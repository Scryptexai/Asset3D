"""
pipeline.py
===========
Blender-headless optimisation pipeline for the Low-Poly Modular Dungeon Kit.

For every raw image-to-3D asset in ``raw_glb/`` this script produces the
marketplace deliverables:

    import & cleanup  -> decimation -> UV unwrap -> PBR bake (high->low)
    -> LOD0..LOD3 -> convex collision mesh -> FBX (Unity) + GLB (Godot)

Run it exactly like a studio would::

    blender --background --python scripts/pipeline.py -- --asset wall
    blender --background --python scripts/pipeline.py -- --all
    python scripts/blender_runner.py --run pipeline.py -- --all   # bpy module mode

Implementation notes
--------------------
* The *raw* mesh is the bake source ("high poly"); the decimated + unwrapped
  copy is the delivery mesh ("low poly"). Bakes are all ray-cast transfers, so
  the low-poly keeps the surface detail of the high-poly (normal + cavity AO)
  and its colour (albedo transfer).
* Albedo is transferred with an **emit bake** rather than a diffuse bake: no
  lighting integration means no noise and no light leaks, which is what a
  texture-transfer bake should be.
* ORM is packed the glTF way (R=AO, G=Roughness, B=Metallic). AO comes from a
  real high->low occlusion bake; roughness/metallic are transferred from the
  raw material analysis.
* Every LOD shares one UV layout and one texture set, which is the convention
  Unity/Unreal expect for LOD chains.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    from blender_runner import ensure_host_ready
    ensure_host_ready()

import bpy
import numpy as np
from mathutils import Vector

import utils
from config import (ASSETS, BAKE_MARGIN, CAGE_EXTRUSION, EXPORTS_DIR, GODOT_DIR, LOD_RATIOS,
                    MANIFEST_PATH, MODELS_DIR, PIPELINE_VERSION, RAW_GLB_DIR,
                    RENDERS_DIR, TEMP_DIR, TEXTURES_DIR, TEXTURE_SIZE,
                    UV_ISLAND_MARGIN, asset_by_key, collision_mesh_name,
                    lod_mesh_name, texture_name)

LOG = utils.get_logger("pipeline")


# --------------------------------------------------------------------------- #
# Stage 1 - import & cleanup
# --------------------------------------------------------------------------- #
def import_raw_asset(desc: dict) -> tuple[object, list]:
    """Import ``raw_glb/<name>.glb`` and return (joined object, all imported)."""
    glb = RAW_GLB_DIR / f"{desc['name']}.glb"
    if not glb.exists():
        raise FileNotFoundError(
            f"{glb} missing - run scripts/recon.py (or a neural generator) first")
    utils.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh in {glb}")
    obj = utils.join_objects(meshes) if len(meshes) > 1 else meshes[0]
    obj.name = f"{desc['name']}_HI"
    utils.select_only([obj])
    return obj, meshes


def cleanup_high_poly(obj) -> dict:
    """Weld + normalise the raw mesh before it is used as a bake source.

    glTF stores independent vertex streams per triangle corner, so the imported
    mesh is an unwelded soup; welding is mandatory before decimation, otherwise
    the collapse modifier cannot propagate across shared edges.
    """
    utils.ensure_object_mode()
    weld = utils.weld_mesh(obj, dist=1e-5)
    utils.recalc_normals_outside(obj)
    utils.select_only([obj])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.object.mode_set(mode="OBJECT")
    LOG.info("weld: %d -> %d verts (%d faces)", weld["verts_before"],
             weld["verts_after"], weld["faces"])
    return {"hi_tris": utils.raw_tri_count(obj.data),
            "hi_non_manifold": utils.count_non_manifold(obj),
            "weld": weld}


# --------------------------------------------------------------------------- #
# Stage 2 + 3 - decimation and UV unwrap
# --------------------------------------------------------------------------- #
def build_lod0(hi_obj, desc: dict):
    """Duplicate the high-poly, decimate to target tris and unwrap it."""
    lod0 = utils.duplicate_object(hi_obj, lod_mesh_name(desc["name"], "LOD0"))
    bpy.context.view_layer.objects.active = lod0
    stats = utils.decimate_to_target(lod0, desc["tris_lod0"], logger=LOG)
    utils.unwrap_smart(lod0, angle_limit_deg=desc["uv_angle_limit"],
                       island_margin=UV_ISLAND_MARGIN, logger=LOG)
    stats["tris"] = utils.raw_tri_count(lod0.data)
    stats["uv_islands"] = _count_uv_islands(lod0)
    return lod0, stats


def _count_uv_islands(obj) -> int:
    """Count connected UV islands (used for the QA report)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.active
    if uv_layer is None:
        bm.free()
        return 0
    seen, islands = set(), 0
    for face in bm.faces:
        if face.index in seen:
            continue
        islands += 1
        stack = [face]
        seen.add(face.index)
        while stack:
            f = stack.pop()
            for edge in f.edges:
                for linked in edge.link_faces:
                    if linked.index in seen:
                        continue
                    # same island only if the UV coordinates are shared
                    if _shares_uv(f, linked, uv_layer):
                        seen.add(linked.index)
                        stack.append(linked)
    bm.free()
    return islands


def _shares_uv(fa, fb, uv_layer, eps: float = 1e-5) -> bool:
    ua = [tuple(round(c, 5) for c in l[uv_layer].uv) for l in fa.loops]
    ub = [tuple(round(c, 5) for c in l[uv_layer].uv) for l in fb.loops]
    return any(a in ub for a in ua)


# --------------------------------------------------------------------------- #
# Stage 4 - PBR baking
# --------------------------------------------------------------------------- #
class BakeSession:
    """Context that prepares Cycles and a bake target image on an object."""

    def __init__(self, low_obj, size: int = TEXTURE_SIZE, samples: int = 32,
                 margin: int = BAKE_MARGIN):
        self.low = low_obj
        self.size = size
        self.samples = samples
        self.margin = margin
        self._temp_images: list = []
        self._temp_materials: list = []
        self._configure_scene()

    # -- scene ---------------------------------------------------------- #
    def _configure_scene(self) -> None:
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = self.samples
        scene.cycles.use_denoising = False
        scene.cycles.max_bounces = 2
        scene.render.bake.margin = self.margin
        scene.render.bake.use_clear = True
        scene.render.bake.margin_type = "ADJACENT_FACES"
        scene.render.bake.use_selected_to_active = True
        scene.render.bake.cage_extrusion = CAGE_EXTRUSION
        scene.render.bake.use_cage = False

    # -- helpers -------------------------------------------------------- #
    def make_target(self, name: str, is_data: bool, color=(0, 0, 0, 1)):
        img = bpy.data.images.new(name, self.size, self.size, alpha=False,
                                  is_data=is_data)
        img.generated_color = color
        self._temp_images.append(img)
        return img

    def bind_target(self, img, mat_name: str):
        """Create a throw-away material whose active node is ``img``."""
        mat = utils.build_image_material(mat_name, img)
        if self.low.data.materials:
            self.low.data.materials[0] = mat
        else:
            self.low.data.materials.append(mat)
        # The *active* image texture node receives the bake result.
        self.low.data.materials[0].node_tree.nodes.active = \
            self.low.data.materials[0].node_tree.nodes["Image Texture"]
        self._temp_materials.append(mat)
        return mat

    def bake(self, bake_type: str, img, *, selected_to_active: bool,
             target_mat: str, pass_color_only: bool = False) -> None:
        scene = bpy.context.scene
        scene.cycles.bake_type = bake_type
        scene.render.bake.use_selected_to_active = selected_to_active
        if bake_type == "DIFFUSE":
            scene.render.bake.use_pass_direct = False
            scene.render.bake.use_pass_indirect = False
            scene.render.bake.use_pass_color = True
        self.bind_target(img, target_mat)
        utils.select_only([self.low])
        if selected_to_active:
            for ob in bpy.context.scene.objects:
                if ob.name.startswith(self.hi_prefix) and ob is not self.low:
                    ob.select_set(True)
                ob.hide_render = False
        bpy.ops.object.bake(type=bake_type, use_clear=True, margin=self.margin,
                            margin_type="ADJACENT_FACES")
        LOG.info("bake %-7s -> %s", bake_type, img.name)

    # -- accessors ------------------------------------------------------ #
    @property
    def hi_prefix(self) -> str:
        return self._hi_prefix

    @hi_prefix.setter
    def hi_prefix(self, value: str) -> None:
        self._hi_prefix = value

    def cleanup(self) -> None:
        for img in self._temp_images:
            if img.name in bpy.data.images:
                bpy.data.images.remove(img)
        for mat in self._temp_materials:
            if mat.name in bpy.data.materials:
                bpy.data.materials.remove(mat)


def _emission_material(name: str, image) -> object:
    """Emission shader with ``image`` as colour - the transfer bake source."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emi = nt.nodes.new("ShaderNodeEmission")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
    nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    nt.nodes.active = tex
    return mat


def _load_image(path: Path, non_color: bool = False):
    img = bpy.data.images.load(str(path), check_existing=False)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def bake_pbr_set(desc: dict, hi_obj, low_obj) -> dict:
    """Bake the full PBR set from ``hi_obj`` onto ``low_obj``'s UV layout."""
    size = TEXTURE_SIZE
    session = BakeSession(low_obj, size=size, samples=desc.get("bake_samples", 32))
    session.hi_prefix = hi_obj.name

    # Remember the high-poly's own materials, we swap in emission for transfers.
    hi_mats = list(hi_obj.data.materials)

    albedo_img = session.make_target("bake_albedo", is_data=False, color=(0.5, 0.5, 0.5, 1))
    normal_img = session.make_target("bake_normal", is_data=True, color=(0.5, 0.5, 1.0, 1))
    # Ambient occlusion is low-frequency: baking it at half resolution and
    # upsampling is visually identical at 2048 but ~4x cheaper on CPU.
    ao_size = max(256, size // 2)
    ao_img = bpy.data.images.new("bake_ao", ao_size, ao_size, alpha=False,
                                 is_data=True)
    ao_img.generated_color = (1, 1, 1, 1)
    session._temp_images.append(ao_img)
    rough_img = session.make_target("bake_rough", is_data=True, color=(0.8, 0.8, 0.8, 1))
    metal_img = session.make_target("bake_metal", is_data=True, color=(0, 0, 0, 1))

    raw_dir = RAW_GLB_DIR
    albedo_src = _load_image(raw_dir / f"{desc['name']}_raw_albedo.png", non_color=False)
    orm_src = _load_image(raw_dir / f"{desc['name']}_raw_orm.png", non_color=True)

    # --- albedo: emit transfer ---------------------------------------- #
    emi_albedo = _emission_material("HI_emit_albedo", albedo_src)
    hi_obj.data.materials.clear()
    hi_obj.data.materials.append(emi_albedo)
    session.bake("EMIT", albedo_img, selected_to_active=True,
                 target_mat="LO_albedo_target")

    # --- normal: geometry transfer ------------------------------------ #
    hi_obj.data.materials.clear()
    for m in hi_mats:
        hi_obj.data.materials.append(m)
    session.bake("NORMAL", normal_img, selected_to_active=True,
                 target_mat="LO_normal_target")

    # --- AO: cavity occlusion from the high-poly ----------------------- #
    bpy.context.scene.cycles.samples = desc.get("ao_samples", 24)
    session.bake("AO", ao_img, selected_to_active=True, target_mat="LO_ao_target")
    bpy.context.scene.cycles.samples = session.samples

    # --- roughness / metallic: emit transfer from the raw ORM ---------- #
    emi_orm = _emission_material("HI_emit_orm", orm_src)
    hi_obj.data.materials.clear()
    hi_obj.data.materials.append(emi_orm)
    session.bake("EMIT", rough_img, selected_to_active=True,
                 target_mat="LO_rough_target")
    session.bake("EMIT", metal_img, selected_to_active=True,
                 target_mat="LO_metal_target")

    # restore
    hi_obj.data.materials.clear()
    for m in hi_mats:
        hi_obj.data.materials.append(m)

    # --- pack ORM and write textures ----------------------------------- #
    ao = utils.image_to_numpy(ao_img)
    if ao.shape[0] != size:
        from scipy import ndimage as _ndi
        ao = _ndi.zoom(ao, (size / ao.shape[0], size / ao.shape[1], 1.0), order=1)
        ao = _ndi.gaussian_filter(ao, (1.5, 1.5, 0))
    rough = utils.image_to_numpy(rough_img)
    metal = utils.image_to_numpy(metal_img)
    orm = np.empty_like(ao)
    orm[..., 0] = ao[..., 0]
    orm[..., 1] = rough[..., 1] if not np.allclose(rough[..., 1], rough[..., 0]) else rough[..., 0]
    orm[..., 2] = metal[..., 0]
    orm[..., 3] = 1.0

    orm_img = bpy.data.images.new("bake_orm", size, size, alpha=False, is_data=True)
    utils.numpy_to_image(orm_img, orm)

    tex_paths = {
        "Albedo": utils.save_image(albedo_img, TEXTURES_DIR / texture_name(desc["name"], "Albedo")),
        "Normal": utils.save_image(normal_img, TEXTURES_DIR / texture_name(desc["name"], "Normal")),
        "ORM": utils.save_image(orm_img, TEXTURES_DIR / texture_name(desc["name"], "ORM")),
    }

    # Reload from disk so the exported material references the shipped files.
    tex_images = {
        "albedo": _load_image(tex_paths["Albedo"], non_color=False),
        "normal": _load_image(tex_paths["Normal"], non_color=True),
        "orm": _load_image(tex_paths["ORM"], non_color=True),
    }
    session.cleanup()
    for img in (albedo_src, orm_src):
        if img.name in bpy.data.images:
            bpy.data.images.remove(img)
    del hi_mats
    return {"paths": tex_paths, "images": tex_images}


# --------------------------------------------------------------------------- #
# Stage 5 - LOD chain
# --------------------------------------------------------------------------- #
def build_lod_chain(lod0, desc: dict, material) -> dict:
    """Decimate LOD0 into LOD1..LOD3, all sharing UVs and material."""
    base_tris = utils.raw_tri_count(lod0.data)
    lods = {"LOD0": lod0}
    for name, ratio in LOD_RATIOS.items():
        if name == "LOD0":
            continue
        target = max(24, int(round(base_tris * ratio)))
        clone = utils.duplicate_object(lod0, lod_mesh_name(desc["name"], name))
        utils.select_only([clone])
        stats = utils.decimate_to_target(clone, target, logger=LOG)
        clone.data.materials.clear()
        clone.data.materials.append(material)
        lods[name] = clone
        LOG.info("%s: %d tris (target %d)", name,
                 utils.raw_tri_count(clone.data), target)
    # re-apply material to LOD0 (bake swapped it for targets)
    lod0.data.materials.clear()
    lod0.data.materials.append(material)
    return lods


# --------------------------------------------------------------------------- #
# Stage 6 - collision
# --------------------------------------------------------------------------- #
def build_collision(lod0, desc: dict):
    hull = utils.convex_hull_copy(lod0, collision_mesh_name(desc["name"]))
    # Convex hull of a 1400-tri architectural module can be heavy; simplify it.
    target = desc.get("collision_tris", 64)
    if utils.raw_tri_count(hull.data) > target:
        utils.decimate_to_target(hull, target, logger=LOG)
    return hull


# --------------------------------------------------------------------------- #
# Stage 7 - export
# --------------------------------------------------------------------------- #
GLB_TEXTURE_FORMAT = "webp"     # overridable from the CLI (--glb-texture)


def export_asset(desc: dict, lods: dict, collision, material) -> dict:
    """Write FBX (Models/), GLB (Godot/) for every LOD + the collision mesh."""
    name = desc["name"]
    written = {"models": [], "godot": []}

    for lod_name, obj in lods.items():
        obj.name = lod_mesh_name(name, lod_name)
        utils.set_origin(obj, desc.get("pivot", "bottom"))
        obj.data.materials.clear()
        obj.data.materials.append(material)

        fbx = MODELS_DIR / f"{obj.name}.fbx"
        utils.export_fbx(obj, fbx)
        written["models"].append(fbx.name)

        glb = GODOT_DIR / f"{obj.name}.glb"
        utils.export_glb(obj, glb, texture_format=GLB_TEXTURE_FORMAT)
        written["godot"].append(glb.name)

    collision.name = collision_mesh_name(name)
    utils.set_origin(collision, desc.get("pivot", "bottom"))
    collision.data.materials.clear()
    fbx = MODELS_DIR / f"{collision.name}.fbx"
    utils.export_fbx(collision, fbx)
    written["models"].append(fbx.name)
    glb = GODOT_DIR / f"{collision.name}.glb"
    utils.export_glb(collision, glb)
    written["godot"].append(glb.name)

    return written


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def state_path(desc: dict) -> Path:
    """Sidecar recording which pipeline version produced an asset's exports."""
    return EXPORTS_DIR / ".pipeline" / f"{desc['name']}.json"


def is_up_to_date(desc: dict) -> bool:
    """True when this asset was exported by the *current* pipeline version.

    Timestamps alone are not enough: a code change (e.g. a new material graph)
    must invalidate cached exports even though the files are newer than the raw
    GLB, so the pipeline version is the source of truth.
    """
    raw = RAW_GLB_DIR / f"{desc['name']}.glb"
    sidecar = state_path(desc)
    if not raw.exists() or not sidecar.exists():
        return False
    try:
        state = utils.read_json(sidecar)
    except Exception:                                   # noqa: BLE001
        return False
    if state.get("pipeline_version") != PIPELINE_VERSION:
        return False
    files = ([MODELS_DIR / f"{lod_mesh_name(desc['name'], l)}.fbx" for l in LOD_RATIOS]
             + [GODOT_DIR / f"{lod_mesh_name(desc['name'], l)}.glb" for l in LOD_RATIOS]
             + [GODOT_DIR / f"{collision_mesh_name(desc['name'])}.glb"]
             + [TEXTURES_DIR / texture_name(desc["name"], k)
                for k in ("Albedo", "Normal", "ORM")])
    return all(f.exists() for f in files)


def process_asset(key: str, *, render_preview: bool = False) -> dict:
    """Run the full optimisation pipeline for a single asset key."""
    desc = asset_by_key(key)
    LOG.info("==================== %s ====================", desc["name"])

    hi_obj, _ = import_raw_asset(desc)
    hi_stats = cleanup_high_poly(hi_obj)

    lod0, lod_stats = build_lod0(hi_obj, desc)
    bakes = bake_pbr_set(desc, hi_obj, lod0)

    material = utils.build_pbr_material(
        f"M_{desc['name']}", bakes["images"]["albedo"],
        bakes["images"]["normal"], bakes["images"]["orm"])

    lods = build_lod_chain(lod0, desc, material)
    collision = build_collision(lods["LOD0"], desc)
    written = export_asset(desc, lods, collision, material)

    tri_counts = {n: utils.raw_tri_count(o.data) for n, o in lods.items()}
    man = {n: utils.count_non_manifold(o) for n, o in lods.items()}
    meta = {
        "key": key,
        "name": desc["name"],
        "category": desc["tags"][0],
        "tags": desc["tags"],
        "dimensions_m": list(desc["dims"]),
        "source": {
            "reference": f"refs/raw/{desc['ref']}",
            "raw_glb": f"raw_glb/{desc['name']}.glb",
            "raw_backend": utils.read_json(RAW_GLB_DIR / f"{desc['name']}.json").get("backend")
            if (RAW_GLB_DIR / f"{desc['name']}.json").exists() else "unknown",
        },
        "geometry": {
            "high_poly_tris": hi_stats["hi_tris"],
            "lod_tris": tri_counts,
            "collision_tris": utils.raw_tri_count(collision.data),
            "uv_islands_lod0": lod_stats.get("uv_islands"),
            "decimate_passes": lod_stats.get("passes"),
        },
        "validation": {
            "non_manifold_lod0": man["LOD0"],
            "non_manifold_all_lods": sum(man.values()),
            "watertight_lod0": man["LOD0"] == 0,
            "lod_chain_complete": set(tri_counts) == set(LOD_RATIOS),
            "has_collision": collision is not None,
        },
        "textures": {
            k: f"exports/Textures/{v.name}" for k, v in bakes["paths"].items()
        },
        "files": {
            "models": [f"exports/Models/{f}" for f in written["models"]],
            "godot": [f"exports/Godot/{f}" for f in written["godot"]],
        },
    }

    if render_preview:
        import preview
        preview.render_views(
            GODOT_DIR / f"{lod_mesh_name(desc['name'], 'LOD0')}.glb",
            RENDERS_DIR / f"{desc['key']}_lod0_preview.png",
            res=480, samples=40, label=f"{desc['name']} - LOD0")

    utils.write_json(state_path(desc), {
        "pipeline_version": PIPELINE_VERSION,
        "asset": desc["name"],
        "blender": bpy.app.version_string,
        "texture_size": TEXTURE_SIZE,
        "glb_texture_format": GLB_TEXTURE_FORMAT,
    })
    LOG.info("%s done: %s | non-manifold LOD0=%d", desc["name"], tri_counts, man["LOD0"])
    return meta


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Low-poly optimisation pipeline")
    ap.add_argument("--asset", action="append", help="asset key (repeatable)")
    ap.add_argument("--all", action="store_true", help="process every configured asset")
    ap.add_argument("--preview", action="store_true", help="render a LOD0 preview")
    ap.add_argument("--glb-texture", default="webp", choices=["webp", "jpeg", "png"],
                    help="embedded texture format for the Godot GLBs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip assets whose exports are already newer than their raw GLB")
    ns = ap.parse_args(argv)

    global GLB_TEXTURE_FORMAT
    GLB_TEXTURE_FORMAT = ns.glb_texture

    keys = [a["key"] for a in ASSETS] if ns.all or not ns.asset else ns.asset
    results = []
    for key in keys:
        desc = asset_by_key(key)
        if ns.skip_existing and is_up_to_date(desc):
            LOG.info("skipping %s (exports newer than raw_glb)", desc["name"])
            continue
        results.append(process_asset(key, render_preview=ns.preview))

    # Merge into the existing manifest so `--skip-existing` runs keep the
    # entries of assets that were not re-processed.
    manifest_path = MANIFEST_PATH
    previous = utils.read_json(manifest_path) if manifest_path.exists() else {}
    by_name = {a["name"]: a for a in previous.get("assets", [])}
    for entry in results:
        by_name[entry["name"]] = entry
    ordered = [by_name[a["name"]] for a in ASSETS if a["name"] in by_name]
    manifest = {"kit": "Low-Poly Modular Dungeon Kit",
                "version": "1.0.0",
                "pipeline_version": PIPELINE_VERSION,
                "texture_size": TEXTURE_SIZE,
                "lod_ratios": LOD_RATIOS,
                "assets": ordered}
    utils.write_json(manifest_path, manifest)
    LOG.info("manifest written: %s (%d assets)", manifest_path, len(ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
