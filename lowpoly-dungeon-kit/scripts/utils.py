"""
utils.py
========
Reusable helpers shared by the Blender-side pipeline stages.

The module is imported from inside Blender (``bpy`` available) but degrades
gracefully for pure-CPython use: only the functions that touch ``bpy`` require
Blender, and importing the module outside Blender is still safe.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - trivial import guard
    import bpy  # type: ignore
    import bmesh  # type: ignore
    from mathutils import Vector  # type: ignore
    IN_BLENDER = True
except ImportError:  # pragma: no cover
    bpy = None  # type: ignore
    bmesh = None  # type: ignore
    Vector = None  # type: ignore
    IN_BLENDER = False

import numpy as np

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FORMAT = "[%(asctime)s] %(levelname)-7s %(name)-14s %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Idempotent logger factory (safe to call twice in the same process)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


class Stopwatch:
    """Tiny context manager for timing pipeline stages."""

    def __init__(self, logger: logging.Logger, label: str):
        self.logger, self.label = logger, label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        self.logger.info("%s finished in %.1fs", self.label, self.elapsed)
        return False


# --------------------------------------------------------------------------- #
# Generic file helpers
# --------------------------------------------------------------------------- #
def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def write_json(path: str | Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
    return path


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Blender object helpers
# --------------------------------------------------------------------------- #
def require_blender() -> None:
    if not IN_BLENDER:
        raise RuntimeError("this function requires Blender (bpy)")


def reset_scene() -> None:
    """Wipe the current Blender scene and orphan data (headless-safe)."""
    require_blender()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                 bpy.data.armatures, bpy.data.actions):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def select_only(objects: Sequence) -> None:
    """Deselect everything, then select + activate ``objects``."""
    require_blender()
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objects:
        ob.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def ensure_object_mode() -> None:
    require_blender()
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def tri_count(obj) -> int:
    """Triangle count of ``obj`` after evaluating modifiers."""
    require_blender()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    ev.to_mesh_clear()
    return n


def raw_tri_count(mesh) -> int:
    """Triangle count of a (already triangulated or not) mesh datablock."""
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def weld_mesh(obj, dist: float = 1e-4) -> dict:
    """Weld coincident vertices and drop geometrically loose elements.

    Operators such as ``mesh.remove_doubles`` are unreliable in background mode
    on very large imported glTF meshes (glTF splits vertices per loop, so a
    100k-triangle asset arrives as ~300k unwelded vertices).  Doing the merge
    through ``bmesh.ops`` is deterministic and an order of magnitude faster.
    """
    require_blender()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    before = (len(bm.verts), len(bm.faces))
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=dist)
    # Drop wire edges / stray vertices left behind by the merge.
    loose_edges = [e for e in bm.edges if not e.link_faces]
    if loose_edges:
        bmesh.ops.delete(bm, geom=loose_edges, context="EDGES_FACES")
    loose_verts = [v for v in bm.verts if not v.link_faces]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    me.update()
    after = (len(me.vertices), len(me.polygons))
    return {"verts_before": before[0], "verts_after": after[0],
            "faces": after[1]}


def recalc_normals_outside(obj) -> None:
    """Force consistent outward normals (bmesh, background-safe)."""
    require_blender()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def delete_loose_geometry(obj) -> int:
    """Remove wire edges and vertices without faces; return how many were cut."""
    require_blender()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    geom = [e for e in bm.edges if not e.link_faces]
    geom += [v for v in bm.verts if not v.link_faces]
    n = len(geom)
    if geom:
        bmesh.ops.delete(bm, geom=geom, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return n


def apply_modifier(obj, mod_name: str) -> None:
    require_blender()
    select_only([obj])
    bpy.ops.object.modifier_apply(modifier=mod_name)


def join_objects(objects: Iterable):
    require_blender()
    objects = list(objects)
    if len(objects) == 1:
        return objects[0]
    select_only(objects)
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def set_active(obj) -> None:
    require_blender()
    bpy.context.view_layer.objects.active = obj


def duplicate_object(obj, name: str):
    require_blender()
    new = obj.copy()
    new.data = obj.data.copy()
    bpy.context.collection.objects.link(new)
    new.name = name
    return new


def delete_objects(objects: Iterable) -> None:
    require_blender()
    objects = [o for o in objects if o is not None]
    if not objects:
        return
    select_only(objects)
    bpy.ops.object.delete(use_global=False)


def hide_from_render(*objects) -> None:
    require_blender()
    for ob in objects:
        ob.hide_render = True


# --------------------------------------------------------------------------- #
# Mesh repair
# --------------------------------------------------------------------------- #
def clean_mesh(obj, merge_distance: float = 1e-4, logger=None) -> dict:
    """Delete loose elements, weld doubles, recalc normals, triangulate.

    Returns a small report dict with the non-manifold edge count before and
    after the repair so the QA stage can assert on it.
    """
    require_blender()
    logger = logger or get_logger("utils.clean")
    ensure_object_mode()
    select_only([obj])

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    before = count_non_manifold(obj)

    # Fill any holes left by loose-face deletion, then re-triangulate.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.fill_holes(sides=0)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")

    after = count_non_manifold(obj)
    report = {"non_manifold_before": before, "non_manifold_after": after,
              "tris": raw_tri_count(obj.data)}
    logger.info("clean: non-manifold %d -> %d, tris=%d",
                before, after, report["tris"])
    return report


def count_non_manifold(obj) -> int:
    """Number of non-manifold edges (boundary + >2-face edges)."""
    require_blender()
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def has_loose_geometry(obj) -> bool:
    require_blender()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    loose = any(len(v.link_faces) == 0 for v in bm.verts)
    bm.free()
    return loose


def decimate_to_target(obj, target_tris: int, logger=None) -> dict:
    """Collapse-decimate ``obj`` in place until it reaches ``target_tris``.

    Blender's decimate ratio is only an approximation, so we iterate: compute
    the current triangle count, derive the ratio, apply, and repeat while the
    result is above the tolerance band.
    """
    require_blender()
    logger = logger or get_logger("utils.decimate")
    current = raw_tri_count(obj.data)
    start = current
    if current <= target_tris:
        return {"from": start, "to": current, "passes": 0}

    pass_no = 0
    while current > target_tris and pass_no < 6:
        ratio = target_tris / float(current)
        # shrink the ratio slightly to counter Blender's rounding up
        ratio = max(0.01, min(1.0, ratio * 0.985))
        mod = obj.modifiers.new(name=f"Decimate_{pass_no}", type="DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
        apply_modifier(obj, mod.name)
        new = raw_tri_count(obj.data)
        logger.info("decimate pass %d: %d -> %d tris (ratio %.3f)",
                    pass_no, current, new, ratio)
        if new >= current:  # no progress, bail out
            break
        current = new
        pass_no += 1
    # Decimation can leave orphaned vertices behind; scrub them so the exported
    # mesh (and its non-manifold count) stays clean.
    delete_loose_geometry(obj)
    recalc_normals_outside(obj)
    return {"from": start, "to": raw_tri_count(obj.data), "passes": pass_no}


def unwrap_smart(obj, angle_limit_deg: float = 66.0,
                 island_margin: float = 0.004, logger=None) -> None:
    """Smart UV Project on the active object (creates Layer 0 if missing)."""
    require_blender()
    logger = logger or get_logger("utils.uv")
    ensure_object_mode()
    select_only([obj])
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=np.radians(angle_limit_deg),
        island_margin=island_margin,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    logger.info("smart UV project: %d UV islands-ish (layer '%s')",
                len(obj.data.uv_layers), obj.data.uv_layers.active.name)


def box_project_uv(obj, bbox_scale: float = 1.0,
                   uv_crop: tuple[float, float, float, float] | None = None) -> None:
    """Axis-aligned box projection from world/object coordinates.

    Used for the raw (un-optimised) meshes: every face is projected along its
    dominant normal axis, which gives a sensible texture mapping without
    requiring a lengthy unwrap.

    ``uv_crop`` maps the object bbox onto a sub-rectangle of the texture
    ``(u0, u1, v0, v1)`` in normalised coordinates.  The raw assets use the
    silhouette's bbox so no geometry ever samples the studio background.
    """
    require_blender()
    me = obj.data
    uv = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    me.update()
    mn = np.array([min(v.co[i] for v in me.vertices) for i in range(3)])
    mx = np.array([max(v.co[i] for v in me.vertices) for i in range(3)])
    extent = np.maximum(mx - mn, 1e-6)

    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)

    for poly in me.polygons:
        n = poly.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        u_ax, v_ax = [(1, 2), (0, 2), (0, 1)][ax]
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            v = co[vi]
            u = (v[u_ax] - mn[u_ax]) / extent[u_ax] / max(bbox_scale, 1e-6)
            w = (v[v_ax] - mn[v_ax]) / extent[v_ax] / max(bbox_scale, 1e-6)
            if uv_crop is not None:
                u0, u1, v0, v1 = uv_crop
                u = u0 + u * (u1 - u0)
                w = v0 + w * (v1 - v0)
            uv.data[li].uv = (u, w)


# --------------------------------------------------------------------------- #
# Materials / images
# --------------------------------------------------------------------------- #
def new_image(name: str, size: int, color=(0.0, 0.0, 0.0, 1.0), is_data: bool = False,
              float_buffer: bool = False):
    require_blender()
    img = bpy.data.images.new(name, width=size, height=size, alpha=False,
                              float_buffer=float_buffer, is_data=is_data)
    img.generated_color = color
    return img


def image_to_numpy(img) -> np.ndarray:
    """Return the image buffer as ``(H, W, 4)`` float32 (bottom-up rows)."""
    require_blender()
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)


def numpy_to_image(img, array: np.ndarray) -> None:
    """Write an ``(H, W, 4)`` float32 array back into a Blender image."""
    require_blender()
    h, w = array.shape[:2]
    img.pixels.foreach_set(array.reshape(-1).astype(np.float32))
    img.update()


def save_image(img, path: str | Path, color_depth: str = "8",
               compression: int = 100) -> Path:
    """Save a Blender image to PNG with explicit settings (deterministic)."""
    require_blender()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.file_format = "PNG"
    settings = bpy.context.scene.render.image_settings
    prev = (settings.file_format, settings.color_mode, settings.color_depth,
            settings.compression)
    settings.file_format = "PNG"
    settings.color_mode = "RGB"
    settings.color_depth = color_depth
    settings.compression = compression
    img.filepath_raw = str(path)
    img.save()
    (settings.file_format, settings.color_mode, settings.color_depth,
     settings.compression) = prev
    return path


def emate_material_image(mat) -> None:
    """Compatibility shim kept for readability of call sites."""
    return None


def build_image_material(name: str, image, *, strength: float = 1.0,
                         emission: bool = False, alpha_clip: bool = False):
    """Create a Principled (or Emission) material driven by ``image``.

    The object must have a UV map; the texture node uses the active UV layer.
    """
    require_blender()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.location = (-400, 0)
    tex.extension = "EXTEND"

    if emission:
        emi = nt.nodes.new("ShaderNodeEmission")
        emi.inputs["Strength"].default_value = strength
        nt.links.new(tex.outputs["Color"], emi.inputs["Color"])
        nt.links.new(emi.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.85
        bsdf.inputs["Metallic"].default_value = 0.0
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def gltf_settings_group():
    """Return the ``glTF Settings`` node group used to export an occlusion map.

    glTF has no way to declare AO from a plain Principled BSDF, so Blender's
    exporter looks for a node group with this exact name and an ``Occlusion``
    input. Using it makes the ORM texture (R = AO) be referenced as
    ``occlusionTexture`` in the GLB -- which is the whole point of packing ORM.
    """
    require_blender()
    name = "glTF Settings"
    group = bpy.data.node_groups.get(name)
    if group is None:
        group = bpy.data.node_groups.new(name, "ShaderNodeTree")
        if hasattr(group, "interface"):          # Blender 4.0+
            group.interface.new_socket("Occlusion", in_out="INPUT",
                                       socket_type="NodeSocketFloat")
        else:                                     # pragma: no cover - Blender 3.x
            group.inputs.new("NodeSocketFloat", "Occlusion")
    return group


def build_pbr_material(name: str, albedo, normal, orm, *, uv_scale: float = 1.0):
    """Assemble the final glTF-compatible PBR material used in exports.

    glTF/metallic-roughness wants ORM packed as ``R=AO, G=Roughness, B=Metal``
    which is exactly what the bake stage produces, so a single image node feeds
    both the Roughness and Metallic sockets via a Separate Color node.
    """
    require_blender()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    uvmap = nt.nodes.new("ShaderNodeUVMap"); uvmap.location = (-900, 0)
    uvmap.uv_map = "UVMap"

    def add_tex(img, loc):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        n.location = loc
        n.extension = "EXTEND"
        nt.links.new(uvmap.outputs["UV"], n.inputs["Vector"])
        return n

    t_albedo = add_tex(albedo, (-500, 250))
    t_normal = add_tex(normal, (-500, -50))
    t_orm = add_tex(orm, (-500, -350))

    nt.links.new(t_albedo.outputs["Color"], bsdf.inputs["Base Color"])

    nmap = nt.nodes.new("ShaderNodeNormalMap"); nmap.location = (-200, -50)
    nt.links.new(t_normal.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (-200, -350)
    nt.links.new(t_orm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])

    # Export the AO channel so the GLB carries an occlusionTexture.
    settings = nt.nodes.new("ShaderNodeGroup")
    settings.node_tree = gltf_settings_group()
    settings.location = (300, -350)
    nt.links.new(sep.outputs["Red"], settings.inputs["Occlusion"])

    return mat


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def export_fbx(obj, filepath: str | Path, *, scale: float = 1.0) -> Path:
    """Export a single object as FBX with Unity-friendly settings."""
    require_blender()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ensure_object_mode()
    obj.data.name = obj.name        # marketplace-friendly mesh datablock name
    select_only([obj])
    bpy.ops.export_scene.fbx(
        filepath=str(filepath),
        use_selection=True,
        object_types={"MESH"},
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        global_scale=scale,
        mesh_smooth_type="FACE",
        use_mesh_modifiers=True,
        use_tspace=True,
        add_leaf_bones=False,
        bake_anim=False,
        path_mode="AUTO",
        embed_textures=False,
        axis_forward="-Z",
        axis_up="Y",
    )
    return filepath


def export_glb(obj, filepath: str | Path, *, draco: bool = False,
               texture_format: str = "webp", quality: int = 90) -> Path:
    """Export a single object as a self-contained binary glTF (.glb).

    ``texture_format``
        * ``"webp"`` - embedded ``EXT_texture_webp`` images (Godot 4 reads them
          natively; ~10x smaller than PNG).  This is the kit default, because a
          2048 PNG set per LOD would push a single GLB past 9 MB.
        * ``"jpeg"`` - glTF-core JPEG (maximum loader compatibility).
        * ``"png"``  - lossless, largest.
    """
    require_blender()
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    ensure_object_mode()
    obj.data.name = obj.name        # marketplace-friendly mesh datablock name
    select_only([obj])
    fmt = {"webp": "WEBP", "jpeg": "JPEG", "png": "AUTO"}.get(texture_format.lower(), "WEBP")
    bpy.ops.export_scene.gltf(
        filepath=str(filepath),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format=fmt,
        export_image_quality=quality,
        export_image_webp_fallback=False,
        export_draco_mesh_compression_enable=draco,
        export_animations=False,
        export_skins=False,
    )
    return filepath


def set_origin(obj, pivot: str = "bottom") -> None:
    """Move the mesh so that the object origin is predictable for level design.

    ``bottom`` -> origin at the bounding-box bottom centre (grid snapping)
    ``top``    -> origin at the bounding-box top centre (floor pieces: walk
                  surface at local z = 0)
    ``center`` -> origin at the geometric centre
    """
    require_blender()
    mn = Vector((min(v.co[i] for v in obj.data.vertices) for i in range(3)))
    mx = Vector((max(v.co[i] for v in obj.data.vertices) for i in range(3)))
    cx, cy = (mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0
    if pivot == "bottom":
        target = Vector((cx, cy, mn.z))
    elif pivot == "top":
        target = Vector((cx, cy, mx.z))
    else:
        target = (mn + mx) / 2.0
    for v in obj.data.vertices:
        v.co -= target
    obj.data.update()


def mesh_bounds(obj) -> dict:
    require_blender()
    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    zs = [v.co.z for v in obj.data.vertices]
    return {"min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)],
            "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]}


def convex_hull_copy(obj, name: str):
    """Convex-hull collision proxy built from ``obj``'s evaluated mesh."""
    require_blender()
    hull = duplicate_object(obj, name)
    # Strip everything but geometry.
    hull.modifiers.clear()
    select_only([hull])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    mod = hull.modifiers.new(name="Hull", type="BOOLEAN")  # placeholder, replaced
    hull.modifiers.remove(mod)
    select_only([hull])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    hull.data.materials.clear()
    return hull


__all__ = [name for name in dir() if not name.startswith("_")]
