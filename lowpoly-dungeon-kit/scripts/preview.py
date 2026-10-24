"""
preview.py
==========
Headless Cycles turntable renderer used for QA and marketplace previews.

It renders an already-exported file (``.glb`` or ``.fbx``) from several
orthographic angles with a neutral studio setup and assembles a contact sheet,
so every artefact that ships is inspected *after* export -- exactly what a
buyer will load in Unity/Godot.

Usage
-----
    python scripts/preview.py --input raw_glb/SM_Dungeon_Wall_A.glb \
        --out renders/qa_wall.png --res 512 --samples 48
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    from blender_runner import ensure_host_ready
    ensure_host_ready()

import math

import bpy
from mathutils import Vector

import utils
from config import RENDERS_DIR, TEMP_DIR

LOG = utils.get_logger("preview")

# (label, azimuth degrees, elevation degrees)
DEFAULT_VIEWS = [
    ("front", 0.0, 8.0),
    ("three_quarter", 45.0, 22.0),
    ("side", 90.0, 10.0),
    ("top", 25.0, 62.0),
]


# --------------------------------------------------------------------------- #
# Scene setup
# --------------------------------------------------------------------------- #
def _clear() -> None:
    utils.reset_scene()


def import_file(path: str | Path):
    """Import a GLB or FBX and return the list of imported mesh objects."""
    path = Path(path)
    before = set(bpy.context.scene.objects)
    suffix = path.suffix.lower()
    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path))
    else:
        raise ValueError(f"unsupported preview input: {path}")
    new = [o for o in bpy.context.scene.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh found in {path}")
    return new, meshes


def world_bounds(objects) -> tuple[Vector, Vector]:
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for ob in objects:
        for corner in ob.bound_box:
            w = ob.matrix_world @ Vector(corner)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    return mn, mx


def setup_studio(center: Vector, radius: float, *, floor_z: float = 0.0,
                 world_grey: float = 0.78) -> None:
    """Neutral three-point studio with a light backdrop (Cycles, CPU)."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"

    # World: soft neutral ambient.
    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (world_grey, world_grey, world_grey, 1.0)
    bg.inputs["Strength"].default_value = 0.55
    scene.world = world

    # Key / fill / rim are *suns* so the lighting is scale independent.
    for name, energy, angle, rot in (
        ("Key", 3.4, 0.09, (math.radians(52), 0, math.radians(35))),
        ("Fill", 1.1, 0.5, (math.radians(60), 0, math.radians(-115))),
        ("Rim", 2.0, 0.2, (math.radians(115), 0, math.radians(200))),
    ):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        light.angle = angle
        ob = bpy.data.objects.new(name, light)
        ob.rotation_euler = rot
        bpy.context.collection.objects.link(ob)

    # Backdrop: a large ground plane *below* the asset's bounding box. Placing it
    # at z=0 would occlude modules whose pivot already sits on the ground (e.g.
    # a floor tile spanning z in [-0.35, 0]).
    bpy.ops.mesh.primitive_plane_add(size=radius * 12,
                                     location=(center.x, center.y, floor_z - radius * 0.02))
    floor = bpy.context.object
    floor.name = "Backdrop"
    mat = bpy.data.materials.new("BackdropMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.42, 0.43, 0.46, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.9
    floor.data.materials.append(mat)


def make_camera(center: Vector, radius: float) -> object:
    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = radius * 2.35
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    target = bpy.data.objects.new("CamTarget", None)
    target.location = center
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return cam


def place_camera(cam, center: Vector, radius: float, azim: float, elev: float) -> None:
    a, e = math.radians(azim), math.radians(elev)
    d = radius * 4.0
    cam.location = center + Vector((
        d * math.cos(e) * math.sin(a),
        -d * math.cos(e) * math.cos(a),
        d * math.sin(e),
    ))
    bpy.context.view_layer.update()


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render_views(input_path: str | Path, out_path: str | Path, *,
                 views=None, res: int = 512, samples: int = 48,
                 label: str | None = None, flat: bool = False) -> Path:
    """Render ``views`` of ``input_path`` into a single PNG contact sheet."""
    views = views or DEFAULT_VIEWS
    out_path = Path(out_path)
    _clear()
    objs, meshes = import_file(input_path)
    if flat:
        # QA mode: untextured clay render, isolates geometry problems from
        # texture / UV projection problems.
        clay = bpy.data.materials.new("Clay")
        clay.use_nodes = True
        b = clay.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.55, 0.55, 0.56, 1)
        b.inputs["Roughness"].default_value = 0.65
        for ob in meshes:
            ob.data.materials.clear()
            ob.data.materials.append(clay)

    mn, mx = world_bounds(meshes)
    center = (mn + mx) / 2.0
    radius = max((mx - mn).length / 2.0, 1e-3)

    setup_studio(center, radius, floor_z=mn.z)
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.render.resolution_x = res
    bpy.context.scene.render.resolution_y = res
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "RGB"

    cam = make_camera(center, radius)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    shots: list[Path] = []
    for name, azim, elev in views:
        place_camera(cam, center, radius, azim, elev)
        frame = TEMP_DIR / f"_prev_{out_path.stem}_{name}.png"
        bpy.context.scene.render.filepath = str(frame)
        bpy.ops.render.render(write_still=True)
        shots.append(frame)
        LOG.info("rendered %s view -> %s", name, frame.name)

    sheet = assemble_contact_sheet(shots, out_path, labels=[v[0] for v in views],
                                   title=label or Path(input_path).stem)
    for s in shots:
        s.unlink(missing_ok=True)
    LOG.info("contact sheet: %s", sheet)
    return sheet


def assemble_contact_sheet(images: list[Path], out_path: Path, *,
                           labels: list[str] | None = None,
                           title: str = "", gutter: int = 8,
                           bg=(28, 30, 34)) -> Path:
    """Combine rendered frames into a horizontal contact sheet (PIL)."""
    from PIL import Image, ImageDraw

    imgs = [Image.open(p).convert("RGB") for p in images]
    w, h = imgs[0].size
    n = len(imgs)
    title_h = 34 if title else 0
    sheet = Image.new("RGB", (n * w + (n + 1) * gutter, h + title_h + gutter), bg)
    for i, im in enumerate(imgs):
        sheet.paste(im, (gutter + i * (w + gutter), title_h))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((gutter + 2, 10), title, fill=(235, 235, 235))
    if labels:
        for i, lab in enumerate(labels):
            draw.text((gutter + i * (w + gutter) + 6, title_h + 6), lab, fill=(255, 210, 130))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Headless turntable preview renderer")
    ap.add_argument("--input", required=True, help=".glb or .fbx to render")
    ap.add_argument("--out", default=None, help="output PNG (default renders/<stem>.png)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--flat", action="store_true", help="untextured clay render")
    ns = ap.parse_args(argv)

    inp = Path(ns.input)
    out = Path(ns.out) if ns.out else RENDERS_DIR / f"{inp.stem}_preview.png"
    render_views(inp, out, res=ns.res, samples=ns.samples, flat=ns.flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
