"""
showcase.py
===========
Assembles a small dungeon room **from the exported Godot GLBs** and renders it
headlessly.  This is the end-to-end proof that the kit kit-bashes correctly:
grid snapping, module pivots, tiling seams, PBR materials and LOD0 texel density
are all visible in one image.

Layout (metres, all modules on a 1 m grid)
-----------------------------------------
    8 x 8 m floor made of four 4 x 4 m tiles
    two 4 m walls on the south edge, two on the west edge (rotated 90 deg)
    a door module inside the east opening
    two pillars flanking the open east corners
    the treasure chest in the middle of the room

Usage
-----
    python scripts/showcase.py --out renders/showcase_room.png
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

import preview
import utils
from config import GODOT_DIR, RENDERS_DIR

LOG = utils.get_logger("showcase")

# (asset key, x, y, rotation_z_degrees)
#
# Rotation matters: every module is authored with its sculpted face towards -Y
# (the same convention as the concept art), so a module must be rotated for its
# detail to face *into* the room:
#   rot 180 -> faces +Y     rot  90 -> faces +X     rot -90 -> faces -X
LAYOUT = [
    # 8 x 8 m floor: four 4 x 4 m tiles. Pivots are top-centre (walk surface at
    # local z = 0) *and* centre-of-footprint, so tiles belong at 2 m / 6 m.
    ("floor", 2.0, 2.0, 0.0),
    ("floor", 6.0, 2.0, 0.0),
    ("floor", 2.0, 6.0, 0.0),
    ("floor", 6.0, 6.0, 0.0),
    # south wall (y = 0, 0.45 m thick, footprint x +-2 -> modules at 2 m / 6 m)
    # rot 180 puts the sculpted face towards +Y, i.e. into the room.
    ("wall", 2.0, 0.0, 180.0),
    ("wall", 6.0, 0.0, 180.0),
    # west wall (x = 0): rot 90 puts the sculpted face towards +X, into the room
    ("wall", 0.0, 2.0, 90.0),
    ("wall", 0.0, 6.0, 90.0),
    # east entrance: door recessed into the opening, detail facing -X (inside)
    ("door", 7.75, 4.0, -90.0),
    # pillars cap the two open east corners
    ("pillar", 8.0, 0.5, 0.0),
    ("pillar", 8.0, 7.5, 0.0),
    # chest turned towards the hero camera (which looks from -X/-Y)
    ("chest", 4.0, 4.4, 200.0),
]

NAMES = {
    "floor": "SM_Dungeon_Floor_A",
    "wall": "SM_Dungeon_Wall_A",
    "pillar": "SM_Dungeon_Pillar_A",
    "door": "SM_Dungeon_Door_A",
    "chest": "SM_Dungeon_Chest_A",
}

# (label, azimuth deg, elevation deg, framing multiplier on the bbox diagonal)
VIEWS = [
    ("hero", 196.0, 24.0, 1.05),
    ("top", 196.0, 72.0, 1.05),
    ("detail", 132.0, 16.0, 0.40),
]


def _load(asset_key: str):
    """Import a GLB once and return the template object."""
    path = GODOT_DIR / f"{NAMES[asset_key]}_LOD0.glb"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run the pipeline first")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    new = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
    if not new:
        raise RuntimeError(f"no mesh in {path}")
    return new[0]


def build_room() -> list:
    utils.reset_scene()
    templates = {k: _load(k) for k in NAMES}
    placed = []
    for key, x, y, rot in LAYOUT:
        src = templates[key]
        ob = src.copy()
        ob.data = src.data           # instances share mesh data (cheap)
        bpy.context.collection.objects.link(ob)
        ob.location = (x, y, 0.0)
        ob.rotation_euler = (0.0, 0.0, math.radians(rot))
        ob.name = f"{NAMES[key]}_{len(placed):02d}"
        placed.append(ob)

    # Hide the templates from the render.
    for ob in templates.values():
        ob.hide_render = True
        ob.hide_set(True)
    # Objects were just transformed: flush so matrix_world (and therefore all
    # bounding-box maths below) reflects the placement.
    bpy.context.view_layer.update()
    return placed


def render(out_path: Path, res: int = 640, samples: int = 48) -> Path:
    objects = build_room()

    # Frame the room, not the templates.
    visible = [o for o in objects if not o.hide_render]
    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))
    for ob in visible:
        for corner in ob.bound_box:
            w = ob.matrix_world @ Vector(corner)
            for i in range(3):
                mn[i] = min(mn[i], w[i])
                mx[i] = max(mx[i], w[i])
    center = (mn + mx) / 2.0
    radius = (mx - mn).length / 2.0
    diagonal = (mx - mn).length
    LOG.info("room bounds %s .. %s (diagonal %.1f m)",
             [round(v, 2) for v in mn], [round(v, 2) for v in mx], diagonal)

    preview.setup_studio(center, radius, floor_z=mn.z - 0.01)
    scene = bpy.context.scene
    scene.cycles.samples = samples
    scene.render.resolution_x = res
    scene.render.resolution_y = int(res * 0.75)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    cam = preview.make_camera(center, radius)
    shots = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for label, azim, elev, zoom in VIEWS:
        # Frame on the bbox diagonal: ortho_scale is the framed *width*, so this
        # keeps the whole room in frame from any azimuth.
        cam.data.ortho_scale = diagonal * zoom
        if label == "detail":
            target = Vector((mn.x + 0.9, mn.y + 0.9, mn.z + 1.4))
        else:
            target = center
        cam.constraints[0].target.location = target
        preview.place_camera(cam, target, radius, azim, elev)
        frame = out_path.parent / f"_show_{label}.png"
        scene.render.filepath = str(frame)
        bpy.ops.render.render(write_still=True)
        shots.append(frame)
        LOG.info("rendered %s", label)

    sheet = preview.assemble_contact_sheet(
        shots, out_path, labels=[v[0] for v in VIEWS],
        title="Low-Poly Modular Dungeon Kit - kit-bashed room (LOD0)")
    for s in shots:
        s.unlink(missing_ok=True)
    return sheet


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Render a kit-bashed dungeon room")
    ap.add_argument("--out", default=str(RENDERS_DIR / "showcase_room.png"))
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--samples", type=int, default=48)
    ns = ap.parse_args(argv)
    render(Path(ns.out), res=ns.res, samples=ns.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
