"""
config.py
=========
Single source of truth for the Low-Poly Modular Dungeon Kit build.

Every path, asset descriptor, texture size and LOD ratio used anywhere in the
pipeline lives here.  Nothing in this module imports ``bpy`` so it can be used
both by plain-CPython tooling and by scripts executed inside Blender.

Axis convention
---------------
Blender is Z-up; Unity is Y-up (the FBX exporter converts for us) and glTF is
Y-up (the glTF exporter converts as well).  All dimensions below are expressed
in **Blender space** as ``(width_x, depth_y, height_z)`` in metres so the kit is
metric and snaps to a 1 m grid, which is what Unity/Godot users expect.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
KIT_ROOT = Path(__file__).resolve().parents[1]

REFS_DIR = KIT_ROOT / "refs"                 # final 2D concept art (Phase 1)
REFS_RAW_DIR = REFS_DIR / "raw"              # untouched generator output
RAW_GLB_DIR = KIT_ROOT / "raw_glb"           # image-to-3D output (Phase 2)
SCRIPTS_DIR = KIT_ROOT / "scripts"

EXPORTS_DIR = KIT_ROOT / "exports"
MODELS_DIR = EXPORTS_DIR / "Models"          # Unity-ready FBX
TEXTURES_DIR = EXPORTS_DIR / "Textures"      # shared PBR texture set
GODOT_DIR = EXPORTS_DIR / "Godot"            # Godot-ready GLB

RENDERS_DIR = KIT_ROOT / "renders"           # QA / marketplace previews
TEMP_DIR = KIT_ROOT / "temp"                 # scratch, git-ignored
DOCS_DIR = KIT_ROOT / "Documentation"
MANIFEST_PATH = EXPORTS_DIR / "manifest.json"

for _d in (REFS_DIR, REFS_RAW_DIR, RAW_GLB_DIR, MODELS_DIR, TEXTURES_DIR,
           GODOT_DIR, RENDERS_DIR, TEMP_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Global pipeline settings
# --------------------------------------------------------------------------- #
#: Bumped whenever the pipeline's output changes in a way that invalidates
#: cached exports (materials, bake settings, LOD policy, ...). ``--skip-existing``
#: compares this against the value stored next to each asset's exports.
PIPELINE_VERSION = "1.1.0"

TEXTURE_SIZE = 2048          # marketplace standard for hero props
BAKE_MARGIN = 16             # px of padding around UV islands
CAGE_EXTRUSION = 0.35        # metres, projection cage for the albedo transfer
UV_ISLAND_MARGIN = 0.004

# name -> fraction of LOD0 triangle count
LOD_RATIOS = {
    "LOD0": 1.00,
    "LOD1": 0.50,
    "LOD2": 0.20,
    "LOD3": 0.05,
}
LOD_NAMES = list(LOD_RATIOS)

# Naming convention (Unreal/Unity style prefixes)
PREFIX_STATIC_MESH = "SM_"
PREFIX_TEXTURE = "T_"

# --------------------------------------------------------------------------- #
# Asset descriptors
# --------------------------------------------------------------------------- #
# mode:
#   "relief"  -> bas-relief: flat back, front displaced by the concept's
#                luminance height map (walls, floors, doors - flat modules that
#                must keep a crisp, tileable silhouette)
#   "inflate" -> Teddy-style medial-axis inflation, rounded closed volume
#                (pillars, free-standing props)
# plane:
#   "xz" -> image lies in the X/Z plane, depth goes to Y (elevation views)
#   "xy" -> image lies in the X/Y plane, depth goes to Z (top-down views)
ASSETS = [
    {
        "key": "wall",
        "name": "SM_Dungeon_Wall_A",
        "ref": "wall.png",
        "dims": (4.0, 0.45, 3.0),   # 4 m module, 3 m tall, snaps to grid
        "plane": "xz",
        "mode": "relief",
        "voxel_res": 104,           # longest-axis samples for the SDF grid
        "core": 0.42,               # fraction of total depth that is solid core
        "relief": 0.58,             # fraction of depth used by the relief bump
        "tris_lod0": 1400,
        "uv_angle_limit": 66.0,
        "pivot": "bottom",          # origin at bottom-centre, on the grid
        "collision": "convex_hull",
        "tags": ["modular", "architecture", "wall"],
    },
    {
        "key": "floor",
        "name": "SM_Dungeon_Floor_A",
        "ref": "floor.png",
        "dims": (4.0, 4.0, 0.35),
        "plane": "xy",
        "mode": "relief",
        "voxel_res": 104,
        "core": 0.55,
        "relief": 0.45,
        "tris_lod0": 1200,
        "uv_angle_limit": 66.0,
        "pivot": "top",             # walk surface sits at local z = 0
        "collision": "convex_hull",
        "tags": ["modular", "architecture", "floor"],
    },
    {
        "key": "pillar",
        "name": "SM_Dungeon_Pillar_A",
        "ref": "pillar.png",
        "dims": (1.0, 1.0, 3.0),
        "plane": "xz",
        "mode": "lathe",
        "voxel_res": 118,
        "core": 0.0,
        "relief": 1.0,
        "tris_lod0": 1500,
        "uv_angle_limit": 66.0,
        "pivot": "bottom",
        "collision": "convex_hull",
        "tags": ["modular", "architecture", "pillar"],
    },
    {
        "key": "door",
        "name": "SM_Dungeon_Door_A",
        "ref": "door.png",
        "dims": (2.0, 0.5, 3.0),
        "plane": "xz",
        "mode": "relief",
        "voxel_res": 104,
        "core": 0.5,
        "relief": 0.5,
        "tris_lod0": 1400,
        "uv_angle_limit": 66.0,
        "pivot": "bottom",
        "collision": "convex_hull",
        "tags": ["modular", "architecture", "door"],
    },
    {
        "key": "chest",
        "name": "SM_Dungeon_Chest_A",
        "ref": "chest.png",
        "dims": (1.0, 0.7, 0.8),
        "plane": "xz",
        # A chest rests against a surface: flat back + sculpted front reads far
        # better in-game than a fully inflated pillow.
        "mode": "relief",
        "voxel_res": 96,
        # A chest must read as a solid volume: most of its depth is core, the
        # relief only carries the plank/iron detail.
        "core": 0.74,
        "relief": 0.26,
        "relief_sigma": 2.2,
        "tris_lod0": 1300,
        "smooth_iters": 0,
        "uv_angle_limit": 66.0,
        "pivot": "bottom",
        "collision": "convex_hull",
        "tags": ["prop", "interactive", "container"],
    },
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def asset_by_key(key: str) -> dict:
    """Return the descriptor for ``key`` (e.g. ``"wall"``)."""
    for a in ASSETS:
        if a["key"] == key:
            return a
    raise KeyError(f"unknown asset key: {key!r}")


def texture_name(asset_name: str, kind: str) -> str:
    """``SM_Dungeon_Wall_A`` + ``Albedo`` -> ``T_Dungeon_Wall_A_Albedo_2048.png``.

    The ``SM_`` prefix is replaced by ``T_`` per the kit naming convention.
    """
    stem = asset_name[len(PREFIX_STATIC_MESH):] if asset_name.startswith(PREFIX_STATIC_MESH) else asset_name
    return f"{PREFIX_TEXTURE}{stem}_{kind}_{TEXTURE_SIZE}.png"


def lod_mesh_name(asset_name: str, lod: str) -> str:
    return f"{asset_name}_{lod}"


def collision_mesh_name(asset_name: str) -> str:
    return f"{PREFIX_STATIC_MESH}{asset_name[len(PREFIX_STATIC_MESH):]}_Collision"


__all__ = [
    "KIT_ROOT", "REFS_DIR", "REFS_RAW_DIR", "RAW_GLB_DIR", "SCRIPTS_DIR",
    "EXPORTS_DIR", "MODELS_DIR", "TEXTURES_DIR", "GODOT_DIR", "RENDERS_DIR",
    "TEMP_DIR", "DOCS_DIR", "MANIFEST_PATH", "TEXTURE_SIZE", "BAKE_MARGIN",
    "CAGE_EXTRUSION", "UV_ISLAND_MARGIN", "LOD_RATIOS", "LOD_NAMES", "ASSETS",
    "asset_by_key", "texture_name", "lod_mesh_name", "collision_mesh_name",
]
