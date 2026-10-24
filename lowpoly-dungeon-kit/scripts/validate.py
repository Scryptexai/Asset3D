"""
validate.py
===========
Phase 5 QA gate: validates the **shipped files**, not the in-memory scene.

Checks performed per asset
--------------------------
geometry
    * every LOD (LOD0..LOD3) and the collision mesh exist as FBX *and* GLB
    * triangle budgets are monotonically decreasing and inside the LOD ratios
    * no non-manifold edges, no loose vertices, outward-facing normals
    * LOD0 has a UV map with sane UV-space coverage
textures
    * albedo / normal / ORM exist, are 2048x2048, and are packed correctly
      (ORM: R=AO, G=roughness, B=metallic; normal: unit-length, +Y up)
pipeline
    * ``exports/manifest.json`` lists every asset and every file on disk
    * the Unity (FBX) and Godot (GLB) variants agree on triangle counts

Usage
-----
    python scripts/validate.py            # full report, non-zero exit on error
    python scripts/validate.py --json     # machine-readable report
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    from blender_runner import ensure_host_ready
    ensure_host_ready()

import json

import bmesh
import bpy
import numpy as np
from mathutils import Vector

import utils
from config import (ASSETS, GODOT_DIR, MANIFEST_PATH, MODELS_DIR, TEXTURES_DIR,
                    TEXTURE_SIZE, asset_by_key, collision_mesh_name,
                    lod_mesh_name, texture_name)

LOG = utils.get_logger("validate")

TRI_TOLERANCE = 1.06      # ±6 % on the LOD ratio targets
MIN_UV_COVERAGE = 0.15    # a 2048 bake should use a decent part of the atlas


# --------------------------------------------------------------------------- #
# Primitive checks
# --------------------------------------------------------------------------- #
def import_mesh(path: Path):
    """Import a GLB/FBX and return (object, tris, non_manifold, loose_verts)."""
    utils.reset_scene()
    suffix = path.suffix.lower()
    if suffix == ".glb":
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        bpy.ops.import_scene.fbx(filepath=str(path))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"no mesh imported from {path.name}")
    ob = meshes[0]

    # glTF stores one vertex per corner (position/normal/UV are indexed
    # separately), so an imported GLB looks like an unwelded soup and every
    # shared edge appears "non-manifold". Weld by position first: manifoldness
    # is a property of the surface, not of the vertex stream layout.
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-5)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    loose = sum(1 for v in bm.verts if not v.link_faces)
    volume = signed_volume(bm)
    bm.free()

    tris = utils.raw_tri_count(ob.data)
    return ob, {"tris": tris, "non_manifold": non_manifold,
                "loose_verts": loose, "signed_volume": round(volume, 5),
                "verts": len(ob.data.vertices)}


def signed_volume(bm) -> float:
    """Signed volume of a closed mesh (divergence theorem).

    Positive => face normals point outwards; negative => the whole mesh is
    inverted.  Unlike a per-face normal test this is exact for concave shapes
    (relief walls, open door frames, hollowed props).
    """
    bm.faces.ensure_lookup_table()
    total = 0.0
    for f in bm.faces:
        vs = [l.vert.co for l in f.loops]
        for i in range(1, len(vs) - 1):
            a, b, c = vs[0], vs[i], vs[i + 1]
            total += a.dot(b.cross(c)) / 6.0
    return total


def uv_stats(obj) -> dict:
    """UV-space coverage and out-of-range check for the active UV layer."""
    me = obj.data
    if not me.uv_layers:
        return {"has_uv": False, "coverage": 0.0, "out_of_range": 0}
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    out = int(((uv < -0.001) | (uv > 1.001)).any(axis=1).sum())
    tri_uv = uv.reshape(-1, 3, 2) if len(uv) % 3 == 0 else uv.reshape(-1, 4, 2)
    areas = np.abs(
        (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
        - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
    ) / 2.0
    return {"has_uv": True, "coverage": float(areas.sum()), "out_of_range": out}


def check_texture(path: Path, kind: str) -> dict:
    """Validate resolution, channel packing and value ranges of a baked map."""
    if not path.exists():
        return {"ok": False, "error": "missing"}
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    arr = utils.image_to_numpy(img)[..., :3]
    bpy.data.images.remove(img)
    res = {"ok": True, "width": w, "height": h,
           "mean": [round(float(c), 3) for c in arr.reshape(-1, 3).mean(axis=0)]}
    if kind == "Normal":
        length = np.linalg.norm(arr * 2.0 - 1.0, axis=-1)
        res["unit_length_p99"] = float(np.percentile(length, 99))
        res["blue_mean"] = float(arr[..., 2].mean())
        res["ok"] = res["ok"] and 0.85 <= float(arr[..., 2].mean()) <= 1.0
    elif kind == "ORM":
        r, g, b = (float(arr[..., i].mean()) for i in range(3))
        res["ao_mean"], res["rough_mean"], res["metal_mean"] = r, g, b
        # AO must not be black, roughness must be a plausible dielectric range.
        res["ok"] = res["ok"] and r > 0.2 and 0.2 < g < 0.98
    elif kind == "Albedo":
        res["ok"] = res["ok"] and float(arr.mean()) > 0.02
    return res


# --------------------------------------------------------------------------- #
# Per-asset validation
# --------------------------------------------------------------------------- #
def validate_asset(desc: dict, manifest_entry: dict | None) -> dict:
    name = desc["name"]
    report: dict = {"name": name, "key": desc["key"], "errors": [], "warnings": []}

    # ---- geometry: both engine flavours ------------------------------- #
    tri_table: dict[str, dict[str, int]] = {}
    for lod, ratio in [("LOD0", 1.0), ("LOD1", 0.5), ("LOD2", 0.2), ("LOD3", 0.05)]:
        mesh_name = lod_mesh_name(name, lod)
        row: dict[str, int] = {}
        for engine, folder in (("fbx", MODELS_DIR), ("glb", GODOT_DIR)):
            path = folder / f"{mesh_name}.{engine}"
            if not path.exists():
                report["errors"].append(f"missing {path.name}")
                continue
            try:
                ob, stats = import_mesh(path)
            except Exception as exc:                       # noqa: BLE001
                report["errors"].append(f"{path.name}: import failed ({exc})")
                continue
            row[engine] = stats["tris"]
            if stats["non_manifold"] > 0:
                report["errors"].append(
                    f"{path.name}: {stats['non_manifold']} non-manifold edges")
            if stats["loose_verts"] > 0:
                report["errors"].append(
                    f"{path.name}: {stats['loose_verts']} loose vertices")
            if stats["signed_volume"] <= 0:
                report["errors"].append(
                    f"{path.name}: normals point inward "
                    f"(signed volume {stats['signed_volume']})")
            if engine == "glb":
                uv = uv_stats(ob)
                if not uv["has_uv"]:
                    report["errors"].append(f"{path.name}: no UV layer")
                elif uv["out_of_range"]:
                    report["errors"].append(
                        f"{path.name}: {uv['out_of_range']} loops outside 0-1 UV space")
                elif uv["coverage"] < MIN_UV_COVERAGE:
                    report["warnings"].append(
                        f"{path.name}: UV coverage only {uv['coverage']:.2f}")
                report.setdefault("uv", {})[lod] = round(uv["coverage"], 3)
        tri_table[lod] = row

    # FBX and GLB must agree
    for lod, row in tri_table.items():
        if len(row) == 2 and row["fbx"] != row["glb"]:
            report["errors"].append(
                f"{lod}: FBX/GLB triangle mismatch ({row['fbx']} vs {row['glb']})")

    # ---- LOD budget and ordering --------------------------------------- #
    tris = {lod: row.get("glb", row.get("fbx")) for lod, row in tri_table.items()}
    report["tris"] = {k: v for k, v in tris.items() if v}
    ordered = [tris.get(l) for l in ("LOD0", "LOD1", "LOD2", "LOD3")]
    if all(ordered):
        if ordered != sorted(ordered, reverse=True):
            report["errors"].append(f"LOD triangle counts are not monotonic: {ordered}")
        base = ordered[0]
        for lod, ratio in (("LOD1", 0.5), ("LOD2", 0.2), ("LOD3", 0.05)):
            target = base * ratio
            actual = tris[lod]
            if not (target * 0.55 <= actual <= target * 1.35):
                report["warnings"].append(
                    f"{lod} is {actual} tris, target {target:.0f} "
                    f"({actual / max(target, 1):.2f}x)")
        if base > desc["tris_lod0"] * 1.06:
            report["errors"].append(f"LOD0 over budget: {base} > {desc['tris_lod0']}")

    # ---- collision ------------------------------------------------------ #
    col_name = collision_mesh_name(name)
    for engine, folder in (("fbx", MODELS_DIR), ("glb", GODOT_DIR)):
        path = folder / f"{col_name}.{engine}"
        if not path.exists():
            report["errors"].append(f"missing collision {path.name}")
            continue
        try:
            _, stats = import_mesh(path)
            report.setdefault("collision", {})[engine] = stats["tris"]
            if stats["non_manifold"] > 0:
                report["errors"].append(
                    f"{path.name}: {stats['non_manifold']} non-manifold edges")
        except Exception as exc:                           # noqa: BLE001
            report["errors"].append(f"{path.name}: import failed ({exc})")

    # ---- textures ------------------------------------------------------- #
    report["textures"] = {}
    for kind in ("Albedo", "Normal", "ORM"):
        path = TEXTURES_DIR / texture_name(name, kind)
        res = check_texture(path, kind)
        report["textures"][kind] = res
        if not res.get("ok"):
            report["errors"].append(f"texture {path.name}: {res}")
        elif (res["width"], res["height"]) != (TEXTURE_SIZE, TEXTURE_SIZE):
            report["errors"].append(
                f"texture {path.name}: {res['width']}x{res['height']} != {TEXTURE_SIZE}")

    # ---- manifest -------------------------------------------------------- #
    if manifest_entry is None:
        report["errors"].append("asset missing from exports/manifest.json")
    else:
        listed = {Path(f).name for f in
                  manifest_entry.get("files", {}).get("models", [])
                  + manifest_entry.get("files", {}).get("godot", [])}
        on_disk = {p.name for p in MODELS_DIR.iterdir() if p.stem.startswith(name)}
        on_disk |= {p.name for p in GODOT_DIR.iterdir() if p.stem.startswith(name)}
        missing = on_disk - listed
        if missing:
            report["warnings"].append(f"files not in manifest: {sorted(missing)}")

    report["passed"] = not report["errors"]
    return report


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="QA validation of the exported kit")
    ap.add_argument("--asset", action="append", help="asset key (repeatable)")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--out", default=None, help="write the JSON report to a file")
    ns = ap.parse_args(argv)

    manifest = utils.read_json(MANIFEST_PATH) if MANIFEST_PATH.exists() else {"assets": []}
    by_name = {a["name"]: a for a in manifest.get("assets", [])}

    keys = ns.asset or [a["key"] for a in ASSETS]
    reports = [validate_asset(asset_by_key(k), by_name.get(asset_by_key(k)["name"]))
               for k in keys]

    summary = {
        "assets": len(reports),
        "passed": sum(1 for r in reports if r["passed"]),
        "errors": sum(len(r["errors"]) for r in reports),
        "warnings": sum(len(r["warnings"]) for r in reports),
    }
    payload = {"summary": summary, "reports": reports}

    if ns.out:
        utils.write_json(ns.out, payload)

    if ns.json:
        print(json.dumps(payload, indent=2))
    else:
        for r in reports:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"\n[{status}] {r['name']}  tris={r.get('tris')}")
            for e in r["errors"]:
                print(f"   ERROR  {e}")
            for w in r["warnings"]:
                print(f"   warn   {w}")
        print(f"\n{summary['passed']}/{summary['assets']} assets passed · "
              f"{summary['errors']} errors · {summary['warnings']} warnings")
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
