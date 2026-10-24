"""
build_all.py
============
One-command rebuild of the whole kit.

    python scripts/build_all.py                    # auto (SF3D -> TripoSR -> recon)
    python scripts/build_all.py --backend sf3d     # strict: require SF3D
    python scripts/build_all.py --backend recon    # offline / CPU-only build
    python scripts/build_all.py --skip-generate --skip-preview

Stages
------
1. ``generate.py``     concept art -> high-poly PBR GLB      (raw_glb/)
2. ``pipeline.py``     optimise, bake, LOD, collision, export (exports/)
3. ``validate.py``     QA gate over the exported files        (renders/qa_report.json)
4. ``preview.py``      contact sheets rendered from the exported GLBs (renders/)

Every stage runs through :mod:`blender_runner`, so the script works with either a
``blender`` binary or the ``bpy`` module.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import utils
from config import ASSETS, GODOT_DIR, RENDERS_DIR, asset_by_key, lod_mesh_name
from blender_runner import run_script
from generators import FALLBACK_CHAIN

LOG = utils.get_logger("build_all")


def stage(name: str, script: str, args: list[str], *, timeout: int = 3600) -> int:
    LOG.info("=== %s ===", name)
    t0 = time.perf_counter()
    proc = run_script(script, args, timeout=timeout, verbose=False)
    dt = time.perf_counter() - t0
    LOG.info("%s finished in %.1fs (exit %d)", name, dt, proc.returncode)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Rebuild the Low-Poly Modular Dungeon Kit")
    ap.add_argument("--backend", default="auto",
                    help="image-to-3D engine for stage 1: auto|sf3d|triposr|recon "
                         f"(auto = {' -> '.join(FALLBACK_CHAIN)})")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--skip-pipeline", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-preview", action="store_true")
    ap.add_argument("--glb-texture", default="webp", choices=["webp", "jpeg", "png"])
    ap.add_argument("--skip-existing", action="store_true",
                    help="reuse assets already exported by this pipeline version")
    ns = ap.parse_args(argv)

    failures: list[str] = []

    if not ns.skip_generate:
        if stage("generate", "generate.py", ["--backend", ns.backend, "--all"]):
            failures.append("generate")

    if not ns.skip_pipeline:
        pipe_args = ["--all", "--glb-texture", ns.glb_texture]
        if ns.skip_existing:
            pipe_args.append("--skip-existing")
        if stage("pipeline", "pipeline.py", pipe_args, timeout=7200):
            failures.append("pipeline")

    if not ns.skip_validate:
        report = RENDERS_DIR / "qa_report.json"
        if stage("validate", "validate.py", ["--out", str(report)]):
            failures.append("validate")

    if not ns.skip_preview:
        for desc in ASSETS:
            glb = GODOT_DIR / f"{lod_mesh_name(desc['name'], 'LOD0')}.glb"
            if not glb.exists():
                LOG.warning("preview skipped, missing %s", glb.name)
                continue
            out = RENDERS_DIR / f"{desc['key']}_lod0_preview.png"
            if stage(f"preview:{desc['key']}", "preview.py",
                     ["--input", str(glb), "--out", str(out),
                      "--res", "480", "--samples", "40"]):
                failures.append(f"preview:{desc['key']}")

    if failures:
        LOG.error("build finished with failures: %s", ", ".join(failures))
        return 1
    LOG.info("build finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
