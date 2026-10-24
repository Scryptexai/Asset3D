"""
generate.py
===========
Phase 1/2 driver: reference images -> high-poly PBR GLBs in ``raw_glb/``.

Reference images
----------------
``refs/raw/`` holds the concept renders produced by the image generator with the
prompt template in :data:`PROMPT_TEMPLATE` (kept here so the art direction is
reproducible).  ``refs/`` holds the approved, final copies that the pipeline
consumes.

Image-to-3D
-----------
Engines are pluggable (see :mod:`generators`):

    --backend sf3d      **primary**  Stability AI Stable Fast 3D (UV + PBR native)
    --backend triposr   **fallback** VAST-AI TripoSR (texture-atlas baked)
    --backend recon     offline fallback: SDF + marching cubes, no GPU/weights
    --backend auto      sf3d -> triposr -> recon, first available wins (default)

All engines produce the same artefact: a high-poly ``.glb`` with UVs and a PBR
material plus a metadata ``.json``, in ``raw_glb/``. Everything downstream is
engine-agnostic.

Usage
-----
    python scripts/generate.py --check-backends          # preflight report
    python scripts/generate.py --all                     # auto (SF3D first)
    python scripts/generate.py --backend sf3d --all      # strict
    python scripts/generate.py --backend triposr --asset door
    python scripts/generate.py --backend recon --all     # offline / CPU-only
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    from blender_runner import ensure_host_ready
    ensure_host_ready()

import utils
from config import ASSETS, REFS_DIR, REFS_RAW_DIR, asset_by_key
from generators import (BACKENDS, DEFAULT_BACKEND, FALLBACK_CHAIN, backend_chain,
                        check_all, load_backend)

LOG = utils.get_logger("generate")

#: Art-direction prompt used for every reference render. ``{asset}`` is replaced
#: by the per-asset noun. Keeping it in code makes the concept set reproducible.
PROMPT_TEMPLATE = (
    "A single modular low-poly dungeon {asset}, isolated on a pure white "
    "background, orthographic view, game-ready asset, stylized clean geometry, "
    "chunky stylized stone blocks, no texture noise, coherent grey and warm "
    "brown palette, flat shading with slight ambient occlusion, high contrast "
    "edges for readability, PBR material concept, centered composition, no "
    "ground plane, no scene context, clean topology, low surface detail, "
    "stylized dark fantasy vibe."
)

ASSET_NOUNS = {
    "wall": "stone wall module",
    "floor": "floor tile module",
    "pillar": "pillar column module",
    "door": "wooden door module with iron bands and stone frame",
    "chest": "treasure chest prop with iron bands",
}


def prompt_for(key: str) -> str:
    return PROMPT_TEMPLATE.format(asset=ASSET_NOUNS.get(key, key))


def promote_references() -> list[Path]:
    """Copy approved concept art from ``refs/raw/`` into ``refs/``.

    The pipeline reads ``refs/<name>.png``; ``refs/raw/`` keeps the untouched
    generator output so the art direction stays auditable.
    """
    promoted = []
    for desc in ASSETS:
        src = REFS_RAW_DIR / desc["ref"]
        dst = REFS_DIR / desc["ref"]
        if not src.exists():
            LOG.warning("missing concept art: %s", src)
            continue
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            dst.write_bytes(src.read_bytes())
            promoted.append(dst)
    return promoted


def print_backend_report(reports: list[dict]) -> None:
    """Human-readable availability table for all engines."""
    print("\nImage-to-3D engine preflight")
    print("=" * 74)
    for r in reports:
        flag = "READY  " if r.get("available") else "BLOCKED"
        print(f"[{flag}] {r.get('backend', '?'):8s} "
              f"cuda={r.get('cuda')} weights={r.get('weights_cached')}")
        if r.get("repo"):
            print(f"           repo: {r['repo']}")
        for problem in r.get("problems", []):
            print(f"           - {problem}")
    print("=" * 74)
    ready = [r["backend"] for r in reports if r.get("available")]
    chain = " -> ".join(FALLBACK_CHAIN)
    print(f"chain: {chain}")
    print(f"engines ready on this host: {ready or ['recon (offline)']}\n")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Reference promotion + image-to-3D")
    ap.add_argument("--backend", default="auto",
                    choices=sorted(BACKENDS) + ["auto"],
                    help="engine to use; 'auto' = sf3d -> triposr -> recon")
    ap.add_argument("--asset", action="append", help="asset key (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check-backends", action="store_true",
                    help="print engine availability and exit")
    ap.add_argument("--print-prompts", action="store_true",
                    help="print the art-direction prompt for each asset and exit")
    ns = ap.parse_args(argv)

    if ns.check_backends:
        print_backend_report(check_all())
        return 0

    if ns.print_prompts:
        for key in ASSET_NOUNS:
            print(f"\n[{key}]\n{prompt_for(key)}")
        return 0

    promote_references()
    keys = [a["key"] for a in ASSETS] if ns.all or not ns.asset else ns.asset

    chain = backend_chain(ns.backend)
    strict = ns.backend not in ("auto",)
    LOG.info("engine chain: %s%s", " -> ".join(chain),
             " (strict)" if strict else " (first available wins)")

    ok, failed, used = 0, [], {}
    for key in keys:
        desc = asset_by_key(key)
        last_error = None
        for engine in chain:
            try:
                utils.reset_scene()
                meta = load_backend(engine).generate(desc)
            except Exception as exc:                    # noqa: BLE001
                last_error = f"{engine}: {exc}"
                LOG.warning("%s: %s failed (%s)", desc["name"], engine,
                            str(exc).splitlines()[0][:160])
                if strict:
                    break
                continue
            ok += 1
            used[engine] = used.get(engine, 0) + 1
            LOG.info("%s -> %s (%s tris) via %s", desc["name"], meta["glb"],
                     meta["raw_tris"], engine)
            last_error = None
            break
        if last_error:
            failed.append((key, last_error))

    LOG.info("generated %d/%d assets; engines used: %s", ok, len(keys), used or "none")
    for key, err in failed:
        LOG.error("  %s: %s", key, err)
    if failed:
        LOG.error("hint: run `python scripts/generate.py --check-backends` "
                  "to see why the neural engines are unavailable")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
