"""
recon_backend.py
================
Adapter that exposes :mod:`recon` (analytical SDF reconstruction) through the
generator interface, so ``generate.py --backend recon`` behaves exactly like a
neural generator from the caller's point of view.
"""

from __future__ import annotations

from pathlib import Path

import recon
from config import RAW_GLB_DIR

REQUIRED_MODULES = ["numpy", "scipy", "skimage", "bpy"]


def check_environment() -> dict:
    """The offline backend only needs Blender plus numpy/scipy/scikit-image."""
    from generators import _common

    modules = _common.import_report(REQUIRED_MODULES)
    missing = _common.format_problems(modules)
    problems = [f"missing modules: {missing}"] if missing else []
    return {
        "backend": "recon",
        "available": not problems,
        "repo": "built-in",
        "cuda": False,
        "weights_cached": True,     # analytical: no model weights involved
        "modules": modules,
        "problems": problems,
    }


def generate(desc: dict, ref_image: Path | None = None,
             out_dir: Path = RAW_GLB_DIR) -> dict:
    """Reconstruct one asset; returns the metadata dict written next to the GLB."""
    return recon.reconstruct_asset(desc["key"], out_dir=out_dir)


__all__ = ["generate", "check_environment"]
