"""
sf3d_backend.py
===============
**Primary** image-to-3D engine: Stability AI **Stable Fast 3D (SF3D)**.

Repo: https://github.com/Stability-AI/stable-fast-3d
Weights: https://huggingface.co/stabilityai/stable-fast-3d

SF3D is the engine of record for this kit because it is the only one of the two
that natively outputs **UV-unwrapped geometry with a PBR material** (albedo +
metallic/roughness + normal, baked into a texture atlas). That means the raw GLB
in ``raw_glb/`` already carries real PBR maps, and the Blender stage's job is
optimisation (retopo/bake/LOD) rather than inventing materials.

Install (once, on a CUDA workstation)
-------------------------------------
    git clone https://github.com/Stability-AI/stable-fast-3d external/stable-fast-3d
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r external/stable-fast-3d/requirements.txt
    # SF3D's texture baker / UV unwrapper are compiled CUDA extensions:
    pip install external/stable-fast-3d/texture_baker external/stable-fast-3d/uv_unwrapper
    export SF3D_REPO=$PWD/external/stable-fast-3d
    huggingface-cli download stabilityai/stable-fast-3d --local-dir $HF_HOME/sf3d

Then::

    python scripts/generate.py --backend sf3d --all

Notes
-----
* ``remesh="none"`` and ``vertex_count=-1`` keep the dense reconstruction; the
  Blender pipeline owns decimation so that the bake source keeps maximum detail.
* ``bake_resolution=2048`` matches the kit's final texture size, so the albedo
  transfer in Phase 3 stays sharp.
* The mesh is exported as GLB with normals, then normalised to the kit's metric
  module size by :mod:`generators._common`.
"""

from __future__ import annotations

import os
from pathlib import Path

import utils
from config import RAW_GLB_DIR
from generators import _common

LOG = utils.get_logger("gen.sf3d")

REPO_ENV = "SF3D_REPO"
MODEL_ID = "stabilityai/stable-fast-3d"
REQUIRED_MODULES = ["torch", "trimesh", "omegaconf"]
#: Optional: nicer background matting. Without it the adapter falls back to the
#: pipeline's own silhouette segmentation, which is what the shipped kit used.
OPTIONAL_MODULES = ["rembg"]
#: baked texture atlas resolution (matches exports/Textures/*_2048.png)
DEFAULT_BAKE_RESOLUTION = 2048


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def check_environment() -> dict:
    """Report exactly what is missing for SF3D to run here.

    Used by ``generate.py --check-backends`` so a failure is diagnosed *before*
    a long download/build, and so the report is actionable.
    """
    repo = _common.repo_on_path(REPO_ENV, "sf3d", "SF3D")
    modules = _common.import_report(REQUIRED_MODULES)
    cuda = _common.cuda_available()
    weights = _weights_present()

    problems: list[str] = []
    if repo is None:
        problems.append(f"{REPO_ENV} not set (clone Stability-AI/stable-fast-3d)")
    if not cuda:
        problems.append("no CUDA device (SF3D's texture baker needs one)")
    if not weights:
        problems.append("weights not in HF cache (needs huggingface.co access)")
    missing = _common.format_problems(modules)
    if missing:
        problems.append(f"missing modules: {missing}")
    optional = _common.import_report(OPTIONAL_MODULES)

    return {
        "backend": "sf3d",
        "available": not problems,
        "repo": str(repo) if repo else None,
        "cuda": cuda,
        "weights_cached": weights,
        "modules": modules,
        "optional_modules": optional,
        "problems": problems,
    }


def _weights_present() -> bool:
    """Cheap check for the HF snapshot in the local cache (no network needed)."""
    candidates = []
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    candidates.append(Path(hf_home) / "hub" / f"models--{MODEL_ID.replace('/', '--')}")
    local = os.environ.get("SF3D_MODEL_DIR")
    if local:
        candidates.append(Path(local))
    for path in candidates:
        if not path.exists():
            continue
        if path.is_dir() and any(path.rglob("*.safetensors")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(desc: dict, ref_image: Path | None = None,
             out_dir: Path = RAW_GLB_DIR, *, bake_resolution: int = DEFAULT_BAKE_RESOLUTION,
             remesh: str = "none", vertex_count: int = -1,
             foreground_ratio: float = 0.85, device: str | None = None,
             model_path: str | None = None) -> dict:
    """Generate one asset with SF3D and write ``raw_glb/<SM_Name>.glb``."""
    report = check_environment()
    if not report["available"]:
        raise RuntimeError("SF3D unavailable: " + " | ".join(report["problems"]))

    import torch                                        # noqa: PLC0415
    from sf3d.system import SF3D                        # noqa: PLC0415
    from sf3d.utils import get_device                   # noqa: PLC0415

    ref_image = Path(ref_image or _common.default_ref_image(desc))
    if not ref_image.exists():
        raise FileNotFoundError(ref_image)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or get_device()
    model_path = model_path or os.environ.get("SF3D_MODEL_DIR") or MODEL_ID
    LOG.info("loading SF3D from %s on %s", model_path, device)
    model = SF3D.from_pretrained(model_path, config_name="config.yaml",
                                 weight_name="model.safetensors")
    model.to(device)
    model.eval()

    image = _common.prepare_reference(ref_image, foreground_ratio)
    LOG.info("running SF3D inference (bake_resolution=%d, remesh=%s)",
             bake_resolution, remesh)
    with torch.no_grad():
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if device == "cuda" else _nullcontext())
        with ctx:
            mesh, glob = model.run_image(
                [image],
                bake_resolution=bake_resolution,
                remesh=remesh,
                vertex_count=vertex_count,
            )
    if isinstance(mesh, list):
        mesh = mesh[0]
    # SF3D is Y-up; rotate into the kit's Z-up authoring frame before measuring.
    mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    norm = _common.normalise_mesh(mesh, desc["dims"], desc.get("pivot", "bottom"),
                                  logger=LOG)
    glb_path = _common.export_glb(mesh, out_dir / f"{desc['name']}.glb")

    meta = _common.write_meta(
        desc, "sf3d", glb_path, ref_image, len(mesh.faces),
        extra={
            "engine": {"name": "Stable Fast 3D", "repo": "Stability-AI/stable-fast-3d",
                       "model": model_path, "device": device,
                       "bake_resolution": bake_resolution, "remesh": remesh,
                       "vertex_count": vertex_count},
            "has_uv": bool(getattr(mesh.visual, "uv", None) is not None),
            "has_pbr": bool(getattr(mesh.visual, "material", None) is not None),
            "normalisation": norm,
        })
    LOG.info("SF3D wrote %s (%d tris, UV=%s, PBR=%s)", glb_path.name,
             meta["raw_tris"], meta["has_uv"], meta["has_pbr"])
    return meta


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


__all__ = ["generate", "check_environment"]
