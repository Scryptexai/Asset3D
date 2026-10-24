"""
triposr_backend.py
==================
**Fallback** image-to-3D engine: VAST-AI **TripoSR**.

Repo: https://github.com/VAST-AI-Research/TripoSR
Weights: https://huggingface.co/stabilityai/TripoSR

TripoSR is used when SF3D cannot run (older GPU, no CUDA texture-baker build,
model unavailable). It is faster but produces vertex colours by default, so the
adapter runs its texture-atlas baking path (``tsr.bake_texture``) to hand the
Blender stage a mesh with a real UV map and albedo texture — the same artefact
contract as SF3D.

Install (once)
--------------
    git clone https://github.com/VAST-AI-Research/TripoSR external/TripoSR
    pip install -r external/TripoSR/requirements.txt     # incl. torchmcubes, xatlas
    export TRIPOSR_REPO=$PWD/external/TripoSR
    huggingface-cli download stabilityai/TripoSR --local-dir $HF_HOME/triposr

Then::

    python scripts/generate.py --backend triposr --all
"""

from __future__ import annotations

import os
from pathlib import Path

import utils
from config import RAW_GLB_DIR
from generators import _common

LOG = utils.get_logger("gen.triposr")

REPO_ENV = "TRIPOSR_REPO"
MODEL_ID = "stabilityai/TripoSR"
REQUIRED_MODULES = ["torch", "trimesh", "omegaconf", "xatlas"]
#: Optional: nicer background matting (see sf3d_backend).
OPTIONAL_MODULES = ["rembg"]
DEFAULT_MC_RESOLUTION = 256
DEFAULT_TEXTURE_RESOLUTION = 2048


def check_environment() -> dict:
    """Report exactly what is missing for TripoSR to run here."""
    repo = _common.repo_on_path(REPO_ENV, "tsr", "TripoSR")
    modules = _common.import_report(REQUIRED_MODULES)
    cuda = _common.cuda_available()
    weights = _weights_present()

    problems: list[str] = []
    if repo is None:
        problems.append(f"{REPO_ENV} not set (clone VAST-AI-Research/TripoSR)")
    if not cuda:
        problems.append("no CUDA device (CPU inference needs >16 GB RAM anyway)")
    if not weights:
        problems.append("weights not in HF cache (needs huggingface.co access)")
    missing = _common.format_problems(modules)
    if missing:
        problems.append(f"missing modules: {missing}")
    optional = _common.import_report(OPTIONAL_MODULES)

    return {
        "backend": "triposr",
        "available": not problems,
        "repo": str(repo) if repo else None,
        "cuda": cuda,
        "weights_cached": weights,
        "modules": modules,
        "optional_modules": optional,
        "problems": problems,
    }


def _weights_present() -> bool:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    candidates = [Path(hf_home) / "hub" / f"models--{MODEL_ID.replace('/', '--')}"]
    local = os.environ.get("TRIPOSR_MODEL_DIR")
    if local:
        candidates.append(Path(local))
    for path in candidates:
        if not path.is_dir():
            continue
        if any(path.rglob("*.ckpt")) or any(path.rglob("*.safetensors")):
            return True
    return False


def generate(desc: dict, ref_image: Path | None = None,
             out_dir: Path = RAW_GLB_DIR, *,
             mc_resolution: int = DEFAULT_MC_RESOLUTION,
             bake_texture: bool = True,
             texture_resolution: int = DEFAULT_TEXTURE_RESOLUTION,
             foreground_ratio: float = 0.85, device: str | None = None,
             model_path: str | None = None) -> dict:
    """Generate one asset with TripoSR and write ``raw_glb/<SM_Name>.glb``."""
    report = check_environment()
    if not report["available"]:
        raise RuntimeError("TripoSR unavailable: " + " | ".join(report["problems"]))

    import numpy as np                                  # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    from PIL import Image                               # noqa: PLC0415
    from tsr.system import TSR                          # noqa: PLC0415

    ref_image = Path(ref_image or _common.default_ref_image(desc))
    if not ref_image.exists():
        raise FileNotFoundError(ref_image)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = model_path or os.environ.get("TRIPOSR_MODEL_DIR") or MODEL_ID
    LOG.info("loading TripoSR from %s on %s", model_path, device)
    model = TSR.from_pretrained(model_path, config_name="config.yaml",
                                weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to(device)

    image = _common.prepare_reference(ref_image, foreground_ratio)
    # TripoSR expects the object composited on mid-grey (see its run.py).
    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.shape[2] == 4:
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    pil = Image.fromarray((arr * 255.0).astype(np.uint8))

    with torch.no_grad():
        scene_codes = model([pil], device=device)
    meshes = model.extract_mesh(scene_codes, not bake_texture, resolution=mc_resolution)
    mesh = meshes[0]

    if bake_texture:
        from tsr.bake_texture import bake_texture as bake_atlas   # noqa: PLC0415
        import xatlas                                            # noqa: PLC0415
        from PIL import Image as _Image                           # noqa: PLC0415

        LOG.info("baking texture atlas at %d px", texture_resolution)
        baked = bake_atlas(mesh, model, scene_codes[0], texture_resolution)
        verts = mesh.vertices[baked["vmapping"]]
        normals = mesh.vertex_normals[baked["vmapping"]]
        import trimesh                                            # noqa: PLC0415
        albedo = _Image.fromarray(
            (baked["colors"] * 255.0).astype(np.uint8)).transpose(_Image.FLIP_TOP_BOTTOM)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=albedo, metallicFactor=0.0, roughnessFactor=0.85)
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=baked["indices"],
            vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(uv=baked["uvs"], material=material),
            process=False)

    # TripoSR is Y-up; rotate into the kit's Z-up authoring frame.
    mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    norm = _common.normalise_mesh(mesh, desc["dims"], desc.get("pivot", "bottom"),
                                  logger=LOG)
    glb_path = _common.export_glb(mesh, out_dir / f"{desc['name']}.glb")

    meta = _common.write_meta(
        desc, "triposr", glb_path, ref_image, len(mesh.faces),
        extra={
            "engine": {"name": "TripoSR", "repo": "VAST-AI-Research/TripoSR",
                       "model": model_path, "device": device,
                       "mc_resolution": mc_resolution,
                       "bake_texture": bake_texture,
                       "texture_resolution": texture_resolution},
            "has_uv": bool(getattr(mesh.visual, "uv", None) is not None),
            "has_pbr": bool(getattr(mesh.visual, "material", None) is not None),
            "normalisation": norm,
        })
    LOG.info("TripoSR wrote %s (%d tris)", glb_path.name, meta["raw_tris"])
    return meta


__all__ = ["generate", "check_environment"]
