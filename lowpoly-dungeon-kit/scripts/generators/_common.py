"""
_common.py
==========
Shared plumbing for the neural image-to-3D backends.

The backends differ only in how they produce a mesh; everything around that
(image preparation, metric normalisation, GLB export, metadata) is identical and
lives here so the adapters stay small and auditable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import utils
from config import KIT_ROOT, REFS_RAW_DIR, RAW_GLB_DIR

LOG = utils.get_logger("gen.common")


# --------------------------------------------------------------------------- #
# Environment probing
# --------------------------------------------------------------------------- #
def repo_on_path(env_var: str, marker: str, label: str) -> Path | None:
    """Put a local checkout on ``sys.path`` and return it (or ``None``)."""
    repo = os.environ.get(env_var)
    if not repo:
        return None
    path = Path(repo).expanduser().resolve()
    if not (path / marker).exists():
        LOG.warning("%s=%s does not contain %s/", env_var, path, marker)
        return None
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:                                   # noqa: BLE001
        return False


def import_report(modules: list[str]) -> dict[str, str]:
    """Try importing each module; map module -> "" (ok) or the error string."""
    out: dict[str, str] = {}
    for name in modules:
        try:
            __import__(name)
            out[name] = ""
        except Exception as exc:                        # noqa: BLE001
            out[name] = f"{type(exc).__name__}: {exc}"
    return out


def format_problems(problems: dict[str, str]) -> str:
    return "; ".join(f"{k} ({v})" for k, v in problems.items() if v)


# --------------------------------------------------------------------------- #
# Image preparation
# --------------------------------------------------------------------------- #
def prepare_reference(ref_image: Path, foreground_ratio: float = 0.85):
    """Load a concept render, cut the background and re-frame it.

    Uses the backend's own ``remove_background``/``resize_foreground`` helpers
    when available (they are part of both SF3D and TripoSR), with a graceful
    fallback to our own flood-fill segmentation so the backend still works if
    ``rembg`` is not installed.
    """
    from PIL import Image

    image = Image.open(ref_image).convert("RGBA")
    alpha_min = image.getchannel("A").getextrema()[0]
    if alpha_min >= 250:                     # opaque concept art: cut it out
        try:
            import rembg
            image = rembg.remove(image, session=rembg.new_session())
        except Exception as exc:                        # noqa: BLE001
            LOG.warning("rembg unavailable (%s); using built-in segmentation", exc)
            image = _fallback_segment(image)
    return _resize_foreground(image, foreground_ratio)


def _fallback_segment(image):
    """Silhouette cut-out using the bpy-free :mod:`segment` helper.

    The neural backends must never import Blender, so they cannot reuse
    ``recon`` (which imports ``bpy``); this keeps them runnable on a plain GPU
    host while sharing the exact same segmentation algorithm.
    """
    import numpy as np
    from PIL import Image

    from segment import segment_silhouette

    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    mask = segment_silhouette(rgb)
    rgb_u8 = np.asarray(image.convert("RGB"))
    rgba = np.dstack([rgb_u8, (mask * 255).astype("uint8")])
    return Image.fromarray(rgba, "RGBA")


def _resize_foreground(image, ratio: float):
    """Scale the cut-out so it fills ``ratio`` of the square canvas."""
    import numpy as np
    from PIL import Image

    a = np.asarray(image.getchannel("A"))
    ys, xs = np.where(a > 8)
    if xs.size == 0:
        return image.resize((1024, 1024), Image.LANCZOS)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = image.crop((x0, y0, x1, y1))
    side = max(crop.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))
    target = int(round(side / max(ratio, 1e-3)))
    square = Image.new("RGBA", (target, target), (0, 0, 0, 0))
    square.paste(canvas, ((target - side) // 2, (target - side) // 2))
    return square.resize((1024, 1024), Image.LANCZOS)


# --------------------------------------------------------------------------- #
# Metric normalisation + export
# --------------------------------------------------------------------------- #
def normalise_mesh(mesh, dims: tuple[float, float, float], pivot: str = "bottom",
                   logger=None):
    """Centre the mesh in XY, sit it on the ground and fit it to the kit metric.

    A generated mesh is scaled **uniformly** (never squashed) so the model's
    proportions survive. The scale is chosen so the mesh fits *inside* the
    module box: the most constrained axis lands exactly on its target dimension
    and every other axis stays within its budget. Non-uniform fitting is
    available for backends whose proportions are already authored
    (``uniform=False``).
    """
    import numpy as np

    logger = logger or LOG
    target = np.asarray(dims, dtype=float)                  # Blender XYZ metres
    # glTF/trimesh is Y-up: map Blender (x, y=depth, z=height) -> (x, y, z=depth)
    target_gltf = np.asarray([target[0], target[2], target[1]], dtype=float)

    extents = np.asarray(mesh.extents, dtype=float)
    axis = int(np.argmax(extents / np.maximum(target_gltf, 1e-6)))
    scale = float(target_gltf[axis] / max(extents[axis], 1e-6))
    mesh.apply_scale(scale)

    centre = mesh.bounding_box.centroid
    mesh.apply_translation([-centre[0], -centre[1], 0.0])
    mn = mesh.bounds[0]
    if pivot == "bottom":
        mesh.apply_translation([0.0, -mn[1], 0.0])
    elif pivot == "top":
        mesh.apply_translation([0.0, -mesh.bounds[1][1], 0.0])

    deviation = {
        "axis_fitted": "xyz"[axis].upper(),
        "scale": round(scale, 4),
        "extent_m": [round(float(v), 3) for v in mesh.extents],
        "target_m": [round(float(v), 3) for v in target_gltf],
    }
    logger.info("normalised mesh: fitted %s to %.2f m (scale %.4f), extents %s",
                deviation["axis_fitted"], target_gltf[axis], scale,
                deviation["extent_m"])
    return deviation


def export_glb(mesh, path: Path, *, include_normals: bool = True) -> Path:
    """Export a trimesh object as a binary GLB with PBR material intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), include_normals=include_normals)
    if not path.exists():
        raise RuntimeError(f"export produced no file: {path}")
    return path


def write_meta(desc: dict, backend: str, glb_path: Path, ref_image: Path,
               tris: int, extra: dict | None = None) -> dict:
    """Write the per-asset metadata JSON next to the GLB."""
    meta = {
        "key": desc["key"],
        "name": desc["name"],
        "backend": backend,
        "source_image": _rel(ref_image),
        "glb": _rel(glb_path),
        "dims_m": list(desc["dims"]),
        "raw_tris": int(tris),
    }
    meta.update(extra or {})
    utils.write_json(Path(glb_path).with_suffix(".json"), meta)
    return meta


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(KIT_ROOT))
    except ValueError:
        return str(path)


def default_ref_image(desc: dict) -> Path:
    return REFS_RAW_DIR / desc["ref"]


def default_out_dir() -> Path:
    return RAW_GLB_DIR
