"""
segment.py
==========
Silhouette extraction for studio-style concept renders (numpy/scipy only).

This module deliberately has **no Blender dependency**: it is used both by
:mod:`recon` (which runs inside Blender) and by the neural image-to-3D adapters
(which must not import ``bpy`` at all, so they can run on a plain CUDA box).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["segment_silhouette", "size_filter", "largest_component"]


def segment_silhouette(rgb: np.ndarray, *, white_threshold: float = 0.93,
                       sat_threshold: float = 0.06) -> np.ndarray:
    """Extract the object mask from a pure-white studio render.

    Two passes are combined so that neither a bright grey stone nor a dark
    object on white is mis-segmented:

    1. ``background`` = pixels that are near-white *and* near-neutral.
    2. A flood fill from the image border keeps only background regions that
       are actually connected to the frame, so a white highlight inside the
       object never becomes a hole.
    """
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = mx - mn
    bg_candidate = (mn > white_threshold) & (sat < sat_threshold)

    lbl, n = ndimage.label(bg_candidate)
    if n:
        border_labels = set(np.unique(np.concatenate([
            lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])))
        border_labels.discard(0)
        border_mask = np.isin(lbl, list(border_labels))
    else:
        border_mask = bg_candidate

    mask = ~border_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = largest_component(mask)
    mask = size_filter(mask)
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(bool)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest 4-connected blob."""
    lbl, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def size_filter(mask: np.ndarray, rel_radius: float = 0.008) -> np.ndarray:
    """Delete thin protrusions while preserving the full extent of thick parts.

    Morphological opening alone would round off legitimately pointed details;
    opening followed by geodesic reconstruction inside the original mask removes
    only structures thinner than the disc, keeping thick regions pixel-exact.
    Long 1-3 px streaks of slightly-warm "white" are otherwise extruded into
    razor-thin fins by the reconstruction.
    """
    from skimage.morphology import disk, opening, reconstruction

    if not mask.any():
        return mask
    r = max(2, int(round(rel_radius * min(mask.shape))))
    opened = opening(mask, disk(r))
    if not opened.any():
        return mask
    return reconstruction(opened, mask, method="dilation")
