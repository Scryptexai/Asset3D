"""
recon.py
========
Image -> triangle-mesh reconstruction that runs **inside Blender**.

This module is the analytical fallback backend of the image-to-3D stage.  It
turns a 2D concept render into a *real* triangle soup: the silhouette is
segmented, a signed distance field is generated in metric space and the surface
is extracted with marching cubes, yielding vertices + faces that are then
written to a binary glTF (``.glb``) together with a PBR texture set.

Why a fallback exists
---------------------
The shipped kit was produced in a sandbox with (a) no CUDA device and (b) no
network route to model weights, which makes neural generators impossible to
run *there*.  The pipeline therefore exposes a generator interface
(``scripts/generators/``); TRELLIS.2 and Hunyuan3D 2.0 adapters plug into the
exact same slot and produce the same kind of artefact (a high-poly textured
GLB in ``raw_glb/``) on a CUDA machine.  Nothing downstream cares which
backend produced the file.

Geometry model
--------------
Two reconstruction modes are supported, both of them volumetric (never
procedural primitives):

``slab``
    ``half_thickness(u, v) = core/2 + relief * f(distance_to_silhouette)``
    A solid core with a relief bump that follows the distance transform of the
    silhouette.  Ideal for architecture modules (walls, floors, doors): keeps
    the back face flat and the front face sculpted.

``inflate``
    ``half_thickness(u, v) = max_thickness * g(distance_to_silhouette)``
    Teddy-style medial-axis inflation: produces a rounded, closed volume from
    an arbitrary silhouette (pillars, props such as the chest).

Because the field is a distance field, the extracted surface is watertight by
construction (no boundary edges), which the QA stage verifies.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

if __name__ == "__main__":  # allow `python scripts/recon.py` on bare containers
    from blender_runner import ensure_host_ready
    ensure_host_ready()

import numpy as np
from scipy import ndimage
from skimage import measure

import bpy  # noqa: F401  (this module is executed inside Blender)

import utils
from config import RAW_GLB_DIR, REFS_RAW_DIR, TEMP_DIR, asset_by_key

LOG = utils.get_logger("recon")

# --------------------------------------------------------------------------- #
# Image segmentation
# --------------------------------------------------------------------------- #
def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as a float32 ``(H, W, 3)`` array, bottom-up (Blender order)."""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(h, w, 4)[..., :3].copy()
    arr[..., :3] = np.clip(arr[..., :3], 0.0, 1.0)
    # Un-premultiply nothing: concept art is stored as straight alpha.
    alpha = buf.reshape(h, w, 4)[..., 3].copy()
    bpy.data.images.remove(img)
    if alpha.min() < 0.99:  # PNG with real alpha channel
        a = np.clip(alpha, 0, 1)[..., None]
        arr = arr * a + (1.0 - a)
    return arr


def composite_on_white(path: str | Path) -> np.ndarray:
    """Return the reference flattened onto white (alpha included)."""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(h, w, 4)
    rgb, a = px[..., :3], np.clip(px[..., 3:4], 0, 1)
    bpy.data.images.remove(img)
    return np.clip(rgb * a + (1.0 - a), 0, 1)


# Segmentation lives in :mod:`segment` so the neural image-to-3D backends can use
# it without importing Blender. Re-exported here for backwards compatibility.
from segment import largest_component, segment_silhouette, size_filter  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# Signed distance field
# --------------------------------------------------------------------------- #
def _resize_mask(mask: np.ndarray, nw: int, nh: int) -> np.ndarray:
    zoom = (nh / mask.shape[0], nw / mask.shape[1])
    out = ndimage.zoom(mask.astype(np.float32), zoom, order=1)
    return np.clip(out, 0.0, 1.0)


def build_sdf(mask: np.ndarray, *, rgb: np.ndarray | None = None,
              dims_uv: tuple[float, float], depth_m: float, plane: str,
              mode: str, voxel_res: int, core: float,
              relief: float, relief_sigma: float = 1.2) -> tuple[np.ndarray, tuple]:
    """Rasterise the silhouette into a metric signed distance volume.

    Parameters
    ----------
    mask
        Boolean silhouette in image space (bottom-up rows).
    rgb
        Concept colour, used by ``relief`` mode as the height source.
    dims_uv, depth_m
        Metric size of the module in the image plane and along the third axis.
    mode
        * ``"relief"``  - bas-relief extrusion: flat back, front surface
          displaced by the concept's luminance height map.  Crisp rectangular
          silhouette, ideal for modular architecture (walls, floors, doors).
        * ``"inflate"`` - Teddy-style medial-axis inflation of the silhouette
          into a rounded closed volume (pillars, free-standing props).
        * ``"slab"``    - symmetric thickness profile, legacy single-piece
          modules (kept for completeness).

    Returns
    -------
    (sdf, spacing)
        ``sdf`` is ``(nz, nh, nw)`` and positive inside the solid; ``spacing``
        is the metric sample pitch ``(sz, sy, sx)``.
    """
    h, w = mask.shape
    longest = max(dims_uv)
    s = longest / float(voxel_res)                  # metres per voxel (longest axis)

    # Grid resolution follows the *metric* size; the mask is resampled onto it,
    # absorbing any mismatch between concept aspect ratio and module size.
    nw = int(np.clip(round(dims_uv[0] / s), 24, 320))
    nh = int(np.clip(round(dims_uv[1] / s), 24, 320))

    m = _clean_binary_mask(_resize_mask(mask, nw, nh) > 0.5)
    if not m.any():
        raise RuntimeError("empty silhouette after resampling")

    sx = dims_uv[0] / nw          # metres per voxel along the image X axis
    sy = dims_uv[1] / nh          # metres per voxel along the image Y axis

    # ------------------------------------------------------------------ #
    # Modular borders.  The silhouette is resampled into the module's metric
    # box; every side of that box the *art itself* reaches is a tiling seam.
    # A seam must be a flush, full-thickness, perfectly straight face, so the
    # field there must NOT be tapered.  Sides the artwork does not reach (the
    # arched top of the door) keep their tapered, rounded profile.
    # ------------------------------------------------------------------ #
    m_ext = m.copy()
    if m[:, 0].mean() > 0.15:
        m_ext[:, :3] = True
    if m[:, -1].mean() > 0.15:
        m_ext[:, -3:] = True
    if m[0, :].mean() > 0.15:
        m_ext[:3, :] = True
    if m[-1, :].mean() > 0.15:
        m_ext[-3:, :] = True

    # Distance (metres) from every pixel to the silhouette boundary.
    dt = ndimage.distance_transform_edt(m_ext).astype(np.float32)
    dt *= float(0.5 * (sx + sy))
    dt_max = float(dt.max())
    if dt_max <= 0:
        raise RuntimeError("degenerate silhouette")
    t = dt / dt_max                                  # 0 at border, 1 at core

    # No distance-based taper. The solid keeps its full thickness right up to
    # the mask boundary, so marching cubes closes the surface with a clean,
    # near-vertical wall through the zero crossing. A wide taper would thin the
    # module's tiling seams to a blade and stop the pieces butting together.
    ramp = np.ones_like(dt, dtype=np.float32)

    if mode == "lathe":
        return _lathe_sdf(m, dims_uv=dims_uv, depth_m=depth_m, s=(sx, sy))

    if mode == "relief":
        if rgb is None:
            raise ValueError("relief mode needs the concept colour array")
        # Height detail only exists inside the artwork; the extended seam bands
        # therefore come out as flat, full-thickness walls.
        height = _normalised_height(_resize_to(rgb, (nh, nw)), m,
                                    smooth_sigma=relief_sigma)
        # Height axis runs from the back plane (0) to the front (depth_m).
        front = (core * depth_m + relief * depth_m * height) * ramp
        front = np.maximum(front, 0.0)
        # A field that thins below one voxel makes the front and back
        # iso-surfaces merge, and marching cubes then emits a comb of razor
        # slivers along the whole silhouette. Keep a minimum solid thickness
        # over the eroded core and let it drop to zero on the outer ring.
        solid_core = ndimage.binary_erosion(m, np.ones((3, 3), bool), iterations=1)
        front = np.where(solid_core, np.maximum(front, 2.0 * s), 0.0)
        sz = max(min(s, depth_m / 48.0), 1e-4)
        nz = int(np.clip(math.ceil(depth_m / sz) + 4, 8, 160))
        # Samples straddle the back plane so the flat back face never lands
        # exactly on the volume border (which would open the mesh).
        z = np.linspace(-0.5, nz - 0.5, nz, dtype=np.float32) * sz
        sdf = np.minimum(front[None, :, :] - z[:, None, None], z[:, None, None])
        # One negative ring around the volume closes the surface at the module
        # border while keeping the seam walls at *full* thickness, so tiles and
        # wall modules butt together without a bevelled gap.
        sdf = np.pad(sdf, 1, mode="constant", constant_values=-1.0)
        return sdf.astype(np.float32), (sz, sy, sx)

    half_depth = max(depth_m, 1e-3) * 0.5
    if mode == "slab":
        fade = ramp
        half = (core * half_depth * (0.35 + 0.65 * fade)
                + relief * half_depth * (t ** 0.85) * fade)
    elif mode == "inflate":
        # Medial-axis inflation; power < 1 fattens thin appendages into tubes.
        half = half_depth * (t ** 0.55)
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"unknown reconstruction mode {mode!r}")

    half = half * ramp
    half = np.where(dt > 0.0, np.maximum(half, 2.0 * min(sx, sy)), 0.0)
    half = ndimage.gaussian_filter(half, sigma=0.9, mode="nearest")
    half[0, :] = half[-1, :] = 0.0
    half[:, 0] = half[:, -1] = 0.0

    sz = max(min(s, depth_m / 48.0), 1e-4)
    nz = int(np.clip(math.ceil(depth_m / sz) + 4, 8, 160))
    z = (np.arange(nz, dtype=np.float32) - (nz - 1) / 2.0) * sz
    sdf = half[None, :, :] - np.abs(z)[:, None, None]
    return sdf.astype(np.float32), (sz, sy, sx)


def _lathe_sdf(mask: np.ndarray, *, dims_uv: tuple[float, float],
               depth_m: float, s: tuple[float, float]) -> tuple[np.ndarray, tuple]:
    """Surface-of-revolution reconstruction from a front-view silhouette.

    Every image row becomes one horizontal cross-section of the module: its
    width and centre define a **superellipse** (rounded square) whose depth is
    derived from the row width.  Vertical caps are cut flat, so the result is a
    true modular column instead of the tapered "pencil" that medial-axis
    inflation produces.

    Returns ``(sdf, spacing)`` with ``sdf`` shaped ``(ny, nz, nx)``.
    """
    sx, sy = s
    nh, nw = mask.shape
    rows = np.arange(nh)

    has = mask.any(axis=1)
    if not has.any():
        raise RuntimeError("empty silhouette")
    # Row width / centre in metres, edge-extended into the two padding rows that
    # exist so the caps can be cut inside the sampling grid.
    widths = np.zeros(nh, dtype=np.float32)
    centres = np.zeros(nh, dtype=np.float32)
    for r in rows[has]:
        idx = np.where(mask[r])[0]
        widths[r] = (idx[-1] - idx[0] + 1) * sx
        centres[r] = ((idx[0] + idx[-1]) / 2.0) * sx
    first, last = np.where(has)[0][[0, -1]]
    widths[:first] = widths[first]
    widths[last + 1:] = widths[last]
    centres[:first] = centres[first]
    centres[last + 1:] = centres[last]

    # Longitudinal smoothing keeps the profile chunky instead of jittery.
    widths = ndimage.gaussian_filter1d(widths, max(1.0, nh / 90.0), mode="nearest")
    centres = ndimage.gaussian_filter1d(centres, max(1.0, nh / 90.0), mode="nearest")

    # Superellipse exponent: 2 = ellipse, 2.8 ~ rounded square (stylized stone).
    n_exp = 2.8
    # Depth/width ratio so the section fills the module's depth envelope.
    depth_ratio = np.clip(depth_m / max(float(widths.max()), 1e-4), 0.35, 1.0)

    x = (np.arange(nw, dtype=np.float32) * sx)
    z = np.zeros(nh, dtype=np.float32)
    z[1:-1] = (rows[1:-1] - first) * sy
    z[0] = -sy
    z[-1] = (last - first) * sy + sy

    half_w = np.maximum(widths / 2.0, 1e-4)
    b = half_w * depth_ratio                                   # (nh,)
    tx = np.clip(np.abs(x[None, :] - centres[:, None]) / half_w[:, None], 0.0, 1.0)
    profile = (1.0 - tx ** n_exp) ** (1.0 / n_exp)             # 1 at centre, 0 at rim
    h = (b[:, None] * profile).astype(np.float32)              # (nh, nw)

    # Flat caps half a voxel inside the padded rows.
    cap_lo = z[0] + 0.5 * sy
    cap_hi = z[-1] - 0.5 * sy
    sz = max(min(sx, sy) * 0.6, 1e-4)
    ny = int(np.clip(math.ceil(2 * b.max() / sz) + 4, 8, 160))
    y = (np.arange(ny, dtype=np.float32) - (ny - 1) / 2.0) * sz

    sdf_y = h[None, :, :] - np.abs(y)[:, None, None]            # (ny, nh, nw)
    sdf_z = np.minimum((z - cap_lo), (cap_hi - z))[None, :, None]  # (1, nh, 1)
    sdf = np.minimum(sdf_y, sdf_z)
    # The extracted surface must never reach the volume border, otherwise
    # marching cubes leaves boundary edges there.
    sdf[:, :, 0] = sdf[:, :, -1] = np.minimum(sdf[:, :, 0], -1.0)
    sdf[:, 0, :] = sdf[:, -1, :] = np.minimum(sdf[:, 0, :], -1.0)
    return sdf.astype(np.float32), (sz, sy, sx)


def _clean_binary_mask(m: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Remove 1-2 voxel protrusions/notches from a resampled silhouette.

    Resampling a 1500 px concept down to a ~100 voxel grid inevitably leaves
    single-voxel spikes.  Each spike would otherwise be extruded into a thin
    full-depth blade by the reconstruction, producing a comb of fins around the
    whole silhouette.  An opening (erode+dilate) deletes protrusions thinner
    than the structuring element; a closing restores small notches afterwards.
    """
    st = np.ones((3, 3), dtype=bool)
    m = ndimage.binary_closing(m, structure=st, iterations=1)
    m = ndimage.binary_opening(m, structure=st, iterations=iterations)
    m = ndimage.binary_closing(m, structure=st, iterations=1)
    # keep the largest component only
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lbl, range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    return m


def _normalised_height(rgb: np.ndarray, mask: np.ndarray,
                       lo_pct: float = 12.0, hi_pct: float = 88.0,
                       smooth_sigma: float = 1.0) -> np.ndarray:
    """Perceptual height map of the concept art, normalised to ``[0, 1]``.

    Bright painted surfaces read as raised, dark mortar lines as recessed -- the
    same convention the concept artist used.  Percentile stretching keeps the
    relief independent of each render's exposure, and a light blur removes
    pixel noise without destroying the block edges.
    """
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lum = ndimage.gaussian_filter(lum, smooth_sigma)
    vals = lum[mask]
    if vals.size < 16:
        return np.zeros_like(lum)
    lo, hi = np.percentile(vals, lo_pct), np.percentile(vals, hi_pct)
    hgt = np.clip((lum - lo) / max(hi - lo, 1e-4), 0.0, 1.0)
    # Soften extreme highlights (specular paint) so the mesh keeps chunky forms.
    hgt = hgt ** 0.9
    return (hgt * mask).astype(np.float32)


def inpaint_background(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the studio white background with edge-padded object colour.

    Without this, the UV margins of a box-projected mesh would sample the pure
    white backdrop and ring every asset with a bright halo.  Nearest-foreground
    padding is the standard fix used when baking texture atlases.
    """
    if mask.all():
        return rgb
    idx = ndimage.distance_transform_edt(~mask, return_distances=False,
                                        return_indices=True)
    filled = rgb[idx[0], idx[1]]
    # Blend towards the object's own average colour with distance from the
    # silhouette: adjacent texels keep real detail, far texels become a flat
    # neutral tone instead of radial smears.
    dist = ndimage.distance_transform_edt(~mask)
    avg = rgb[mask].mean(axis=0) if mask.any() else np.array([0.5, 0.5, 0.5])
    w = np.clip(dist / 24.0, 0.0, 1.0)[..., None]
    soft = filled * (1.0 - w) + avg * w
    soft = ndimage.gaussian_filter(soft, (2, 2, 0))
    out = np.where(mask[..., None], rgb, soft)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Mesh assembly
# --------------------------------------------------------------------------- #
def _mc_to_blender(verts: np.ndarray, spacing: tuple[float, float, float],
                   grid_shape: tuple[int, int, int], plane: str) -> np.ndarray:
    """Convert marching-cubes output (index space x spacing) to Blender metres.

    ``verts`` are ordered along the volume axes ``(z, v, u)``; the grid is
    re-centred on its own midpoint so the resulting mesh is symmetric in the
    image plane and centred on the depth axis.
    """
    sz, sy, sx = spacing
    nz, nh, nw = grid_shape
    x = verts[:, 2] * sx - (nw - 1) * sx / 2.0
    v = verts[:, 1] * sy - (nh - 1) * sy / 2.0
    d = verts[:, 0] * sz - (nz - 1) * sz / 2.0
    if plane == "xz":
        return np.stack([x, d, v], axis=1)      # X, Y(depth), Z(height)
    return np.stack([x, v, d], axis=1)          # X, Y(plan), Z(thickness)


def create_mesh_object(name: str, verts: np.ndarray, faces: np.ndarray,
                       smooth_iters: int = 4):
    """Build a Blender object from raw vertex/face arrays and relax the staircase."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.validate(verbose=False)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)

    utils.select_only([ob])
    # Marching cubes produces axis-aligned terracing; a couple of Laplacian
    # passes give the chunky-but-organic low-poly feel of the concept art.
    mod = ob.modifiers.new("Relax", "SMOOTH")
    mod.factor = 0.5
    mod.iterations = smooth_iters
    utils.apply_modifier(ob, mod.name)
    return ob


def fit_to_dims(ob, dims: tuple[float, float, float]) -> None:
    """Non-uniformly fit the reconstructed volume to the metric module size."""
    co = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
    ob.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mn, mx = co.min(axis=0), co.max(axis=0)
    size = np.maximum(mx - mn, 1e-6)
    target = np.array(dims, dtype=np.float32)
    co = (co - mn) / size * target
    ob.data.vertices.foreach_set("co", co.reshape(-1).astype(np.float32))
    ob.data.update()


def shade_flat(ob) -> None:
    utils.select_only([ob])
    bpy.ops.object.shade_flat()


# --------------------------------------------------------------------------- #
# Raw PBR texture set (albedo / normal / ORM) from the concept art
# --------------------------------------------------------------------------- #
def classical_pbr_split(albedo_rgb: np.ndarray, mask: np.ndarray,
                        *, normal_strength: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Derive a plausible normal + ORM pair from a concept render.

    This is the "PBR dasar" that ships with the *raw* GLB.  The optimised,
    high-quality maps are re-baked from geometry in Phase 3; these only need to
    make the raw asset readable in a viewer.

    * hue analysis separates the three materials in this palette: iron is the
      only *cool* (B >= R) and desaturated surface, stone is warm-neutral and
      wood is strongly saturated. Brightness is useless here because lit stone
      and iron bands overlap almost completely in value.
    * a Sobel gradient of luminance becomes a tangent-space normal map.
    """
    lum = albedo_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mx = albedo_rgb.max(axis=2)
    mn = albedo_rgb.min(axis=2)
    sat = mx - mn
    cool = albedo_rgb[..., 2] - albedo_rgb[..., 0]      # blue minus red

    # Metal: desaturated *and* cooler than the warm-grey stone around it.
    # (A brightness test would classify every highlight on the stone as metal,
    # which is exactly the mistake that turns a stone floor metallic.)
    metal = np.clip(cool / 0.012, 0.0, 1.0) * np.clip(1.0 - sat / 0.06, 0.0, 1.0)
    metal = ndimage.uniform_filter(metal, 3) * mask

    # Roughness: forged iron is smoother than stone, wood is in between.
    stone_rough = np.clip(0.90 - 0.10 * lum, 0.55, 0.95)
    wood = np.clip((sat - 0.10) / 0.15, 0.0, 1.0)
    roughness = np.where(metal > 0.5, 0.45, stone_rough * (1.0 - 0.18 * wood))

    # Height -> normal (Sobel), flipped to tangent space (OpenGL, +Y up).
    hgt = ndimage.gaussian_filter(lum, 1.2)
    gx = ndimage.sobel(hgt, axis=1) * normal_strength
    gy = ndimage.sobel(hgt, axis=0) * normal_strength
    nx, ny, nz = -gx, -gy, np.ones_like(gx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    normal = np.stack([nx * inv, ny * inv, nz * inv], axis=-1) * 0.5 + 0.5

    ao = np.clip(0.55 + 0.45 * ndimage.gaussian_filter(lum, 6), 0, 1)
    orm = np.stack([ao, roughness, metal], axis=-1)
    return normal.astype(np.float32), np.clip(orm, 0, 1).astype(np.float32)


def save_rgba_numpy(arr_rgb: np.ndarray, path: Path, size: int | None = None) -> Path:
    """Write a float RGB array to PNG through Blender's image API."""
    h, w = arr_rgb.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = np.clip(arr_rgb, 0, 1)
    img = bpy.data.images.new(path.stem, width=w, height=h, alpha=False)
    utils.numpy_to_image(img, rgba)
    if size and size != w:
        img.scale(size, size)
    out = utils.save_image(img, path)
    bpy.data.images.remove(img)
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def reconstruct_asset(key: str, *, out_dir: str | Path = RAW_GLB_DIR,
                      texture_size: int = 1024, keep_blend: bool = False) -> dict:
    """Run the full image -> GLB reconstruction for one asset descriptor.

    Steps
    -----
    1. segment the silhouette from the white studio background;
    2. build the metric SDF and extract it with marching cubes;
    3. relax + fit to the modular metric size, box-project UVs;
    4. derive albedo/normal/ORM and pack a PBR material;
    5. export ``raw_glb/<name>.glb``.

    Returns a metadata dict that is written to ``raw_glb/<name>.json``.
    """
    desc = asset_by_key(key)
    ref_path = REFS_RAW_DIR / desc["ref"]
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    LOG.info("=== reconstructing %s from %s", desc["name"], ref_path.name)

    rgb = composite_on_white(ref_path)
    mask = segment_silhouette(rgb)
    frac = mask.mean()
    LOG.info("silhouette: %.1f%% of %dx%d px", frac * 100, mask.shape[1], mask.shape[0])
    if not (0.02 < frac < 0.97):
        raise RuntimeError(f"segmentation failed for {ref_path.name} (coverage {frac:.3f})")

    dims = desc["dims"]
    if desc["plane"] == "xz":
        dims_uv = (dims[0], dims[2])          # image spans X (width) and Z (height)
        depth_m = dims[1]
    else:
        dims_uv = (dims[0], dims[1])          # image spans X and Y (top view)
        depth_m = dims[2]

    sdf, spacing = build_sdf(
        mask, rgb=rgb, dims_uv=dims_uv, depth_m=depth_m,
        plane=desc["plane"], mode=desc["mode"], voxel_res=desc["voxel_res"],
        core=desc["core"], relief=desc["relief"],
        relief_sigma=desc.get("relief_sigma", 1.2))

    # Extracted in index space; _mc_to_blender applies the metric spacing once.
    verts, faces, _, _ = measure.marching_cubes(sdf, level=0.0)
    LOG.info("marching cubes: %d verts / %d faces (grid %s)", len(verts), len(faces), sdf.shape)

    bverts = _mc_to_blender(verts, spacing, sdf.shape, desc["plane"])
    ob = create_mesh_object(desc["name"] + "_Raw", bverts, faces,
                            smooth_iters=desc.get("smooth_iters", 4))
    fit_to_dims(ob, dims)
    utils.ensure_object_mode()
    utils.select_only([ob])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    shade_flat(ob)

    # ---- texture set ----------------------------------------------------- #
    albedo_resized = inpaint_background(_resize_to(rgb, mask.shape), mask)
    normal_map, orm = classical_pbr_split(albedo_resized, mask)
    tex_dir = Path(out_dir)
    albedo_path = save_rgba_numpy(albedo_resized, tex_dir / f"{desc['name']}_raw_albedo.png")
    normal_path = save_rgba_numpy(normal_map, tex_dir / f"{desc['name']}_raw_normal.png")
    orm_path = save_rgba_numpy(orm, tex_dir / f"{desc['name']}_raw_orm.png")

    # UVs span the silhouette bbox only, so the box projection never samples
    # the (edge-padded) studio background.
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    H, W = mask.shape
    pad = 1.5 / max(H, W)
    uv_crop = (float(cols[0] / W) + pad, float((cols[-1] + 1) / W) - pad,
               float(rows[0] / H) + pad, float((rows[-1] + 1) / H) - pad)
    utils.box_project_uv(ob, uv_crop=uv_crop)
    mat = _raw_material(desc, albedo_path, normal_path, orm_path)
    ob.data.materials.append(mat)

    utils.set_origin(ob, desc.get("pivot", "bottom"))
    tris = utils.raw_tri_count(ob.data)

    glb_path = Path(out_dir) / f"{desc['name']}.glb"
    # WebP keeps the raw deliverable a few MB instead of tens of MB; the bake
    # stage reads the lossless PNGs from disk, not the GLB.
    utils.export_glb(ob, glb_path, texture_format="webp", quality=92)

    meta = {
        "key": key,
        "name": desc["name"],
        "backend": "recon-sdf-marchingcubes",
        "source_image": str(ref_path.relative_to(Path(__file__).resolve().parents[1])),
        "glb": str(glb_path.relative_to(Path(__file__).resolve().parents[1])),
        "dims_m": list(dims),
        "raw_tris": tris,
        "raw_verts": len(ob.data.vertices),
        "watertight": utils.count_non_manifold(ob) == 0,
        "textures": {
            "albedo": albedo_path.name,
            "normal": normal_path.name,
            "orm": orm_path.name,
        },
    }
    utils.write_json(Path(out_dir) / f"{desc['name']}.json", meta)
    LOG.info("raw asset written: %s (%d tris, %.1f KB)",
             glb_path.name, tris, glb_path.stat().st_size / 1024.0)

    if not keep_blend:
        utils.delete_objects([ob])
        bpy.data.materials.remove(mat)
    return meta


def _resize_to(rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest/bilinear resize of an RGB array to ``shape`` (H, W)."""
    if rgb.shape[:2] == shape:
        return rgb
    zoom = (shape[0] / rgb.shape[0], shape[1] / rgb.shape[1], 1.0)
    return np.clip(ndimage.zoom(rgb, zoom, order=1), 0, 1)


def _raw_material(desc: dict, albedo_path: Path, normal_path: Path, orm_path: Path):
    """glTF-friendly material for the raw asset (albedo + normal + ORM)."""
    a = bpy.data.images.load(str(albedo_path), check_existing=False)
    n = bpy.data.images.load(str(normal_path), check_existing=False)
    n.colorspace_settings.name = "Non-Color"
    o = bpy.data.images.load(str(orm_path), check_existing=False)
    o.colorspace_settings.name = "Non-Color"
    return utils.build_pbr_material(f"M_{desc['name']}_Raw", a, n, o)


__all__ = [
    "load_image_rgb", "composite_on_white", "segment_silhouette", "build_sdf",
    "create_mesh_object", "fit_to_dims", "classical_pbr_split",
    "save_rgba_numpy", "reconstruct_asset",
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    from config import ASSETS

    ap = argparse.ArgumentParser(description="Image -> GLB reconstruction (Blender)")
    ap.add_argument("--asset", action="append", default=None,
                    help="asset key (repeatable); default = all configured assets")
    ap.add_argument("--out-dir", default=str(RAW_GLB_DIR))
    ap.add_argument("--tex-size", type=int, default=1024,
                    help="resolution of the raw PBR maps (optimised maps are 2048)")
    ns = ap.parse_args(argv)

    keys = ns.asset or [a["key"] for a in ASSETS]
    ok = 0
    for key in keys:
        utils.reset_scene()
        meta = reconstruct_asset(key, out_dir=ns.out_dir, texture_size=ns.tex_size)
        ok += 1
        print(f"[recon] {meta['name']}: {meta['raw_tris']} tris, "
              f"watertight={meta['watertight']}")
    print(f"[recon] done: {ok}/{len(keys)} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
