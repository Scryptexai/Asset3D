"""
generators
==========
Pluggable image-to-3D engines for Phase 2 of the pipeline.

Contract
--------
Every backend exposes::

    generate(desc, ref_image=None, out_dir=RAW_GLB_DIR) -> dict
    check_environment() -> dict        # availability report, no side effects

and writes a **high-poly GLB with UVs and a PBR material** to
``raw_glb/<desc['name']>.glb`` plus a sibling ``.json``. Nothing downstream
(Blender optimisation, LODs, QA, exports) knows which engine produced the file.

Engines
-------
``sf3d``      **Primary.** Stability AI Stable Fast 3D — natively produces UV +
              PBR (albedo, metallic/roughness, normal). Real triangle mesh.
``triposr``   **Fallback.** VAST-AI TripoSR — fast triplane reconstruction; the
              adapter uses its texture-atlas bake path for UV + albedo.
``recon``     **Offline fallback.** Silhouette -> SDF -> marching cubes, run
              inside Blender. Requires no GPU and no model weights, which is
              what allowed the shipped kit to be produced on this host.

Selection
---------
    --backend auto      sf3d -> triposr -> recon, first available wins (default)
    --backend sf3d      strict: fail if SF3D cannot run
    --backend triposr   strict
    --backend recon     strict

See ``Documentation/README.md`` section 4 for install instructions.
"""

from __future__ import annotations

from importlib import import_module

BACKENDS = {
    "sf3d": "generators.sf3d_backend",
    "triposr": "generators.triposr_backend",
    "recon": "generators.recon_backend",
}

#: Order used by ``--backend auto``: best quality first, guaranteed-offline last.
FALLBACK_CHAIN = ["sf3d", "triposr", "recon"]

DEFAULT_BACKEND = "sf3d"


def load_backend(name: str):
    """Import and return a backend module."""
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    module = import_module(BACKENDS[name])
    if not hasattr(module, "generate"):
        raise AttributeError(f"{BACKENDS[name]} does not implement generate()")
    return module


def backend_chain(requested: str = "auto") -> list[str]:
    """Resolve a requested backend into the ordered list to attempt."""
    if requested in ("auto", "fallback"):
        return list(FALLBACK_CHAIN)
    if requested == "all":
        return list(FALLBACK_CHAIN)
    load_backend(requested)          # validate early
    return [requested]


def check_all() -> list[dict]:
    """Availability report for every backend (used by ``--check-backends``)."""
    reports = []
    for name in BACKENDS:
        try:
            reports.append(load_backend(name).check_environment())
        except Exception as exc:                        # noqa: BLE001
            reports.append({"backend": name, "available": False,
                            "problems": [f"{type(exc).__name__}: {exc}"]})
    return reports


def pick_backend(requested: str = "auto", logger=None) -> str:
    """Return the first backend in the chain that reports itself available."""
    chain = backend_chain(requested)
    for name in chain:
        if name == "recon":
            # The offline reconstruction only needs Blender, which the caller
            # already guarantees by importing this module inside it.
            return name
        try:
            report = load_backend(name).check_environment()
        except Exception as exc:                        # noqa: BLE001
            if logger:
                logger.warning("%s preflight failed: %s", name, exc)
            continue
        if report.get("available"):
            return name
        if logger:
            logger.warning("%s unavailable: %s", name,
                           "; ".join(report.get("problems", [])))
    return "recon"


__all__ = ["BACKENDS", "FALLBACK_CHAIN", "DEFAULT_BACKEND", "load_backend",
           "backend_chain", "check_all", "pick_backend"]
