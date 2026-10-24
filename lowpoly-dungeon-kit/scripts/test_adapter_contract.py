"""
test_adapter_contract.py
========================
Contract test for the neural image-to-3D adapters **without a GPU or weights**.

Why this exists
---------------
SF3D and TripoSR both need (a) a CUDA device and (b) model weights downloaded
from Hugging Face. Neither is available on a CPU-only, network-restricted build
host, so the adapters cannot be exercised against the real models there. This
test substitutes *stub* ``torch`` / ``sf3d`` / ``tsr`` packages that reproduce the
exact public API of the real projects (same module paths, same call signatures)
and feed back a genuine triangle mesh. That verifies everything the adapter is
responsible for:

* engine discovery and preflight reporting,
* the inference call (positional/keyword arguments match the real signature),
* Y-up -> Z-up conversion,
* metric normalisation and pivot placement,
* GLB export with UVs + PBR material preserved,
* metadata contract (``raw_glb/<name>.json``).

With real weights installed, ``scripts/generate.py --backend sf3d`` runs the same
code path unchanged.

Usage
-----
    python scripts/test_adapter_contract.py            # both adapters
    python scripts/test_adapter_contract.py --backend sf3d
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

# bpy needs the stub GL/X11 libraries before it is imported; ensure_host_ready()
# re-execs this process with the right LD_* environment if necessary.
from blender_runner import ensure_host_ready              # noqa: E402

ensure_host_ready()

import utils                                            # noqa: E402
from config import REFS_RAW_DIR, asset_by_key          # noqa: E402

LOG = utils.get_logger("test.contract")

# --------------------------------------------------------------------------- #
# Stub engines — mirror the real module layout and signatures
# --------------------------------------------------------------------------- #
_STUB_TORCH = '''
"""Minimal torch stand-in: enough API for the adapters, no CUDA."""
class _NoGrad:
    def __enter__(self): return None
    def __exit__(self, *a): return False
def no_grad(): return _NoGrad()
class _Autocast(_NoGrad):
    def __init__(self, *a, **k): pass
def autocast(*a, **k): return _Autocast()
bfloat16 = "bfloat16"
class cuda:
    @staticmethod
    def is_available(): return True
'''

_STUB_SF3D_SYSTEM = '''
"""Stub of sf3d.system.SF3D with the real signature from the upstream repo."""
import trimesh, numpy as np
from PIL import Image

class SF3D:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config_name, weight_name):
        assert isinstance(config_name, str) and isinstance(weight_name, str)
        obj = cls()
        obj.pretrained_model_name_or_path = pretrained_model_name_or_path
        return obj

    def to(self, device): self.device = device; return self
    def eval(self): return self

    def run_image(self, image, bake_resolution, remesh="none",
                  vertex_count=-1, estimate_illumination=False):
        """Return a real UV-mapped, PBR-material mesh (Y-up, like SF3D)."""
        assert isinstance(bake_resolution, int) and bake_resolution > 0
        mesh = trimesh.creation.box(extents=(1.0, 1.2, 0.6), subdivisions=2)
        uv = trimesh.creation.box(extents=(1.0, 1.2, 0.6)).vertices[:, :2]
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.vertices[:, :2] * 0.5 + 0.25,
            material=trimesh.visual.material.PBRMaterial(
                name="stub_pbr", baseColorTexture=Image.new("RGB", (64, 64), (140, 130, 120)),
                metallicFactor=0.0, roughnessFactor=0.8))
        return mesh, {"stub": True}
'''

_STUB_SF3D_UTILS = '''
def get_device(): return "cuda"
def remove_background(image, rembg_session=None, force=False, **kw): return image
def resize_foreground(image, ratio, out_size=None): return image
'''

_STUB_TSR_SYSTEM = '''
"""Stub of tsr.system.TSR with the real signature from the upstream repo."""
import trimesh

class _Renderer:
    def set_chunk_size(self, n): self.chunk = n

class TSR:
    def __init__(self): self.renderer = _Renderer()
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config_name, weight_name):
        return cls()
    def to(self, device): self.device = device; return self
    def __call__(self, images, device=None): return ["stub-scene-code"]
    def extract_mesh(self, scene_codes, has_vertex_color, resolution=256, threshold=25.0):
        m = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        return [m]
'''

_STUB_TSR_BAKE = '''
def bake_texture(mesh, model, scene_code, texture_resolution):
    v = mesh.vertices
    return {"vmapping": list(range(len(v))), "indices": mesh.faces,
            "uvs": v[:, :2] * 0.5 + 0.25,
            "colors": __import__("numpy").tile([0.55, 0.5, 0.45], (64, 64, 1))}
'''

_STUB_XATLAS = "def export(*a, **k):\n    return None\n"


def build_stub_env(root: Path) -> dict:
    """Materialise the stub packages and return the environment to use."""
    def write(path: Path, text: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    write(root / "torch" / "__init__.py", _STUB_TORCH)
    write(root / "sf3d" / "__init__.py", "")
    write(root / "sf3d" / "system.py", _STUB_SF3D_SYSTEM)
    write(root / "sf3d" / "utils.py", _STUB_SF3D_UTILS)
    write(root / "tsr" / "__init__.py", "")
    write(root / "tsr" / "system.py", _STUB_TSR_SYSTEM)
    write(root / "tsr" / "bake_texture.py", _STUB_TSR_BAKE)
    write(root / "xatlas.py", _STUB_XATLAS)

    # Fake HF cache so the backend's weight probe succeeds.
    for model, pattern in (("stabilityai/stable-fast-3d", "model.safetensors"),
                           ("stabilityai/TripoSR", "model.ckpt")):
        d = root / "hf" / "hub" / f"models--{model.replace('/', '--')}" / "snapshots" / "stub"
        write(d / pattern, b"stub".decode())

    env = {
        "SF3D_REPO": str(root),
        "TRIPOSR_REPO": str(root),
        "HF_HOME": str(root / "hf"),
    }
    return env


# --------------------------------------------------------------------------- #
# Assertions
# --------------------------------------------------------------------------- #
def assert_glb(path: Path, dims: tuple[float, float, float]) -> dict:
    import trimesh

    assert path.exists(), f"missing GLB {path}"
    scene = trimesh.load(str(path), force="scene")
    geoms = list(scene.geometry.values())
    assert geoms, "GLB contains no geometry"
    mesh = geoms[0]
    assert len(mesh.faces) > 0, "mesh has no faces"
    assert mesh.visual is not None, "mesh has no visual (material/UV)"
    uv = getattr(mesh.visual, "uv", None)
    assert uv is not None and len(uv) == len(mesh.vertices), "missing per-vertex UVs"

    extents = mesh.extents
    target = [dims[0], dims[2], dims[1]]           # Blender XYZ -> glTF XYZ
    # The mesh is scaled uniformly (proportions preserved) to fit *inside* the
    # module box: no axis may exceed its target, and the most constrained axis
    # must touch it exactly.
    for i in range(3):
        assert extents[i] <= target[i] * 1.01, (
            f"axis {i} overflows the module box: {extents[i]:.3f} > {target[i]:.3f} m")
    tight = min(range(3), key=lambda i: target[i] / max(extents[i], 1e-6))
    assert abs(extents[tight] - target[tight]) < 0.02 * target[tight], (
        f"tightest axis {tight} is {extents[tight]:.3f} m, expected {target[tight]:.3f} m")
    assert abs(mesh.bounds[0][1]) < 1e-3, "mesh is not sitting on the ground plane"

    return {"faces": len(mesh.faces), "extents": [round(float(e), 3) for e in extents],
            "watertight": bool(mesh.is_watertight),
            "has_uv": True, "has_material": mesh.visual.material is not None}


def run_backend(name: str, env: dict, out_dir: Path) -> dict:
    import os

    os.environ.update(env)
    for mod in ("torch", "sf3d", "sf3d.system", "sf3d.utils", "tsr", "tsr.system",
                "tsr.bake_texture", "xatlas"):
        sys.modules.pop(mod, None)
    sys.path.insert(0, env["SF3D_REPO"])

    if name == "sf3d":
        from generators import sf3d_backend as backend
        desc = asset_by_key("wall")
    else:
        from generators import triposr_backend as backend
        desc = asset_by_key("chest")

    report = backend.check_environment()
    assert report["available"], f"{name} preflight failed: {report['problems']}"

    meta = backend.generate(desc, out_dir=out_dir)
    stats = assert_glb(Path(meta["glb"]), desc["dims"])
    assert meta["name"] == desc["name"]
    assert Path(meta["glb"]).with_suffix(".json").exists(), "metadata JSON missing"

    # the produced file must be re-importable by the Blender pipeline
    import bpy
    import utils as u
    u.reset_scene()
    bpy.ops.import_scene.gltf(filepath=str(Path(meta["glb"]).resolve()))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    assert meshes, "Blender could not import the GLB"
    assert meshes[0].data.uv_layers, "imported mesh lost its UV map"

    LOG.info("%s OK: %s", name, stats)
    return stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Adapter contract test (stub engines)")
    ap.add_argument("--backend", action="append", choices=["sf3d", "triposr"])
    ns = ap.parse_args(argv)
    names = ns.backend or ["sf3d", "triposr"]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="engine_stubs_") as tmp:
        root = Path(tmp)
        env = build_stub_env(root)
        out_dir = root / "raw_glb"
        for name in names:
            try:
                run_backend(name, env, out_dir)
                print(f"[PASS] {name} adapter contract")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
            except Exception as exc:                    # noqa: BLE001
                failures += 1
                print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
