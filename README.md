# Asset3D — Low-Poly Modular Dungeon Kit

This repository contains a complete, reproducible asset-production pipeline and
the commercial deliverables it produces.

```
lowpoly-dungeon-kit/
├── refs/          # concept art (2D)
├── raw_glb/       # image-to-3D output (high-poly GLB + PBR)
├── scripts/       # Blender-headless pipeline (documented, modular)
├── exports/       # Models/ (Unity FBX) · Textures/ (2048² PBR) · Godot/ (GLB)
├── renders/       # QA + store-page previews
└── Documentation/ # README.md (setup & import) · LICENSE.txt (commercial)
```

**Start here → [`lowpoly-dungeon-kit/Documentation/README.md`](lowpoly-dungeon-kit/Documentation/README.md)**

Quick rebuild:

```bash
cd lowpoly-dungeon-kit
pip install -r scripts/requirements.txt          # numpy, scipy, scikit-image, pillow
export BLENDER_BIN=/path/to/blender              # or: pip install bpy

python scripts/generate.py --check-backends      # which image-to-3D engine can run?
python scripts/build_all.py                      # generate → optimise → QA → previews
```

Image-to-3D engine order: **Stable Fast 3D** (primary, native UV + PBR) →
**TripoSR** (fallback) → `recon` (offline, no GPU/weights). Both neural engines
need a CUDA device and Hugging Face weights; see
[`Documentation/README.md`](lowpoly-dungeon-kit/Documentation/README.md#4-image-to-3d-engines-sf3d-primary--triposr-fallback).

| Stage | Script | What it does |
|---|---|---|
| 1 | `scripts/generate.py` | concept art → high-poly PBR GLB (`raw_glb/`); engines: **SF3D** (primary), **TripoSR** (fallback), `recon` (offline) |
| 2 | `scripts/pipeline.py` | import → clean → decimate → UV unwrap → PBR bake → LOD0-3 → convex collision → FBX + GLB export |
| 3 | `scripts/validate.py` | QA gate over the exported files (manifoldness, LOD budgets, texture packing, manifest) |
| 4 | `scripts/preview.py`, `scripts/showcase.py` | headless Cycles turntables and a kit-bashed room render |
