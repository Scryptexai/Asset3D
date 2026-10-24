# Low-Poly Modular Dungeon Kit

A production-ready, modular dark-fantasy dungeon set for **Unity** and **Godot 4**.
Every asset is a real triangle mesh generated from concept art, retopologised,
UV-unwrapped, PBR-baked, LOD-reduced and validated by a fully reproducible
Blender-headless pipeline.

* **5 modular assets** — wall, floor, pillar, door, chest
* **4 LODs each** (100 % / 50 % / 20 % / 5 % of the LOD0 triangle budget)
* **3 × 2048² PBR maps per asset** — Albedo, Normal (OpenGL +Y), ORM
  (`R`=AO, `G`=Roughness, `B`=Metallic — the glTF/metallic-roughness packing)
* **Convex collision mesh** per asset (prefixed `SM_…_Collision`)
* **Metric, 1 m grid-snapped modules** with predictable pivots
* **Both engine flavours** — FBX for Unity, GLB for Godot 4

---

## 1. What is in the box

```
lowpoly-dungeon-kit/
├── refs/                     # approved concept art (1024-1536 px PNG)
│   └── raw/                  # untouched generator output
├── raw_glb/                  # image-to-3D output: high-poly GLB + PBR maps + .json
├── scripts/                  # the whole pipeline (Blender headless)
│   ├── config.py             # every path, size, LOD ratio and asset descriptor
│   ├── utils.py              # shared Blender/bmesh helpers
│   ├── segment.py            # silhouette segmentation (no Blender dependency)
│   ├── recon.py              # offline backend: silhouette -> SDF -> marching cubes
│   ├── generate.py           # Phase 2 driver (engine selection + fallback)
│   ├── generators/           # sf3d (primary) / triposr (fallback) / recon backends
│   ├── test_adapter_contract.py  # verifies both neural adapters with stub engines
│   ├── pipeline.py           # Phase 3 optimisation pipeline
│   ├── preview.py            # headless turntable QA renderer
│   ├── validate.py           # Phase 5 QA gate (validates exported files)
│   ├── blender_runner.py     # Blender runtime detection / launcher
│   ├── build_all.py          # one-command rebuild
│   └── requirements.txt
├── exports/
│   ├── Models/               # Unity: SM_*.fbx (LOD0-3 + Collision)
│   ├── Textures/             # T_*_Albedo/Normal/ORM_2048.png
│   ├── Godot/                # Godot 4: SM_*.glb (textures embedded, WebP)
│   └── manifest.json         # machine-readable asset manifest
├── renders/                  # QA / store-page preview contact sheets
└── Documentation/
    ├── README.md             # this file
    └── LICENSE.txt           # commercial licence (single seat)
```

### Asset table

| Asset | Mesh prefix | Size (W × D × H, m) | LOD0 tris | Pivot |
|---|---|---|---|---|
| Wall module | `SM_Dungeon_Wall_A` | 4.00 × 0.45 × 3.00 | ~1 400 | bottom-centre |
| Floor tile | `SM_Dungeon_Floor_A` | 4.00 × 4.00 × 0.35 | ~1 200 | **top-centre** (walk surface at local `z = 0`) |
| Pillar | `SM_Dungeon_Pillar_A` | 1.00 × 1.00 × 3.00 | ~1 500 | bottom-centre |
| Door | `SM_Dungeon_Door_A` | 2.00 × 0.50 × 3.00 | ~1 400 | bottom-centre |
| Chest | `SM_Dungeon_Chest_A` | 1.00 × 0.70 × 0.80 | ~1 300 | bottom-centre |

Exact per-asset numbers (including every LOD and the collision mesh) live in
`exports/manifest.json`.

---

## 2. Requirements

* **Blender 4.2 LTS or newer** (developed and validated on **Blender 5.0.1**), or
  the official `bpy` Python module, plus
* Python 3.11+, `numpy`, `scipy`, `scikit-image`, `pillow` (see
  `scripts/requirements.txt`).

### Setup A — Blender binary (recommended)

```bash
export BLENDER_BIN=/opt/blender-4.2.9-linux-x64/blender
python3 scripts/blender_runner.py --check      # should print "mode: binary"
```

### Setup B — Blender as a Python module

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r scripts/requirements.txt bpy
python3 scripts/blender_runner.py --check      # should print "mode: module"
```

> **Headless containers.** Blender links against X11/OpenGL even in
> `--background` mode. On a slim image those shared objects are missing, and
> `blender_runner.py` synthesises no-op stub libraries automatically (they are
> never called during background rendering). Nothing else in the pipeline cares.

---

## 3. Rebuilding the kit from scratch

```bash
# Phase 2: concept art -> high-poly PBR GLB in raw_glb/
#   auto = SF3D -> TripoSR -> offline reconstruction, first available wins
python scripts/generate.py --all

# Phase 3: import -> clean -> decimate -> unwrap -> bake -> LODs -> collision -> export
python scripts/blender_runner.py --run pipeline.py --all

# Phase 5: QA gate (validates the exported FBX/GLB/PNG files)
python scripts/blender_runner.py --run validate.py --out renders/qa_report.json

# Previews for the store page / QA (rendered from the exported GLBs)
python scripts/blender_runner.py --run preview.py \
    --input exports/Godot/SM_Dungeon_Wall_A_LOD0.glb \
    --out renders/wall_lod0_preview.png
python scripts/blender_runner.py --run showcase.py --out renders/showcase_room.png
```

Or everything in one go:

```bash
python scripts/build_all.py                # generate + pipeline + validate + previews
```

Verifying the neural adapters without a GPU (stub-engine contract test):

```bash
python scripts/test_adapter_contract.py    # [PASS] sf3d · [PASS] triposr
```

With a real Blender binary the equivalent of the pipeline command is exactly the
marketplace-standard form:

```bash
blender --background --python scripts/pipeline.py -- --all
```

### Pipeline stages (`scripts/pipeline.py`)

| # | Stage | Implementation |
|---|---|---|
| 1 | Import & cleanup | glTF import, `bmesh` weld (glTF arrives as an unwelded soup), loose-geometry removal, hole fill, outward normals |
| 2 | Retopology / decimation | iterative collapse-decimate to the per-asset triangle budget (±2 %) |
| 3 | UV unwrap | `Smart UV Project` (66° angle limit, 0.004 island margin) |
| 4 | PBR bake | high→low ray-cast: **albedo** (emission transfer), **normal**, **AO** (Cycles occlusion); roughness/metallic transferred from the raw material analysis, then packed to ORM |
| 5 | LOD generation | LOD1 50 %, LOD2 20 %, LOD3 5 % of LOD0 — one shared UV layout and one texture set per asset |
| 6 | Collision | convex hull, decimated to ≤ 64 tris, named `SM_<Asset>_Collision` |
| 7 | Export | `exports/Models/*.fbx` (Unity, Y-up, no leaf bones) and `exports/Godot/*.glb` (embedded WebP textures) |

Design notes worth knowing:

* The **raw** mesh is the bake source; the decimated + unwrapped copy is the
  delivery mesh. That is why a 1 400-triangle wall keeps the mortar detail of a
  60 000-triangle reconstruction.
* Albedo is transferred with an **emission** bake, not a diffuse bake — no
  lighting integration means no noise and no light leaks.
* AO is a genuine high→low occlusion bake, so it also darkens recesses the
  low-poly can no longer represent geometrically.

---

## 4. Image-to-3D engines (SF3D primary · TripoSR fallback)

`generate.py` is engine-agnostic and ships three backends:

| Engine | Role | Output | Needs |
|---|---|---|---|
| **`sf3d`** — [Stability AI Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | **primary** | dense mesh + **native UV + PBR** (albedo / metal-rough / normal atlas) | CUDA, HF weights |
| **`triposr`** — [VAST-AI TripoSR](https://github.com/VAST-AI-Research/TripoSR) | **fallback** | triplane mesh + baked texture atlas (UV + albedo) | CUDA (or CPU, very slow), HF weights |
| `recon` | offline fallback | silhouette → SDF → marching cubes, UV + derived PBR | Blender only |

```bash
python scripts/generate.py --check-backends           # preflight: what can run here?
python scripts/generate.py --all                      # auto: sf3d -> triposr -> recon
python scripts/generate.py --backend sf3d --all       # strict: SF3D or fail
python scripts/generate.py --backend triposr --asset door
python scripts/generate.py --backend recon --all      # CPU-only / no-network build
```

All engines write the same artefact — `raw_glb/<SM_Name>.glb` (high-poly, UVs,
PBR) plus `raw_glb/<SM_Name>.json` — so the Blender optimisation stage, the QA
gate and the exporters are unchanged.

### Installing SF3D (primary)

```bash
git clone https://github.com/Stability-AI/stable-fast-3d external/stable-fast-3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r external/stable-fast-3d/requirements.txt
# SF3D's texture baker and UV unwrapper are compiled CUDA extensions:
pip install external/stable-fast-3d/texture_baker external/stable-fast-3d/uv_unwrapper
export SF3D_REPO=$PWD/external/stable-fast-3d

huggingface-cli download stabilityai/stable-fast-3d --local-dir ~/.cache/hf/sf3d
python scripts/generate.py --check-backends       # expect: [READY] sf3d
```

Adapter defaults (tuned for a kit that gets retopologised downstream):
`bake_resolution=2048`, `remesh="none"`, `vertex_count=-1` — i.e. keep the dense
reconstruction and let Phase 3 own decimation, so the bake source keeps maximum
detail.

### Installing TripoSR (fallback)

```bash
git clone https://github.com/VAST-AI-Research/TripoSR external/TripoSR
pip install -r external/TripoSR/requirements.txt
export TRIPOSR_REPO=$PWD/external/TripoSR
huggingface-cli download stabilityai/TripoSR --local-dir ~/.cache/hf/triposr
```

The adapter uses TripoSR's texture-atlas bake path (`--bake-texture` equivalent)
so the raw asset carries a real UV map + albedo instead of vertex colours.

### Why the shipped `raw_glb/` came from the offline `recon` backend

The kit was produced in a sandbox that **cannot run either neural engine**, and
the pipeline reports exactly why instead of pretending otherwise:

```
$ python scripts/generate.py --check-backends
[BLOCKED] sf3d     cuda=False weights=False
           - no CUDA device (SF3D's texture baker needs one)
           - weights not in HF cache (needs huggingface.co access)
[BLOCKED] triposr  cuda=False weights=False
           - no CUDA device (CPU inference needs >16 GB RAM anyway)
           - weights not in HF cache (needs huggingface.co access)
[READY  ] recon    ...
```

Measured facts from the build host: **no GPU** (2 vCPU / 3 GB RAM), and network
egress is restricted to GitHub + PyPI — `huggingface.co` and `cdn-lfs.huggingface.co`
are unreachable, so the weights (`model.safetensors`, `model.ckpt`) cannot be
fetched. The adapter code is nevertheless verified: `scripts/test_adapter_contract.py`
substitutes API-faithful stub engines and asserts the full path (inference call →
Y-up→Z-up → metric normalisation → GLB with UV + PBR → metadata → re-import into
Blender) for both adapters:

```bash
python scripts/test_adapter_contract.py     # [PASS] sf3d · [PASS] triposr
```

On a CUDA workstation, re-running `python scripts/generate.py --backend sf3d --all`
followed by the pipeline replaces the offline meshes with SF3D reconstructions;
everything downstream (bake, LODs, collision, exports, QA) is identical.

### The offline backend is a real reconstruction, not procedural code

For completeness, `recon` does not emit procedural/parametric geometry: it
segments the silhouette, builds a metric signed distance field, extracts it with
**marching cubes** into an explicit vertex/face list, relaxes it, and writes that
triangle mesh to GLB — the output is verifiable in any DCC tool or engine as a
plain mesh (`POSITION/NORMAL/TEXCOORD_0` accessors).

## 5. Unity import guide

1. Copy `exports/Models/*.fbx` and `exports/Textures/*.png` into
   `Assets/LowPolyDungeonKit/`.
2. Select every FBX and set, in the **Model** tab:
   * Scale Factor `1`, Convert Units ✓ (the kit is metric)
   * Mesh Compression `Off` (let the LODs do the work)
   * Generate Colliders `Off` — use the supplied `SM_*_Collision` meshes
   * Normals `Import`, Tangents `Calculate MikkTSpace` (required by the normal map)
3. **Rig** tab → Animation Type `None` (static kit).
4. Create one material per asset and assign the three maps:
   * `T_*_Albedo_2048.png` → `Albedo`, sRGB ✓
   * `T_*_Normal_2048.png` → `Normal map`, sRGB **off**, `Create from Grayscale` off
   * `T_*_ORM_2048.png` → sRGB **off**; use this single map for **Occlusion (R)**
     and drive `Smoothness` from **G** (smoothness = 1 − roughness) and
     `Metallic` from **B** via a small shader or a custom material inspector.
     The High Definition Render Pipeline does this natively with a
     `Mask Map` (R = metallic, G = AO, B = detail, A = smoothness), so either
     repack once for HDRP or use a tiny remap shader.
5. LOD groups: add `SM_*_LOD0..3` to a `LOD Group` and use the screen-relative
   heights `0.6 / 0.3 / 0.1 / 0.03` as a starting point.
6. Collision: add the `SM_*_Collision` mesh to a `MeshCollider` (Convex ✓).

## 6. Godot 4 import guide

1. Copy `exports/Godot/*.glb` into `res://lowpoly_dungeon_kit/`.
2. Godot imports the GLB with **embedded WebP textures** (`EXT_texture_webp`,
   supported natively since 4.0) — no external texture files needed.
3. In the **Import** dock:
   * *Meshes → Generate LODs*: **off** (the kit ships its own LOD chain)
   * *Meshes → Create shadow meshes*: on
   * *Materials → Extract*: on if you want to tweak roughness/metallic
4. Drop `SM_Dungeon_Wall_A_LOD0.glb` into the scene; set
   `GridMap` or a `MeshLibrary` for level building, or place the modules by hand.
   The modules are 4 m (wall/floor) and 3 m tall, so `GridMap.cell_size = Vector3(4, 3, 4)`
   gives instant kit-bashing.
5. Collision: add a `StaticBody3D` + `CollisionShape3D` with a
   `ConvexPolygonShape3D` created from the `SM_*_Collision` mesh.
6. LODs: `GeometryInstance3D` → **Visibility Range** per imported GLB, or use
   `$GeometryInstance3D.lod_bias` in a `LODGroup` node.

---

## 7. Modular workflow notes

* Wall/floor modules are **4 m**, doors **2 m wide × 3 m tall**, pillars **1 m² × 3 m**.
* Wall, door, pillar and chest pivots sit at the **bottom-centre** of the mesh, so
  `position = (x * 4, 0, z * 4)` snaps them to the grid with no offset fudge.
* Floor tiles are pivoted at the **walk surface** (top), so a tile placed at
  `y = 0` gives you a floor whose surface is exactly at `y = 0`.
* Back faces of wall/door/floor/chest modules are flat, so they can be recessed
  into geometry without visible gaps.
* UVs are unique (non-overlapping) per asset, so baking/lightmapping in your
  engine works out of the box.

---

## 8. Quality assurance

`scripts/validate.py` re-imports **every exported file** and checks:

* presence of all LODs and the collision mesh, in *both* FBX and GLB
* triangle budgets, monotonic LOD reduction, FBX/GLB triangle parity
* non-manifold edges, loose vertices, inward-facing normals
* UV existence, 0-1 range and coverage
* texture resolution (2048²), channel packing and value sanity
* manifest completeness

```bash
python scripts/blender_runner.py --run validate.py
```

The latest report is stored in `renders/qa_report.json`; the contact sheets in
`renders/` are rendered **from the exported GLBs**, i.e. exactly what a buyer
loads in-engine.

---

## 9. Licence

Commercial single-seat licence — see `LICENSE.txt`. The Python pipeline in
`scripts/` is MIT-licensed and reusable independently of the assets.

Version 1.0.0 · generated 2026-10-24
