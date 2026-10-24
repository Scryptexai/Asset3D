# Store listing copy & technical spec sheet

Ready-to-paste descriptions, tags and spec tables for **itch.io** and the
**Unity Asset Store**, plus the numbers a reviewer will ask for.

---

## 1. Short description (≤ 200 chars)

> Hand-crafted low-poly modular dungeon kit: stone walls, floors, pillars, doors
> and a treasure chest. 4 LODs, PBR textured, convex collision, Unity & Godot ready.

## 2. itch.io description

```markdown
# Low-Poly Modular Dungeon Kit

Build dark-fantasy dungeons in minutes. Every piece is a clean, game-ready
low-poly mesh with 4 LOD levels, 2048² PBR textures and a matching convex
collision mesh — drop them on a 1 m grid and kit-bash.

## What's inside
- 5 modular assets — wall, floor, pillar, door, treasure chest
- 4 LODs per asset (100 % / 50 % / 20 % / 5 %)
- 2048² PBR set per asset: Albedo, Normal (OpenGL), ORM (AO/Roughness/Metallic)
- Convex collision mesh per asset (`SM_*_Collision`)
- **Unity**: 20 FBX + 15 PNG (external textures, MikkTSpace tangents)
- **Godot 4**: 20 GLB with embedded WebP textures (no extra files to wire up)
- Metric, grid-snapped modules: walls/floors 4 m, doors 2 × 3 m, pillars 1 × 3 m
- QA report and manifest included — every file validated by re-import

## Perfect for
First-person dungeon crawlers, roguelikes, VR dungeon builders, tabletop
simulators, low-poly RPGs, game-jam prototypes.

## Technical notes
- LOD0 triangle budget: 1 182 – 1 476 tris per asset
- UVs are unique (non-overlapping) — lightmapping and re-baking work out of the box
- Flat-shaded, chunky silhouette keeps the stylised look readable at distance
- No animations, no rig — this is a static architecture/prop kit

Licence: commercial, single seat. See LICENSE.txt.
```

## 3. Unity Asset Store fields

| Field | Value |
|---|---|
| Title | Low-Poly Modular Dungeon Kit |
| Category | 3D/Environments/Fantasy |
| Version | 1.0.0 |
| Unity version | 2021.3 LTS or newer (Built-in, URP and HDRP compatible) |
| Render pipeline | Built-in / URP / HDRP (standard metallic-roughness inputs) |
| Texture format | PNG, 2048 × 2048, sRGB for Albedo, linear for Normal/ORM |
| Collision | Convex meshes provided, no runtime generation |
| Files | 20 FBX + 15 PNG + documentation |
| AI disclosure | Concept art is AI-generated; base meshes come from an automated image-to-3D stage (Stable Fast 3D when a CUDA device is available, otherwise the kit's offline reconstruction backend). All delivered game assets were retopologised, re-textured, LOD-reduced and validated by supervised automated tooling. See LICENSE.txt §6. |

## 4. Tags

`low poly`, `modular`, `dungeon`, `dark fantasy`, `environment`, `props`,
`stone`, `medieval`, `LOD`, `PBR`, `game ready`, `kitbash`, `3d`, `unity`,
`godot`, `indoor`, `architecture`

## 5. Spec sheet (measured from the shipped files)

Run `python scripts/blender_runner.py --run validate.py` to regenerate; the
authoritative numbers live in `exports/manifest.json` and `renders/qa_report.json`.

| Asset | LOD0 | LOD1 | LOD2 | LOD3 | Collision | Size (W×D×H) | Pivot |
|---|---|---|---|---|---|---|---|
| `SM_Dungeon_Wall_A` | 1 378 | 678 | 270 | 66 | 62 | 4.00 × 0.45 × 3.00 m | bottom-centre |
| `SM_Dungeon_Floor_A` | 1 182 | 582 | 232 | 58 | 62 | 4.00 × 4.00 × 0.35 m | top-centre |
| `SM_Dungeon_Pillar_A` | 1 476 | 726 | 290 | 72 | 64 | 1.00 × 1.00 × 3.00 m | bottom-centre |
| `SM_Dungeon_Door_A` | 1 378 | 678 | 270 | 66 | 62 | 2.00 × 0.50 × 3.00 m | bottom-centre |
| `SM_Dungeon_Chest_A` | 1 280 | 630 | 252 | 62 | 64 | 1.00 × 0.70 × 0.80 m | bottom-centre |

Totals: **LOD0 6 694 tris** for the whole kit, 20 meshes + 5 collision meshes,
15 texture maps (≈ 43 MB of lossless PNG).

## 6. Reviewer checklist

- [x] All meshes closed / manifold — 0 non-manifold edges in every LOD and collision mesh
- [x] Normals outward (signed volume positive for all 25 exported meshes)
- [x] UVs present, inside 0-1, non-overlapping
- [x] Texture resolution uniform (2048²), channels packed per glTF spec
- [x] LOD chain monotonic, FBX and GLB agree on triangle counts
- [x] Metrics are real-world metres, pivots consistent per category
- [x] Convex collision mesh per asset, ≤ 64 tris
- [x] Licence file and documentation included

## 7. Screenshot shot-list (for the store page)

1. `renders/showcase_room.png` — kit-bashed 8 × 8 m dungeon room (hero shot)
2. `renders/wall_lod0_preview.png` — wall turntable
3. `renders/door_lod0_preview.png` — door turntable
4. `renders/pillar_lod0_preview.png` — pillar turntable
5. `renders/chest_lod0_preview.png` — chest turntable
6. `renders/floor_lod0_preview.png` — floor turntable

All previews are rendered headlessly from the **exported** GLBs with Cycles, so
they show exactly what ships.
