# SimReady friction audit (decision 10 input)

Reference material for the team, 5 Nov 2026. Not report text. Read-only study: no pipeline code changed. Reading script: `simready_audit.py`; figure: `simready_friction.png`.

## Summary

- NVIDIA's SimReady Central is a search front end (Hugging Face Space `nvidia/simready-central`) over OEM datasets. The "2,000+ free OpenUSD assets" library is the dataset `nvidia/simready-assets`: 2,677 asset folders, CC-BY-4.0, ungated, about 490 GB with meshes and textures.
- Friction is stored per **physics material**, not per object–surface pair. Every asset has a small text layer `simready_usd/payloads/Physics/physics.usda` that applies `PhysicsMaterialAPI` to each prim under `/RootNode/PhysicsMaterials` and authors `physics:staticFriction`, `physics:dynamicFriction`, `physics:restitution` and `physics:density`. Each mesh binds one of these materials (`material:binding:physics` in `base.usda`). Two touching bodies get a pair coefficient from the engine's own combination rule.
- Read on 5 Nov 2026 with `usd-core` 26.11: 2,669 of the 2,677 assets have a physics layer (the other 8 have no payloads at all); 2,668 of them carry at least one physics material; **9,597 physics material prims, 3,014 distinct material names**.
- The library does **not** carry one default value, but it does not carry measured values either. Each material class carries one fixed (static, dynamic) pair: plastic (0.4, 0.4), steel (0.3, 0.3), painted metal (0.5, 0.4), aluminium (0.61, 0.47), cardboard (0.7, 0.7), paper (0.5, 0.45), rubber (0.8, 0.8), glass (0.4, 0.4), wood (0.5, 0.5), fabric (0.5, 0.5), ceramic (0.35, 0.3), and so on. 31 of the 33 classes with ten or more prims put one pair on at least 95% of their prims. Dynamic 0.4 alone is 44% of all prims; the eight most common pairs cover 92%.
- All nine rig object bases and all three surfaces have a plausible class in the library, four of the objects by name (oak, glossy ceramic, stainless steel, acrylic). None has a measured or pair-specific value; melamine and MDF map only to "laminate wood" (four prims) and generic wood.
- **Conclusion (spec form 2):** the values vary and cover the rig's materials, so a C2-SimReady baseline (per-material asset friction, combined by each simulator's rule) is possible. The caveats below say what such a baseline would and would not test.
- Time: 26 minutes from the first Hub query to the final tables (download 279 s, USD parse 10 s), plus writing this note. 2,669 assets read.

## 1. What SimReady Central is and what was read

**The library.** `nvidia/simready-assets` (Hugging Face dataset, last modified 4 Nov 2026, 1.3 M downloads, licence CC-BY-4.0, no gating). Its card points to the Space `nvidia/simready-central` ("Semantic search for 3D assets: USD + RTX simulation previews", a docker app over OpenSearch, Neo4j and SigLIP2 embeddings) and to `nvidia/simready-central-library`, which is gated for contributor onboarding and holds only a README. A community catalogue (`thisisrick25/nvidia-simready-catalog`, computed 4 Nov 2026) lists 2,677 assets, 489.68 GB, 183,398 files, 14 taxonomy groups (2,314 "common_assets"), 2,669 with PhysX, MuJoCo and Newton variants, 8 with no payloads, 5 articulated.

**Partner datasets** indexed by the Space, not read here (the brief's 2,000+ figure is the NVIDIA set): ForgeCAD 121 asset folders, NdotLight 50, Vertiv 41, VAST-AI Tripo 19, frankarobotics 9, deeproboticslab 8, standardbots 1, nvidia/simready-dsx 1, Sandflow (different layout). All CC-BY-4.0. ForgeCAD, frankarobotics, deeproboticslab, standardbots and Tripo show the same `physics.usda` layout in their file lists; Vertiv and NdotLight do not.

**Asset layout** (one folder per asset):

```
<asset>/com.nvidia.simready.packaging.json          package id, licence, hashes
<asset>/.metadata/                                   bill of materials, conformance reports
<asset>/simready_usd/sm_<asset>_01.usd               main stage (binary usdc)
<asset>/simready_usd/payloads/base.usda              prim tree, variant sets, per-mesh material:binding:physics
<asset>/simready_usd/payloads/Physics/physics.usda   neutral physics layer: PhysicsMaterialAPI values (read here)
<asset>/simready_usd/payloads/materials.usda         visual materials; mirrors the numbers as custom
                                                     pxr:usd:physics_* attributes and adds pxr:usd:physics_type
                                                     and omni:simready:nonvisual:base tokens (read here)
<asset>/simready_usd/payloads/instances*.usda        per-mesh mass, inertia, collision approximation
<asset>/simready_usd/payloads/{PhysX,MuJoCo,Newton}/enabled.usda   engine overlays
<asset>/simready_usd/payloads/geometries.usd         mesh data (large; not downloaded)
```

**How friction is authored.** In `physics.usda`, for example (asset `CT_injection_system_a01`):

```
over "physics_mat_m_opaque__metal__steel_clean_a01" (prepend apiSchemas = ["PhysicsMaterialAPI"])
{
    float physics:density = 7850
    float physics:dynamicFriction = 0.3
    float physics:restitution = 0.1
    float physics:staticFriction = 0.3
}
```

That asset has 20 meshes bound to 9 physics materials. The same numbers appear on the visual material in `materials.usda` as `custom double pxr:usd:physics_dynamicFriction = 0.3` next to `custom string pxr:usd:physics_type = "steel"` (the mirror disagrees with the physics layer on 55 of 9,597 prims). The Newton overlay adds `newton:rollingFriction = 0.0001` and `newton:torsionalFriction = 0.005` to every material; the PhysX and MuJoCo overlays add no friction of their own. Material names follow `physics_mat_[m_]<opacity>__<base>__<variant>_<version>`, where `<base>` is a class token (metal, plastic, rubber, glass, wood, cardboard, paper, fabric, ceramic, organic, ...). The non-visual base token comes from the SimReady Foundation's fixed vocabulary (requirement NVM.002: aluminum, steel, plastic, plexiglass, nylon, rubber, wood, cardboard, paper, fabric, leather, ...).

**What the spec requires.** SimReady Foundation requirement PMT.001 ("collider-material-binding", tagged essential) says every collider must bind a physics material and that materials "should have appropriate values for density, friction, and restitution". No value, source or tolerance is prescribed. The values in the library therefore come from NVIDIA's authoring pipeline, with no provenance recorded in the assets.

**How it was read.** `simready_audit.py` lists the asset folders through the Hub API, downloads only `Physics/physics.usda` and `materials.usda` for each asset with `huggingface_hub` (5,338 files, 119 MB, 10 parallel workers, 279 s), opens each physics layer with `pxr.Usd` and reads every prim carrying `UsdPhysics.MaterialAPI` (10 s for all assets), reads the `physics_type` and non-visual base tokens from `materials.usda` with `pxr.Sdf`, and writes one row per physics material prim. Sixteen files were missing: the 8 assets with no payloads (`consumable_microfiber_towel_folded_large_a01`, `consumable_microfiber_towel_stack_folded_a01`, `fireProtection_suppressionCylinder_fixed_b01`, `interiorPart_rear_seat_b01`, `powerTool_welding_station_d01`, `serviceTool_ac_recharge_recovery_machine_c01`, `suspension_coilover_front_remoteReservoir_b01`, `testingTool_battery_analyzer_b01`, the same 8 the community catalogue flags). One asset, `electricalSystem_alternator_c01`, has a physics layer with no physics material. Every physics material prim in every other asset has all four attributes authored.

## 2. Distribution of the friction values

`simready_friction.png`: one slot per distinct value, static in blue and dynamic in orange. 27 distinct values occur, from 0 (water) to 2.2; the table keeps those with 0.5% or more of the prims.

| value | dynamic: prims | share | static: prims | share |
|---|---|---|---|---|
| 0.3 | 1,982 | 20.7% | 1,919 | 20.0% |
| 0.35 | 143 | 1.5% | 59 | 0.6% |
| 0.4 | 4,259 | 44.4% | 3,494 | 36.4% |
| 0.45 | 599 | 6.2% | 144 | 1.5% |
| 0.47 | 477 | 5.0% | 0 | 0 |
| 0.5 | 574 | 6.0% | 1,752 | 18.3% |
| 0.55 | 0 | 0 | 148 | 1.5% |
| 0.6 | 136 | 1.4% | 62 | 0.6% |
| 0.61 | 0 | 0 | 477 | 5.0% |
| 0.65 | 0 | 0 | 48 | 0.5% |
| 0.7 | 730 | 7.6% | 805 | 8.4% |
| 0.8 | 582 | 6.1% | 582 | 6.1% |
| all other values | 115 | 1.2% | 107 | 1.1% |

Static equals dynamic on 7,032 prims (73.3%). The most common (static, dynamic) pairs: (0.4, 0.4) 34.9%, (0.3, 0.3) 20.0%, (0.5, 0.4) 7.9%, (0.7, 0.7) 7.6%, (0.5, 0.45) 6.2%, (0.8, 0.8) 6.1%, (0.61, 0.47) 5.0%, (0.5, 0.5) 4.1%; cumulative 91.8% for these eight, 96.9% for the top twelve.

**Does most of the library carry one default value?** Not one: dynamic 0.4 is 44% of prims, only 454 of 2,668 assets (17%) carry 0.4 on every material, and 1,148 assets (43%) have a single dynamic value across their materials. What the library carries is **one value per material class**. By the `physics_type` token:

| physics_type | prims | (static, dynamic) | share of class |
|---|---|---|---|
| plastic | 2,915 | (0.4, 0.4) | 99.4% |
| steel | 1,262 | (0.3, 0.3) | 99.8% |
| painted_metal | 743 | (0.5, 0.4) | 99.7% |
| cardboard | 718 | (0.7, 0.7) | 99.4% |
| paper | 583 | (0.5, 0.45) | 99.7% |
| rubber | 569 | (0.8, 0.8) | 99.6% |
| stainless_steel_thin | 535 | (0.3, 0.3) | 100% |
| aluminum | 477 | (0.61, 0.47) | 99.2% |
| glass | 357 | (0.4, 0.4) | 99.7% |
| wood | 189 | (0.5, 0.5) | 100% |
| fabric | 161 | (0.5, 0.5) | 100% |
| (no token) | 112 | (0.4, 0.4) | 36.6% |
| metal | 111 | (0.3, 0.3) | 100% |
| ceramic | 59 | (0.35, 0.3) | 100% |
| bronze | 53 | (0.4, 0.35) | 100% |
| concrete | 53 | (0.7, 0.6) | 100% |
| vinyl | 52 | (0.4, 0.4) | 100% |
| leather | 45 | (0.55, 0.5) | 97.8% |
| foods (apple, bagel, orange, banana, chocolate) | 365 | (0.45, 0.4) to (0.65, 0.6) | 99–100% each |

31 of the 33 classes with ten or more prims put one pair on 95% or more of their prims; the two exceptions are the 112 prims with no `physics_type` token and water. So the library is a class lookup table of about 35 entries applied at authoring time, with no spread, no source and no speed or load dependence. The full per-class, per-base and per-value tables are in the script's `simready_audit_tables.md` output.

## 3. Rig materials against the library

Matching rule (`simready_audit.py`, `RIG_MATERIALS`): a "specific" match is a keyword in the variant part of the material name within an allowed class (so a metal material named `crate_plywood` does not count as plywood); otherwise a "class" match takes every material whose base token, `physics_type` or non-visual base is the rig material's class. The modal pair is what a C2-SimReady baseline would use.

| rig material | role | match | matched on | materials (assets, prims) | modal (static, dynamic) | share | examples |
|---|---|---|---|---|---|---|---|
| PLA (3D printed) | object | class | plastic | 888 (1,423, 3,218) | (0.4, 0.4) | 91% | `plastic__black_a01`, `plastic__glossy_white_a01` |
| ABS sheet (extruded) | object | class | plastic | 888 (1,423, 3,218) | (0.4, 0.4) | 91% | as PLA; no ABS or PLA named anywhere |
| furniture felt | object | class | fabric | 97 (125, 166) | (0.5, 0.5) | 97% | `fabric__tweed_herringbone_a01`, `fabric__cotton_blue_o01`; no felt |
| neoprene rubber 60A | object | class | rubber | 172 (499, 646) | (0.8, 0.8) | 89% | `rubber__new_black_a01`, `rubber__conveyorBelt_a01`; no neoprene |
| corrugated cardboard | object | class | cardboard | 191 (341, 742) | (0.7, 0.7) | 97% | `cardboard__brown_a`, `cardboard__unique_box_cube_a` |
| cast acrylic | object | specific | acrylic | 1 (2, 2) | (0.4, 0.4) | 100% | `plastic__acrylic_fence_a01`; the `plexiglass` base (5 prims) is also (0.4, 0.4) |
| oak, grain along travel | object | specific | oak | 2 (3, 3) | (0.5, 0.5) | 100% | `wood__oak_b01`, `wood__oak_a02`; all 189 wood prims are (0.5, 0.5) |
| glazed ceramic tile | object | specific | glossy ceramic | 3 (8, 8) | (0.35, 0.3) | 100% | `ceramic__smoothGlossy_a01`, `ceramic__glossy_white_a01`; all ceramic is (0.35, 0.3) |
| stainless steel 304 | object | specific | stainless | 22 (161, 197) | (0.3, 0.3) | 98% | `metal__steelStainless_clean_a01`, `metal__steelStainless_a01` |
| white melamine shelf board | surface | specific | laminate | 2 (4, 4) | (0.4, 0.4) | 75% | `wood__laminate_gray_a` (3 prims at 0.4), one packing-table laminate at 0.5; no melamine |
| birch plywood, sanded | surface | specific | plywood | 4 (8, 10) | (0.5, 0.5) | 60% | `wood__plywoodRough_b01`, `wood__unique_crate_plywood_b`; the crate materials also carry stray 0 and 1 values; no birch |
| painted MDF (satin) | surface | class | wood | 136 (210, 237) | (0.5, 0.5) | 82% | `wood__recycled_a`, `wood__new_a`; no MDF, fibreboard or painted wood (paint appears only as `painted_metal`, (0.5, 0.4)) |

Count: 9 of 9 object bases and 3 of 3 surfaces have a plausible class value; 4 objects and 2 surfaces match by name, the rest by class only. 0 of 12 have a value specific to the rig's material (grade, finish, grain) and 0 of the 27 rig pairs have a pair value.

**What a C2-SimReady baseline would predict.** Each side's modal dynamic value combined by each engine's rule (Drake `2ab/(a+b)`, `CalcContactFrictionFromSurfaceProperties`; PhysX default `PxCombineMode::eAVERAGE`; MuJoCo element-wise maximum unless a geom has priority). The last column is the published lookup table's nominal (`friction_lookup.csv`, `mu_k_nominal`, 17 of 27 rig pairs have one).

| pair | object | surface | μ_obj | μ_surf | Drake | PhysX | MuJoCo | lookup μ_k |
|---|---|---|---|---|---|---|---|---|
| M01 | PLA | melamine | 0.4 | 0.4 | 0.40 | 0.40 | 0.40 | 0.29 |
| M02 | ABS | melamine | 0.4 | 0.4 | 0.40 | 0.40 | 0.40 | 0.30 |
| M03 | felt | melamine | 0.5 | 0.4 | 0.44 | 0.45 | 0.50 | – |
| M04 | neoprene | melamine | 0.8 | 0.4 | 0.53 | 0.60 | 0.80 | 0.83 |
| M05 | cardboard | melamine | 0.7 | 0.4 | 0.51 | 0.55 | 0.70 | 0.20 |
| M06 | acrylic | melamine | 0.4 | 0.4 | 0.40 | 0.40 | 0.40 | – |
| M07 | oak | melamine | 0.5 | 0.4 | 0.44 | 0.45 | 0.50 | 0.39 |
| M08 | ceramic tile | melamine | 0.3 | 0.4 | 0.34 | 0.35 | 0.40 | – |
| M09 | stainless 304 | melamine | 0.3 | 0.4 | 0.34 | 0.35 | 0.40 | 0.25 |
| M10 | PLA | plywood | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | 0.33 |
| M11 | ABS | plywood | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | 0.30 |
| M12 | felt | plywood | 0.5 | 0.5 | 0.50 | 0.50 | 0.50 | – |
| M13 | neoprene | plywood | 0.8 | 0.5 | 0.62 | 0.65 | 0.80 | 0.83 |
| M14 | cardboard | plywood | 0.7 | 0.5 | 0.58 | 0.60 | 0.70 | 0.21 |
| M15 | acrylic | plywood | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | – |
| M16 | oak | plywood | 0.5 | 0.5 | 0.50 | 0.50 | 0.50 | 0.34 |
| M17 | ceramic tile | plywood | 0.3 | 0.5 | 0.38 | 0.40 | 0.50 | – |
| M18 | stainless 304 | plywood | 0.3 | 0.5 | 0.38 | 0.40 | 0.50 | 0.22 |
| M19 | PLA | painted MDF | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | 0.29 |
| M20 | ABS | painted MDF | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | 0.30 |
| M21 | felt | painted MDF | 0.5 | 0.5 | 0.50 | 0.50 | 0.50 | – |
| M22 | neoprene | painted MDF | 0.8 | 0.5 | 0.62 | 0.65 | 0.80 | 0.83 |
| M23 | cardboard | painted MDF | 0.7 | 0.5 | 0.58 | 0.60 | 0.70 | 0.20 |
| M24 | acrylic | painted MDF | 0.4 | 0.5 | 0.44 | 0.45 | 0.50 | – |
| M25 | oak | painted MDF | 0.5 | 0.5 | 0.50 | 0.50 | 0.50 | 0.39 |
| M26 | ceramic tile | painted MDF | 0.3 | 0.5 | 0.38 | 0.40 | 0.50 | – |
| M27 | stainless 304 | painted MDF | 0.3 | 0.5 | 0.38 | 0.40 | 0.50 | – |

Two things stand out. The three engines' rules disagree by up to 0.27 on the same pair (M04: 0.53, 0.60, 0.80), more than the rig's whole simulated range of 0.10–0.50 spans in places, so a C2-SimReady value is engine-specific by construction. And the class values sit well away from the published lookup where both exist: cardboard 0.51–0.70 against 0.20; plastics 0.40–0.50 against 0.29–0.33; stainless steel 0.34–0.50 against 0.22–0.25; oak 0.44–0.50 against 0.34–0.39; only rubber (0.53–0.80 against 0.83) is close, and only under MuJoCo's maximum rule.

## 4. Conclusion

The values vary and cover the rig's materials, so a C2-SimReady baseline (per-material asset friction, combined by each simulator's rule) is possible. It is not a background finding in the spec's sense: the library is not one default, every rig material has a class in it, and the per-material numbers can be read from the assets without downloading meshes.

What that baseline would and would not be. It is a 35-class lookup table with one number per class and no provenance, so it tests the same thing as C2 (a published table) with coarser inputs and an extra, engine-dependent step; it carries no spread, so it would have to be given one (the lookup's tier-C factor is the natural choice, decision 3). For the three surfaces the class is generic wood or a four-prim laminate, which is the weakest part of the match. What the branch would learn from it is therefore less "what friction does SimReady know for our pairs" and more "how much the simulator's combination rule alone moves μ for identical inputs" (0.34 to 0.50 for steel on wood across Drake, PhysX and MuJoCo), which is a real input to the Drake-versus-Isaac comparison. Whether that is worth the branch's cost is decision 10, the team's.

## 5. Provenance, timing and how to repeat

- Start of audit: 2026-11-05 17:37 UTC (first Hub query). Final tables: 18:03 UTC. **26 minutes**, of which the library download took 279 s and the USD parse 10 s; the rest was finding the library, inspecting one asset's layers and the Foundation spec, and writing the script. This note took a further half hour.
- Assets read: 2,669 physics layers of 2,677 folders; 9,597 physics material prims; 5,338 files, 119 MB.
- Software: Python 3.13, `usd-core` 26.11, `huggingface_hub` 2.4.0, `pyarrow` 26.0.1 (only to inspect the community catalogue), all installed into `.venv` with pip. The machine's disk was at 100% when the audit started (the first install failed with "no space left on device"); Angelina freed space before the download.
- Downloaded layers are kept, outside the repo, at `/private/tmp/claude-501/-Users-angelinale-slide-to-target/simready_download/nvidia_simready_assets/<asset>/simready_usd/payloads/...` (plus one full sample asset's payloads and the community catalogue's parquet files beside it). The per-prim table `simready_friction_values.csv`, `simready_audit_summary.json` and `simready_audit_tables.md` are in the session scratchpad `/private/tmp/claude-501/-Users-angelinale-slide-to-target/dc561f25-ac06-41c6-a331-44f9c7ac6272/scratchpad/out/`. They are regenerated by the commands below.

```bash
.venv/bin/python simready_audit.py --download-dir /path/to/empty/dir --out-dir /path/to/out --figure simready_friction.png
```

```bash
.venv/bin/python simready_audit.py --out-dir /path/to/out --analyze-only
```

The first command lists, downloads (resumable; existing files are kept), reads and summarises; the second redoes the tables and figure from the CSV. `--limit 12` reads twelve assets for a quick check. Rerunning on a later snapshot of the dataset will change the counts; the class values are what to compare.

## Sources

- Dataset: https://huggingface.co/datasets/nvidia/simready-assets (card, file tree, `CT_injection_system_a01` sample)
- Space: https://huggingface.co/spaces/nvidia/simready-central ; contributor dataset (gated): https://huggingface.co/datasets/nvidia/simready-central-library
- Community catalogue: https://huggingface.co/datasets/thisisrick25/nvidia-simready-catalog (asset count, size, no-payload list; it has no friction columns)
- SimReady Foundation spec: https://github.com/NVIDIA/simready-foundation , `nv_core/tiers/simready_foundation_tier_core/.../physics_materials/requirements/collider-material-binding.md` (PMT.001) and `.../nonvisual_materials/requirements/material-base.md` (NVM.002)
- Partner datasets: ForgeCAD, Vertiv, frankarobotics, deeproboticslab, standardbots, VAST-AI, NdotLight, Sandflow `simready` datasets on Hugging Face (file counts from the Hub API, 5 Nov 2026)
- Combination rules: MuJoCo modeling guide, contact parameters (element-wise maximum unless priority differs), https://mujoco.readthedocs.io/en/stable/modeling.html ; PhysX `PxMaterial::setFrictionCombineMode` default `eAVERAGE`, https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/_api_build/classPxMaterial.html ; Drake `2ab/(a+b)` as used in `scene.py`
- OpenUSD physics schema: https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_materials
