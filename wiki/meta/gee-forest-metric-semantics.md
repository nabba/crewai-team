---
aliases:
- gee forest metric semantics
author: idle_scheduler.wiki_synthesis
confidence: medium
created_at: '2026-05-16T21:40:35Z'
date: '2026-05-16'
related: []
relationships: []
section: meta
source: workspace/skills/gee_forest_metric_semantics.md
status: active
tags:
- self-improvement
- skills
- auto-synthesised
title: 'Forest metrics in GEE: pick the metric you actually need'
updated_at: '2026-05-16T21:40:35Z'
version: 1
---

# Forest metrics in GEE: pick the metric you actually need

*kb: episteme | id: skill_episteme_gee_forest_metric_semantics | status: active | usage: 0 | created: 2026-05-02T20:35:00+00:00*

# Why this exists

A user asking for *"forest age maps"* or *"deforestation maps"* is using shorthand for one of several distinct quantities. Picking the wrong one produces a script that runs cleanly but answers the wrong question. The 2026-05-02 Estonia dispatch shipped a working pipeline that computed *"years since most recent disturbance"* and labelled it *"forest age"* — the user noticed because intact 200-year-old spruce stands got `age = 0` (same value as land cleared yesterday).

This skill names the common metrics, the data needed for each, and the tradeoffs.

---

# Deforestation — usually unambiguous

For *"deforestation per year"*, the canonical answer is **Hansen Global Forest Change** (`UMD/hansen/global_forest_change_<latest>_v1_<n>`):

- `lossyear` band: per-pixel year code (1 = 2001, 2 = 2002, …, 24 = 2024)
- `loss` band: binary (1 = ever lost, 0 = not)
- `treecover2000` band: % canopy cover in 2000

Use the latest dataset version available — at time of writing `UMD/hansen/global_forest_change_2024_v1_12` is current. Older snapshots (e.g. `_2023_v1_11`) miss the most recent year.

For *area-per-year*, multiply the per-year mask by `ee.Image.pixelArea()` server-side and use `Reducer.sum().group(groupField=0)` in one call. See `gee_batching_pattern.md` for the round-trip rule.

---

# Forest age — three distinct quantities, one shorthand

When a user says *"forest age"*, they almost always mean one of:

### 1. Stand age — biological age of the standing forest

The age in years of the trees currently growing on a pixel. A 200-year-old primary spruce stand has `stand_age = 200`. **Hansen alone CANNOT compute this** — Hansen only knows about the satellite era (post-2000). For a true stand-age product you need:

- **Besnard et al. (2021)** global forest age, 1 km, ESS&D — model-based estimate combining inventory + biomass + remote sensing. Available via direct download or as an Earth Engine asset (search the catalog).
- **National forest inventory polygons** with stand-age attributes. For Estonia: the Estonian Land Board (Maa-amet) publishes forest inventory data, and **RMK (Riigimetsa Majandamise Keskus)** publishes stand-level data including age class. For other countries: equivalent national agencies — most EU member states have one.
- **GEDI Level 4B** + machine learning — research-grade, generally not what an end-user wants directly.

### 2. Regrowth age — years since most recent disturbance

For pixels that lost forest at some point in the satellite era, *"how many years ago"*. Computable from Hansen alone (`current_year - (2000 + lossyear)` for pixels where `lossyear > 0`). **Only meaningful for disturbed pixels.** Intact pixels need a sentinel value (NaN / masked / explicit "≥24 years" marker) — NOT zero, which silently conflates "never disturbed" with "disturbed yesterday".

### 3. Time-since-last-event — generalization of regrowth age

Like (2) but for events broader than just clear-cutting (fire, selective logging, wind damage). Needs disturbance-detection products like **Sentinel-1 SAR change detection**, **MODIS burned area** (`MODIS/061/MCD64A1`), or **GFW Integrated Alerts** (`projects/glad/lcluc_integrated_alerts`).

---

# Composing the right answer

The honest design pattern when *"forest age"* is requested without further qualifier:

1. Ask the user (if conversation allows): *"Stand age (biological) or regrowth age (years since last disturbance)?"*
2. If you can't ask, default to the conservative: produce **both** outputs side-by-side, clearly labelled. Stand age via Besnard (or RMK if Estonia-specific) and regrowth age via Hansen. A two-column table is much clearer than one ambiguous column.
3. NEVER set intact-forest pixels to `age = 0` in a "regrowth age" map — use `masked` or `null` or an explicit sentinel like `-1` with documentation. Setting to zero is a known footgun (the 2026-05-02 dispatch did exactly this and the user immediately spotted the conflation).
4. When delivering the visual maps: use the `render_map(image, region, name, vis_params, ...)` helper inside `gee_run_script` (added 2026-05-03, audit H10 — see `gee_batching_pattern.md`).  Use a sequential palette (`viridis` / `magma` / white-to-green) for age maps; binary (`palette=['white', 'red']`) for deforestation masks; encode the metric in the filename (`stand_age_*.png` vs `regrowth_age_*.png`) so downstream consumers don't confuse them.

# Other shorthand traps

| Shorthand | Distinct meanings | Right dataset |
|---|---|---|
| *"forest cover"* | Canopy density (% per pixel) vs binary forest/not-forest at some threshold | Hansen `treecover2000` (continuous %) — apply your own threshold for binary |
| *"forest loss"* | Permanent removal vs temporary disturbance vs degradation (selective logging) | Hansen `loss` is "stand-replacing disturbance" — does NOT capture selective logging or short-term fire recovery |
| *"forest gain"* | Tree planting vs natural regrowth vs new forest in previously non-forested areas | Hansen `gain` is "non-forest → forest" 2000-2012 only — discontinued. Use newer products if currentness matters. |
| *"degradation"* | Crown thinning, partial loss, biomass decline | Sentinel-1 SAR backscatter trends; GFW Integrated Alerts; or per-pixel NDVI trend regression — Hansen does NOT cover this |

---

# Output convention reminders

- Every map produced should have **units in metadata** (% canopy / binary 0-1 / years since loss / etc.) — and the units in the legend if rendered.
- For per-year deforestation, output BOTH pixel counts AND area (hectares or km²) — pixel counts depend on resolution, area is portable. `30m × 30m = 0.09 ha`; `Sentinel-2 10m × 10m = 0.01 ha`.
- When delivering a multi-year report, include a *summary table* (year × hectares) AS WELL as the rasters — the user almost always wants both. Only delivering the GeoTIFF exports without a numeric summary forces them to compute it themselves.
- For stand age vs regrowth age, when delivering "forest age" rasters, **encode the meaning in the filename** (`stand_age_*.tif` vs `regrowth_age_*.tif`) — `forest_age_*.tif` alone is ambiguous and downstream consumers will misinterpret it.

---

# Reference

- Hansen et al. 2013, "High-Resolution Global Maps of 21st-Century Forest Cover Change", *Science* 342(6160). The canonical Hansen paper; explains what `lossyear` / `loss` / `gain` actually measure.
- Besnard et al. 2021, "Mapping global forest age from forest inventories, biomass and climate data", *Earth System Science Data* 13. The 1 km global forest age dataset.
- GFW (Global Forest Watch) is the user-facing portal for Hansen + integrated alerts. Their docs explain the same caveats in user-facing language.
- Estonian Land Board (Maa-amet) geoportal — `https://geoportaal.maaamet.ee` — for national inventory data.
- RMK — `https://www.rmk.ee` — Estonian state-forest manager, publishes stand-level data.
