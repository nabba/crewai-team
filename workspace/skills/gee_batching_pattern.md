# Google Earth Engine: One Round-Trip Per Script

*kb: episteme | id: skill_episteme_gee_batching_pattern | status: active | usage: 0 | created: 2026-05-02T00:00:00+00:00*

# The Rule

When using the `gee_run_script` tool, **every `.getInfo()` call is a synchronous round-trip to Google's compute backend** that costs ~20–40 seconds of wall-clock time. The tool budget is 240 s; the orchestrator's zero-output watchdog kills crews after 20 min of silence.

A naive script that calls `.getInfo()` inside a Python `for` loop (one call per year, one per land-cover class, one per region) will almost always time out, even though *the same aggregation done server-side* completes in ~110 s.

**Aggregate everything server-side, then pull the result ONCE.**

# Anti-Pattern (do NOT write this)

```python
# BAD — 13 round-trips × ~30s each → 240s tool timeout → crew dies
gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
loss_year = gfc.select("lossyear").clip(estonia.geometry())
out = {}
for yr in range(12, 25):                     # 13 years (2012–2024)
    mask = loss_year.eq(yr)
    ha = mask.multiply(ee.Image.pixelArea()).divide(1e4).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=estonia.geometry(),
        scale=100, maxPixels=int(1e9),
    ).get("lossyear").getInfo()              # ← round-trip per year
    out[2000 + yr] = ha
result = out
```

Why it dies: each `.getInfo()` blocks until Google computes that one number. 13 sequential round-trips on a slow-pixel reduction cleanly exceed the 240 s tool budget. Google's compute backend can do all 13 reductions *in one pass* for the same wall-clock cost as one reduction — but only if you express the work as a single graph.

# The Right Pattern (canonical Estonia example)

```python
# GOOD — single frequencyHistogram → 1 round-trip → ~109s (verified 2026-05-02)
gfc = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
loss_year = gfc.select("lossyear").clip(estonia.geometry())

# Mask out "no loss" pixels (lossyear == 0) before histogramming so the
# reducer doesn't burn time on the dominant class.
loss_only = loss_year.updateMask(loss_year.gt(0))

hist = loss_only.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),  # ← all years in one pass
    geometry=estonia.geometry(),
    scale=100,
    maxPixels=int(1e9),
    bestEffort=True,
).get("lossyear").getInfo()                   # ← one round-trip total

# hist is {"12": 1234, "13": 5678, ...} — pixel counts per year code
result = {2000 + int(k): round(v, 0) for k, v in hist.items()}
```

If you need *area* rather than pixel count, multiply each pixel by `ee.Image.pixelArea()` server-side first, then use a `Reducer.sum().group(groupField=0)` — still one call.

# Decision Table

| Question shape | Reducer to reach for |
|---|---|
| "How many of class X in region?" | `Reducer.sum()` on a binary mask, one call |
| "How many of *each* class in region?" | `Reducer.frequencyHistogram()`, one call |
| "Total area per class?" | `Reducer.sum().group(groupField=0)` on `pixelArea`, one call |
| "Same metric across N regions?" | Map the reducer over an `ee.FeatureCollection`; one `.getInfo()` on the result list |
| "Time-series across N dates?" | `ee.ImageCollection.map(...)` then `.aggregate_array("prop")`, one call |

# Other Single-Round-Trip Idioms

**Per-region aggregation** — never loop over Python regions:
```python
fc = ee.FeatureCollection("...some admin polygons...")
def add_loss(f):
    s = loss_only.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=f.geometry(),
        scale=100, maxPixels=int(1e9),
    ).get("lossyear")
    return f.set("loss_px", s)
result = fc.map(add_loss).getInfo()   # one round-trip, returns GeoJSON
```

**Per-date time-series** — never loop over Python dates:
```python
ic = ee.ImageCollection("COPERNICUS/S2_SR").filterDate("2024-04-01", "2024-09-30")
def mean_ndvi(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
    m = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=estonia.geometry(),
        scale=100, maxPixels=int(1e9),
    ).get("ndvi")
    return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "ndvi": m})
result = ee.FeatureCollection(ic.map(mean_ndvi)).getInfo()  # one round-trip
```

# Wrapper Behavior

The tool's `_run_user_script` wrapper checks whether your `result` variable has a `.getInfo()` method and calls it for you. So the cleanest scripts assign an `ee.Dictionary`, `ee.Number`, or `ee.FeatureCollection` to `result` and let the wrapper do the single pull:

```python
result = loss_only.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=estonia.geometry(),
    scale=100, maxPixels=int(1e9), bestEffort=True,
).get("lossyear")     # ← still an ee.Dictionary; wrapper unwraps it
```

# Escape Hatch

Server-side aggregation has limits — `frequencyHistogram` blows up if there are millions of distinct values. For genuinely large discrete domains, *export* to a Drive/GCS asset (asynchronous, not subject to the 240 s budget) instead of trying to pull the whole result through `.getInfo()`. But this is rare; for the questions the crew actually gets (per-year, per-class, per-region), one round-trip is achievable.

# Reference

* Empirical timing data, Estonia 24-year deforestation aggregation, 2026-05-02:
  - Naive per-year loop: ≥240 s (timed out)
  - Single `frequencyHistogram`: 109 s (returned `{2012: ..., ..., 2024: ...}`)
* Tool source: `app/tools/gee_tool.py` — see the `gee_run_script` description for the inline rule.
