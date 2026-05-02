# Google Earth Engine integration

Satellite-imagery analysis surfaced to the coding crew via a single
sandboxed tool, `gee_run_script`. Lets agents run aggregations over
Hansen Global Forest Change, Sentinel-2/Landsat composites, NDVI
time-series, MODIS, GEDI lidar, etc., without downloading imagery —
Google's compute does the heavy lifting and returns aggregated
numbers / vector summaries.

Shipped 2026-05-02 (commit `b4208dd`). Description-and-skill guidance
hardening shipped same day after a production crew wrote a per-year
sequential-`getInfo()` script and timed out at 240s.

---

## 1. Components

| Path | Role |
|---|---|
| `app/tools/gee_tool.py` | The tool itself: `create_gee_tools()` factory + `gee_run_script` BaseTool + `_ensure_initialised` lazy-init guard. |
| `app/agents/coder.py` | Wires the tool into the coder agent's inventory via `optional_tool_group("coder", "gee")`. |
| `secrets/gee-service-account.json` | Service-account JSON. Mode 600 on host; bind-mounted read-only into the container at `/app/secrets/`. Gitignored. |
| `docker-compose.yml` | Mounts `./secrets:/app/secrets:ro`. |
| `.env` / `.env.example` | `GOOGLE_APPLICATION_CREDENTIALS` + `GEE_PROJECT` env vars. |
| `requirements.txt` | `earthengine-api>=1.0.0`. |
| `workspace/skills/gee_batching_pattern.md` | Handwritten skill teaching the round-trip rule + worked good/bad pattern. Indexed into ChromaDB by the idle skill-indexer so retrieval surfaces it on relevant queries. |
| `tests/test_gee_tool.py` | 14 unit tests (env resolution, factory, init caching, description guidance, skill presence). |

---

## 2. Setup walkthrough

The full step-by-step lives in `.env.example` for users running BotArmy
fresh. Headline:

1. Create a GCP project at <https://console.cloud.google.com/projectcreate>.
2. Enable Earth Engine API on it.
3. Register the project at <https://code.earthengine.google.com/register>
   (commercial/noncommercial choice).
4. Create a service account and grant **both** `Earth Engine Resource
   Writer` AND `Service Usage Consumer` IAM roles. The latter is the
   one most people miss — without it, `ee.Initialize()` succeeds but
   every API call returns 403.
5. Download the service-account JSON key.
6. Drop it at `secrets/gee-service-account.json`, set
   `GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gee-service-account.json`
   and `GEE_PROJECT=<project-id>` in `.env`.
7. `docker compose up -d --force-recreate gateway`.

Verification (one-liner):

```bash
docker exec -e GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gee-service-account.json \
    crewai-team-gateway-1 python -c "
from app.tools.gee_tool import _ensure_initialised
ok, err = _ensure_initialised()
print('OK' if ok else err)
"
```

If it prints `OK`, the tool is live and `create_gee_tools('coder')`
returns `[gee_run_script]`. If it prints an error mentioning
"Service Usage Consumer", revisit step 4.

---

## 3. The single-roundtrip rule (critical)

Every `.getInfo()` call is a synchronous network round-trip to
Google's compute backend, taking ~20–40 seconds. The tool's wall-clock
budget is 240 s; the orchestrator's zero-output watchdog kills crews
after 20 min of silence. So:

> **One `.getInfo()` per script.** Aggregate server-side, then pull once.

The tool's `description` field hard-codes both an anti-pattern label
(`# BAD`) and the fix label (`# GOOD`) so the LLM sees the contrast
when deciding what script to write:

```python
# BAD (13 round-trips, times out at 240s):
for yr in range(12, 25):
    out[yr] = mask.eq(yr).reduceRegion(...).getInfo()

# GOOD (1 round-trip, ~110s):
hist = loss.updateMask(loss.gt(0)).reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(), ...
).get('lossyear').getInfo()
```

Per-year/per-class/per-region values should use:

- `ee.Reducer.frequencyHistogram()` for categorical bands (year codes,
  land-cover classes).
- `ee.Reducer.group()` for cross-tabulating two bands.
- `ee.FeatureCollection(years.map(per_year_fn))` then a single
  `.getInfo()` for arbitrary per-element computation.

The handwritten skill at `workspace/skills/gee_batching_pattern.md`
covers more patterns + dataset references.

---

## 4. Sandbox semantics

`gee_run_script` runs the supplied Python in an in-process sandbox
dict pre-loaded with:

| Name | Value |
|---|---|
| `ee` | The Earth Engine API module. |
| `estonia` | `ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ADM0_NAME=='Estonia')` — convenience AOI. |
| `result` | Initially `None`. The wrapper calls `.getInfo()` on this once when the script returns, so leaving it as an `ee.Number` / `ee.Dictionary` / `ee.List` is preferred over calling `.getInfo()` yourself. |

The sandbox IS in-process — same trust model as the coding-crew's
`base_crew.run_python` sandbox. Forge-audit gates apply when the tool
is invoked from agent code that's been forged.

Outbound network calls go to `earthengine.googleapis.com` only;
local CPU/RAM cost per call is small (Google does the work).

Heavy outputs (export-to-Drive / export-to-GCS) are logged but never
block the call returning — they run as Earth Engine "tasks" on
Google's side and complete asynchronously.

---

## 5. Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `create_gee_tools('coder')` returns `[]` | env var unset OR JSON file missing | Check `GOOGLE_APPLICATION_CREDENTIALS` + verify file at `/app/secrets/gee-service-account.json` |
| `ee.Initialize()` fails with `Caller does not have required permission to use project ...` | Service account missing `Service Usage Consumer` IAM role | Add the role at `iam-admin/iam?project=<id>` |
| `Project is not registered to use Earth Engine` | Project never registered for Earth Engine | Visit `code.earthengine.google.com/register` for the project |
| Tool times out at 240 s on multi-year aggregation | Naive per-year `getInfo()` loop | Rewrite as single-pass server-side aggregation (§3) |
| Crew produces output about an unrelated topic | Skill-retrieval matched a stale auto-skill on a generic verb | Spawned task: "Fix skills-retrieval surface-keyword contamination" — see `app/skills/` work |

---

## 6. Live verification (Estonia 2001-2024)

```python
# This is what the verified pipeline does today (~150s wall-clock):
hansen = ee.Image("UMD/hansen/global_forest_change_2024_v1_12").clip(estonia.geometry())
loss = hansen.select("lossyear")
hist = loss.updateMask(loss.gt(0)).reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=estonia.geometry(),
    scale=100, maxPixels=int(1e9), bestEffort=True,
).get("lossyear").getInfo()
result = {2000 + int(k): round(v, 0) for k, v in hist.items()}
# → {2001: 47955, 2002: 60799, ..., 2019: 118256, ..., 2024: 91418}
```

Output of one such run is committed at `workspace/output/estonia_forest_report.md`.

---

## 7. Cross-references

- `docs/CONTROL_PLANES.md` §Request-Path — where `gee_run_script`
  sits in the dispatch flow.
- `docs/FORGE.md` — forge gates on agent-generated tools (the
  GEE tool is registered, not forged, so the forge gates don't fire
  on it specifically).
- `app/recovery/strategies/sandbox_execute.py` — recovery loop
  fallback that can attempt to re-run a failed tool call inside
  the sandbox.
- The 2026-05-02 production incident: a coding crew misread
  "execute the plan" and wrote a Weather Forecast System instead
  of using `gee_run_script` for forest analysis. Two follow-up
  tasks spun off: skills-retrieval contamination, crew-level
  zero-progress watchdog.
