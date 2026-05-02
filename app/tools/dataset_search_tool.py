"""
dataset_search_tool.py — discovery tool for geospatial / forest /
remote-sensing datasets the system should reach for.

Built for the 2026-05-02 audit (item 3 of the "what's left" list).
The user's wish was for the system to *"produce the best algorithm on
its own"*; that requires the LLM to know about datasets it isn't
explicitly told about.  Without this tool, the design phase keeps
defaulting to Hansen-only solutions even when (e.g.) Besnard 2021
forest age or RMK national inventory data would produce a better
answer.

Scope: a curated index of ~20 commonly-needed datasets, hand-authored
rather than crawled.  Each entry names the dataset, its access
pattern (GEE asset id, direct URL, WFS endpoint, etc.), the FAST
pattern for using it, and the caveats that matter.  The tool returns
the top-N matches by simple keyword overlap + tag matching.

Out of scope (intentionally):
  * Live querying of the GEE catalog API — would add network calls
    on every search.  Static index covers the 80% case.
  * Crawling national geoportals — too brittle.  Curated entries
    for Estonian agencies + a "see your country's equivalent"
    pointer is more honest.
  * Embedding-based retrieval — overkill for ~20 entries; keyword
    overlap finds the right matches.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# Each entry: (id, name, description, access_pattern, fast_use, caveats, tags).
# Tags are matched case-insensitively against the user query.  Order
# is the tie-breaker when multiple entries score equal — earlier
# wins, so put the canonical / most-recommended option first within
# each category.
_DATASETS: tuple[dict, ...] = (
    # ── Forest cover / loss / disturbance ─────────────────────────
    {
        "id": "hansen-gfc",
        "name": "Hansen Global Forest Change (UMD/Google/USGS)",
        "description": "Annual global tree-cover, loss, and gain since 2000 at 30 m. The canonical source for stand-replacing forest disturbance.",
        "access_pattern": "ee.Image('UMD/hansen/global_forest_change_<latest>_v1_<n>') — at time of writing, latest is `2024_v1_12`. Bands: treecover2000, loss, lossyear, gain, datamask.",
        "fast_use": "For per-year area: `lossyear` band → `frequencyHistogram` reducer in ONE reduceRegion call. See gee_batching_pattern.md.",
        "caveats": "Captures stand-replacing disturbance only (no degradation / selective logging). `gain` band is 2000-2012 only — discontinued. Use latest version; older snapshots miss recent years.",
        "tags": ("forest", "deforestation", "loss", "tree-cover", "hansen", "gfc", "global", "annual", "30m", "estonia"),
    },
    {
        "id": "gfw-integrated-alerts",
        "name": "GFW Integrated Alerts (GLAD-L + GLAD-S2 + RADD)",
        "description": "Near-real-time forest-disturbance alerts combining Landsat (GLAD-L), Sentinel-2 (GLAD-S2) and SAR (RADD).",
        "access_pattern": "ee.ImageCollection('projects/glad/lcluc_integrated_alerts')",
        "fast_use": "filterDate + reduceRegion for area-since-date queries; per-tile alerts arrive within days of the disturbance.",
        "caveats": "Tropical-focused historically; coverage in temperate/boreal regions is improving but check first. Recent alerts may be revised after confirmation.",
        "tags": ("forest", "alerts", "disturbance", "near-real-time", "tropical", "monitoring"),
    },
    # ── Forest age / structure ─────────────────────────────────────
    {
        "id": "besnard-forest-age",
        "name": "Besnard et al. (2021) global forest age (1 km)",
        "description": "Modelled global forest stand-age combining inventory data, biomass, and climate. Genuine BIOLOGICAL stand age — the right answer when a user asks 'forest age' without further qualifier.",
        "access_pattern": "Direct download (NetCDF) from https://doi.org/10.5194/essd-13-4881-2021. May also be available as a community Earth Engine asset — search the EE catalog or community asset index.",
        "fast_use": "After upload to GEE: `ee.Image('users/<your-account>/besnard_2021_forest_age')` then standard reduceRegion. Or process the NetCDF locally with rioxarray + xarray.",
        "caveats": "1 km resolution — coarse for parcel-level work. Snapshot from ~2010-2014 inventory data; doesn't update as forest changes. For Hansen-disturbed pixels post-snapshot, the value is stale.",
        "tags": ("forest", "age", "stand-age", "biomass", "biological", "besnard", "global", "1km"),
    },
    {
        "id": "gedi-l4b",
        "name": "GEDI L4B Biomass / structure (NASA, Maryland)",
        "description": "Spaceborne lidar canopy height + biomass density at 1 km. Direct measurement of vertical forest structure.",
        "access_pattern": "ee.Image('LARSE/GEDI/GEDI04_B_002')",
        "fast_use": "reduceRegion for area-mean biomass; sample for stand-level estimates.",
        "caveats": "Coverage is between ~52°N and ~52°S — most of Estonia (~58°N) is OUT OF GEDI's nominal coverage. Best for tropical / temperate latitudes <52°.",
        "tags": ("forest", "biomass", "structure", "canopy", "lidar", "gedi", "1km"),
    },
    # ── Optical satellite imagery ──────────────────────────────────
    {
        "id": "sentinel-2-sr",
        "name": "Sentinel-2 Surface Reflectance (Copernicus / ESA)",
        "description": "10–60 m multispectral imagery, ~5-day revisit. Best general-purpose optical source for vegetation, water, urban, agriculture.",
        "access_pattern": "ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi).filterDate(start, end).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))",
        "fast_use": "Median composite over a season; QA60-band cloud-mask; NDVI = (B8 - B4) / (B8 + B4).",
        "caveats": "Frequent cloud cover at high latitudes (Estonia: April-Sept usable). Pair with Sentinel-1 SAR for cloud-resilient time-series.",
        "tags": ("optical", "satellite", "sentinel-2", "10m", "vegetation", "ndvi", "cloud", "europe"),
    },
    {
        "id": "sentinel-1-grd",
        "name": "Sentinel-1 SAR (Copernicus / ESA)",
        "description": "C-band radar backscatter at 10 m, ~6-day revisit, cloud-penetrating. Essential for monitoring during cloud-cover periods.",
        "access_pattern": "ee.ImageCollection('COPERNICUS/S1_GRD').filter(ee.Filter.eq('instrumentMode', 'IW')).filterBounds(roi)",
        "fast_use": "VH/VV ratio for surface change detection; per-pixel time-series anomaly for disturbance.",
        "caveats": "Speckle noise — apply a Lee or Refined Lee filter before reductions. Geometric distortions in mountainous terrain.",
        "tags": ("sar", "radar", "sentinel-1", "10m", "cloud-resilient", "all-weather"),
    },
    {
        "id": "landsat-c2",
        "name": "Landsat 8 + 9 Collection 2 (USGS / NASA)",
        "description": "30 m multispectral imagery since 1984. Long historical record — essential for change detection back to the 1980s.",
        "access_pattern": "ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') and 'LC09/C02/T1_L2'",
        "fast_use": "Median composite, NDVI, NBR for fire severity. For pre-2013 use LT04/LT05/LE07.",
        "caveats": "16-day revisit per satellite — pair both for ~8-day combined. Cloud cover same caveats as Sentinel-2.",
        "tags": ("optical", "landsat", "30m", "historical", "long-term", "since-1984"),
    },
    # ── Land cover ──────────────────────────────────────────────────
    {
        "id": "esa-worldcover",
        "name": "ESA WorldCover 10 m global land cover",
        "description": "11-class global land cover at 10 m for 2020 and 2021. Cleanest global LC product available.",
        "access_pattern": "ee.Image('ESA/WorldCover/v200/2021') — 11 classes including tree-cover, shrubland, grassland, cropland, built-up, bare, water.",
        "fast_use": "Per-class area: `frequencyHistogram` over the Map_band.",
        "caveats": "Snapshot — only 2020 and 2021. For multi-year LC change, use Dynamic World or composite Hansen + WorldCover.",
        "tags": ("land-cover", "worldcover", "esa", "10m", "global", "classification"),
    },
    {
        "id": "dynamic-world",
        "name": "Dynamic World — near-real-time land cover (Google / WRI)",
        "description": "10 m near-real-time probabilistic land cover, updated every Sentinel-2 pass. 9 classes.",
        "access_pattern": "ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(roi).filterDate(start, end)",
        "fast_use": "Mode reducer over a date window for stable LC; use the per-class probability bands for fuzzy queries.",
        "caveats": "Probabilistic — pixel can have spread of probabilities across classes. Not a cleaned-up product like WorldCover.",
        "tags": ("land-cover", "dynamic-world", "10m", "near-real-time", "probabilistic"),
    },
    # ── Fires + burned area ─────────────────────────────────────────
    {
        "id": "modis-burned",
        "name": "MODIS Burned Area (MCD64A1)",
        "description": "Monthly global burned area at 500 m since 2000.",
        "access_pattern": "ee.ImageCollection('MODIS/061/MCD64A1') — band: BurnDate.",
        "fast_use": "Per-month burned-area sum; combine with land-cover for forest-fire-specific area.",
        "caveats": "500 m resolution — small fires (<25 ha) under-detected. For higher-res use Sentinel-2 NBR / Landsat NBR + thresholding.",
        "tags": ("fire", "burned-area", "modis", "500m", "monthly", "global"),
    },
    # ── Climate / weather ───────────────────────────────────────────
    {
        "id": "era5-land",
        "name": "ERA5-Land hourly climate reanalysis (ECMWF)",
        "description": "0.1° (~9 km) hourly climate variables — temperature, precipitation, soil moisture, snow, radiation.",
        "access_pattern": "ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')",
        "fast_use": "filterDate + reduceRegion for area-mean per-time-step; for monthly aggregates use 'ECMWF/ERA5_LAND/MONTHLY_AGGR'.",
        "caveats": "Resolution is coarse for plot-level work. For Estonia (small country) use the monthly aggregates unless hourly granularity is essential.",
        "tags": ("climate", "weather", "era5", "temperature", "precipitation", "ecmwf", "9km", "hourly"),
    },
    # ── Population / built-up ──────────────────────────────────────
    {
        "id": "worldpop",
        "name": "WorldPop population density",
        "description": "Global population at 100 m or 1 km, annual since 2000.",
        "access_pattern": "ee.ImageCollection('WorldPop/GP/100m/pop')",
        "fast_use": "filter('country', 'EST') + reduceRegion sum for population in an AOI.",
        "caveats": "Modelled estimates — uncertainty higher in low-density rural areas.",
        "tags": ("population", "worldpop", "100m", "demographic", "global"),
    },
    # ── Hydrology ───────────────────────────────────────────────────
    {
        "id": "hydrobasins",
        "name": "HydroBASINS / HydroLAKES (WWF / McGill)",
        "description": "Hierarchical river basins + global lake polygons. Essential for any watershed-scoped query.",
        "access_pattern": "ee.FeatureCollection('WWF/HydroSHEDS/v1/Basins/hybas_<level>') for level 1-12; lakes via 'WWF/HydroSHEDS/v1/FreeFlowingRivers'.",
        "fast_use": "Filter to AOI bounds; use as the geometry argument for reduceRegion.",
        "caveats": "Static snapshot (2010-era HydroSHEDS). For dynamic surface water see JRC Global Surface Water.",
        "tags": ("hydrology", "watershed", "basin", "river", "lake", "hydrosheds"),
    },
    # ── Administrative boundaries ──────────────────────────────────
    {
        "id": "fao-gaul",
        "name": "FAO GAUL administrative boundaries (level 0/1/2)",
        "description": "Global administrative boundaries at country (level 0), region (1), and district (2). Pre-loaded as `estonia` in the gee_run_script sandbox.",
        "access_pattern": "ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'Estonia'))",
        "fast_use": "The sandbox already pre-loads `estonia` for level 0; for sub-national use level 1 or 2 with the same filter pattern.",
        "caveats": "2015 snapshot — boundary changes since then aren't reflected.",
        "tags": ("admin", "boundary", "fao", "gaul", "country", "estonia", "geometry"),
    },
    {
        "id": "natural-earth",
        "name": "Natural Earth admin boundaries",
        "description": "Public-domain world map at 1:10m, 1:50m, 1:110m scales. Cleaner than FAO GAUL for cartography.",
        "access_pattern": "Direct download from https://www.naturalearthdata.com — upload as EE asset.",
        "fast_use": "Use as the geometry argument for clip / reduceRegion.",
        "caveats": "Cartographic generalization — coastlines smoothed at lower resolutions.",
        "tags": ("admin", "boundary", "natural-earth", "country", "cartography"),
    },
    # ── Estonian national datasets ──────────────────────────────────
    {
        "id": "estonia-rmk",
        "name": "RMK (Riigimetsa Majandamise Keskus) — Estonian state forest data",
        "description": "Stand-level forest inventory for Estonian state forests, including age class, species, basal area, volume.",
        "access_pattern": "Web portal: https://www.rmk.ee — public datasets via download. Some layers exposed via WFS at https://kaart.rmk.ee/geoserver/wfs",
        "fast_use": "Fetch GeoJSON via firecrawl_scrape on the WFS endpoint with a bbox filter; convert to ee.FeatureCollection by uploading or via geojson_to_ee.",
        "caveats": "Estonian state forests only (~40% of the country). For private forests use the SMI or Maa-amet inventory.",
        "tags": ("forest", "inventory", "estonia", "rmk", "stand-age", "national"),
    },
    {
        "id": "estonia-maa-amet",
        "name": "Maa-amet (Estonian Land Board) geoportal",
        "description": "Estonian national geoportal — orthophotos, cadastral parcels, soil, geology, topographic maps.",
        "access_pattern": "WMS/WFS at https://geoportaal.maaamet.ee/eng/Spatial-Data — also public download index.",
        "fast_use": "WFS endpoint for vector data; tile WMS for orthophotos. Use firecrawl_scrape for non-WFS pages.",
        "caveats": "Layers in EPSG:3301 (Estonian National Grid) — reproject if combining with WGS84 sources.",
        "tags": ("estonia", "maa-amet", "national", "orthophoto", "cadastral", "geoportal", "wfs", "wms"),
    },
    # ── Pointer entries (when no curated dataset matches the question)
    {
        "id": "country-equivalent-pointer",
        "name": "Your country's equivalent of RMK / Maa-amet (POINTER, not a dataset)",
        "description": "Most countries have a national land-survey + national-forest agency that publishes inventory data. Find it via 'national land survey <country>' or 'national forest inventory <country>'.",
        "access_pattern": "Search engines + the agency's geoportal (often WMS/WFS). For EU member states, also check INSPIRE national geoportals.",
        "fast_use": "firecrawl_search or web_search for the agency name + 'WFS' or 'open data' to find the access endpoint.",
        "caveats": "Quality, coverage, license terms vary widely. Read the licence before redistribution.",
        "tags": ("national", "inventory", "country", "land-survey", "pointer", "search"),
    },
)


def _score_entry(entry: dict, query_tokens: set[str]) -> int:
    """Score an entry by how many query tokens overlap with its tags
    + name + id.  Tags weighted higher (canonical labels)."""
    score = 0
    for tag in entry["tags"]:
        if tag.lower() in query_tokens:
            score += 3
    name_words = set(re.findall(r"[\w-]+", entry["name"].lower()))
    score += len(name_words & query_tokens)
    if entry["id"].lower() in query_tokens:
        score += 5
    return score


def _format_entry(entry: dict) -> str:
    """Render one entry as a markdown-ish block for LLM consumption."""
    return (
        f"### {entry['name']}\n"
        f"**id:** `{entry['id']}`\n\n"
        f"{entry['description']}\n\n"
        f"**Access:** {entry['access_pattern']}\n\n"
        f"**Fast use:** {entry['fast_use']}\n\n"
        f"**Caveats:** {entry['caveats']}\n"
    )


def search_datasets(query: str, max_results: int = 5) -> str:
    """Search the curated dataset index for entries matching *query*.

    Returns a markdown-formatted block with the top *max_results*
    matches.  When no entry matches above zero score, returns a
    pointer to the broader search tools (web_search / firecrawl).
    """
    if not query or not query.strip():
        return "dataset_search: empty query — pass a description like 'forest age data for Estonia'."
    query_tokens = set(re.findall(r"[\w-]+", query.lower()))
    scored = [(_score_entry(e, query_tokens), e) for e in _DATASETS]
    scored = [(s, e) for s, e in scored if s > 0]
    scored.sort(key=lambda pair: -pair[0])
    if not scored:
        return (
            "dataset_search: no curated matches.  Try broader terms, OR "
            "use `firecrawl_search` / `web_search` to find datasets the "
            "curated index doesn't cover.  Common omissions: country-"
            "specific inventories (use the country-equivalent-pointer), "
            "research-grade datasets only on academic repositories, "
            "commercial sources."
        )
    out_lines = [
        f"# dataset_search: top {min(max_results, len(scored))} matches for '{query[:80]}'",
        "",
    ]
    for _, entry in scored[:max_results]:
        out_lines.append(_format_entry(entry))
        out_lines.append("---")
    out_lines.append(
        "\n_Use the access pattern + fast-use guidance above.  When in "
        "doubt about the right metric (e.g. 'forest age' is ambiguous), "
        "check `gee_forest_metric_semantics.md` skill._"
    )
    return "\n".join(out_lines)


# ── Public factory ──────────────────────────────────────────────────

def create_dataset_search_tools(agent_id: str = "researcher") -> list:
    """Factory returning the dataset_search BaseTool.

    Returns ``[]`` when crewai isn't importable (matches the other
    factories' graceful-degradation behaviour).
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    class _DatasetSearchInput(BaseModel):
        query: str = Field(
            description=(
                "Natural-language description of the data you need.  "
                "Examples: 'deforestation per year for Estonia', "
                "'forest age data', 'high-resolution land cover', "
                "'population density', 'historical satellite imagery "
                "since 1990'.  Returns the top matching curated "
                "datasets with access patterns + caveats."
            ),
        )
        max_results: int = Field(
            default=5,
            description="Maximum number of dataset entries to return (default 5).",
        )

    class DatasetSearchTool(BaseTool):
        name: str = "dataset_search"
        description: str = (
            "Search a curated index of geospatial / forest / "
            "remote-sensing / climate / population / boundary "
            "datasets.  USE THIS in the design phase when the task "
            "needs DATA you don't already know about — instead of "
            "defaulting to Hansen-only solutions for ambiguous "
            "'forest' queries, search first to find the RIGHT dataset.\n\n"
            "The index covers: Hansen GFC, Sentinel-1/2, Landsat, "
            "MODIS, ESA WorldCover, Dynamic World, Besnard 2021 forest "
            "age, GEDI biomass, GFW alerts, ERA5 climate, WorldPop, "
            "HydroBASINS, FAO GAUL admin boundaries, Estonian RMK + "
            "Maa-amet, plus a pointer to country-equivalent national "
            "data sources.\n\n"
            "Returns markdown-formatted entries with: name, access "
            "pattern (GEE asset id, WFS URL, etc.), fast-use guidance, "
            "and caveats that matter (e.g. 'GEDI doesn't cover Estonia "
            "north of 52°N')."
        )
        args_schema: Type[BaseModel] = _DatasetSearchInput

        def _run(self, query: str, max_results: int = 5) -> str:
            try:
                return search_datasets(query, max_results=max_results)
            except Exception as exc:
                logger.exception("dataset_search failed")
                return f"dataset_search: error {type(exc).__name__}: {exc}"

    return [DatasetSearchTool()]
