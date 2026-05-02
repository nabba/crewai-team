"""
geodata_tool.py — Multi-provider geospatial data fetcher.

Wraps three open-data geoportals behind a single CrewAI tool surface so
the coding crew can answer "what does the satellite record say about X
in country Y?" without picking a provider per call:

  1. Google Earth Engine (GEE)        — pixel-level analysis on 1000+
     datasets (Hansen forest change, Sentinel-2 / Landsat composites,
     MODIS, GEDI). Reuses gee_tool._ensure_initialised().
  2. Global Forest Watch (GFW)        — curated forest-change stats
     for any AOI (tree-cover loss, integrated alerts) via the public
     data-api at data-api.globalforestwatch.org.
  3. Copernicus Data Space (CDSE)     — STAC catalogue of every
     Sentinel acquisition; open browsing, OAuth required only for
     bulk download.

Two public tools (registered via ``create_geodata_tools``):

  * geodata_discover — live catalog listings from GFW + Copernicus
    STAC. GEE dataset *discovery* is delegated to the
    ``dataset_search`` tool (curated, qualitative index); we don't
    duplicate that here.
  * geodata_fetch    — for an AOI (country name / ISO-3 / bbox /
    GeoJSON) plus a date range, pull summary statistics from all
    enabled providers IN PARALLEL. Per-provider failures are isolated
    so one slow API can't poison the whole call.

Design-phase vs runtime
-----------------------
``dataset_search`` (see app/tools/dataset_search_tool.py) is the
design-phase entry point: "given my question, what dataset should I
use?" It returns curated metadata, not numbers. ``geodata_fetch`` is
the runtime entry point: "given an AOI + date range, get me the
numbers." Tools cross-reference each other in their descriptions so
the LLM uses them in the right order.

Configuration
-------------
Provider keys are optional — missing keys mean that provider is
skipped, never an error.

* ``GOOGLE_APPLICATION_CREDENTIALS`` / ``GEE_PROJECT`` — see gee_tool.py
* ``GFW_API_KEY``                 — Global Forest Watch (free at
  globalforestwatch.org/help/developers; without it the data-api is
  read-only and rate-limited).
* ``CDSE_USERNAME`` / ``CDSE_PASSWORD`` — Copernicus, only used to
  mint download URLs; STAC search itself is open.

Country resolution
------------------
Accepts: ``country=`` (English name OR ISO-3 code), ``bbox=`` (xmin,
ymin, xmax, ymax in WGS-84), or ``geojson=`` (Polygon/MultiPolygon
geometry dict). Country names hit a small built-in ISO-3→bbox table
first; unknown names fall back to OSM Nominatim with the standard
User-Agent.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── HTTP session (reuse pattern from web_fetch.py) ──────────────────
_session = requests.Session()
_session.headers["User-Agent"] = "BotArmy-Geodata/1.0 (+https://github.com/anthropics/claude-code)"

# ── Provider endpoints ──────────────────────────────────────────────
_GFW_API = "https://data-api.globalforestwatch.org"
_CDSE_STAC = "https://catalogue.dataspace.copernicus.eu/stac"
_CDSE_TOKEN = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
_NOMINATIM = "https://nominatim.openstreetmap.org/search"

# ── Built-in country → bbox table (xmin,ymin,xmax,ymax, WGS-84) ─────
# Subset of frequent AOIs; unknown countries fall back to Nominatim.
_COUNTRY_BBOX: dict[str, tuple[str, list[float]]] = {
    # ISO-3:    (display_name,            [W,    S,    E,    N])
    "FIN": ("Finland",                    [20.55, 59.81, 31.59, 70.09]),
    "EST": ("Estonia",                    [21.84, 57.52, 28.21, 59.69]),
    "SWE": ("Sweden",                     [11.11, 55.34, 24.16, 69.06]),
    "NOR": ("Norway",                     [4.65,  57.98, 31.10, 71.18]),
    "DNK": ("Denmark",                    [8.07,  54.56, 15.16, 57.75]),
    "LVA": ("Latvia",                     [20.97, 55.67, 28.24, 58.09]),
    "LTU": ("Lithuania",                  [21.06, 53.90, 26.84, 56.45]),
    "POL": ("Poland",                     [14.12, 49.00, 24.15, 54.84]),
    "DEU": ("Germany",                    [5.87,  47.27, 15.04, 55.06]),
    "FRA": ("France",                     [-5.14, 41.33, 9.56,  51.09]),
    "ESP": ("Spain",                      [-9.39, 35.95, 4.33,  43.79]),
    "ITA": ("Italy",                      [6.62,  35.49, 18.52, 47.10]),
    "GBR": ("United Kingdom",             [-8.65, 49.86, 1.76,  60.86]),
    "NLD": ("Netherlands",                [3.36,  50.75, 7.23,  53.55]),
    "USA": ("United States",              [-125.0,24.40,-66.93, 49.38]),
    "CAN": ("Canada",                     [-141.0,41.68,-52.62, 83.11]),
    "BRA": ("Brazil",                     [-73.99,-33.75,-34.79, 5.27]),
    "RUS": ("Russia",                     [19.64, 41.19, 180.0, 81.86]),
    "CHN": ("China",                      [73.55, 18.16,134.77, 53.56]),
    "IND": ("India",                      [68.11, 6.75, 97.40,  35.51]),
    "AUS": ("Australia",                  [112.92,-43.74,153.64,-10.05]),
    "JPN": ("Japan",                      [122.93,24.04,145.82, 45.52]),
    "IDN": ("Indonesia",                  [95.01, -10.92,141.01,5.91]),
    "ZAF": ("South Africa",               [16.45, -34.83,32.89, -22.13]),
    "KEN": ("Kenya",                      [33.91, -4.68, 41.91, 5.51]),
    "COD": ("Democratic Republic of the Congo",[12.18,-13.46,31.30,5.39]),
    "MEX": ("Mexico",                     [-118.41,14.53,-86.7,32.72]),
    "ARG": ("Argentina",                  [-73.58,-55.06,-53.59,-21.78]),
    "CHL": ("Chile",                      [-75.65,-55.92,-66.42,-17.50]),
    "PER": ("Peru",                       [-81.41,-18.35,-68.65,-0.04]),
    "COL": ("Colombia",                   [-78.99,-4.23,-66.87,12.46]),
    "VNM": ("Vietnam",                    [102.14,8.55, 109.47,23.39]),
    "THA": ("Thailand",                   [97.34, 5.61, 105.64,20.46]),
    "TUR": ("Turkey",                     [25.66, 35.82, 44.81, 42.11]),
    "UKR": ("Ukraine",                    [22.14, 44.39, 40.23, 52.38]),
    "EGY": ("Egypt",                      [24.70, 21.99, 36.87, 31.66]),
    "NGA": ("Nigeria",                    [2.69,  4.27, 14.58,  13.89]),
    "ETH": ("Ethiopia",                   [32.99, 3.40, 47.99,  14.85]),
    "MAR": ("Morocco",                    [-13.17,21.42,-1.12, 35.93]),
    "IRN": ("Iran",                       [44.04, 25.08, 63.32, 39.78]),
    "PAK": ("Pakistan",                   [60.87, 23.69, 77.84, 37.10]),
    "BGD": ("Bangladesh",                 [88.08, 20.74, 92.67, 26.45]),
    "MMR": ("Myanmar",                    [92.19, 9.78, 101.17, 28.55]),
    "PHL": ("Philippines",                [117.17,5.58, 126.60,18.51]),
    "MYS": ("Malaysia",                   [100.09,0.85, 119.27, 7.36]),
    "ISL": ("Iceland",                    [-24.55,63.39,-13.50,66.57]),
    "IRL": ("Ireland",                    [-10.48,51.42,-5.99, 55.39]),
    "PRT": ("Portugal",                   [-9.53, 36.84,-6.19, 42.15]),
    "GRC": ("Greece",                     [19.37, 34.80, 28.24, 41.75]),
    "ROU": ("Romania",                    [20.22, 43.69, 29.63, 48.27]),
    "CZE": ("Czechia",                    [12.09, 48.55, 18.86, 51.06]),
    "AUT": ("Austria",                    [9.53,  46.37, 17.16, 49.02]),
    "CHE": ("Switzerland",                [5.96,  45.82, 10.49, 47.81]),
    "BEL": ("Belgium",                    [2.55,  49.50, 6.41,  51.51]),
    "HUN": ("Hungary",                    [16.11, 45.74, 22.90, 48.59]),
    "BLR": ("Belarus",                    [23.18, 51.26, 32.78, 56.17]),
    "NZL": ("New Zealand",                [166.43,-47.29,178.55,-34.39]),
}
# Common-name → ISO-3 alias for ergonomic lookup. Lowercased keys.
_NAME_TO_ISO3: dict[str, str] = {
    name.lower(): iso for iso, (name, _) in _COUNTRY_BBOX.items()
}
# A few hand-rolled aliases callers commonly pass.
_NAME_TO_ISO3.update({
    "us": "USA", "u.s.": "USA", "u.s.a.": "USA", "america": "USA",
    "uk": "GBR", "u.k.": "GBR", "britain": "GBR",
    "russia": "RUS", "drc": "COD", "congo (dr)": "COD", "vietnam": "VNM",
})

_NOMINATIM_CACHE: dict[str, list[float]] = {}
_NOMINATIM_LOCK = threading.Lock()


# ── AOI resolution ──────────────────────────────────────────────────

def _resolve_country(name_or_iso: str) -> dict[str, Any] | None:
    """Resolve a country name or ISO-3 code to a bbox + display name.

    Returns ``{"iso3": "...", "name": "...", "bbox": [W,S,E,N]}`` or
    ``None`` when the lookup fails. Built-in table is checked first;
    unknown values fall back to OSM Nominatim (cached per process).
    """
    key = name_or_iso.strip()
    if not key:
        return None
    upper = key.upper()
    if upper in _COUNTRY_BBOX:
        name, bbox = _COUNTRY_BBOX[upper]
        return {"iso3": upper, "name": name, "bbox": list(bbox)}
    iso = _NAME_TO_ISO3.get(key.lower())
    if iso:
        name, bbox = _COUNTRY_BBOX[iso]
        return {"iso3": iso, "name": name, "bbox": list(bbox)}

    # Nominatim fallback — one network call, cached per process.
    cache_key = key.lower()
    with _NOMINATIM_LOCK:
        cached = _NOMINATIM_CACHE.get(cache_key)
    if cached:
        return {"iso3": "", "name": key, "bbox": list(cached)}

    try:
        r = _session.get(
            _NOMINATIM,
            params={"country": key, "format": "json", "limit": 1},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        # Nominatim returns boundingbox = [S, N, W, E] as strings.
        s, n, w, e = (float(v) for v in results[0]["boundingbox"])
        bbox = [w, s, e, n]
        with _NOMINATIM_LOCK:
            _NOMINATIM_CACHE[cache_key] = bbox
        return {
            "iso3": "",
            "name": results[0].get("display_name", key),
            "bbox": bbox,
        }
    except Exception as exc:
        logger.warning(f"geodata: nominatim lookup for {key!r} failed: {exc}")
        return None


def _resolve_aoi(
    country: str | None,
    bbox: list[float] | None,
    geojson: dict | None,
) -> dict[str, Any]:
    """Normalise AOI input into ``{name, bbox, geometry}``.

    ``geometry`` is a GeoJSON dict — if the caller supplies only a
    bbox, we synthesise a Polygon so the GFW geostore endpoint (which
    only accepts polygons) can be used. Raises ``ValueError`` when no
    valid AOI is provided.
    """
    if geojson:
        coords = _bbox_from_geojson(geojson)
        return {
            "name": "user-geojson",
            "bbox": coords,
            "geometry": geojson,
        }
    if bbox:
        if len(bbox) != 4:
            raise ValueError("bbox must be [xmin, ymin, xmax, ymax]")
        w, s, e, n = bbox
        return {
            "name": "user-bbox",
            "bbox": [float(w), float(s), float(e), float(n)],
            "geometry": _polygon_from_bbox(w, s, e, n),
        }
    if country:
        resolved = _resolve_country(country)
        if not resolved:
            raise ValueError(f"could not resolve country: {country!r}")
        w, s, e, n = resolved["bbox"]
        return {
            "name": resolved["name"],
            "iso3": resolved.get("iso3", ""),
            "bbox": [w, s, e, n],
            "geometry": _polygon_from_bbox(w, s, e, n),
        }
    raise ValueError("must supply one of: country, bbox, geojson")


def _polygon_from_bbox(w: float, s: float, e: float, n: float) -> dict:
    """Return a GeoJSON Polygon for the given bounding box."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [w, s], [e, s], [e, n], [w, n], [w, s],
        ]],
    }


def _bbox_from_geojson(geom: dict) -> list[float]:
    """Compute the bounding box of a GeoJSON Polygon/MultiPolygon."""
    def walk(coords: Any) -> list[tuple[float, float]]:
        if (isinstance(coords, list) and coords
                and isinstance(coords[0], (int, float))):
            return [(float(coords[0]), float(coords[1]))]
        out: list[tuple[float, float]] = []
        for c in coords:
            out.extend(walk(c))
        return out

    pts = walk(geom.get("coordinates", []))
    if not pts:
        raise ValueError("empty geometry")
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


# ── Provider: Global Forest Watch ───────────────────────────────────

def _gfw_headers() -> dict[str, str]:
    key = os.environ.get("GFW_API_KEY", "").strip()
    return {"x-api-key": key} if key else {}


def gfw_list_datasets(timeout: int = 20) -> dict[str, Any]:
    """List GFW data-api datasets (open endpoint, ~50-100 entries)."""
    try:
        r = _session.get(
            f"{_GFW_API}/datasets",
            headers=_gfw_headers(),
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        return {
            "ok": True,
            "count": len(data),
            "datasets": [
                {"id": d.get("dataset"), "metadata": d.get("metadata", {})}
                for d in data[:200]
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def gfw_query_aoi(
    aoi: dict[str, Any],
    date_from: str | None = None,
    date_to: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Register the AOI as a GFW geostore and run two canned queries.

    Returns tree-cover-loss area + integrated deforestation alerts for
    the AOI. Both endpoints are public; rate-limited without an API
    key. Date strings are ISO ``YYYY-MM-DD``.
    """
    headers = _gfw_headers()
    headers["Content-Type"] = "application/json"
    try:
        # 1. Register geostore (ephemeral, ~24h TTL).
        gs = _session.post(
            f"{_GFW_API}/geostore",
            headers=headers,
            json={"geometry": aoi["geometry"]},
            timeout=timeout,
        )
        gs.raise_for_status()
        geostore_id = (
            gs.json().get("data", {}).get("gfw_geostore_id")
            or gs.json().get("data", {}).get("id")
        )
        if not geostore_id:
            return {"ok": False, "error": "GFW geostore registration returned no id"}

        out: dict[str, Any] = {"ok": True, "geostore_id": geostore_id, "queries": {}}

        # 2. Tree-cover loss (Hansen, area-summed by year).
        loss_sql = (
            "SELECT SUM(area__ha) AS area_ha, umd_tree_cover_loss__year AS year "
            "FROM data WHERE umd_tree_cover_density_2000__threshold = 30 "
            "GROUP BY umd_tree_cover_loss__year ORDER BY umd_tree_cover_loss__year"
        )
        try:
            r1 = _session.get(
                f"{_GFW_API}/dataset/umd_tree_cover_loss/latest/query",
                headers=_gfw_headers(),
                params={"sql": loss_sql, "geostore_id": geostore_id,
                        "geostore_origin": "gfw"},
                timeout=timeout,
            )
            r1.raise_for_status()
            out["queries"]["tree_cover_loss_by_year"] = r1.json().get("data", [])
        except Exception as exc:
            out["queries"]["tree_cover_loss_by_year"] = {
                "error": f"{type(exc).__name__}: {exc}"}

        # 3. Integrated deforestation alerts (most recent N days).
        if date_from or date_to:
            alert_sql = (
                "SELECT COUNT(*) AS alert_count "
                "FROM data WHERE gfw_integrated_alerts__date "
                f">= '{date_from or '2024-01-01'}' "
                f"AND gfw_integrated_alerts__date <= '{date_to or '2099-12-31'}'"
            )
            try:
                r2 = _session.get(
                    f"{_GFW_API}/dataset/gfw_integrated_alerts/latest/query",
                    headers=_gfw_headers(),
                    params={"sql": alert_sql, "geostore_id": geostore_id,
                            "geostore_origin": "gfw"},
                    timeout=timeout,
                )
                r2.raise_for_status()
                out["queries"]["integrated_alerts_count"] = r2.json().get("data", [])
            except Exception as exc:
                out["queries"]["integrated_alerts_count"] = {
                    "error": f"{type(exc).__name__}: {exc}"}

        return out
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ── Provider: Copernicus Data Space (STAC) ──────────────────────────

def cdse_list_collections(timeout: int = 20) -> dict[str, Any]:
    """List Copernicus STAC collections (Sentinel-1/2/3/5P, etc.)."""
    try:
        r = _session.get(f"{_CDSE_STAC}/collections", timeout=timeout)
        r.raise_for_status()
        cols = r.json().get("collections", [])
        return {
            "ok": True,
            "count": len(cols),
            "collections": [
                {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "description": (c.get("description") or "")[:240],
                }
                for c in cols
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def cdse_search_aoi(
    aoi: dict[str, Any],
    date_from: str | None = None,
    date_to: str | None = None,
    collections: list[str] | None = None,
    limit: int = 20,
    timeout: int = 30,
) -> dict[str, Any]:
    """STAC search for scenes intersecting the AOI within the window.

    ``collections`` defaults to Sentinel-2 L2A (the most-used optical
    layer). Output is a compact list of scene IDs + cloud cover so the
    LLM can decide which to hand off to GEE for pixel-level analysis.
    """
    cols = collections or ["SENTINEL-2"]
    body: dict[str, Any] = {
        "bbox": aoi["bbox"],
        "limit": limit,
        "collections": cols,
    }
    if date_from or date_to:
        body["datetime"] = f"{date_from or '..'}/{date_to or '..'}"

    try:
        r = _session.post(
            f"{_CDSE_STAC}/search",
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        return {
            "ok": True,
            "matched": data.get("context", {}).get("matched", len(feats)),
            "returned": len(feats),
            "scenes": [
                {
                    "id": f.get("id"),
                    "collection": f.get("collection"),
                    "datetime": (f.get("properties") or {}).get("datetime"),
                    "cloud_cover": (f.get("properties") or {}).get("eo:cloud_cover"),
                    "self": next(
                        (lk["href"] for lk in (f.get("links") or [])
                         if lk.get("rel") == "self"),
                        None,
                    ),
                }
                for f in feats
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# ── Provider: Google Earth Engine ───────────────────────────────────
# GEE dataset *discovery* lives in app/tools/dataset_search_tool.py —
# it's a curated index with access patterns + caveats, much richer
# than a flat id list. We only do GEE *fetching* here.

def gee_query_aoi(
    aoi: dict[str, Any],
    date_from: str | None = None,
    date_to: str | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run three canned EE reductions for the AOI.

    Returns: total tree-cover-loss area (Hansen), majority Dynamic
    World land-cover class, and Sentinel-2 NDVI mean. Each reduction
    is server-side; one .getInfo() per metric.
    """
    try:
        from app.tools.gee_tool import _ensure_initialised
    except Exception as exc:
        return {"ok": False, "error": f"gee_tool unavailable: {exc}"}
    ok, err = _ensure_initialised()
    if not ok:
        return {"ok": False, "error": err or "ee.Initialize failed"}

    try:
        import ee
    except ImportError:
        return {"ok": False, "error": "earthengine-api not installed"}

    w, s, e, n = aoi["bbox"]
    region = ee.Geometry.Rectangle([w, s, e, n])
    out: dict[str, Any] = {"ok": True, "metrics": {}}

    # Hansen tree-cover loss (area, ha).
    try:
        hansen = ee.Image("UMD/hansen/global_forest_change_2023_v1_11")
        loss_area = (
            hansen.select("loss")
            .multiply(ee.Image.pixelArea())
            .divide(10_000)
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=30,
                maxPixels=1e10,
                bestEffort=True,
            )
        )
        out["metrics"]["hansen_tree_cover_loss_ha_2001_2023"] = loss_area.getInfo()
    except Exception as exc:
        out["metrics"]["hansen_tree_cover_loss_ha_2001_2023"] = {
            "error": f"{type(exc).__name__}: {exc}"}

    # Sentinel-2 NDVI mean over the date range.
    try:
        df = date_from or "2024-01-01"
        dt = date_to or "2024-12-31"
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(df, dt)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        )
        ndvi = s2.map(
            lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ).mean()
        ndvi_mean = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=100,
            maxPixels=1e10,
            bestEffort=True,
        )
        out["metrics"][f"sentinel2_ndvi_mean_{df}_to_{dt}"] = ndvi_mean.getInfo()
    except Exception as exc:
        out["metrics"]["sentinel2_ndvi_mean"] = {
            "error": f"{type(exc).__name__}: {exc}"}

    # Dynamic World land-cover histogram.
    try:
        df = date_from or "2024-01-01"
        dt = date_to or "2024-12-31"
        dw = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(region)
            .filterDate(df, dt)
            .select("label")
            .mode()
        )
        hist = dw.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=100,
            maxPixels=1e10,
            bestEffort=True,
        )
        out["metrics"]["dynamic_world_class_histogram"] = hist.getInfo()
    except Exception as exc:
        out["metrics"]["dynamic_world_class_histogram"] = {
            "error": f"{type(exc).__name__}: {exc}"}

    return out


# ── Parallel orchestrator ───────────────────────────────────────────

_FETCH_PROVIDERS = ("gee", "gfw", "cdse")
_DISCOVER_PROVIDERS = ("gfw", "cdse")  # GEE handled by dataset_search


def fetch_all(
    country: str | None = None,
    bbox: list[float] | None = None,
    geojson: dict | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    providers: list[str] | None = None,
    per_provider_timeout: int = 90,
) -> dict[str, Any]:
    """Fan out to every enabled provider in parallel.

    Per-provider failures are isolated: one slow / broken API does not
    block the others. The orchestrator returns whatever finished
    within ``per_provider_timeout`` seconds.
    """
    aoi = _resolve_aoi(country, bbox, geojson)
    chosen = [p for p in (providers or _FETCH_PROVIDERS) if p in _FETCH_PROVIDERS]
    if not chosen:
        chosen = list(_FETCH_PROVIDERS)

    jobs = {
        "gee": lambda: gee_query_aoi(aoi, date_from, date_to),
        "gfw": lambda: gfw_query_aoi(aoi, date_from, date_to),
        "cdse": lambda: cdse_search_aoi(aoi, date_from, date_to),
    }

    started = time.time()
    results: dict[str, Any] = {"aoi": aoi, "providers": {}}
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        futures = {pool.submit(jobs[p]): p for p in chosen}
        for fut in as_completed(futures, timeout=per_provider_timeout + 5):
            name = futures[fut]
            try:
                results["providers"][name] = fut.result(timeout=per_provider_timeout)
            except FutureTimeout:
                results["providers"][name] = {
                    "ok": False,
                    "error": f"timed out after {per_provider_timeout}s",
                }
            except Exception as exc:
                results["providers"][name] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    results["elapsed_s"] = round(time.time() - started, 2)
    return results


def discover_all(providers: list[str] | None = None) -> dict[str, Any]:
    """Live catalog listings from GFW data-api + Copernicus STAC.

    GEE dataset discovery is intentionally NOT done here — see the
    ``dataset_search`` tool (app/tools/dataset_search_tool.py) for a
    curated index with access patterns + caveats. If the caller
    explicitly asks for ``providers=['gee']`` we return a pointer
    rather than silently dropping the request.
    """
    chosen = list(providers) if providers else list(_DISCOVER_PROVIDERS)
    out: dict[str, Any] = {}
    if "gee" in chosen:
        out["gee"] = {
            "ok": False,
            "info": (
                "GEE catalog discovery moved to the `dataset_search` "
                "tool — it returns curated entries with access "
                "patterns, fast-use guidance and caveats. Call "
                "dataset_search(query='...') instead."
            ),
        }
    chosen = [p for p in chosen if p in _DISCOVER_PROVIDERS]
    if not chosen:
        return out
    jobs = {
        "gfw": gfw_list_datasets,
        "cdse": cdse_list_collections,
    }
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        futs = {pool.submit(jobs[p]): p for p in chosen}
        for fut in as_completed(futs, timeout=60):
            name = futs[fut]
            try:
                out[name] = fut.result(timeout=30)
            except Exception as exc:
                out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


# ── Public CrewAI factory ───────────────────────────────────────────

def create_geodata_tools(agent_id: str = "coder") -> list:
    """Build the CrewAI tool list for multi-provider geodata fetching.

    Returns ``[]`` only if pydantic / crewai aren't importable. All
    three providers are always exposed — individual providers
    self-report as "not configured" at call time when their key is
    missing, which is the correct UX for the LLM (it can pick another
    provider on the next call).
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    class _DiscoverInput(BaseModel):
        providers: list[str] | None = Field(
            default=None,
            description=(
                "Optional list of providers to query: any of 'gee', "
                "'gfw', 'cdse'. Defaults to all three. Use this to "
                "skip a known-broken provider without changing code."
            ),
        )

    class _FetchInput(BaseModel):
        country: str | None = Field(
            default=None,
            description=(
                "Country name (English) or ISO-3 code (e.g. 'Finland', "
                "'FIN', 'Brazil', 'BRA'). One of country/bbox/geojson "
                "is required. The tool resolves a built-in table "
                "first; unknown names hit OSM Nominatim once."
            ),
        )
        bbox: list[float] | None = Field(
            default=None,
            description=(
                "Bounding box [xmin, ymin, xmax, ymax] in WGS-84 "
                "decimal degrees. Use for sub-national AOIs."
            ),
        )
        geojson: dict | None = Field(
            default=None,
            description=(
                "GeoJSON Polygon or MultiPolygon geometry dict. Use "
                "for irregular AOIs (e.g. a national park boundary). "
                "Must be the geometry, not a Feature."
            ),
        )
        date_from: str | None = Field(
            default=None,
            description="ISO date YYYY-MM-DD (inclusive). Defaults to 2024-01-01.",
        )
        date_to: str | None = Field(
            default=None,
            description="ISO date YYYY-MM-DD (inclusive). Defaults to 2024-12-31.",
        )
        providers: list[str] | None = Field(
            default=None,
            description=(
                "Subset of ['gee','gfw','cdse']. Defaults to all three "
                "running in parallel."
            ),
        )
        per_provider_timeout: int = Field(
            default=90,
            description="Wall-clock budget per provider in seconds.",
        )

    class GeodataDiscoverTool(BaseTool):
        name: str = "geodata_discover"
        description: str = (
            "Live catalog listings from Global Forest Watch data-api "
            "and Copernicus Data Space STAC, fetched in parallel. Use "
            "this when you need to see what GFW or Copernicus expose "
            "RIGHT NOW (e.g. a recently-added dataset). For Google "
            "Earth Engine dataset selection use the `dataset_search` "
            "tool instead — it's a curated index with access patterns "
            "+ caveats, higher-signal for design-phase decisions. "
            "After picking a dataset, use `geodata_fetch` to pull "
            "stats for an AOI."
        )
        args_schema: Type[BaseModel] = _DiscoverInput

        def _run(self, providers: list[str] | None = None) -> str:
            out = discover_all(providers)
            return json.dumps(out, indent=2, default=str)[:8000]

    class GeodataFetchTool(BaseTool):
        name: str = "geodata_fetch"
        description: str = (
            "Fetch summary geospatial statistics for an AOI from "
            "multiple providers IN PARALLEL: GEE (Hansen forest loss, "
            "Sentinel-2 NDVI, Dynamic World land cover), GFW "
            "(tree-cover loss timeline, integrated alert counts), "
            "Copernicus STAC (Sentinel scene catalogue).\n\n"
            "WORKFLOW: use `dataset_search` FIRST to pick the right "
            "dataset for the question (forest age vs. deforestation, "
            "land cover at what resolution, etc.), then call this "
            "tool to actually pull the numbers. The two are "
            "complementary — design phase vs. runtime phase.\n\n"
            "AOI input is FLEXIBLE: pass a country name OR ISO-3 code "
            "(e.g. 'BRA'), a bbox [xmin,ymin,xmax,ymax], or a GeoJSON "
            "geometry. Date range is optional but recommended. "
            "Per-provider failures are isolated — you always get a "
            "result for the providers that worked. Pick a single "
            "provider via the providers= argument when you already "
            "know which one you need."
        )
        args_schema: Type[BaseModel] = _FetchInput

        def _run(
            self,
            country: str | None = None,
            bbox: list[float] | None = None,
            geojson: dict | None = None,
            date_from: str | None = None,
            date_to: str | None = None,
            providers: list[str] | None = None,
            per_provider_timeout: int = 90,
        ) -> str:
            try:
                out = fetch_all(
                    country=country, bbox=bbox, geojson=geojson,
                    date_from=date_from, date_to=date_to,
                    providers=providers,
                    per_provider_timeout=per_provider_timeout,
                )
            except ValueError as exc:
                return f"geodata_fetch: bad AOI input — {exc}"
            return json.dumps(out, indent=2, default=str)[:12000]

    return [GeodataDiscoverTool(), GeodataFetchTool()]
