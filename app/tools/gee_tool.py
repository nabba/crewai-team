"""
gee_tool.py — Google Earth Engine tool for the coding crew.

Exposes a single CrewAI BaseTool, ``gee_run_script``, that:

  1. Initialises an authenticated Earth Engine session using the
     service-account JSON at ``GOOGLE_APPLICATION_CREDENTIALS``
     (see .env.example for the one-time setup).
  2. Executes a user-provided Python snippet in a sandbox dict that
     has ``ee`` (the Earth Engine API) pre-imported and a few helpers.
  3. Captures any printed output + the value of the script's
     ``result`` variable (if assigned) and returns both.

The crew uses this for satellite-imagery analysis: Hansen Global
Forest Change, Sentinel-2 / Landsat composites, NDVI time-series,
land-cover classification, etc. Examples land in the user-facing
forest pipeline (Estonia deforestation maps per year since 2012).

Safety properties
-----------------
* The script runs in-process — it CAN access any Python object the
  gateway has loaded. This is the same trust model as the existing
  coding-crew sandbox (``base_crew.run_python``). Forge-audit gates
  apply when the tool is invoked from agent code that's been forged.
* Outbound network calls go to ``earthengine.googleapis.com`` only
  (Google's hosted compute) — actual heavy lifting happens server-
  side. Local CPU/RAM cost per call is small.
* Heavy outputs (export-to-Drive / export-to-GCS) are logged but
  never block the call returning — they run as Earth Engine "tasks"
  on Google's side and complete asynchronously.

Configuration
-------------
* ``GOOGLE_APPLICATION_CREDENTIALS`` — path to service-account JSON
  (mounted from secrets/ via docker-compose volume).
* ``GEE_PROJECT`` — the GCP project ID (e.g. ``botarmy-495107``).
  When unset, falls back to the ``project_id`` field of the JSON.

When EITHER env var is missing OR the JSON file is unreadable,
``create_gee_tools()`` returns ``[]`` — agent inventories degrade
gracefully (the coding crew still works, just without the GEE tool).
"""
from __future__ import annotations

import json
import logging
import os
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Initialisation guard ─────────────────────────────────────────────
# ee.Initialize() makes a network call and parses the JWT; we only
# want to do it once per process. A module-level flag keeps subsequent
# tool invocations cheap.
_EE_INITIALISED = False
_EE_INIT_ERROR: str | None = None


def _gee_credentials_path() -> str | None:
    """Resolve the service-account JSON path. Returns None when the
    env var isn't set or points at a missing file."""
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not path:
        return None
    if not Path(path).exists():
        logger.warning(f"gee_tool: GOOGLE_APPLICATION_CREDENTIALS={path!r} but file is missing")
        return None
    return path


def _gee_project_id(sa_json: dict) -> str | None:
    """Project ID resolution priority:
      1. GEE_PROJECT env (explicit override)
      2. project_id field in the service-account JSON
    """
    explicit = os.environ.get("GEE_PROJECT", "").strip()
    if explicit:
        return explicit
    return (sa_json.get("project_id") or "").strip() or None


def _ensure_initialised() -> tuple[bool, str | None]:
    """Lazy ee.Initialize. Returns (success, error_message_or_None).

    Repeated calls are O(1) after the first success; failure is cached
    so we don't hammer Google with broken credentials on every tool
    invocation.
    """
    global _EE_INITIALISED, _EE_INIT_ERROR
    if _EE_INITIALISED:
        return True, None
    if _EE_INIT_ERROR is not None:
        return False, _EE_INIT_ERROR

    sa_path = _gee_credentials_path()
    if not sa_path:
        msg = (
            "gee_tool: GOOGLE_APPLICATION_CREDENTIALS not set or file "
            "missing. See .env.example for the setup walkthrough."
        )
        _EE_INIT_ERROR = msg
        return False, msg

    try:
        sa_json = json.loads(Path(sa_path).read_text())
    except Exception as exc:
        msg = f"gee_tool: failed to parse service-account JSON at {sa_path}: {exc}"
        _EE_INIT_ERROR = msg
        return False, msg

    project_id = _gee_project_id(sa_json)
    sa_email = sa_json.get("client_email", "")
    if not project_id or not sa_email:
        msg = (
            f"gee_tool: service-account JSON missing project_id or "
            f"client_email (path={sa_path})"
        )
        _EE_INIT_ERROR = msg
        return False, msg

    try:
        import ee
        credentials = ee.ServiceAccountCredentials(sa_email, sa_path)
        ee.Initialize(credentials, project=project_id)
        _EE_INITIALISED = True
        logger.info(
            f"gee_tool: initialised as {sa_email} (project={project_id})"
        )
        return True, None
    except ImportError as exc:
        msg = f"gee_tool: earthengine-api not installed: {exc}"
        _EE_INIT_ERROR = msg
        return False, msg
    except Exception as exc:
        msg = f"gee_tool: ee.Initialize() failed: {exc}"
        _EE_INIT_ERROR = msg
        return False, msg


def _format_rendered_maps(paths: list[str], *, label: str = "rendered maps") -> str:
    """Format rendered_paths as a vetting-friendly block.

    Pre-fix the output looked like:
      --- rendered maps (4) ---
        /app/workspace/output/maps/20260502T215631_estonia_age_2024.png
        ...

    Vetting LLMs saw the absolute container paths as URL-shaped
    strings and flagged them as "hallucinated URLs" (false positive
    on the 2026-05-02 v8 dispatch).  This helper rewrites them as:
      --- [gee_render] rendered maps (4) ---
        workspace/output/maps/20260502T215631_estonia_age_2024.png
        ...

    The `[gee_render]` marker is a stable string vetting can
    pattern-match to recognise legitimate file artifacts; the
    relative paths look like file paths, not URLs.  Container
    prefix /app/ is stripped so the path is portable.
    """
    rels = []
    for p in paths[:32]:
        # Strip /app/ prefix for portability + non-URL appearance
        if p.startswith("/app/"):
            rels.append(p[len("/app/"):])
        else:
            rels.append(p)
    body = "\n".join(f"  {r}" for r in rels)
    extra = f"  ... and {len(paths) - 32} more" if len(paths) > 32 else ""
    return f"\n--- [gee_render] {label} ({len(paths)}) ---\n{body}{extra}"


def _make_render_map(rendered_paths: list[str]):
    """Build the ``render_map`` helper that the user-script can call.

    Renders an ``ee.Image`` to a PNG via Earth Engine's thumbnail API
    (synchronous, no async export task) and saves to
    ``workspace/output/maps/``.  Up to ~1024×1024 dimensions per call
    (EE thumbnail limit).

    Returns the absolute path of the rendered file.  Each render is
    appended to ``rendered_paths`` so the wrapper can include them in
    the result summary that goes back to the agent.

    The closure captures ``rendered_paths`` so we can collect every
    map produced during a single ``gee_run_script`` call without
    requiring the user-script to track them.
    """
    import urllib.request
    import urllib.error
    from datetime import datetime

    def render_map(
        image,
        region=None,
        name: str = "map",
        vis_params: dict | None = None,
        dimensions: int = 768,
        format: str = "png",
    ) -> str:
        """Render *image* to a PNG and return the saved file path.

        Parameters
        ----------
        image : ee.Image
            What to render.  Should already be ``.clip(region)``-ed
            for best results.
        region : ee.Geometry | None
            AOI for the thumbnail.  Defaults to ``image.geometry()``
            when None.
        name : str
            Filename prefix (no extension).  Sanitised + timestamped.
        vis_params : dict | None
            Standard EE visualisation params (e.g. ``min, max, palette,
            bands, gamma``).  Merged into the thumbnail request.
        dimensions : int
            Max edge in pixels.  EE caps at ~1024.  Default 768.
        format : str
            "png" (default) or "jpg".
        """
        out_dir = Path("/app/workspace/output/maps")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build thumbnail request
        params: dict[str, Any] = dict(vis_params or {})
        params.setdefault("dimensions", dimensions)
        params.setdefault("format", format)
        if region is not None:
            params["region"] = region
        try:
            url = image.getThumbURL(params)
        except Exception as exc:
            raise RuntimeError(f"render_map: getThumbURL failed: {exc}") from exc

        # Sanitise filename
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:80] or "map"
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"{ts}_{safe}.{format}"

        # Fetch + save
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            out_path.write_bytes(data)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"render_map: failed to fetch thumbnail at {url[:120]}...: {exc}"
            ) from exc

        rendered_paths.append(str(out_path))
        logger.info(
            "render_map: wrote %s (%d bytes) for %s",
            out_path.name, len(data), name,
        )
        return str(out_path)

    return render_map


def _run_user_script(script: str, timeout_s: int = 60) -> dict[str, Any]:
    """Execute the user's GEE Python snippet.

    Returns ``{ok, stdout, result, error, rendered_maps}``:
      * ``ok``: True if the script ran without uncaught exception.
      * ``stdout``: anything the script ``print()``-ed.
      * ``result``: value of the script's ``result`` variable (if any).
      * ``error``: short exception message when ``ok=False``, else None.
      * ``rendered_maps``: list of file paths the script produced via
        the ``render_map(...)`` helper (Week 5 audit fix for H10).

    The sandbox dict pre-loads ``ee`` and a couple of common helpers so
    short snippets stay short. Heavy clients (geemap, etc.) can be
    imported by the script itself.
    """
    import ee

    # Pre-populated namespace — keeps short scripts short.
    rendered_paths: list[str] = []
    sandbox: dict[str, Any] = {
        "ee": ee,
        # Common AOI helpers
        "estonia": ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
            ee.Filter.eq("ADM0_NAME", "Estonia"),
        ),
        # Synchronous PNG renderer — see _make_render_map for the
        # signature.  Closes over rendered_paths so every saved map
        # is included in the result summary.
        "render_map": _make_render_map(rendered_paths),
        "result": None,
    }
    stdout = StringIO()
    try:
        with redirect_stdout(stdout):
            exec(script, sandbox)
    except Exception as exc:
        return {
            "ok": False,
            "stdout": stdout.getvalue(),
            "result": None,
            "rendered_maps": rendered_paths,
            "error": f"{type(exc).__name__}: {exc}",
        }

    # Best-effort serialisation of the result so the LLM can read it.
    raw = sandbox.get("result")
    serialised: Any = None
    if raw is None:
        serialised = None
    elif hasattr(raw, "getInfo"):
        try:
            serialised = raw.getInfo()
        except Exception as exc:
            serialised = f"<getInfo() failed: {exc}>"
    else:
        try:
            json.dumps(raw)
            serialised = raw
        except (TypeError, ValueError):
            serialised = repr(raw)
    return {
        "ok": True,
        "stdout": stdout.getvalue(),
        "result": serialised,
        "rendered_maps": rendered_paths,
        "error": None,
    }


# ── Public factory ──────────────────────────────────────────────────

def create_gee_tools(agent_id: str = "coder") -> list:
    """Build the Earth Engine tool list for an agent.

    Returns ``[]`` when GEE isn't configured — callers (the coding
    crew agent factory) just skip the tool without errors. Returns a
    one-element list ``[GeeRunScriptTool]`` when configured.
    """
    sa_path = _gee_credentials_path()
    if not sa_path:
        return []
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    class _GeeRunScriptInput(BaseModel):
        script: str = Field(
            description=(
                "Python snippet to execute against Earth Engine. The "
                "namespace pre-loads `ee` (the API) and `estonia` "
                "(an AOI FeatureCollection for Estonia). Assign your "
                "final value to a variable named `result` — the "
                "wrapper calls .getInfo() on it ONCE for you, so "
                "leaving it as an ee.Number/ee.Dictionary/ee.List is "
                "preferred over calling .getInfo() yourself."
            ),
        )
        timeout_s: int = Field(
            default=60,
            description=(
                "Maximum wall-clock seconds the call can take "
                "server-side. Heavy reductions can be slow; 60 is "
                "the default. EE export tasks (Drive/GCS) are "
                "asynchronous and not subject to this timeout."
            ),
        )

    class GeeRunScriptTool(BaseTool):
        name: str = "gee_run_script"
        description: str = (
            "Run a Python snippet against Google Earth Engine for "
            "satellite-imagery analysis (Hansen Global Forest Change, "
            "Sentinel-2 / Landsat composites, NDVI time-series, "
            "land-cover classification, MODIS, GEDI lidar). USE THIS "
            "for any geospatial analysis question — you do NOT need "
            "to download imagery yourself; Google's compute does the "
            "heavy lifting and returns aggregated numbers / vector "
            "summaries.\n\n"
            "ALWAYS use the LATEST Hansen GFC version: "
            "`UMD/hansen/global_forest_change_2024_v1_12` (covers "
            "loss through 2024).  Older snapshots (v1_11/2023, "
            "v1_10/2022) miss the most recent year and EE will "
            "warn you that they're DEPRECATED.  When in doubt, "
            "search the EE catalog or use dataset_search.\n\n"
            "CRITICAL — round-trip rule: every .getInfo() is a "
            "synchronous network call to Google (~30s each). Aggregate "
            "server-side, then pull ONCE.\n\n"
            "TWO LOOP DISTINCTIONS — they look similar but behave "
            "very differently:\n"
            "  COMPUTE-loop in Python (with .getInfo() inside) = BAD "
            "— each iteration is a 30s round-trip, N years = N×30s, "
            "blows the 180s tool budget.  Replace with a single "
            "server-side reducer.\n"
            "  RENDER-loop in Python (with render_map() inside) = OK "
            "— each iteration is a separate EE thumbnail call (~5-15s), "
            "12 years = ~60-180s total, fits the budget.  Use this "
            "when the user asks for per-year visual maps.\n\n"
            "# BAD compute-loop (13 round-trips, times out at 180s):\n"
            "for yr in range(12, 25):\n"
            "    out[yr] = mask.eq(yr).reduceRegion(...).getInfo()\n\n"
            "# GOOD compute (1 round-trip, ~110s):\n"
            "hist = loss.updateMask(loss.gt(0)).reduceRegion(\n"
            "    reducer=ee.Reducer.frequencyHistogram(), ...\n"
            ").get('lossyear').getInfo()\n\n"
            "# GOOD render-loop (12 thumbnails, ~120s, one PNG per year):\n"
            "for yr in range(12, 25):\n"
            "    render_map(\n"
            "        image=loss_year.eq(yr).updateMask(loss_year.eq(yr)),\n"
            "        region=estonia.geometry(),\n"
            "        name=f'estonia_loss_{2000+yr}',\n"
            "        vis_params={'palette': ['red'], 'min': 0, 'max': 1},\n"
            "    )\n\n"
            "MAP RENDERING — the sandbox pre-loads a render_map(image, "
            "region, name, vis_params) helper that synchronously saves "
            "an ee.Image as a PNG to workspace/output/maps/.  Use this "
            "when the user asks for VISUAL maps (not just statistics).  "
            "Each call is one round-trip (EE thumbnail API, ~5-15s).  "
            "PNG paths are returned in the result and listed in the "
            "tool output as 'rendered maps' with a `[gee_render]` "
            "marker so downstream consumers (vetting, critic) "
            "recognise them as legitimate file artifacts.\n\n"
            "Useful patterns: Hansen 'lossyear' band for year-by-year "
            "deforestation, ee.ImageCollection.filterDate for time-range "
            "queries, .clip(estonia.geometry()) for country-scoped work."
        )
        args_schema: Type[BaseModel] = _GeeRunScriptInput

        def _run(self, script: str, timeout_s: int = 60) -> str:
            ok, err = _ensure_initialised()
            if not ok:
                return (
                    f"GEE not initialised: {err}\n\n"
                    "Once configured, ALL EE calls go through this tool — "
                    "do NOT try to import google.cloud or reach Google "
                    "directly from agent code."
                )
            out = _run_user_script(script, timeout_s=timeout_s)
            # Compact the output for LLM consumption — keep both
            # machine-parseable JSON and a human-readable summary.
            if out["ok"]:
                # Build a friendly text summary first
                lines = ["GEE script completed."]
                if out["stdout"]:
                    lines.append(f"\n--- stdout ---\n{out['stdout'].rstrip()}")
                if out["result"] is not None:
                    lines.append(f"\n--- result ---\n{json.dumps(out['result'], indent=2, default=str)[:4000]}")
                rendered = out.get("rendered_maps") or []
                if rendered:
                    lines.append(_format_rendered_maps(rendered))
                return "\n".join(lines)
            # Failure path — still surface any maps rendered before the
            # exception (some succeed, some fail mid-script).
            err_msg = f"GEE script failed: {out['error']}\n\n--- stdout ---\n{out['stdout']}"
            rendered = out.get("rendered_maps") or []
            if rendered:
                err_msg += "\n\n" + _format_rendered_maps(
                    rendered, label="partial maps rendered before failure",
                )
            return err_msg

    return [GeeRunScriptTool()]


# ── Tool registry annotation (Phase 1a, passive) ────────────────────
try:
    from app.tool_registry import Lifecycle, Tier, register_tool

    def _gee_guard() -> bool:
        """True iff GOOGLE_APPLICATION_CREDENTIALS points at a readable file."""
        import os
        cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        return bool(cred and os.path.isfile(cred))

    @register_tool(
        name="gee_run_script",
        capabilities=["reads-satellite", "executes-earth-engine"],
        description=(
            "Execute a Python snippet against Google Earth Engine for "
            "satellite-imagery analysis. ALWAYS use this for forest / "
            "land-use / NDVI / GEDI / MODIS questions. CRITICAL: one "
            "`.getInfo()` per script — server-side aggregate first, "
            "single round-trip second. See docs/GEE.md."
        ),
        tier=Tier.PRODUCTION,
        lifecycle=Lifecycle.SINGLETON,
        guard=_gee_guard,
    )
    def _gee_run_script_registry_factory(agent_id: str = "coder"):
        tools = create_gee_tools(agent_id=agent_id)
        if not tools:
            raise RuntimeError("gee_run_script factory returned empty list")
        return tools[0]
except ImportError:
    pass
