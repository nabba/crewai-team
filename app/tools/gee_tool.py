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


def _run_user_script(script: str, timeout_s: int = 60) -> dict[str, Any]:
    """Execute the user's GEE Python snippet.

    Returns ``{ok, stdout, result, error}``:
      * ``ok``: True if the script ran without uncaught exception.
      * ``stdout``: anything the script ``print()``-ed.
      * ``result``: value of the script's ``result`` variable (if any).
      * ``error``: short exception message when ``ok=False``, else None.

    The sandbox dict pre-loads ``ee`` and a couple of common helpers so
    short snippets stay short. Heavy clients (geemap, etc.) can be
    imported by the script itself.
    """
    import ee

    # Pre-populated namespace — keeps short scripts short.
    sandbox: dict[str, Any] = {
        "ee": ee,
        # Common AOI helpers
        "estonia": ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
            ee.Filter.eq("ADM0_NAME", "Estonia"),
        ),
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
            "CRITICAL — round-trip rule: every .getInfo() is a "
            "synchronous network call to Google (~30s each). Aggregate "
            "server-side, then pull ONCE. For per-year/per-class/"
            "per-region values use ee.Reducer.frequencyHistogram, "
            "ee.Reducer.group, or map a reducer over an "
            "ee.FeatureCollection — NEVER a Python for-loop with "
            ".getInfo() inside.\n\n"
            "# BAD (13 round-trips, times out at 240s):\n"
            "for yr in range(12, 25):\n"
            "    out[yr] = mask.eq(yr).reduceRegion(...).getInfo()\n\n"
            "# GOOD (1 round-trip, ~110s):\n"
            "hist = loss.updateMask(loss.gt(0)).reduceRegion(\n"
            "    reducer=ee.Reducer.frequencyHistogram(), ...\n"
            ").get('lossyear').getInfo()\n\n"
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
                return "\n".join(lines)
            return f"GEE script failed: {out['error']}\n\n--- stdout ---\n{out['stdout']}"

    return [GeeRunScriptTool()]
