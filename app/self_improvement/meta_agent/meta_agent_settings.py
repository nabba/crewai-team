"""
app.self_improvement.meta_agent.meta_agent_settings — persistent on/off
switch for the meta-agent recipe path.

Mirrors app.crews.delegation_settings exactly. When ON for a crew, the
``run_single_agent_crew`` dispatch routes through the meta-agent
selector; when OFF, the dispatch uses factory defaults (no augmentation,
no recipe learning side-effect).

Backed by a simple JSON file in the workspace so the setting survives
container restarts without env redeploys. The dashboard Org Chart
toggles this via the /api/cp/meta-agent endpoints.

Resolution order in feature_flag.is_meta_agent_enabled:
    1. Per-crew env var ``META_AGENT_<CREW>`` set explicitly  (ops override)
    2. Master env var ``META_AGENT=1``                        (ops master)
    3. Persisted settings JSON (this module)                  (dashboard)
    4. Default: OFF

This module never touches the env vars — those remain ops-only.

IMMUTABLE — infrastructure-level module.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(os.environ.get(
    "META_AGENT_SETTINGS_PATH",
    "/app/workspace/meta_agent_settings.json",
))
_LOCK = threading.Lock()

# Default state: OFF for every crew. Same conservative rollout pattern as
# delegation_settings — operator opts in explicitly.
#
# The crew list mirrors delegation_settings._DEFAULTS so the React Org
# Chart can render both panels with the same crew rows. Meta-agent only
# fires from run_single_agent_crew, so additional crews going through
# that chokepoint can be added here when they're ready.
_DEFAULTS: dict[str, bool] = {
    "research": False,
    "coding": False,
    "writing": False,
}


def _load() -> dict[str, bool]:
    if not _SETTINGS_PATH.exists():
        return dict(_DEFAULTS)
    try:
        raw = json.loads(_SETTINGS_PATH.read_text())
        if not isinstance(raw, dict):
            return dict(_DEFAULTS)
        merged = dict(_DEFAULTS)
        for k, v in raw.items():
            if isinstance(v, bool) and k in _DEFAULTS:
                merged[k] = v
        return merged
    except Exception:
        logger.debug("meta_agent_settings: load failed", exc_info=True)
        return dict(_DEFAULTS)


def _save(state: dict[str, bool]) -> None:
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


def get_all() -> dict[str, bool]:
    """Return {crew_name: bool} for every configurable crew."""
    with _LOCK:
        return _load()


def is_enabled(crew: str) -> bool:
    """Return True if the persisted setting has the meta-agent ON for this crew.

    The full feature-flag check (env override → settings → default) lives
    in feature_flag.is_meta_agent_enabled. This function only reads the
    JSON layer.
    """
    return bool(get_all().get(crew, False))


def set_enabled(crew: str, enabled: bool) -> dict[str, bool]:
    """Enable or disable the meta-agent for a specific crew. Returns the
    full updated state. Unknown crews are ignored."""
    if crew not in _DEFAULTS:
        return get_all()
    with _LOCK:
        state = _load()
        state[crew] = bool(enabled)
        _save(state)
    logger.info(f"meta_agent_settings: {crew} → {'ON' if enabled else 'OFF'}")
    return state
