"""CompanionConfig — durable per-workspace settings.

Stored inside ``control_plane.projects.config_json`` under the ``companion``
key, so no DB migration is needed. Runtime state (cost ledger, vruntime,
last_tick_at) is NOT here — see ``app.companion.state``.

Bounds in this module are infrastructure-level: per CLAUDE.md's safety
invariants, the Self-Improver agent cannot widen them.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MIN_DAILY_BUDGET_USD = 0.0
MAX_DAILY_BUDGET_USD = 100.0
DEFAULT_DAILY_BUDGET_USD = 1.0

MIN_SURFACE_THRESHOLD = 0.5
MAX_SURFACE_THRESHOLD = 0.95
DEFAULT_SURFACE_THRESHOLD = 0.7

MIN_NOVELTY_THRESHOLD = 0.3
MAX_NOVELTY_THRESHOLD = 0.95
DEFAULT_NOVELTY_THRESHOLD = 0.7

MIN_TRANSFERABILITY_THRESHOLD = 0.5
MAX_TRANSFERABILITY_THRESHOLD = 0.95
DEFAULT_TRANSFERABILITY_THRESHOLD = 0.7

# Phase 7: critic-panel aggregate must clear this for surfacing.
MIN_PANEL_THRESHOLD = 0.4
MAX_PANEL_THRESHOLD = 0.9
DEFAULT_PANEL_THRESHOLD = 0.6

DEFAULT_QUIET_HOURS_START = 2
DEFAULT_QUIET_HOURS_END = 6


@dataclass
class CompanionConfig:
    """Per-workspace Companion settings."""

    enabled: bool = True
    seed_prompt: str | None = None
    daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD
    surface_threshold: float = DEFAULT_SURFACE_THRESHOLD
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD
    transferability_threshold: float = DEFAULT_TRANSFERABILITY_THRESHOLD
    panel_threshold: float = DEFAULT_PANEL_THRESHOLD
    quiet_hours_start: int = DEFAULT_QUIET_HOURS_START
    quiet_hours_end: int = DEFAULT_QUIET_HOURS_END
    sources: list[dict] = field(default_factory=list)

    def clamp(self) -> "CompanionConfig":
        self.daily_budget_usd = _clamp(self.daily_budget_usd,
                                        MIN_DAILY_BUDGET_USD,
                                        MAX_DAILY_BUDGET_USD)
        self.surface_threshold = _clamp(self.surface_threshold,
                                         MIN_SURFACE_THRESHOLD,
                                         MAX_SURFACE_THRESHOLD)
        self.novelty_threshold = _clamp(self.novelty_threshold,
                                         MIN_NOVELTY_THRESHOLD,
                                         MAX_NOVELTY_THRESHOLD)
        self.transferability_threshold = _clamp(
            self.transferability_threshold,
            MIN_TRANSFERABILITY_THRESHOLD,
            MAX_TRANSFERABILITY_THRESHOLD,
        )
        self.panel_threshold = _clamp(self.panel_threshold,
                                       MIN_PANEL_THRESHOLD,
                                       MAX_PANEL_THRESHOLD)
        self.quiet_hours_start = max(0, min(23, int(self.quiet_hours_start)))
        self.quiet_hours_end = max(0, min(23, int(self.quiet_hours_end)))
        return self

    def is_quiet_hour(self, hour: int) -> bool:
        s, e = self.quiet_hours_start, self.quiet_hours_end
        if s == e:
            return False
        if s < e:
            return s <= hour < e
        return hour >= s or hour < e  # wraps midnight

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CompanionConfig":
        if not raw:
            return cls()
        fields = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**fields).clamp()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo


def load(project_id: str) -> CompanionConfig | None:
    """Read CompanionConfig from CP projects.config_json.companion.

    Returns None if the project doesn't exist; default-config if it exists
    without companion settings yet.
    """
    try:
        row = _get_project_by_id(project_id)
    except Exception as exc:
        logger.debug("companion.config: get_by_id failed for %s: %s",
                     project_id, exc)
        return None
    if not row:
        return None
    cfg = (row.get("config_json") or {}).get("companion") or {}
    return CompanionConfig.from_dict(cfg)


def _get_project_by_id(project_id: str) -> dict | None:
    """Indirection for ``app.control_plane.projects.get_projects().get_by_id()``.

    The local seam keeps tests from importing the full DB stack just to stub
    a project lookup.
    """
    from app.control_plane.projects import get_projects
    return get_projects().get_by_id(project_id)


def save(project_id: str, config: CompanionConfig) -> bool:
    """Persist CompanionConfig to ``CP.projects.config_json.companion``.

    Read-modify-write merge: the rest of ``config_json`` is preserved so
    we don't stomp settings owned by other modules. Returns False on any
    failure (project missing, DB unavailable, write error). Bounds are
    re-clamped before write so callers can't slip out-of-range values past
    the API into storage.
    """
    cfg = config.clamp()
    try:
        existing = _get_full_config_json(project_id)
    except Exception as exc:
        logger.warning("companion.config: read for save failed (%s): %s",
                       project_id, exc)
        return False
    if existing is None:
        return False
    if not isinstance(existing, dict):
        existing = {}
    existing["companion"] = cfg.to_dict()
    try:
        return _save_full_config_json(project_id, existing)
    except Exception as exc:
        logger.warning("companion.config: save failed for %s: %s",
                       project_id, exc)
        return False


def _get_full_config_json(project_id: str) -> dict | None:
    """Indirection for testability — returns the project's whole config_json,
    or None if the project doesn't exist."""
    row = _get_project_by_id(project_id)
    if row is None:
        return None
    return row.get("config_json") or {}


def _save_full_config_json(project_id: str, config_json: dict) -> bool:
    """Indirection for testability — UPDATE the project's config_json."""
    import json as _json
    from app.control_plane.db import execute
    execute(
        "UPDATE control_plane.projects SET config_json = %s WHERE id = %s",
        (_json.dumps(config_json), project_id),
        fetch=False,
    )
    return True
