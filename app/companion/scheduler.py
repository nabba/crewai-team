"""Fairness scheduler — pick the next workspace for a Companion tick.

CFS-inspired: each workspace has a vruntime that grows with cycle wall-time;
the workspace with the lowest ``vruntime / weight`` is chosen next. A 12 h
temporal floor guarantees no active workspace is starved over a day,
regardless of weight.

Phase 2 wires the affect-bridge factual_grounding signal:
  - below the grounding floor → ALL cycles paused (system in adverse state)
  - otherwise → grounding ∈ [0,1] maps to weight ∈ [WEIGHT_FLOOR, WEIGHT_CEIL]
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.companion import budget as _budget
from app.companion import state as _state
from app.companion.config import CompanionConfig

logger = logging.getLogger(__name__)

TEMPORAL_FLOOR_S = 12 * 3600
WEIGHT_FLOOR = 0.5
WEIGHT_CEIL = 2.0

# Below this affect-bridge factual_grounding score, cycles are paused.
# 0.3 mirrors the existing convention for adverse epistemic coupling
# (distress / frozen / depleted / overwhelm attractors).
AFFECT_GROUNDING_FLOOR = 0.3


@dataclass(frozen=True)
class Candidate:
    project_id: str
    config: CompanionConfig
    state: _state.WorkspaceState
    weight: float


def collect_candidates(now_local_hour: int | None = None) -> list[Candidate]:
    """Workspaces eligible for a Companion tick right now.

    Filters out: disabled, in quiet hours, budget-exhausted, in adverse
    affect attractor (Phase 2+ — currently a stub returning None).
    """
    candidates: list[Candidate] = []
    try:
        rows = _list_projects()
    except Exception as exc:
        logger.debug("companion.scheduler: list_all failed: %s", exc)
        return []

    if now_local_hour is None:
        now_local_hour = _local_hour()

    skip_reason_for_affect = _affect_blocks_cycles()

    for row in rows:
        pid = row.get("id")
        if not pid:
            continue
        cfg_raw = (row.get("config_json") or {}).get("companion") or {}
        cfg = CompanionConfig.from_dict(cfg_raw)
        if not cfg.enabled:
            continue
        st = _state.load(pid)
        if cfg.is_quiet_hour(now_local_hour):
            _record_skip(st, "quiet_hours")
            continue
        if _budget.is_exhausted(pid, cfg):
            _record_skip(st, "budget_exhausted")
            continue
        if skip_reason_for_affect:
            _record_skip(st, f"affect:{skip_reason_for_affect}")
            continue
        candidates.append(Candidate(pid, cfg, st, _affect_weight()))
    return candidates


def select_next(now_unix: float | None = None,
                now_local_hour: int | None = None) -> Candidate | None:
    """Pick the next workspace to tick, or None if nothing is eligible."""
    if now_unix is None:
        now_unix = time.time()
    cands = collect_candidates(now_local_hour=now_local_hour)
    if not cands:
        return None

    starved = [
        c for c in cands
        if c.state.last_tick_at == 0
        or (now_unix - c.state.last_tick_at) > TEMPORAL_FLOOR_S
    ]
    if starved:
        return min(starved, key=lambda c: c.state.last_tick_at)

    return min(cands, key=lambda c: c.state.vruntime_s / max(c.weight, WEIGHT_FLOOR))


def record_tick(project_id: str, cycle_cost_s: float, weight: float,
                now_unix: float | None = None) -> None:
    """Update vruntime + last_tick_at after a cycle completes."""
    if now_unix is None:
        now_unix = time.time()
    s = _state.load(project_id)
    eff_w = max(min(float(weight), WEIGHT_CEIL), WEIGHT_FLOOR)
    s.vruntime_s = round(s.vruntime_s + max(0.0, float(cycle_cost_s)) / eff_w, 4)
    s.last_tick_at = now_unix
    s.cycles_total += 1
    s.last_skip_reason = None
    _state.save(s)


def _list_projects() -> list[dict]:
    """Indirection for ``app.control_plane.projects.get_projects().list_all()``.

    The local seam keeps tests from depending on the full DB stack (psycopg2,
    pool init) just to stub project lists.
    """
    from app.control_plane.projects import get_projects
    return get_projects().list_all() or []


def _record_skip(state_obj: _state.WorkspaceState, reason: str) -> None:
    if state_obj.last_skip_reason != reason:
        state_obj.last_skip_reason = reason
        _state.save(state_obj)


def _local_hour() -> int:
    """Current Helsinki hour 0-23. Falls back to UTC if zoneinfo lookup fails."""
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Europe/Helsinki")).hour
    except Exception:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).hour


def _affect_blocks_cycles() -> str | None:
    """If the affect bridge reports adverse epistemic coupling, return reason.

    Reads ``live_factual_grounding()`` ∈ [0.0, 1.0]. Below the floor, the
    system is signalling distress / frozen / depletion; the Companion stays
    quiet rather than burning cycles in adverse states. None on missing
    signal (treat as neutral, do not block).
    """
    grounding = _live_grounding()
    if grounding is None:
        return None
    if grounding < AFFECT_GROUNDING_FLOOR:
        return f"low_grounding({grounding:.2f})"
    return None


def _affect_weight() -> float:
    """Map factual_grounding ∈ [0,1] → CFS weight in [WEIGHT_FLOOR, WEIGHT_CEIL].

    Higher grounding → higher weight → the workspace gets more cycles per
    unit vruntime. Falls back to 1.0 (neutral) on missing signal.
    """
    grounding = _live_grounding()
    if grounding is None:
        return 1.0
    return max(WEIGHT_FLOOR, min(WEIGHT_CEIL, 0.5 + float(grounding)))


def _live_grounding() -> float | None:
    """Indirection over the affect-bridge import for testability."""
    try:
        from app.epistemic.affect_bridge import live_factual_grounding
        return live_factual_grounding()
    except Exception:
        return None
