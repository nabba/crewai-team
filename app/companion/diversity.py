"""MAP-Elites per-workspace diversity hook.

Each Companion cycle's converged output is recorded on a workspace-scoped
MAP-Elites grid (role = ``companion:<workspace_id>``). The grid retains
the best representative per behavioural cell along three feature
dimensions (complexity, cost_efficiency, specialization), so the next
cycle's prompt can read from void cells to deliberately push into
under-explored corners of the idea space — counters the Diverge phase's
tendency toward consensus.

Two integration points:

  - ``record_cycle(workspace_id, idea_id, text, fitness)`` is called
    after persistence in ``cycle.py``. Wraps the existing
    ``map_elites_wiring.record_crew_outcome`` with a CrewOutcome built
    from the cycle's output.

  - ``sparse_cell_hints(workspace_id, max_hints)`` returns short prompt
    fragments derived from void cells flanked by reasonably-performing
    neighbours. ``cycle._compose_prompt`` threads these as
    "explore these directions" guidance.

Failures (MAP-Elites schema unavailable, signature empty, fitness
missing) are logged and absorbed — the cycle proceeds without
diversity guidance rather than aborting.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ROLE_PREFIX = "companion:"
DEFAULT_HINT_LIMIT = 3


def workspace_role(workspace_id: str) -> str:
    """Stable MAP-Elites role string for one workspace."""
    safe = "".join(c for c in (workspace_id or "") if c.isalnum() or c in "-_")
    return f"{ROLE_PREFIX}{safe or 'default'}"


def record_cycle(workspace_id: str, idea_id: str, text: str,
                  fitness: float, *,
                  panel_score: float = 0.0,
                  duration_s: float = 0.0,
                  difficulty: int = 3) -> bool:
    """Append the cycle's converged output to the workspace grid.

    Returns True on successful write, False on any path that didn't
    produce a stored entry (empty text, missing fitness, MAP-Elites
    write-failure all silently log + return False).
    """
    if not text or not text.strip():
        return False
    if fitness is None:
        return False
    try:
        outcome = _build_outcome(workspace_id, idea_id, text, fitness,
                                  panel_score=panel_score,
                                  duration_s=duration_s,
                                  difficulty=difficulty)
        return bool(_invoke_record(outcome))
    except Exception as exc:
        logger.debug("companion.diversity: record_cycle failed: %s", exc)
        return False


def sparse_cell_hints(workspace_id: str, *,
                       max_hints: int = DEFAULT_HINT_LIMIT) -> list[str]:
    """Return up to max_hints short prompt fragments for under-explored cells.

    Each hint describes the target feature vector (complexity,
    cost_efficiency, specialization) of a void cell flanked by
    high-fitness neighbours — the highest-leverage exploration signals.
    """
    if max_hints <= 0:
        return []
    try:
        voids = _invoke_get_voids(workspace_id)
    except Exception as exc:
        logger.debug("companion.diversity: void lookup failed: %s", exc)
        return []
    return [_format_void_hint(v) for v in voids[:max_hints]
            if v is not None]


def coverage(workspace_id: str) -> dict:
    """Best-effort coverage report for the workspace's grid.

    Returns ``{}`` on any failure. Useful for telemetry / Phase 12
    "mode-collapse metric" diagnostics — coverage rising over time means
    the workspace is exploring more of the behavioural space.
    """
    try:
        return _invoke_coverage(workspace_id)
    except Exception as exc:
        logger.debug("companion.diversity: coverage failed: %s", exc)
        return {}


# ── Internal: outcome construction ─────────────────────────────────────────

def _build_outcome(workspace_id: str, idea_id: str, text: str,
                    fitness: float, *,
                    panel_score: float, duration_s: float, difficulty: int):
    """Compose a CrewOutcome from the cycle's signal.

    Fitness is supplied externally — the cycle already has novelty,
    quality, transferability, panel_score; the caller picks the
    composite. We map fitness → confidence/completeness so the existing
    ``compute_fitness`` lands on roughly the same value the cycle gave us.
    """
    from app.map_elites_wiring import CrewOutcome
    fitness = max(0.0, min(1.0, float(fitness)))
    if fitness >= 0.7:
        confidence = "high"
    elif fitness >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"
    return CrewOutcome(
        crew_name=workspace_role(workspace_id),
        task_description=(
            f"Workspace Companion contemplation cycle "
            f"(panel {panel_score:.2f}, idea {idea_id[:12]})"
        ),
        result=text,
        backstory_snippet=(
            "Workspace Companion · idle contemplation cycle · seeded by the "
            "workspace's grand task and its accumulated polished ideas."
        ),
        difficulty=int(difficulty),
        duration_s=float(duration_s),
        confidence=confidence,
        completeness="complete" if fitness >= 0.5 else "partial",
        passed_quality_gate=fitness >= 0.5,
        has_result=True,
        is_failure_pattern=fitness <= 0.2,
    )


# ── Internal: indirection over MAP-Elites for testability ─────────────────

def _invoke_record(outcome) -> bool:
    """Indirection over ``map_elites_wiring.record_crew_outcome``."""
    from app.map_elites_wiring import record_crew_outcome
    return record_crew_outcome(outcome)


def _invoke_get_voids(workspace_id: str) -> list[dict]:
    """Indirection over ``MAPElitesDB.get_voids`` for one workspace."""
    from app.map_elites import get_db
    db = get_db(workspace_role(workspace_id))
    if hasattr(db, "get_voids"):
        return list(db.get_voids() or [])
    # Older MAP-Elites versions exposed it on the grid directly.
    if hasattr(db, "_grids") and db._grids:
        return list(db._grids[0].get_void_cells() or [])
    return []


def _invoke_coverage(workspace_id: str) -> dict:
    """Indirection over ``MAPElitesDB.get_coverage_report``."""
    from app.map_elites import get_db
    db = get_db(workspace_role(workspace_id))
    if hasattr(db, "get_coverage_report"):
        return dict(db.get_coverage_report() or {})
    return {}


# ── Hint formatting ────────────────────────────────────────────────────────

def _format_void_hint(void: dict) -> str:
    """Turn a void-cell descriptor into a one-line prompt fragment."""
    target = (void or {}).get("feature_target") or {}
    parts = []
    for dim, value in target.items():
        parts.append(f"{dim}={value:.2f}")
    if not parts:
        return "Explore an under-tried direction."
    return "Explore a direction with " + ", ".join(parts) + "."
