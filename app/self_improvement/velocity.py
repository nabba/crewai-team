"""Self-improvement velocity — observational rollup of "how fast are
we improving, and is it working?"

PROGRAM §51 — Q16 Theme 4 (decade-resilience hardening, recursive
self-improvement boundaries). The Tier-3 amendment + self-quarantine
+ Goodhart hard gate are structural safety. Velocity is the OTHER
question: *is the system actually getting better?*

Without an observable velocity surface, we can't tell whether:

  * `capability_gap_analyzer` is producing useful CRs or just noise.
  * Architecture requests that landed are actually being used.
  * Meta-agent recipes are converging or collapsing.
  * `lessons_learned` is growing — and being CONSULTED.
  * The Forge tool pipeline is alive or dormant.

This module is **observational only**. It never mutates state, never
files a CR, never gates anything. The numbers feed:

  * REST endpoint ``GET /api/cp/self-improvement/velocity``
  * (Optional future) React dashboard at ``/cp/self-improvement``

Each section is failure-isolated: a broken upstream source returns
an empty/placeholder payload but never breaks the whole roll-up.

Data sources
------------

  * Change requests (``app.change_requests.store.list_all``) — by
    requestor, by status, by quarter.
  * Architecture requests (``app.architecture_requests.store.list_all``)
    + adoption probe (``app.architecture_requests.adoption.measure``).
  * Meta-agent recipes (``app.self_improvement.meta_agent.store.list_recipes``).
  * Lessons-learned KB (``workspace/self_heal/lessons_learned.json`` /
    ``workspace/companion/lessons_learned.json``).
  * Forge graduations — discovered from
    ``workspace/forge/graduations.jsonl`` if present.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_DAYS = 365


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_self_improvement_velocity_enabled
        return get_self_improvement_velocity_enabled()
    except Exception:
        return os.getenv(
            "SELF_IMPROVEMENT_VELOCITY_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _quarter_key(ts: float) -> str:
    """UTC year-quarter key, e.g. "2026Q2"."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{q}"


def _parse_iso(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


# ── 1. Change-requests velocity ──────────────────────────────────────────


def _crs_by_quarter(window_days: int = _DEFAULT_WINDOW_DAYS) -> dict[str, Any]:
    """Count CRs in window by quarter × (requestor, status)."""
    out: dict[str, Any] = {
        "available": False,
        "n_total": 0,
        "by_quarter": {},
        "by_requestor": {},
        "by_status": {},
        "applied_rate_overall": None,
    }
    try:
        from app.change_requests import store, Status
        crs = store.list_all(limit=10_000)
    except Exception:
        return out
    out["available"] = True
    cutoff = time.time() - window_days * 86400
    n_total = 0
    by_quarter: dict[str, Counter] = defaultdict(Counter)
    by_requestor: Counter = Counter()
    by_status: Counter = Counter()
    n_applied = 0
    n_resolved = 0
    for cr in crs:
        ts = _parse_iso(getattr(cr, "created_at", None))
        if ts is None or ts < cutoff:
            continue
        n_total += 1
        requestor = getattr(cr, "requestor", "(unknown)") or "(unknown)"
        status = getattr(cr.status, "value", str(cr.status)) if hasattr(cr, "status") else "(unknown)"
        q = _quarter_key(ts)
        by_quarter[q][status] += 1
        by_requestor[requestor] += 1
        by_status[status] += 1
        if status in ("applied",):
            n_applied += 1
            n_resolved += 1
        elif status in ("rejected", "rolled_back", "timeout", "tier_immutable_refused"):
            n_resolved += 1
    out["n_total"] = n_total
    # Normalise quarter table to plain dicts.
    out["by_quarter"] = {
        q: dict(counter) for q, counter in sorted(by_quarter.items())
    }
    out["by_requestor"] = dict(by_requestor.most_common())
    out["by_status"] = dict(by_status.most_common())
    if n_resolved > 0:
        out["applied_rate_overall"] = round(n_applied / n_resolved, 3)
    return out


# ── 2. Architecture-request adoption histogram ───────────────────────────


def _architecture_adoption_histogram() -> dict[str, Any]:
    """Distribution of adoption scores across APPLIED architecture
    requests in the 30–60d post-applied window (the same window the
    healing monitor probes). Buckets: 0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8,
    0.8–1.0."""
    out: dict[str, Any] = {
        "available": False,
        "n_measured": 0,
        "histogram": {},
        "below_rollback_threshold": 0,
    }
    try:
        from app.architecture_requests import store as arch_store
        from app.architecture_requests.adoption import measure
    except Exception:
        return out
    try:
        all_reqs = arch_store.list_all(limit=500)
    except Exception:
        return out
    out["available"] = True
    buckets = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    histogram = {b: 0 for b in buckets}
    below_threshold = 0
    n = 0
    for req in all_reqs:
        # Probe ONLY APPLIED requests (the adoption monitor's contract).
        status_val = getattr(req.status, "value", str(req.status)) if hasattr(req, "status") else ""
        if status_val not in ("applied", "stable"):
            continue
        try:
            result = measure(req.id) if hasattr(req, "id") else None
        except Exception:
            continue
        if result is None:
            continue
        score = getattr(result, "score", None)
        if score is None or not isinstance(score, (int, float)):
            continue
        n += 1
        if score < 0.2:
            histogram["0.0-0.2"] += 1
        elif score < 0.4:
            histogram["0.2-0.4"] += 1
        elif score < 0.6:
            histogram["0.4-0.6"] += 1
        elif score < 0.8:
            histogram["0.6-0.8"] += 1
        else:
            histogram["0.8-1.0"] += 1
        if score < 0.20:
            below_threshold += 1
    out["n_measured"] = n
    out["histogram"] = histogram
    out["below_rollback_threshold"] = below_threshold
    return out


# ── 3. Meta-agent recipe convergence ─────────────────────────────────────


def _recipe_selection_summary(top_n: int = 10) -> dict[str, Any]:
    """Summarise the top-N most-used recipes + total active recipes.
    The full Gini probe lives in ``app/healing/monitors/feedback_loop_drift``
    — here we surface the headline numbers."""
    out: dict[str, Any] = {
        "available": False,
        "n_active": 0,
        "top": [],
        "total_uses": 0,
        "total_successes": 0,
        "global_success_rate": None,
    }
    try:
        from app.self_improvement.meta_agent import store as ma_store
        recipes = ma_store.list_recipes(limit=500)
    except Exception:
        return out
    out["available"] = True
    out["n_active"] = len(recipes)
    total_uses = sum(int(getattr(r, "uses", 0)) for r in recipes)
    total_successes = sum(int(getattr(r, "successes", 0)) for r in recipes)
    out["total_uses"] = total_uses
    out["total_successes"] = total_successes
    if total_uses > 0:
        out["global_success_rate"] = round(total_successes / total_uses, 3)
    top = sorted(
        recipes,
        key=lambda r: int(getattr(r, "uses", 0)),
        reverse=True,
    )[:top_n]
    for r in top:
        uses = int(getattr(r, "uses", 0))
        successes = int(getattr(r, "successes", 0))
        rate = round(successes / uses, 3) if uses > 0 else None
        out["top"].append({
            "id": getattr(r, "id", None),
            "crew_name": getattr(r, "crew_name", None),
            "uses": uses,
            "successes": successes,
            "success_rate": rate,
        })
    return out


# ── 4. Lessons-learned growth + reuse ────────────────────────────────────


def _lessons_learned_summary() -> dict[str, Any]:
    """Total lessons recorded + recent additions. The "reuse" metric
    is approximated by counting lessons with ``consulted_at_history``
    non-empty; older versions of the KB may not have that field, in
    which case we report n_with_consultations=None."""
    out: dict[str, Any] = {
        "available": False,
        "sources": [],
        "n_total": 0,
        "n_added_last_30d": 0,
        "n_with_consultations": None,
    }
    candidates = [
        _workspace() / "self_heal" / "lessons_learned.json",
        _workspace() / "companion" / "lessons_learned.json",
    ]
    n_total = 0
    n_recent = 0
    n_consulted = 0
    n_consultation_field_seen = 0
    cutoff = time.time() - 30 * 86400
    for p in candidates:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out["sources"].append(str(p.relative_to(_workspace())))
        # Tolerate two shapes: list of dicts OR dict with "lessons" key.
        if isinstance(data, dict):
            lessons = data.get("lessons") or data.get("entries") or []
        elif isinstance(data, list):
            lessons = data
        else:
            lessons = []
        for lesson in lessons:
            if not isinstance(lesson, dict):
                continue
            n_total += 1
            created = _parse_iso(
                lesson.get("created_at")
                or lesson.get("recorded_at")
                or lesson.get("ts")
            )
            if created is not None and created >= cutoff:
                n_recent += 1
            if "consulted_at_history" in lesson:
                n_consultation_field_seen += 1
                history = lesson.get("consulted_at_history") or []
                if history:
                    n_consulted += 1
    out["available"] = bool(out["sources"])
    out["n_total"] = n_total
    out["n_added_last_30d"] = n_recent
    if n_consultation_field_seen > 0:
        out["n_with_consultations"] = n_consulted
    return out


# ── 5. Forge graduations ─────────────────────────────────────────────────


def _forge_graduations_summary() -> dict[str, Any]:
    """Count SHADOW → CANARY → PROMOTED transitions from the forge
    graduations log. The log is optional; absent file → zero counts."""
    out: dict[str, Any] = {
        "available": False,
        "n_total": 0,
        "by_stage": {},
        "n_last_90d": 0,
    }
    p = _workspace() / "forge" / "graduations.jsonl"
    if not p.exists():
        return out
    out["available"] = True
    cutoff = time.time() - 90 * 86400
    n_total = 0
    by_stage: Counter = Counter()
    n_recent = 0
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                n_total += 1
                stage = row.get("to_stage") or row.get("stage") or "(unknown)"
                by_stage[stage] += 1
                ts = _parse_iso(row.get("ts") or row.get("timestamp"))
                if ts is not None and ts >= cutoff:
                    n_recent += 1
    except OSError:
        pass
    out["n_total"] = n_total
    out["by_stage"] = dict(by_stage.most_common())
    out["n_last_90d"] = n_recent
    return out


# ── 6. Top-level rollup ──────────────────────────────────────────────────


def velocity_summary(
    *,
    window_days: int = _DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    """One-shot aggregation. Every section is failure-isolated so a
    broken source doesn't break the whole report.

    Returns:
        ``{
            "generated_at": ISO-8601,
            "window_days": int,
            "change_requests": {...},
            "architecture_adoption": {...},
            "recipes": {...},
            "lessons_learned": {...},
            "forge_graduations": {...},
        }``

    When the master switch is OFF, returns ``{"disabled": True}``.
    """
    if not _enabled():
        return {"disabled": True}
    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": int(window_days),
    }
    for key, fn in (
        ("change_requests", lambda: _crs_by_quarter(window_days=window_days)),
        ("architecture_adoption", _architecture_adoption_histogram),
        ("recipes", _recipe_selection_summary),
        ("lessons_learned", _lessons_learned_summary),
        ("forge_graduations", _forge_graduations_summary),
    ):
        try:
            out[key] = fn()
        except Exception:
            logger.debug(
                "velocity: %s aggregation raised", key, exc_info=True,
            )
            out[key] = {"available": False, "error": True}
    return out
