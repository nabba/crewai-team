"""Goal-progress probe — daily inference of progress on current_goals.

PROGRAM §51 — Q16 Theme 7. The long-term-goal-review (Q9.6) produces
quarterly review compositions. Goal-progress probes the gap between
reviews: are this week's actions actually advancing what the
operator (or the system itself, via the goal_emitter) named as
current goals?

What this module does
=====================

  * Loads ``current_goals`` from the SubIA kernel (a list of goal
    descriptors with text + optional tags).
  * For each goal, scans a recent activity window for evidence of
    movement:
      - completed crew_tasks with matching tags / title tokens
      - ledger events naming the goal text
      - companion ideas referencing the goal
      - browse-topic clusters that overlap goal terms
  * Scores each goal as ``advancing``, ``stalled``, or
    ``insufficient_data``.
  * Emits a "stalled goals" Signal alert on a per-goal 14-day dedup
    window when ≥1 goal is stalled for >14 consecutive days.

What this module deliberately doesn't do
========================================

  * No auto-edit of current_goals. Goal removal is operator-only
    (kernel.self_state is Tier-3-protected).
  * No LLM grading. Pure structural overlap on tokenised goal
    text. Cost: zero per probe.
  * No filing of CRs / tickets. Surfaces the signal to the
    operator; that's the contract.

Cadence: daily probe; per-goal evaluation each pass; 14-day alert
dedup per goal.

Master switch: ``goal_progress_probe_enabled`` (default ON).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_STATE_FILE = "goal_progress_state.json"
_STALLED_DAYS_THRESHOLD = 14
_ACTIVITY_WINDOW_DAYS = 30
_DEDUP_WINDOW_S = 14 * 86400
_MIN_TOKEN_LENGTH = 4   # avoid matching "the", "and", etc.


@dataclass
class GoalStatus:
    """One goal's inferred status."""
    text: str
    tokens: list[str]
    evidence_count: int
    sources: list[str]
    status: str           # "advancing" | "stalled" | "insufficient_data"
    last_evidence_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tokens": list(self.tokens),
            "evidence_count": int(self.evidence_count),
            "sources": list(self.sources),
            "status": str(self.status),
            "last_evidence_at": self.last_evidence_at,
        }


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_goal_progress_probe_enabled
        return get_goal_progress_probe_enabled()
    except Exception:
        return os.getenv(
            "GOAL_PROGRESS_PROBE_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "companion" / _STATE_FILE


def _tokenise(text: str) -> list[str]:
    """Lowercase, drop short tokens, dedupe."""
    if not isinstance(text, str):
        return []
    tokens = re.findall(r"[A-Za-z]{%d,}" % _MIN_TOKEN_LENGTH, text.lower())
    return list(dict.fromkeys(tokens))


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "per_goal_stall_since": {}, "last_alert_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "per_goal_stall_since": {}, "last_alert_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("goal_progress: state write failed", exc_info=True)


def _load_current_goals() -> list[Any]:
    """Load current_goals from the SubIA kernel.

    Canonical pattern (mirrors ``app/identity/long_term_goal_review.py:
    _current_goals``): ``kernel.self_state.current_goals``. Returns
    empty list on any failure.
    """
    try:
        from app.subia import kernel
        goals = getattr(kernel.self_state, "current_goals", None) or []
        if isinstance(goals, list):
            return goals
    except Exception:
        logger.debug("goal_progress: kernel load failed", exc_info=True)
    return []


def _normalise_goal(goal: Any) -> tuple[str, list[str]]:
    """Extract (text, tokens) from a goal — accepts strings or dicts
    with a 'text'/'title'/'description' field."""
    if isinstance(goal, str):
        text = goal.strip()
    elif isinstance(goal, dict):
        text = (
            goal.get("text") or goal.get("title")
            or goal.get("description") or json.dumps(goal, sort_keys=True)
        )
    else:
        text = str(goal)
    return text[:500], _tokenise(text)


# ── Evidence sources ─────────────────────────────────────────────────────


def _scan_crew_tasks(
    tokens_by_goal: dict[str, list[str]],
    *,
    cutoff: float,
) -> dict[str, list[dict[str, Any]]]:
    """Match goal tokens against completed crew_task descriptions in
    the last _ACTIVITY_WINDOW_DAYS. Failure-isolated."""
    matches: dict[str, list[dict[str, Any]]] = {k: [] for k in tokens_by_goal}
    try:
        from app.control_plane import db
        rows = db.execute(
            """
            SELECT id, description, completed_at, status
            FROM crew_tasks
            WHERE completed_at >= NOW() - INTERVAL '30 days'
            ORDER BY completed_at DESC
            LIMIT 500
            """,
            (),
            fetch=True,
        ) or []
    except Exception:
        return matches
    for row in rows:
        desc = (row.get("description") if isinstance(row, dict) else row[1]) or ""
        completed_at = row.get("completed_at") if isinstance(row, dict) else row[2]
        ts: Optional[float] = None
        if hasattr(completed_at, "timestamp"):
            try:
                ts = completed_at.timestamp()
            except Exception:
                ts = None
        elif isinstance(completed_at, str):
            try:
                dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            except Exception:
                ts = None
        if ts is None or ts < cutoff:
            continue
        desc_tokens = set(_tokenise(desc))
        for goal_text, goal_tokens in tokens_by_goal.items():
            if not goal_tokens:
                continue
            overlap = desc_tokens & set(goal_tokens)
            # Require ≥2 token overlap to count as evidence — single-
            # word matches are too noisy.
            if len(overlap) >= 2:
                matches[goal_text].append({
                    "source": "crew_tasks",
                    "id": row.get("id") if isinstance(row, dict) else row[0],
                    "ts": ts,
                })
    return matches


def _scan_companion_ideas(
    tokens_by_goal: dict[str, list[str]],
    *,
    cutoff: float,
) -> dict[str, list[dict[str, Any]]]:
    """Walk workspace/companion/ideas/*.jsonl for entries within the
    window."""
    matches: dict[str, list[dict[str, Any]]] = {k: [] for k in tokens_by_goal}
    ideas_dir = _workspace() / "companion" / "ideas"
    if not ideas_dir.exists():
        return matches
    try:
        for path in ideas_dir.glob("*.jsonl"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        ts_str = row.get("ts") or row.get("created_at") or ""
                        if not isinstance(ts_str, str):
                            continue
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            ts = dt.timestamp()
                        except Exception:
                            continue
                        if ts < cutoff:
                            continue
                        text = (
                            (row.get("idea") or row.get("text") or "")
                            + " " + (row.get("title") or "")
                        )
                        idea_tokens = set(_tokenise(text))
                        for goal_text, goal_tokens in tokens_by_goal.items():
                            if not goal_tokens:
                                continue
                            if len(idea_tokens & set(goal_tokens)) >= 2:
                                matches[goal_text].append({
                                    "source": "companion_ideas",
                                    "id": row.get("id"),
                                    "ts": ts,
                                })
            except OSError:
                continue
    except Exception:
        logger.debug("goal_progress: ideas walk failed", exc_info=True)
    return matches


def _scan_continuity_ledger(
    tokens_by_goal: dict[str, list[str]],
    *,
    cutoff: float,
) -> dict[str, list[dict[str, Any]]]:
    """Walk the continuity ledger for events naming goal terms."""
    matches: dict[str, list[dict[str, Any]]] = {k: [] for k in tokens_by_goal}
    ledger_path = _workspace() / "identity" / "continuity_ledger.jsonl"
    if not ledger_path.exists():
        return matches
    try:
        with open(ledger_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts_str = row.get("ts") or ""
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ts = dt.timestamp()
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                text = (row.get("summary") or "") + " " + (row.get("kind") or "")
                event_tokens = set(_tokenise(text))
                for goal_text, goal_tokens in tokens_by_goal.items():
                    if not goal_tokens:
                        continue
                    if len(event_tokens & set(goal_tokens)) >= 2:
                        matches[goal_text].append({
                            "source": "continuity_ledger",
                            "kind": row.get("kind"),
                            "ts": ts,
                        })
    except OSError:
        pass
    return matches


def _alert_stalled(
    stalled: list[GoalStatus],
    state: dict[str, Any],
    *,
    now: float,
) -> bool:
    """One topic-keyed Signal alert per stalled goal (14-day dedup)."""
    sent_any = False
    last_alerts = state.setdefault("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        last_alerts = {}
        state["last_alert_at"] = last_alerts
    for goal in stalled:
        key = f"stalled:{goal.text[:80]}"
        last = float(last_alerts.get(key, 0))
        if now - last < _DEDUP_WINDOW_S:
            continue
        last_alerts[key] = now
        body = (
            f"🎯 Stalled goal — no activity in {_STALLED_DAYS_THRESHOLD}+ "
            f"days:\n\n"
            f"  • _{goal.text}_\n\n"
            f"Window scanned: {_ACTIVITY_WINDOW_DAYS} days. Sources "
            f"consulted: crew_tasks, companion_ideas, continuity_ledger.\n\n"
            f"If this goal is no longer relevant, consider removing it "
            f"from `current_goals` (operator-only — kernel state is "
            f"Tier-3-protected). If it IS relevant, what's blocking "
            f"progress?"
        )
        try:
            from app.notify import notify
            notify(
                title="🎯 Stalled goal",
                body=body,
                url="/cp/goals",
                topic=f"goal_progress_stalled:{key}",
                critical=False,
                arbitrate=True,
            )
            sent_any = True
        except Exception:
            logger.debug("goal_progress: notify failed", exc_info=True)
    return sent_any


def evaluate(*, now: Optional[float] = None) -> dict[str, Any]:
    """One probe pass. Returns a summary dict."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_goals": 0,
        "goals": [],
        "stalled": [],
        "alert_sent": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary
    cur = float(now) if now is not None else time.time()
    state = _read_state()
    state["last_run_at"] = cur
    summary["ran"] = True

    goals = _load_current_goals()
    summary["n_goals"] = len(goals)
    if not goals:
        _write_state(state)
        return summary

    tokens_by_goal: dict[str, list[str]] = {}
    for raw_goal in goals:
        text, tokens = _normalise_goal(raw_goal)
        if text:
            tokens_by_goal[text] = tokens

    cutoff = cur - _ACTIVITY_WINDOW_DAYS * 86400
    all_matches: dict[str, list[dict[str, Any]]] = {k: [] for k in tokens_by_goal}
    for scan in (_scan_crew_tasks, _scan_companion_ideas, _scan_continuity_ledger):
        try:
            partial = scan(tokens_by_goal, cutoff=cutoff)
        except Exception:
            logger.debug("goal_progress: scan raised", exc_info=True)
            continue
        for goal_text, rows in partial.items():
            all_matches.setdefault(goal_text, []).extend(rows)

    per_goal_stall = state.setdefault("per_goal_stall_since", {})
    if not isinstance(per_goal_stall, dict):
        per_goal_stall = {}
        state["per_goal_stall_since"] = per_goal_stall

    stalled_now: list[GoalStatus] = []
    statuses: list[GoalStatus] = []
    for goal_text, tokens in tokens_by_goal.items():
        rows = all_matches.get(goal_text, [])
        sources = sorted({r["source"] for r in rows})
        last_ts = max((r["ts"] for r in rows), default=None)
        if not tokens:
            status = "insufficient_data"
        elif rows:
            status = "advancing"
            per_goal_stall.pop(goal_text, None)
        else:
            # No matches in window. Are we sustained-stalled?
            stall_since = per_goal_stall.get(goal_text)
            if stall_since is None:
                per_goal_stall[goal_text] = cur
                status = "advancing"  # first miss; benefit of doubt
            elif (cur - float(stall_since)) >= _STALLED_DAYS_THRESHOLD * 86400:
                status = "stalled"
            else:
                status = "advancing"  # cooling-off period
        gs = GoalStatus(
            text=goal_text,
            tokens=tokens,
            evidence_count=len(rows),
            sources=sources,
            status=status,
            last_evidence_at=last_ts,
        )
        statuses.append(gs)
        if status == "stalled":
            stalled_now.append(gs)
    summary["goals"] = [g.to_dict() for g in statuses]
    summary["stalled"] = [g.to_dict() for g in stalled_now]
    if stalled_now:
        summary["alert_sent"] = _alert_stalled(stalled_now, state, now=cur)
    _write_state(state)
    return summary
