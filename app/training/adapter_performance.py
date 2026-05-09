"""Adapter retirement — performance arm (Phase C #1, 2026-05-09).

The Wave 2 ``adapter_lifecycle`` module handles disk hygiene:
orphan cleanup, dead-pointer detection, bloat alerts. It does NOT
grade adapters on quality. Result: a long-tail of low-eval-score
adapters could sit in the registry forever, slowing the ensemble
and consuming runtime memory.

This module adds the missing performance arm:

  1. Snapshot ``workspace/training_adapters/registry.json`` weekly.
     Each snapshot is appended to
     ``workspace/training/adapter_quality_history.jsonl`` —
     append-only, so we have a long timeseries of ``eval_score``
     per adapter.
  2. Compute a ``health_score`` per adapter:
        health = (eval_score / QUALITY_GATE)
                 × age_decay(days_since_created)
        age_decay(d) = 1.0 for d ≤ 30, then linear decay to 0.5 at 180 d
     Adapters with health < ``RETIREMENT_THRESHOLD`` (default 0.6)
     are candidates.
  3. Cross-check the meta-agent recipe-outcome ledger (Phase A): if
     any recipe whose tools include this adapter has a 7-day
     win-rate < 0.5, that's a strong signal — escalate the
     candidate.
  4. For each candidate, append a proposal to
     ``workspace/training/retirement_proposals.jsonl``::

        {ts, adapter_name, health_score, reason,
         suggested_action: "retire" | "re-train" | "deprecate"}

     and emit a Signal alert. Operator decides; this module never
     mutates the registry directly (registry.json is the source of
     truth and edits go through the same change-request gate as code).

Cadence: 7 days. Adapter quality moves slowly; weekly is fast enough.

Master switch: ``ADAPTER_PERFORMANCE_ENABLED`` (default ON).
``RETIREMENT_THRESHOLD`` env var tunes the threshold per-deploy.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


_REGISTRY_PATH = Path("/app/workspace/training_adapters/registry.json")
_HISTORY_PATH = Path("/app/workspace/training/adapter_quality_history.jsonl")
_PROPOSALS_PATH = Path("/app/workspace/training/retirement_proposals.jsonl")
_STATE_FILE = "adapter_performance.json"
_RUN_CADENCE_S = 7 * 24 * 3600

_DEFAULT_THRESHOLD = 0.6
_DEFAULT_QUALITY_GATE = 0.75


def _enabled() -> bool:
    return os.getenv("ADAPTER_PERFORMANCE_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _retirement_threshold() -> float:
    raw = os.getenv("RETIREMENT_THRESHOLD", str(_DEFAULT_THRESHOLD)).strip()
    try:
        return max(0.1, min(0.95, float(raw)))
    except ValueError:
        return _DEFAULT_THRESHOLD


def _quality_gate() -> float:
    """The training-pipeline's QUALITY_GATE constant. 0.75 by default."""
    try:
        from app.training_pipeline import QUALITY_GATE
        return float(QUALITY_GATE)
    except Exception:
        return _DEFAULT_QUALITY_GATE


# ── Snapshot ──────────────────────────────────────────────────────────────


def _read_registry() -> dict[str, dict[str, Any]]:
    """Return the registry dict. Empty on any failure."""
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("adapter_performance: registry unreadable", exc_info=True)
        return {}


_HISTORY_MAX_LINES = 5000  # Phase F #7: ~50 weeks × 100 adapters


def _snapshot_to_history(registry: dict[str, dict]) -> int:
    """Append one history row per adapter. Returns count appended."""
    if not registry:
        return 0
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    written = 0
    try:
        with _HISTORY_PATH.open("a", encoding="utf-8") as f:
            for name, info in registry.items():
                row = {
                    "ts": ts, "adapter": name,
                    "eval_score": float(info.get("eval_score", 0.0) or 0.0),
                    "examples_count": int(info.get("examples_count", 0) or 0),
                    "promoted": bool(info.get("promoted", False)),
                    "created_at": info.get("created_at", ""),
                }
                f.write(json.dumps(row, sort_keys=True))
                f.write("\n")
                written += 1
    except OSError:
        logger.debug("adapter_performance: history write failed", exc_info=True)
    # Phase F #7: cap retention so the JSONL doesn't grow unbounded.
    try:
        from app.utils.jsonl_retention import cap_jsonl
        cap_jsonl(_HISTORY_PATH, _HISTORY_MAX_LINES)
    except Exception:
        logger.debug("adapter_performance: history cap failed", exc_info=True)
    return written


# ── Health score ─────────────────────────────────────────────────────────


def _age_decay(days: float) -> float:
    """Decay factor: 1.0 for d≤30, linear to 0.5 at 180, floored at 0.3 past 365."""
    if days <= 30:
        return 1.0
    if days >= 365:
        return 0.3
    # Linear from 1.0 at d=30 to 0.5 at d=180; then linear from 0.5 to
    # 0.3 between 180 and 365.
    if days <= 180:
        return max(0.3, 1.0 - (days - 30) * (1.0 - 0.5) / (180 - 30))
    return max(0.3, 0.5 - (days - 180) * (0.5 - 0.3) / (365 - 180))


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _adapter_health(info: dict, now: datetime, quality_gate: float) -> dict:
    """Compute a health score + reasons for one adapter."""
    eval_score = float(info.get("eval_score", 0.0) or 0.0)
    created_at = _parse_iso(info.get("created_at", ""))
    days = (now - created_at).days if created_at else 0
    quality_ratio = eval_score / quality_gate if quality_gate > 0 else 0.0
    decay = _age_decay(days)
    score = round(min(1.5, quality_ratio * decay), 3)
    return {
        "health_score": score, "eval_score": eval_score,
        "days_since_created": days, "age_decay": round(decay, 3),
        "quality_ratio": round(quality_ratio, 3),
    }


# ── Recipe-ledger cross-check ────────────────────────────────────────────


def _recipe_winrate_for_adapter(adapter_name: str, days: int = 7) -> Optional[float]:
    """Fetch the recent win-rate across recipes mentioning this adapter.

    Phase F #2 (2026-05-09): the earlier implementation iterated
    ``RecipeOutcome.tool_names`` — a field that doesn't exist on
    ``RecipeOutcome`` (it lives on ``AgentRecipe.extra_tool_names``).
    The check therefore always returned None and the cross-signal
    was dead.

    Correct shape:

      1. ``list_recipes`` to find every recipe whose
         ``extra_tool_names`` mentions ``adapter_name``.
      2. For each matching recipe, walk its outcomes (the recipe-id
         join restores the (recipe, outcome) pairing).
      3. Roll up to a single win-rate over the lookback window.

    Returns None when PG is unavailable, when no recipe mentions the
    adapter, or when no outcomes fall in the window — in all three
    cases the caller treats it as "no signal" and ignores.
    """
    try:
        from app.self_improvement.meta_agent.store import (
            list_outcomes, list_recipes,
        )
    except Exception:
        return None

    try:
        all_recipes = list_recipes(limit=500)
    except Exception:
        return None

    matching_recipe_ids: set[str] = set()
    for r in all_recipes or []:
        tools = getattr(r, "extra_tool_names", None) or []
        if any(adapter_name in str(t) for t in tools):
            rid = getattr(r, "id", None) or getattr(r, "recipe_id", None)
            if rid:
                matching_recipe_ids.add(str(rid))
    if not matching_recipe_ids:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total = 0
    success = 0
    for rid in matching_recipe_ids:
        try:
            outcomes = list_outcomes(rid, limit=200)
        except Exception:
            continue
        for o in outcomes or []:
            recorded = (
                getattr(o, "recorded_at", "")
                if not isinstance(o, dict) else o.get("recorded_at", "")
            )
            ts = _parse_iso(recorded)
            if ts is None or ts < cutoff:
                continue
            total += 1
            success_flag = (
                getattr(o, "success", False)
                if not isinstance(o, dict) else bool(o.get("success", False))
            )
            if success_flag:
                success += 1
    if total == 0:
        return None
    return success / total


# ── Proposal writer ──────────────────────────────────────────────────────


_PROPOSALS_MAX_LINES = 1000  # Phase F #7


def _write_proposal(adapter_name: str, health: dict, reason: str,
                    action: str) -> None:
    """Append a retirement proposal to the JSONL log."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "adapter_name": adapter_name,
        "health_score": health["health_score"],
        "eval_score": health["eval_score"],
        "days_since_created": health["days_since_created"],
        "age_decay": health["age_decay"],
        "reason": reason, "suggested_action": action,
    }
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            _PROPOSALS_PATH, json.dumps(row, sort_keys=True),
            _PROPOSALS_MAX_LINES,
        )
    except Exception:
        logger.debug("adapter_performance: proposal append failed", exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally."""
    summary: dict[str, Any] = {
        "ran": False, "snapshotted": 0, "candidates": 0, "alerted": False,
    }
    if not _enabled():
        return summary

    try:
        from app.healing.handlers._common import (
            audit_event, read_state_json, send_signal_alert, write_state_json,
        )
    except Exception:
        logger.debug("adapter_performance: helpers unavailable", exc_info=True)
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    registry = _read_registry()
    summary["snapshotted"] = _snapshot_to_history(registry)
    if not registry:
        write_state_json(_STATE_FILE, state)
        return summary

    threshold = _retirement_threshold()
    quality_gate = _quality_gate()
    now = datetime.now(timezone.utc)

    candidates: list[tuple[str, dict, str, str]] = []
    for name, info in registry.items():
        health = _adapter_health(info, now, quality_gate)
        reasons: list[str] = []
        action = "retire"

        if health["eval_score"] < quality_gate * 0.85:
            reasons.append(
                f"eval_score {health['eval_score']:.2f} below 85% of "
                f"QUALITY_GATE ({quality_gate:.2f})"
            )
        if health["days_since_created"] >= 90 and health["eval_score"] < quality_gate:
            reasons.append(
                f"old ({health['days_since_created']}d) AND under quality gate"
            )
            action = "deprecate"

        # Recipe-ledger cross-check.
        winrate = _recipe_winrate_for_adapter(name)
        if winrate is not None and winrate < 0.5:
            reasons.append(
                f"recipe-ledger 7d win-rate {winrate:.2f} < 0.50"
            )
            action = "re-train" if health["health_score"] >= threshold else "retire"

        if health["health_score"] >= threshold and not reasons:
            continue  # healthy, skip

        if not reasons:
            reasons.append(f"composite health_score {health['health_score']:.2f} below threshold {threshold:.2f}")

        reason_str = "; ".join(reasons)
        _write_proposal(name, health, reason_str, action)
        candidates.append((name, health, reason_str, action))

    summary["candidates"] = len(candidates)
    write_state_json(_STATE_FILE, state)

    if candidates:
        lines = [
            f"🧬 Adapter performance: {len(candidates)} retirement "
            f"candidate(s) in workspace/training_adapters/registry.json:",
            "",
        ]
        for name, health, reason, action in candidates[:10]:
            lines.append(
                f"  • `{name}` — health {health['health_score']:.2f} "
                f"(eval={health['eval_score']:.2f}, age={health['days_since_created']}d) "
                f"→ {action}"
            )
            lines.append(f"    {reason}")
        lines.append("")
        lines.append(
            "Proposals written to "
            "`workspace/training/retirement_proposals.jsonl`. "
            "Operator decides; no auto-retirement."
        )
        try:
            send_signal_alert("\n".join(lines), tag="adapter_performance")
            summary["alerted"] = True
        except Exception:
            logger.debug("adapter_performance: alert send failed", exc_info=True)

    audit_event(
        "adapter_performance_pass",
        snapshotted=summary["snapshotted"],
        candidates=summary["candidates"],
        alerted=summary["alerted"],
    )
    return summary
