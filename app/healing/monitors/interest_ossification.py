"""interest_ossification — interest-model imbalance/staleness probe.

PROGRAM §49 — Q14.5 (year-2+ resilience §10.5). The companion's
``interest_model.py`` produces a top-30 topic list via
``recency × frequency × source-diversity`` scoring. Over years,
three failure modes emerge:

  1. **Concentrated** — recency × frequency reinforcement causes a
     handful of topics to dominate; everything else falls below
     ``_MIN_FREQ``. The companion's idle contemplation becomes
     monotonic.
  2. **Diffuse** — recency decay outweighs reinforcement; no topic
     accumulates enough mass for the system to know what the user
     cares about. The interest signal becomes noise.
  3. **Ossified** — the top-30 list is stable (low churn) for many
     weeks; the recency half-life and minimum-frequency floor have
     conspired to freeze the list. Genuinely new interests can't
     enter.

This monitor watches all three on weekly cadence.

Metrics computed:

  * **Shannon entropy** of the top-30 score distribution, normalised
    to the maximum (uniform). Concentrated → low entropy ratio
    (≈0.0–0.3). Healthy → middle (≈0.4–0.8). Diffuse → ≈0.9–1.0.
  * **Jaccard overlap** with the previous week's top-30 (after
    converting to sets of topic names). Low overlap = healthy
    churn; very high overlap for ≥4 consecutive weeks = ossified.

Alert routing:

  * Entropy < ``_CONCENTRATED_THRESHOLD`` (0.30) → topic
    ``interest_concentrated`` alert.
  * Entropy > ``_DIFFUSE_THRESHOLD`` (0.90) → topic
    ``interest_diffuse`` alert.
  * Jaccard ≥ ``_OSSIFIED_OVERLAP`` (0.95) for ≥
    ``_OSSIFIED_PERSIST_WEEKS`` (4) consecutive runs → topic
    ``interest_ossified`` alert.

Surfaces in the daily briefing's "📊 Companion interest" section
(composer pulls one-line summary on alert).

Master switch: ``interest_ossification_monitor_enabled`` (default ON).
Cadence: weekly.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


NAME = "interest_ossification"
CADENCE_SECONDS = 7 * 24 * 3600
MASTER_SWITCH_KEY = "interest_ossification_monitor_enabled"

_CONCENTRATED_THRESHOLD = 0.30
_DIFFUSE_THRESHOLD = 0.90
_OSSIFIED_OVERLAP = 0.95
_OSSIFIED_PERSIST_WEEKS = 4
_DEDUP_WINDOW_S = 14 * 86400
_STATE_FILE_NAME = "interest_ossification_state.json"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_interest_ossification_monitor_enabled
        return get_interest_ossification_monitor_enabled()
    except Exception:
        return os.getenv(
            "INTEREST_OSSIFICATION_MONITOR_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _profile_path() -> Path:
    return _workspace() / "companion" / "interest_profile.json"


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {
            "last_run_at": 0.0,
            "last_top30_names": [],
            "consecutive_high_overlap_weeks": 0,
            "last_alert_at": {},
        }
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_run_at": 0.0,
            "last_top30_names": [],
            "consecutive_high_overlap_weeks": 0,
            "last_alert_at": {},
        }


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug(
            "interest_ossification: state write failed", exc_info=True,
        )


def _shannon_entropy_ratio(scores: list[float]) -> float:
    """Return entropy / max_entropy in [0, 1]. 0 = all mass on one
    topic; 1 = uniform distribution."""
    if not scores:
        return 0.0
    total = sum(scores)
    if total <= 0:
        return 0.0
    probs = [s / total for s in scores if s > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs))
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _read_top30() -> list[dict[str, Any]]:
    p = _profile_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    topics = data.get("topics") or []
    return [
        {"name": t["name"], "score": float(t.get("score", 0))}
        for t in topics[:30]
        if "name" in t
    ]


def run(*, now: Optional[float] = None) -> dict[str, Any]:
    """One weekly pass. Returns summary dict.

    Failure-isolated: missing interest_profile.json → skip with
    summary['skipped']=True (no alert)."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_topics": 0,
        "entropy_ratio": 0.0,
        "jaccard_overlap": 0.0,
        "consecutive_high_overlap_weeks": 0,
        "alerts": [],
        "ledger_emitted": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < CADENCE_SECONDS:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    top30 = _read_top30()
    summary["n_topics"] = len(top30)
    if len(top30) < 5:
        # Interest profile too small to draw conclusions.
        _write_state(state)
        return summary

    scores = [t["score"] for t in top30]
    names = [t["name"] for t in top30]
    entropy_ratio = _shannon_entropy_ratio(scores)
    summary["entropy_ratio"] = round(entropy_ratio, 4)

    last_names = list(state.get("last_top30_names") or [])
    jaccard = _jaccard(names, last_names) if last_names else 0.0
    summary["jaccard_overlap"] = round(jaccard, 4)

    # Update ossification streak.
    if last_names and jaccard >= _OSSIFIED_OVERLAP:
        state["consecutive_high_overlap_weeks"] = int(
            state.get("consecutive_high_overlap_weeks", 0)
        ) + 1
    else:
        state["consecutive_high_overlap_weeks"] = 0
    summary["consecutive_high_overlap_weeks"] = state[
        "consecutive_high_overlap_weeks"
    ]
    state["last_top30_names"] = names

    last_alerts = state.setdefault("last_alert_at", {})
    if not isinstance(last_alerts, dict):
        last_alerts = {}
        state["last_alert_at"] = last_alerts

    def _maybe_alert(topic: str, title: str, body: str) -> bool:
        last = float(last_alerts.get(topic, 0))
        if cur - last < _DEDUP_WINDOW_S:
            return False
        try:
            from app.notify import notify
            notify(
                title=title,
                body=body,
                url="/cp/companion",
                topic=topic,
                critical=False,
                arbitrate=True,
            )
            last_alerts[topic] = cur
            return True
        except Exception:
            logger.debug(
                "interest_ossification: notify failed",
                exc_info=True,
            )
            return False

    alerts_fired: list[str] = []
    landmark_summary_parts: list[str] = []

    if entropy_ratio < _CONCENTRATED_THRESHOLD:
        top5 = [t["name"] for t in top30[:5]]
        ok = _maybe_alert(
            "interest_concentrated",
            "📊 Companion interest: concentrated",
            (
                f"The top-30 interest distribution has collapsed to "
                f"a handful of topics (entropy ratio "
                f"{entropy_ratio:.2f}, threshold "
                f"{_CONCENTRATED_THRESHOLD}). Top 5: "
                f"{', '.join(top5)}. Consider whether the recency "
                f"half-life is too short or the diversity bonus is "
                f"too weak."
            ),
        )
        if ok:
            alerts_fired.append("interest_concentrated")
            landmark_summary_parts.append(
                f"concentrated (entropy={entropy_ratio:.2f})"
            )

    if entropy_ratio > _DIFFUSE_THRESHOLD:
        ok = _maybe_alert(
            "interest_diffuse",
            "📊 Companion interest: diffuse / noisy",
            (
                f"The top-30 interest distribution is too uniform "
                f"(entropy ratio {entropy_ratio:.2f} > "
                f"{_DIFFUSE_THRESHOLD}). The companion can't tell "
                f"what's important. Consider whether the diversity "
                f"bonus is too strong or the minimum-frequency "
                f"floor too low."
            ),
        )
        if ok:
            alerts_fired.append("interest_diffuse")
            landmark_summary_parts.append(
                f"diffuse (entropy={entropy_ratio:.2f})"
            )

    if (
        state["consecutive_high_overlap_weeks"] >= _OSSIFIED_PERSIST_WEEKS
    ):
        ok = _maybe_alert(
            "interest_ossified",
            "📊 Companion interest: ossified",
            (
                f"The top-30 list has barely changed in "
                f"{state['consecutive_high_overlap_weeks']} weeks "
                f"(Jaccard overlap {jaccard:.2f} ≥ "
                f"{_OSSIFIED_OVERLAP}). Genuinely new interests "
                f"may not be entering. Consider whether the "
                f"frequency floor needs lowering or recency "
                f"half-life needs shortening."
            ),
        )
        if ok:
            alerts_fired.append("interest_ossified")
            landmark_summary_parts.append(
                f"ossified ({state['consecutive_high_overlap_weeks']}w stable)"
            )

    summary["alerts"] = alerts_fired

    if alerts_fired:
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="interest_ossification",
                actor="interest_ossification_monitor",
                summary=(
                    f"Interest-model imbalance: "
                    + "; ".join(landmark_summary_parts)
                ),
                detail={
                    "entropy_ratio": entropy_ratio,
                    "jaccard_overlap": jaccard,
                    "consecutive_high_overlap_weeks": state[
                        "consecutive_high_overlap_weeks"
                    ],
                    "n_topics": len(top30),
                    "top_5": [t["name"] for t in top30[:5]],
                    "alerts": alerts_fired,
                },
            )
            summary["ledger_emitted"] = True
        except Exception:
            logger.debug(
                "interest_ossification: ledger emit failed", exc_info=True,
            )

    _write_state(state)
    return summary
