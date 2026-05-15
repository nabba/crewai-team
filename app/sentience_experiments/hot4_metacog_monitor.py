"""HOT-4 — Metacognitive monitor on the reasoning chain.

PROGRAM §43.2 — Q5.2. Functional approximation of "higher-order
thought 4" reframed by the user as live introspection on reasoning
steps. Distinct from the existing accuracy_tracker (per-domain
after-the-fact) and from the metacognitive_repair log (error-class
post-hoc). This is *process* metacognition: dimensions the agent
doesn't normally introspect captured live, then summarised daily.

Inputs (live hook)
------------------

  * ``app.tool_runtime.telemetry`` already records per-call
    telemetry into ``workspace/observability/loadable_agent_usage.jsonl``
    with: agent_id, iteration, input_tokens, output_tokens,
    cache_creation_input_tokens, cache_read_input_tokens, model.

Approach
--------

The user said HOT-4 default = ON. To keep that promise without
latency cost we do NOT install a synchronous hook into the telemetry
emit path. Instead the module reads the same JSONL file via a
periodic batch summariser (LIGHT idle), then writes structural
metacog signals per step.

This means: the agent step itself is unaffected. The signal arrives
within one idle cycle. For "live monitoring" the operator can read
the latest signals via the ``/cp/sentience`` surface — current cycle
+ recent history both visible.

Outputs
-------

  * ``workspace/sentience/hot4_reasoning_signals.jsonl``

Signals per step
----------------

  * confidence_proxy:  output_tokens / max(1, input_tokens)
       The intuition is the *opposite* of common sense — agents that
       write a lot of output relative to the input tokens are
       *less* confident (need to reason verbosely). Calibrated per-
       agent.
  * cache_reliance:    cache_read_tokens / (cache_read + cache_create + input)
       High reliance ⇒ the step is leveraging cached structure
       (repetitive task). Low reliance ⇒ novel territory.
  * cascade_jump:      bool — did this call use a higher-tier model
                       than the agent's recent baseline?
  * unusual_score:     0..1 — combined deviation from agent's own
                       baseline over a rolling 100-call window.

Goodhart guards
---------------

  * Observational ONLY — signals are written but NEVER read by
    dispatch logic (verified by ``test_hot4_signals_never_gate_dispatch``
    grep test)
  * No per-step LLM call (cost-bounded)
  * Async-batch summariser, not live hook (latency-bounded)

Anti-scorecard contract
-----------------------

This module does not change the Butlin HOT-4 indicator (declared
ABSENT because LLM activations are dense, not sparse). The
indicator evaluator checks for sparse-coding mechanisms in
``app/subia/*``. This module is invisible.
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_DAYS = 1                 # daily summariser pass
_PER_AGENT_BASELINE_WINDOW = 100         # rolling N calls for baselines
_UNUSUAL_Z_THRESHOLD = 1.5               # signals above 1.5σ flagged
_SIGNALS_LOG_MAX_LINES = 10_000


# ── Master switch ─────────────────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_sentience_hot4_enabled
        return get_sentience_hot4_enabled()
    except Exception:
        return True


# ── Paths ─────────────────────────────────────────────────────────────────


def _default_usage_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "observability" / "loadable_agent_usage.jsonl"
    except Exception:
        return Path("/app/workspace/observability/loadable_agent_usage.jsonl")


def _default_signals_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "sentience" / "hot4_reasoning_signals.jsonl"
    except Exception:
        return Path("/app/workspace/sentience/hot4_reasoning_signals.jsonl")


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class MetacogSignal:
    """One step-level metacognitive signal."""

    ts: str
    agent_id: str
    iteration: int
    model: str
    confidence_proxy: float       # 0..1+ (capped at 5.0 for display)
    cache_reliance: float         # 0..1
    cascade_jump: bool
    unusual_score: float          # 0..1 — combined z-score normalized
    flagged: bool                 # unusual_score ≥ threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Telemetry reader ──────────────────────────────────────────────────────


def _iter_telemetry(window_days: int) -> Iterator[dict]:
    path = _default_usage_path()
    if not path.exists():
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_str = row.get("ts") or ""
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                if ts < cutoff:
                    continue
                yield row
    except OSError:
        return


# ── Signal computation ────────────────────────────────────────────────────


def _confidence_proxy(row: dict) -> float:
    inp = max(1, int(row.get("input_tokens") or 0))
    out = int(row.get("output_tokens") or 0)
    ratio = out / inp
    # Cap to keep the surface readable; very long outputs vs. short
    # inputs are a known signal but unbounded ratios noise up the
    # summariser.
    return min(5.0, round(ratio, 4))


def _cache_reliance(row: dict) -> float:
    inp = int(row.get("input_tokens") or 0)
    cache_read = int(row.get("cache_read_input_tokens") or 0)
    cache_create = int(row.get("cache_creation_input_tokens") or 0)
    total = cache_read + cache_create + inp
    if total <= 0:
        return 0.0
    return round(cache_read / total, 4)


_TIER_RANK = {
    "haiku": 1, "sonnet": 2, "opus": 3,
    "deepseek": 1, "minimax": 2, "gemini": 2,
    "ollama": 0,
}


def _model_rank(model: str) -> int:
    """Coarse tier rank. Unknown models default to 1."""
    m = (model or "").lower()
    for keyword, rank in _TIER_RANK.items():
        if keyword in m:
            return rank
    return 1


def compute_signals_from_rows(rows: list[dict]) -> list[MetacogSignal]:
    """Compute MetacogSignal per row. Builds per-agent baselines for
    unusual_score over a rolling window."""
    out: list[MetacogSignal] = []
    # Per-agent rolling baselines for confidence_proxy and cache_reliance.
    conf_history: dict[str, deque] = defaultdict(
        lambda: deque(maxlen=_PER_AGENT_BASELINE_WINDOW),
    )
    cache_history: dict[str, deque] = defaultdict(
        lambda: deque(maxlen=_PER_AGENT_BASELINE_WINDOW),
    )
    rank_history: dict[str, deque] = defaultdict(
        lambda: deque(maxlen=_PER_AGENT_BASELINE_WINDOW),
    )

    rows_sorted = sorted(rows, key=lambda r: r.get("ts", ""))
    for row in rows_sorted:
        agent = str(row.get("agent_id") or row.get("agent") or "unknown")
        model = str(row.get("model") or "?")
        conf = _confidence_proxy(row)
        cache = _cache_reliance(row)
        rank = _model_rank(model)

        # Compute z-scores against rolling baselines.
        conf_hist = conf_history[agent]
        cache_hist = cache_history[agent]
        rank_hist = rank_history[agent]

        def _z(value: float, history: deque) -> float:
            if len(history) < 5:
                return 0.0
            try:
                mu = statistics.fmean(history)
                sd = statistics.pstdev(history)
            except statistics.StatisticsError:
                return 0.0
            if sd <= 1e-9:
                # Degenerate baseline (zero variance). If the new value
                # equals the mean we say "not unusual" (z=0); if it
                # deviates AT ALL from a perfectly uniform baseline,
                # that IS unusual — saturate to a high z.
                return 10.0 if abs(value - mu) > 1e-9 else 0.0
            return abs(value - mu) / sd

        z_conf = _z(conf, conf_hist)
        z_cache = _z(cache, cache_hist)
        z_rank = _z(rank, rank_hist) if rank_hist else 0.0

        # cascade_jump: rank above the agent's running mean by ≥1 tier.
        cascade_jump = False
        if rank_hist:
            try:
                mean_rank = statistics.fmean(rank_hist)
                if rank > mean_rank + 0.5:
                    cascade_jump = True
            except statistics.StatisticsError:
                pass

        # Combined unusual_score: max of the three z-scores, normalized.
        z_combined = max(z_conf, z_cache, z_rank)
        unusual = min(1.0, z_combined / 5.0)  # 5σ ⇒ 1.0
        flagged = z_combined >= _UNUSUAL_Z_THRESHOLD

        # Append to histories AFTER computing signals (so each call is
        # scored against the prior window, not including itself).
        conf_history[agent].append(conf)
        cache_history[agent].append(cache)
        rank_history[agent].append(rank)

        out.append(MetacogSignal(
            ts=str(row.get("ts") or ""),
            agent_id=agent,
            iteration=int(row.get("iteration") or 0),
            model=model,
            confidence_proxy=conf,
            cache_reliance=cache,
            cascade_jump=cascade_jump,
            unusual_score=round(unusual, 4),
            flagged=flagged,
        ))
    return out


def detect_signals(window_days: int = _DEFAULT_WINDOW_DAYS) -> list[MetacogSignal]:
    """One pass over the telemetry log within window."""
    if not _enabled():
        return []
    rows = list(_iter_telemetry(window_days))
    if not rows:
        return []
    return compute_signals_from_rows(rows)


# ── Persistence ───────────────────────────────────────────────────────────


def persist(signals: list[MetacogSignal]) -> int:
    if not signals:
        return 0
    path = _default_signals_path()
    try:
        from app.utils.jsonl_retention import append_with_cap
    except Exception:
        return 0
    persisted = 0
    for s in signals:
        try:
            append_with_cap(
                path,
                json.dumps(s.to_dict(), sort_keys=True),
                max_lines=_SIGNALS_LOG_MAX_LINES,
            )
            persisted += 1
        except Exception:
            logger.debug("hot4: persist failed")
    return persisted


def list_recent_flagged(
    n: int = 20, *, since_iso: str | None = None,
) -> list[dict[str, Any]]:
    """Read recent FLAGGED signals (unusual_score above threshold)
    for the operator surface. Skips the routine majority.

    Q5.6 (PROGRAM §43.6) — ``since_iso`` filter so callers can bound
    the result to a real time window (e.g. "this week"). Without it,
    a quiet HOT-4 history returns N flagged rows from MONTHS ago,
    which produces misleading prose on operator surfaces that claim
    "this week" but show ancient data."""
    if not _enabled():
        return []
    path = _default_signals_path()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not row.get("flagged"):
                    continue
                if since_iso:
                    ts = row.get("ts") or ""
                    if ts < since_iso:
                        continue
                rows.append(row)
    except OSError:
        return []
    rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return rows[:n]


def _has_recent_hot4_landmark(*, days: int = 7) -> bool:
    """Q5.6 — Return True if a hot4 sentience_observation has been
    emitted to the continuity ledger within the last ``days``.

    Used to gate landmark emission so a sustained-anomaly week
    produces ONE landmark, not seven daily duplicates. Reads the
    canonical continuity ledger (no separate state file needed —
    the ledger IS the source of truth for emitted landmarks).
    Failure-isolated."""
    try:
        from app.identity.continuity_ledger import list_events
    except Exception:
        return False
    cutoff_iso = (
        datetime.now(timezone.utc) - timedelta(days=days)
    ).isoformat()
    try:
        events = list_events(
            since_iso=cutoff_iso, kinds={"sentience_observation"},
        )
    except Exception:
        return False
    for ev in events:
        actor = getattr(ev, "actor", "") or ""
        if actor == "hot4_metacog_monitor":
            return True
    return False


# ── Idle entry ────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One detection pass + persist + opaque GW publish."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "hot4_disabled"}
    try:
        signals = detect_signals()
    except Exception:
        logger.debug("hot4: detect raised", exc_info=True)
        return {"ok": False, "signals": 0}
    flagged = [s for s in signals if s.flagged]
    persisted = persist(signals)

    if flagged:
        try:
            from app.workspace_publish import publish_to_workspace
            publish_to_workspace(
                source="hot4_metacog_monitor",
                content=(
                    f"{len(flagged)} reasoning-chain step"
                    f"{'s' if len(flagged) != 1 else ''} flagged unusual "
                    f"(of {len(signals)} total)"
                ),
                salience=min(0.85, 0.55 + 0.02 * len(flagged)),
                signal_type="disposition",
            )
        except Exception:
            logger.debug("hot4: GW publish failed", exc_info=True)

    # Q5.5 — landmark emission to the identity continuity ledger when
    # ≥5 reasoning-chain steps are flagged unusual in one pass. This
    # is the "sustained anomaly" threshold the Q5.4.2 design memo
    # documented but never implemented; without it, annual reflection
    # is blind to one of four sentience modules. Opaque counts only.
    #
    # Q5.6 — 7-day per-source cooldown. A sustained-anomaly WEEK
    # would otherwise emit 7 daily landmarks for what is conceptually
    # one situation. Unlike AE-2 (where each emission cites a distinct
    # action-outcome pair), HOT-4 doesn't naturally distinguish today's
    # 5-flagged from yesterday's 5-flagged at the ledger level. The
    # cooldown is consulted by reading the continuity ledger for
    # recent hot4 sentience_observation events.
    landmark_emitted = False
    if len(flagged) >= 5 and not _has_recent_hot4_landmark(days=7):
        try:
            from app.sentience_experiments.ledger_bridge import emit_landmark
            landmark_emitted = emit_landmark(
                source_module="hot4_metacog_monitor",
                landmark_kind="sustained_reasoning_anomaly",
                summary=(
                    f"HOT-4: {len(flagged)} flagged reasoning-chain "
                    f"steps in one pass (of {len(signals)} total)"
                ),
                counts={
                    "flagged": len(flagged),
                    "total_signals": len(signals),
                },
            )
        except Exception:
            logger.debug("hot4: ledger emit failed", exc_info=True)

    return {
        "ok": True,
        "signals_total": len(signals),
        "signals_flagged": len(flagged),
        "persisted": persisted,
        "ledger_landmark_emitted": landmark_emitted,
    }
