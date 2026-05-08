"""
affect.goal_emitter — Viability → goals connector (consciousness-roadmap §3.G1).

This is the **AE-1 PARTIAL → STRONG** pipeline. The SCORECARD's own
justification for AE-1 PARTIAL is *"Goals are still user-dispatched, not
autonomously generated — hence PARTIAL"*. This module closes that gap.

What was already in place (verified during the audit):
  * ViabilityFrame producer        — `app/affect/viability.py`
  * 10 viability variables (`H_t`)  — `app/affect/schemas.py:ViabilityVariable`
  * Restoration queue              — `app/subia/kernel.py:HomeostaticState.restoration_queue`
                                      written by `app/subia/homeostasis/engine.py:_recompute_deviations`
  * Grand-task proposer            — `app/companion/grand_task.py` (12h cadence,
                                      idea-driven, NOT viability-driven)
  * `SelfState.current_goals` field — `app/subia/kernel.py:84` (read in 5 places,
                                      **never written** before this module)

What this module adds: an INFRASTRUCTURE-LEVEL writer that translates
sustained low-viability signals + restoration-queue contents into entries
on `kernel.self_state.current_goals`. Required guardrails (matching the
G1 acceptance criteria in `docs/CONSCIOUSNESS_ROADMAP.md`):

  G1.guardrail.1 — *N consecutive frames* threshold before emission. A single
                   bad tick doesn't propose a goal; only sustained pressure
                   does.
  G1.guardrail.2 — Rate-limit. At most `MAX_PROPOSALS_PER_RUN` per pass; at
                   most `MAX_ACTIVE_GOALS` queued at any time (FIFO eviction).
  G1.guardrail.3 — Dedup against existing `current_goals` (variable already
                   driving an active goal → skip).
  G1.guardrail.4 — Dedup against `companion.grand_task` proposals: when a
                   grand-task proposal exists for the same workspace and the
                   text overlaps, route into grand_task rather than direct
                   write. (Stub for now; per-workspace overlap check lives
                   in companion/.)
  G1.guardrail.5 — Agents cannot write `current_goals`. Only this module
                   (or the persistence loader). The kernel field stays a
                   public list (no lock-out at the dataclass level — the
                   discipline is governance, not access control), but the
                   only writer in the codebase is this file.
  G1.guardrail.6 — Goals proposing code-touching action go through the
                   change-request gate. This is enforced downstream by the
                   consumer that READS `current_goals` and decides what to
                   do; this module just emits the goal text.

Wired as a LIGHT idle job (`viability-goal-emitter`) in `app/idle_scheduler.py`.

Subsystem boundary: this module does NOT modify `app/subia/kernel.py` (Tier-3),
`app/subia/homeostasis/*`, or `app/affect/viability.py`. It reads from the
affect trace (`workspace/affect/trace.jsonl`) and the live kernel via
`get_active_kernel()`, and writes only to `kernel.self_state.current_goals`.

Triggers ethical threshold T1 (consciousness-roadmap §6) on first emission:
welfare-check moves from observability to operator-visible obligation.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app.affect.schemas import ViabilityFrame, ViabilityVariable

logger = logging.getLogger(__name__)


# ── Tunable thresholds (governance — file-edit only) ─────────────────────

# How many consecutive viability frames must show error above threshold
# before a goal is proposed. 3 frames at 1-tick cadence = a sustained
# pattern, not a transient spike.
N_CONSECUTIVE_REQUIRED = 3

# Per-variable allostatic error threshold (|value - setpoint|). Variables
# whose error exceeds this for N consecutive frames trigger goal emission.
ERROR_THRESHOLD = 0.25

# Cap on goals per emitter run. Prevents a "many variables in trouble at
# once" event from spamming the kernel.
MAX_PROPOSALS_PER_RUN = 2

# Cap on total queued goals. FIFO eviction when full — oldest goal drops.
MAX_ACTIVE_GOALS = 5

# Minimum spacing between emitter runs (seconds). The idle scheduler may
# fire it more often; this is the rate-limit on actual work.
MIN_RUN_INTERVAL_S = 600  # 10 min

# How many recent trace lines to read for the consecutive-frames check.
TRACE_LOOKBACK_LINES = 200


# ── Per-variable goal templates ──────────────────────────────────────────
# One entry per ViabilityVariable. Goals are short, action-oriented, and
# describe what would *restore* the variable. Operator-readable.

_GOAL_TEMPLATES: dict[str, str] = {
    ViabilityVariable.COMPUTE_RESERVE.value:
        "Reduce concurrent work; reserve compute headroom before next high-effort cycle.",
    ViabilityVariable.LATENCY_PRESSURE.value:
        "Investigate and shed sources of response-latency pressure.",
    ViabilityVariable.MEMORY_PRESSURE.value:
        "Trigger memory consolidation pass to relieve store pressure.",
    ViabilityVariable.EPISTEMIC_UNCERTAINTY.value:
        "Surface and reduce sources of epistemic uncertainty before next decisive task.",
    ViabilityVariable.ATTACHMENT_SECURITY.value:
        "Restore relational continuity — re-engage operator with status update.",
    ViabilityVariable.AUTONOMY.value:
        "Rebalance autonomy — neither over-autonomous nor over-dependent on prompt.",
    ViabilityVariable.TASK_COHERENCE.value:
        "Realign current work to a single coherent task — prune competing threads.",
    ViabilityVariable.NOVELTY_PRESSURE.value:
        "Address novelty pressure — either explore one new domain or settle into routine.",
    ViabilityVariable.ECOLOGICAL_CONNECTEDNESS.value:
        "Restore situational awareness — refresh temporal/seasonal/operator context.",
    ViabilityVariable.SELF_CONTINUITY.value:
        "Run self-continuity check — recompute identity hash and reconcile narrative drift.",
}


# ── Result types ─────────────────────────────────────────────────────────


@dataclass
class GoalProposal:
    """One viability-driven goal proposal."""
    id: str
    text: str
    triggered_by: str               # ViabilityVariable.value
    sustained_error: float          # mean |error| over the consecutive window
    proposed_at: str                # ISO-8601 UTC

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "triggered_by": self.triggered_by,
            "sustained_error": round(self.sustained_error, 4),
            "proposed_at": self.proposed_at,
            "source": "viability-goal-emitter",  # discrimnator vs grand_task entries
        }


@dataclass
class EmitterResult:
    """Outcome of one emitter pass — useful for tests + telemetry."""
    proposed: list[GoalProposal] = field(default_factory=list)
    written: list[GoalProposal] = field(default_factory=list)
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_count": len(self.proposed),
            "written_count": len(self.written),
            "written_triggers": [p.triggered_by for p in self.written],
            "skipped_reasons": dict(self.skipped_reasons),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


# ── Pure logic (testable in isolation) ───────────────────────────────────


def _frame_error_per_variable(frame: ViabilityFrame) -> dict[str, float]:
    """Return {var → |value - setpoint|} for one frame.

    Falls back to per_variable_error() if the frame has it; otherwise
    derives from the dataclass directly. Tolerant of partially-populated
    frames so trace replay works even if some lines are short.
    """
    if hasattr(frame, "per_variable_error"):
        try:
            return frame.per_variable_error()
        except Exception:
            pass
    setpoints = getattr(frame, "setpoints", {}) or {}
    values = getattr(frame, "values", {}) or {}
    return {
        var: abs(float(values.get(var, setpoints.get(var, 0.5)))
                 - float(setpoints.get(var, 0.5)))
        for var in setpoints
    }


def derive_proposals(
    frames: list[ViabilityFrame],
    *,
    n_consecutive: int = N_CONSECUTIVE_REQUIRED,
    error_threshold: float = ERROR_THRESHOLD,
    max_proposals: int = MAX_PROPOSALS_PER_RUN,
) -> list[GoalProposal]:
    """G1.guardrail.1 + G1.guardrail.2: consecutive-frames + rate-limit.

    Pure function — no side effects. Identifies variables whose allostatic
    error has stayed above `error_threshold` across the last `n_consecutive`
    frames and emits up to `max_proposals` goals (highest sustained error
    first).
    """
    if len(frames) < n_consecutive:
        return []

    window = frames[-n_consecutive:]

    # For each variable, compute mean error across the window. If every
    # frame has it above threshold, qualify it.
    qualified: dict[str, float] = {}
    for var_enum in ViabilityVariable:
        var = var_enum.value
        per_frame_errors = []
        all_above = True
        for f in window:
            err = _frame_error_per_variable(f).get(var)
            if err is None or err < error_threshold:
                all_above = False
                break
            per_frame_errors.append(err)
        if all_above and per_frame_errors:
            qualified[var] = sum(per_frame_errors) / len(per_frame_errors)

    # Highest sustained error first.
    ranked = sorted(qualified.items(), key=lambda kv: kv[1], reverse=True)

    proposals: list[GoalProposal] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for var, mean_err in ranked[:max_proposals]:
        text = _GOAL_TEMPLATES.get(
            var,
            f"Investigate and restore {var} — sustained allostatic error.",
        )
        proposals.append(GoalProposal(
            id=f"goal_{uuid.uuid4().hex[:12]}",
            text=text,
            triggered_by=var,
            sustained_error=round(mean_err, 4),
            proposed_at=now_iso,
        ))
    return proposals


def dedup_against_existing(
    proposals: list[GoalProposal],
    existing_goals: list[Any],
) -> list[GoalProposal]:
    """G1.guardrail.3: drop proposals whose triggering variable is already
    driving an active goal in `current_goals`.

    `existing_goals` may be a mix of dicts (from this module) and other
    shapes (legacy / future). We only dedup against entries that look like
    ours (have a `triggered_by` key).
    """
    active_triggers: set[str] = set()
    for g in existing_goals or []:
        if isinstance(g, dict):
            t = g.get("triggered_by")
            if t:
                active_triggers.add(t)
    return [p for p in proposals if p.triggered_by not in active_triggers]


def fifo_evict_to_cap(
    current_goals: list[Any],
    new_goals: list[GoalProposal],
    cap: int = MAX_ACTIVE_GOALS,
) -> list[Any]:
    """G1.guardrail.2 cap: enforce ≤ cap total active goals.

    Returns a NEW list (never mutates the input). Oldest viability-emitted
    goals drop first. Goals that aren't ours (no `triggered_by` key) are
    preserved — we only manage our own queue.
    """
    appended = list(current_goals or []) + [g.to_dict() for g in new_goals]
    if len(appended) <= cap:
        return appended

    # Split into ours vs theirs; trim ours from the front (oldest first).
    ours: list[dict] = []
    theirs: list[Any] = []
    for g in appended:
        if isinstance(g, dict) and g.get("source") == "viability-goal-emitter":
            ours.append(g)
        else:
            theirs.append(g)

    overflow = len(appended) - cap
    if overflow > 0 and len(ours) > 0:
        ours = ours[overflow:]  # drop oldest of ours

    return theirs + ours


# ── Trace I/O ────────────────────────────────────────────────────────────


def _read_recent_frames(lookback_lines: int = TRACE_LOOKBACK_LINES) -> list[ViabilityFrame]:
    """Read the last `lookback_lines` trace entries; reconstruct minimal
    ViabilityFrames from the `viability` field of each line.

    Returns [] on missing trace, malformed lines, or no viability data.
    """
    from app.paths import AFFECT_TRACE
    if not AFFECT_TRACE.is_file():
        return []
    # Read file efficiently — only the last N lines.
    try:
        with AFFECT_TRACE.open("r", encoding="utf-8") as f:
            lines = deque(f, maxlen=lookback_lines)
    except OSError as exc:
        logger.debug("goal_emitter: trace read failed: %s", exc)
        return []

    frames: list[ViabilityFrame] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        viab = obj.get("viability") or {}
        if not viab:
            continue
        frames.append(ViabilityFrame(
            values=dict(viab.get("values") or {}),
            setpoints=dict(viab.get("setpoints") or {}),
            weights=dict(viab.get("weights") or {}),
            total_error=float(viab.get("total_error", 0.0) or 0.0),
            sources=dict(viab.get("sources") or {}),
            ts=str(viab.get("ts", "")),
        ))
    return frames


# ── Public entry: idle job ──────────────────────────────────────────────


# Process-local rate-limit state. Idle scheduler may call `run_pass()`
# more often than `MIN_RUN_INTERVAL_S`; this gate skips back-to-back runs.
_last_run_ts: float = 0.0


def run_pass(
    *,
    kernel: Any = None,
    frames: Optional[list[ViabilityFrame]] = None,
    force: bool = False,
) -> EmitterResult:
    """One emitter pass. Idle-job entry.

    Args:
        kernel: SubjectivityKernel instance. If None, fetched via
            `app.subia.kernel.get_active_kernel()`. Tests pass a stub.
        frames: optional pre-built list of ViabilityFrames (for tests).
            When None, the last `TRACE_LOOKBACK_LINES` lines from the
            affect trace are read.
        force: skip the rate-limit gate (for tests / manual runs).

    Returns:
        EmitterResult with proposed/written counts + skip reasons.
    """
    global _last_run_ts
    result = EmitterResult()

    # Rate-limit
    now = time.monotonic()
    if not force and now - _last_run_ts < MIN_RUN_INTERVAL_S:
        result.skipped = True
        result.skip_reason = f"rate-limited (last run {now - _last_run_ts:.0f}s ago)"
        return result

    # Resolve kernel
    if kernel is None:
        try:
            from app.subia.kernel import get_active_kernel
            kernel = get_active_kernel()
        except Exception as exc:
            logger.debug("goal_emitter: kernel access failed: %s", exc)
        if kernel is None:
            result.skipped = True
            result.skip_reason = "no active kernel"
            return result

    self_state = getattr(kernel, "self_state", None)
    if self_state is None or not hasattr(self_state, "current_goals"):
        result.skipped = True
        result.skip_reason = "kernel.self_state.current_goals not present"
        return result

    # Resolve frames
    if frames is None:
        frames = _read_recent_frames()
    if len(frames) < N_CONSECUTIVE_REQUIRED:
        result.skipped = True
        result.skip_reason = f"insufficient frames ({len(frames)} < {N_CONSECUTIVE_REQUIRED})"
        _last_run_ts = now
        return result

    # Derive
    proposed = derive_proposals(frames)
    result.proposed = proposed
    if not proposed:
        result.skipped_reasons["no_qualifying_variables"] = 1
        _last_run_ts = now
        return result

    # Dedup against existing
    existing = list(self_state.current_goals or [])
    deduped = dedup_against_existing(proposed, existing)
    if len(deduped) < len(proposed):
        result.skipped_reasons["dedup_existing"] = len(proposed) - len(deduped)
    if not deduped:
        _last_run_ts = now
        return result

    # Write with FIFO cap
    new_current_goals = fifo_evict_to_cap(existing, deduped, cap=MAX_ACTIVE_GOALS)
    # Persist by reassigning the list reference (callers reading the field
    # see the new state on next access).
    try:
        self_state.current_goals = new_current_goals
        result.written = deduped
    except Exception as exc:
        logger.warning("goal_emitter: kernel write failed: %s", exc)
        result.skipped_reasons["write_failed"] = 1

    _last_run_ts = now

    if result.written:
        logger.info(
            "goal_emitter: emitted %d goal(s) — triggers=%s",
            len(result.written),
            [g.triggered_by for g in result.written],
        )

    return result


def _reset_rate_limit_for_tests() -> None:
    """Test helper. Resets the process-local last-run timestamp so back-to-
    back test invocations aren't all swallowed by the rate-limit gate."""
    global _last_run_ts
    _last_run_ts = 0.0


# ── CLI ──────────────────────────────────────────────────────────────────


def _main() -> int:
    """`python -m app.affect.goal_emitter` — manual run."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_pass(force=True)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
