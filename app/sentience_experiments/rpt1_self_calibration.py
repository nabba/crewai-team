"""RPT-1 — Forward predictions + calibration ledger.

PROGRAM §43.2 — Q5.2. Functional approximation reframed by the user:
the system MAKES predictions about itself + its world, later SCORES
them against outcomes, and tracks CALIBRATION over time. Classic
Brier-score discipline applied at the system level.

Distinct from existing ``app/subia/prediction/accuracy_tracker.py``
which tracks the predictive layer's *internal* per-domain accuracy.
This module tracks predictions the system makes about ITSELF and
ITS OWN governance/operator outcomes:

  * "this Tier-3 amendment will be approved"
  * "this CR will apply within 2h"
  * "this idle job will succeed"
  * "this welfare breach will recover within 10 min"

Three responsibilities
----------------------

1. **Registration** — ``register_prediction(claim_kind, claim_text,
   predicted_p, resolution_at, scorer)`` persists a Forecast row
   to ``workspace/sentience/rpt1_predictions.jsonl``.
2. **Reconciliation** — ``reconcile_due()`` walks unresolved
   forecasts whose ``resolution_at`` has passed, runs the registered
   scorer to determine actual outcome, persists the resolved row.
3. **Calibration aggregation** — ``aggregate_calibration(window_days)``
   computes Brier score + ECE + calibration curve per claim_kind over
   rolling windows. Result persisted to
   ``workspace/sentience/rpt1_calibration_state.json``.

Goodhart guards
---------------

  * Calibration state DOES NOT feed back into the predictive layer
    automatically. That would be a closed loop where the system
    overfits to its own past predictions.
  * The scorer callable is registered alongside the forecast — must
    be a deterministic outcome resolver, not an LLM call (avoids
    self-judging-its-own-predictions Goodhart pattern).
  * Bucket-based aggregation only after ≥10 resolutions per kind
    (single-sample calibration is meaningless).

Anti-scorecard contract
-----------------------

This module does NOT change the Butlin RPT-1 indicator (declared
ABSENT because LLMs are feed-forward at inference). The evaluator
checks for algorithmic recurrence inside ``app/subia/*``; this
module is invisible.
"""
from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_PREDICTIONS_LOG_MAX_LINES = 10_000
_CALIBRATION_WINDOW_DAYS = 30
_MIN_RESOLUTIONS_PER_KIND = 10
_BUCKETS = 10  # 10 calibration buckets over [0.0, 1.0]
# Q5.5 — stale-forecast grace period. After ``resolution_at`` passes
# AND the scorer still returns None for ``_STALE_GRACE_DAYS`` more,
# the forecast is terminated with score_error="stale_unresolved" so
# it doesn't sit in eternal limbo. The reconciler's terminal-error
# short-circuit (Q5.4.1#4) then skips it on subsequent passes.
_STALE_GRACE_DAYS = 60


# ── Master switch ─────────────────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_sentience_rpt1_enabled
        return get_sentience_rpt1_enabled()
    except Exception:
        return True


# ── Paths ─────────────────────────────────────────────────────────────────


def _default_predictions_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "sentience" / "rpt1_predictions.jsonl"
    except Exception:
        return Path("/app/workspace/sentience/rpt1_predictions.jsonl")


def _default_calibration_state_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "sentience" / "rpt1_calibration_state.json"
    except Exception:
        return Path("/app/workspace/sentience/rpt1_calibration_state.json")


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class Forecast:
    """One self-prediction with deferred resolution."""

    id: str
    claim_kind: str             # "tier3_approval" | "cr_apply" | "idle_job_success" | etc.
    claim_text: str             # human-readable description (≤ 200 chars)
    predicted_p: float          # 0..1
    registered_at: str
    resolution_at: str          # ISO ts when scorer should be invoked
    scorer_ref: str             # named scorer (resolved via _resolve_scorer)
    scorer_args: dict[str, Any] = field(default_factory=dict)
    actual: bool | None = None  # populated by reconciler
    resolved_at: str | None = None
    score_error: str | None = None  # set if scorer raised

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Forecast":
        return cls(
            id=d.get("id", ""),
            claim_kind=d.get("claim_kind", ""),
            claim_text=d.get("claim_text", ""),
            predicted_p=float(d.get("predicted_p", 0.0)),
            registered_at=d.get("registered_at", ""),
            resolution_at=d.get("resolution_at", ""),
            scorer_ref=d.get("scorer_ref", ""),
            scorer_args=dict(d.get("scorer_args") or {}),
            actual=d.get("actual"),
            resolved_at=d.get("resolved_at"),
            score_error=d.get("score_error"),
        )


@dataclass
class CalibrationReport:
    """Per-kind aggregate over the window."""

    claim_kind: str
    window_days: int
    n_resolutions: int
    brier_score: float          # mean squared error (lower better)
    ece: float                  # expected calibration error
    bucket_curve: list[dict]    # [{bin_low, bin_high, n, mean_predicted, fraction_actual}]
    last_updated: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Scorer registry ───────────────────────────────────────────────────────


# Named scorers — module-level functions registered here. A forecast
# stores a scorer_ref string; reconcile_due resolves it via this map.
# Using strings (not callables) keeps forecasts JSON-serializable.

_SCORERS: dict[str, Callable[[dict], bool | None]] = {}


def register_scorer(name: str, scorer: Callable[[dict], bool | None]) -> None:
    """Register a named outcome scorer. The scorer takes the forecast's
    ``scorer_args`` dict and returns True (claim came true), False
    (claim was false), or None (outcome cannot yet be determined —
    forecast stays unresolved).

    Q5.4.1 — Refuses callables defined in LLM or agent modules. The
    docstring promise of "deterministic outcome resolver, not an LLM
    call" is now enforced at registration time. Self-judging-its-own-
    predictions is the Goodhart pattern this guard exists to catch."""
    # Inspect the callable's __module__ — refuse anything that lives
    # under app.llm, app.agents, app.crews, or anthropic_*. This
    # catches the obvious mistake; sophisticated subversion still
    # possible but requires deliberate effort.
    mod = (getattr(scorer, "__module__", "") or "").lower()
    forbidden_prefixes = (
        "app.llm",
        "app.agents",
        "app.crews",
        "anthropic",
        "openai",
    )
    for prefix in forbidden_prefixes:
        if mod.startswith(prefix):
            raise ValueError(
                f"rpt1.register_scorer: refused scorer from module "
                f"{mod!r}. Scorers must be deterministic outcome "
                f"resolvers (not LLM calls). Move the scorer to a "
                f"pure-Python module outside {prefix!r}."
            )
    _SCORERS[name] = scorer


def _resolve_scorer(name: str) -> Callable[[dict], bool | None] | None:
    return _SCORERS.get(name)


# Built-in scorers — operator can register more from outside.


def _proposal_state_value(proposal) -> str:
    """Extract a lower-cased state string from a proposal, regardless
    of whether the state attribute is a string, an enum with .value,
    or something else. Returns empty string when state can't be
    resolved. Failure-isolated.

    Q5.6 — the original scorer had a fragile path that pre-computed
    ``.lower()`` outside the enum-aware try/except, which raised
    AttributeError on enum-only state attributes and bubbled up to
    return None for ALL cases. This helper isolates the extraction."""
    state = getattr(proposal, "state", None)
    if state is None:
        return ""
    # Enum with .value
    value = getattr(state, "value", None)
    if isinstance(value, str):
        return value.lower()
    # Plain string
    if isinstance(state, str):
        return state.lower()
    # Fallback: str() representation
    try:
        return str(state).lower()
    except Exception:
        return ""


def _scorer_tier3_approval(args: dict) -> bool | None:
    """Resolve a tier3 amendment forecast.

    args = {"plan_id": str}.
    True if proposal state ∈ {applied, stable}; False if proposal
    reached a terminal non-approval state (including eligibility_failed);
    None if still in flight.

    Q5.6 (PROGRAM §43.6) — eligibility_failed is now a terminal-False
    outcome rather than perpetual-None. The forecast "this amendment
    will be approved" is honestly answered "no, it didn't even reach
    review" when eligibility fails. Without this, eligibility-failed
    proposals sat unresolved for the full 60-day stale-timeout window
    (Q5.5#3) before terminating with score_error — wasting reconciler
    cycles and losing the overconfidence calibration signal."""
    plan_id = args.get("plan_id")
    if not plan_id:
        return None
    try:
        from app.governance_amendment.protocol import load_proposal
        proposal = load_proposal(plan_id)
        if proposal is None:
            return None
        state_val = _proposal_state_value(proposal)
        if state_val in ("applied", "stable"):
            return True
        if state_val in (
            "rejected", "rolled_back", "reverted",
            "eligibility_failed",  # Q5.6 — terminal non-approval
        ):
            return False
        return None
    except Exception:
        return None


def _scorer_cr_apply(args: dict) -> bool | None:
    """Resolve a CR-apply forecast. args = {"cr_id": str}."""
    cr_id = args.get("cr_id")
    if not cr_id:
        return None
    try:
        from app.change_requests.lifecycle import load_request
        cr = load_request(cr_id)
        if cr is None:
            return None
        state = getattr(cr, "state", None)
        sv = (state.value if hasattr(state, "value") else str(state)).lower()
        if sv == "applied":
            return True
        if sv in ("rejected", "rolled_back", "timeout"):
            return False
        return None
    except Exception:
        return None


# Register built-ins so they're always available.
register_scorer("tier3_approval", _scorer_tier3_approval)
register_scorer("cr_apply", _scorer_cr_apply)


# ── Public API: registration ──────────────────────────────────────────────


def register_prediction(
    *,
    claim_kind: str,
    claim_text: str,
    predicted_p: float,
    resolution_at: datetime | str,
    scorer_ref: str,
    scorer_args: dict | None = None,
) -> Forecast | None:
    """Register a forecast. Returns the Forecast (or None when disabled).

    ``predicted_p`` is clamped to [0.0, 1.0]. ``resolution_at`` accepts
    datetime or ISO string. Failure-isolated."""
    if not _enabled():
        return None
    p = max(0.0, min(1.0, float(predicted_p)))
    if isinstance(resolution_at, datetime):
        res_iso = resolution_at.isoformat()
    else:
        res_iso = str(resolution_at)
    fc = Forecast(
        id=uuid.uuid4().hex[:12],
        claim_kind=str(claim_kind),
        claim_text=str(claim_text)[:200],
        predicted_p=p,
        registered_at=datetime.now(timezone.utc).isoformat(),
        resolution_at=res_iso,
        scorer_ref=str(scorer_ref),
        scorer_args=dict(scorer_args or {}),
    )
    _append_forecast(fc)
    return fc


def _append_forecast(fc: Forecast) -> None:
    path = _default_predictions_path()
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            path,
            json.dumps(fc.to_dict(), sort_keys=True),
            max_lines=_PREDICTIONS_LOG_MAX_LINES,
        )
    except Exception:
        # Manual append as fallback.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fc.to_dict(), sort_keys=True) + "\n")
        except OSError:
            logger.debug("rpt1: append failed")


def _read_all_forecasts() -> list[Forecast]:
    path = _default_predictions_path()
    if not path.exists():
        return []
    out: list[Forecast] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(Forecast.from_dict(d))
                except (json.JSONDecodeError, KeyError):
                    continue
    except OSError:
        return []
    # Multi-write artifact: same id may have multiple rows (registered +
    # resolved). Keep the LATEST per id (last write wins).
    by_id: dict[str, Forecast] = {}
    for fc in out:
        by_id[fc.id] = fc
    return list(by_id.values())


# ── Reconciliation ────────────────────────────────────────────────────────


def reconcile_due(*, now: datetime | None = None) -> dict[str, Any]:
    """Walk unresolved forecasts past their resolution_at; run scorers;
    append resolved rows. Returns a summary."""
    if not _enabled():
        return {"ok": False, "skipped": True}
    now_dt = now or datetime.now(timezone.utc)
    forecasts = _read_all_forecasts()
    resolved_now = 0
    errors = 0
    for fc in forecasts:
        if fc.actual is not None:
            continue
        # Q5.4.1 — terminal-error short-circuit: a forecast that
        # already has score_error set should NOT be re-scored every
        # pass forever. The original ship set score_error + resolved_at
        # but left actual=None, so the reconciler's actual-is-None
        # check kept selecting these rows. Now they're skipped.
        if fc.score_error:
            continue
        # Parse resolution_at.
        try:
            res_dt = datetime.fromisoformat(fc.resolution_at.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if res_dt > now_dt:
            continue  # not yet due
        scorer = _resolve_scorer(fc.scorer_ref)
        if scorer is None:
            fc.score_error = f"unknown_scorer:{fc.scorer_ref}"
            fc.resolved_at = now_dt.isoformat()
            _append_forecast(fc)
            errors += 1
            continue
        try:
            outcome = scorer(fc.scorer_args)
        except Exception as exc:
            fc.score_error = f"scorer_raised:{type(exc).__name__}"
            fc.resolved_at = now_dt.isoformat()
            _append_forecast(fc)
            errors += 1
            continue
        if outcome is None:
            # Q5.5 — stale-forecast timeout: a forecast that has been
            # past resolution_at for ≥_STALE_GRACE_DAYS days AND whose
            # scorer is still returning None means the underlying
            # entity (proposal / CR) is permanently stuck. Terminate
            # so it doesn't sit in eternal limbo. Reconcile-skip via
            # Q5.4.1#4 picks it up on next pass.
            stale_after = res_dt + timedelta(days=_STALE_GRACE_DAYS)
            if now_dt >= stale_after:
                fc.score_error = "stale_unresolved"
                fc.resolved_at = now_dt.isoformat()
                _append_forecast(fc)
                errors += 1
                continue
            # Not yet stale. Leave unresolved; next pass may succeed.
            continue
        fc.actual = bool(outcome)
        fc.resolved_at = now_dt.isoformat()
        _append_forecast(fc)
        resolved_now += 1
    return {
        "ok": True,
        "resolved_now": resolved_now,
        "errors": errors,
        "checked": sum(1 for f in forecasts if f.actual is None),
    }


# ── Calibration aggregation ───────────────────────────────────────────────


def _bucket_for(p: float) -> int:
    """Map probability to bucket index in [0, _BUCKETS-1]."""
    if p >= 1.0:
        return _BUCKETS - 1
    if p < 0.0:
        return 0
    return int(p * _BUCKETS)


def aggregate_calibration(
    *, window_days: int = _CALIBRATION_WINDOW_DAYS,
) -> dict[str, CalibrationReport]:
    """Compute per-kind Brier + ECE + bucket curve over recent
    resolutions. Returns dict[claim_kind, CalibrationReport]."""
    if not _enabled():
        return {}
    forecasts = _read_all_forecasts()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    by_kind: dict[str, list[Forecast]] = {}
    for fc in forecasts:
        if fc.actual is None:
            continue
        if (fc.resolved_at or "") < cutoff:
            continue
        by_kind.setdefault(fc.claim_kind, []).append(fc)

    out: dict[str, CalibrationReport] = {}
    now_iso = datetime.now(timezone.utc).isoformat()
    for kind, items in by_kind.items():
        if len(items) < _MIN_RESOLUTIONS_PER_KIND:
            continue
        # Brier score: mean of (p - actual)²
        brier = sum(
            (fc.predicted_p - (1.0 if fc.actual else 0.0)) ** 2
            for fc in items
        ) / len(items)
        # ECE + bucket curve
        buckets: list[list[Forecast]] = [[] for _ in range(_BUCKETS)]
        for fc in items:
            buckets[_bucket_for(fc.predicted_p)].append(fc)
        ece = 0.0
        curve = []
        for i, bucket in enumerate(buckets):
            n = len(bucket)
            if n == 0:
                curve.append({
                    "bin_low": round(i / _BUCKETS, 2),
                    "bin_high": round((i + 1) / _BUCKETS, 2),
                    "n": 0,
                    "mean_predicted": None,
                    "fraction_actual": None,
                })
                continue
            mean_p = sum(fc.predicted_p for fc in bucket) / n
            frac_actual = sum(1 for fc in bucket if fc.actual) / n
            ece += (n / len(items)) * abs(mean_p - frac_actual)
            curve.append({
                "bin_low": round(i / _BUCKETS, 2),
                "bin_high": round((i + 1) / _BUCKETS, 2),
                "n": n,
                "mean_predicted": round(mean_p, 4),
                "fraction_actual": round(frac_actual, 4),
            })
        out[kind] = CalibrationReport(
            claim_kind=kind,
            window_days=window_days,
            n_resolutions=len(items),
            brier_score=round(brier, 4),
            ece=round(ece, 4),
            bucket_curve=curve,
            last_updated=now_iso,
        )
    return out


def persist_calibration(reports: dict[str, CalibrationReport]) -> None:
    """Write per-kind reports to the state file. Atomic write."""
    path = _default_calibration_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reports": {k: r.to_dict() for k, r in reports.items()},
        }
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        logger.debug("rpt1: persist_calibration failed")


def load_calibration_state() -> dict[str, Any]:
    """Read the persisted calibration state. Empty dict on absence."""
    path = _default_calibration_state_path()
    if not path.exists():
        return {"updated_at": None, "reports": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"updated_at": None, "reports": {}}


# ── Idle entry ────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One pass: reconcile due forecasts + recompute calibration.
    Idle-job entry. No-op when master switch OFF."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "rpt1_disabled"}
    # Snapshot previous state for landmark-detection.
    prev_state = load_calibration_state()
    prev_kinds = set((prev_state.get("reports") or {}).keys())

    rec = reconcile_due()
    reports = aggregate_calibration()
    persist_calibration(reports)

    # Q5.4.2 — landmark emission to the identity continuity ledger
    # when a new claim_kind crosses the min-resolutions threshold
    # for the first time. This is the "the system gained calibration
    # confidence on a new domain" signal annual reflection needs.
    new_kinds = set(reports.keys()) - prev_kinds
    landmark_emitted = False
    if new_kinds:
        try:
            from app.sentience_experiments.ledger_bridge import emit_landmark
            top_new = sorted(new_kinds)[0]
            landmark_emitted = emit_landmark(
                source_module="rpt1_self_calibration",
                landmark_kind="first_calibration",
                summary=(
                    f"RPT-1: first calibration achieved for "
                    f"kind={top_new!r} (n={reports[top_new].n_resolutions}, "
                    f"Brier={reports[top_new].brier_score:.3f})"
                ),
                counts={
                    "new_kinds": len(new_kinds),
                    "total_kinds": len(reports),
                },
            )
        except Exception:
            logger.debug("rpt1: ledger emit failed", exc_info=True)

    return {
        "ok": True,
        "reconcile": rec,
        "kinds_with_calibration": list(reports.keys()),
        "n_kinds": len(reports),
        "new_kinds_this_pass": sorted(new_kinds),
        "ledger_landmark_emitted": landmark_emitted,
    }
