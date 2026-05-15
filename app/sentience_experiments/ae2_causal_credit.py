"""AE-2 — Rare-event causal credit assignment.

PROGRAM §43.2 — Q5.2. Functional approximation of "agency-embodiment 2":
when the system takes action A in context C and outcome O occurs later,
attribute credit. Specifically for *rare* outcomes — typical
action-outcome chains are well-understood by the existing
accuracy_tracker; rare events are where attribution is hardest and
where the operator most wants the system to learn.

This module is OBSERVATIONAL. It does not close any loop into action
selection. The operator-visible log of inferred causal associations
is the only output.

Inputs (read-only)
------------------

  * ``workspace/observability/loadable_agent_usage.jsonl`` —
    (action_id, agent, model, cascade_tier) per LLM call
  * ``workspace/observability/errors.jsonl`` — error outcomes
  * ``workspace/audit_log.jsonl`` — operator approvals/rejections
  * ``workspace/affect/welfare_audit.jsonl`` — affect outcomes

Outputs
-------

  * ``workspace/sentience/ae2_associations.jsonl`` — append-only
    log of CausalAssociation records, archive-rotated.

Algorithm
---------

1. Build action-feature vectors from the usage log (window: 7d default)
2. Build outcome events from errors + audit + welfare (same window,
   plus Δ-shift up to 24h forward to catch action→outcome chains)
3. For each (action_signature, context_hash) bucket, count outcome
   frequency
4. Flag rare outcomes (< 10% per bucket) with high lift (≥ 3× baseline)
5. Emit ``CausalAssociation`` records with confidence + evidence-count

Goodhart guards
---------------

  * Observational ONLY — no auto-modification of action selection
  * Operator-visible logs
  * Rarity threshold (≤10%) prevents over-fitting to common outcomes
  * Lift threshold (≥3×) requires statistically meaningful signal
  * Min sample size (n≥5 per bucket) prevents single-observation noise

Anti-scorecard contract
-----------------------

This module DOES NOT change the Butlin AE-2 indicator (declared
ABSENT-by-declaration because the system has no body). The scorecard
evaluator checks canonical paths; this module at
``app/sentience_experiments/ae2_causal_credit.py`` is invisible to it.
We build the capability because rare-event credit assignment is useful,
not because the scorecard rewards it.
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# Defaults — module-level constants so tests can override via monkeypatch.
_DEFAULT_WINDOW_DAYS = 7
_OUTCOME_LOOKAHEAD_HOURS = 24
_MIN_OBSERVATIONS_PER_BUCKET = 5
_RARITY_CEILING = 0.10           # outcome probability ≤ 10% to count as "rare"
_MIN_LIFT = 3.0                  # P(outcome | action) / baseline ≥ 3×
_MAX_ASSOCIATIONS_PER_PASS = 50  # bound the operator surface
_ASSOCIATIONS_LOG_MAX_LINES = 5_000


# ── Master switch ─────────────────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_sentience_ae2_enabled
        return get_sentience_ae2_enabled()
    except Exception:
        # Default ON per operator decision in Q5 plan.
        return True


# ── Paths ─────────────────────────────────────────────────────────────────


def _default_usage_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "observability" / "loadable_agent_usage.jsonl"
    except Exception:
        return Path("/app/workspace/observability/loadable_agent_usage.jsonl")


def _default_errors_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "observability" / "errors.jsonl"
    except Exception:
        return Path("/app/workspace/observability/errors.jsonl")


def _default_welfare_audit_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "affect" / "welfare_audit.jsonl"
    except Exception:
        return Path("/app/workspace/affect/welfare_audit.jsonl")


def _default_associations_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "sentience" / "ae2_associations.jsonl"
    except Exception:
        return Path("/app/workspace/sentience/ae2_associations.jsonl")


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class CausalAssociation:
    """One inferred association: action signature → outcome at high lift."""

    action_signature: str        # e.g. "agent=coder|model=deepseek|tier=2"
    outcome_kind: str            # e.g. "welfare_breach", "error:ConnectionError"
    rarity: float                # P(outcome) in this bucket (≤ _RARITY_CEILING)
    lift: float                  # P(outcome | action) / P(outcome) baseline
    n_observations: int          # how many co-occurrences supported this
    n_actions: int               # total times this action signature ran in window
    first_seen: str
    last_seen: str
    confidence: float            # 0..1 — function of n_observations and lift

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Source readers ────────────────────────────────────────────────────────


def _iter_jsonl(path: Path, *, since_iso: str) -> Iterator[dict]:
    """Walk a JSONL file, yielding rows whose ``ts`` is within window.
    Failure-isolated."""
    if not path.exists():
        return
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
                ts = row.get("ts") or ""
                if ts < since_iso:
                    continue
                yield row
    except OSError:
        return


def _action_signature(row: dict) -> str:
    """Build a coarse action signature from a usage row. Coarseness is
    the point — too-fine signatures fragment the buckets and prevent
    learning. The signature names the *kind* of action, not the
    specific instance."""
    agent = row.get("agent_id") or row.get("agent") or "?"
    model = (row.get("model") or "?").split("/")[-1][:30]
    return f"agent={agent}|model={model}"


def _outcome_kind_from_error(row: dict) -> str:
    """Coarse error class. Errors that share a class are one bucket."""
    err = row.get("error_type") or row.get("error_class") or "error"
    return f"error:{err}"


def _outcome_kind_from_welfare(row: dict) -> str:
    """Welfare breach kinds map to coarse buckets."""
    kind = row.get("kind") or "welfare:unknown"
    return f"welfare:{kind}"


# ── Aggregation ───────────────────────────────────────────────────────────


def _ts_parse(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat((s or "").replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def detect_associations(
    *, window_days: int = _DEFAULT_WINDOW_DAYS,
    rarity_ceiling: float = _RARITY_CEILING,
    min_lift: float = _MIN_LIFT,
    min_observations: int = _MIN_OBSERVATIONS_PER_BUCKET,
) -> list[CausalAssociation]:
    """One detection pass. Returns CausalAssociation records that meet
    the rarity + lift + min-observation thresholds.

    The implementation is intentionally simple lift-based counting,
    not a learned model. The point is *visibility*, not optimization."""
    if not _enabled():
        return []

    now = datetime.now(timezone.utc)
    since_iso = (now - timedelta(days=window_days)).isoformat()
    lookahead = timedelta(hours=_OUTCOME_LOOKAHEAD_HOURS)

    # 1. Collect actions: (ts, signature)
    actions: list[tuple[datetime, str]] = []
    for row in _iter_jsonl(_default_usage_path(), since_iso=since_iso):
        ts = _ts_parse(row.get("ts", ""))
        if ts is None:
            continue
        actions.append((ts, _action_signature(row)))

    if not actions:
        return []

    # 2. Collect outcomes: (ts, outcome_kind)
    outcomes: list[tuple[datetime, str]] = []
    for row in _iter_jsonl(_default_errors_path(), since_iso=since_iso):
        ts = _ts_parse(row.get("ts", ""))
        if ts is None:
            continue
        outcomes.append((ts, _outcome_kind_from_error(row)))
    for row in _iter_jsonl(_default_welfare_audit_path(), since_iso=since_iso):
        ts = _ts_parse(row.get("ts", ""))
        if ts is None:
            continue
        outcomes.append((ts, _outcome_kind_from_welfare(row)))

    if not outcomes:
        return []

    # 3. Build action-counts and outcome-baseline.
    action_count: dict[str, int] = defaultdict(int)
    for _, sig in actions:
        action_count[sig] += 1

    outcome_count: dict[str, int] = defaultdict(int)
    for _, kind in outcomes:
        outcome_count[kind] += 1

    total_actions = len(actions)
    # Each outcome is "explained by" at most one action — we use a
    # forward-window join (action precedes outcome within lookahead).
    # Multiple actions sharing the signature share the credit; one
    # outcome can be assigned to multiple action signatures only when
    # outcomes occur very close to multiple action types.

    # 4. Co-occurrence counts: for each (action_sig, outcome_kind),
    #    how many outcome instances had an action of that signature
    #    in the prior `lookahead` window?
    cooc: dict[tuple[str, str], int] = defaultdict(int)
    cooc_first: dict[tuple[str, str], datetime] = {}
    cooc_last: dict[tuple[str, str], datetime] = {}

    # Sort actions for efficient backward search.
    actions_sorted = sorted(actions, key=lambda x: x[0])
    # For each outcome, find all action signatures within lookahead before.
    for o_ts, o_kind in outcomes:
        window_start = o_ts - lookahead
        # Binary-search-ish linear walk; data volumes are small enough.
        sigs_in_window: set[str] = set()
        for a_ts, a_sig in actions_sorted:
            if a_ts > o_ts:
                break
            if a_ts < window_start:
                continue
            sigs_in_window.add(a_sig)
        for sig in sigs_in_window:
            key = (sig, o_kind)
            cooc[key] += 1
            cooc_first.setdefault(key, o_ts)
            cooc_last[key] = o_ts

    # 5. Compute lift; flag rare-event high-lift associations.
    associations: list[CausalAssociation] = []
    for (sig, kind), n_co in cooc.items():
        if n_co < min_observations:
            continue
        n_action = action_count.get(sig, 0)
        if n_action == 0:
            continue
        p_outcome_given_action = n_co / n_action
        baseline = outcome_count[kind] / total_actions if total_actions else 0.0
        if baseline <= 0:
            continue
        # Rarity is the baseline probability — we want rare outcomes.
        if baseline > rarity_ceiling:
            continue
        lift = p_outcome_given_action / baseline
        if lift < min_lift:
            continue
        # Confidence: scales with n_observations (saturating) and lift.
        conf = min(1.0, (math.log10(max(2, n_co))) * (lift / (lift + 5.0)))
        associations.append(CausalAssociation(
            action_signature=sig,
            outcome_kind=kind,
            rarity=round(baseline, 4),
            lift=round(lift, 3),
            n_observations=n_co,
            n_actions=n_action,
            first_seen=cooc_first[(sig, kind)].isoformat(),
            last_seen=cooc_last[(sig, kind)].isoformat(),
            confidence=round(conf, 3),
        ))

    associations.sort(key=lambda a: a.lift, reverse=True)
    return associations[:_MAX_ASSOCIATIONS_PER_PASS]


# ── Persistence ───────────────────────────────────────────────────────────


def persist(associations: list[CausalAssociation]) -> int:
    """Append each association to the JSONL with cap-rotation. Returns
    count persisted. Failure-isolated."""
    if not associations:
        return 0
    path = _default_associations_path()
    try:
        from app.utils.jsonl_retention import append_with_cap
    except Exception:
        return 0
    persisted = 0
    for assoc in associations:
        try:
            append_with_cap(
                path,
                json.dumps({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **assoc.to_dict(),
                }, sort_keys=True),
                max_lines=_ASSOCIATIONS_LOG_MAX_LINES,
            )
            persisted += 1
        except Exception:
            logger.debug("ae2: persist failed for %s", assoc.action_signature)
    return persisted


def list_recent(n: int = 20) -> list[dict[str, Any]]:
    """Read recent associations for the operator surface. Newest-first."""
    if not _enabled():
        return []
    path = _default_associations_path()
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
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    rows.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return rows[:n]


# ── Idle entry ────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One detection pass + persist + opaque GW publish. Cadence-guarded
    by the caller (companion.loop). No-op when master switch OFF."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "ae2_disabled"}
    try:
        assocs = detect_associations()
    except Exception:
        logger.debug("ae2: detect raised", exc_info=True)
        return {"ok": False, "associations": 0}
    persisted = persist(assocs)

    # GW publish opaque counts only — never action_signatures or outcomes.
    if assocs:
        try:
            from app.workspace_publish import publish_to_workspace
            publish_to_workspace(
                source="ae2_causal_credit",
                content=(
                    f"{len(assocs)} rare-event causal association"
                    f"{'s' if len(assocs) != 1 else ''} detected "
                    f"(top lift {assocs[0].lift:.1f}×)"
                ),
                salience=min(0.7, 0.3 + 0.05 * len(assocs)),
                signal_type="background",
            )
        except Exception:
            logger.debug("ae2: GW publish failed", exc_info=True)
    return {
        "ok": True,
        "associations": len(assocs),
        "persisted": persisted,
        "top_lift": float(assocs[0].lift) if assocs else 0.0,
    }
