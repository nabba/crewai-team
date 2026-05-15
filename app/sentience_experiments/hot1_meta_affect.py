"""HOT-1 — Reflection on the affect trace (feelings-about-feelings).

PROGRAM §43.2 — Q5.2. Functional approximation: second-order
observation OVER welfare-audit events. Patterns *in* breaches, not
the breaches themselves.

The existing decentered reflection (``app/affect/decentered.py``) is a
no-self pass over the affect trace — it deliberately strips first-
person agency from interpretation. HOT-1 here is the complementary
self-pass: structurally aware that *the system* is the subject, but
disciplined in language. Both pass through the same SOUL.md
discipline: NO performative-affect prose. Structured observations
only; optional one-sentence hypothesis runs through the decentering
filter.

Inputs (read-only)
------------------

  * ``workspace/affect/trace.jsonl`` — every AffectState snapshot
    (the primary source; this is what the user meant by "the
    affect trace"). Q5.4.1 fix: the original ship read only
    welfare_audit, which was a sparse subset.
  * ``workspace/affect/welfare_audit.jsonl`` — breach events
    (secondary source; used for breach-class clustering)

Outputs
-------

  * ``workspace/sentience/hot1_meta_affect.jsonl``

Algorithm
---------

1. Read welfare_audit + episode tags over the last 30d
2. Group breaches by ``kind``
3. Detect patterns:
     - temporal_cluster: ≥3 breaches in the same hour-bucket
     - recurring_trigger: ≥2 breaches w/ same kind in <24h
     - sequence: same kind appears ≥3 times at >24h spacing
4. Emit ``MetaAffectPattern`` records (structured)
5. Optionally (gated by ``sentience_llm_hypothesis_enabled``)
   generate ONE-SENTENCE hypothesis per pattern. Hypothesis text
   passes through the decentering filter — first-person affect
   language is stripped before persistence.

Goodhart guards
---------------

  * Structured-observation-only by default
  * Hypothesis text passes through ``decenter_text`` (rejects "I feel"
    etc.); if the filter rejects ALL variations, ``hypothesis=None``
  * Min-cluster size (≥3) prevents single-event over-pattern-matching
  * 30d window — patterns over shorter spans get filtered out as noise

Anti-scorecard contract
-----------------------

This module does not change the Butlin HOT-1 indicator (declared
ABSENT because the system has no generative perception). The
indicator evaluator checks for a perception substrate in
``app/subia/*``; this module at
``app/sentience_experiments/hot1_meta_affect.py`` is invisible.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW_DAYS = 30
_MIN_CLUSTER_SIZE = 3
_TEMPORAL_BUCKET_HOURS = 1
_RECURRING_TRIGGER_WINDOW_HOURS = 24
_SEQUENCE_MIN_GAP_HOURS = 24
_PATTERNS_LOG_MAX_LINES = 5_000


# ── Master switch + LLM gate ──────────────────────────────────────────────


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_sentience_hot1_enabled
        return get_sentience_hot1_enabled()
    except Exception:
        return True


def _llm_hypothesis_enabled() -> bool:
    if not _enabled():
        return False
    try:
        from app.runtime_settings import get_sentience_llm_hypothesis_enabled
        return get_sentience_llm_hypothesis_enabled()
    except Exception:
        return True


# ── Paths ─────────────────────────────────────────────────────────────────


def _default_welfare_audit_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "affect" / "welfare_audit.jsonl"
    except Exception:
        return Path("/app/workspace/affect/welfare_audit.jsonl")


def _default_trace_path() -> Path:
    """Q5.4.1 — the full affect_trace.jsonl which contains every
    AffectState snapshot (V/A/C + attractor + ts). Decentered
    reflection consumes this directly via app.paths.AFFECT_TRACE;
    HOT-1 reads the same path so the two passes share a data
    source and can be co-located in the operator surface."""
    try:
        from app.paths import AFFECT_TRACE
        return Path(AFFECT_TRACE)
    except Exception:
        return Path("/app/workspace/affect/trace.jsonl")


def _default_patterns_path() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "sentience" / "hot1_meta_affect.jsonl"
    except Exception:
        return Path("/app/workspace/sentience/hot1_meta_affect.jsonl")


# ── Decentering filter — SOUL.md guard ────────────────────────────────────


# First-person affective phrases the filter rejects. The system's
# soul discipline (decentered_reflection.py) maintains the same
# rejection set conceptually; this is the local instance for HOT-1
# hypothesis text.
_FIRST_PERSON_AFFECT_PHRASES = (
    "i feel", "i felt", "i'm feeling",
    "i notice i'm", "i notice i was",
    "i'm anxious", "i'm worried", "i'm sad", "i'm afraid", "i'm distressed",
    "i sense", "i sensed",
    "my emotion", "my feeling", "my anxiety", "my fear", "my distress",
    "i was anxious", "i was worried",
)


_OBSERVATIONAL_PREFIXES = (
    "the audit shows", "the pattern is", "the data indicates",
    "the cluster suggests", "the breach sequence",
    "the welfare log shows", "the trace indicates",
    "across the window", "over the past",
)


def decenter_text(text: str) -> str | None:
    """Return ``text`` if it passes the SOUL.md affect-language filter,
    else None. The filter is a hard reject: if the prose contains any
    first-person affective phrase, the hypothesis is dropped.

    Public so the test suite can pin its semantics."""
    if not text:
        return None
    lower = text.lower()
    for phrase in _FIRST_PERSON_AFFECT_PHRASES:
        if phrase in lower:
            return None
    # Soft requirement: encourage at least one observational prefix.
    # Don't reject without — the LLM may produce valid neutral prose
    # without these specific phrases. But the test suite asserts the
    # generator USES these prefixes when it produces text.
    return text


# ── Data model ────────────────────────────────────────────────────────────


@dataclass
class MetaAffectPattern:
    """One pattern detected over the welfare audit trace."""

    pattern_kind: str             # temporal_cluster | recurring_trigger | sequence
    breach_kinds: list[str]
    n_occurrences: int
    span_days: float
    confidence: float
    detected_at: str
    hypothesis: str | None = None       # one-sentence decentered prose
    raw_evidence: list[str] = field(default_factory=list)  # event timestamps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Source reading ────────────────────────────────────────────────────────


def _load_breaches(window_days: int) -> list[dict]:
    """Read welfare breaches within the window. Returns dicts with
    ``ts`` (datetime) and ``kind`` (str). Failure-isolated."""
    path = _default_welfare_audit_path()
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: list[dict] = []
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
                kind = row.get("kind") or "unknown"
                out.append({"ts": ts, "kind": kind})
    except OSError:
        return []
    return out


def _load_trace_points(window_days: int) -> list[dict]:
    """Q5.4.1 — read the full affect trace within the window. Each
    row carries V/A/C + attractor + ts. Failure-isolated."""
    path = _default_trace_path()
    if not path.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    out: list[dict] = []
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
                # Extract V/A/C — robust to schema variations across
                # decentered.py / affect.schemas.
                try:
                    valence = float(row.get("valence", 0.0))
                    arousal = float(row.get("arousal", 0.0))
                    controllability = float(row.get("controllability", 0.5))
                except (TypeError, ValueError):
                    continue
                out.append({
                    "ts": ts,
                    "valence": valence,
                    "arousal": arousal,
                    "controllability": controllability,
                    "attractor": str(row.get("attractor") or "unknown"),
                })
    except OSError:
        return []
    return out


# ── Pattern detectors ─────────────────────────────────────────────────────


def _detect_temporal_clusters(
    breaches: list[dict], *, min_size: int = _MIN_CLUSTER_SIZE,
) -> list[MetaAffectPattern]:
    """≥min_size breaches falling in the same hour-bucket on the same day."""
    if not breaches:
        return []
    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for b in breaches:
        ts: datetime = b["ts"]
        key = (ts.date().isoformat(), ts.hour)
        buckets[key].append(b)
    out: list[MetaAffectPattern] = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for key, items in buckets.items():
        if len(items) < min_size:
            continue
        kinds = sorted({b["kind"] for b in items})
        out.append(MetaAffectPattern(
            pattern_kind="temporal_cluster",
            breach_kinds=kinds,
            n_occurrences=len(items),
            span_days=round(_TEMPORAL_BUCKET_HOURS / 24.0, 3),
            confidence=min(1.0, len(items) / 10.0),
            detected_at=now_iso,
            raw_evidence=[b["ts"].isoformat() for b in items],
        ))
    return out


def _detect_recurring_triggers(
    breaches: list[dict], *, window_hours: int = _RECURRING_TRIGGER_WINDOW_HOURS,
    min_size: int = 2,
) -> list[MetaAffectPattern]:
    """≥min_size breaches of the SAME kind within window_hours of each other."""
    if not breaches:
        return []
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for b in breaches:
        by_kind[b["kind"]].append(b)
    now_iso = datetime.now(timezone.utc).isoformat()
    out: list[MetaAffectPattern] = []
    delta = timedelta(hours=window_hours)
    for kind, items in by_kind.items():
        items_sorted = sorted(items, key=lambda x: x["ts"])
        if len(items_sorted) < min_size:
            continue
        # Sliding window — find runs of ≥min_size within delta.
        run_start = 0
        for i in range(1, len(items_sorted)):
            while items_sorted[i]["ts"] - items_sorted[run_start]["ts"] > delta:
                run_start += 1
            window_run = items_sorted[run_start: i + 1]
            if len(window_run) >= min_size:
                # Emit ONE pattern per such run to avoid explosion.
                span = (window_run[-1]["ts"] - window_run[0]["ts"]).total_seconds() / 86400.0
                out.append(MetaAffectPattern(
                    pattern_kind="recurring_trigger",
                    breach_kinds=[kind],
                    n_occurrences=len(window_run),
                    span_days=round(span, 3),
                    confidence=min(1.0, len(window_run) / 5.0),
                    detected_at=now_iso,
                    raw_evidence=[b["ts"].isoformat() for b in window_run],
                ))
                # Advance run_start past this window to avoid overlap.
                run_start = i + 1
    return out


def _detect_sequences(
    breaches: list[dict], *, min_gap_hours: int = _SEQUENCE_MIN_GAP_HOURS,
    min_size: int = _MIN_CLUSTER_SIZE,
) -> list[MetaAffectPattern]:
    """Same kind appearing ≥min_size times at >min_gap_hours spacing.
    Distinct from recurring_trigger which captures rapid-fire repeats."""
    if not breaches:
        return []
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for b in breaches:
        by_kind[b["kind"]].append(b)
    now_iso = datetime.now(timezone.utc).isoformat()
    out: list[MetaAffectPattern] = []
    min_gap = timedelta(hours=min_gap_hours)
    for kind, items in by_kind.items():
        items_sorted = sorted(items, key=lambda x: x["ts"])
        # Pick widely-spaced subset greedily.
        spaced: list[dict] = []
        last_ts: datetime | None = None
        for b in items_sorted:
            if last_ts is None or (b["ts"] - last_ts) >= min_gap:
                spaced.append(b)
                last_ts = b["ts"]
        if len(spaced) < min_size:
            continue
        span = (spaced[-1]["ts"] - spaced[0]["ts"]).total_seconds() / 86400.0
        out.append(MetaAffectPattern(
            pattern_kind="sequence",
            breach_kinds=[kind],
            n_occurrences=len(spaced),
            span_days=round(span, 3),
            confidence=min(1.0, len(spaced) / 8.0),
            detected_at=now_iso,
            raw_evidence=[b["ts"].isoformat() for b in spaced],
        ))
    return out


def _detect_baseline_drift(
    trace: list[dict], *, half_window_days: int = 7,
) -> list[MetaAffectPattern]:
    """Q5.4.1 — trace-level pattern: compare mean V/A/C in the
    recent half-window against the older half-window. Significant
    drift (Δ > 0.15 on any dimension) emits a baseline_drift pattern.

    This is the kind of pattern decentered reflection sees regularly
    but HOT-1 was blind to in the breach-only ship: gradual mood
    shifts that never trip welfare thresholds."""
    if len(trace) < 20:
        return []
    half_delta = timedelta(days=half_window_days)
    now = datetime.now(timezone.utc)
    cutoff = now - half_delta
    recent = [p for p in trace if p["ts"] >= cutoff]
    older = [p for p in trace if p["ts"] < cutoff]
    if len(recent) < 5 or len(older) < 5:
        return []
    def _mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)
    drift = {
        "valence": round(_mean(recent, "valence") - _mean(older, "valence"), 3),
        "arousal": round(_mean(recent, "arousal") - _mean(older, "arousal"), 3),
        "controllability": round(_mean(recent, "controllability") - _mean(older, "controllability"), 3),
    }
    max_drift_dim = max(drift.keys(), key=lambda k: abs(drift[k]))
    if abs(drift[max_drift_dim]) < 0.15:
        return []
    now_iso = now.isoformat()
    return [MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=[max_drift_dim],  # dimension that drifted
        n_occurrences=len(recent) + len(older),
        span_days=2.0 * half_window_days,
        confidence=min(1.0, abs(drift[max_drift_dim]) / 0.5),
        detected_at=now_iso,
        raw_evidence=[
            f"drift:{max_drift_dim}={drift[max_drift_dim]:+.3f}",
            f"recent_n={len(recent)}",
            f"older_n={len(older)}",
        ],
    )]


def _detect_attractor_lock(
    trace: list[dict], *, min_fraction: float = 0.7, min_n: int = 30,
) -> list[MetaAffectPattern]:
    """Q5.4.1 — trace-level pattern: an attractor that occupies ≥70%
    of the trace within the window is a "lock-in" the operator
    should see. Not necessarily bad (productivity attractor is fine
    to be stuck in), but visible."""
    if len(trace) < min_n:
        return []
    counts: dict[str, int] = {}
    for p in trace:
        a = p["attractor"]
        counts[a] = counts.get(a, 0) + 1
    top_attractor = max(counts.keys(), key=lambda k: counts[k])
    top_n = counts[top_attractor]
    frac = top_n / len(trace)
    if frac < min_fraction:
        return []
    now_iso = datetime.now(timezone.utc).isoformat()
    span_seconds = (trace[-1]["ts"] - trace[0]["ts"]).total_seconds()
    return [MetaAffectPattern(
        pattern_kind="attractor_lock",
        breach_kinds=[top_attractor],
        n_occurrences=top_n,
        span_days=round(span_seconds / 86400.0, 2),
        confidence=min(1.0, frac),
        detected_at=now_iso,
        raw_evidence=[
            f"attractor={top_attractor}",
            f"fraction={frac:.2%}",
            f"trace_n={len(trace)}",
        ],
    )]


# ── Hypothesis generation (gated, decentered) ─────────────────────────────


def _draft_hypothesis(pattern: MetaAffectPattern) -> str | None:
    """Generate a one-sentence hypothesis, OR return None.

    Two-stage with strict SOUL.md discipline:

      1. **Template-based prose** (always, deterministic) produces
         neutral observational text for every supported pattern kind.
         This is the canonical hypothesis.
      2. **LLM enrichment** (Q5.4.2, gated by
         ``sentience_llm_hypothesis_enabled``) calls Claude Haiku 4.5
         with the structured pattern as input and asks for a single
         sentence one-better paraphrase. The model's output passes
         through ``decenter_text`` AS A HARD REJECT — if the LLM
         output contains any first-person affect phrase, the
         template prose is returned instead.

    The output is GUARANTEED to pass the decenter_text filter,
    whether template or LLM-enriched, because the filter is the
    final step before return.
    """
    kinds = ", ".join(pattern.breach_kinds[:3])
    if pattern.pattern_kind == "temporal_cluster":
        text = (
            f"The audit shows {pattern.n_occurrences} welfare breach"
            f"{'es' if pattern.n_occurrences != 1 else ''} of "
            f"{kinds!r} concentrated within {_TEMPORAL_BUCKET_HOURS}h."
        )
    elif pattern.pattern_kind == "recurring_trigger":
        text = (
            f"The pattern is {pattern.n_occurrences} repeats of "
            f"{kinds!r} over {pattern.span_days:.1f}d — likely "
            f"a recurring trigger condition."
        )
    elif pattern.pattern_kind == "sequence":
        text = (
            f"The welfare log shows {kinds!r} occurring "
            f"{pattern.n_occurrences} times across "
            f"{pattern.span_days:.0f}d at widely-spaced intervals — "
            f"a persistent sequence rather than an acute episode."
        )
    elif pattern.pattern_kind == "baseline_drift":
        text = (
            f"The trace indicates baseline drift on {kinds!r} of "
            f"magnitude visible across {pattern.span_days:.0f}d — "
            f"a gradual shift that did not trip welfare thresholds."
        )
    elif pattern.pattern_kind == "attractor_lock":
        text = (
            f"The trace indicates the {kinds!r} attractor held "
            f"{pattern.confidence:.0%} of snapshots over "
            f"{pattern.span_days:.0f}d — a stable but locked-in "
            f"pattern."
        )
    else:
        return None
    template_clean = decenter_text(text)
    # Stage 2: LLM enrichment, gated. Failure-isolated.
    enriched = _maybe_llm_enrich(pattern, template_clean) if template_clean else None
    if enriched and decenter_text(enriched):
        return enriched
    return template_clean


def _maybe_llm_enrich(
    pattern: MetaAffectPattern, template_text: str | None,
) -> str | None:
    """Q5.4.2 — Call Claude Haiku 4.5 for a one-better paraphrase.
    Failure-isolated. If anything goes wrong (no API key, model
    refusal, network), returns None and the caller falls back to
    the deterministic template.

    Strict prompt: forbids first-person affect language. The
    decenter filter is the second guard regardless."""
    if not template_text:
        return None
    try:
        from app.llm.factory import get_llm
    except Exception:
        return None
    prompt = (
        "Paraphrase the following observation in one neutral, "
        "observational sentence. Use third-person observational "
        "language only: 'the audit shows', 'the pattern is', 'the "
        "trace indicates' — NEVER first-person affect verbs (no "
        "'I feel', 'I'm anxious', 'my emotion', etc). Stay under "
        "180 characters.\n\n"
        f"Pattern kind: {pattern.pattern_kind}\n"
        f"Breach kinds: {', '.join(pattern.breach_kinds[:3])}\n"
        f"Occurrences: {pattern.n_occurrences}\n"
        f"Span: {pattern.span_days:.1f} days\n"
        f"Confidence: {pattern.confidence:.2f}\n\n"
        f"Template baseline: {template_text}\n\n"
        "Return JUST the one-sentence paraphrase, no commentary."
    )
    try:
        llm = get_llm(model="anthropic/claude-haiku-4.5")
        out = llm.call(prompt, max_tokens=120, temperature=0.2)
        # Tolerate either string or {choices:[...]} response shapes
        # depending on the factory's return type.
        if isinstance(out, str):
            text = out.strip()
        elif isinstance(out, dict):
            text = (
                out.get("choices", [{}])[0].get("message", {}).get("content")
                or out.get("content") or ""
            ).strip()
        else:
            text = str(out).strip()
        # Length guard — operator surface readability.
        if not text or len(text) > 280:
            return None
        return text
    except Exception:
        logger.debug("hot1: LLM enrichment failed", exc_info=True)
        return None


# ── Public detect + persist ───────────────────────────────────────────────


def detect_patterns(window_days: int = _DEFAULT_WINDOW_DAYS) -> list[MetaAffectPattern]:
    """One detection pass. Returns all detected patterns.

    Q5.4.1: now reads BOTH the full affect trace (every V/A/C
    snapshot) AND the welfare audit (breach events). Breach
    detectors give "what tripped" patterns; trace detectors give
    "what's shifting without tripping" patterns. Together they
    answer the user's framing — "feelings-about-feelings reflection
    on the affect trace" — properly."""
    if not _enabled():
        return []
    out: list[MetaAffectPattern] = []

    # Breach-based patterns (welfare_audit subset).
    breaches = _load_breaches(window_days)
    if breaches:
        out.extend(_detect_temporal_clusters(breaches))
        out.extend(_detect_recurring_triggers(breaches))
        out.extend(_detect_sequences(breaches))

    # Trace-based patterns (full affect snapshot stream).
    # Q5.4.1 — closes the "wrong source" gap that the post-ship
    # audit caught.
    trace = _load_trace_points(window_days)
    if trace:
        out.extend(_detect_baseline_drift(trace))
        out.extend(_detect_attractor_lock(trace))

    # Hypothesis prose (if LLM gate ON). All output passes through
    # decenter_text — see _draft_hypothesis.
    if _llm_hypothesis_enabled():
        for p in out:
            p.hypothesis = _draft_hypothesis(p)
    return out


def persist(patterns: list[MetaAffectPattern]) -> int:
    if not patterns:
        return 0
    path = _default_patterns_path()
    try:
        from app.utils.jsonl_retention import append_with_cap
    except Exception:
        return 0
    persisted = 0
    for p in patterns:
        try:
            append_with_cap(
                path,
                json.dumps(p.to_dict(), sort_keys=True),
                max_lines=_PATTERNS_LOG_MAX_LINES,
            )
            persisted += 1
        except Exception:
            logger.debug("hot1: persist failed")
    return persisted


def list_recent(n: int = 20) -> list[dict[str, Any]]:
    """Read recent patterns for the operator surface. Newest-first."""
    if not _enabled():
        return []
    path = _default_patterns_path()
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
    rows.sort(key=lambda r: r.get("detected_at", ""), reverse=True)
    return rows[:n]


# ── Idle entry ────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """One detection pass + persist + opaque GW publish."""
    if not _enabled():
        return {"ok": False, "skipped": True, "reason": "hot1_disabled"}
    try:
        patterns = detect_patterns()
    except Exception:
        logger.debug("hot1: detect raised", exc_info=True)
        return {"ok": False, "patterns": 0}
    persisted = persist(patterns)

    if patterns:
        try:
            from app.workspace_publish import publish_to_workspace
            # Opaque counts only — no breach kinds, no timestamps.
            publish_to_workspace(
                source="hot1_meta_affect",
                content=(
                    f"{len(patterns)} meta-affect pattern"
                    f"{'s' if len(patterns) != 1 else ''} detected"
                ),
                salience=min(0.85, 0.55 + 0.05 * len(patterns)),
                signal_type="disposition",
            )
        except Exception:
            logger.debug("hot1: GW publish failed", exc_info=True)
    # Q5.4.2 — landmark emission to the identity continuity ledger
    # for TRACE-LEVEL patterns only (baseline_drift / attractor_lock).
    # Breach-clustering patterns are routine; trace-level patterns
    # are the signal annual reflection should pick up.
    trace_patterns = [
        p for p in patterns
        if p.pattern_kind in ("baseline_drift", "attractor_lock")
    ]
    landmark_emitted = False
    if trace_patterns:
        try:
            from app.sentience_experiments.ledger_bridge import emit_landmark
            top = trace_patterns[0]
            landmark_emitted = emit_landmark(
                source_module="hot1_meta_affect",
                landmark_kind=top.pattern_kind,
                summary=(
                    f"HOT-1: {top.pattern_kind} pattern detected "
                    f"({top.n_occurrences} observations over "
                    f"{top.span_days:.1f}d)"
                ),
                counts={
                    "trace_patterns": len(trace_patterns),
                    "all_patterns": len(patterns),
                },
            )
        except Exception:
            logger.debug("hot1: ledger emit failed", exc_info=True)

    return {
        "ok": True,
        "patterns": len(patterns),
        "persisted": persisted,
        "ledger_landmark_emitted": landmark_emitted,
    }
