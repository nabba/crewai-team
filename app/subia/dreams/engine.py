"""
subia.dreams.engine — Backward counterfactual replay engine.

Consciousness-roadmap §3.G2: the *real* "dreams" subsystem (recombination
+ simulation + learning from synthetic past experience). Distinct from:
  * `app/subia/reverie/`               — concept-walk synthesis
  * CIL Step 5 PREDICT                 — forward counterfactual prediction
  * Anthropic Managed Agents *dreams*  — session-transcript curation

Pipeline (one `run_pass()`):

  1. SAMPLE — pull fragments from the affect trace + recent narrative
              chapters. JSONL files only; no DB calls. Bounded by
              `MAX_FRAGMENTS_PER_PASS`.
  2. RECOMBINE — synthesize K alternative-past scenarios from the
              fragments by applying one of three perturbation kinds:
                * AFFECT_FLIP    — inverted dominant_affect
                * ITEM_SWAP      — swap salient items between two fragments
                * SEQUENCE_SHUFFLE — re-order the fragments' chronology
  3. PREDICT — run each scenario through an injected `predict_fn`
              (defaults to a no-op stub; production wires
              `PredictiveLayer.predict_and_compare`). The function should
              return a (confidence, surprise) pair. NO ground-truth check
              — the scenarios are synthetic; we record the predictor's
              own confidence + surprise as the learning signal.
  4. AUDIT — append outcomes to `workspace/dreams/replay_audit.jsonl`
              with hash-chained `supersedes` linkage (same convention as
              the wiki-index reconciler). Never modifies the retrospective
              signal store directly — that store has its own ingestion
              path.

Subsystem boundary: this module reads `workspace/affect/trace.jsonl` and
`workspace/affect/chapters/*.md` (read-only). Writes only to
`workspace/dreams/`. Does NOT touch belief/, kernel/, current_goals,
or any TIER_IMMUTABLE file.

T2 ethical threshold (consciousness-roadmap §6): on first sustained run
the operator should review the audit stream before any retrospective
rescan starts pulling from it. This module supports that — it doesn't
push into retrospective; retrospective decides whether to pull.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Tunables (governance — file-edit only) ───────────────────────────────

MAX_FRAGMENTS_PER_PASS = 8
MAX_SCENARIOS_PER_PASS = 4
TRACE_LOOKBACK_LINES = 500
RNG_SEED_DEFAULT: Optional[int] = None  # None = nondeterministic; tests pin it


# ── Result types ─────────────────────────────────────────────────────────


class PerturbationKind(str, enum.Enum):
    """How a scenario diverges from the real past."""
    AFFECT_FLIP = "affect_flip"
    ITEM_SWAP = "item_swap"
    SEQUENCE_SHUFFLE = "sequence_shuffle"


@dataclass
class FragmentSource:
    """One sampled fragment from a memory store."""
    source: str               # "affect_trace" | "chapter"
    ts: str
    content_summary: str      # short, human-readable
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ts": self.ts,
            "content_summary": self.content_summary[:200],
        }


@dataclass
class ReplayScenario:
    """A counterfactual constructed from K fragments + a perturbation."""
    id: str
    fragments: list[FragmentSource]
    perturbation: PerturbationKind
    perturbation_note: str    # human-readable description of what changed
    constructed_at: str

    def to_synthesized_context(self) -> str:
        """Render the scenario as a string suitable for handing to a
        predictor. Just a human-readable summary; predictor decides what
        to do with it.
        """
        lines = [f"COUNTERFACTUAL ({self.perturbation.value}):"]
        lines.append(f"Perturbation: {self.perturbation_note}")
        lines.append("Fragments:")
        for i, f in enumerate(self.fragments):
            lines.append(f"  [{i+1}] {f.source} @ {f.ts}: {f.content_summary[:160]}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "perturbation": self.perturbation.value,
            "perturbation_note": self.perturbation_note,
            "fragment_count": len(self.fragments),
            "fragments": [f.to_dict() for f in self.fragments],
            "constructed_at": self.constructed_at,
        }


@dataclass
class ReplayOutcome:
    """The result of running predict on one scenario."""
    scenario_id: str
    perturbation: PerturbationKind
    predictor_confidence: float
    predictor_surprise: float
    predictor_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "perturbation": self.perturbation.value,
            "predictor_confidence": round(self.predictor_confidence, 4),
            "predictor_surprise": round(self.predictor_surprise, 4),
            "predictor_error": self.predictor_error,
        }


@dataclass
class PassResult:
    """Outcome of one run_pass() — exposed for tests and operator."""
    sampled_count: int = 0
    scenarios_count: int = 0
    outcomes: list[ReplayOutcome] = field(default_factory=list)
    audit_id: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampled_count": self.sampled_count,
            "scenarios_count": self.scenarios_count,
            "outcomes": [o.to_dict() for o in self.outcomes],
            "audit_id": self.audit_id,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


# ── 1. SAMPLE ────────────────────────────────────────────────────────────


def _read_recent_trace_lines(n: int) -> list[dict]:
    """Last `n` JSONL lines from the affect trace, parsed."""
    try:
        from app.paths import AFFECT_TRACE
    except Exception:
        return []
    if not AFFECT_TRACE.is_file():
        return []
    out: list[dict] = []
    try:
        with AFFECT_TRACE.open("r", encoding="utf-8") as f:
            for line in deque(f, maxlen=n):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        logger.debug("dreams.engine: trace read failed: %s", exc)
    return out


def _read_recent_chapters(limit: int) -> list[dict]:
    """Up to `limit` recent narrative chapters as parsed dicts."""
    try:
        from app.paths import AFFECT_CHAPTERS_DIR
    except Exception:
        return []
    if not AFFECT_CHAPTERS_DIR.is_dir():
        return []

    md_files = sorted(AFFECT_CHAPTERS_DIR.glob("*.md"))[-limit:]
    out: list[dict] = []
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        # Crude frontmatter parse; chapter shape is small + stable.
        ts = ""
        first_lines = text.splitlines()[:10]
        for ln in first_lines:
            if ln.startswith("ts:"):
                ts = ln.split(":", 1)[1].strip().strip("'\"")
                break
        out.append({
            "path": str(p),
            "ts": ts or p.stem,
            "body": text[:1500],   # bounded
        })
    return out


def sample_fragments(
    *,
    rng: Optional[random.Random] = None,
    max_fragments: int = MAX_FRAGMENTS_PER_PASS,
    trace_lines: Optional[list[dict]] = None,
    chapter_dicts: Optional[list[dict]] = None,
) -> list[FragmentSource]:
    """Pull a small random sample of fragments from the affect trace + recent
    chapters. Returns at most `max_fragments`. Pure when given inputs.

    Production caller passes nothing — defaults read from disk.
    Tests pass `trace_lines=[...]` and `chapter_dicts=[...]` for determinism.
    """
    rng = rng or random.Random()
    fragments: list[FragmentSource] = []

    trace = trace_lines if trace_lines is not None else _read_recent_trace_lines(TRACE_LOOKBACK_LINES)
    chapters = chapter_dicts if chapter_dicts is not None else _read_recent_chapters(7)

    # Trace fragments: sample up to ⅔ of budget from trace.
    trace_budget = max(1, (max_fragments * 2) // 3)
    if trace:
        for raw in rng.sample(trace, min(trace_budget, len(trace))):
            affect = (raw or {}).get("affect") or {}
            ts = affect.get("ts", "")
            dominant = affect.get("dominant_affect", "neutral")
            fragments.append(FragmentSource(
                source="affect_trace",
                ts=str(ts),
                content_summary=f"affect={dominant}",
                raw=raw,
            ))

    # Chapter fragments: sample remaining budget from chapters.
    chapter_budget = max(0, max_fragments - len(fragments))
    if chapters and chapter_budget:
        for ch in rng.sample(chapters, min(chapter_budget, len(chapters))):
            body = ch.get("body", "")
            # First non-empty line of the body, after frontmatter.
            summary = ""
            for line in body.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith(("---", "#", "ts:", "title:")):
                    summary = stripped[:200]
                    break
            fragments.append(FragmentSource(
                source="chapter",
                ts=str(ch.get("ts", "")),
                content_summary=summary or "(empty chapter)",
                raw={"path": ch.get("path")},
            ))

    return fragments[:max_fragments]


# ── 2. RECOMBINE ─────────────────────────────────────────────────────────


_AFFECT_OPPOSITES = {
    "calm": "agitated",
    "agitated": "calm",
    "focus": "scattered",
    "scattered": "focus",
    "neutral": "ambivalent",
    "ambivalent": "neutral",
    "engaged": "withdrawn",
    "withdrawn": "engaged",
}


def _flip_affect(label: str) -> str:
    return _AFFECT_OPPOSITES.get(label, f"not-{label}")


def construct_scenarios(
    fragments: list[FragmentSource],
    *,
    rng: Optional[random.Random] = None,
    max_scenarios: int = MAX_SCENARIOS_PER_PASS,
) -> list[ReplayScenario]:
    """Build alternative-past scenarios by perturbing fragment combinations.

    Three perturbation kinds rotate through the output. Each scenario uses
    2-3 fragments. Pure given (fragments, rng).
    """
    rng = rng or random.Random()
    if len(fragments) < 2:
        return []

    scenarios: list[ReplayScenario] = []
    perturbations = list(PerturbationKind)
    now_iso = datetime.now(timezone.utc).isoformat()

    for i in range(max_scenarios):
        kind = perturbations[i % len(perturbations)]
        # Sample 2-3 fragments per scenario.
        k = rng.randint(2, min(3, len(fragments)))
        chosen = rng.sample(fragments, k)
        chosen_copy = [
            FragmentSource(
                source=f.source, ts=f.ts,
                content_summary=f.content_summary,
                raw=dict(f.raw),
            )
            for f in chosen
        ]

        if kind is PerturbationKind.AFFECT_FLIP:
            # Find a trace fragment, flip its dominant_affect summary.
            note = "no affect_trace fragment available"
            for f in chosen_copy:
                if f.source == "affect_trace" and "affect=" in f.content_summary:
                    original = f.content_summary.split("affect=", 1)[1]
                    flipped = _flip_affect(original)
                    f.content_summary = f"affect={flipped} (was {original})"
                    note = f"flipped dominant_affect: {original} → {flipped}"
                    break
        elif kind is PerturbationKind.ITEM_SWAP:
            # Swap content_summary between first two fragments.
            if len(chosen_copy) >= 2:
                a, b = chosen_copy[0], chosen_copy[1]
                a.content_summary, b.content_summary = b.content_summary, a.content_summary
                note = f"swapped content between fragments at {a.ts} and {b.ts}"
            else:
                note = "insufficient fragments for swap"
        elif kind is PerturbationKind.SEQUENCE_SHUFFLE:
            # Reverse chronological order of fragments.
            chosen_copy = list(reversed(chosen_copy))
            note = "reversed fragment chronology"
        else:
            note = "unknown perturbation"

        scenarios.append(ReplayScenario(
            id=f"replay_{uuid.uuid4().hex[:12]}",
            fragments=chosen_copy,
            perturbation=kind,
            perturbation_note=note,
            constructed_at=now_iso,
        ))

    return scenarios


# ── 3. PREDICT ───────────────────────────────────────────────────────────


PredictFn = Callable[[ReplayScenario], tuple[float, float]]
"""Adapter signature: takes a scenario, returns (confidence, surprise).

Production wiring: `PredictiveLayer.predict_and_compare(channel, context)` —
we adapt its return value into (confidence, surprise). Tests pass a stub.
"""


def _stub_predict_fn(scenario: ReplayScenario) -> tuple[float, float]:
    """Default no-op predictor: returns neutral confidence + zero surprise.

    Used when the engine runs without a real predictor wired. Produces
    well-shaped outcomes so the audit log is consistent in dev and tests.
    """
    return (0.5, 0.0)


# ── Production predictor adapter ─────────────────────────────────────────

# Channel name reserved on `PredictiveLayer` for replay walks. Distinct
# from real channels (text, etc.) so the predictor's per-channel running
# accuracy stays clean — replay outcomes shouldn't pollute real
# prediction-error stats.
REPLAY_CHANNEL = "counterfactual_replay"


def production_predict_fn(
    layer: Optional[Any] = None,
) -> PredictFn:
    """Build a predict_fn wired to the live `PredictiveLayer`.

    Production wiring (idle scheduler hook) calls this factory to get a
    closure suitable as `run_pass(predict_fn=...)`. Tests can pass a
    custom layer; default uses the singleton `get_predictive_layer()`.

    **Semantics — important caveat.** Counterfactual replay has no
    ground-truth "actual outcome" by construction (the scenario is
    synthetic). We therefore do not call the full `predict_and_compare`
    error pipeline; instead we extract just the predictor's *own*
    confidence on the synthesized context. Surprise is reported as 0.0
    because there is no real comparison to make.

    Sustained low confidence on a particular perturbation kind across
    many replay passes is the actionable signal — it indicates a
    systematic blind spot the retrospective rescan
    (`accuracy_tracker.has_sustained_error`) can act on, even without a
    surprise number.

    Failure mode: any exception during predictor access falls through to
    `(0.5, 0.0)` — the same neutral signal the stub returns. Replay must
    NEVER crash the calling subsystem.
    """
    def _adapter(scenario: ReplayScenario) -> tuple[float, float]:
        try:
            nonlocal layer
            if layer is None:
                from app.subia.prediction.layer import get_predictive_layer
                layer = get_predictive_layer()
            predictor = layer.get_predictor(REPLAY_CHANNEL)
            context = scenario.to_synthesized_context()
            prediction = predictor.generate_prediction(context, "")
            confidence = float(getattr(prediction, "confidence", 0.5))
            return (confidence, 0.0)
        except Exception as exc:
            logger.debug(
                "dreams.engine: production_predict_fn failed for %s: %s",
                getattr(scenario, "id", "<unknown>"),
                type(exc).__name__,
            )
            return (0.5, 0.0)

    return _adapter


def _run_predictions(
    scenarios: list[ReplayScenario],
    predict_fn: PredictFn,
) -> list[ReplayOutcome]:
    out: list[ReplayOutcome] = []
    for s in scenarios:
        try:
            conf, surprise = predict_fn(s)
            out.append(ReplayOutcome(
                scenario_id=s.id,
                perturbation=s.perturbation,
                predictor_confidence=float(conf),
                predictor_surprise=float(surprise),
            ))
        except Exception as exc:
            out.append(ReplayOutcome(
                scenario_id=s.id,
                perturbation=s.perturbation,
                predictor_confidence=0.0,
                predictor_surprise=0.0,
                predictor_error=f"{type(exc).__name__}: {exc}",
            ))
    return out


# ── 4. AUDIT ─────────────────────────────────────────────────────────────


def _audit_path() -> Path:
    """Resolve the audit log path. Lazy so DREAMS_ROOT changes (in tests)
    are picked up correctly."""
    from app.paths import DREAMS_ROOT
    return DREAMS_ROOT / "replay_audit.jsonl"


def _last_audit_id() -> Optional[str]:
    audit = _audit_path()
    if not audit.is_file():
        return None
    try:
        last = ""
        with audit.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        if not last:
            return None
        return json.loads(last).get("id")
    except (OSError, json.JSONDecodeError):
        return None


def _append_audit(
    *,
    scenarios: list[ReplayScenario],
    outcomes: list[ReplayOutcome],
) -> str:
    """Append a hash-chained audit entry. `superseded_by` invariant carried
    over from the wiki-index reconciler — never delete; chain entries.
    """
    audit = _audit_path()
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit_id = uuid.uuid4().hex[:16]
    entry = {
        "id": audit_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "supersedes": _last_audit_id(),
        "scenarios_count": len(scenarios),
        "outcomes_count": len(outcomes),
        "perturbation_breakdown": _perturbation_counts(outcomes),
        "outcomes": [o.to_dict() for o in outcomes],
        "scenarios": [s.to_dict() for s in scenarios],
    }
    with audit.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return audit_id


def _perturbation_counts(outcomes: list[ReplayOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.perturbation.value] = counts.get(o.perturbation.value, 0) + 1
    return counts


# ── Public entry: idle job ───────────────────────────────────────────────


def run_pass(
    *,
    predict_fn: Optional[PredictFn] = None,
    rng: Optional[random.Random] = None,
    trace_lines: Optional[list[dict]] = None,
    chapter_dicts: Optional[list[dict]] = None,
    max_fragments: int = MAX_FRAGMENTS_PER_PASS,
    max_scenarios: int = MAX_SCENARIOS_PER_PASS,
) -> PassResult:
    """One backward-counterfactual replay pass.

    Args:
        predict_fn: scenario → (confidence, surprise). Defaults to a stub
            (no real prediction). Production wires `PredictiveLayer
            .predict_and_compare`.
        rng: seeded for tests; None = nondeterministic in production.
        trace_lines / chapter_dicts: bypass disk reads for tests.
        max_fragments / max_scenarios: bounds for this pass.

    Returns:
        PassResult with sampled count, scenarios, outcomes, and audit id.
    """
    result = PassResult()

    try:
        fragments = sample_fragments(
            rng=rng,
            max_fragments=max_fragments,
            trace_lines=trace_lines,
            chapter_dicts=chapter_dicts,
        )
        result.sampled_count = len(fragments)
        if len(fragments) < 2:
            result.skipped = True
            result.skip_reason = (
                f"insufficient fragments ({len(fragments)} < 2)"
            )
            return result

        scenarios = construct_scenarios(
            fragments, rng=rng, max_scenarios=max_scenarios,
        )
        result.scenarios_count = len(scenarios)
        if not scenarios:
            result.skipped = True
            result.skip_reason = "no scenarios constructed"
            return result

        outcomes = _run_predictions(scenarios, predict_fn or _stub_predict_fn)
        result.outcomes = outcomes
        result.audit_id = _append_audit(scenarios=scenarios, outcomes=outcomes)

        logger.info(
            "dreams.engine: replay pass done — %d scenarios, audit_id=%s",
            len(scenarios), result.audit_id,
        )
        return result

    except Exception as exc:
        logger.warning("dreams.engine: pass failed: %s", exc, exc_info=True)
        result.error = f"{type(exc).__name__}: {exc}"
        return result


# ── CLI ──────────────────────────────────────────────────────────────────


def _main() -> int:
    """`python -m app.subia.dreams.engine` — manual run."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_pass()
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if not result.error else 1


if __name__ == "__main__":
    raise SystemExit(_main())
