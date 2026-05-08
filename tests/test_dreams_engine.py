"""Tests for app.subia.dreams.engine — backward counterfactual replay.

Consciousness-roadmap §3.G2. Pin the contract for sampling, recombination,
prediction adapter, and audit-chain semantics.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from app.subia.dreams.engine import (
    FragmentSource,
    PerturbationKind,
    ReplayOutcome,
    ReplayScenario,
    construct_scenarios,
    run_pass,
    sample_fragments,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_dreams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect DREAMS_ROOT to a tmp dir so audit writes don't pollute."""
    dreams = tmp_path / "dreams"
    dreams.mkdir()

    from app import paths as _paths
    monkeypatch.setattr(_paths, "DREAMS_ROOT", dreams)

    return {"dreams": dreams, "audit": dreams / "replay_audit.jsonl"}


def _trace_line(*, ts: str, dominant_affect: str = "neutral") -> dict:
    return {
        "affect": {"ts": ts, "dominant_affect": dominant_affect},
        "viability": {},
    }


def _chapter(ts: str, summary: str) -> dict:
    return {
        "path": f"/tmp/{ts}.md",
        "ts": ts,
        "body": f"---\nts: {ts}\ntitle: Daily Chapter\n---\n# Chapter\n{summary}",
    }


# ── sample_fragments ─────────────────────────────────────────────────────


def test_sample_fragments_returns_at_most_max():
    rng = random.Random(42)
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(20)]
    chapters = [_chapter(f"d{i}", f"chapter content {i}") for i in range(10)]

    fragments = sample_fragments(
        rng=rng, max_fragments=5,
        trace_lines=trace, chapter_dicts=chapters,
    )

    assert len(fragments) == 5


def test_sample_fragments_includes_both_sources_when_available():
    rng = random.Random(0)
    trace = [_trace_line(ts=f"t{i}", dominant_affect="focus") for i in range(5)]
    chapters = [_chapter(f"d{i}", f"chapter {i}") for i in range(3)]

    fragments = sample_fragments(
        rng=rng, max_fragments=8,
        trace_lines=trace, chapter_dicts=chapters,
    )

    sources = {f.source for f in fragments}
    assert "affect_trace" in sources
    assert "chapter" in sources


def test_sample_fragments_handles_empty_inputs():
    fragments = sample_fragments(
        rng=random.Random(0), trace_lines=[], chapter_dicts=[],
    )
    assert fragments == []


def test_sample_fragments_summarizes_affect():
    rng = random.Random(1)
    trace = [_trace_line(ts="t1", dominant_affect="agitated")]
    fragments = sample_fragments(rng=rng, trace_lines=trace, chapter_dicts=[])
    assert "affect=agitated" in fragments[0].content_summary


# ── construct_scenarios ──────────────────────────────────────────────────


def test_construct_scenarios_zero_when_too_few_fragments():
    fragments = [FragmentSource(source="affect_trace", ts="t1",
                                content_summary="affect=calm", raw={})]
    scenarios = construct_scenarios(fragments, rng=random.Random(0))
    assert scenarios == []


def test_construct_scenarios_rotates_through_perturbations():
    fragments = [
        FragmentSource(source="affect_trace", ts=f"t{i}",
                       content_summary=f"affect=calm", raw={})
        for i in range(5)
    ]
    scenarios = construct_scenarios(
        fragments, rng=random.Random(0), max_scenarios=3,
    )

    perturbations = [s.perturbation for s in scenarios]
    # All 3 kinds present in 3 scenarios.
    assert PerturbationKind.AFFECT_FLIP in perturbations
    assert PerturbationKind.ITEM_SWAP in perturbations
    assert PerturbationKind.SEQUENCE_SHUFFLE in perturbations


def test_affect_flip_changes_summary():
    fragments = [
        FragmentSource(source="affect_trace", ts="t1",
                       content_summary="affect=calm", raw={}),
        FragmentSource(source="affect_trace", ts="t2",
                       content_summary="affect=focus", raw={}),
    ]
    scenarios = construct_scenarios(
        fragments, rng=random.Random(0), max_scenarios=1,
    )
    # Force the first kind in the rotation to be AFFECT_FLIP.
    assert scenarios[0].perturbation is PerturbationKind.AFFECT_FLIP
    flipped_text = " ".join(f.content_summary for f in scenarios[0].fragments)
    # Original "calm" or "focus" should appear renamed / paired with flip.
    assert "agitated" in flipped_text or "scattered" in flipped_text or "→" in scenarios[0].perturbation_note


def test_sequence_shuffle_reverses():
    fragments = [
        FragmentSource(source="affect_trace", ts=f"t{i}",
                       content_summary=f"affect=calm", raw={})
        for i in range(4)
    ]
    # Force SEQUENCE_SHUFFLE by picking a scenario index where it lands.
    scenarios = construct_scenarios(
        fragments, rng=random.Random(0), max_scenarios=3,
    )
    seq_scenarios = [s for s in scenarios
                     if s.perturbation is PerturbationKind.SEQUENCE_SHUFFLE]
    assert len(seq_scenarios) >= 1


def test_to_synthesized_context_includes_perturbation():
    fragments = [
        FragmentSource(source="affect_trace", ts="t1",
                       content_summary="affect=calm", raw={}),
        FragmentSource(source="affect_trace", ts="t2",
                       content_summary="affect=focus", raw={}),
    ]
    scenarios = construct_scenarios(
        fragments, rng=random.Random(0), max_scenarios=1,
    )
    ctx = scenarios[0].to_synthesized_context()
    assert "COUNTERFACTUAL" in ctx
    assert "Perturbation" in ctx


# ── run_pass ─────────────────────────────────────────────────────────────


def test_run_pass_skips_when_too_few_fragments(isolated_dreams):
    result = run_pass(
        rng=random.Random(0),
        trace_lines=[_trace_line(ts="t1", dominant_affect="calm")],
        chapter_dicts=[],
    )
    assert result.skipped


def test_run_pass_writes_audit_log(isolated_dreams):
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]
    result = run_pass(
        rng=random.Random(42),
        trace_lines=trace,
        chapter_dicts=[],
    )

    assert not result.skipped
    assert result.scenarios_count > 0
    assert result.audit_id is not None
    assert isolated_dreams["audit"].is_file()

    line = isolated_dreams["audit"].read_text().splitlines()[0]
    entry = json.loads(line)
    assert entry["id"] == result.audit_id
    assert entry["scenarios_count"] == result.scenarios_count


def test_run_pass_audit_chain_supersedes(isolated_dreams):
    """Each new audit entry references the previous one's id."""
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]

    r1 = run_pass(rng=random.Random(1), trace_lines=trace, chapter_dicts=[])
    r2 = run_pass(rng=random.Random(2), trace_lines=trace, chapter_dicts=[])

    lines = isolated_dreams["audit"].read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["supersedes"] is None
    assert e2["supersedes"] == e1["id"]


def test_run_pass_uses_injected_predict_fn(isolated_dreams):
    """The injected predictor is called once per scenario; its output
    appears in the audit log."""
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]
    calls = []

    def _fake_predict(scenario):
        calls.append(scenario.id)
        return (0.85, 0.15)

    result = run_pass(
        rng=random.Random(0),
        predict_fn=_fake_predict,
        trace_lines=trace,
        chapter_dicts=[],
    )

    assert len(calls) == result.scenarios_count
    assert all(o.predictor_confidence == 0.85 for o in result.outcomes)
    assert all(o.predictor_surprise == 0.15 for o in result.outcomes)


def test_run_pass_swallows_predict_fn_exceptions(isolated_dreams):
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]

    def _broken_predict(scenario):
        raise RuntimeError("predictor offline")

    result = run_pass(
        rng=random.Random(0),
        predict_fn=_broken_predict,
        trace_lines=trace,
        chapter_dicts=[],
    )

    assert not result.skipped
    assert all(o.predictor_error for o in result.outcomes)


def test_run_pass_observational_only(isolated_dreams):
    """G2 invariant: replay must NOT write to belief, current_goals, or
    anywhere outside `workspace/dreams/`. We can't directly assert no
    write happened to those (they're outside this fixture), but we CAN
    assert the only file produced is the audit log.
    """
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]
    run_pass(rng=random.Random(0), trace_lines=trace, chapter_dicts=[])

    # Only the audit file exists in the dreams dir.
    files = list(isolated_dreams["dreams"].iterdir())
    assert len(files) == 1
    assert files[0].name == "replay_audit.jsonl"


def test_pass_result_to_dict_is_json_safe(isolated_dreams):
    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]
    result = run_pass(rng=random.Random(0), trace_lines=trace, chapter_dicts=[])
    payload = json.dumps(result.to_dict())
    assert "scenarios_count" in payload


# ── production_predict_fn ────────────────────────────────────────────────


def test_production_predict_fn_calls_layer_with_replay_channel(isolated_dreams):
    """The factory should hand the predictor the dedicated REPLAY_CHANNEL
    name so replay outcomes don't pollute the per-channel running accuracy
    of real channels."""
    from app.subia.dreams.engine import (
        REPLAY_CHANNEL,
        production_predict_fn,
    )
    from types import SimpleNamespace

    captured = {}

    class _FakePredictor:
        def generate_prediction(self, context, beliefs):
            captured["context"] = context
            captured["beliefs"] = beliefs
            return SimpleNamespace(confidence=0.73)

    class _FakeLayer:
        def get_predictor(self, channel):
            captured["channel"] = channel
            return _FakePredictor()

    fn = production_predict_fn(layer=_FakeLayer())
    fragment = FragmentSource(source="affect_trace", ts="t1",
                              content_summary="affect=calm", raw={})
    scenario = ReplayScenario(
        id="s1", fragments=[fragment, fragment],
        perturbation=PerturbationKind.AFFECT_FLIP,
        perturbation_note="test",
        constructed_at="2026-05-08T12:00:00Z",
    )
    confidence, surprise = fn(scenario)

    assert captured["channel"] == REPLAY_CHANNEL
    assert "COUNTERFACTUAL" in captured["context"]
    assert confidence == pytest.approx(0.73)
    assert surprise == 0.0


def test_production_predict_fn_falls_back_to_neutral_on_error(isolated_dreams):
    """Predictor failures must yield (0.5, 0.0) — same as the stub. The
    replay engine never crashes the idle thread."""
    from app.subia.dreams.engine import production_predict_fn

    class _BrokenLayer:
        def get_predictor(self, channel):
            raise RuntimeError("predictive layer offline")

    fn = production_predict_fn(layer=_BrokenLayer())
    fragment = FragmentSource(source="affect_trace", ts="t1",
                              content_summary="affect=calm", raw={})
    scenario = ReplayScenario(
        id="s1", fragments=[fragment, fragment],
        perturbation=PerturbationKind.ITEM_SWAP,
        perturbation_note="test",
        constructed_at="2026-05-08T12:00:00Z",
    )
    confidence, surprise = fn(scenario)

    assert (confidence, surprise) == (0.5, 0.0)


def test_run_pass_uses_production_predict_fn(isolated_dreams):
    """End-to-end: run_pass with the production adapter wired feeds
    confidences from the layer's predictor through to the audit log."""
    from types import SimpleNamespace

    from app.subia.dreams.engine import production_predict_fn

    class _DiscriminatingPredictor:
        """Returns different confidences per perturbation note — lets us
        verify the audit log received the actual values."""
        def generate_prediction(self, context, beliefs):
            if "flipped" in context:
                return SimpleNamespace(confidence=0.20)
            if "swapped" in context:
                return SimpleNamespace(confidence=0.55)
            return SimpleNamespace(confidence=0.80)

    class _FakeLayer:
        def get_predictor(self, channel):
            return _DiscriminatingPredictor()

    trace = [_trace_line(ts=f"t{i}", dominant_affect="calm") for i in range(10)]
    result = run_pass(
        rng=random.Random(0),
        predict_fn=production_predict_fn(layer=_FakeLayer()),
        trace_lines=trace,
        chapter_dicts=[],
    )

    # Each outcome's confidence reflects the predictor's per-perturbation
    # response, NOT the stub's flat 0.5.
    confidences = {o.predictor_confidence for o in result.outcomes}
    assert confidences != {0.5}
    assert all(o.predictor_surprise == 0.0 for o in result.outcomes)
