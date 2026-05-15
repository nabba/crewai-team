"""PROGRAM §43.5 — Q5.5 follow-up tests.

Six findings from the third post-ship audit:
  #1  HOT-1 LLM enrichment was calling non-existent ``app.llm.factory``
  #2  HOT-4 had no landmark emission (annual reflection blind to it)
  #3  RPT-1 forecasts could sit unresolved forever
  #4  AE-2 emitted the same landmark on every pass (no first-emission dedup)
  #5  predicted_p collapsed "no history" and "proven 0%" cases
  #6  ledger_bridge had a dead `_MAX_EMISSIONS_PER_PASS` constant
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_app_config(monkeypatch, api_key=None):
    """Inject a stub ``app.config`` so the LLM enrichment path can run
    on dev environments without pydantic_settings.

    The real module fails to import on minimal dev envs; the test
    only needs ``get_anthropic_api_key``. Production environments
    use the real module."""
    stub = MagicMock()
    stub.get_anthropic_api_key = MagicMock(return_value=api_key)
    monkeypatch.setitem(sys.modules, "app.config", stub)


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#1 — HOT-1 LLM enrichment uses canonical Anthropic API
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hot1():
    return _load_isolated(
        "hot1_q55", "app/sentience_experiments/hot1_meta_affect.py",
    )


def test_hot1_llm_enrich_uses_canonical_anthropic_pattern():
    """Source-level: _maybe_llm_enrich imports anthropic.Anthropic +
    get_anthropic_api_key, NOT the non-existent app.llm.factory."""
    src = Path("app/sentience_experiments/hot1_meta_affect.py").read_text()
    assert "from anthropic import Anthropic" in src
    assert "from app.config import get_anthropic_api_key" in src
    # The dead path must be gone.
    assert "from app.llm.factory import get_llm" not in src


def test_hot1_llm_enrich_returns_none_without_api_key(hot1, monkeypatch):
    """No API key → fallback path returns None, template wins."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    _stub_app_config(monkeypatch, api_key=None)
    # Stub anthropic too so the import doesn't fail on dev envs.
    monkeypatch.setitem(sys.modules, "anthropic", MagicMock())
    assert hot1._maybe_llm_enrich(p, "template") is None


def test_hot1_llm_enrich_exercises_real_call_path(hot1, monkeypatch):
    """LOAD-BEARING — exercise the actual Anthropic call path with a
    stubbed client. The Q5.4 test only mocked _maybe_llm_enrich
    directly, missing that the import was broken.

    This test stubs ``anthropic.Anthropic`` itself, so the production
    code path (import → client.messages.create → _extract_text_from_resp)
    runs end-to-end."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    # Provide an API key so the function gets past the guard.
    _stub_app_config(monkeypatch, api_key="test-key")
    # Build a fake Anthropic client whose messages.create returns a
    # response with .content[0].text shape.
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "The trace indicates a notable valence shift."
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp
    fake_anthropic_class = MagicMock(return_value=fake_client)
    # Inject into the anthropic module so the production code's
    # ``from anthropic import Anthropic`` picks up the stub.
    monkeypatch.setitem(sys.modules, "anthropic", MagicMock(Anthropic=fake_anthropic_class))
    out = hot1._maybe_llm_enrich(p, "Template baseline.")
    assert out == "The trace indicates a notable valence shift."
    # The client was called with the right model.
    args, kwargs = fake_client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 120


def test_hot1_llm_enrich_extracts_multi_block_text(hot1, monkeypatch):
    """_extract_text_from_resp tolerates blocks without .type and
    concatenates multi-block responses."""
    block1 = MagicMock()
    block1.type = "text"
    block1.text = "First half. "
    block2 = MagicMock()
    block2.type = "text"
    block2.text = "Second half."
    resp = MagicMock()
    resp.content = [block1, block2]
    assert hot1._extract_text_from_resp(resp) == "First half. Second half."


def test_hot1_llm_enrich_rejects_first_person_via_decenter_filter(hot1, monkeypatch):
    """Even with a working LLM, output containing first-person affect
    must be rejected by the decenter filter."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    _stub_app_config(monkeypatch, api_key="test-key")
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "I feel that the system is drifting."
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp
    monkeypatch.setitem(
        sys.modules, "anthropic",
        MagicMock(Anthropic=MagicMock(return_value=fake_client)),
    )
    # _maybe_llm_enrich returns the text — caller's _draft_hypothesis
    # is responsible for filtering. Verify _draft_hypothesis falls
    # back to template when LLM produces forbidden prose.
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    hyp = hot1._draft_hypothesis(p)
    assert "i feel" not in (hyp or "").lower()


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#2 — HOT-4 landmark emission
# ─────────────────────────────────────────────────────────────────────────


def test_hot4_landmark_emits_at_threshold(monkeypatch, tmp_path):
    """≥5 flagged signals in one pass triggers a sentience_observation."""
    hot4 = _load_isolated(
        "hot4_q55", "app/sentience_experiments/hot4_metacog_monitor.py",
    )
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    monkeypatch.setattr(hot4, "_default_usage_path", lambda: tmp_path / "absent.jsonl")

    # Patch detect_signals to return 5 flagged + a few non-flagged.
    def _five_flagged():
        out = []
        for i in range(7):
            out.append(hot4.MetacogSignal(
                ts=f"2026-05-13T10:{i:02d}:00+00:00",
                agent_id="coder", iteration=i, model="haiku",
                confidence_proxy=0.1, cache_reliance=0.5,
                cascade_jump=False,
                unusual_score=0.9 if i < 5 else 0.1,
                flagged=(i < 5),
            ))
        return out
    monkeypatch.setattr(hot4, "detect_signals", _five_flagged)
    monkeypatch.setattr(hot4, "persist", lambda signals: len(signals))

    # Stub emit_landmark to capture the call.
    captured: list[dict] = []
    def fake_emit(**kwargs):
        captured.append(kwargs)
        return True
    monkeypatch.setattr(
        "app.sentience_experiments.ledger_bridge.emit_landmark", fake_emit,
    )
    result = hot4.run()
    assert result["ledger_landmark_emitted"] is True
    assert len(captured) == 1
    assert captured[0]["source_module"] == "hot4_metacog_monitor"
    assert captured[0]["landmark_kind"] == "sustained_reasoning_anomaly"
    # Opaque counts only — no agent_ids or model names in the summary.
    summary = captured[0]["summary"]
    assert "coder" not in summary
    assert "haiku" not in summary


def test_hot4_landmark_silent_below_threshold(monkeypatch, tmp_path):
    """<5 flagged → no landmark emission."""
    hot4 = _load_isolated(
        "hot4_q55b", "app/sentience_experiments/hot4_metacog_monitor.py",
    )
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    monkeypatch.setattr(hot4, "_default_usage_path", lambda: tmp_path / "absent.jsonl")

    def _three_flagged():
        return [
            hot4.MetacogSignal(
                ts=f"2026-05-13T10:0{i}:00+00:00",
                agent_id="x", iteration=i, model="m",
                confidence_proxy=0.1, cache_reliance=0.5,
                cascade_jump=False, unusual_score=0.9, flagged=True,
            ) for i in range(3)
        ]
    monkeypatch.setattr(hot4, "detect_signals", _three_flagged)
    monkeypatch.setattr(hot4, "persist", lambda signals: len(signals))
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.sentience_experiments.ledger_bridge.emit_landmark",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = hot4.run()
    assert result["ledger_landmark_emitted"] is False
    assert captured == []


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#3 — RPT-1 stale-forecast timeout
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rpt1():
    return _load_isolated(
        "rpt1_q55", "app/sentience_experiments/rpt1_self_calibration.py",
    )


def test_rpt1_stale_forecast_terminated(rpt1, monkeypatch, tmp_path):
    """A forecast past resolution_at by ≥60d with a scorer that
    keeps returning None should be terminated with
    score_error='stale_unresolved'."""
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)
    rpt1.register_scorer("indeterminate", lambda args: None)
    # Resolution_at was 70 days ago.
    very_old = datetime.now(timezone.utc) - timedelta(days=70)
    fc = rpt1.register_prediction(
        claim_kind="kx", claim_text="t", predicted_p=0.5,
        resolution_at=very_old, scorer_ref="indeterminate",
    )
    summary = rpt1.reconcile_due()
    assert summary["errors"] >= 1
    forecasts = rpt1._read_all_forecasts()
    matching = [f for f in forecasts if f.id == fc.id]
    assert len(matching) == 1
    assert matching[0].score_error == "stale_unresolved"


def test_rpt1_stale_forecast_within_grace_not_terminated(rpt1, monkeypatch, tmp_path):
    """A forecast past resolution_at by <60d should NOT be terminated."""
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)
    rpt1.register_scorer("indeterminate2", lambda args: None)
    # Resolution_at was 10 days ago — well within grace.
    recent_past = datetime.now(timezone.utc) - timedelta(days=10)
    fc = rpt1.register_prediction(
        claim_kind="kx", claim_text="t", predicted_p=0.5,
        resolution_at=recent_past, scorer_ref="indeterminate2",
    )
    rpt1.reconcile_due()
    forecasts = rpt1._read_all_forecasts()
    matching = [f for f in forecasts if f.id == fc.id]
    assert len(matching) == 1
    assert matching[0].score_error is None
    assert matching[0].actual is None  # still unresolved, not stale


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#4 — AE-2 landmark dedup
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ae2():
    return _load_isolated(
        "ae2_q55", "app/sentience_experiments/ae2_causal_credit.py",
    )


def test_ae2_landmark_dedup_state_round_trip(ae2, tmp_path):
    """Read/write of the dedup state file is symmetric.

    Q5.6 changed the return type from set to list to preserve
    insertion order for FIFO eviction at the cap. The dedup
    semantics are still set-like (membership check via ``in``)
    but the underlying representation is ordered."""
    state_path = tmp_path / "landmarks.json"
    import app.sentience_experiments.ae2_causal_credit as real_ae2
    original = real_ae2._default_landmark_state_path
    real_ae2._default_landmark_state_path = lambda: state_path
    try:
        assert real_ae2._load_emitted_landmarks() == []
        real_ae2._save_emitted_landmarks(["a||x", "b||y"])
        assert set(real_ae2._load_emitted_landmarks()) == {"a||x", "b||y"}
    finally:
        real_ae2._default_landmark_state_path = original


def test_ae2_landmark_only_emits_on_first_observation(monkeypatch, tmp_path):
    """Running the module twice with the same strong association
    should emit a landmark on the FIRST pass only."""
    ae2 = _load_isolated(
        "ae2_q55b", "app/sentience_experiments/ae2_causal_credit.py",
    )
    monkeypatch.setattr(ae2, "_enabled", lambda: True)
    state_path = tmp_path / "landmarks.json"
    assoc_path = tmp_path / "assoc.jsonl"
    monkeypatch.setattr(ae2, "_default_landmark_state_path", lambda: state_path)
    monkeypatch.setattr(ae2, "_default_associations_path", lambda: assoc_path)
    monkeypatch.setattr(ae2, "_default_usage_path", lambda: tmp_path / "u.jsonl")
    monkeypatch.setattr(ae2, "_default_errors_path", lambda: tmp_path / "e.jsonl")
    monkeypatch.setattr(ae2, "_default_welfare_audit_path", lambda: tmp_path / "w.jsonl")
    monkeypatch.setattr(ae2, "_default_audit_log_path", lambda: tmp_path / "a.jsonl")

    # Force the same single high-density association on every pass.
    fake_assoc = ae2.CausalAssociation(
        action_signature="agent=coder|model=x",
        outcome_kind="error:Fatal",
        outcome_rate=0.02,
        outcome_density_ratio=8.0,
        n_observations=10,
        n_actions=20,
        first_seen="2026-05-13T00:00:00+00:00",
        last_seen="2026-05-13T01:00:00+00:00",
        confidence=0.8,
    )
    monkeypatch.setattr(ae2, "detect_associations", lambda **kw: [fake_assoc])
    monkeypatch.setattr(ae2, "persist", lambda assocs: len(assocs))

    captured: list[dict] = []
    def fake_emit(**kwargs):
        captured.append(kwargs)
        return True
    monkeypatch.setattr(
        "app.sentience_experiments.ledger_bridge.emit_landmark", fake_emit,
    )
    # First pass: emit landmark.
    r1 = ae2.run()
    assert r1["ledger_landmark_emitted"] is True
    assert len(captured) == 1
    # Second pass: same association, should NOT emit again.
    r2 = ae2.run()
    assert r2["ledger_landmark_emitted"] is False
    assert len(captured) == 1  # unchanged


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#5 — predicted_p history-vs-zero distinction
# ─────────────────────────────────────────────────────────────────────────


def test_relevant_history_by_kind_empty_has_no_resolved_history():
    """Empty result must include has_resolved_history=False."""
    rh = _load_isolated(
        "rh_q55", "app/identity/relevant_history.py",
    )
    result = rh._empty_by_kind("soul_edit", 365)
    assert result["has_resolved_history"] is False


def test_relevant_history_by_kind_with_resolutions_has_resolved_history(monkeypatch, tmp_path):
    """A kind with applied+rolled_back > 0 should report has_resolved_history=True."""
    rh = _load_isolated("rh_q55b", "app/identity/relevant_history.py")
    try:
        import app.runtime_settings  # noqa: F401
    except Exception:
        pytest.skip("app.runtime_settings unavailable")
    monkeypatch.setattr(
        "app.runtime_settings.get_ledger_governor_enabled", lambda: True,
    )
    import app.identity.continuity_ledger as cl
    monkeypatch.setattr(cl, "list_events", lambda **kwargs: [])
    cr_log = tmp_path / "audit.jsonl"
    now_iso = datetime.now(timezone.utc).isoformat()
    cr_log.write_text("\n".join([
        json.dumps({"ts": now_iso, "payload": {
            "event": "applied", "path": "app/tools/x.py",
            "request_id": "cr1", "status": "applied",
        }}),
        json.dumps({"ts": now_iso, "payload": {
            "event": "rolled_back", "path": "app/tools/y.py",
            "request_id": "cr2", "status": "rolled_back",
        }}),
    ]) + "\n", encoding="utf-8")
    monkeypatch.setattr(rh, "_cr_audit_path", lambda: cr_log)
    result = rh.relevant_history_by_kind("app/tools/z.py")
    assert result["has_resolved_history"] is True
    assert result["success_rate"] == 0.5


def test_tier3_producer_uses_has_resolved_history():
    """Source-level: tier-3 producer reads has_resolved_history."""
    src = Path("app/tools/request_tier3_amendment.py").read_text()
    assert "has_resolved_history" in src


def test_cr_producer_uses_has_resolved_history():
    """Source-level: CR producer reads has_resolved_history."""
    src = Path("app/change_requests/lifecycle.py").read_text()
    assert "has_resolved_history" in src


# ─────────────────────────────────────────────────────────────────────────
#   Q5.5#6 — ledger_bridge per-process emission ceiling
# ─────────────────────────────────────────────────────────────────────────


def test_ledger_bridge_per_process_ceiling(monkeypatch):
    """After _MAX_EMISSIONS_PER_PROCESS hits, further emissions return False."""
    bridge = _load_isolated(
        "bridge_q55", "app/sentience_experiments/ledger_bridge.py",
    )
    # Don't actually write to the real ledger.
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event",
        lambda **kwargs: True,
    )
    bridge._reset_emission_counter_for_tests()
    cap = bridge._MAX_EMISSIONS_PER_PROCESS
    # Below cap: all succeed.
    for _ in range(cap):
        assert bridge.emit_landmark(
            source_module="test_source",
            landmark_kind="x",
            summary="non-empty summary text",
        ) is True
    # At cap: refuses.
    assert bridge.emit_landmark(
        source_module="test_source",
        landmark_kind="x",
        summary="this should refuse",
    ) is False
    # A different source_module has its own counter.
    assert bridge.emit_landmark(
        source_module="other_source",
        landmark_kind="x",
        summary="different source has its own budget",
    ) is True


def test_ledger_bridge_dead_constant_removed():
    """Source-level: the old _MAX_EMISSIONS_PER_PASS constant is gone."""
    src = Path("app/sentience_experiments/ledger_bridge.py").read_text()
    assert "_MAX_EMISSIONS_PER_PASS" not in src
    assert "_MAX_EMISSIONS_PER_PROCESS" in src
