"""Tests for ``app.healing.llm_output_drift`` (Phase D #6)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.healing import llm_output_drift as drift
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(drift, "_BASELINE_PATH", tmp_path / "llm_drift_baseline.json")
    monkeypatch.setattr(drift, "_HISTORY_PATH", tmp_path / "llm_drift_history.jsonl")
    monkeypatch.setattr(drift, "_PROBES_PATH", tmp_path / "llm_drift_probes.json")

    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    monkeypatch.setattr(
        "app.healing.handlers._common.audit_event",
        lambda *a, **k: None,
    )
    yield tmp_path, sent


def test_hash_embed_deterministic():
    from app.healing.llm_output_drift import _hash_embed
    a = _hash_embed("hello world")
    b = _hash_embed("hello world")
    assert a == b


def test_hash_embed_different_for_different_text():
    from app.healing.llm_output_drift import _hash_embed
    a = _hash_embed("hello world")
    c = _hash_embed("totally unrelated content here")
    assert a != c


def test_cosine_self_is_one():
    from app.healing.llm_output_drift import _cosine, _hash_embed
    v = _hash_embed("estonia")
    assert math.isclose(_cosine(v, v), 1.0, abs_tol=1e-6)


def test_cosine_orthogonal_is_zero():
    from app.healing.llm_output_drift import _cosine
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert _cosine(a, b) == 0.0


def test_first_run_seeds_baseline(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import llm_output_drift as drift

    monkeypatch.setattr(drift, "_ask_llm", lambda q: "Tallinn")
    summary = drift.run()
    assert summary["ran"] is True
    assert summary["baseline_seeded"] is True
    assert (tmp_path / "llm_drift_baseline.json").exists()
    # No alert on first-run seeding even though similarity is 1.0.
    assert sent == []


def test_no_drift_no_alert(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import llm_output_drift as drift

    monkeypatch.setattr(drift, "_ask_llm", lambda q: "Tallinn")
    drift.run()  # seed
    # Reset cadence + clear last_alert, rerun with same answer.
    state_path = tmp_path / "self_heal" / "llm_output_drift.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    summary = drift.run()
    assert summary["avg_similarity"] is not None
    assert summary["avg_similarity"] >= 0.99
    assert sent == []


def test_drift_alerts(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import llm_output_drift as drift

    answers = iter(["Tallinn"] * 5 + [
        "Wholly different response — completely unrelated text "
        "with none of the original tokens preserved at all "
        "absolutely nothing in common"
    ] * 5)
    monkeypatch.setattr(drift, "_ask_llm", lambda q: next(answers))

    drift.run()  # seed
    state_path = tmp_path / "self_heal" / "llm_output_drift.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))

    summary = drift.run()
    assert summary["avg_similarity"] is not None
    # With completely orthogonal answers, similarity should be far
    # below the 0.85 threshold.
    assert summary["avg_similarity"] < 0.85
    assert summary["alerted"] is True


def test_llm_unavailable_no_alert(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import llm_output_drift as drift
    monkeypatch.setattr(drift, "_ask_llm", lambda q: None)
    summary = drift.run()
    assert summary["avg_similarity"] is None
    assert sent == []


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("LLM_OUTPUT_DRIFT_ENABLED", "0")
    from app.healing import llm_output_drift as drift
    summary = drift.run()
    assert summary["ran"] is False


def test_custom_probes_loaded(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.healing import llm_output_drift as drift
    custom = [{"id": "custom1", "question": "tell me a joke"}]
    (tmp_path / "llm_drift_probes.json").write_text(json.dumps(custom))
    probes = drift._load_probes()
    assert probes == custom
