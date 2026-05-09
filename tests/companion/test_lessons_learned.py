"""Tests for ``app.companion.lessons_learned`` (Phase D #7)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.companion import lessons_learned as ll
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(ll, "_KB_PATH", tmp_path / "lessons_learned.json")

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


def test_embed_deterministic():
    from app.companion.lessons_learned import _embed
    a = _embed("rewrite the auth module to use jwt")
    b = _embed("rewrite the auth module to use jwt")
    assert a == b


def test_cosine_self_is_one():
    from app.companion.lessons_learned import _embed, _cosine
    v = _embed("forest carbon flux")
    assert abs(_cosine(v, v) - 1.0) < 1e-6


def test_cluster_groups_similar(isolated):
    from app.companion.lessons_learned import _cluster_into_kb
    events = [
        {"source": "change_request", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "rewrite auth module to use jwt sessions",
         "decision_reason": "auth changes go through legal review first"},
        {"source": "change_request", "ts": "2026-05-08T00:00:00+00:00",
         "proposal_text": "rewrite auth module to use jwt tokens",
         "decision_reason": "auth changes need legal review"},
    ]
    lessons = _cluster_into_kb(events, [])
    # Both events should land in the same cluster (similar text).
    assert len(lessons) == 1
    assert lessons[0]["count"] == 2


def test_cluster_keeps_distinct_separate(isolated):
    from app.companion.lessons_learned import _cluster_into_kb
    events = [
        {"source": "feedback_down", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "build a forest carbon dashboard",
         "decision_reason": "out of scope"},
        {"source": "feedback_down", "ts": "2026-05-08T00:00:00+00:00",
         "proposal_text": "rewrite billing engine in rust",
         "decision_reason": "no need"},
    ]
    lessons = _cluster_into_kb(events, [])
    assert len(lessons) == 2


def test_run_writes_kb(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.companion import lessons_learned as ll

    fake_events = [
        {"source": "change_request", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "rewrite auth module again — use jwt",
         "decision_reason": "auth path frozen pending audit"},
        {"source": "change_request", "ts": "2026-05-08T00:00:00+00:00",
         "proposal_text": "rewrite auth module switch to jwt tokens",
         "decision_reason": "auth path frozen pending audit"},
    ]
    monkeypatch.setattr(ll, "_from_change_requests", lambda c: fake_events)
    monkeypatch.setattr(ll, "_from_companion_feedback", lambda c: [])
    monkeypatch.setattr(ll, "_from_goodhart_reports", lambda c: [])

    summary = ll.run()
    assert summary["ran"] is True
    assert summary["events_seen"] == 2
    assert summary["lessons_total"] == 1
    assert (tmp_path / "lessons_learned.json").exists()


def test_check_against_returns_close_match(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.companion import lessons_learned as ll

    fake_events = [
        {"source": "change_request", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "rewrite auth module use jwt sessions",
         "decision_reason": "frozen pending legal audit"},
        {"source": "change_request", "ts": "2026-05-08T00:00:00+00:00",
         "proposal_text": "rewrite auth module move to jwt tokens",
         "decision_reason": "frozen pending legal audit"},
    ]
    monkeypatch.setattr(ll, "_from_change_requests", lambda c: fake_events)
    monkeypatch.setattr(ll, "_from_companion_feedback", lambda c: [])
    monkeypatch.setattr(ll, "_from_goodhart_reports", lambda c: [])
    ll.run()

    # Fresh proposal with similar shape should hit the lesson.
    matches = ll.check_against("rewrite auth module use jwt")
    assert matches
    # Hashing-trick cosine ≥ 0.4 is meaningful signal; not absolute.
    assert matches[0]["similarity"] >= 0.4
    assert "legal" in matches[0]["sample_reason"]


def test_check_against_unrelated_returns_empty(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.companion import lessons_learned as ll
    monkeypatch.setattr(ll, "_from_change_requests", lambda c: [
        {"source": "change_request", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "rewrite auth module use jwt",
         "decision_reason": "frozen"},
        {"source": "change_request", "ts": "2026-05-08T00:00:00+00:00",
         "proposal_text": "rewrite auth module use jwt tokens",
         "decision_reason": "frozen"},
    ])
    monkeypatch.setattr(ll, "_from_companion_feedback", lambda c: [])
    monkeypatch.setattr(ll, "_from_goodhart_reports", lambda c: [])
    ll.run()
    matches = ll.check_against("plant 100 trees in estonia winter")
    assert matches == []


def test_singleton_clusters_filtered(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.companion import lessons_learned as ll
    # One event → one singleton cluster → should NOT be persisted.
    monkeypatch.setattr(ll, "_from_change_requests", lambda c: [
        {"source": "change_request", "ts": "2026-05-09T00:00:00+00:00",
         "proposal_text": "lone proposal",
         "decision_reason": "no"},
    ])
    monkeypatch.setattr(ll, "_from_companion_feedback", lambda c: [])
    monkeypatch.setattr(ll, "_from_goodhart_reports", lambda c: [])
    summary = ll.run()
    assert summary["lessons_total"] == 0


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("LESSONS_LEARNED_ENABLED", "0")
    from app.companion import lessons_learned as ll
    summary = ll.run()
    assert summary["ran"] is False
