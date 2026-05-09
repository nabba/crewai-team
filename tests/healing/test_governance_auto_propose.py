"""Tests for ``app.governance_ratchet.auto_propose`` (Phase C #5)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.governance_ratchet import auto_propose
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(auto_propose, "_PROPOSALS_PATH",
                        tmp_path / "governance_proposals.jsonl")

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


def _promotions(n: int, *, safety: float, quality: float) -> list[dict]:
    """Build n synthetic promotion rows with given safety+quality scores."""
    rows = []
    for i in range(n):
        rows.append({
            "action": "promotion_decision",
            "detail_json": {
                "safety_score": safety, "quality_score": quality,
            },
        })
    return rows


def test_no_promotions_no_proposals(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose
    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: [])
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 0)
    summary = auto_propose.run()
    assert summary["ran"] is True
    assert summary["proposed"] == []


def test_proposes_safety_when_conditions_met(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose

    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: _promotions(30, safety=0.99, quality=0.85))
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 1)
    # Make effective_value() reproducible.
    monkeypatch.setattr("app.governance_ratchet.protocol.effective_value",
                        lambda name: 0.95)

    summary = auto_propose.run()
    assert "safety_minimum" in summary["proposed"]
    assert "quality_minimum" not in summary["proposed"]
    assert any("safety_minimum" in body for body in sent)
    rows = (tmp_path / "governance_proposals.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    proposal = json.loads(rows[0])
    assert proposal["threshold"] == "safety_minimum"
    assert proposal["proposed_value"] > proposal["current_effective"]


def test_skips_when_rollback_rate_too_high(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose

    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: _promotions(30, safety=0.99, quality=0.85))
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 5)  # 5/30 = 16% > 5%
    monkeypatch.setattr("app.governance_ratchet.protocol.effective_value",
                        lambda name: 0.95)

    summary = auto_propose.run()
    assert summary["proposed"] == []


def test_skips_when_too_few_promotions(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose

    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: _promotions(10, safety=0.99, quality=0.85))
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 0)
    monkeypatch.setattr("app.governance_ratchet.protocol.effective_value",
                        lambda name: 0.95)

    summary = auto_propose.run()
    assert summary["proposed"] == []


def test_skips_when_no_headroom(isolated, monkeypatch):
    """Average must be ≥ effective + 0.03 to propose; just above effective doesn't qualify."""
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose

    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: _promotions(30, safety=0.96, quality=0.71))
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 0)
    monkeypatch.setattr("app.governance_ratchet.protocol.effective_value",
                        lambda name: 0.95 if name == "safety_minimum" else 0.70)

    summary = auto_propose.run()
    assert summary["proposed"] == []


def test_dedup_within_window(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import auto_propose

    monkeypatch.setattr(auto_propose, "_query_recent_promotions",
                        lambda days: _promotions(30, safety=0.99, quality=0.99))
    monkeypatch.setattr(auto_propose, "_query_recent_rollbacks",
                        lambda days: 0)
    monkeypatch.setattr("app.governance_ratchet.protocol.effective_value",
                        lambda name: 0.95 if name == "safety_minimum" else 0.70)

    auto_propose.run()
    initial = len(sent)

    # Reset cadence; rerun. Same conditions but proposal is in dedup window.
    state_path = tmp_path / "self_heal" / "governance_auto_propose.json"
    state = json.loads(state_path.read_text())
    state["last_run_at"] = 0.0
    state_path.write_text(json.dumps(state))
    auto_propose.run()
    # No new alerts within 14-day dedup.
    assert len(sent) == initial


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("GOVERNANCE_AUTO_PROPOSE_ENABLED", "0")
    from app.governance_ratchet import auto_propose
    summary = auto_propose.run()
    assert summary["ran"] is False
