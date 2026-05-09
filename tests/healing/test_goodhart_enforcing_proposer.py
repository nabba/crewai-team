"""Tests for ``app.governance_ratchet.goodhart_enforcing_proposer`` (Phase D #3)."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(ghep, "_PROPOSALS_PATH",
                        tmp_path / "governance_proposals.jsonl")
    monkeypatch.setattr(ghep, "_GOODHART_REPORTS",
                        tmp_path / "goodhart_reports.json")
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


def test_skips_when_gate_off(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep
    monkeypatch.setattr(ghep, "_gate_mode", lambda: "off")
    summary = ghep.run()
    assert summary["mode"] == "off"
    assert summary["proposed"] is False


def test_skips_when_gate_already_enforcing(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep
    monkeypatch.setattr(ghep, "_gate_mode", lambda: "enforcing")
    summary = ghep.run()
    assert summary["proposed"] is False


def test_proposes_when_conditions_met(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep

    monkeypatch.setattr(ghep, "_gate_mode", lambda: "advisory")
    monkeypatch.setattr(
        ghep, "_read_recent_promotions",
        lambda days: [{"detail_json": {}} for _ in range(40)],
    )
    monkeypatch.setattr(
        ghep, "_read_recent_gaming_signals",
        lambda days: [
            {"severity": "low", "signal_type": "x", "description": "y",
             "detected_at": time.time() - 86400}
            for _ in range(2)
        ],
    )
    summary = ghep.run()
    assert summary["proposed"] is True
    assert any("Advisory → Enforcing" in body for body in sent)
    rows = (tmp_path / "governance_proposals.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    proposal = json.loads(rows[0])
    assert proposal["threshold"] == "goodhart_enforcing"
    assert proposal["proposed_value"] == "enforcing"


def test_skips_when_block_pct_too_high(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep

    monkeypatch.setattr(ghep, "_gate_mode", lambda: "advisory")
    # 40 promotions, 5 high signals → 12.5% block rate, > 5% cap.
    monkeypatch.setattr(
        ghep, "_read_recent_promotions",
        lambda days: [{"detail_json": {}} for _ in range(40)],
    )
    monkeypatch.setattr(
        ghep, "_read_recent_gaming_signals",
        lambda days: [
            {"severity": "high", "signal_type": "x", "description": "y",
             "detected_at": time.time() - 86400}
            for _ in range(5)
        ],
    )
    summary = ghep.run()
    assert summary["proposed"] is False


def test_skips_when_too_few_promotions(isolated, monkeypatch):
    tmp_path, sent = isolated
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep

    monkeypatch.setattr(ghep, "_gate_mode", lambda: "advisory")
    monkeypatch.setattr(ghep, "_read_recent_promotions", lambda days: [{}] * 10)
    monkeypatch.setattr(ghep, "_read_recent_gaming_signals", lambda days: [])
    summary = ghep.run()
    assert summary["proposed"] is False


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("GOODHART_ENFORCING_PROPOSER_ENABLED", "0")
    from app.governance_ratchet import goodhart_enforcing_proposer as ghep
    summary = ghep.run()
    assert summary["ran"] is False
