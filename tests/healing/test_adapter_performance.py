"""Tests for ``app.training.adapter_performance`` (Phase C #1)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.training import adapter_performance
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(adapter_performance, "_REGISTRY_PATH",
                        tmp_path / "registry.json")
    monkeypatch.setattr(adapter_performance, "_HISTORY_PATH",
                        tmp_path / "adapter_quality_history.jsonl")
    monkeypatch.setattr(adapter_performance, "_PROPOSALS_PATH",
                        tmp_path / "retirement_proposals.jsonl")
    monkeypatch.setattr(adapter_performance,
                        "_recipe_winrate_for_adapter", lambda *a, **kw: None)

    yield tmp_path


def test_age_decay_curve():
    from app.training.adapter_performance import _age_decay
    assert _age_decay(0) == 1.0
    assert _age_decay(30) == 1.0
    assert _age_decay(180) == pytest.approx(0.5)
    assert _age_decay(365) == 0.3
    # Linear ramp at 90 days should be between 0.5 and 1.0.
    assert 0.5 < _age_decay(90) < 1.0


def test_health_score_old_low_eval(isolated):
    from app.training.adapter_performance import _adapter_health
    now = datetime.now(timezone.utc)
    info = {
        "eval_score": 0.5,
        "created_at": (now - timedelta(days=120)).isoformat(),
    }
    h = _adapter_health(info, now, quality_gate=0.75)
    assert h["health_score"] < 0.6
    assert h["days_since_created"] >= 119


def test_run_no_registry_noop(isolated):
    from app.training import adapter_performance
    summary = adapter_performance.run()
    assert summary["ran"] is True
    assert summary["candidates"] == 0


def test_run_proposes_low_quality_adapter(isolated, monkeypatch):
    from app.training import adapter_performance
    sent: list[str] = []
    monkeypatch.setattr(
        "app.healing.handlers._common.send_signal_alert",
        lambda body, **kw: sent.append(body) or True,
    )
    monkeypatch.setattr(
        "app.healing.handlers._common.audit_event",
        lambda *a, **k: None,
    )

    now = datetime.now(timezone.utc)
    registry = {
        "weak_adapter": {
            "name": "weak_adapter", "eval_score": 0.4,
            "created_at": (now - timedelta(days=120)).isoformat(),
            "examples_count": 1000, "promoted": True, "agent_roles": ["coder"],
        },
        "good_adapter": {
            "name": "good_adapter", "eval_score": 0.85,
            "created_at": (now - timedelta(days=20)).isoformat(),
            "examples_count": 2000, "promoted": True, "agent_roles": ["writer"],
        },
    }
    (isolated / "registry.json").write_text(json.dumps(registry))

    summary = adapter_performance.run()
    assert summary["candidates"] == 1
    proposals = (isolated / "retirement_proposals.jsonl").read_text().strip()
    assert "weak_adapter" in proposals
    assert "good_adapter" not in proposals


def test_history_appended(isolated):
    from app.training import adapter_performance
    now = datetime.now(timezone.utc)
    registry = {
        "a": {
            "name": "a", "eval_score": 0.85,
            "created_at": (now - timedelta(days=10)).isoformat(),
            "examples_count": 1, "promoted": True, "agent_roles": [],
        },
    }
    (isolated / "registry.json").write_text(json.dumps(registry))
    summary = adapter_performance.run()
    assert summary["snapshotted"] == 1
    history = (isolated / "adapter_quality_history.jsonl").read_text().strip()
    assert "a" in history
    assert "0.85" in history


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("ADAPTER_PERFORMANCE_ENABLED", "0")
    from app.training import adapter_performance
    summary = adapter_performance.run()
    assert summary["ran"] is False
