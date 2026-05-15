"""PROGRAM §43.6 — Q5.6 follow-up tests.

Five findings from the fourth post-ship audit:
  P1#1  HOT-4 list_recent_flagged time-bound + briefing filter
  P1#2  _scorer_tier3_approval handles eligibility_failed → False
  P2#3  AE-2 dedup state cap (FIFO at 10K entries)
  P2#4  HOT-4 landmark 7-day cooldown via continuity-ledger consult
  P2#5  HOT-1 deliberate-separation docstring note
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


# ─────────────────────────────────────────────────────────────────────────
#   P1#1 — HOT-4 list_recent_flagged time-bound
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hot4():
    return _load_isolated(
        "hot4_q56", "app/sentience_experiments/hot4_metacog_monitor.py",
    )


def test_hot4_list_recent_flagged_respects_since_iso(hot4, monkeypatch, tmp_path):
    """Rows with ts < since_iso are filtered out."""
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    sig_path = tmp_path / "signals.jsonl"
    monkeypatch.setattr(hot4, "_default_signals_path", lambda: sig_path)
    # 3 rows: one 30d old, one 5d old, one today.
    now = datetime.now(timezone.utc)
    rows = [
        {"ts": (now - timedelta(days=30)).isoformat(),
         "agent_id": "x", "iteration": 0, "model": "m",
         "confidence_proxy": 1, "cache_reliance": 0, "cascade_jump": False,
         "unusual_score": 0.9, "flagged": True},
        {"ts": (now - timedelta(days=5)).isoformat(),
         "agent_id": "x", "iteration": 0, "model": "m",
         "confidence_proxy": 1, "cache_reliance": 0, "cascade_jump": False,
         "unusual_score": 0.9, "flagged": True},
        {"ts": now.isoformat(),
         "agent_id": "x", "iteration": 0, "model": "m",
         "confidence_proxy": 1, "cache_reliance": 0, "cascade_jump": False,
         "unusual_score": 0.9, "flagged": True},
    ]
    sig_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                        encoding="utf-8")
    # Without since_iso → all 3.
    all_rows = hot4.list_recent_flagged(n=10)
    assert len(all_rows) == 3
    # With since_iso=7-days-ago → only 2.
    week_ago = (now - timedelta(days=7)).isoformat()
    recent = hot4.list_recent_flagged(n=10, since_iso=week_ago)
    assert len(recent) == 2


def test_briefing_filters_hot4_to_seven_days(monkeypatch):
    """Source-level: the briefing passes a 7-day since_iso to
    list_recent_flagged."""
    src = Path("app/life_companion/daily_briefing.py").read_text()
    # Verify the filter is in place.
    assert "list_recent_flagged(n=20, since_iso=" in src
    assert "_td(days=7)" in src or "timedelta(days=7)" in src


# ─────────────────────────────────────────────────────────────────────────
#   P1#2 — Tier-3 scorer handles eligibility_failed
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rpt1():
    return _load_isolated(
        "rpt1_q56", "app/sentience_experiments/rpt1_self_calibration.py",
    )


def test_scorer_tier3_approval_treats_eligibility_failed_as_false(rpt1, monkeypatch):
    """eligibility_failed must return False (terminal non-approval),
    not None (perpetually unresolved)."""
    # Build a fake proposal at ELIGIBILITY_FAILED.
    class _State:
        value = "eligibility_failed"
    class _Proposal:
        state = _State()
    fake_load = MagicMock(return_value=_Proposal())
    # The scorer imports load_proposal inside the try; stub the module.
    fake_module = MagicMock(load_proposal=fake_load)
    monkeypatch.setitem(
        sys.modules, "app.governance_amendment.protocol", fake_module,
    )
    outcome = rpt1._scorer_tier3_approval({"plan_id": "abc"})
    assert outcome is False


def test_scorer_tier3_approval_still_returns_none_for_in_flight(rpt1, monkeypatch):
    """Proposals at STAGED/COOLDOWN states still return None."""
    class _State:
        value = "staged"
    class _Proposal:
        state = _State()
    fake_module = MagicMock(load_proposal=MagicMock(return_value=_Proposal()))
    monkeypatch.setitem(
        sys.modules, "app.governance_amendment.protocol", fake_module,
    )
    outcome = rpt1._scorer_tier3_approval({"plan_id": "abc"})
    assert outcome is None


def test_scorer_tier3_approval_still_returns_true_for_applied(rpt1, monkeypatch):
    class _State:
        value = "applied"
    class _Proposal:
        state = _State()
    fake_module = MagicMock(load_proposal=MagicMock(return_value=_Proposal()))
    monkeypatch.setitem(
        sys.modules, "app.governance_amendment.protocol", fake_module,
    )
    outcome = rpt1._scorer_tier3_approval({"plan_id": "abc"})
    assert outcome is True


# ─────────────────────────────────────────────────────────────────────────
#   P2#3 — AE-2 dedup state cap with FIFO eviction
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ae2():
    return _load_isolated(
        "ae2_q56", "app/sentience_experiments/ae2_causal_credit.py",
    )


def test_ae2_dedup_state_preserves_insertion_order(ae2, tmp_path, monkeypatch):
    """The persisted list preserves insertion order (newest at end)."""
    state_path = tmp_path / "landmarks.json"
    monkeypatch.setattr(ae2, "_default_landmark_state_path", lambda: state_path)
    keys = ["a||x", "b||y", "c||z"]
    ae2._save_emitted_landmarks(keys)
    loaded = ae2._load_emitted_landmarks()
    assert loaded == keys


def test_ae2_dedup_state_caps_at_threshold(ae2, tmp_path, monkeypatch):
    """When state exceeds _LANDMARK_STATE_CAP, drop the oldest batch."""
    state_path = tmp_path / "landmarks.json"
    monkeypatch.setattr(ae2, "_default_landmark_state_path", lambda: state_path)
    # Force cap to a small value for testability.
    monkeypatch.setattr(ae2, "_LANDMARK_STATE_CAP", 10)
    monkeypatch.setattr(ae2, "_LANDMARK_STATE_DROP_BATCH", 3)
    # Save 15 keys — should trim to 12 (15 - 3).
    keys = [f"sig{i}||outcome" for i in range(15)]
    ae2._save_emitted_landmarks(keys)
    loaded = ae2._load_emitted_landmarks()
    assert len(loaded) == 12
    # The OLDEST keys (sig0/1/2) were dropped; newest preserved.
    assert "sig0||outcome" not in loaded
    assert "sig1||outcome" not in loaded
    assert "sig2||outcome" not in loaded
    assert "sig14||outcome" in loaded


def test_ae2_dedup_state_below_cap_unchanged(ae2, tmp_path, monkeypatch):
    """Below the cap, save round-trips exactly."""
    state_path = tmp_path / "landmarks.json"
    monkeypatch.setattr(ae2, "_default_landmark_state_path", lambda: state_path)
    monkeypatch.setattr(ae2, "_LANDMARK_STATE_CAP", 100)
    keys = [f"k{i}" for i in range(50)]
    ae2._save_emitted_landmarks(keys)
    loaded = ae2._load_emitted_landmarks()
    assert loaded == keys


# ─────────────────────────────────────────────────────────────────────────
#   P2#4 — HOT-4 7-day landmark cooldown
# ─────────────────────────────────────────────────────────────────────────


def test_hot4_has_recent_landmark_true_within_window(hot4, monkeypatch):
    """A hot4 sentience_observation within 7d → cooldown active."""
    class _Event:
        def __init__(self, actor):
            self.actor = actor
            self.ts = datetime.now(timezone.utc).isoformat()
    fake_events = [_Event("hot4_metacog_monitor")]
    fake_module = MagicMock(list_events=MagicMock(return_value=fake_events))
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_module,
    )
    assert hot4._has_recent_hot4_landmark(days=7) is True


def test_hot4_has_recent_landmark_false_when_empty(hot4, monkeypatch):
    fake_module = MagicMock(list_events=MagicMock(return_value=[]))
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_module,
    )
    assert hot4._has_recent_hot4_landmark(days=7) is False


def test_hot4_has_recent_landmark_ignores_other_actors(hot4, monkeypatch):
    """Events from other source modules don't trigger hot4 cooldown."""
    class _Event:
        actor = "ae2_causal_credit"
        ts = datetime.now(timezone.utc).isoformat()
    fake_module = MagicMock(list_events=MagicMock(return_value=[_Event()]))
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_module,
    )
    assert hot4._has_recent_hot4_landmark(days=7) is False


def test_hot4_skips_landmark_when_cooldown_active(hot4, monkeypatch, tmp_path):
    """≥5 flagged + recent landmark → no new emission."""
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    monkeypatch.setattr(
        hot4, "_default_usage_path", lambda: tmp_path / "absent.jsonl",
    )
    # Detection returns ≥5 flagged.
    def _five():
        return [hot4.MetacogSignal(
            ts=f"2026-05-13T10:0{i}:00+00:00",
            agent_id="x", iteration=i, model="m",
            confidence_proxy=0.1, cache_reliance=0.5,
            cascade_jump=False, unusual_score=0.9, flagged=True,
        ) for i in range(6)]
    monkeypatch.setattr(hot4, "detect_signals", _five)
    monkeypatch.setattr(hot4, "persist", lambda s: len(s))
    # Cooldown active → no emit.
    monkeypatch.setattr(hot4, "_has_recent_hot4_landmark", lambda *, days=7: True)
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.sentience_experiments.ledger_bridge.emit_landmark",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = hot4.run()
    assert result["ledger_landmark_emitted"] is False
    assert captured == []


def test_hot4_emits_landmark_when_cooldown_expired(hot4, monkeypatch, tmp_path):
    """≥5 flagged + no recent landmark → emit."""
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    monkeypatch.setattr(
        hot4, "_default_usage_path", lambda: tmp_path / "absent.jsonl",
    )
    def _five():
        return [hot4.MetacogSignal(
            ts=f"2026-05-13T10:0{i}:00+00:00",
            agent_id="x", iteration=i, model="m",
            confidence_proxy=0.1, cache_reliance=0.5,
            cascade_jump=False, unusual_score=0.9, flagged=True,
        ) for i in range(6)]
    monkeypatch.setattr(hot4, "detect_signals", _five)
    monkeypatch.setattr(hot4, "persist", lambda s: len(s))
    monkeypatch.setattr(hot4, "_has_recent_hot4_landmark", lambda *, days=7: False)
    captured: list[dict] = []
    monkeypatch.setattr(
        "app.sentience_experiments.ledger_bridge.emit_landmark",
        lambda **kw: (captured.append(kw), True)[1],
    )
    result = hot4.run()
    assert result["ledger_landmark_emitted"] is True
    assert len(captured) == 1


# ─────────────────────────────────────────────────────────────────────────
#   P2#5 — HOT-1 deliberate-separation docstring
# ─────────────────────────────────────────────────────────────────────────


def test_hot1_docstring_documents_deliberate_separation():
    """The module docstring closes the cross-reference question
    explicitly."""
    src = Path("app/sentience_experiments/hot1_meta_affect.py").read_text()
    assert "Deliberate non-wire" in src
    assert "no-self vs. self-pass" in src
    # And the closing condition for re-opening.
    assert "re-open" in src or "re-opened" in src
