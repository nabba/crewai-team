"""Tests for ``app.companion.feedback_weights`` (Phase D #4)."""
from __future__ import annotations

import json
import time

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.companion import feedback_weights
    monkeypatch.setattr(feedback_weights, "_STATE_PATH",
                        tmp_path / "feedback_weights.json")
    yield tmp_path / "feedback_weights.json"


def test_no_state_default_one(isolated):
    from app.companion.feedback_weights import current_multiplier
    assert current_multiplier("ws-1") == 1.0


def test_one_thumbs_down_drops_to_0_8(isolated):
    from app.companion.feedback_weights import current_multiplier, record_negative
    record_negative("ws-1")
    m = current_multiplier("ws-1")
    assert 0.79 <= m <= 0.81


def test_three_thumbs_down_floored_at_min(isolated):
    from app.companion.feedback_weights import current_multiplier, record_negative
    for _ in range(3):
        record_negative("ws-1")
    m = current_multiplier("ws-1")
    assert m >= 0.4   # floor
    # But MAY be slightly above the floor depending on decay; baseline
    # without decay would be 1 - 0.6 = 0.4 floored.
    assert m <= 0.5


def test_decay_pushes_multiplier_back_up(isolated, monkeypatch):
    from app.companion import feedback_weights
    feedback_weights.record_negative("ws-1")
    # Seed first_observed_at to 7 days ago (>2 halflives).
    state = json.loads(isolated.read_text())
    state["ws-1"]["first_observed_at"] = time.time() - 7 * 86400
    isolated.write_text(json.dumps(state))
    m = feedback_weights.current_multiplier("ws-1")
    # After 7d (≈2.3 halflives), multiplier should be near 1.0.
    assert m > 0.9


def test_thumbs_up_counteracts(isolated):
    from app.companion.feedback_weights import (
        current_multiplier, record_negative, record_positive,
    )
    record_negative("ws-1")
    record_negative("ws-1")
    m1 = current_multiplier("ws-1")
    record_positive("ws-1")
    m2 = current_multiplier("ws-1")
    assert m2 > m1


def test_distinct_workspaces_isolated(isolated):
    from app.companion.feedback_weights import current_multiplier, record_negative
    record_negative("ws-A")
    assert current_multiplier("ws-A") < 1.0
    assert current_multiplier("ws-B") == 1.0


def test_disabled_returns_one(isolated, monkeypatch):
    monkeypatch.setenv("COMPANION_FEEDBACK_WEIGHTS_ENABLED", "0")
    from app.companion.feedback_weights import current_multiplier, record_negative
    record_negative("ws-1")
    assert current_multiplier("ws-1") == 1.0


def test_reset_clears_state(isolated):
    from app.companion.feedback_weights import (
        current_multiplier, record_negative, reset,
    )
    record_negative("ws-1")
    assert current_multiplier("ws-1") < 1.0
    reset()
    assert current_multiplier("ws-1") == 1.0
