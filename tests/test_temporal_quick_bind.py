"""Tests for app.subia.temporal.binding.temporal_quick_bind.

Consciousness-roadmap §3.G4 — compressed-loop binding cadence. The full
`temporal_bind` reducer requires FEEL/ATTEND/OWN/PREDICT/MONITOR; the
compressed CIL only runs Steps 1-3, so a cheap variant computes what
it can from FEEL+ATTEND alone.

These tests pin the contract: which fields are populated on a quick-bind
vs left at dataclass defaults.
"""

from __future__ import annotations

import pytest

from app.subia.temporal.binding import (
    BoundMoment,
    temporal_bind,
    temporal_quick_bind,
)


def test_quick_bind_returns_bound_moment():
    bm = temporal_quick_bind(feel={}, attend={})
    assert isinstance(bm, BoundMoment)


def test_quick_bind_populates_dominant_affect():
    bm = temporal_quick_bind(
        feel={"dominant_affect": "calm-focus"},
        attend={},
    )
    assert bm.dominant_affect == "calm-focus"


def test_quick_bind_defaults_dominant_affect_to_neutral():
    bm = temporal_quick_bind(feel={}, attend={})
    assert bm.dominant_affect == "neutral"


def test_quick_bind_populates_salient_focus_top_5():
    items = [
        {"id": f"item-{i}", "salience": 0.9 - 0.05 * i}
        for i in range(10)
    ]
    bm = temporal_quick_bind(
        feel={},
        attend={"focal_items": items},
    )
    # Top 5 by salience, descending
    assert len(bm.salient_focus) == 5
    salience_values = [it["salience"] for it in bm.salient_focus]
    assert salience_values == sorted(salience_values, reverse=True)


def test_quick_bind_handles_missing_focal_items():
    bm = temporal_quick_bind(feel={}, attend={"focal_items": None})
    assert bm.salient_focus == []


def test_quick_bind_leaves_confidence_at_default():
    """The whole point of the quick variant — these fields are NOT computable
    from FEEL+ATTEND alone (need PREDICT + MONITOR)."""
    bm = temporal_quick_bind(
        feel={"urgency": 0.9},
        attend={"focal_items": [{"id": "x", "salience": 0.8}]},
    )
    assert bm.confidence_unified == 0.5  # dataclass default
    assert bm.conflicts == []             # dataclass default
    assert bm.predict == {}
    assert bm.own == {}
    assert bm.monitor == {}


def test_quick_bind_does_not_apply_stability_bias():
    """temporal_bind weights items present across retention frames; the
    quick variant has no retention input and must not invent stability."""
    items = [
        {"id": "stable", "salience": 0.5},
        {"id": "fresh", "salience": 0.6},
    ]
    bm = temporal_quick_bind(feel={}, attend={"focal_items": items})
    # Higher salience wins regardless of any "stability" — fresh ranks first.
    assert bm.salient_focus[0]["id"] == "fresh"


def test_quick_bind_handles_none_inputs():
    """Defensive: callers may pass None on either argument."""
    bm = temporal_quick_bind(feel=None, attend=None)
    assert isinstance(bm, BoundMoment)
    assert bm.dominant_affect == "neutral"
    assert bm.salient_focus == []


def test_quick_bind_distinct_from_full_bind_with_same_inputs():
    """Sanity: full-bind with PREDICT+MONITOR computes a non-default
    confidence; quick-bind with the same FEEL+ATTEND does not."""
    feel = {"urgency": 0.9}
    attend = {"focal_items": [{"id": "x", "salience": 0.8}]}
    full = temporal_bind(
        feel=feel, attend=attend,
        predict={"confidence": 0.9},
        monitor={"confidence": 0.8},
    )
    quick = temporal_quick_bind(feel=feel, attend=attend)
    # Full bind weighted-averages predict + monitor confidences.
    assert full.confidence_unified != 0.5
    # Quick bind has no predict/monitor → stays at default.
    assert quick.confidence_unified == 0.5
