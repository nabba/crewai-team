"""Tests for the Q18 drill baseline + comparison (app/resilience_drills/baseline.py)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_baseline_dir(monkeypatch, tmp_path):
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    yield


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _obs(drill, **measurements):
    from app.resilience_drills.baseline import Observation
    return Observation(drill_name=drill, observed_at=_now_iso(),
                       measurements=measurements)


def test_round_trip_observation():
    from app.resilience_drills.baseline import Observation
    o = _obs("d", a=1, b="two")
    d = o.to_dict()
    o2 = Observation.from_dict(d)
    assert o2.measurements == o.measurements


def test_ratify_persists_baseline():
    from app.resilience_drills import baseline as bl
    o = _obs("vendor_independence", n_fallbacks=1, providers_ready=["groq"])
    b = bl.ratify_from_observation(o, operator="operator-react",
                                    notes="single fallback acceptable for our posture")
    assert b.drill_name == "vendor_independence"
    loaded = bl.load("vendor_independence")
    assert loaded is not None
    assert loaded.measurements["n_fallbacks"] == 1


def test_compare_exact_match_passes():
    from app.resilience_drills import baseline as bl
    o = _obs("d", x=1, y="z")
    b = bl.ratify_from_observation(o, operator="op")
    r = bl.compare(o, b)
    assert r.ok is True
    assert r.regressions == []


def test_compare_exact_mismatch_fails():
    from app.resilience_drills import baseline as bl
    baseline_obs = _obs("d", x=1)
    b = bl.ratify_from_observation(baseline_obs, operator="op")
    later = _obs("d", x=2)
    r = bl.compare(later, b)
    assert r.ok is False
    assert len(r.regressions) == 1
    assert r.regressions[0].key == "x"


def test_compare_min_rule():
    from app.resilience_drills import baseline as bl
    bo = _obs("d", n=2)
    b = bl.ratify_from_observation(bo, operator="op",
                                    tolerances={"n": {"rule": "min", "value": 1}})
    # observed=2 ≥ min=1 → pass
    assert bl.compare(_obs("d", n=2), b).ok is True
    # observed=0 < min=1 → fail
    r = bl.compare(_obs("d", n=0), b)
    assert r.ok is False
    assert r.regressions[0].rule == "min"


def test_compare_max_rule():
    from app.resilience_drills import baseline as bl
    bo = _obs("d", n=3)
    b = bl.ratify_from_observation(bo, operator="op",
                                    tolerances={"n": {"rule": "max", "value": 5}})
    assert bl.compare(_obs("d", n=5), b).ok is True
    assert bl.compare(_obs("d", n=6), b).ok is False


def test_compare_range_rule():
    from app.resilience_drills import baseline as bl
    bo = _obs("d", n=3)
    b = bl.ratify_from_observation(bo, operator="op",
                                    tolerances={"n": {"rule": "range", "min": 1, "max": 5}})
    assert bl.compare(_obs("d", n=3), b).ok is True
    assert bl.compare(_obs("d", n=0), b).ok is False
    assert bl.compare(_obs("d", n=6), b).ok is False


def test_compare_subset_of_rule():
    """subset_of: observation list MUST be a subset of the allowed list.
    Gained members ARE violations."""
    from app.resilience_drills import baseline as bl
    bo = _obs("d", providers=["groq", "anthropic"])
    b = bl.ratify_from_observation(
        bo, operator="op",
        tolerances={"providers": {"rule": "subset_of",
                                   "value": ["groq", "anthropic"]}},
    )
    assert bl.compare(_obs("d", providers=["groq"]), b).ok is True
    assert bl.compare(_obs("d", providers=["groq", "anthropic"]), b).ok is True
    # New unexpected provider in observation → violation
    r = bl.compare(_obs("d", providers=["groq", "anthropic", "openai"]), b)
    assert r.ok is False


def test_compare_superset_of_rule_catches_regression():
    """superset_of is the load-bearing rule for vendor_independence:
    'observation must include every member of the required list'. Lost
    members are regressions — the operator's chosen baseline degraded."""
    from app.resilience_drills import baseline as bl
    bo = _obs("d", providers_ready=["groq", "anthropic"])
    b = bl.ratify_from_observation(
        bo, operator="op",
        tolerances={"providers_ready": {"rule": "superset_of",
                                         "value": ["groq", "anthropic"]}},
    )
    assert bl.compare(_obs("d", providers_ready=["groq", "anthropic"]), b).ok is True
    # Lost groq → regression
    r = bl.compare(_obs("d", providers_ready=["anthropic"]), b)
    assert r.ok is False
    assert r.regressions[0].rule == "superset_of"
    assert "groq" in r.regressions[0].detail


def test_compare_missing_key_flagged_separately():
    """Baseline keys absent from observation are NOT regressions but
    they DO get flagged in missing_keys for operator review."""
    from app.resilience_drills import baseline as bl
    bo = _obs("d", x=1, y=2)
    b = bl.ratify_from_observation(bo, operator="op")
    r = bl.compare(_obs("d", x=1), b)
    assert r.ok is False
    assert r.missing_keys == ["y"]
    assert r.regressions == []  # not a value mismatch — a structural one


def test_compare_extra_observation_key_not_regression():
    """Observation may add new measurement keys that didn't exist when
    the baseline was ratified — that's evolution, not regression."""
    from app.resilience_drills import baseline as bl
    bo = _obs("d", x=1)
    b = bl.ratify_from_observation(bo, operator="op")
    r = bl.compare(_obs("d", x=1, new_thing=42), b)
    assert r.ok is True


def test_compare_unknown_rule_surfaces_as_regression():
    """Defensive: an unknown rule name is treated as a regression so
    it surfaces in operator view rather than silently passing."""
    from app.resilience_drills import baseline as bl
    bo = _obs("d", x=1)
    b = bl.ratify_from_observation(
        bo, operator="op",
        tolerances={"x": {"rule": "made_up_rule"}},
    )
    r = bl.compare(_obs("d", x=1), b)
    assert r.ok is False
    assert "unknown" in r.regressions[0].detail.lower()


def test_list_all_baselines():
    from app.resilience_drills import baseline as bl
    bl.ratify_from_observation(_obs("a", n=1), operator="op")
    bl.ratify_from_observation(_obs("b", n=2), operator="op")
    out = bl.list_all_baselines()
    names = sorted(b.drill_name for b in out)
    assert names == ["a", "b"]


def test_load_returns_none_for_unratified_drill():
    from app.resilience_drills import baseline as bl
    assert bl.load("never_ratified") is None


def test_vendor_independence_baseline_use_case():
    """The load-bearing use case from PROGRAM §57: operator ratifies
    'one fallback + Groq is acceptable; alert me only if it gets
    worse'. Future runs compare to that. Adding providers is fine;
    losing them is a regression."""
    from app.resilience_drills import baseline as bl
    # Initial observation: 1 fallback (groq), no ollama
    initial = _obs(
        "vendor_independence",
        n_fallbacks=1,
        providers_ready=["anthropic", "groq"],
        ollama_reachable=False,
    )
    # Operator says: this is fine. Lock in min n_fallbacks=1,
    # superset of [groq] (anthropic is dominant, not a "fallback").
    b = bl.ratify_from_observation(
        initial, operator="operator-react",
        tolerances={
            "n_fallbacks": {"rule": "min", "value": 1},
            "providers_ready": {"rule": "superset_of", "value": ["groq"]},
            "ollama_reachable": {"rule": "exact"},
        },
        notes="single non-dominant fallback acceptable for posture",
    )
    # Future observation: same state → OK
    same = _obs("vendor_independence", n_fallbacks=1,
                providers_ready=["anthropic", "groq"], ollama_reachable=False)
    assert bl.compare(same, b).ok is True

    # Future observation: groq key expired → regression
    degraded = _obs("vendor_independence", n_fallbacks=0,
                    providers_ready=["anthropic"], ollama_reachable=False)
    r = bl.compare(degraded, b)
    assert r.ok is False
    keys = sorted(reg.key for reg in r.regressions)
    assert "n_fallbacks" in keys
    assert "providers_ready" in keys

    # Future observation: added ollama → still HEALTHY (more fallbacks
    # is better, ollama_reachable changed but operator ratified True
    # ... wait, no, they ratified False. So adding ollama IS a change.
    # The point is: superset_of is satisfied, n_fallbacks may have
    # gone up, but the ollama_reachable=True vs False is an exact
    # mismatch. That's deliberate — if operator ratified "no ollama"
    # and ollama appears, the configuration drifted and they should
    # know.)
    new_ollama = _obs("vendor_independence", n_fallbacks=2,
                      providers_ready=["anthropic", "groq", "ollama"],
                      ollama_reachable=True)
    r = bl.compare(new_ollama, b)
    # Mixed: providers_ready OK (still superset), n_fallbacks OK (≥1),
    # but ollama_reachable diverged — that surfaces as a regression
    # which the operator can re-ratify.
    assert r.ok is False
    assert any(reg.key == "ollama_reachable" for reg in r.regressions)
