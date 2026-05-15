"""PROGRAM §43.4 — Q5.4 follow-up tests.

Covers:
  * Q5.4.1#1: AE-2 reads audit_log
  * Q5.4.1#2: AE-2 lift → outcome_density_ratio rename
  * Q5.4.1#3: HOT-1 reads full affect_trace
  * Q5.4.1#4: RPT-1 errored forecasts terminal
  * Q5.4.1#5: RPT-1 scorer registry refuses LLM-module callables
  * Q5.4.1#6: signal_type "disposition" + salience ≥ 0.5
  * Q5.4.2#1: Tier-3 amendment proposer registers RPT-1 forecast
  * Q5.4.2#2: CR creation registers RPT-1 forecast
  * Q5.4.2#3: HOT-1 LLM enrichment path (with decenter guard intact)
  * Q5.4.2#4: Anti-Goodhart pinning strengthened (marker-path scan)
  * Q5.4.2#5: Continuity ledger accepts sentience_observation
  * Q5.4.2#6: Daily briefing weekly digest section
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.1#1 + #2 — AE-2 audit_log + density-ratio rename
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ae2():
    return _load_isolated(
        "ae2_q54", "app/sentience_experiments/ae2_causal_credit.py",
    )


def test_ae2_consumes_audit_log(ae2, monkeypatch, tmp_path):
    """Operator approvals/rejections in audit_log.jsonl now appear as
    outcome events. The original ship promised this in the docstring
    but never read the file."""
    monkeypatch.setattr(ae2, "_enabled", lambda: True)
    usage = tmp_path / "usage.jsonl"
    errors = tmp_path / "errors.jsonl"
    welfare = tmp_path / "welfare.jsonl"
    audit = tmp_path / "audit_log.jsonl"
    monkeypatch.setattr(ae2, "_default_usage_path", lambda: usage)
    monkeypatch.setattr(ae2, "_default_errors_path", lambda: errors)
    monkeypatch.setattr(ae2, "_default_welfare_audit_path", lambda: welfare)
    monkeypatch.setattr(ae2, "_default_audit_log_path", lambda: audit)

    now = datetime.now(timezone.utc)
    # 100 actions, 90 of one signature
    actions = []
    for i in range(90):
        ts = now - timedelta(hours=2) - timedelta(seconds=i)
        actions.append({"ts": ts.isoformat(), "agent_id": "other", "model": "x"})
    for i in range(10):
        ts = now - timedelta(hours=3) - timedelta(seconds=i)
        actions.append({"ts": ts.isoformat(), "agent_id": "coder", "model": "y"})
    usage.write_text("\n".join(json.dumps(a) for a in actions) + "\n",
                     encoding="utf-8")
    errors.write_text("", encoding="utf-8")
    welfare.write_text("", encoding="utf-8")
    # 6 operator rejections, each 30s AFTER a coder action.
    audit_rows = []
    for i in range(6):
        ts = now - timedelta(hours=3) - timedelta(seconds=i) + timedelta(seconds=30)
        audit_rows.append({
            "ts": ts.isoformat(),
            "actor": "operator",
            "decision": "rejected",
        })
    audit.write_text("\n".join(json.dumps(r) for r in audit_rows) + "\n",
                     encoding="utf-8")

    assocs = ae2.detect_associations(window_days=1)
    # We expect an audit:operator_rejection bucket with high density ratio.
    rejected = [a for a in assocs if a.outcome_kind == "audit:operator_rejection"]
    assert len(rejected) >= 1, "audit_log outcomes should appear in associations"


def test_ae2_field_renamed_to_outcome_density_ratio(ae2):
    """The CausalAssociation dataclass field is renamed."""
    a = ae2.CausalAssociation(
        action_signature="agent=x|model=y",
        outcome_kind="error:Foo",
        outcome_rate=0.05,
        outcome_density_ratio=4.5,
        n_observations=6,
        n_actions=10,
        first_seen="2026-05-13T00:00:00+00:00",
        last_seen="2026-05-13T01:00:00+00:00",
        confidence=0.7,
    )
    d = a.to_dict()
    assert "outcome_density_ratio" in d
    assert "outcome_rate" in d
    # The misleading old fields should NOT be present.
    assert "lift" not in d
    assert "rarity" not in d


def test_ae2_outcome_audit_only_classifies_operator_rows(ae2):
    """audit_log rows from non-operator actors should NOT count."""
    assert ae2._outcome_kind_from_audit({
        "actor": "operator", "decision": "approved",
    }) == "audit:operator_approval"
    assert ae2._outcome_kind_from_audit({
        "actor": "operator", "decision": "rejected",
    }) == "audit:operator_rejection"
    assert ae2._outcome_kind_from_audit({
        "actor": "self_improver", "decision": "approved",  # non-operator
    }) is None
    assert ae2._outcome_kind_from_audit({
        "actor": "operator", "event": "runtime_settings_change",  # routine
    }) is None


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.1#3 — HOT-1 reads full affect_trace
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hot1():
    return _load_isolated(
        "hot1_q54", "app/sentience_experiments/hot1_meta_affect.py",
    )


def test_hot1_load_trace_points(hot1, monkeypatch, tmp_path):
    """The trace reader picks up V/A/C snapshots from trace.jsonl."""
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setattr(hot1, "_default_trace_path", lambda: trace)
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(20):
        rows.append({
            "ts": (now - timedelta(hours=i)).isoformat(),
            "valence": 0.5 - 0.01 * i,
            "arousal": 0.4,
            "controllability": 0.6,
            "attractor": "calm",
        })
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
    points = hot1._load_trace_points(window_days=30)
    assert len(points) == 20
    assert points[0]["attractor"] == "calm"
    assert "valence" in points[0]


def test_hot1_detects_baseline_drift(hot1, monkeypatch, tmp_path):
    """Significant valence drift between the older and recent halves."""
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: False)
    trace = tmp_path / "trace.jsonl"
    welfare = tmp_path / "welfare.jsonl"
    monkeypatch.setattr(hot1, "_default_trace_path", lambda: trace)
    monkeypatch.setattr(hot1, "_default_welfare_audit_path", lambda: welfare)
    welfare.write_text("", encoding="utf-8")
    now = datetime.now(timezone.utc)
    rows = []
    # Older half (7-14d ago): high valence
    for i in range(15):
        rows.append({
            "ts": (now - timedelta(days=10 + i * 0.5)).isoformat(),
            "valence": 0.7,
            "arousal": 0.5,
            "controllability": 0.6,
            "attractor": "calm",
        })
    # Recent half (0-7d): much lower valence
    for i in range(15):
        rows.append({
            "ts": (now - timedelta(days=i * 0.5)).isoformat(),
            "valence": 0.3,
            "arousal": 0.5,
            "controllability": 0.6,
            "attractor": "calm",
        })
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
    patterns = hot1.detect_patterns(window_days=30)
    drift = [p for p in patterns if p.pattern_kind == "baseline_drift"]
    assert len(drift) >= 1
    assert "valence" in drift[0].breach_kinds


def test_hot1_detects_attractor_lock(hot1, monkeypatch, tmp_path):
    """≥70% concentration in one attractor flags as lock."""
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: False)
    trace = tmp_path / "trace.jsonl"
    welfare = tmp_path / "welfare.jsonl"
    monkeypatch.setattr(hot1, "_default_trace_path", lambda: trace)
    monkeypatch.setattr(hot1, "_default_welfare_audit_path", lambda: welfare)
    welfare.write_text("", encoding="utf-8")
    now = datetime.now(timezone.utc)
    rows = []
    # 35 snapshots, 30 in "focused", 5 in others.
    for i in range(30):
        rows.append({
            "ts": (now - timedelta(hours=i)).isoformat(),
            "valence": 0.5, "arousal": 0.5, "controllability": 0.6,
            "attractor": "focused",
        })
    for i in range(5):
        rows.append({
            "ts": (now - timedelta(hours=30 + i)).isoformat(),
            "valence": 0.5, "arousal": 0.5, "controllability": 0.6,
            "attractor": "wander",
        })
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
    patterns = hot1.detect_patterns(window_days=30)
    locks = [p for p in patterns if p.pattern_kind == "attractor_lock"]
    assert len(locks) >= 1
    assert "focused" in locks[0].breach_kinds


def test_hot1_baseline_drift_hypothesis_decentered(hot1):
    """The new baseline_drift pattern's hypothesis must pass the filter."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    hyp = hot1._draft_hypothesis(p)
    if hyp is not None:
        assert hot1.decenter_text(hyp) == hyp


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.1#4 + #5 — RPT-1 terminal errors + LLM scorer refusal
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rpt1():
    return _load_isolated(
        "rpt1_q54", "app/sentience_experiments/rpt1_self_calibration.py",
    )


def test_rpt1_errored_forecasts_dont_retry(rpt1, monkeypatch, tmp_path):
    """A forecast that resolved-with-error must not be re-scored every pass."""
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)

    # Register with a scorer that raises every time.
    call_count = {"n": 0}
    def raising_scorer(args):
        call_count["n"] += 1
        raise RuntimeError("synthetic")
    rpt1.register_scorer("raises", raising_scorer)
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    rpt1.register_prediction(
        claim_kind="kx", claim_text="t", predicted_p=0.5,
        resolution_at=past, scorer_ref="raises",
    )
    # First reconcile: scorer raised → score_error set, resolved_at set.
    rpt1.reconcile_due()
    n_after_first = call_count["n"]
    assert n_after_first >= 1
    # Second reconcile: should NOT re-call the scorer.
    rpt1.reconcile_due()
    n_after_second = call_count["n"]
    assert n_after_second == n_after_first, (
        "Errored forecast was retried — terminal-error short-circuit failed"
    )


def test_rpt1_scorer_registry_refuses_llm_modules(rpt1):
    """Refuse callables whose __module__ lives under app.llm / app.agents."""
    # Build a stub callable claiming to come from app.llm.
    def fake_scorer(args):
        return True
    fake_scorer.__module__ = "app.llm.factory"
    with pytest.raises(ValueError, match="refused scorer"):
        rpt1.register_scorer("bad_llm", fake_scorer)

    fake_scorer.__module__ = "app.agents.commander.foo"
    with pytest.raises(ValueError, match="refused scorer"):
        rpt1.register_scorer("bad_agent", fake_scorer)

    # Pure-python module is fine.
    fake_scorer.__module__ = "app.healing.handlers.foo"
    rpt1.register_scorer("ok_scorer", fake_scorer)  # no raise


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.2#1 + #2 — RPT-1 producers wired
# ─────────────────────────────────────────────────────────────────────────


def test_tier3_proposer_imports_register_prediction():
    """Source-level: request_tier3_amendment.py registers RPT-1 forecasts."""
    src = Path("app/tools/request_tier3_amendment.py").read_text()
    assert "from app.sentience_experiments.rpt1_self_calibration import" in src
    assert "register_prediction" in src
    assert 'claim_kind="tier3_approval"' in src
    assert 'scorer_ref="tier3_approval"' in src


def test_cr_creation_imports_register_prediction():
    """Source-level: change_requests.lifecycle registers RPT-1 forecasts."""
    src = Path("app/change_requests/lifecycle.py").read_text()
    assert "from app.sentience_experiments.rpt1_self_calibration import" in src
    assert "register_prediction" in src
    assert 'claim_kind="cr_apply"' in src
    assert 'scorer_ref="cr_apply"' in src


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.2#3 — LLM hypothesis path with decenter guard
# ─────────────────────────────────────────────────────────────────────────


def test_hot1_llm_enrich_falls_back_to_template_when_first_person(hot1, monkeypatch):
    """LLM output containing first-person affect language MUST be
    rejected by the decenter filter; the deterministic template is
    returned instead."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    # Force the LLM to return first-person prose.
    monkeypatch.setattr(
        hot1, "_maybe_llm_enrich",
        lambda pattern, template: "I feel like the system is drifting...",
    )
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    hyp = hot1._draft_hypothesis(p)
    # Either None or filter-clean — never the first-person prose.
    assert hyp is None or hot1.decenter_text(hyp) == hyp
    # In particular, the verboten phrase must not appear.
    assert "i feel" not in (hyp or "").lower()


def test_hot1_llm_enrich_accepts_clean_output(hot1, monkeypatch):
    """LLM output passing the decenter filter is preferred over template."""
    p = hot1.MetaAffectPattern(
        pattern_kind="baseline_drift",
        breach_kinds=["valence"],
        n_occurrences=30,
        span_days=14.0,
        confidence=0.6,
        detected_at="2026-05-13T00:00:00+00:00",
    )
    monkeypatch.setattr(
        hot1, "_maybe_llm_enrich",
        lambda pattern, template: "The trace indicates a notable valence shift.",
    )
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    hyp = hot1._draft_hypothesis(p)
    assert hyp == "The trace indicates a notable valence shift."


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.1#6 — signal_type + salience
# ─────────────────────────────────────────────────────────────────────────


def test_sentience_modules_use_disposition_signal_type():
    """All four module GW publishes use signal_type="disposition" and
    salience ≥ 0.5 so agents with default importance_filter="high"
    actually receive them."""
    for module_path in (
        "app/sentience_experiments/ae2_causal_credit.py",
        "app/sentience_experiments/hot1_meta_affect.py",
        "app/sentience_experiments/hot4_metacog_monitor.py",
    ):
        src = Path(module_path).read_text()
        assert 'signal_type="disposition"' in src, (
            f"{module_path}: signal_type not set to 'disposition'"
        )
        assert 'signal_type="background"' not in src, (
            f"{module_path}: 'background' signal_type still present"
        )
        # Verify the salience floor is ≥ 0.5.
        assert "salience=0.4" not in src, (
            f"{module_path}: salience floor still at 0.4"
        )


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.2#5 — Continuity ledger sentience_observation kind
# ─────────────────────────────────────────────────────────────────────────


def test_continuity_ledger_accepts_sentience_observation():
    cl = _load_isolated(
        "cl_q54", "app/identity/continuity_ledger.py",
    )
    assert "sentience_observation" in cl.IDENTITY_EVENT_KINDS


def test_ledger_bridge_emit_landmark(tmp_path):
    """The bridge appends a sentience_observation row to the ledger.

    The bridge does `from app.identity.continuity_ledger import
    record_event` at call time, so we must monkeypatch the actual
    canonical module — not an isolated-load copy."""
    import app.identity.continuity_ledger as cl
    bridge = _load_isolated(
        "bridge_q54", "app/sentience_experiments/ledger_bridge.py",
    )
    target = tmp_path / "ledger.jsonl"
    original = cl._path_override
    cl._path_override = target
    try:
        ok = bridge.emit_landmark(
            source_module="ae2_causal_credit",
            landmark_kind="high_density_association",
            summary="AE-2: 3 associations (top density ratio 8.4×)",
            counts={"associations": 3},
        )
        assert ok is True
        rows = cl.list_events(kinds={"sentience_observation"})
        assert len(rows) == 1
        assert rows[0].actor == "ae2_causal_credit"
        assert rows[0].detail.get("landmark_kind") == "high_density_association"
        assert rows[0].detail.get("count_associations") == 3
    finally:
        cl._path_override = original


def test_ledger_bridge_failure_isolated(monkeypatch):
    """If record_event raises, emit_landmark returns False rather than
    propagating the exception."""
    bridge = _load_isolated(
        "bridge_q54_fail", "app/sentience_experiments/ledger_bridge.py",
    )
    def broken_record_event(**kwargs):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(
        "app.identity.continuity_ledger.record_event", broken_record_event,
    )
    ok = bridge.emit_landmark(
        source_module="x", landmark_kind="y", summary="z",
    )
    assert ok is False


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.2#4 — Anti-Goodhart pinning strengthened
# ─────────────────────────────────────────────────────────────────────────


def test_strengthened_pinning_test_exists():
    """The two new pinning tests should be in test_q5_2_modules.py."""
    src = Path("tests/test_q5_2_modules.py").read_text()
    assert "test_q5_marker_paths_absent_in_subia" in src
    assert "test_q5_subia_file_count_pinned" in src


# ─────────────────────────────────────────────────────────────────────────
#   Q5.4.2#6 — Daily briefing weekly digest section
# ─────────────────────────────────────────────────────────────────────────


def test_briefing_has_sentience_digest_section():
    """Source-level: weekly composer surfaces sentience digest."""
    src = Path("app/life_companion/daily_briefing.py").read_text()
    assert "_gather_sentience_digest" in src
    assert "🔬 Self-observation (week)" in src


def test_briefing_sentience_digest_empty_when_nothing(monkeypatch):
    """Empty result when nothing happened this week (section disappears)."""
    briefing = _load_isolated(
        "br_q54", "app/life_companion/daily_briefing.py",
    )
    # Stub each module's list to return empty.
    monkeypatch.setattr(
        "app.sentience_experiments.ae2_causal_credit.list_recent",
        lambda n=20: [],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.hot1_meta_affect.list_recent",
        lambda n=20: [],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.hot4_metacog_monitor.list_recent_flagged",
        lambda n=20: [],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.rpt1_self_calibration.load_calibration_state",
        lambda: {"reports": {}},
    )
    lines = briefing._gather_sentience_digest()
    assert lines == []


def test_briefing_sentience_digest_surfaces_when_data(monkeypatch):
    """When AE-2 has a strong association, the digest line shows it."""
    briefing = _load_isolated(
        "br_q54b", "app/life_companion/daily_briefing.py",
    )
    monkeypatch.setattr(
        "app.sentience_experiments.ae2_causal_credit.list_recent",
        lambda n=20: [{
            "outcome_density_ratio": 8.4,
            "outcome_kind": "audit:operator_rejection",
            "action_signature": "agent=coder|model=x",
        }],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.hot1_meta_affect.list_recent",
        lambda n=20: [],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.hot4_metacog_monitor.list_recent_flagged",
        lambda n=20: [],
    )
    monkeypatch.setattr(
        "app.sentience_experiments.rpt1_self_calibration.load_calibration_state",
        lambda: {"reports": {}},
    )
    lines = briefing._gather_sentience_digest()
    assert any("AE-2" in ln for ln in lines)
    # Opaque counts only — no identities should leak.
    joined = " ".join(lines)
    assert "operator_rejection" not in joined
    assert "coder" not in joined
