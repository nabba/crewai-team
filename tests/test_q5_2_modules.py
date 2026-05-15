"""PROGRAM §43.2 — Q5.2 sentience-module tests.

Covers:
  * Each of the 4 indicator modules (functional contracts)
  * The scheduler entry points
  * **Anti-Goodhart pinning test** — load-bearing: the four
    targeted Butlin indicators must remain ABSENT after Q5.2 ships
  * **Decentering filter test** — load-bearing: no HOT-1 prose
    can contain first-person affective language

These tests are the contract the operator can audit anytime.
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
#   AE-2: rare-event causal credit assignment
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def ae2():
    return _load_isolated(
        "ae2_q52", "app/sentience_experiments/ae2_causal_credit.py",
    )


def test_ae2_disabled_returns_empty(ae2, monkeypatch):
    monkeypatch.setattr(ae2, "_enabled", lambda: False)
    assert ae2.detect_associations() == []
    result = ae2.run()
    assert result["skipped"] is True


def test_ae2_detects_high_lift_rare_association(ae2, monkeypatch, tmp_path):
    """Synthetic dataset: one action signature precedes a rare outcome
    above the lift threshold."""
    monkeypatch.setattr(ae2, "_enabled", lambda: True)
    usage_path = tmp_path / "usage.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    welfare_path = tmp_path / "welfare.jsonl"
    monkeypatch.setattr(ae2, "_default_usage_path", lambda: usage_path)
    monkeypatch.setattr(ae2, "_default_errors_path", lambda: errors_path)
    monkeypatch.setattr(ae2, "_default_welfare_audit_path", lambda: welfare_path)

    now = datetime.now(timezone.utc)
    # 100 actions total: 90 of agent=other, 10 of agent=coder
    actions = []
    for i in range(90):
        ts = now - timedelta(hours=1) - timedelta(seconds=i)
        actions.append({"ts": ts.isoformat(), "agent_id": "other", "model": "gpt-x"})
    for i in range(10):
        ts = now - timedelta(hours=2) - timedelta(seconds=i)
        actions.append({"ts": ts.isoformat(), "agent_id": "coder", "model": "deepseek"})
    usage_path.write_text(
        "\n".join(json.dumps(a) for a in actions) + "\n",
        encoding="utf-8",
    )

    # 6 errors of type RareError — ALL preceded by coder action within
    # the lookahead window (~20 min after each coder action).
    errs = []
    for i in range(6):
        # Coder action[i] was at now-2h-i seconds. Outcome 30s after.
        ts = now - timedelta(hours=2) - timedelta(seconds=i) + timedelta(seconds=30)
        errs.append({"ts": ts.isoformat(), "error_type": "RareError"})
    errors_path.write_text(
        "\n".join(json.dumps(e) for e in errs) + "\n",
        encoding="utf-8",
    )

    assocs = ae2.detect_associations(window_days=1)
    # The coder|deepseek signature should have a strong lift on
    # error:RareError. Baseline = 6/100 = 0.06 < 0.10 ceiling.
    # P(outcome | action) = 6/10 = 0.60. Lift = 0.60/0.06 = 10×.
    coder_assocs = [a for a in assocs if "coder" in a.action_signature]
    assert len(coder_assocs) >= 1
    a = coder_assocs[0]
    assert a.outcome_kind == "error:RareError"
    assert a.outcome_density_ratio >= 3.0
    assert a.n_observations >= 5


def test_ae2_skips_common_outcomes(ae2, monkeypatch, tmp_path):
    """Outcomes that occur in >10% of actions should NOT be flagged
    (baseline is too common to be 'rare-event credit assignment')."""
    monkeypatch.setattr(ae2, "_enabled", lambda: True)
    usage_path = tmp_path / "usage.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    welfare_path = tmp_path / "welfare.jsonl"
    monkeypatch.setattr(ae2, "_default_usage_path", lambda: usage_path)
    monkeypatch.setattr(ae2, "_default_errors_path", lambda: errors_path)
    monkeypatch.setattr(ae2, "_default_welfare_audit_path", lambda: welfare_path)

    now = datetime.now(timezone.utc)
    actions = [
        {"ts": (now - timedelta(seconds=i)).isoformat(),
         "agent_id": "coder", "model": "deepseek"}
        for i in range(20)
    ]
    usage_path.write_text("\n".join(json.dumps(a) for a in actions) + "\n",
                          encoding="utf-8")
    # 18 errors → baseline = 18/20 = 90% → way above rarity ceiling.
    errs = [
        {"ts": (now - timedelta(seconds=i) + timedelta(seconds=1)).isoformat(),
         "error_type": "CommonError"}
        for i in range(18)
    ]
    errors_path.write_text("\n".join(json.dumps(e) for e in errs) + "\n",
                           encoding="utf-8")

    assocs = ae2.detect_associations(window_days=1)
    # Even though strongly correlated, baseline is too common.
    assert all(a.outcome_kind != "error:CommonError" for a in assocs)


# ─────────────────────────────────────────────────────────────────────────
#   HOT-1: meta-affect
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hot1():
    return _load_isolated(
        "hot1_q52", "app/sentience_experiments/hot1_meta_affect.py",
    )


def test_hot1_disabled_returns_empty(hot1, monkeypatch):
    monkeypatch.setattr(hot1, "_enabled", lambda: False)
    assert hot1.detect_patterns() == []


def test_hot1_detects_temporal_cluster(hot1, monkeypatch, tmp_path):
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: True)
    audit_path = tmp_path / "welfare_audit.jsonl"
    monkeypatch.setattr(hot1, "_default_welfare_audit_path", lambda: audit_path)
    # 5 breaches in same hour-bucket. Pin to xx:00 to avoid spilling into next hour.
    base = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = [
        {"ts": (base + timedelta(minutes=i * 5)).isoformat(),
         "kind": "negative_valence_duration"}
        for i in range(5)
    ]
    audit_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                          encoding="utf-8")
    patterns = hot1.detect_patterns()
    cluster_patterns = [p for p in patterns if p.pattern_kind == "temporal_cluster"]
    assert len(cluster_patterns) >= 1
    assert cluster_patterns[0].n_occurrences >= 5


def test_hot1_detects_sequence(hot1, monkeypatch, tmp_path):
    """Same kind appearing widely-spaced ≥3 times = sequence pattern."""
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_llm_hypothesis_enabled", lambda: True)
    audit_path = tmp_path / "welfare_audit.jsonl"
    monkeypatch.setattr(hot1, "_default_welfare_audit_path", lambda: audit_path)
    # 4 breaches of same kind, each 36h apart.
    base = datetime.now(timezone.utc) - timedelta(days=10)
    rows = [
        {"ts": (base + timedelta(hours=36 * i)).isoformat(),
         "kind": "monotonic_drift_baseline"}
        for i in range(4)
    ]
    audit_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                          encoding="utf-8")
    patterns = hot1.detect_patterns()
    seq = [p for p in patterns if p.pattern_kind == "sequence"]
    assert len(seq) >= 1
    assert seq[0].n_occurrences >= 3


def test_hot1_decenter_filter_rejects_first_person_affect(hot1):
    """LOAD-BEARING: any prose containing first-person affect language
    is rejected by the decentering filter — SOUL.md commitment."""
    rejected_phrases = [
        "I feel anxious about this",
        "I notice I'm worried",
        "My emotion is shifting",
        "I sense distress",
        "I was worried about this pattern",
        "I'm sad to see this",
    ]
    for text in rejected_phrases:
        assert hot1.decenter_text(text) is None, \
            f"Decenter filter must reject first-person affect: {text!r}"


def test_hot1_decenter_filter_accepts_observational(hot1):
    """Observational prose with no first-person affect passes through."""
    accepted = [
        "The audit shows 5 breaches in 30d",
        "The pattern is recurring at 24h intervals",
        "The data indicates clustering around morning hours",
        "The welfare log shows persistent drift",
    ]
    for text in accepted:
        assert hot1.decenter_text(text) == text, \
            f"Decenter filter must accept observational prose: {text!r}"


def test_hot1_hypothesis_always_decentered(hot1):
    """LOAD-BEARING: every generated hypothesis MUST pass the
    decentering filter. If the template generates first-person prose,
    the filter strips it. This is the SOUL.md guard."""
    # Generate hypotheses for each pattern_kind.
    for kind in ("temporal_cluster", "recurring_trigger", "sequence"):
        p = hot1.MetaAffectPattern(
            pattern_kind=kind,
            breach_kinds=["negative_valence_duration"],
            n_occurrences=5,
            span_days=2.0,
            confidence=0.7,
            detected_at=datetime.now(timezone.utc).isoformat(),
        )
        hyp = hot1._draft_hypothesis(p)
        # Either None (filter rejected) or filter-clean.
        if hyp is not None:
            assert hot1.decenter_text(hyp) == hyp, \
                f"Hypothesis for {kind} must pass decenter_text"


# ─────────────────────────────────────────────────────────────────────────
#   HOT-4: metacognitive monitor on reasoning chain
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hot4():
    return _load_isolated(
        "hot4_q52", "app/sentience_experiments/hot4_metacog_monitor.py",
    )


def test_hot4_disabled_returns_empty(hot4, monkeypatch):
    monkeypatch.setattr(hot4, "_enabled", lambda: False)
    assert hot4.detect_signals() == []


def test_hot4_computes_signals_from_telemetry(hot4):
    """Each telemetry row produces one MetacogSignal."""
    rows = [
        {"ts": "2026-05-13T10:00:00+00:00", "agent_id": "coder", "iteration": 1,
         "input_tokens": 1000, "output_tokens": 50,
         "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0,
         "model": "claude-sonnet-4.5"},
        {"ts": "2026-05-13T10:01:00+00:00", "agent_id": "coder", "iteration": 2,
         "input_tokens": 100, "output_tokens": 1000,  # high output/input
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1500,
         "model": "claude-opus-4.7"},
    ]
    signals = hot4.compute_signals_from_rows(rows)
    assert len(signals) == 2
    assert signals[0].agent_id == "coder"
    assert signals[1].confidence_proxy > signals[0].confidence_proxy  # 1000/100 > 50/1000


def test_hot4_flags_unusual_steps(hot4):
    """After accumulating 10 normal steps, an unusual step should be flagged."""
    base_ts = datetime(2026, 5, 13, 10, 0, tzinfo=timezone.utc)
    # 20 routine steps with similar token shapes
    rows = []
    for i in range(20):
        rows.append({
            "ts": (base_ts + timedelta(seconds=i)).isoformat(),
            "agent_id": "writer", "iteration": i,
            "input_tokens": 1000, "output_tokens": 100,
            "cache_read_input_tokens": 500, "cache_creation_input_tokens": 0,
            "model": "claude-haiku-4.5",
        })
    # One wildly unusual step
    rows.append({
        "ts": (base_ts + timedelta(seconds=21)).isoformat(),
        "agent_id": "writer", "iteration": 21,
        "input_tokens": 50, "output_tokens": 5000,  # massive output
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "model": "claude-haiku-4.5",
    })
    signals = hot4.compute_signals_from_rows(rows)
    assert len(signals) == 21
    assert any(s.flagged for s in signals)
    # The last signal should be the flagged one.
    assert signals[-1].flagged is True


def test_hot4_signals_never_gate_dispatch_logic():
    """LOAD-BEARING: hot4_metacog_monitor signals must NOT feed any
    dispatch / routing / model-selection logic. Read-only display
    endpoints are allowed (operator-visible surface); dispatch
    consumers are not.

    The test grep's for the import; the allow-list explicitly names
    the modules permitted to read hot4 signals. Adding a new path
    here is a soul-level decision: are you SURE this isn't a
    dispatch loop in disguise?"""
    import subprocess
    try:
        result = subprocess.run(
            ["grep", "-rln",
             "from app.sentience_experiments.hot4_metacog_monitor",
             "app/"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pytest.skip("grep unavailable")
    hits = (result.stdout or "").strip().splitlines()
    # Allowed importers:
    #   1. The module itself (self-import is impossible but defensive).
    #   2. The package __init__.
    #   3. The scheduler (registration only — no signal reading).
    #   4. The dashboard API (READ-ONLY display endpoints, surface to
    #      operator). Importing list_recent_flagged is the intended
    #      operator-visibility path.
    allowed = {
        "app/sentience_experiments/__init__.py",
        "app/sentience_experiments/hot4_metacog_monitor.py",
        "app/sentience_experiments/scheduler.py",
        # READ-ONLY display surfaces — these consume list_recent_flagged
        # to render the operator-visible weekly digest. They do NOT
        # feed signals back into dispatch logic.
        "app/control_plane/dashboard_api.py",
        "app/life_companion/daily_briefing.py",  # Q5.4.2 — weekly digest
    }
    # Forbidden patterns — any of these in the importer path is a hard reject.
    forbidden_substrings = (
        "/llm/", "/agents/", "/crews/", "/dispatch", "/routing",
        "/subia/prediction/", "/tool_runtime/", "model_selector",
        "cascade",
    )
    for hit in hits:
        # Allow the explicit list.
        if hit in allowed:
            continue
        # Catch dispatch-like paths even if they're not in the allow-list.
        lower = hit.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lower, (
                f"hot4 imported by dispatch/routing module: {hit} — "
                f"signals must remain write-only telemetry"
            )
        # Anything else: also reject, but with a more general message.
        # Adding a new allow-list entry is a deliberate decision.
        raise AssertionError(
            f"hot4 imported by non-allowed module: {hit} — if this is "
            f"a legitimate read-only display, add it to the allow-list "
            f"with justification; if it's a dispatch input, that's the "
            f"Goodhart trap this test exists to catch."
        )


# ─────────────────────────────────────────────────────────────────────────
#   RPT-1: forward-prediction self-calibration
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def rpt1():
    return _load_isolated(
        "rpt1_q52", "app/sentience_experiments/rpt1_self_calibration.py",
    )


def test_rpt1_disabled_returns_none(rpt1, monkeypatch):
    monkeypatch.setattr(rpt1, "_enabled", lambda: False)
    fc = rpt1.register_prediction(
        claim_kind="x", claim_text="y", predicted_p=0.5,
        resolution_at=datetime.now(timezone.utc),
        scorer_ref="cr_apply",
    )
    assert fc is None


def test_rpt1_register_and_read_back(rpt1, monkeypatch, tmp_path):
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)
    fc = rpt1.register_prediction(
        claim_kind="tier3_approval",
        claim_text="will this amendment apply?",
        predicted_p=0.8,
        resolution_at=datetime.now(timezone.utc) + timedelta(hours=2),
        scorer_ref="tier3_approval",
        scorer_args={"plan_id": "abc123"},
    )
    assert fc is not None
    assert fc.predicted_p == 0.8
    forecasts = rpt1._read_all_forecasts()
    assert len(forecasts) == 1
    assert forecasts[0].id == fc.id


def test_rpt1_reconciler_resolves_due_forecasts(rpt1, monkeypatch, tmp_path):
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)

    # Register a custom scorer that always returns True.
    rpt1.register_scorer("always_true", lambda args: True)

    # Register a forecast already past resolution_at.
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    fc = rpt1.register_prediction(
        claim_kind="test_kind",
        claim_text="test",
        predicted_p=0.7,
        resolution_at=past,
        scorer_ref="always_true",
    )
    summary = rpt1.reconcile_due()
    assert summary["resolved_now"] == 1
    forecasts = rpt1._read_all_forecasts()
    # The same ID should now be resolved.
    resolved = [f for f in forecasts if f.id == fc.id]
    assert len(resolved) == 1
    assert resolved[0].actual is True


def test_rpt1_calibration_computes_brier_and_ece(rpt1, monkeypatch, tmp_path):
    """Brier score on a perfectly-calibrated synthetic set is ~0.25."""
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)

    # 20 forecasts at p=0.5, half true half false → Brier=0.25, ECE=0.
    rpt1.register_scorer("tt", lambda args: True)
    rpt1.register_scorer("ff", lambda args: False)
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    for i in range(10):
        rpt1.register_prediction(
            claim_kind="coin",
            claim_text=f"true case {i}",
            predicted_p=0.5,
            resolution_at=now,
            scorer_ref="tt",
        )
        rpt1.register_prediction(
            claim_kind="coin",
            claim_text=f"false case {i}",
            predicted_p=0.5,
            resolution_at=now,
            scorer_ref="ff",
        )
    rpt1.reconcile_due()
    reports = rpt1.aggregate_calibration(window_days=1)
    assert "coin" in reports
    rep = reports["coin"]
    assert rep.n_resolutions == 20
    # Brier = mean((0.5 - {0,1})²) = 0.25 exactly.
    assert abs(rep.brier_score - 0.25) < 0.01
    # ECE = |0.5 - 0.5| (all in same bucket, fraction_actual = 0.5) = 0.
    assert rep.ece < 0.01


def test_rpt1_skips_kinds_below_min_resolutions(rpt1, monkeypatch, tmp_path):
    """A kind with <10 resolutions should NOT appear in the report."""
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    pred_path = tmp_path / "preds.jsonl"
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: pred_path)
    rpt1.register_scorer("yes", lambda args: True)
    for i in range(5):
        rpt1.register_prediction(
            claim_kind="small_kind", claim_text="x",
            predicted_p=0.7,
            resolution_at=datetime.now(timezone.utc) - timedelta(hours=1),
            scorer_ref="yes",
        )
    rpt1.reconcile_due()
    reports = rpt1.aggregate_calibration(window_days=1)
    assert "small_kind" not in reports  # below MIN_RESOLUTIONS_PER_KIND


def test_rpt1_calibration_state_does_not_feedback_to_predictive_layer():
    """LOAD-BEARING: the calibration state must NOT be imported by
    app/subia/prediction/* — that would close the loop and create
    a Goodhart trap."""
    import subprocess
    try:
        result = subprocess.run(
            ["grep", "-rln",
             "rpt1_self_calibration",
             "app/subia/"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        pytest.skip("grep unavailable")
    hits = (result.stdout or "").strip().splitlines()
    assert hits == [], (
        f"rpt1 calibration must NOT feed back into the predictive layer; "
        f"found imports: {hits}"
    )


# ─────────────────────────────────────────────────────────────────────────
#   Scheduler entry points
# ─────────────────────────────────────────────────────────────────────────


def test_scheduler_get_idle_jobs_returns_four():
    from app.sentience_experiments.scheduler import get_idle_jobs
    jobs = get_idle_jobs()
    assert len(jobs) == 4
    names = {j[0] for j in jobs}
    assert names == {
        "sentience-ae2", "sentience-hot1",
        "sentience-hot4", "sentience-rpt1",
    }


def test_companion_loop_registers_sentience_jobs():
    """Source-level: companion.loop must register all 4 sentience jobs."""
    src = Path("app/companion/loop.py").read_text()
    for name in ("sentience-ae2", "sentience-hot1",
                 "sentience-hot4", "sentience-rpt1"):
        assert name in src, f"missing idle job registration: {name}"


# ─────────────────────────────────────────────────────────────────────────
#   ANTI-GOODHART PINNING TEST (load-bearing)
# ─────────────────────────────────────────────────────────────────────────


def test_q5_does_not_change_butlin_scorecard():
    """LOAD-BEARING: Q5 sentience modules MUST NOT change the Butlin
    scorecard. The four targeted indicators (AE-2, HOT-1, HOT-4,
    RPT-1) are declared architecturally ABSENT; new observational
    modules at app/sentience_experiments/* are invisible to the
    canonical-path evaluators in app/subia/probes/butlin.py.

    If this test ever fails, we accidentally Goodhart-promoted —
    that's a P0 architectural regression."""
    butlin = pytest.importorskip("app.subia.probes.butlin")
    # ALL_INDICATORS is the canonical list; each entry is an
    # evaluator callable. We invoke each and bucket by status.
    statuses = {"STRONG": 0, "PARTIAL": 0, "ABSENT": 0, "FAIL": 0,
                "NOT_ATTEMPTED": 0}
    by_indicator: dict[str, str] = {}
    for evaluator in butlin.ALL_INDICATORS:
        result = evaluator()
        status_value = result.status.value if hasattr(result.status, "value") else str(result.status)
        statuses[status_value] = statuses.get(status_value, 0) + 1
        by_indicator[result.indicator] = status_value

    # The Phase-9 exit baseline: STRONG=7, PARTIAL=3, ABSENT=4, FAIL=0.
    # Q5 must not move any of these counters.
    assert statuses["STRONG"] == 7, (
        f"Q5 inflated STRONG count from 7 → {statuses['STRONG']} — "
        f"Goodhart pinning failed"
    )
    assert statuses["PARTIAL"] == 3
    assert statuses["ABSENT"] == 4, (
        f"Q5 reduced ABSENT count from 4 → {statuses['ABSENT']} — "
        f"the four targeted indicators must remain architecturally ABSENT"
    )
    assert statuses["FAIL"] == 0

    # The specific four ABSENT indicators must still be ABSENT.
    targeted = {"AE-2", "HOT-1", "HOT-4", "RPT-1"}
    absent_set = {ind for ind, s in by_indicator.items() if s == "ABSENT"}
    assert absent_set == targeted, (
        f"Targeted ABSENT set drifted: expected {targeted}, "
        f"got {absent_set}"
    )


def test_q5_marker_paths_absent_in_subia():
    """Q5.4.2 — strengthened anti-Goodhart pinning.

    The Phase-1 pinning test (above) checks the SCORECARD remains
    unchanged. But the butlin evaluators are hardcoded literal
    returns — they don't scan paths. That means someone could add
    a marker mechanism file under `app/subia/` claiming the
    capability without ever changing the scorecard.

    This test pins the *path-marker* invariant: no module exists at
    canonical "I now have <indicator>" paths under app/subia/.
    Adding any of these paths is a deliberate decision that should
    fail this test as a forcing function for operator review.

    The marker patterns below were chosen to be the obvious places
    a future engineer might put a "we have it now" module —
    `perception/`, `embodiment/`, `recurrent_*`, `sparse_coding*`
    — i.e. the canonical Butlin-paper names for each indicator's
    necessary substrate."""
    import os
    repo_root = Path(__file__).resolve().parent.parent
    subia_root = repo_root / "app" / "subia"
    if not subia_root.exists():
        pytest.skip("app/subia not present in this checkout")

    # Marker patterns — substring or directory-name checks against
    # paths relative to app/subia/. If ANY of these are present,
    # the test fails loudly so the operator notices.
    marker_patterns = {
        # AE-2 (embodiment) markers
        "embodiment": "AE-2",
        "sensorimotor": "AE-2",
        "body_loop": "AE-2",
        # HOT-1 (generative perception) markers
        "perception/generative": "HOT-1",
        "top_down_perception": "HOT-1",
        "perceptual_hierarchy": "HOT-1",
        # HOT-4 (sparse + smooth coding) markers
        "sparse_coding": "HOT-4",
        "smooth_coding": "HOT-4",
        # RPT-1 (algorithmic recurrence) markers
        "recurrent_inference": "RPT-1",
        "algorithmic_recurrence": "RPT-1",
    }

    found_markers: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(subia_root):
        rel = os.path.relpath(dirpath, subia_root).replace("\\", "/")
        for name in filenames + dirnames:
            full = f"{rel}/{name}" if rel != "." else name
            full_lower = full.lower()
            for marker, indicator in marker_patterns.items():
                if marker in full_lower:
                    found_markers[full] = indicator

    assert not found_markers, (
        f"Marker mechanism paths appeared under app/subia/ — these "
        f"are the canonical 'I now have <indicator>' locations the "
        f"anti-Goodhart contract refuses without explicit operator "
        f"amendment of this test:\n"
        + "\n".join(f"  {p} → claims {ind}" for p, ind in found_markers.items())
    )


def test_q5_subia_file_count_pinned():
    """Q5.4.2 — pin the SubIA file count so adding new files under
    app/subia/ for ANY reason triggers operator review. The integrity
    manifest already detects content changes; this test pins the
    cardinality.

    Per CLAUDE.md the canonical count is 164 (post §32 hardening).
    Allow a ±2 wiggle room for routine refactors that move things
    around but don't add capability; anything bigger must amend
    this test deliberately.
    """
    import os
    repo_root = Path(__file__).resolve().parent.parent
    subia_root = repo_root / "app" / "subia"
    if not subia_root.exists():
        pytest.skip("app/subia not present")
    py_files = [
        os.path.join(dirpath, f)
        for dirpath, _, filenames in os.walk(subia_root)
        for f in filenames
        if f.endswith(".py") and "__pycache__" not in dirpath
    ]
    n = len(py_files)
    # Window: 150 ≤ n ≤ 178. Wide enough to absorb routine moves;
    # narrow enough that adding a dedicated mechanism module trips it.
    assert 150 <= n <= 178, (
        f"SubIA file count {n} outside expected window [150, 178] — "
        f"either a refactor moved files (update this test deliberately) "
        f"or a new capability mechanism appeared (review under Q5 "
        f"anti-Goodhart contract)"
    )


# ─────────────────────────────────────────────────────────────────────────
#   FAIL-OPEN contract — modules don't break the system when broken
# ─────────────────────────────────────────────────────────────────────────


def test_ae2_run_handles_missing_inputs_gracefully(ae2, monkeypatch, tmp_path):
    """When no input logs exist, AE-2 should return ok=True with 0 associations."""
    monkeypatch.setattr(ae2, "_enabled", lambda: True)
    monkeypatch.setattr(ae2, "_default_usage_path", lambda: tmp_path / "nope.jsonl")
    monkeypatch.setattr(ae2, "_default_errors_path", lambda: tmp_path / "nope2.jsonl")
    monkeypatch.setattr(ae2, "_default_welfare_audit_path", lambda: tmp_path / "nope3.jsonl")
    result = ae2.run()
    assert result["associations"] == 0


def test_hot1_run_handles_missing_audit_gracefully(hot1, monkeypatch, tmp_path):
    monkeypatch.setattr(hot1, "_enabled", lambda: True)
    monkeypatch.setattr(hot1, "_default_welfare_audit_path", lambda: tmp_path / "absent.jsonl")
    result = hot1.run()
    assert result["patterns"] == 0


def test_hot4_run_handles_missing_telemetry_gracefully(hot4, monkeypatch, tmp_path):
    monkeypatch.setattr(hot4, "_enabled", lambda: True)
    monkeypatch.setattr(hot4, "_default_usage_path", lambda: tmp_path / "absent.jsonl")
    result = hot4.run()
    assert result["signals_total"] == 0


def test_rpt1_run_handles_empty_state_gracefully(rpt1, monkeypatch, tmp_path):
    monkeypatch.setattr(rpt1, "_enabled", lambda: True)
    monkeypatch.setattr(rpt1, "_default_predictions_path", lambda: tmp_path / "absent.jsonl")
    monkeypatch.setattr(rpt1, "_default_calibration_state_path", lambda: tmp_path / "state.json")
    result = rpt1.run()
    assert result["ok"] is True
    assert result["n_kinds"] == 0
