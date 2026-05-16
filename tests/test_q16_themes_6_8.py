"""Q16 Themes 6-8 — QoS + companion depth + sentience consumption.

PROGRAM §51 Q16 third batch.

Theme 6 — quality of service:
  * 39th monitor app/healing/monitors/latency_slo.py
  * Answer regression suite app/qos/answer_regression.py
  * RPT-1 advisory surface (/api/cp/quality/rpt1-calibration)

Theme 7 — companion depth:
  * app/companion/accuracy_log.py
  * app/companion/goal_progress.py
  * app/privacy/annual_review.py

Theme 8 — sentience consumption:
  * app/healing/hot1_consultation.py
  * app/philosophy/panel_digest.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, REPO_ROOT / rel_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_notify(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: captured.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)
    return captured


# ═════════════════════════════════════════════════════════════════════════
#   Theme 6.1 — latency_slo
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_latency(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_latency_slo_monitor_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_latency_monitor(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/monitors/latency_slo.py", "_q16t6_latency_slo",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(
        mod, "_audit_log_path", lambda: tmp_path / "audit.log",
    )
    return mod


def test_latency_slo_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "latency_slo.py"
    assert p.is_file()
    src = p.read_text()
    assert "def run(" in src
    assert "_REGRESSION_RATIO_WARN" in src
    assert "history_snapshot" in src
    # No mutating any external state besides the state file.
    assert "create_request" not in src
    assert "approve(" not in src


def test_latency_slo_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_latency_monitor(monkeypatch, tmp_path)
    _stub_rs_latency(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out.get("skipped") is True


def test_latency_slo_no_audit_log_silent(monkeypatch, tmp_path) -> None:
    mod = _load_latency_monitor(monkeypatch, tmp_path)
    _stub_rs_latency(monkeypatch)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out["ran"] is True
    assert out["rollup"]["n"] == 0
    assert out["alerts"] == []


def test_latency_slo_computes_percentiles(monkeypatch, tmp_path) -> None:
    mod = _load_latency_monitor(monkeypatch, tmp_path)
    _stub_rs_latency(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    # Write 50 paired request_received / response_sent rows with
    # varying latencies.
    rows = []
    for i in range(50):
        trace_id = f"t-{i}"
        ts0 = now - 86400 + i * 10
        ts1 = ts0 + 0.1 + (i % 10) * 0.05  # 100-550ms
        rows.append({
            "ts": datetime.fromtimestamp(ts0, tz=timezone.utc).isoformat(),
            "event": "request_received",
            "trace_id": trace_id,
        })
        rows.append({
            "ts": datetime.fromtimestamp(ts1, tz=timezone.utc).isoformat(),
            "event": "response_sent",
            "trace_id": trace_id,
        })
    (tmp_path / "audit.log").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
    )
    out = mod.run(now=now)
    assert out["ran"] is True
    rollup = out["rollup"]
    assert rollup["n"] == 50
    assert 0 < rollup["p50_s"] <= 1.0
    assert rollup["p99_s"] >= rollup["p50_s"]


def test_latency_slo_regression_alert(monkeypatch, tmp_path) -> None:
    """When current p95 ≥ 2× baseline median, an alert fires."""
    mod = _load_latency_monitor(monkeypatch, tmp_path)
    _stub_rs_latency(monkeypatch)
    captured = _stub_notify(monkeypatch)
    # Pre-seed history with 4 weeks of slow rollups (p95 ~ 0.5s).
    history = [
        {
            "ts": time.time() - (5 - i) * 7 * 86400,
            "n": 100,
            "p50_s": 0.2,
            "p95_s": 0.5,
            "p99_s": 0.7,
        } for i in range(4)
    ]
    state = {
        "last_run_at": time.time() - 8 * 86400,
        "history": history,
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    # Write 50 SLOW rows so the current p95 is ~1.5s (>2× the 0.5s baseline).
    now = time.time()
    rows = []
    for i in range(50):
        trace_id = f"t-{i}"
        ts0 = now - 86400 + i * 10
        ts1 = ts0 + 0.5 + (i % 5) * 0.3  # 500ms-1.7s
        rows.append({
            "ts": datetime.fromtimestamp(ts0, tz=timezone.utc).isoformat(),
            "event": "request_received",
            "trace_id": trace_id,
        })
        rows.append({
            "ts": datetime.fromtimestamp(ts1, tz=timezone.utc).isoformat(),
            "event": "response_sent",
            "trace_id": trace_id,
        })
    (tmp_path / "audit.log").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
    )
    out = mod.run(now=now)
    assert any(a["kind"].startswith("regression_") for a in out["alerts"])


def test_latency_slo_history_snapshot(monkeypatch, tmp_path) -> None:
    mod = _load_latency_monitor(monkeypatch, tmp_path)
    _stub_rs_latency(monkeypatch)
    _stub_notify(monkeypatch)
    (tmp_path / "state.json").write_text(json.dumps({
        "last_run_at": time.time(),
        "history": [{"n": 10, "p50_s": 0.1, "p95_s": 0.2, "p99_s": 0.3}],
        "last_alert_at": {},
    }))
    snap = mod.history_snapshot()
    assert "history" in snap and "baselines" in snap


# ═════════════════════════════════════════════════════════════════════════
#   Theme 6.2 — answer regression
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_answer(monkeypatch, *, enabled: bool = True, llm: bool = False) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_answer_regression_enabled = lambda: enabled
    fake_rs.get_answer_regression_llm_enabled = lambda: llm
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_answer_module(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/qos/answer_regression.py", "_q16t6_answer_regression",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    return mod


def test_answer_regression_module_exists() -> None:
    p = REPO_ROOT / "app" / "qos" / "answer_regression.py"
    assert p.is_file()
    src = p.read_text()
    assert "FROZEN_QA_PAIRS" in src
    assert "def run_regression" in src
    assert "Verdict" in src
    # Frozen corpus contains at least 5 pairs.
    assert src.count("QAPair(") >= 5


def test_answer_regression_disabled_returns_none(monkeypatch, tmp_path) -> None:
    mod = _load_answer_module(monkeypatch, tmp_path)
    _stub_rs_answer(monkeypatch, enabled=False)
    out = mod.run_regression(
        answer_fn=lambda qa: "stub",
        judge_fn=lambda qa, ans: (10, "pass", "stub"),
        force=True,
    )
    assert out is None


def test_answer_regression_cadence_gate(monkeypatch, tmp_path) -> None:
    """Two runs back-to-back: second is skipped by the 90-day gate
    unless force=True."""
    mod = _load_answer_module(monkeypatch, tmp_path)
    _stub_rs_answer(monkeypatch, enabled=True)
    first = mod.run_regression(
        answer_fn=lambda qa: qa.reference_answer,
        judge_fn=lambda qa, ans: (8, "pass", "stub"),
        force=True,
    )
    assert first is not None
    second = mod.run_regression(
        answer_fn=lambda qa: qa.reference_answer,
        judge_fn=lambda qa, ans: (8, "pass", "stub"),
        force=False,
    )
    assert second is None


def test_answer_regression_aggregates_verdicts(monkeypatch, tmp_path) -> None:
    mod = _load_answer_module(monkeypatch, tmp_path)
    _stub_rs_answer(monkeypatch)
    # Pass on every odd id, fail on every even.
    call_idx = {"i": 0}

    def fake_answer(qa):
        return qa.reference_answer if "even" not in qa.id else "wrong"

    def fake_judge(qa, ans):
        call_idx["i"] += 1
        if call_idx["i"] % 2 == 0:
            return 2, "fail", "stub"
        return 9, "pass", "stub"

    run = mod.run_regression(
        answer_fn=fake_answer,
        judge_fn=fake_judge,
        force=True,
    )
    assert run is not None
    assert run.n_questions == len(mod.FROZEN_QA_PAIRS)
    assert run.n_pass + run.n_fail + run.n_partial + run.n_error == run.n_questions
    # Persisted snapshot exists.
    assert mod.latest_run() is not None


def test_answer_regression_handles_broken_answer_fn(monkeypatch, tmp_path) -> None:
    mod = _load_answer_module(monkeypatch, tmp_path)
    _stub_rs_answer(monkeypatch)

    def broken(qa):
        raise RuntimeError("nope")

    run = mod.run_regression(
        answer_fn=broken,
        judge_fn=lambda qa, ans: (5, "partial", "stub"),
        force=True,
    )
    assert run is not None
    assert run.n_error == run.n_questions


# ═════════════════════════════════════════════════════════════════════════
#   Theme 6.3 — RPT-1 advisory REST
# ═════════════════════════════════════════════════════════════════════════


def test_qos_api_module_exists() -> None:
    p = REPO_ROOT / "app" / "api" / "qos_api.py"
    assert p.is_file()
    src = p.read_text()
    assert '@router.get("/latency")' in src
    assert '@router.get("/answer-regression")' in src
    assert '@router.get("/rpt1-calibration")' in src
    assert "advisory_only" in src


def test_qos_api_router_mounted_in_main() -> None:
    p = REPO_ROOT / "app" / "main.py"
    src = p.read_text()
    assert "qos_api" in src
    assert "qos_router" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 7.1 — companion accuracy log
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_accuracy(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_companion_accuracy_log_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_accuracy_log(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/companion/accuracy_log.py", "_q16t7_accuracy_log",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    return mod


def test_accuracy_log_module_exists() -> None:
    p = REPO_ROOT / "app" / "companion" / "accuracy_log.py"
    assert p.is_file()
    src = p.read_text()
    assert "def log_suggestion" in src
    assert "def log_action" in src
    assert "def accuracy_summary" in src


def test_accuracy_log_records_and_summarises(monkeypatch, tmp_path) -> None:
    mod = _load_accuracy_log(monkeypatch, tmp_path)
    _stub_rs_accuracy(monkeypatch)
    sid1 = mod.log_suggestion(kind="person_dormancy", payload={"id": 1})
    sid2 = mod.log_suggestion(kind="person_dormancy", payload={"id": 2})
    sid3 = mod.log_suggestion(kind="cross_modal_pattern", payload={"id": 3})
    mod.log_action(suggestion_id=sid1, action="clicked")
    mod.log_action(suggestion_id=sid2, action="ignored")
    mod.log_action(suggestion_id=sid3, action="replied")
    summary = mod.accuracy_summary(window_days=30)
    assert summary["available"] is True
    assert summary["n_suggestions"] == 3
    assert summary["n_with_action"] == 3
    by_kind = summary["by_kind"]
    assert by_kind["person_dormancy"]["total"] == 2
    assert by_kind["person_dormancy"]["accepted"] == 1
    assert by_kind["cross_modal_pattern"]["accepted"] == 1


def test_accuracy_log_disabled_skips_recording(monkeypatch, tmp_path) -> None:
    mod = _load_accuracy_log(monkeypatch, tmp_path)
    _stub_rs_accuracy(monkeypatch, enabled=False)
    sid = mod.log_suggestion(kind="x", payload={"y": 1})
    mod.log_action(suggestion_id=sid, action="clicked")
    summary = mod.accuracy_summary()
    # No file → available=False
    assert summary["available"] is False


def test_accuracy_log_never_stores_payload(monkeypatch, tmp_path) -> None:
    """The payload text is hashed; verify the hash but not the body
    appears on disk."""
    mod = _load_accuracy_log(monkeypatch, tmp_path)
    _stub_rs_accuracy(monkeypatch)
    secret = "VERY SPECIFIC SECRET TEXT"
    mod.log_suggestion(kind="x", payload={"text": secret})
    log_path = tmp_path / "companion" / "accuracy_log.jsonl"
    assert log_path.exists()
    body = log_path.read_text()
    assert secret not in body
    assert "payload_hash" in body


# ═════════════════════════════════════════════════════════════════════════
#   Theme 7.2 — goal_progress probe
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_goals(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_goal_progress_probe_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_goal_module(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/companion/goal_progress.py", "_q16t7_goal_progress",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    return mod


def test_goal_progress_module_exists() -> None:
    p = REPO_ROOT / "app" / "companion" / "goal_progress.py"
    assert p.is_file()
    src = p.read_text()
    assert "def evaluate" in src
    assert "current_goals" in src
    assert "_STALLED_DAYS_THRESHOLD" in src


def test_goal_progress_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_goal_module(monkeypatch, tmp_path)
    _stub_rs_goals(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.evaluate(now=time.time())
    assert out.get("skipped") is True


def test_goal_progress_no_goals_silent(monkeypatch, tmp_path) -> None:
    mod = _load_goal_module(monkeypatch, tmp_path)
    _stub_rs_goals(monkeypatch)
    _stub_notify(monkeypatch)
    monkeypatch.setattr(mod, "_load_current_goals", lambda: [])
    out = mod.evaluate(now=time.time())
    assert out["ran"] is True
    assert out["n_goals"] == 0
    assert out["goals"] == []


def test_goal_progress_advancing_when_evidence_present(
    monkeypatch, tmp_path,
) -> None:
    mod = _load_goal_module(monkeypatch, tmp_path)
    _stub_rs_goals(monkeypatch)
    _stub_notify(monkeypatch)
    monkeypatch.setattr(
        mod, "_load_current_goals",
        lambda: ["learn woodworking dovetail joinery techniques"],
    )
    # No crew_tasks, no companion ideas, but inject continuity ledger
    # rows matching "woodworking" + "joinery".
    ledger_path = tmp_path / "identity" / "continuity_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    ledger_path.write_text(json.dumps({
        "ts": ts,
        "kind": "tier3_amendment",
        "summary": "studied woodworking joinery techniques",
    }) + "\n")
    out = mod.evaluate(now=time.time())
    assert out["n_goals"] == 1
    goal_row = out["goals"][0]
    assert goal_row["status"] == "advancing"
    assert goal_row["evidence_count"] >= 1


def test_goal_progress_stalled_after_threshold(monkeypatch, tmp_path) -> None:
    """A goal with NO evidence + stall_since past threshold becomes
    stalled and emits an alert."""
    mod = _load_goal_module(monkeypatch, tmp_path)
    _stub_rs_goals(monkeypatch)
    captured = _stub_notify(monkeypatch)
    monkeypatch.setattr(
        mod, "_load_current_goals",
        lambda: ["learn quantum chromodynamics deeply"],
    )
    now = time.time()
    # Pre-seed state with a stall_since 30 days ago.
    state_path = tmp_path / "companion" / "goal_progress_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "last_run_at": now - 100,
        "per_goal_stall_since": {
            "learn quantum chromodynamics deeply": now - 30 * 86400,
        },
        "last_alert_at": {},
    }))
    out = mod.evaluate(now=now)
    stalled = out["stalled"]
    assert len(stalled) == 1
    assert any("🎯" in c.get("title", "") for c in captured)


# ═════════════════════════════════════════════════════════════════════════
#   Theme 7.3 — annual privacy review
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_privacy(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_annual_privacy_review_enabled = lambda: enabled
    # Stub for source-state lookups inside the composer.
    fake_rs._ensure_initialized = lambda: {}
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_privacy_module(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/privacy/annual_review.py", "_q16t7_privacy_review",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    return mod


def test_privacy_module_exists() -> None:
    p = REPO_ROOT / "app" / "privacy" / "annual_review.py"
    assert p.is_file()
    src = p.read_text()
    assert "_DATA_SOURCES" in src
    assert "def compose_review" in src
    # Catalogue contains the major surfaces.
    for name in (
        "Signal messages", "Person correlation", "Browser-history",
        "Apple Health", "Google Workspace",
    ):
        assert name in src


def test_privacy_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_privacy_module(monkeypatch, tmp_path)
    _stub_rs_privacy(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run_once()
    assert out.get("skipped") is True


def test_privacy_composes_audit_file(monkeypatch, tmp_path) -> None:
    mod = _load_privacy_module(monkeypatch, tmp_path)
    _stub_rs_privacy(monkeypatch)
    _stub_notify(monkeypatch)
    out = mod.run_once()
    assert out["ran"] is True
    written = out["wrote"]
    assert written is not None
    p = Path(written)
    assert p.exists()
    body = p.read_text()
    assert "# Annual privacy audit" in body
    assert "Signal messages" in body
    assert "Apple Health" in body


# ═════════════════════════════════════════════════════════════════════════
#   Theme 8.1 — HOT-1 consultation hook
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_hot1(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_hot1_consultation_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_hot1_module(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/hot1_consultation.py", "_q16t8_hot1_consultation",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(
        mod, "_observation_path",
        lambda: tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl",
    )
    return mod


def test_hot1_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "hot1_consultation.py"
    assert p.is_file()
    src = p.read_text()
    assert "def consult" in src
    assert "recommendation" in src
    assert "hint_for_prompt" in src


def test_hot1_disabled_returns_empty(monkeypatch, tmp_path) -> None:
    mod = _load_hot1_module(monkeypatch, tmp_path)
    _stub_rs_hot1(monkeypatch, enabled=False)
    out = mod.consult(pattern_signature="x", file_path="app/foo.py")
    assert out["available"] is False


def test_hot1_no_observations_returns_empty(monkeypatch, tmp_path) -> None:
    mod = _load_hot1_module(monkeypatch, tmp_path)
    _stub_rs_hot1(monkeypatch)
    out = mod.consult(pattern_signature="x", file_path="app/foo.py")
    assert out["available"] is False


def test_hot1_chronic_failure_recommends_skip(monkeypatch, tmp_path) -> None:
    mod = _load_hot1_module(monkeypatch, tmp_path)
    _stub_rs_hot1(monkeypatch)
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = []
    # 4 declined attempts, no successes.
    for i in range(4):
        rows.append(json.dumps({
            "ts": now_iso,
            "kind": "metacognitive_repair_proposal",
            "originating_error": {"pattern_signature": "PAT-1"},
            "higher_order_thought": {"declined": True, "self_assessed_confidence": 0.2},
            "outcome": None,
        }))
    obs_path.write_text("\n".join(rows) + "\n")
    out = mod.consult(pattern_signature="PAT-1", file_path="app/foo.py")
    assert out["available"] is True
    assert out["n_declined"] == 4
    assert out["recommendation"] == "skip"
    assert out["hint_for_prompt"] is not None


def test_hot1_prior_success_recommends_normal(monkeypatch, tmp_path) -> None:
    mod = _load_hot1_module(monkeypatch, tmp_path)
    _stub_rs_hot1(monkeypatch)
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        json.dumps({
            "ts": now_iso,
            "originating_error": {"pattern_signature": "PAT-2"},
            "higher_order_thought": {"declined": False, "self_assessed_confidence": 0.8},
            "outcome": "applied",
        })
    ]
    obs_path.write_text("\n".join(rows) + "\n")
    out = mod.consult(pattern_signature="PAT-2", file_path="app/foo.py")
    assert out["recommendation"] == "proceed_normally"
    assert out["n_applied"] == 1


def test_hot1_rollback_recommends_caveat(monkeypatch, tmp_path) -> None:
    mod = _load_hot1_module(monkeypatch, tmp_path)
    _stub_rs_hot1(monkeypatch)
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    rows = [
        json.dumps({
            "ts": now_iso,
            "originating_error": {"pattern_signature": "PAT-3"},
            "higher_order_thought": {"declined": False, "self_assessed_confidence": 0.6},
            "outcome": "rolled_back",
        })
    ]
    obs_path.write_text("\n".join(rows) + "\n")
    out = mod.consult(pattern_signature="PAT-3", file_path="app/foo.py")
    assert out["recommendation"] == "proceed_with_caveat"
    assert "rolled back" in (out["hint_for_prompt"] or "")


def test_structured_diagnosis_imports_hot1_consultation() -> None:
    """Wiring check: structured_diagnosis must call hot1_consult."""
    p = REPO_ROOT / "app" / "healing" / "structured_diagnosis.py"
    src = p.read_text()
    assert "hot1_consultation" in src
    assert "consult as _hot1_consult" in src or "from app.healing.hot1_consultation import consult" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 8.2 — philosophy panel quarterly digest
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_philosophy(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_philosophy_digest_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_philosophy_module(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/philosophy/panel_digest.py", "_q16t8_panel_digest",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    return mod


def test_philosophy_digest_module_exists() -> None:
    p = REPO_ROOT / "app" / "philosophy" / "panel_digest.py"
    assert p.is_file()
    src = p.read_text()
    assert "def compose_digest" in src
    assert "def run_once" in src
    assert "_MIN_DAYS_BETWEEN_DIGESTS" in src


def test_philosophy_digest_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_philosophy_module(monkeypatch, tmp_path)
    _stub_rs_philosophy(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run_once()
    assert out.get("skipped") is True


def test_philosophy_digest_no_consultations_silent(monkeypatch, tmp_path) -> None:
    mod = _load_philosophy_module(monkeypatch, tmp_path)
    _stub_rs_philosophy(monkeypatch)
    _stub_notify(monkeypatch)
    out = mod.run_once()
    assert out["ran"] is True
    assert out["wrote"] is None


def test_philosophy_digest_composes_when_consultations_exist(monkeypatch, tmp_path) -> None:
    mod = _load_philosophy_module(monkeypatch, tmp_path)
    _stub_rs_philosophy(monkeypatch)
    _stub_notify(monkeypatch)
    cache = tmp_path / "philosophy" / "panel_cache.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    # 3 consultations in the current quarter.
    rows = []
    for i in range(3):
        rows.append(json.dumps({
            "key": f"k{i}",
            "consulted_at": now.isoformat(),
            "result": {
                "question": "is recipe X aligned with the operator's values?",
                "perspectives": [],
                "unresolved_tensions": [
                    "stoic vs utilitarian on aesthetic trade-off",
                ],
                "coverage": 0.7,
                "consulted_at": now.isoformat(),
            },
        }))
    cache.write_text("\n".join(rows) + "\n")
    out = mod.run_once()
    assert out["ran"] is True
    assert out["wrote"] is not None
    digest_path = Path(out["wrote"])
    body = digest_path.read_text()
    assert "Philosophy panel digest" in body
    assert "Unresolved tensions" in body
    assert "stoic vs utilitarian" in body


# ═════════════════════════════════════════════════════════════════════════
#   Wiring tests
# ═════════════════════════════════════════════════════════════════════════


def test_runtime_settings_has_themes_6_8_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "latency_slo_monitor_enabled",
        "answer_regression_enabled",
        "answer_regression_llm_enabled",
        "companion_accuracy_log_enabled",
        "goal_progress_probe_enabled",
        "annual_privacy_review_enabled",
        "hot1_consultation_enabled",
        "philosophy_digest_enabled",
    ):
        assert f'"{key}"' in src
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_monitors_init_registers_themes_6_8_monitors() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    for name in (
        "latency_slo", "answer_regression", "goal_progress",
        "annual_privacy_review", "philosophy_digest",
    ):
        assert f'"{name}"' in src, f"monitor {name} not in cadence map"
