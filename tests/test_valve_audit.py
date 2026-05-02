"""Tests for the reducing-valve audit logger and replay job.

These tests cover:
  - log_rejection() writes JSONL rows with the expected schema
  - kill switch via VALVE_AUDIT_ENABLED=0 silences the logger
  - failures inside log_rejection() never propagate
  - replay loads / samples / produces a daily summary correctly
  - the LLM second-opinion is gated behind VALVE_AUDIT_LLM_REPLAY=1
  - integration: F4 quality-gate rejection actually emits a log row
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.observability import valve_audit, valve_audit_replay


# ── Logger ───────────────────────────────────────────────────────────────────


def _redirect_workspace(tmp_path: Path, monkeypatch) -> Path:
    """Repoint WORKSPACE_ROOT so audit logs land under tmp_path."""
    import app.paths as paths_mod
    monkeypatch.setattr(paths_mod, "WORKSPACE_ROOT", tmp_path)
    return tmp_path


def test_log_rejection_writes_expected_schema(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("VALVE_AUDIT_ENABLED", "1")

    valve_audit.log_rejection(
        filter_id="F4", callsite="test:1",
        input_text="hello world",
        reason="too_short", score=11.0, threshold=20.0,
        extra={"crew_name": "writing"},
    )

    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["filter_id"] == "F4"
    assert row["reason"] == "too_short"
    assert row["score"] == 11.0
    assert row["threshold"] == 20.0
    assert row["input_text"] == "hello world"
    assert len(row["input_hash"]) == 16
    assert row["extra"] == {"crew_name": "writing"}
    assert "ts" in row


def test_log_rejection_kill_switch(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("VALVE_AUDIT_ENABLED", "0")

    valve_audit.log_rejection(
        filter_id="F4", callsite="test:1", reason="too_short",
    )

    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    assert not log_file.exists(), "kill switch should suppress all writes"


def test_log_rejection_truncates_long_text(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("VALVE_AUDIT_ENABLED", "1")

    huge = "x" * 10_000
    valve_audit.log_rejection(
        filter_id="F8", callsite="test:1", input_text=huge,
        reason="below_quality", score=0.4, threshold=0.7,
    )
    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    row = json.loads(log_file.read_text(encoding="utf-8").strip())
    # Truncated to cap, hash is full-text
    assert len(row["input_text"]) <= 4096
    assert len(row["input_hash"]) == 16


def test_log_rejection_swallows_exceptions(monkeypatch):
    """The audit MUST NOT raise into a filter's hot path."""
    monkeypatch.setenv("VALVE_AUDIT_ENABLED", "1")

    # Force the path resolver to raise
    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(valve_audit, "_log_path", boom)

    # Must not raise
    valve_audit.log_rejection(filter_id="F1", callsite="test:1", reason="x")


# ── Replay ───────────────────────────────────────────────────────────────────


def test_replay_loads_only_target_date(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("\n".join([
        json.dumps({"ts": "2026-04-30T12:00:00+00:00", "filter_id": "F4",
                    "reason": "too_short", "score": 5.0, "threshold": 20.0,
                    "input_text": "old", "input_hash": "a" * 16}),
        json.dumps({"ts": "2026-05-01T12:00:00+00:00", "filter_id": "F4",
                    "reason": "too_short", "score": 5.0, "threshold": 20.0,
                    "input_text": "today", "input_hash": "b" * 16}),
    ]) + "\n", encoding="utf-8")

    rows = valve_audit_replay._load_rejections_for_date(log_file, "2026-05-01")
    assert len(rows) == 1
    assert rows[0]["input_text"] == "today"


def test_stratified_sample_caps_per_filter():
    rejections = (
        [{"filter_id": "F4", "i": i} for i in range(80)]
        + [{"filter_id": "F8", "i": i} for i in range(20)]
    )
    sampled = valve_audit_replay._stratified_sample(rejections, per_filter_cap=50)
    by_id = {}
    for r in sampled:
        by_id.setdefault(r["filter_id"], []).append(r)
    assert len(by_id["F4"]) == 50
    assert len(by_id["F8"]) == 20


def test_stratified_sample_is_deterministic():
    """Re-running the sampler on the same input must produce the same sample."""
    rejections = [{"filter_id": "F4", "i": i} for i in range(100)]
    a = valve_audit_replay._stratified_sample(rejections, 50)
    b = valve_audit_replay._stratified_sample(rejections, 50)
    assert [r["i"] for r in a] == [r["i"] for r in b]


def test_loose_replay_too_short_below_relaxed():
    rejection = {
        "filter_id": "F4", "reason": "too_short",
        "score": 5.0, "threshold": 20.0,
    }
    # Original threshold 20; relaxed 10. Score 5 < 10 → would NOT pass.
    assert valve_audit_replay._loose_replay("F4", "too_short", rejection) is False


def test_loose_replay_too_short_above_relaxed():
    rejection = {
        "filter_id": "F4", "reason": "too_short",
        "score": 15.0, "threshold": 20.0,
    }
    # Score 15 >= relaxed 10 → would pass.
    assert valve_audit_replay._loose_replay("F4", "too_short", rejection) is True


def test_loose_replay_skips_categorical_reasons():
    """no_text / cooldown / pattern-match rejections have no relaxed equivalent."""
    for filter_id, reason in [
        ("F8", "no_text"),
        ("F8", "cooldown"),
        ("F4", "quality_failure_pattern"),
    ]:
        rej = {"filter_id": filter_id, "reason": reason}
        assert valve_audit_replay._loose_replay(filter_id, reason, rej) is None


def test_replay_persists_daily_summary(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("VALVE_AUDIT_LLM_REPLAY", "0")  # loose replay only

    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(60):
        rows.append(json.dumps({
            "ts": "2026-05-01T12:00:00+00:00",
            "filter_id": "F4", "reason": "too_short",
            "score": float(i % 20), "threshold": 20.0,
            "input_text": f"x{i}", "input_hash": f"{i:016x}",
        }))
    log_file.write_text("\n".join(rows) + "\n", encoding="utf-8")

    summary = valve_audit_replay.run_daily_replay(target_date="2026-05-01")
    assert summary["rejections_total"] == 60
    assert summary["sampled_total"] == 50  # capped per filter
    assert len(summary["filters"]) == 1
    f4 = summary["filters"][0]
    assert f4["filter_id"] == "F4"
    assert f4["rejections_total"] == 60
    # FRR is None when LLM replay is off
    assert f4["false_rejection_rate"] is None
    assert f4["disagreement_rate"] is not None

    # Verdicts and summary written
    assert (tmp_path / "logs" / "valve_audit_verdicts.jsonl").exists()
    assert (tmp_path / "logs" / "valve_audit_summary.jsonl").exists()


def test_replay_empty_day(tmp_path, monkeypatch):
    _redirect_workspace(tmp_path, monkeypatch)
    summary = valve_audit_replay.run_daily_replay(target_date="2026-05-01")
    assert summary["rejections_total"] == 0
    assert summary["filters"] == []


def test_replay_llm_gated_off_by_default(tmp_path, monkeypatch):
    """When VALVE_AUDIT_LLM_REPLAY is unset, _llm_second_opinion must not fire."""
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.delenv("VALVE_AUDIT_LLM_REPLAY", raising=False)
    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps({
        "ts": "2026-05-01T12:00:00+00:00",
        "filter_id": "F4", "reason": "too_short",
        "score": 5.0, "threshold": 20.0,
        "input_text": "x", "input_hash": "a" * 16,
    }) + "\n", encoding="utf-8")

    # Patch the second-opinion path to assert it isn't reached.
    def explode(rejection):
        raise AssertionError("LLM was called when it should have been gated off")

    monkeypatch.setattr(valve_audit_replay, "_llm_second_opinion", explode)

    summary = valve_audit_replay.run_daily_replay(target_date="2026-05-01")
    assert summary["sampled_total"] == 1
    f4 = summary["filters"][0]
    assert f4["false_rejection_rate"] is None


# ── Integration smoke test ───────────────────────────────────────────────────


def test_quality_gate_rejection_emits_audit_row(tmp_path, monkeypatch):
    """When _passes_quality_gate rejects, a row appears in valve_audit.jsonl.

    Loads execution.py directly via importlib so we don't pull in the full
    crewai dependency chain via the package __init__ — the function is a
    pure Python gate that doesn't need the rest of the agent stack.
    """
    import importlib.util
    _redirect_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("VALVE_AUDIT_ENABLED", "1")

    src_path = (
        Path(__file__).parent.parent
        / "app" / "agents" / "commander" / "execution.py"
    )
    spec = importlib.util.spec_from_file_location("commander_execution", src_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _passes_quality_gate = module._passes_quality_gate

    assert _passes_quality_gate("hi", "writing") is False  # too_short

    log_file = tmp_path / "logs" / "valve_audit.jsonl"
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    row = json.loads(lines[0])
    assert row["filter_id"] == "F4"
    assert row["reason"] == "too_short"
