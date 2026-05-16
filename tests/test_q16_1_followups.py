"""Q16.1 follow-ups — 11-item audit-driven follow-on commit.

Tests for the new artifacts:
  * Item 2 — HOT-1 outcome reconciler
  * Item 3 — accuracy_log producer wired in person_suggestions
  * Item 4 — corpus version bump (already pinned in themes_6_8 tests)
  * Item 5 — vendor_independence docstring update
  * Item 6 — host_substrate_metrics scripts exist + executable
  * Item 7 — operator_anomaly.last_critical_alert_at() + vacation_mode
    decoupling
  * Item 9 — velocity_digest quarterly composer
  * Item 11 — CONSCIOUSNESS_HOT1_OBSERVATIONS.md amendment
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
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
#   Item 2 — HOT-1 outcome reconciler
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_reconciler(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_hot1_outcome_reconciler_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_reconciler(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/hot1_outcome_reconciler.py",
        "_q16_1_reconciler",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(
        mod, "_cr_audit_path",
        lambda: tmp_path / "change_requests" / "audit.jsonl",
    )
    monkeypatch.setattr(
        mod, "_hot1_observation_path",
        lambda: tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl",
    )
    monkeypatch.setattr(
        mod, "_overlay_path",
        lambda: tmp_path / "healing" / "hot1_outcomes_overlay.json",
    )
    monkeypatch.setattr(
        mod, "_state_path",
        lambda: tmp_path / "healing" / "hot1_outcome_reconciler_state.json",
    )
    return mod


def test_reconciler_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "hot1_outcome_reconciler.py"
    assert p.is_file()
    src = p.read_text()
    assert "def reconcile_once" in src
    assert "def lookup_outcome" in src
    assert "requestor" in src
    assert "error_diagnosis" in src
    # Append-only contract: the reconciler writes to the overlay file,
    # NOT to the source HOT-1 JSONL. Verify by source-grep that the
    # observation path is only READ.
    assert "_hot1_observation_path()" in src
    # Every direct write is to the overlay or state file, not the obs log.
    obs_writes = [
        line for line in src.splitlines()
        if "_hot1_observation_path" in line and "write" in line
    ]
    assert obs_writes == [], (
        f"reconciler writes to observation log — violates append-only: "
        f"{obs_writes}"
    )


def test_reconciler_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch, enabled=False)
    out = mod.reconcile_once(now=time.time())
    assert out.get("skipped") is True


def test_reconciler_no_audit_log_silent(monkeypatch, tmp_path) -> None:
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    out = mod.reconcile_once(now=time.time())
    assert out["ran"] is True
    assert out["n_terminal_events"] == 0
    assert out["n_overlay_entries_after"] == 0


def test_reconciler_matches_cr_to_hot1_by_pattern(monkeypatch, tmp_path) -> None:
    """End-to-end: HOT-1 row + CR audit event with matching
    pattern_signature → overlay entry written."""
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    now = time.time()
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    hot1_ts = datetime.fromtimestamp(now - 100, tz=timezone.utc).isoformat()
    obs_path.write_text(json.dumps({
        "ts": hot1_ts,
        "kind": "metacognitive_repair_proposal",
        "originating_error": {"pattern_signature": "PAT-A"},
        "higher_order_thought": {"declined": False},
        "outcome": None,
    }) + "\n")
    audit_path = tmp_path / "change_requests" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "audit_event": "applied",
        "requestor": "error_diagnosis",
        "origin_pattern_signature": "PAT-A",
        "decided_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "request_id": "cr-1",
    }) + "\n")
    out = mod.reconcile_once(now=now)
    assert out["n_terminal_events"] == 1
    assert out["n_observations"] == 1
    assert out["n_new_outcomes"] == 1
    assert mod.lookup_outcome(pattern_signature="PAT-A", ts_iso=hot1_ts) == "applied"


def test_reconciler_skips_non_error_diagnosis_cr(monkeypatch, tmp_path) -> None:
    """A CR with requestor != error_diagnosis must NOT update HOT-1 outcomes."""
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    obs_path.write_text(json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "originating_error": {"pattern_signature": "PAT-B"},
        "higher_order_thought": {"declined": False},
    }) + "\n")
    audit_path = tmp_path / "change_requests" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "audit_event": "applied",
        "requestor": "wiki_index_reconciler",  # not error_diagnosis
        "origin_pattern_signature": "PAT-B",
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    out = mod.reconcile_once(now=time.time())
    assert out["n_new_outcomes"] == 0


def test_reconciler_idempotent_on_rerun(monkeypatch, tmp_path) -> None:
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    now = time.time()
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    obs_path.write_text(json.dumps({
        "ts": datetime.fromtimestamp(now - 100, tz=timezone.utc).isoformat(),
        "originating_error": {"pattern_signature": "PAT-C"},
        "higher_order_thought": {"declined": False},
    }) + "\n")
    audit_path = tmp_path / "change_requests" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "audit_event": "applied",
        "requestor": "error_diagnosis",
        "origin_pattern_signature": "PAT-C",
        "decided_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
    }) + "\n")
    first = mod.reconcile_once(now=now)
    second = mod.reconcile_once(now=now + 60)
    assert first["n_new_outcomes"] == 1
    # Second run sees same data; overlay unchanged.
    assert second["n_new_outcomes"] == 0


def test_reconciler_rewrites_nothing_to_original_log(monkeypatch, tmp_path) -> None:
    """The append-only contract: the original metacognitive_repair.jsonl
    must be byte-identical after reconciliation."""
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = (json.dumps({
        "ts": datetime.now(timezone.utc).isoformat(),
        "originating_error": {"pattern_signature": "PAT-D"},
        "higher_order_thought": {"declined": False},
    }) + "\n").encode("utf-8")
    obs_path.write_bytes(original_bytes)
    audit_path = tmp_path / "change_requests" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "audit_event": "applied",
        "requestor": "error_diagnosis",
        "origin_pattern_signature": "PAT-D",
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }) + "\n")
    mod.reconcile_once(now=time.time())
    # Original log must be untouched byte-for-byte.
    assert obs_path.read_bytes() == original_bytes


def test_reconciler_run_respects_weekly_cadence(monkeypatch, tmp_path) -> None:
    """run() (the monitor-driver entry) gates on weekly cadence, but
    reconcile_once() does not — the test confirms the right surface
    is gated."""
    mod = _load_reconciler(monkeypatch, tmp_path)
    _stub_rs_reconciler(monkeypatch)
    state_path = tmp_path / "healing" / "hot1_outcome_reconciler_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    # Pretend a recent run.
    state_path.write_text(json.dumps({
        "last_run_at": time.time() - 100,
        "last_cr_audit_ts": 0,
    }))
    out = mod.run(now=time.time())
    assert out.get("ran") is False or out.get("reason") == "cadence"


def test_hot1_consultation_consumes_overlay(monkeypatch, tmp_path) -> None:
    """When the reconciler has written an outcome to the overlay,
    hot1_consultation.consult counts it (n_applied > 0) and
    recommends proceed_normally."""
    # Stub runtime_settings shared between the two modules.
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_hot1_consultation_enabled = lambda: True
    fake_rs.get_hot1_outcome_reconciler_enabled = lambda: True
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    # Load both modules pointing at tmp_path.
    consultation = _isolated_module(
        "app/healing/hot1_consultation.py", "_q16_1_consultation",
    )
    monkeypatch.setattr(consultation, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(
        consultation, "_observation_path",
        lambda: tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl",
    )
    reconciler = _isolated_module(
        "app/healing/hot1_outcome_reconciler.py", "_q16_1_reconciler_v2",
    )
    monkeypatch.setattr(reconciler, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(
        reconciler, "_overlay_path",
        lambda: tmp_path / "healing" / "hot1_outcomes_overlay.json",
    )
    # Patch consultation to find the same reconciler module.
    monkeypatch.setitem(
        sys.modules, "app.healing.hot1_outcome_reconciler", reconciler,
    )

    # Set up: one HOT-1 observation with no outcome in the row, but
    # an overlay entry saying it was applied.
    ts_iso = datetime.now(timezone.utc).isoformat()
    obs_path = tmp_path / "subia" / "observations" / "metacognitive_repair.jsonl"
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    obs_path.write_text(json.dumps({
        "ts": ts_iso,
        "originating_error": {"pattern_signature": "PAT-E"},
        "higher_order_thought": {"declined": False, "self_assessed_confidence": 0.8},
    }) + "\n")
    overlay_path = tmp_path / "healing" / "hot1_outcomes_overlay.json"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(json.dumps({
        f"PAT-E::{ts_iso}": "applied",
    }))
    out = consultation.consult(
        pattern_signature="PAT-E", file_path="app/foo.py",
    )
    assert out["n_applied"] == 1
    assert out["recommendation"] == "proceed_normally"


# ═════════════════════════════════════════════════════════════════════════
#   Item 3 — accuracy_log producer in person_suggestions
# ═════════════════════════════════════════════════════════════════════════


def test_person_suggestions_imports_accuracy_log() -> None:
    p = REPO_ROOT / "app" / "companion" / "person_suggestions.py"
    src = p.read_text()
    assert "accuracy_log" in src
    assert "log_suggestion" in src
    # Privacy boundary: the wired call passes only person_id +
    # category, never names or notes.
    assert "log_suggestion(" in src
    assert '"person_id"' in src


# ═════════════════════════════════════════════════════════════════════════
#   Item 5 — vendor_independence docstring update
# ═════════════════════════════════════════════════════════════════════════


def test_vendor_independence_docstring_marks_extension_point() -> None:
    p = (
        REPO_ROOT / "app" / "resilience_drills" / "drills" / "vendor_independence.py"
    )
    src = p.read_text()
    assert "EXTENSION POINT" in src
    assert "smoke_completion" in src
    assert "Activation requires TWO operator steps" in src


# ═════════════════════════════════════════════════════════════════════════
#   Item 6 — host_substrate_metrics scripts
# ═════════════════════════════════════════════════════════════════════════


def test_host_substrate_metrics_scripts_exist() -> None:
    for rel in (
        "scripts/host_substrate_metrics.sh",
        "scripts/host_substrate_metrics.plist",
        "scripts/install_host_substrate_metrics.sh",
    ):
        p = REPO_ROOT / rel
        assert p.is_file(), f"missing {rel}"


def test_host_substrate_metrics_shell_scripts_executable() -> None:
    for rel in (
        "scripts/host_substrate_metrics.sh",
        "scripts/install_host_substrate_metrics.sh",
    ):
        p = REPO_ROOT / rel
        mode = p.stat().st_mode
        assert mode & stat.S_IXUSR, f"{rel} is not executable"


def test_host_substrate_metrics_writes_to_correct_path() -> None:
    """The collector script writes to workspace/healing/host_metrics.jsonl
    — the path the in-container monitor reads."""
    p = REPO_ROOT / "scripts" / "host_substrate_metrics.sh"
    src = p.read_text()
    assert "host_metrics.jsonl" in src
    assert "healing" in src


# ═════════════════════════════════════════════════════════════════════════
#   Item 7 — operator_anomaly.last_critical_alert_at()
# ═════════════════════════════════════════════════════════════════════════


def test_operator_anomaly_exposes_public_api(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/healing/monitors/operator_anomaly.py",
        "_q16_1_op_anomaly",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    # No state file → returns None.
    assert mod.last_critical_alert_at("new_sender") is None
    # Inject state.
    (tmp_path / "state.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "state.json").write_text(json.dumps({
        "last_alert_at": {"new_sender": 1747497600.0, "cadence_quiet": 0},
    }))
    assert mod.last_critical_alert_at("new_sender") == pytest.approx(1747497600.0)
    # Zero / non-numeric → None.
    assert mod.last_critical_alert_at("cadence_quiet") is None
    assert mod.last_critical_alert_at("nonexistent_kind") is None


def test_vacation_mode_sweep_calls_public_anomaly_api() -> None:
    """Source-grep: vacation_mode.sweep must NOT read operator_anomaly's
    state file directly anymore."""
    p = REPO_ROOT / "app" / "vacation_mode" / "sweep.py"
    src = p.read_text()
    # The function imports the public helper.
    assert "from app.healing.monitors.operator_anomaly import" in src
    assert "last_critical_alert_at" in src
    # And does NOT contain the old direct file read.
    assert "operator_anomaly_state.json" not in src


# ═════════════════════════════════════════════════════════════════════════
#   Item 9 — velocity_digest
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_velocity_digest(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_velocity_digest_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_velocity_digest(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/self_improvement/velocity_digest.py",
        "_q16_1_velocity_digest",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    return mod


def test_velocity_digest_module_exists() -> None:
    p = REPO_ROOT / "app" / "self_improvement" / "velocity_digest.py"
    assert p.is_file()
    src = p.read_text()
    assert "def compose_digest" in src
    assert "def run_once" in src
    assert "_MIN_DAYS_BETWEEN_DIGESTS" in src


def test_velocity_digest_disabled_skips(monkeypatch, tmp_path) -> None:
    mod = _load_velocity_digest(monkeypatch, tmp_path)
    _stub_rs_velocity_digest(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run_once()
    assert out.get("skipped") is True


def test_velocity_digest_composes_and_persists_snapshot(monkeypatch, tmp_path) -> None:
    mod = _load_velocity_digest(monkeypatch, tmp_path)
    _stub_rs_velocity_digest(monkeypatch)
    captured = _stub_notify(monkeypatch)

    # Stub the velocity_summary aggregator.
    fake_velocity = type(sys)("app.self_improvement.velocity")
    fake_velocity.velocity_summary = lambda window_days=90: {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "change_requests": {
            "n_total": 24, "applied_rate_overall": 0.72,
            "by_requestor": {"capability_gap_analyzer": 12, "library_radar": 8},
        },
        "recipes": {"n_active": 15, "global_success_rate": 0.81},
        "architecture_adoption": {"n_measured": 5, "below_rollback_threshold": 1},
        "lessons_learned": {"n_total": 47, "n_added_last_30d": 3},
        "forge_graduations": {"n_last_90d": 0},
    }
    monkeypatch.setitem(
        sys.modules, "app.self_improvement.velocity", fake_velocity,
    )

    out = mod.run_once()
    assert out["ran"] is True
    assert out["wrote"] is not None
    p = Path(out["wrote"])
    assert p.exists()
    body = p.read_text()
    assert "Self-improvement velocity digest" in body
    assert "24" in body  # CR total
    assert "applied-rate" in body.lower()
    # First run: no prior snapshot → no deltas.
    assert "No material quarter-over-quarter deltas" in body or "Quarter-over-quarter deltas" not in body


def test_velocity_digest_surfaces_applied_rate_drop(monkeypatch, tmp_path) -> None:
    """Two consecutive composes where applied-rate drops 20pp → delta
    bullet appears in the second digest."""
    mod = _load_velocity_digest(monkeypatch, tmp_path)
    _stub_rs_velocity_digest(monkeypatch)
    _stub_notify(monkeypatch)

    rates = [0.85, 0.60]  # 25pp drop
    state = {"i": 0}

    fake_velocity = type(sys)("app.self_improvement.velocity")
    def _vs(window_days=90):
        rate = rates[state["i"] % len(rates)]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_days": window_days,
            "change_requests": {"n_total": 30, "applied_rate_overall": rate},
            "recipes": {"n_active": 10, "global_success_rate": 0.7},
            "architecture_adoption": {"n_measured": 0, "below_rollback_threshold": 0},
            "lessons_learned": {"n_total": 50, "n_added_last_30d": 5},
            "forge_graduations": {"n_last_90d": 0},
        }
    fake_velocity.velocity_summary = _vs
    monkeypatch.setitem(
        sys.modules, "app.self_improvement.velocity", fake_velocity,
    )

    # First compose: store snapshot with high applied-rate.
    mod.compose_digest(now=time.time() - 100)
    state["i"] = 1
    # Second compose (after the cadence gate would block; bypass via
    # direct call).
    p = mod.compose_digest(now=time.time())
    assert p is not None
    body = p.read_text()
    assert "applied-rate" in body.lower()
    # Material drop should appear as a delta bullet.
    assert "▼" in body or "shifted" in body


# ═════════════════════════════════════════════════════════════════════════
#   Item 10 — goal_progress simplification
# ═════════════════════════════════════════════════════════════════════════


def test_goal_progress_uses_canonical_kernel_access() -> None:
    p = REPO_ROOT / "app" / "companion" / "goal_progress.py"
    src = p.read_text()
    # Canonical pattern (matches app/identity/long_term_goal_review.py).
    assert "kernel.self_state" in src
    # Old defensive double-fallback should be gone.
    assert "_get_kernel" not in src


# ═════════════════════════════════════════════════════════════════════════
#   Item 11 — CONSCIOUSNESS_HOT1_OBSERVATIONS.md amendment
# ═════════════════════════════════════════════════════════════════════════


def test_hot1_obs_doc_names_new_consumers() -> None:
    p = REPO_ROOT / "docs" / "CONSCIOUSNESS_HOT1_OBSERVATIONS.md"
    assert p.is_file()
    src = p.read_text()
    assert "hot1_consultation" in src
    assert "hot1_outcome_reconciler" in src
    assert "overlay" in src.lower()
    assert "append-only" in src.lower()


# ═════════════════════════════════════════════════════════════════════════
#   Item 1 — PROGRAM.md §51 entry
# ═════════════════════════════════════════════════════════════════════════


def test_program_md_has_section_51() -> None:
    p = REPO_ROOT / "PROGRAM.md"
    assert p.is_file()
    src = p.read_text()
    assert "## 51 2026-05-16 — Q16" in src
    # All 8 themes accounted for.
    for theme in (
        "Theme 1 — substrate longevity",
        "Theme 2 — vendor independence",
        "Theme 3 — operator-unavailable autonomy",
        "Theme 4 — recursive self-improvement boundaries",
        "Theme 5 — knowledge management at decade-scale",
        "Theme 6 — quality of service",
        "Theme 7 — companion depth",
        "Theme 8 — sentience consumption",
    ):
        assert theme in src, f"PROGRAM.md §51 missing: {theme}"
    # Q16.1 follow-up section.
    assert "Q16.1" in src


# ═════════════════════════════════════════════════════════════════════════
#   Wiring tests
# ═════════════════════════════════════════════════════════════════════════


def test_runtime_settings_has_q16_1_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "hot1_outcome_reconciler_enabled",
        "velocity_digest_enabled",
    ):
        assert f'"{key}"' in src
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_monitors_init_registers_q16_1_runners() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    for name in ("hot1_outcome_reconciler", "velocity_digest"):
        assert f'"{name}"' in src, f"missing {name} in cadence map"
