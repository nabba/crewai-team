"""PROGRAM §45.1 — Q7.1 architecture-request adoption + rollback tests.

Covers:
  * adoption.compute_score saturating-weighted-sum semantics
  * adoption.measure handles missing request + non-completed + too-young
  * adoption signal probes (imports, idle runs, outputs, interactions)
  * architecture_adoption monitor proposes CR on low score
  * monitor silent when score above threshold
  * monitor dedups per-request via flagged state file
  * Master switch gates lifecycle.create_request (ProtocolDisabled)
  * Master switch gates monitor (skipped path)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
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
#   adoption.compute_score pure-function tests
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def adoption():
    return _load_isolated(
        "adoption_q71", "app/architecture_requests/adoption.py",
    )


def test_score_zero_when_no_signal(adoption):
    assert adoption.compute_score(0, 0, 0, 0) == 0.0


def test_score_one_when_saturated(adoption):
    """All four signals at or above saturation → 1.0."""
    s = adoption.compute_score(
        n_imports=10,                  # > 5 saturation
        n_idle_runs=20,                # > 10 saturation
        n_outputs=10,                  # > 5 saturation
        n_operator_interactions=10,    # > 3 saturation
    )
    assert s == 1.0


def test_score_partial_when_only_imports(adoption):
    """Only imports at saturation → weights["imports"] = 0.40."""
    s = adoption.compute_score(5, 0, 0, 0)
    assert abs(s - 0.40) < 0.001


def test_score_low_adoption_threshold(adoption):
    """Below LOW_ADOPTION_THRESHOLD when imports=1 and nothing else."""
    s = adoption.compute_score(1, 0, 0, 0)
    # 1/5 * 0.40 = 0.08 → below 0.2
    assert s < adoption.LOW_ADOPTION_THRESHOLD


def test_score_saturation_clamp(adoption):
    """Going above saturation doesn't increase score above 1.0."""
    s1 = adoption.compute_score(5, 10, 5, 3)
    s2 = adoption.compute_score(50, 100, 50, 30)
    assert s1 == s2 == 1.0


# ─────────────────────────────────────────────────────────────────────────
#   adoption.measure short-circuit paths
# ─────────────────────────────────────────────────────────────────────────


def test_measure_returns_none_when_request_not_found(adoption, monkeypatch):
    monkeypatch.setattr(
        "app.architecture_requests.store.get", lambda rid: None,
    )
    assert adoption.measure("nonexistent") is None


def test_measure_returns_none_when_not_completed(adoption, monkeypatch):
    class _Status:
        value = "implementing"
    class _Req:
        status = _Status()
        completed_at = None
        package_path = "app/foo/"
        integration_points = []
    monkeypatch.setattr(
        "app.architecture_requests.store.get", lambda rid: _Req(),
    )
    assert adoption.measure("req1") is None


def test_measure_returns_none_when_too_young(adoption, monkeypatch):
    """Request COMPLETED 5d ago — window is 30d default — returns None."""
    five_days_ago = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    class _Status:
        value = "completed"
    class _Req:
        id = "young"
        status = _Status()
        completed_at = five_days_ago
        package_path = "app/foo/"
        integration_points = []
    monkeypatch.setattr(
        "app.architecture_requests.store.get", lambda rid: _Req(),
    )
    assert adoption.measure("young") is None


def test_measure_computes_score_for_old_completed(adoption, monkeypatch, tmp_path):
    """Request COMPLETED 35d ago — measured returns a report."""
    old_ts = (datetime.now(timezone.utc) - timedelta(days=35)).isoformat()
    class _Status:
        value = "completed"
    class _Req:
        id = "old"
        status = _Status()
        completed_at = old_ts
        package_path = "app/example_sub/"
        integration_points = []
    monkeypatch.setattr(
        "app.architecture_requests.store.get", lambda rid: _Req(),
    )
    # All probes return 0 (no real workspace).
    monkeypatch.setattr(adoption, "_count_imports", lambda *a, **kw: 0)
    monkeypatch.setattr(adoption, "_count_idle_runs", lambda *a, **kw: 0)
    monkeypatch.setattr(adoption, "_count_outputs", lambda *a, **kw: 0)
    monkeypatch.setattr(adoption, "_count_operator_interactions", lambda *a, **kw: 0)
    report = adoption.measure("old", repo_root=tmp_path, workspace_root=tmp_path)
    assert report is not None
    assert report.score == 0.0
    assert report.is_low_adoption is True


# ─────────────────────────────────────────────────────────────────────────
#   Signal probes
# ─────────────────────────────────────────────────────────────────────────


def test_count_outputs_counts_files_after_completed(adoption, tmp_path):
    out_dir = tmp_path / "example_sub"
    out_dir.mkdir()
    (out_dir / "newer.json").write_text("{}")
    (out_dir / "older.json").write_text("{}")
    # Backdate older.
    old_ts = time.time() - 60 * 86400
    import os
    os.utime(out_dir / "older.json", (old_ts, old_ts))
    since_iso = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    n = adoption._count_outputs(
        "app/example_sub/", since_iso=since_iso, workspace_root=tmp_path,
    )
    # Only newer counts.
    assert n == 1


def test_count_idle_runs_matches_audit_entries(adoption, tmp_path):
    obs_dir = tmp_path / "observability"
    obs_dir.mkdir()
    log = obs_dir / "idle_jobs.jsonl"
    rows = [
        {"ts": "2026-04-01T10:00:00+00:00", "job_name": "my-job"},     # before window
        {"ts": "2026-05-12T10:00:00+00:00", "job_name": "my-job"},     # in window
        {"ts": "2026-05-13T10:00:00+00:00", "job_name": "other-job"},  # not ours
        {"ts": "2026-05-14T10:00:00+00:00", "job_name": "my-job"},     # in window
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    integration_points = [{"kind": "idle_job", "target": "my-job"}]
    n = adoption._count_idle_runs(
        integration_points,
        since_iso="2026-05-01T00:00:00+00:00",
        workspace_root=tmp_path,
    )
    assert n == 2


# ─────────────────────────────────────────────────────────────────────────
#   architecture_adoption monitor
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def monitor():
    return _load_isolated(
        "arch_adoption_mon", "app/healing/monitors/architecture_adoption.py",
    )


def test_monitor_skipped_when_master_off(monitor, monkeypatch):
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_adoption_monitor_enabled",
            lambda: False,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    result = monitor.run()
    assert result.get("skipped_disabled") is True


def test_monitor_flags_low_adoption(monitor, monkeypatch, tmp_path):
    """Low-adoption report → CR proposed + flagged state updated."""
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_adoption_monitor_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_requests_enabled",
            lambda: True,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    monkeypatch.setattr(
        monitor, "_default_flagged_state_path",
        lambda: tmp_path / "flagged.json",
    )
    monkeypatch.setattr(
        "app.architecture_requests.adoption.list_completed_eligible_for_measurement",
        lambda **kw: ["req_low"],
    )
    # Fake low-adoption report.
    class _Report:
        request_id = "req_low"
        package_path = "app/example/"
        window_days = 30
        completed_at = "2026-04-13T00:00:00+00:00"
        n_imports = 0
        n_idle_runs = 0
        n_outputs = 0
        n_operator_interactions = 0
        score = 0.05
        is_low_adoption = True
    monkeypatch.setattr(
        "app.architecture_requests.adoption.measure",
        lambda rid: _Report(),
    )
    crs_created: list[dict] = []
    fake_cr = MagicMock()
    fake_cr.id = "cr_xyz"
    monkeypatch.setattr(
        "app.change_requests.lifecycle.create_request",
        lambda **kw: (crs_created.append(kw), fake_cr)[1],
    )
    monkeypatch.setattr("app.notify.notify", lambda **kw: None)
    result = monitor.run()
    assert result["flagged"] == 1
    assert len(crs_created) == 1
    assert crs_created[0]["requestor"] == "architecture_adoption_monitor"
    # Dedup state updated.
    flagged = json.loads((tmp_path / "flagged.json").read_text())
    assert "req_low" in flagged


def test_monitor_dedup_via_flagged_state(monitor, monkeypatch, tmp_path):
    """Second pass for the same request_id is skipped (already flagged)."""
    monkeypatch.setattr(
        monitor, "_default_flagged_state_path",
        lambda: tmp_path / "flagged.json",
    )
    # Seed flagged state.
    (tmp_path / "flagged.json").write_text(
        json.dumps({"req_already": "2026-05-01T00:00:00+00:00"})
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_adoption_monitor_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_requests_enabled",
            lambda: True,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    monkeypatch.setattr(
        "app.architecture_requests.adoption.list_completed_eligible_for_measurement",
        lambda **kw: ["req_already"],
    )
    crs_created: list[dict] = []
    monkeypatch.setattr(
        "app.change_requests.lifecycle.create_request",
        lambda **kw: (crs_created.append(kw), MagicMock(id="x"))[1],
    )
    monkeypatch.setattr("app.notify.notify", lambda **kw: None)
    result = monitor.run()
    assert result["skipped_already_flagged"] == 1
    assert crs_created == []


def test_monitor_silent_when_score_above_threshold(monitor, monkeypatch, tmp_path):
    """High-adoption report → no CR, no alert."""
    monkeypatch.setattr(
        monitor, "_default_flagged_state_path",
        lambda: tmp_path / "flagged.json",
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_adoption_monitor_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_requests_enabled",
            lambda: True,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    monkeypatch.setattr(
        "app.architecture_requests.adoption.list_completed_eligible_for_measurement",
        lambda **kw: ["req_healthy"],
    )
    class _Report:
        request_id = "req_healthy"
        package_path = "app/x/"
        score = 0.85
        is_low_adoption = False
        n_imports = 5; n_idle_runs = 10; n_outputs = 5; n_operator_interactions = 3
        window_days = 30
        completed_at = "2026-04-13T00:00:00+00:00"
    monkeypatch.setattr(
        "app.architecture_requests.adoption.measure",
        lambda rid: _Report(),
    )
    crs: list = []
    monkeypatch.setattr(
        "app.change_requests.lifecycle.create_request",
        lambda **kw: crs.append(kw),
    )
    monkeypatch.setattr("app.notify.notify", lambda **kw: None)
    result = monitor.run()
    assert result["flagged"] == 0
    assert crs == []


# ─────────────────────────────────────────────────────────────────────────
#   Master switch gates lifecycle.create_request
# ─────────────────────────────────────────────────────────────────────────


def test_lifecycle_protocol_disabled_when_master_off(monkeypatch):
    """When architecture_requests_enabled = False, create_request raises
    ProtocolDisabled (callers can distinguish from validation failure)."""
    from app.architecture_requests.lifecycle import (
        create_request, ProtocolDisabled,
    )
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_requests_enabled",
            lambda: False,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    with pytest.raises(ProtocolDisabled):
        create_request(
            requestor="test",
            intent="x",
            motivation="y",
            package_path="app/test/",
            file_layout=[],
            integration_points=[],
            env_switches={},
            test_plan="z",
        )


def test_lifecycle_proceeds_when_master_on(monkeypatch, tmp_path):
    """When switch is ON, create_request proceeds (subject to validation)."""
    from app.architecture_requests import store
    from app.architecture_requests.lifecycle import create_request
    try:
        import app.runtime_settings  # noqa: F401
        monkeypatch.setattr(
            "app.runtime_settings.get_architecture_requests_enabled",
            lambda: True,
        )
    except Exception:
        pytest.skip("runtime_settings unavailable")
    monkeypatch.setattr(store, "_DEFAULT_BASE_DIR", tmp_path)
    monkeypatch.setattr(store, "_base_dir", lambda: tmp_path)
    # Real validator runs; we just want to verify protocol isn't
    # short-circuited. Use a minimal valid proposal under a path that
    # passes validation.
    req = create_request(
        requestor="test_agent",
        intent="add a new subsystem for testing",
        motivation="testing — Q7.1 protocol gate",
        package_path="app/q71_test_subsystem/",
        file_layout=[],
        integration_points=[],
        env_switches={},
        test_plan="manual",
    )
    # Either it succeeded (status = PROPOSED) or was refused by
    # validator (TIER_IMMUTABLE_REFUSED) — but NOT ProtocolDisabled.
    assert req is not None


# ─────────────────────────────────────────────────────────────────────────
#   React + config_api wiring
# ─────────────────────────────────────────────────────────────────────────


def test_react_settings_has_arch_requests_fields():
    src = Path("dashboard-react/src/api/queries.ts").read_text()
    assert "architecture_requests_enabled" in src
    assert "architecture_adoption_monitor_enabled" in src


def test_settings_page_mounts_arch_requests_card():
    src = Path("dashboard-react/src/components/SettingsPage.tsx").read_text()
    assert "ArchitectureRequestsCard" in src


def test_arch_requests_card_exists():
    path = Path("dashboard-react/src/components/ArchitectureRequestsCard.tsx")
    assert path.exists()
    body = path.read_text()
    assert "architecture_requests_enabled" in body
    assert "architecture_adoption_monitor_enabled" in body


def test_config_api_handles_master_switch():
    src = Path("app/api/config_api.py").read_text()
    assert '"architecture_requests_enabled" in payload' in src
    assert '"architecture_adoption_monitor_enabled" in payload' in src


def test_arch_adoption_monitor_registered():
    src = Path("app/healing/monitors/__init__.py").read_text()
    assert '"architecture_adoption"' in src
    assert "from app.healing.monitors import architecture_adoption" in src
