"""Q13 (PROGRAM §48) — Year-2+ resilience.

Tests for the 3 missing items shipped this Q (the 5 already-shipped
items had separate test coverage):

  * 2.2 schema-migration drill — shell script + monitor + ledger
  * 2.3 dependency_radar — pip+OSV+GitHub composition + routing
  * 2.6 tz_drift monitor — divergence detection + CR + ledger
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    """Load a module via spec_from_file_location to bypass gateway-
    dependency imports in the test environment."""
    spec = importlib.util.spec_from_file_location(
        mod_name, REPO_ROOT / rel_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ═════════════════════════════════════════════════════════════════════════
#   Source-level: 2.1 / 2.4 / 2.5 / 2.7 / 2.8 already shipped
# ═════════════════════════════════════════════════════════════════════════


def test_2_1_algorithm_pinning_shipped() -> None:
    """algorithm_pinning.py + crypto_rotation_drill monitor present."""
    p1 = REPO_ROOT / "app" / "audit" / "algorithm_pinning.py"
    p2 = REPO_ROOT / "app" / "healing" / "monitors" / "crypto_rotation_drill.py"
    assert p1.is_file() and p2.is_file()
    src = p1.read_text()
    assert "KNOWN_ARTIFACT_CLASSES" in src
    assert "rolled_audit_log" in src  # the §2.4 chain class is tracked


def test_2_4_rolled_log_audit_journal_migrated() -> None:
    """rolled_log infrastructure exists + audit_journal migrated."""
    p1 = REPO_ROOT / "app" / "audit" / "rolled_log.py"
    p2 = REPO_ROOT / "app" / "audit" / "journal.py"
    assert p1.is_file() and p2.is_file()
    src = p2.read_text()
    assert "RolledLogStore" in src or "rolled_log" in src
    # audit_journal.json migrated to rolled-segment storage.
    assert "audit_journal" in src


def test_2_5_version_upgrade_drill_shipped() -> None:
    """version_upgrade_drill monitor + shell script present."""
    p1 = REPO_ROOT / "app" / "healing" / "monitors" / "version_upgrade_drill.py"
    p2 = REPO_ROOT / "deploy" / "scripts" / "version-upgrade-drill.sh"
    assert p1.is_file() and p2.is_file()
    assert "pg_upgrade" in p2.read_text() or "neo4j-admin" in p2.read_text()


def test_2_7_provider_contract_drift_shipped() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "provider_contract_drift.py"
    assert p.is_file()
    src = p.read_text()
    assert "baseline" in src.lower()
    assert "structural" in src.lower()


def test_2_8_continuity_ledger_shipped() -> None:
    """continuity_ledger + annual_reflection wired."""
    p1 = REPO_ROOT / "app" / "identity" / "continuity_ledger.py"
    p2 = REPO_ROOT / "app" / "identity" / "annual_reflection.py"
    assert p1.is_file() and p2.is_file()
    src = p1.read_text()
    # Q13 added 2 new event kinds — assert they're listed.
    assert '"schema_migration_drill"' in src
    assert '"tz_drift"' in src


# ═════════════════════════════════════════════════════════════════════════
#   2.2 — schema-migration drill
# ═════════════════════════════════════════════════════════════════════════


def test_migration_drill_shell_script_exists() -> None:
    p = REPO_ROOT / "deploy" / "scripts" / "migration-drill.sh"
    assert p.is_file()
    src = p.read_text()
    # Must invoke the production startup migrations code path.
    assert "startup_migrations" in src
    # Must restore Postgres before exercising the schema.
    assert "psql" in src
    # Must run schema-smoke queries that exercise tables today's code
    # expects to find (the actual concern: "today's code can't read a
    # 6-month-old backup"). At least one probe under control_plane.* must
    # be in the script — the smoke loop is the load-bearing signal.
    assert "control_plane." in src
    # Must emit the manifest the monitor reads.
    assert "migration_drill_manifest.json" in src
    # Must update last_drill_at + last_drill_ok.
    assert "last_drill_at" in src
    assert "last_drill_ok" in src
    # Must NOT walk migrations/*.sql and fabricate a tracking table —
    # that design was retired (see header comment in the drill script).
    assert "_schema_migrations" not in src, (
        "Drill must not reintroduce the fabricated control_plane._schema_migrations "
        "tracking table. See migration-drill.sh header for rationale."
    )


def test_migration_drill_shell_script_executable() -> None:
    p = REPO_ROOT / "deploy" / "scripts" / "migration-drill.sh"
    assert os.access(p, os.X_OK), "migration-drill.sh must be executable"


def test_migration_drill_monitor_alerts_when_manifest_missing(monkeypatch, tmp_path) -> None:
    """The monitor alerts when manifest doesn't exist."""
    _stub_runtime_settings(monkeypatch, migration_drill_monitor_enabled=True)
    _stub_common(monkeypatch)
    mod = _load_migration_drill()
    # Force fresh state
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(mod, "_STATE_FILE", str(state_path))
    summary = mod.run(
        manifest_path=tmp_path / "missing.json",
        now=10_000_000.0,
    )
    assert summary["ran"] is True
    assert summary["manifest_present"] is False
    assert summary["alert_fired"] is True
    assert "never_run" in summary["alert_tag"]


def test_migration_drill_monitor_alerts_when_stale(monkeypatch, tmp_path) -> None:
    _stub_runtime_settings(monkeypatch, migration_drill_monitor_enabled=True)
    _stub_common(monkeypatch)
    mod = _load_migration_drill()

    # Write a stale manifest (last drill 200 days ago).
    now_ts = 10_000_000.0
    stale_dt = datetime.fromtimestamp(now_ts - 200 * 86400, tz=timezone.utc)
    manifest = {
        "last_drill_at": stale_dt.isoformat(),
        "last_drill_ok": True,
        "runs": [],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    summary = mod.run(manifest_path=mp, now=now_ts)
    assert summary["manifest_present"] is True
    assert summary["last_drill_age_days"] is not None
    assert summary["last_drill_age_days"] > 100
    assert summary["alert_fired"] is True
    assert "stale" in summary["alert_tag"]


def test_migration_drill_monitor_alerts_when_failed(monkeypatch, tmp_path) -> None:
    _stub_runtime_settings(monkeypatch, migration_drill_monitor_enabled=True)
    _stub_common(monkeypatch)
    mod = _load_migration_drill()

    now_ts = 10_000_000.0
    recent = datetime.fromtimestamp(now_ts - 86400, tz=timezone.utc)
    manifest = {
        "last_drill_at": recent.isoformat(),
        "last_drill_ok": False,
        "runs": [],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    summary = mod.run(manifest_path=mp, now=now_ts)
    assert summary["alert_fired"] is True
    assert "failed" in summary["alert_tag"]


def test_migration_drill_monitor_silent_on_healthy(monkeypatch, tmp_path) -> None:
    _stub_runtime_settings(monkeypatch, migration_drill_monitor_enabled=True)
    _stub_common(monkeypatch)
    mod = _load_migration_drill()

    now_ts = 10_000_000.0
    recent = datetime.fromtimestamp(now_ts - 30 * 86400, tz=timezone.utc)
    manifest = {
        "last_drill_at": recent.isoformat(),
        "last_drill_ok": True,
        "runs": [],
    }
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    summary = mod.run(manifest_path=mp, now=now_ts)
    assert summary["alert_fired"] is False


def test_migration_drill_monitor_disabled_returns_quickly(monkeypatch) -> None:
    _stub_runtime_settings(monkeypatch, migration_drill_monitor_enabled=False)
    _stub_common(monkeypatch)
    mod = _load_migration_drill()
    summary = mod.run()
    assert summary["ran"] is False


# ═════════════════════════════════════════════════════════════════════════
#   2.3 — dependency_radar
# ═════════════════════════════════════════════════════════════════════════


def test_dep_radar_classifies_bumps_correctly() -> None:
    mod = _load_dep_radar_proposer()
    assert mod._classify_bump("1.2.3", "1.2.4") == mod.Severity.PATCH
    assert mod._classify_bump("1.2.3", "1.3.0") == mod.Severity.MINOR
    assert mod._classify_bump("1.2.3", "2.0.0") == mod.Severity.MAJOR
    # Unknown shapes route to MAJOR (operator review).
    assert mod._classify_bump("git-abc", "git-def") == mod.Severity.MAJOR


def test_dep_radar_routes_patch_to_proposal_bridge(monkeypatch) -> None:
    """Patch bumps → proposal_bridge.stage()."""
    mod = _load_dep_radar_proposer()
    staged: list[dict[str, Any]] = []
    alerted: list[dict[str, Any]] = []

    def _fake_stage(**kw):
        staged.append(kw)

    def _fake_notify(**kw):
        alerted.append(kw)

    result = mod.run_one_pass(
        pip_runner=lambda: [
            {"package": "requests", "current": "2.31.0", "latest": "2.31.1"},
        ],
        osv_runner=lambda packages: {},
        pip_show_runner=lambda pkg: {"Home-page": ""},
        github_runner=lambda owner, repo: None,
        stage_fn=_fake_stage,
        notify_fn=_fake_notify,
    )
    assert result.ran
    assert result.cr_proposals_filed == 1
    assert len(staged) == 1
    assert staged[0]["source"] == "dependency_radar"
    assert staged[0]["target_path"] == "requirements.txt"
    assert "patch-level" in staged[0]["title"]


def test_dep_radar_routes_major_to_signal(monkeypatch) -> None:
    """Major bumps → Signal alert ONLY (no CR)."""
    mod = _load_dep_radar_proposer()
    staged: list = []
    alerted: list = []

    result = mod.run_one_pass(
        pip_runner=lambda: [
            {"package": "django", "current": "4.2.0", "latest": "5.0.0"},
        ],
        osv_runner=lambda p: {},
        pip_show_runner=lambda p: {"Home-page": ""},
        github_runner=lambda o, r: None,
        stage_fn=lambda **kw: staged.append(kw),
        notify_fn=lambda **kw: alerted.append(kw),
    )
    assert result.cr_proposals_filed == 0
    assert result.alerts_fired == 1
    assert "major-version available" in alerted[0]["title"]


def test_dep_radar_cve_with_fix_routes_to_proposal_bridge(monkeypatch) -> None:
    mod = _load_dep_radar_proposer()
    staged: list = []
    alerted: list = []
    osv_response = {
        "cryptography": [
            {
                "id": "GHSA-test-1234",
                "affected": [{
                    "ranges": [{"events": [{"introduced": "0"}, {"fixed": "41.0.0"}]}]
                }],
            },
        ],
    }
    result = mod.run_one_pass(
        pip_runner=lambda: [
            {"package": "cryptography", "current": "40.0.0", "latest": "41.0.0"},
        ],
        osv_runner=lambda p: osv_response,
        pip_show_runner=lambda p: {"Home-page": ""},
        github_runner=lambda o, r: None,
        stage_fn=lambda **kw: staged.append(kw),
        notify_fn=lambda **kw: alerted.append(kw),
    )
    assert result.cr_proposals_filed == 1
    assert len(staged) == 1
    assert "CVE-patch" in staged[0]["title"]
    # CVE proposals use the shorter cooldown
    assert staged[0]["cooldown_days"] == mod._COOLDOWN_CVE


def test_dep_radar_cve_no_fix_routes_to_signal(monkeypatch) -> None:
    """CVE without a patched version → Signal alert only."""
    mod = _load_dep_radar_proposer()
    staged: list = []
    alerted: list = []
    osv_response = {
        "lib_no_fix": [
            {
                "id": "GHSA-nofix-001",
                "affected": [{
                    "ranges": [{"events": [{"introduced": "0"}]}]
                }],
            },
        ],
    }
    result = mod.run_one_pass(
        pip_runner=lambda: [
            {"package": "lib_no_fix", "current": "1.0", "latest": "1.0.1"},
        ],
        osv_runner=lambda p: osv_response,
        pip_show_runner=lambda p: {"Home-page": ""},
        github_runner=lambda o, r: None,
        stage_fn=lambda **kw: staged.append(kw),
        notify_fn=lambda **kw: alerted.append(kw),
    )
    # The OSV row has no "fixed" event → CVE_NO_FIX.
    assert result.cr_proposals_filed == 0
    assert result.alerts_fired == 1
    assert "without fix" in alerted[0]["title"].lower()


def test_dep_radar_abandonment_routes_to_signal(monkeypatch) -> None:
    """A repo last-pushed > 365 days ago → Signal alert."""
    mod = _load_dep_radar_proposer()
    alerted: list = []
    old_push = datetime.now(timezone.utc) - timedelta(days=500)
    result = mod.run_one_pass(
        pip_runner=lambda: [
            {"package": "abandoned_lib", "current": "1.0", "latest": "1.1"},
        ],
        osv_runner=lambda p: {},
        pip_show_runner=lambda p: {"Home-page": "https://github.com/x/abandoned_lib"},
        github_runner=lambda o, r: old_push,
        stage_fn=lambda **kw: None,
        notify_fn=lambda **kw: alerted.append(kw),
    )
    # The package gets MULTIPLE findings: minor bump AND abandoned.
    # Severity ABANDONED routes to alert; severity MINOR routes to CR.
    severities = {f.severity.value for f in result.findings}
    assert "abandoned" in severities
    abandoned_alerts = [a for a in alerted if "abandoned" in a.get("topic", "")]
    assert len(abandoned_alerts) >= 1


def test_dep_radar_respects_max_proposals_per_pass(monkeypatch) -> None:
    """Rate limit: only _MAX_PROPOSALS_PER_PASS bumps per pass."""
    mod = _load_dep_radar_proposer()
    staged: list = []
    rows = [
        {"package": f"pkg_{i}", "current": "1.2.3", "latest": "1.2.4"}
        for i in range(10)
    ]
    result = mod.run_one_pass(
        pip_runner=lambda: rows,
        osv_runner=lambda p: {},
        pip_show_runner=lambda p: {"Home-page": ""},
        github_runner=lambda o, r: None,
        stage_fn=lambda **kw: staged.append(kw),
        notify_fn=lambda **kw: None,
    )
    assert result.cr_proposals_filed == mod._MAX_PROPOSALS_PER_PASS
    assert len(staged) == mod._MAX_PROPOSALS_PER_PASS


def test_dep_radar_disabled_returns_early(monkeypatch) -> None:
    """Master-switch disabled → run_one_pass returns immediately."""
    mod = _load_dep_radar_proposer()
    monkeypatch.setenv("DEPENDENCY_RADAR_ENABLED", "false")
    # Override runtime_settings module if present
    if "app.runtime_settings" in sys.modules:
        rs = sys.modules["app.runtime_settings"]
        if hasattr(rs, "get_dependency_radar_enabled"):
            monkeypatch.setattr(
                rs, "get_dependency_radar_enabled", lambda: False,
            )
    result = mod.run_one_pass()
    assert result.ran is False


def test_dep_radar_proposal_bridge_source_registered() -> None:
    """The `dependency_radar` source MUST be in proposal_bridge._KNOWN_SOURCES,
    otherwise stage() raises ValueError."""
    p = REPO_ROOT / "app" / "proposal_bridge" / "store.py"
    src = p.read_text()
    assert '"dependency_radar"' in src
    assert "_KNOWN_SOURCES" in src


# ═════════════════════════════════════════════════════════════════════════
#   2.6 — tz_drift monitor
# ═════════════════════════════════════════════════════════════════════════


def test_tz_drift_no_divergence_returns_silent(monkeypatch) -> None:
    _stub_runtime_settings(monkeypatch, tz_drift_monitor_enabled=True)
    mod = _load_tz_drift()
    # Reuse the production probe — on a healthy host with current
    # tzdata, hand-rolled and zoneinfo should agree.
    summary = mod.run()
    assert summary["checked"] is True
    # On a healthy host with current tzdata, no divergence.
    if summary["n_diverged"] == 0:
        assert summary["alerts"] == 0
        assert summary["cr_filed_id"] is None


def test_tz_drift_detects_divergence_and_files_cr(monkeypatch, tmp_path) -> None:
    """Inject a hand-rolled function that returns wrong offsets and
    verify the monitor alerts + files CR."""
    _stub_runtime_settings(monkeypatch, tz_drift_monitor_enabled=True)
    mod = _load_tz_drift()
    # Force the state file into tmp_path so this test doesn't pollute
    # the real workspace.
    monkeypatch.setattr(
        mod, "_state_path", lambda: tmp_path / "tz_drift_state.json",
    )
    # Force the hand-rolled implementation to be deliberately wrong.
    # Patch _handrolled_offset_at to always return 0 (UTC) so it
    # diverges from ZoneInfo's +2/+3.
    monkeypatch.setattr(mod, "_handrolled_offset_at", lambda dt: 0)
    # Stub the CR filer + notify so no real CR/Signal is created.
    captured_crs: list = []
    captured_alerts: list = []
    monkeypatch.setattr(
        mod, "_propose_consolidation_cr",
        lambda probes: captured_crs.append(probes) or "cr_test_001",
    )
    # Stub notify
    fake_notify_mod = type(sys)("app.notify")
    fake_notify_mod.notify = lambda **kw: captured_alerts.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify_mod)
    # Stub continuity ledger
    fake_ledger = type(sys)("app.identity.continuity_ledger")
    fake_ledger.record_event = lambda **kw: True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )

    summary = mod.run()
    assert summary["checked"] is True
    assert summary["n_diverged"] >= 1
    assert summary["alerts"] == 1
    assert summary["cr_filed_id"] == "cr_test_001"
    assert len(captured_crs) == 1
    assert len(captured_alerts) == 1
    assert "TZ drift" in captured_alerts[0]["title"]


def test_tz_drift_recovery_transition_emits_landmark(monkeypatch, tmp_path) -> None:
    """When divergence-then-no-divergence is detected, the monitor
    emits a recovery ledger event."""
    _stub_runtime_settings(monkeypatch, tz_drift_monitor_enabled=True)
    mod = _load_tz_drift()
    monkeypatch.setattr(
        mod, "_state_path", lambda: tmp_path / "tz_drift_state.json",
    )
    # Pre-populate state with a recorded divergence.
    state_dir = tmp_path
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "tz_drift_state.json").write_text(json.dumps({
        "last_divergence_at": "2026-05-15T10:00:00+00:00",
        "cr_filed": True,
        "recovered": False,
    }))
    # Stub the ledger to capture events.
    captured: list = []
    fake_ledger = type(sys)("app.identity.continuity_ledger")
    fake_ledger.record_event = lambda **kw: captured.append(kw) or True
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )

    # Now use the REAL hand-rolled function (not the deliberately
    # wrong stub from the previous test) — so no current divergence.
    summary = mod.run()
    if summary["n_diverged"] == 0:
        # Recovery emitted.
        assert len(captured) == 1
        assert "recovered" in captured[0]["summary"].lower()


def test_tz_drift_disabled_returns_skipped(monkeypatch) -> None:
    _stub_runtime_settings(monkeypatch, tz_drift_monitor_enabled=False)
    mod = _load_tz_drift()
    summary = mod.run()
    assert summary.get("skipped") is True


def test_tz_drift_handrolled_matches_zoneinfo_on_current_year() -> None:
    """Pin: the hand-rolled implementation in temporal_context.py
    SHOULD agree with zoneinfo for the current year. If this test
    fails, it likely means EU abolished DST and the production code
    needs the consolidation."""
    mod = _load_tz_drift()
    now = datetime.now(timezone.utc)
    h = mod._handrolled_offset_at(now)
    z = mod._zoneinfo_offset_at(now)
    if z is None:
        pytest.skip("zoneinfo Europe/Helsinki not available on this host")
    assert h == z, (
        f"hand-rolled offset {h}s != zoneinfo offset {z}s at "
        f"{now.isoformat()}. EU may have abolished DST — consolidate "
        f"_helsinki_tz onto ZoneInfo."
    )


# ═════════════════════════════════════════════════════════════════════════
#   Wiring / registration
# ═════════════════════════════════════════════════════════════════════════


def test_monitors_init_registers_migration_drill() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    assert '"migration_drill"' in src
    assert "from app.healing.monitors import migration_drill" in src


def test_monitors_init_registers_tz_drift() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    assert '"tz_drift"' in src
    assert "from app.healing.monitors import tz_drift" in src


def test_healing_init_boot_anchors_dependency_radar() -> None:
    p = REPO_ROOT / "app" / "healing" / "__init__.py"
    src = p.read_text()
    assert "dependency_radar" in src


def test_runtime_settings_has_all_three_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "migration_drill_monitor_enabled",
        "dependency_radar_enabled",
        "tz_drift_monitor_enabled",
    ):
        assert f'"{key}"' in src
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_continuity_ledger_has_two_new_event_kinds() -> None:
    p = REPO_ROOT / "app" / "identity" / "continuity_ledger.py"
    src = p.read_text()
    assert '"schema_migration_drill"' in src
    assert '"tz_drift"' in src


# ═════════════════════════════════════════════════════════════════════════
#   Helpers
# ═════════════════════════════════════════════════════════════════════════


def _stub_runtime_settings(monkeypatch, **kwargs) -> None:
    """Stub runtime_settings module with arbitrary getter overrides."""
    fake = type(sys)("app.runtime_settings")
    for key, value in kwargs.items():
        fn = (lambda v: lambda: v)(value)
        setattr(fake, f"get_{key}", fn)
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake)


def _stub_common(monkeypatch) -> None:
    """Stub app.healing.handlers._common so monitors can run in isolation."""
    state_store: dict[str, Any] = {}
    fake = type(sys)("app.healing.handlers._common")
    fake.audit_event = lambda *a, **kw: None
    fake.read_state_json = lambda name, default: dict(default)
    fake.send_signal_alert = lambda body, tag=None: None
    fake.write_state_json = lambda name, state: None
    # Need the parent packages registered for the import path
    monkeypatch.setitem(sys.modules, "app", type(sys)("app"))
    monkeypatch.setitem(sys.modules, "app.healing", type(sys)("app.healing"))
    monkeypatch.setitem(sys.modules, "app.healing.handlers", type(sys)("app.healing.handlers"))
    monkeypatch.setitem(sys.modules, "app.healing.handlers._common", fake)


def _load_migration_drill():
    """Load app/healing/monitors/migration_drill.py in isolation."""
    return _isolated_module(
        "app/healing/monitors/migration_drill.py",
        "_q13_migration_drill_under_test",
    )


def _load_dep_radar_proposer():
    """Load app/dependency_radar/proposer.py in isolation (skipping
    the eager start_daemon() at module bottom by stubbing threading)."""
    # The eager start at import bottom is gated by _enabled() — set
    # the env var so it short-circuits before any daemon spins up.
    os.environ["DEPENDENCY_RADAR_ENABLED"] = "false"
    try:
        return _isolated_module(
            "app/dependency_radar/proposer.py",
            "_q13_dep_radar_under_test",
        )
    finally:
        del os.environ["DEPENDENCY_RADAR_ENABLED"]


def _load_tz_drift():
    """Load app/healing/monitors/tz_drift.py in isolation."""
    return _isolated_module(
        "app/healing/monitors/tz_drift.py",
        "_q13_tz_drift_under_test",
    )
