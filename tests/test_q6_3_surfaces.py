"""PROGRAM §44.3 — Q6.3 surface tests.

Covers:
  * 4 REST endpoints under /api/cp/drills/* declared in dashboard_api
  * 6 master switches handled in config_api setter block
  * React types updated with new fields
  * Briefing weekly digest gathers drill state
  * DR export includes resilience/
  * Operator guide doc exists + names the four drills
"""
from __future__ import annotations

import importlib.util
import json
import sys
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
#   REST endpoints (source-level)
# ─────────────────────────────────────────────────────────────────────────


def test_dashboard_api_has_four_drill_endpoints():
    src = Path("app/control_plane/dashboard_api.py").read_text()
    for path in (
        '"/drills/registry"',
        '"/drills/audit"',
        '"/drills/run/{name}"',
        '"/drills/posture"',
    ):
        assert path in src, f"missing endpoint {path}"


def test_config_api_handles_all_six_master_switches():
    src = Path("app/api/config_api.py").read_text()
    for setting in (
        "resilience_drills_enabled",
        "drill_backup_restore_enabled",
        "drill_embedding_migration_enabled",
        "drill_secret_rotation_enabled",
        "drill_kill_the_gateway_enabled",
        "drill_staleness_monitor_enabled",
    ):
        assert f'"{setting}" in payload' in src, f"missing handler for {setting}"


# ─────────────────────────────────────────────────────────────────────────
#   React types
# ─────────────────────────────────────────────────────────────────────────


def test_react_runtime_settings_type_extended():
    src = Path("dashboard-react/src/api/queries.ts").read_text()
    for field in (
        "resilience_drills_enabled",
        "drill_kill_the_gateway_enabled",
        "drill_staleness_monitor_enabled",
    ):
        assert field in src, f"RuntimeSettings missing {field}"


def test_settings_page_mounts_resilience_card():
    src = Path("dashboard-react/src/components/SettingsPage.tsx").read_text()
    assert "ResilienceDrillsCard" in src
    assert "import { ResilienceDrillsCard }" in src


def test_resilience_drills_card_exists():
    path = Path("dashboard-react/src/components/ResilienceDrillsCard.tsx")
    assert path.exists()
    body = path.read_text()
    assert "drill_kill_the_gateway_enabled" in body
    # Warning text for the disruptive drill must be present.
    assert "DISRUPTIVE" in body
    assert "EXECUTE KILL DRILL" in body


# ─────────────────────────────────────────────────────────────────────────
#   Briefing weekly digest
# ─────────────────────────────────────────────────────────────────────────


def test_briefing_gathers_drill_digest_empty_when_nothing(monkeypatch):
    briefing = _load_isolated(
        "br_q63", "app/life_companion/daily_briefing.py",
    )
    # Force registry to be empty by clearing — RESTORE in finally so
    # subsequent tests still have populated registrations.
    from app.resilience_drills.protocol import get_registry
    reg = get_registry()
    saved_specs = list(reg._specs.values())
    saved_runners = dict(reg._runners)
    reg.clear_for_tests()
    try:
        monkeypatch.setattr(
            "app.resilience_drills.audit.iter_results",
            lambda since_iso=None: iter([]),
        )
        monkeypatch.setattr(
            "app.resilience_drills.audit.days_since_last_success",
            lambda name: None,
        )
        lines = briefing._gather_resilience_drill_digest()
        assert lines == []
    finally:
        # Repopulate registry so subsequent tests have the drills back.
        for spec in saved_specs:
            reg.register(spec, saved_runners.get(spec.name) or (lambda **kw: None))


def test_briefing_surfaces_recent_failures(monkeypatch):
    briefing = _load_isolated(
        "br_q63b", "app/life_companion/daily_briefing.py",
    )
    # Populate registry.
    import app.resilience_drills.drills  # noqa: F401
    # Recent FAIL row in iter_results.
    recent_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    monkeypatch.setattr(
        "app.resilience_drills.audit.iter_results",
        lambda since_iso=None: iter([
            {"drill_name": "backup_restore", "status": "fail",
             "started_at": recent_iso},
        ]),
    )
    monkeypatch.setattr(
        "app.resilience_drills.audit.days_since_last_success",
        lambda name: 30.0,
    )
    monkeypatch.setattr(
        "app.resilience_drills.protocol.drill_enabled", lambda spec: True,
    )
    lines = briefing._gather_resilience_drill_digest()
    assert any("FAILED" in ln for ln in lines)
    assert any("backup_restore" in ln for ln in lines)


# ─────────────────────────────────────────────────────────────────────────
#   DR export inclusion
# ─────────────────────────────────────────────────────────────────────────


def test_dr_export_includes_resilience():
    """``resilience/`` is in the DR _LEDGER_INCLUDES allow-list so
    drill_audit.jsonl + kill_drill_*.json get backed up."""
    src = Path("app/dr/export_kbs.py").read_text()
    # The allow-list entry must be present + commented as Q6.3.
    assert '"resilience/"' in src
    assert "Q6.3" in src or "§44.3" in src


# ─────────────────────────────────────────────────────────────────────────
#   Operator guide
# ─────────────────────────────────────────────────────────────────────────


def test_resilience_drills_doc_exists_and_names_all_four():
    doc = Path("docs/RESILIENCE_DRILLS.md").read_text()
    for name in ("backup_restore", "embedding_migration",
                 "secret_rotation", "kill_the_gateway"):
        assert name in doc, f"doc missing drill {name}"
    # Names the posture decision.
    assert "RESILIENCE_POSTURE.md" in doc
    # Names master switches.
    assert "drill_kill_the_gateway_enabled" in doc
    # Names the REST endpoints.
    assert "/api/cp/drills/" in doc
