"""Tests for the React-toggleable migrate_live_execute setting.

Pins the contract:
  * runtime_settings exposes get/set_migrate_live_execute
  * Flips emit a `cloud_migration:execute_policy_changed` ledger event
  * Setting works through the /api/cp/settings dispatcher
  * cloud_prep.is_live_execute_enabled is the single resolver
  * Either env var OR runtime_settings True enables the gate (OR semantics)
"""
import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def isolated_workspace(tmp_path, monkeypatch):
    """Re-route WORKSPACE_ROOT so runtime_settings.json + ledger land in tmp."""
    from app import paths as _paths
    monkeypatch.setattr(_paths, "WORKSPACE_ROOT", tmp_path)
    # Reset runtime_settings cache so it re-reads from the (empty) tmp file
    from app import runtime_settings as rs
    monkeypatch.setattr(rs, "_cache", None)
    monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
    # And the ledger path so emissions land in tmp
    from app.identity import continuity_ledger as cl
    monkeypatch.setattr(cl, "_path_override", tmp_path / "identity" / "continuity_ledger.jsonl")
    return tmp_path


# ── Runtime-settings layer ──────────────────────────────────────────


class TestRuntimeSettingsField:
    def test_default_is_false_without_env_var(self, isolated_workspace, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app import runtime_settings as rs
        assert rs.get_migrate_live_execute() is False

    def test_default_seeded_from_env_var(self, isolated_workspace, monkeypatch):
        """First read of a fresh runtime_settings.json picks up the env var."""
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        from app import runtime_settings as rs
        # Reset cache so the next get triggers re-seed
        monkeypatch.setattr(rs, "_cache", None)
        assert rs.get_migrate_live_execute() is True

    def test_set_then_get(self, isolated_workspace, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app import runtime_settings as rs
        rs.set_migrate_live_execute(True)
        assert rs.get_migrate_live_execute() is True
        rs.set_migrate_live_execute(False)
        assert rs.get_migrate_live_execute() is False

    def test_set_persists_to_disk(self, isolated_workspace, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app import runtime_settings as rs
        rs.set_migrate_live_execute(True)
        state = json.loads((isolated_workspace / "runtime_settings.json").read_text())
        assert state["migrate_live_execute"] is True


# ── Ledger emission ────────────────────────────────────────────────


class TestLedgerEmissionOnFlip:
    def test_flip_emits_cloud_migration_event(self, isolated_workspace, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        from app import runtime_settings as rs
        rs.set_migrate_live_execute(True)

        ledger = isolated_workspace / "identity" / "continuity_ledger.jsonl"
        assert ledger.exists()
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        assert len(cm) == 1
        ev = cm[0]
        assert ev["detail"]["phase"] == "execute_policy_changed"
        assert ev["detail"]["prior"] is False
        assert ev["detail"]["new"] is True
        # Operator-facing summary must say what changed
        assert "real cloud spend" in ev["summary"]

    def test_no_event_when_value_unchanged(self, isolated_workspace, monkeypatch):
        """Idempotent set (set(False) when already False) must NOT
        emit a no-op event — annual reflection would see noise."""
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        from app import runtime_settings as rs
        rs.set_migrate_live_execute(False)   # already False default

        ledger = isolated_workspace / "identity" / "continuity_ledger.jsonl"
        if ledger.exists():
            events = [json.loads(line) for line in ledger.read_text().splitlines()]
            cm = [e for e in events if e["kind"] == "cloud_migration"]
            assert cm == [], "no-op set must not emit a ledger event"

    def test_flip_back_to_false_emits_too(self, isolated_workspace, monkeypatch):
        """Going OFF is also identity-shaping (operator decided cloud is no
        longer safe to spend); both transitions deserve a landmark."""
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

        from app import runtime_settings as rs
        rs.set_migrate_live_execute(True)
        rs.set_migrate_live_execute(False)

        ledger = isolated_workspace / "identity" / "continuity_ledger.jsonl"
        events = [json.loads(line) for line in ledger.read_text().splitlines()]
        cm = [e for e in events if e["kind"] == "cloud_migration"]
        assert len(cm) == 2
        # Operator-facing summaries describe both directions
        assert "real cloud spend now possible" in cm[0]["summary"]
        assert "returned to report-only mode" in cm[1]["summary"]


# ── Single resolver — cloud_prep.is_live_execute_enabled ───────────


class TestSingleResolver:
    def test_both_off_returns_false(self, isolated_workspace, monkeypatch):
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app.substrate.cloud_prep import is_live_execute_enabled
        assert is_live_execute_enabled() is False

    def test_env_var_alone_enables(self, isolated_workspace, monkeypatch):
        """Legacy CLI workflow: env var on, runtime_settings False → True."""
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        from app import runtime_settings as rs
        monkeypatch.setattr(rs, "_cache", None)
        rs.set_migrate_live_execute(False)  # explicit
        from app.substrate.cloud_prep import is_live_execute_enabled
        assert is_live_execute_enabled() is True

    def test_runtime_settings_alone_enables(self, isolated_workspace, monkeypatch):
        """React toggle alone: env var off, runtime_settings True → True."""
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app import runtime_settings as rs
        rs.set_migrate_live_execute(True)
        from app.substrate.cloud_prep import is_live_execute_enabled
        assert is_live_execute_enabled() is True

    def test_disable_requires_both_off(self, isolated_workspace, monkeypatch):
        """To actually disable execution, BOTH sources must be False."""
        monkeypatch.delenv("BOTARMY_MIGRATE_LIVE_EXECUTE", raising=False)
        from app import runtime_settings as rs
        rs.set_migrate_live_execute(False)
        from app.substrate.cloud_prep import is_live_execute_enabled
        assert is_live_execute_enabled() is False

    def test_broken_runtime_settings_falls_back_to_env_only(self, monkeypatch, isolated_workspace):
        """If runtime_settings module is broken, env-var path still works."""
        monkeypatch.setenv("BOTARMY_MIGRATE_LIVE_EXECUTE", "1")
        # Simulate runtime_settings explosion by patching the getter
        from app import runtime_settings as rs

        def _boom():
            raise RuntimeError("simulated runtime_settings explosion")

        monkeypatch.setattr(rs, "get_migrate_live_execute", _boom)
        from app.substrate.cloud_prep import is_live_execute_enabled
        # Env var alone is enough
        assert is_live_execute_enabled() is True


# ── Cross-module consistency ───────────────────────────────────────


class TestAllShellsUseSameResolver:
    """Pins that migration._shell, cloud_prep._shell, cutover._shell all
    consult the same is_live_execute_enabled resolver. Without this pin,
    a future edit could re-introduce divergent execute-gate logic across
    the three modules — exactly the drift this refactor was meant to
    eliminate.
    """

    def test_migration_shell_calls_resolver(self):
        src = (Path(__file__).parent.parent / "app/substrate/migration.py").read_text()
        assert "is_live_execute_enabled" in src

    def test_cloud_prep_shell_calls_resolver(self):
        src = (Path(__file__).parent.parent / "app/substrate/cloud_prep.py").read_text()
        # cloud_prep DEFINES the resolver AND uses it inside _shell
        assert "def is_live_execute_enabled" in src
        # _shell body must reference it
        idx = src.find("def _shell(")
        assert idx > 0
        # The next ~40 lines should reference it
        assert "is_live_execute_enabled" in src[idx : idx + 800]

    def test_cutover_shell_calls_resolver(self):
        src = (Path(__file__).parent.parent / "app/substrate/cutover.py").read_text()
        assert "is_live_execute_enabled" in src

    def test_migrate_api_uses_resolver_not_env_direct(self):
        """The REST endpoint must use the resolver, not re-implement the
        env-var check (which would skip runtime_settings)."""
        src = (Path(__file__).parent.parent / "app/control_plane/migrate_api.py").read_text()
        # Source must reference the canonical resolver
        assert "is_live_execute_enabled" in src
        # And must NOT independently check the env var (would cause drift)
        assert 'os.environ.get(\n        "BOTARMY_MIGRATE_LIVE_EXECUTE"' not in src
        assert 'os.environ.get("BOTARMY_MIGRATE_LIVE_EXECUTE"' not in src


# ── Dispatcher (POST /api/cp/settings) ─────────────────────────────


class TestSettingsAPIDispatcher:
    def test_dispatcher_imports_setter(self):
        """The settings dispatcher must import set_migrate_live_execute or
        the React POST won't route correctly."""
        src = (Path(__file__).parent.parent / "app/api/config_api.py").read_text()
        assert "set_migrate_live_execute" in src
        # The dispatch arm must reference the payload key
        assert '"migrate_live_execute" in payload' in src
