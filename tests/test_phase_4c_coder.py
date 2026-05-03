"""Tests for Phase 4c — coder migration to LoadableAgent.

Mirrors tests/test_phase_4a_researcher.py + test_phase_4b_writer.py
for the coder agent. The coder is the highest-stakes Phase 4
migration so far — it executes code, calls Forge, produces
user-deliverable artifacts. Default-OFF + failsafe fallback are
load-bearing.

Covers:
  * **dispatch matrix** — default legacy, flag → loadable, master
    flag → loadable, per-agent off overrides master, failsafe.
  * **eager-toolset parity** — LoadableAgent's eager set differs
    from legacy by exactly the binder control tools; every legacy
    tool is present.
  * **discoverable capabilities** — geodata + currency surface
    correctly.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestCoderDispatch:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_default_is_legacy_agent(self, monkeypatch):
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_CODER", raising=False)
        from app.agents.coder import create_coder
        agent = create_coder()
        assert type(agent).__name__ == "Agent"

    def test_loadable_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_CODER", "1")
        from app.agents.coder import create_coder
        agent = create_coder()
        assert type(agent).__name__ == "LoadableAgent"

    def test_master_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.delenv("LOADABLE_CODER", raising=False)
        from app.agents.coder import create_coder
        agent = create_coder()
        assert type(agent).__name__ == "LoadableAgent"

    def test_per_agent_off_overrides_master(self, monkeypatch):
        """Master ON, coder OFF → coder stays legacy.

        Critical for staged rollout: an operator can have all other
        migrated agents on the new path while keeping coder (the
        highest-stakes one) on legacy until its parity panel passes.
        """
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.setenv("LOADABLE_CODER", "0")
        from app.agents.coder import create_coder
        agent = create_coder()
        assert type(agent).__name__ == "Agent"

    def test_loadable_failure_falls_back_to_legacy(self, monkeypatch):
        """If the loadable factory raises, legacy runs instead.

        Critical for the coder: a Phase 4c bug must not break the
        user-facing PDF/Signal flow. Failsafe means the worst case
        is "experimental path is wasted" not "user can't get a
        report this cycle."
        """
        monkeypatch.setenv("LOADABLE_CODER", "1")
        from app.agents import coder

        with patch.object(
            coder, "_build_loadable_coder",
            side_effect=RuntimeError("simulated bug"),
        ):
            agent = coder.create_coder()
        assert type(agent).__name__ == "Agent"


class TestCoderEagerToolsetParity:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_loadable_eager_count_matches_legacy(self, monkeypatch):
        """Eager toolset parity — loadable adds ONLY the 2 binder
        control tools to the legacy set."""
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_CODER", raising=False)
        from app.agents.coder import create_coder
        legacy = create_coder()
        legacy_names = {t.name for t in legacy.tools}

        monkeypatch.setenv("LOADABLE_CODER", "1")
        from app.agents import coder
        loadable = coder.create_coder()
        loadable_names = {t.name for t in loadable.tools}

        added = loadable_names - legacy_names
        assert added == {"load_tool", "list_available_tools"}, (
            f"Unexpected tool delta: {added}. Loadable should add "
            "ONLY the binder control tools."
        )
        missing = legacy_names - loadable_names
        assert not missing, f"Loadable missing legacy tools: {missing}"

    def test_loadable_has_discoverable_capabilities(self, monkeypatch):
        """Discoverable catalog should resolve to registered tools.

        Coder's discoverable set: fetches-geodata + converts-currency
        + fetches-finance. None of these are in the eager toolset.
        """
        monkeypatch.setenv("LOADABLE_CODER", "1")
        from app.agents.coder import create_coder
        agent = create_coder()
        catalog = agent.binder.catalog_names()
        # Geodata
        assert any(t.startswith("geodata_") for t in catalog), (
            f"No geodata tool in catalog: {catalog}"
        )
        # Currency
        assert "currency_convert" in catalog or "currency_rates" in catalog
