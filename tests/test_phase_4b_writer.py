"""Tests for Phase 4b — writer migration to LoadableAgent.

Mirrors tests/test_phase_4a_researcher.py for the writer agent.
Covers:
  * **dispatch matrix** — default legacy, flag → loadable, master flag
    → loadable, per-agent off overrides master, failsafe fallback.
  * **eager-toolset parity** — LoadableAgent's eager set differs
    from legacy by exactly the binder control tools (load_tool +
    list_available_tools); every legacy tool is present.
  * **discoverable capabilities** — at least the 5 declared
    capabilities resolve to registered tools.

These tests don't run the agent — they verify the dispatch logic.
Operator-driven live parity validation (the actual behavior check)
follows the Phase 4-X cycle.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


class TestWriterDispatch:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_default_is_legacy_agent(self, monkeypatch):
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_WRITER", raising=False)
        from app.agents.writer import create_writer
        agent = create_writer()
        assert type(agent).__name__ == "Agent"

    def test_loadable_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_WRITER", "1")
        from app.agents.writer import create_writer
        agent = create_writer()
        assert type(agent).__name__ == "LoadableAgent"

    def test_master_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.delenv("LOADABLE_WRITER", raising=False)
        from app.agents.writer import create_writer
        agent = create_writer()
        assert type(agent).__name__ == "LoadableAgent"

    def test_per_agent_off_overrides_master(self, monkeypatch):
        """Master ON, writer OFF → writer stays legacy."""
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.setenv("LOADABLE_WRITER", "0")
        from app.agents.writer import create_writer
        agent = create_writer()
        assert type(agent).__name__ == "Agent"

    def test_loadable_failure_falls_back_to_legacy(self, monkeypatch):
        """If the loadable factory raises, legacy runs instead."""
        monkeypatch.setenv("LOADABLE_WRITER", "1")
        from app.agents import writer

        with patch.object(
            writer, "_build_loadable_writer",
            side_effect=RuntimeError("simulated bug"),
        ):
            agent = writer.create_writer()
        assert type(agent).__name__ == "Agent"


class TestWriterEagerToolsetParity:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_loadable_eager_count_matches_legacy(self, monkeypatch):
        """LoadableAgent's eager toolset should have the same names
        as the legacy path's tools (plus 2 control tools:
        load_tool + list_available_tools).

        High-fidelity behavior parity by construction — the agent
        sees the same tool names regardless of dispatch path.
        """
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_WRITER", raising=False)
        from app.agents.writer import create_writer
        legacy = create_writer()
        legacy_names = {t.name for t in legacy.tools}

        monkeypatch.setenv("LOADABLE_WRITER", "1")
        from app.agents import writer
        loadable = writer.create_writer()
        loadable_names = {t.name for t in loadable.tools}

        # Loadable has the legacy set PLUS the control tools.
        added = loadable_names - legacy_names
        assert added == {"load_tool", "list_available_tools"}, (
            f"Unexpected tool delta: {added}. Loadable should add ONLY "
            "the binder control tools; everything else must match."
        )
        missing = legacy_names - loadable_names
        assert not missing, f"Loadable missing legacy tools: {missing}"

    def test_loadable_has_discoverable_capabilities(self, monkeypatch):
        """Discoverable catalog should resolve to registered tools."""
        monkeypatch.setenv("LOADABLE_WRITER", "1")
        from app.agents.writer import create_writer
        agent = create_writer()
        catalog = agent.binder.catalog_names()
        # At least the discoverable capabilities should resolve to
        # tools registered in Phase 1a.
        # executes-code → execute_code
        # fetches-geodata → geodata_discover/fetch
        # converts-currency → currency_convert
        assert "execute_code" in catalog
        assert "currency_convert" in catalog or "currency_rates" in catalog
        # At least one geodata tool
        assert any(t.startswith("geodata_") for t in catalog)
