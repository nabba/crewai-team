"""Tests for Phase 4a — researcher migration to LoadableAgent.

Covers:
  * **feature_flags helper** — per-agent flag matrix (per-agent
    explicit overrides master, master sets default for unflagged
    agents, both unset = legacy default).
  * **researcher dispatch** — flag matrix produces correct agent
    classes; light path always legacy.
  * **failsafe fallback** — experimental factory exception →
    legacy path runs, warning logged.
  * **eager-tool parity** — LoadableAgent's eager toolset matches
    the legacy full path's tool count (so behavior parity is high
    by construction).

These tests don't run the agent — they verify the dispatch logic.
Operator-driven live parity (the actual behavior validation)
follows the Phase 2.5 / Phase 4 cycle: set the flag in staging,
run a representative task panel, compare success rates.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ── feature_flags helper ─────────────────────────────────────────────


class TestFeatureFlags:

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_RESEARCHER", raising=False)
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("researcher") is False

    def test_master_alone_enables(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.delenv("LOADABLE_RESEARCHER", raising=False)
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("researcher") is True
        assert is_loadable_for("coder") is True       # any agent
        assert is_loadable_for("writer") is True

    def test_per_agent_overrides_master_off(self, monkeypatch):
        """Master ON, but per-agent set to '0' → that agent is OFF."""
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.setenv("LOADABLE_RESEARCHER", "0")
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("researcher") is False
        # Other agents inherit master.
        assert is_loadable_for("coder") is True

    def test_per_agent_overrides_master_on(self, monkeypatch):
        """Master OFF, but per-agent set to '1' → that agent is ON."""
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("researcher") is True
        assert is_loadable_for("coder") is False

    def test_per_agent_zero_overrides_master_unset(self, monkeypatch):
        """Per-agent set to '0' overrides even when master is unset."""
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.setenv("LOADABLE_RESEARCHER", "0")
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("researcher") is False

    def test_case_insensitive_agent_name(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.tool_runtime.feature_flags import is_loadable_for
        assert is_loadable_for("Researcher") is True
        assert is_loadable_for("RESEARCHER") is True
        assert is_loadable_for("researcher") is True

    def test_explicit_flag_for_returns_value(self, monkeypatch):
        from app.tool_runtime.feature_flags import explicit_flag_for
        monkeypatch.setenv("LOADABLE_RESEARCHER", "0")
        assert explicit_flag_for("researcher") == "0"
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        assert explicit_flag_for("researcher") == "1"
        monkeypatch.delenv("LOADABLE_RESEARCHER")
        assert explicit_flag_for("researcher") is None

    def test_master_query(self, monkeypatch):
        from app.tool_runtime.feature_flags import is_master_on
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        assert is_master_on() is False
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        assert is_master_on() is True
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "0")
        assert is_master_on() is False


# ── researcher dispatch ──────────────────────────────────────────────


class TestResearcherDispatch:

    def setup_method(self) -> None:
        # Boot registry so capability resolution works in loadable path.
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_default_is_legacy_agent(self, monkeypatch):
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_RESEARCHER", raising=False)
        from app.agents.researcher import create_researcher
        agent = create_researcher()
        assert type(agent).__name__ == "Agent"

    def test_loadable_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.agents.researcher import create_researcher
        agent = create_researcher()
        assert type(agent).__name__ == "LoadableAgent"

    def test_master_flag_produces_loadable_agent(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.delenv("LOADABLE_RESEARCHER", raising=False)
        from app.agents.researcher import create_researcher
        agent = create_researcher()
        assert type(agent).__name__ == "LoadableAgent"

    def test_light_path_always_legacy(self, monkeypatch):
        """Light path is the small-tool fast-path; flag has no effect."""
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.agents.researcher import create_researcher
        agent = create_researcher(light=True)
        assert type(agent).__name__ == "Agent"

    def test_per_agent_off_overrides_master(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        monkeypatch.setenv("LOADABLE_RESEARCHER", "0")
        from app.agents.researcher import create_researcher
        agent = create_researcher()
        assert type(agent).__name__ == "Agent"

    def test_loadable_failure_falls_back_to_legacy(self, monkeypatch):
        """If the loadable factory raises, legacy runs instead."""
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.agents import researcher

        with patch.object(
            researcher, "_build_loadable_researcher",
            side_effect=RuntimeError("simulated bug"),
        ):
            agent = researcher.create_researcher()
        assert type(agent).__name__ == "Agent"


# ── eager-toolset parity ─────────────────────────────────────────────


class TestEagerToolsetParity:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)

    def test_loadable_eager_count_matches_legacy_full(self, monkeypatch):
        """LoadableAgent's eager toolset should have the same names
        as the legacy full path's tools (plus 2 control tools:
        load_tool + list_available_tools).

        High-fidelity behavior parity by construction — the agent
        sees the same tool names regardless of dispatch path.
        """
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        monkeypatch.delenv("LOADABLE_RESEARCHER", raising=False)
        from app.agents.researcher import create_researcher
        legacy = create_researcher()
        legacy_names = {t.name for t in legacy.tools}

        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.agents import researcher
        # Need a fresh module read since previous import may have
        # cached the dispatch result. Just call again.
        loadable = researcher.create_researcher()
        loadable_names = {t.name for t in loadable.tools}

        # Loadable has the legacy set PLUS the control tools.
        added = loadable_names - legacy_names
        assert added == {"load_tool", "list_available_tools"}, (
            f"Unexpected tool delta: {added}. Loadable should add ONLY "
            "the binder control tools; everything else must match."
        )
        # Verify subset: every legacy tool is in loadable.
        missing = legacy_names - loadable_names
        assert not missing, f"Loadable missing legacy tools: {missing}"

    def test_loadable_has_discoverable_capabilities(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_RESEARCHER", "1")
        from app.agents.researcher import create_researcher
        agent = create_researcher()
        # Should have at least pdf_compose + signal_send_attachment
        # available (registered via Phase 1a).
        catalog = agent.binder.catalog_names()
        assert "pdf_compose" in catalog
        assert "signal_send_attachment" in catalog
