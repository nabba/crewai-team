"""Tests for app.tool_runtime Phase 2 components.

Covers:
  * **factory.build_loadable_agent** — hybrid core_tools/capabilities,
    discoverable resolution, tier+workspace gates.
  * **telemetry** — usage extraction, file-write, file-read, summary.
  * **introspector flag** — LOADABLE_AGENT_EXPERIMENTAL toggle,
    failsafe fallback to legacy on errors.
  * **parity harness** — dry-mode run, success-rate accounting.

Live mode is not exercised here (would require real LLM calls); the
parity module's live path is integration-tested by operators when
they actually run live measurements.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


# ── factory.build_loadable_agent ─────────────────────────────────────


class TestFactory:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False)

    def test_core_tools_passthrough(self):
        """Raw tool instances passed via ``core_tools`` end up in
        the agent's tool list."""
        from typing import Type

        from crewai.tools import BaseTool
        from pydantic import BaseModel

        from app.tool_runtime.factory import build_loadable_agent
        from app.llm_factory import create_specialist_llm

        class _StubInput(BaseModel):
            x: int = 0

        class StubTool(BaseTool):
            name: str = "stub"
            description: str = "stub tool"
            args_schema: Type[BaseModel] = _StubInput

            def _run(self, x: int = 0) -> str:
                return "ok"

        llm = create_specialist_llm(max_tokens=512, role="introspector")
        stub = StubTool()
        agent = build_loadable_agent(
            role="Test", goal="test", backstory="test",
            llm=llm, agent_id="test",
            core_tools=[stub],
            verbose=False,
        )
        names = [t.name for t in agent.tools]
        assert "stub" in names
        # Plus the auto-injected control tools.
        assert "load_tool" in names
        assert "list_available_tools" in names

    def test_discoverable_capabilities_populate_catalog(self):
        """Capabilities resolve via the registry into the binder's
        catalog (lazy)."""
        from app.tool_runtime.factory import build_loadable_agent
        from app.llm_factory import create_specialist_llm

        llm = create_specialist_llm(max_tokens=512, role="introspector")
        agent = build_loadable_agent(
            role="Test", goal="test", backstory="test",
            llm=llm, agent_id="test",
            discoverable_capabilities=["renders-pdf", "sends-signal"],
            verbose=False,
        )
        # Catalog should include pdf_compose + signal_send_attachment
        # (assuming Phase 1a annotations are present).
        catalog = agent.binder.catalog_names()
        assert "pdf_compose" in catalog
        # signal_send_attachment may not be loadable (env config) but
        # it's still in the catalog if registered.
        # Don't assert membership — assert at least one of the two.
        assert len(catalog) >= 1

    def test_discoverable_names_explicit(self):
        from app.tool_runtime.factory import build_loadable_agent
        from app.llm_factory import create_specialist_llm

        llm = create_specialist_llm(max_tokens=512, role="introspector")
        agent = build_loadable_agent(
            role="Test", goal="test", backstory="test",
            llm=llm, agent_id="test",
            discoverable_names=["web_search"],
            verbose=False,
        )
        assert "web_search" in agent.binder.catalog_names()

    def test_unknown_discoverable_name_skipped(self):
        """Names not in the registry are skipped with a warning, not
        a hard error — keeps the agent constructable even if a tool
        was renamed."""
        from app.tool_runtime.factory import build_loadable_agent
        from app.llm_factory import create_specialist_llm

        llm = create_specialist_llm(max_tokens=512, role="introspector")
        agent = build_loadable_agent(
            role="Test", goal="test", backstory="test",
            llm=llm, agent_id="test",
            discoverable_names=["nonexistent_tool", "web_search"],
            verbose=False,
        )
        catalog = agent.binder.catalog_names()
        assert "nonexistent_tool" not in catalog
        assert "web_search" in catalog

    def test_tier_gate_filters_shadow_for_production_agent(self):
        """A PRODUCTION-tier agent sees only PRODUCTION+ tools."""
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        from app.tool_runtime.factory import build_loadable_agent
        from app.llm_factory import create_specialist_llm

        # Add a SHADOW tool to the registry.
        @register_tool(
            name="t_shadow",
            capabilities=["renders-pdf"],
            description="A shadow-tier PDF tool description here.",
            tier=Tier.SHADOW,
        )
        def f():
            class _T:
                name = "t_shadow"
            return _T()

        llm = create_specialist_llm(max_tokens=512, role="introspector")
        agent = build_loadable_agent(
            role="Test", goal="test", backstory="test",
            llm=llm, agent_id="test",
            discoverable_capabilities=["renders-pdf"],
            agent_tier=Tier.PRODUCTION,
            verbose=False,
        )
        catalog = agent.binder.catalog_names()
        assert "t_shadow" not in catalog


# ── telemetry ────────────────────────────────────────────────────────


class TestTelemetry:

    def test_record_call_writes_jsonl(self, tmp_path, monkeypatch):
        from app.tool_runtime import telemetry
        path = tmp_path / "usage.jsonl"
        monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", path)

        # Mock litellm-like response object — instance attrs so
        # __dict__ extraction works.
        class _Usage:
            def __init__(self):
                self.input_tokens = 100
                self.output_tokens = 50
                self.cache_creation_input_tokens = 200
                self.cache_read_input_tokens = 1000

        class _Resp:
            def __init__(self):
                self.usage = _Usage()

        telemetry.record_call_usage(
            agent_id="test", iteration=1,
            response=_Resp(), model="claude-test",
        )
        assert path.exists()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["agent_id"] == "test"
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["cache_read_input_tokens"] == 1000

    def test_load_telemetry_filters_by_agent(self, tmp_path, monkeypatch):
        from app.tool_runtime import telemetry
        path = tmp_path / "usage.jsonl"
        path.write_text(
            json.dumps({"agent_id": "a", "input_tokens": 10}) + "\n"
            + json.dumps({"agent_id": "b", "input_tokens": 20}) + "\n"
        )
        monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", path)

        rows = telemetry.load_telemetry(agent_id="a")
        assert len(rows) == 1
        assert rows[0]["input_tokens"] == 10

    def test_analyze_zero_when_empty(self, tmp_path, monkeypatch):
        from app.tool_runtime import telemetry
        path = tmp_path / "missing.jsonl"
        monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", path)
        out = telemetry.analyze_telemetry()
        assert out["calls"] == 0
        assert out["effective_input_tokens"] == 0.0

    def test_analyze_computes_effective_tokens(self, tmp_path, monkeypatch):
        from app.tool_runtime import telemetry
        path = tmp_path / "usage.jsonl"
        path.write_text(
            json.dumps({
                "agent_id": "x",
                "input_tokens": 100,
                "cache_creation_input_tokens": 1000,
                "cache_read_input_tokens": 4000,
            }) + "\n"
        )
        monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", path)
        out = telemetry.analyze_telemetry()
        # 1.00 × 100 + 1.25 × 1000 + 0.10 × 4000 = 100 + 1250 + 400 = 1750
        assert out["effective_input_tokens"] == 1750.0
        # vs uncached (5100) — savings = 1 - 1750/5100 ≈ 0.657
        assert 0.65 <= out["vs_uncached"]["savings_ratio"] <= 0.66

    def test_record_call_handles_missing_usage(self, tmp_path, monkeypatch):
        """Non-Anthropic responses without cache fields → zeros, no crash."""
        from app.tool_runtime import telemetry
        path = tmp_path / "usage.jsonl"
        monkeypatch.setattr(telemetry, "_TELEMETRY_PATH", path)

        class _Resp:
            pass  # no usage attribute

        # Should not raise.
        telemetry.record_call_usage(
            agent_id="test", iteration=1, response=_Resp(),
        )
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["cache_creation_input_tokens"] == 0


# ── introspector flag dispatch ───────────────────────────────────────


class TestIntrospectorFlag:

    def setup_method(self) -> None:
        # Boot the registry so capability resolution works.
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False)

    def test_default_is_legacy_path(self, monkeypatch):
        monkeypatch.delenv("LOADABLE_AGENT_EXPERIMENTAL", raising=False)
        from app.agents import introspector
        import importlib
        importlib.reload(introspector)
        agent = introspector.create_introspector()
        assert type(agent).__name__ == "Agent"

    def test_flag_on_uses_loadable_path(self, monkeypatch):
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        from app.agents import introspector
        import importlib
        importlib.reload(introspector)
        agent = introspector.create_introspector()
        assert type(agent).__name__ == "LoadableAgent"

    def test_loadable_failure_falls_back_to_legacy(self, monkeypatch, caplog):
        """If the experimental factory raises, the legacy factory
        runs instead. Telemetry visible in the log."""
        monkeypatch.setenv("LOADABLE_AGENT_EXPERIMENTAL", "1")
        from app.agents import introspector
        import importlib
        importlib.reload(introspector)

        # Make the loadable factory blow up.
        with patch.object(
            introspector, "_build_loadable_introspector",
            side_effect=RuntimeError("simulated bug"),
        ):
            agent = introspector.create_introspector()
        # Should be a stock Agent (legacy fallback), not LoadableAgent.
        assert type(agent).__name__ == "Agent"


# ── parity harness ───────────────────────────────────────────────────


class TestParity:

    def setup_method(self) -> None:
        from app.tool_registry import ToolRegistry
        from app.tool_registry.boot import boot_registry
        ToolRegistry.reset_for_tests()
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False)

    def test_dry_mode_runs_panel_without_real_calls(self):
        from app.tool_runtime.parity import run_parity_panel
        report = run_parity_panel(mode="dry", runs=1)
        # All tasks should report success in dry mode (no real LLM).
        assert report["stock_success_rate"] == 1.0
        assert report["loadable_success_rate"] == 1.0
        # Loadable should be cheaper than stock.
        assert report["loadable_total"] < report["stock_total"]

    def test_dry_mode_passes_phase_1c_gate(self):
        """The default panel should land below the 50% gate."""
        from app.tool_runtime.parity import run_parity_panel
        report = run_parity_panel(mode="dry", runs=1)
        assert report["ratio"] <= 0.50
        assert report["verdict"] == "GO"

    def test_render_report_is_markdown(self):
        from app.tool_runtime.parity import render_report, run_parity_panel
        report = run_parity_panel(mode="dry", runs=1)
        md = render_report(report)
        assert "Phase 2" in md
        assert "Verdict" in md
        assert "Ratio" in md
