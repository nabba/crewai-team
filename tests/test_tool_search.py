"""Tests for app.tool_registry.discovery + app.tools.tool_search.

Three layers covered:

  * **Subjectless detection** — bare imperative tokens ("ok", "go",
    "execute the plan") return empty when no capabilities are
    provided. Layer 1 of the 4-layer contamination defense.
  * **The 4 hard gates** — quarantine, tier, workspace, distance.
    Each verified independently + composed.
  * **The Weather/Estonia regression** — analogue of the skills-
    retrieval contamination test. A subjectless query after a
    weather conversation must NOT surface a tool whose only signal
    is a weak distance match. The bounded capability vocabulary +
    distance ceiling are the safety net.

Quarantine + ChromaDB index are mocked where realistic; the rest
runs against the real registry.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# ── Subjectless detection (Layer 1) ──────────────────────────────────


class TestSubjectless:

    def test_bare_imperatives_are_subjectless(self):
        from app.tool_registry.discovery import _is_subjectless
        for q in ["", "ok", "go", "do it", "run", "execute the plan",
                  "go now", "please run", "run it now"]:
            assert _is_subjectless(q), f"{q!r} should be subjectless"

    def test_domain_queries_are_not_subjectless(self):
        from app.tool_registry.discovery import _is_subjectless
        for q in ["render a PDF report", "fetch satellite forest data",
                  "send Signal message", "search the web for X",
                  "convert USD to EUR"]:
            assert not _is_subjectless(q), f"{q!r} should have signal"

    def test_long_queries_always_have_signal(self):
        """Long queries can be defensively kept even with filler tokens."""
        from app.tool_registry.discovery import _is_subjectless
        long_q = "ok please run the plan now to do it execute go yes"
        # 10 tokens > 8 → bypass the filler check
        assert not _is_subjectless(long_q)


# ── Discovery API (the 4 hard gates) ─────────────────────────────────


class TestDiscoveryGates:

    def setup_method(self) -> None:
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()

        @register_tool(
            name="g_pdf",
            capabilities=["renders-pdf"],
            description="Production PDF tool description here.",
            tier=Tier.PRODUCTION,
        )
        def f1():
            class _T:
                name = "g_pdf"
            return _T()

        @register_tool(
            name="g_shadow_pdf",
            capabilities=["renders-pdf"],
            description="Shadow PDF tool description here.",
            tier=Tier.SHADOW,
        )
        def f2():
            class _T:
                name = "g_shadow_pdf"
            return _T()

        @register_tool(
            name="g_eesti_only",
            capabilities=["renders-pdf"],
            description="Workspace-pinned PDF tool description here.",
            tier=Tier.PRODUCTION,
            workspace_scope=("eesti-mets",),
        )
        def f3():
            class _T:
                name = "g_eesti_only"
            return _T()

    def test_capability_only_returns_all_three(self):
        """Without gates, capability tag matches everything declaring it."""
        from app.tool_registry import Tier
        from app.tool_registry.discovery import search_tools
        # Test with SHADOW tier so all tiers pass the tier gate.
        matches = search_tools(
            capabilities=["renders-pdf"], agent_tier=Tier.SHADOW,
        )
        names = sorted(m.name for m in matches)
        assert names == ["g_eesti_only", "g_pdf", "g_shadow_pdf"]

    def test_tier_gate_blocks_shadow_for_production_crew(self):
        """Layer 2: a PRODUCTION-tier crew never sees SHADOW tools."""
        from app.tool_registry import Tier
        from app.tool_registry.discovery import search_tools
        matches = search_tools(
            capabilities=["renders-pdf"], agent_tier=Tier.PRODUCTION,
        )
        names = {m.name for m in matches}
        assert "g_shadow_pdf" not in names
        assert "g_pdf" in names

    def test_workspace_gate_filters_pinned_tools(self):
        """Layer 3: workspace-pinned tools don't surface elsewhere."""
        from app.tool_registry.discovery import search_tools
        # In a different workspace, the pinned tool is filtered out.
        matches = search_tools(
            capabilities=["renders-pdf"], workspace="plg",
        )
        names = {m.name for m in matches}
        assert "g_eesti_only" not in names
        assert "g_pdf" in names  # workspace_scope=* — visible everywhere

    def test_workspace_gate_admits_pinned_tool_in_correct_workspace(self):
        from app.tool_registry.discovery import search_tools
        matches = search_tools(
            capabilities=["renders-pdf"], workspace="eesti-mets",
        )
        names = {m.name for m in matches}
        assert "g_eesti_only" in names

    def test_quarantine_filter(self):
        """Layer 1: quarantined tools never surface."""
        from app.tool_registry import Tier
        from app.tool_registry.discovery import search_tools
        # Mock the quarantine list to include g_pdf
        with patch(
            "app.tool_registry.discovery.quarantined_names",
            return_value={"g_pdf"},
        ):
            matches = search_tools(
                capabilities=["renders-pdf"], agent_tier=Tier.SHADOW,
            )
            names = {m.name for m in matches}
            assert "g_pdf" not in names
            assert "g_shadow_pdf" in names  # not quarantined

    def test_subjectless_query_with_no_caps_returns_empty(self):
        """Layer 0: bare imperative + no capabilities = empty list.
        This is the Weather/Estonia regression defense."""
        from app.tool_registry.discovery import search_tools
        assert search_tools(intent="ok") == []
        assert search_tools(intent="go") == []
        assert search_tools(intent="execute the plan") == []
        assert search_tools(intent="") == []

    def test_subjectless_query_with_caps_falls_through(self):
        """Subjectless intent + capability tags → capability path
        still works. The defense protects against hijacking, not
        deliberate capability lookups."""
        from app.tool_registry.discovery import search_tools
        matches = search_tools(intent="go", capabilities=["renders-pdf"])
        assert len(matches) >= 1
        assert all("renders-pdf" in m.spec.capabilities for m in matches)


# ── Capability boost vs semantic match ───────────────────────────────


class TestRanking:

    def setup_method(self) -> None:
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()

        @register_tool(
            name="r_exact_tag",
            capabilities=["renders-pdf"],
            description="A tool that does totally different things.",
            tier=Tier.PRODUCTION,
        )
        def f1():
            class _T:
                name = "r_exact_tag"
            return _T()

    def test_capability_boost_is_applied(self):
        """A tool matching the requested capability tag gets a score
        boost regardless of whether ChromaDB also matches it."""
        from app.tool_registry.discovery import _CAPABILITY_BOOST, search_tools

        # Bypass ChromaDB entirely — just check the capability path.
        with patch(
            "app.tool_registry.discovery.query_index", return_value=[],
        ):
            matches = search_tools(capabilities=["renders-pdf"])

        assert len(matches) == 1
        assert matches[0].name == "r_exact_tag"
        # Score should be boost + tier rank (PRODUCTION = 0.20)
        assert matches[0].score == pytest.approx(_CAPABILITY_BOOST + 0.20)

    def test_distance_ceiling_filters_weak_matches(self):
        """Layer 4: cosine distance > ceiling → reject."""
        from app.tool_registry.discovery import _DISTANCE_CEILING, search_tools

        with patch(
            "app.tool_registry.discovery.query_index",
            return_value=[
                # Above ceiling — should be filtered.
                {"name": "r_exact_tag", "distance": _DISTANCE_CEILING + 0.1, "metadata": {}, "document": ""},
            ],
        ):
            matches = search_tools(intent="something orthogonal")

        # The intent path produced no valid matches; without capabilities
        # there's nothing else to fall back to → empty.
        assert matches == []

    def test_below_ceiling_match_surfaces(self):
        from app.tool_registry.discovery import _DISTANCE_CEILING, search_tools

        with patch(
            "app.tool_registry.discovery.query_index",
            return_value=[
                {"name": "r_exact_tag", "distance": 0.30, "metadata": {}, "document": ""},
            ],
        ):
            matches = search_tools(intent="generate a doc")

        assert len(matches) == 1
        assert matches[0].name == "r_exact_tag"


# ── The Weather/Estonia regression analogue ──────────────────────────


class TestWeatherEstoniaRegression:
    """Mirrors tests/test_skill_retrieval_contamination.py:TestProductionRegression
    but for tools.

    Setup: a stale ``weather_widget`` tool exists in the registry
    (perhaps left over from a Forge experiment). Agent issues a
    subjectless message ("execute the plan") in the middle of an
    Estonia deforestation conversation. The 4-layer defense must NOT
    surface the weather tool.

    Three independent gates protect against this:
      1. Subjectless detection (no caps + bare query → empty).
      2. Capability vocabulary (weather has no relevant tag).
      3. Distance ceiling (weak matches don't surface).
    """

    def setup_method(self) -> None:
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()

        # Stale weather tool — claims `searches-web` (the closest
        # capability), production tier, available everywhere.
        @register_tool(
            name="w_weather_widget",
            capabilities=["searches-web"],
            description="Get the weather forecast for a city or region.",
            tier=Tier.PRODUCTION,
        )
        def f1():
            class _T:
                name = "w_weather_widget"
            return _T()

        # The actually-relevant tool for Estonia work.
        @register_tool(
            name="w_forest_pdf",
            capabilities=["renders-pdf", "renders-chart"],
            description="Render a PDF forest report from Hansen v1.12 data.",
            tier=Tier.PRODUCTION,
        )
        def f2():
            class _T:
                name = "w_forest_pdf"
            return _T()

    def test_subjectless_after_estonia_does_not_surface_weather(self):
        """The headline regression: 'execute the plan' must not
        return the weather tool when no capabilities specified."""
        from app.tool_registry.discovery import search_tools
        matches = search_tools(intent="execute the plan")
        assert matches == [], (
            "Subjectless query surfaced tools without capabilities. "
            "This is the Weather/Estonia hijack vulnerability — "
            "subjectless gate must return empty."
        )

    def test_explicit_capability_request_works(self):
        """Counter-test: the user explicitly asking for 'renders-pdf'
        must still get the forest tool. The defense doesn't disable
        retrieval, it disables UNINTENTIONAL retrieval."""
        from app.tool_registry.discovery import search_tools
        matches = search_tools(capabilities=["renders-pdf"])
        names = {m.name for m in matches}
        assert "w_forest_pdf" in names
        # Weather doesn't declare renders-pdf, so it's not in the result.
        assert "w_weather_widget" not in names

    def test_weak_distance_match_to_weather_blocked(self):
        """Even with a non-subjectless intent, if the only candidate
        is far from the intent (distance > ceiling), it's filtered."""
        from app.tool_registry.discovery import _DISTANCE_CEILING, search_tools

        # Mock ChromaDB to return weather as a weak (above-ceiling) match
        with patch(
            "app.tool_registry.discovery.query_index",
            return_value=[
                {
                    "name": "w_weather_widget",
                    "distance": _DISTANCE_CEILING + 0.05,
                    "metadata": {},
                    "document": "",
                },
            ],
        ):
            matches = search_tools(intent="render a forest deforestation map")
        assert matches == [], (
            "Weather widget surfaced via weak semantic match. "
            "Distance gate must drop matches above the ceiling."
        )


# ── tool_search BaseTool ─────────────────────────────────────────────


class TestToolSearchBaseTool:

    def test_factory_returns_one_tool(self):
        pytest.importorskip("crewai.tools")
        from app.tools.tool_search import create_tool_search_tools
        tools = create_tool_search_tools()
        assert len(tools) == 1
        assert tools[0].name == "tool_search"

    def test_run_with_capability(self):
        pytest.importorskip("crewai.tools")
        from app.tool_registry import (
            Lifecycle, Tier, ToolRegistry, register_tool,
        )
        ToolRegistry.reset_for_tests()

        @register_tool(
            name="ts_pdf",
            capabilities=["renders-pdf"],
            description="A PDF rendering tool description here.",
            tier=Tier.PRODUCTION,
        )
        def f():
            class _T:
                name = "ts_pdf"
            return _T()

        from app.tools.tool_search import create_tool_search_tools
        [t] = create_tool_search_tools()
        out = t._run(capabilities=["renders-pdf"])
        assert "ts_pdf" in out
        assert "Found" in out
        assert "renders-pdf" in out

    def test_run_with_subjectless_returns_helpful_error(self):
        pytest.importorskip("crewai.tools")
        from app.tools.tool_search import create_tool_search_tools
        [t] = create_tool_search_tools()
        out = t._run(intent="ok")
        assert "No matching tools" in out
        assert "capabilities" in out  # tells user how to fix


# ── Quarantine list ──────────────────────────────────────────────────


class TestQuarantine:

    def test_empty_when_file_missing(self, tmp_path, monkeypatch):
        from app.tool_registry import quarantine
        monkeypatch.setattr(
            quarantine, "_QUARANTINE_PATH", tmp_path / "missing.json",
        )
        assert quarantine.quarantined_names() == set()

    def test_loads_entries_from_json(self, tmp_path, monkeypatch):
        from app.tool_registry import quarantine
        path = tmp_path / "quarantine.json"
        path.write_text("""
        {
          "quarantined": [
            {"name": "bad_tool", "reason": "produces wrong output", "since": "2026-05-01"}
          ]
        }
        """)
        monkeypatch.setattr(quarantine, "_QUARANTINE_PATH", path)
        assert "bad_tool" in quarantine.quarantined_names()
        entry = quarantine.quarantine_entry("bad_tool")
        assert entry is not None
        assert entry.reason == "produces wrong output"

    def test_malformed_json_treated_as_empty(self, tmp_path, monkeypatch, caplog):
        from app.tool_registry import quarantine
        path = tmp_path / "broken.json"
        path.write_text("not valid json {{{")
        monkeypatch.setattr(quarantine, "_QUARANTINE_PATH", path)
        # Should NOT raise — operator-side bug shouldn't crash discovery.
        assert quarantine.quarantined_names() == set()
