"""Tests for app/crews/base_crew.py plugin registry + Agent patch + auto-skill."""
from unittest.mock import MagicMock, patch

import pytest

from tests._v2_shim import install_settings_shim

install_settings_shim()

from app.crews import base_crew  # noqa: E402


def _reset():
    base_crew._tool_plugins.clear()
    base_crew._plugin_tools_cache = None


class TestToolPluginRegistry:
    def setup_method(self):
        _reset()

    def test_register_invalidates_cache(self):
        base_crew._plugin_tools_cache = ["stale"]
        base_crew.register_tool_plugin(lambda: [])
        assert base_crew._plugin_tools_cache is None

    def test_get_plugin_tools_flattens_results(self):
        base_crew.register_tool_plugin(lambda: ["a", "b"])
        base_crew.register_tool_plugin(lambda: ["c"])
        tools = base_crew.get_plugin_tools()
        assert tools == ["a", "b", "c"]

    def test_get_plugin_tools_cached_after_first_call(self):
        call_counter = {"n": 0}

        def factory():
            call_counter["n"] += 1
            return ["x"]

        base_crew.register_tool_plugin(factory)
        base_crew.get_plugin_tools()
        base_crew.get_plugin_tools()
        base_crew.get_plugin_tools()
        assert call_counter["n"] == 1

    def test_factory_failures_are_swallowed(self):
        base_crew.register_tool_plugin(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        base_crew.register_tool_plugin(lambda: ["ok"])
        assert base_crew.get_plugin_tools() == ["ok"]

    def test_empty_factory_return_ignored(self):
        base_crew.register_tool_plugin(lambda: None)
        base_crew.register_tool_plugin(lambda: [])
        base_crew.register_tool_plugin(lambda: ["one"])
        assert base_crew.get_plugin_tools() == ["one"]


class TestEstimateToolCalls:
    def test_observation_markers(self):
        result = """
Thought: I should look this up.
Action: search_web
Action Input: ...
Observation: found something
Thought: Let me check another source.
Action: fetch_url
Observation: got data
Observation: more data
"""
        assert base_crew._estimate_tool_calls(result) == 3

    def test_falls_back_to_action_markers(self):
        result = "Action: foo\nAction: bar\nAction: baz"
        assert base_crew._estimate_tool_calls(result) == 3

    def test_long_result_without_markers_returns_threshold(self):
        result = "x" * 3000
        assert base_crew._estimate_tool_calls(result) == base_crew._SKILL_CREATION_THRESHOLD

    def test_short_result_no_markers_is_zero(self):
        assert base_crew._estimate_tool_calls("brief answer") == 0


class TestAgentPatching:
    def test_patch_sets_global_flag(self):
        base_crew._agent_patched = False

        class FakeAgent:
            def __init__(self, *a, **kw):
                self.tools = []

        import sys, types
        fake_mod = types.ModuleType("crewai_fake")
        fake_mod.Agent = FakeAgent
        monkey_modules = {"crewai": fake_mod}
        original = sys.modules.get("crewai")
        sys.modules["crewai"] = fake_mod
        try:
            base_crew._patch_agent_for_plugins()
            assert base_crew._agent_patched is True
            # Second call is a no-op
            first_init = FakeAgent.__init__
            base_crew._patch_agent_for_plugins()
            # _agent_patched still True; no error
            assert base_crew._agent_patched is True
        finally:
            if original is None:
                sys.modules.pop("crewai", None)
            else:
                sys.modules["crewai"] = original
            base_crew._agent_patched = False

    def test_patched_agent_gets_plugin_tools(self):
        """When the registered plugin returns tools, Agent.__init__ should
        auto-append them."""
        _reset()
        base_crew._agent_patched = False

        class FakeTool:
            def __init__(self, name):
                self.name = name

        class FakeAgent:
            def __init__(self, *, tools=None):
                self.tools = list(tools) if tools else []

        import sys, types
        fake_mod = types.ModuleType("crewai_patch_test")
        fake_mod.Agent = FakeAgent
        original = sys.modules.get("crewai")
        sys.modules["crewai"] = fake_mod
        try:
            base_crew.register_tool_plugin(lambda: [FakeTool("plug1"), FakeTool("plug2")])
            base_crew._patch_agent_for_plugins()
            a = FakeAgent(tools=[FakeTool("builtin")])
            names = {t.name for t in a.tools}
            assert names == {"builtin", "plug1", "plug2"}
        finally:
            if original is None:
                sys.modules.pop("crewai", None)
            else:
                sys.modules["crewai"] = original
            base_crew._agent_patched = False
            _reset()

    def test_register_default_plugins_registers_core_sources(self):
        _reset()
        with patch.object(base_crew, "_patch_agent_for_plugins"):
            base_crew._register_default_plugins()
        # At minimum MCP adapter + browser + session_search. Extras (mcp_manager,
        # photos, etc.) may be added later but the core three must be present.
        assert len(base_crew._tool_plugins) >= 3


class TestAutoCreateSkill:
    def test_auto_skill_skips_empty_llm_output(self, monkeypatch):
        # Returns a short string — should be rejected
        llm = MagicMock()
        llm.call = MagicMock(return_value="too short")

        monkeypatch.setattr(
            "app.llm_factory.create_specialist_llm",
            lambda **kw: llm,
        )

        integrate_calls = []

        def fake_integrate(draft, **kw):
            integrate_calls.append(draft)
            return None

        monkeypatch.setattr("app.self_improvement.integrator.integrate",
                            fake_integrate)
        base_crew._auto_create_skill("coding", "task", "result", 6)
        assert integrate_calls == []  # short text rejected

    def test_auto_skill_integrates_long_output_when_vetting_passed(self, monkeypatch):
        # 2026-05-02 audit (H6): _auto_create_skill is now vetting-gated.
        # Set the outcome to True to mirror the orchestrator's behaviour
        # after a successful crew run.
        from app.crews.events import set_vetting_outcome
        set_vetting_outcome("coding", True)

        llm = MagicMock()
        llm.call = MagicMock(return_value=(
            "Topic: How to use the caching layer\n"
            "When to use: every time\n"
            "Procedure:\n1. Initialize\n2. Cache the data\n3. Evict when stale\n"
            "Pitfalls: race conditions in the evict step"
        ))

        monkeypatch.setattr(
            "app.llm_factory.create_specialist_llm",
            lambda **kw: llm,
        )

        drafts = []

        def fake_integrate(draft, **kw):
            drafts.append(draft)
            return MagicMock()

        monkeypatch.setattr("app.self_improvement.integrator.integrate",
                            fake_integrate)

        base_crew._auto_create_skill("coding", "build a caching layer", "result", 6)
        assert len(drafts) == 1
        draft = drafts[0]
        assert draft.topic.startswith("How to use the caching layer")
        assert draft.proposed_kb == "experiential"
        assert "coding" in draft.rationale

    def test_auto_skill_dropped_when_vetting_failed(self, monkeypatch):
        # H6: failed-vetting outcomes must not produce skills.
        from app.crews.events import set_vetting_outcome
        set_vetting_outcome("coding", False)

        llm = MagicMock()
        llm.call = MagicMock(return_value=(
            "Topic: How to use the caching layer\n"
            "When to use: every time\n"
            "Procedure:\n1. Initialize\n2. Cache\n3. Evict\n"
            "Pitfalls: races"
        ))
        monkeypatch.setattr(
            "app.llm_factory.create_specialist_llm",
            lambda **kw: llm,
        )

        drafts = []
        monkeypatch.setattr(
            "app.self_improvement.integrator.integrate",
            lambda draft, **kw: drafts.append(draft) or MagicMock(),
        )

        base_crew._auto_create_skill("coding", "build a caching layer", "result", 6)
        assert drafts == []  # vetting failed → no skill written

    def test_auto_skill_dropped_when_vetting_unknown(self, monkeypatch):
        # H6: conservative default — refusing to act on uncertainty.
        # Use a unique crew name to avoid carryover from other tests.
        llm = MagicMock()
        llm.call = MagicMock(return_value=(
            "Topic: Unique skill\n"
            "When to use: never\n"
            "Procedure:\n1. Step\n2. Step\n3. Step\n"
            "Pitfalls: none"
        ))
        monkeypatch.setattr(
            "app.llm_factory.create_specialist_llm",
            lambda **kw: llm,
        )

        drafts = []
        monkeypatch.setattr(
            "app.self_improvement.integrator.integrate",
            lambda draft, **kw: drafts.append(draft) or MagicMock(),
        )

        # Crew name that no other test set an outcome for
        base_crew._auto_create_skill("never_vetted_crew_xyz", "t", "r", 6)
        assert drafts == []  # unknown outcome → drop

    def test_auto_skill_silent_on_exception(self, monkeypatch):
        def boom(**kw):
            raise RuntimeError("no LLM")

        monkeypatch.setattr("app.llm_factory.create_specialist_llm", boom)
        # Should not raise
        base_crew._auto_create_skill("coding", "t", "r", 6)

    def test_excluded_crews_never_trigger(self):
        assert "self_improvement" in base_crew._SKILL_EXCLUDED_CREWS
        assert "retrospective" in base_crew._SKILL_EXCLUDED_CREWS
        assert "critic" in base_crew._SKILL_EXCLUDED_CREWS
