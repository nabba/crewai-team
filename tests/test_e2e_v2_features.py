"""End-to-end tests for v2 spec features.

These exercise multiple subsystems together to catch integration bugs:
  - Conversation flow: user message → middleware → FTS5 search → session tool
  - MCP pipeline: registry → client → tool adapter → CrewAI tool
  - NL cron flow: parse → cron trigger → job registration → persistence roundtrip
  - Skill flow: integrate draft → persist record → search → filter by context
"""
import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests._v2_shim import install_settings_shim

install_settings_shim()


# ═════════════════════════════════════════════════════════════════════════════
# E2E: conversation → compression middleware → FTS5 → session search tool
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_ConversationFlow:

    def test_user_message_flows_through_middleware_fts_and_search_tool(
        self, tmp_path, monkeypatch
    ):
        # 1. Isolated DB
        import app.conversation_store as cs
        monkeypatch.setattr(cs, "DB_PATH", tmp_path / "conv.db")
        if hasattr(cs._local, "conn"):
            cs._local.conn = None

        # Fresh history store
        import app.history_compression as hc
        hc._histories.clear()

        # 2. Wrap a dummy commander with CompressionMiddleware
        from app.history_compression import CompressionMiddleware

        class DummyCommander:
            last_crew_used = "researcher"
            def handle(self, text, sender, attachments):
                # Simulate the normal main.py add_message calls
                cs.add_message(sender, "user", text)
                cs.add_message(sender, "assistant",
                               "Helsinki sits near 60°N; daylight varies by season.")
                return "Helsinki sits near 60°N; daylight varies by season."

        mw = CompressionMiddleware(DummyCommander())

        # 3. User sends a message
        sender = "+15558887777"
        result = mw.handle("What is the sunrise time in Helsinki today?",
                          sender, [])
        assert "Helsinki" in result

        # 4. Compressed history tracked the exchange
        from app.security import _sender_hash
        h = hc.get_history(_sender_hash(sender))
        roles = [m.role for m in h.current.messages]
        assert "user" in roles
        assert "assistant" in roles

        # 5. FTS5 can find the exchange by keyword
        hits = cs.search_messages("helsinki")
        if not hits:
            pytest.skip("FTS5 not available in this SQLite build")
        assert len(hits) >= 1
        assert any(">>>" in h["content_snippet"] for h in hits)

        # 6. The session_search_tool surfaces these results to an agent
        pytest.importorskip("crewai")
        from app.tools.session_search_tool import create_session_search_tools
        tools = create_session_search_tools()
        assert len(tools) == 1
        out = tools[0]._run(query="helsinki", limit=5)
        assert "matches" in out
        assert "Helsinki" in out or "helsinki" in out.lower()


# ═════════════════════════════════════════════════════════════════════════════
# E2E: MCP pipeline (registry → client → tool adapter → CrewAI tool run)
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_MCPPipeline:

    def test_registry_to_tool_adapter_round_trip(self, monkeypatch):
        """Simulate an MCP server with 2 tools and verify CrewAI sees them."""
        pytest.importorskip("crewai")

        from app.mcp import client, registry

        # 1. Build a fake transport
        class FakeTransport:
            def __init__(self):
                self.is_alive = True
                self._responses = [
                    # initialize
                    {"jsonrpc": "2.0", "id": 1, "result": {}},
                    # tools/list
                    {"jsonrpc": "2.0", "id": 2, "result": {"tools": [
                        {"name": "read_file",
                         "description": "Read a file from disk",
                         "inputSchema": {"type": "object",
                                         "properties": {"path": {"type": "string",
                                                                 "description": "File path"}},
                                         "required": ["path"]}},
                        {"name": "list_dir",
                         "description": "List directory contents",
                         "inputSchema": {"type": "object",
                                         "properties": {"path": {"type": "string"}}}},
                    ]}},
                    # tools/call
                    {"jsonrpc": "2.0", "id": 3, "result": {
                        "content": [{"type": "text", "text": "hello from file"}]
                    }},
                ]
                self._i = 0
                self.notifications = []
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def send_receive(self, msg):
                r = self._responses[self._i]
                self._i += 1
                return r

            def send_notification(self, msg):
                self.notifications.append(msg)

            def stop(self):
                self.stopped = True

        fake_transport = FakeTransport()

        # 2. Plug into MCPClient
        cfg = client.MCPServerConfig(name="filesystem", command="/bin/true")
        c = client.MCPClient(cfg)
        c._transport = fake_transport
        assert c.connect() is True
        assert len(c.tools) == 2

        # 3. Register in the module-level registry
        registry._clients.clear()
        registry._clients["filesystem"] = c

        # 4. Adapt to CrewAI
        from app.mcp.tool_adapter import create_crewai_tools
        tools = create_crewai_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "mcp_filesystem_read_file" in names
        assert "mcp_filesystem_list_dir" in names

        # 5. Calling the CrewAI tool → round-trips to the registry call_tool
        read_tool = next(t for t in tools if "read_file" in t.name)
        out = read_tool._run(path="/tmp/example.txt")
        assert out == "hello from file"

        # 6. format_status reports the server
        status = registry.format_status()
        assert "filesystem" in status
        assert "2 tools" in status

        # 7. Clean shutdown
        registry.disconnect_all()
        assert fake_transport.stopped is True
        assert registry._clients == {}


# ═════════════════════════════════════════════════════════════════════════════
# E2E: NL cron — parse → persist → restore
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_NlCronFlow:

    def test_schedule_persist_restore_roundtrip(self, monkeypatch, tmp_path):
        from app.agents.commander import commands as cmd_mod

        # Isolated persistence file
        monkeypatch.setattr(cmd_mod, "_NL_JOBS_FILE", tmp_path / "nl_jobs.json")

        # Capture jobs added to the scheduler
        class FakeSched:
            def __init__(self):
                self.added = []
            def add_job(self, func, trigger, **kw):
                self.added.append({"id": kw.get("id"), "func": func,
                                   "trigger": trigger, "name": kw.get("name")})

        first_sched = FakeSched()
        fake_main = MagicMock()
        fake_main.scheduler = first_sched
        monkeypatch.setitem(__import__("sys").modules, "app.main", fake_main)

        class DummyCommander:
            last_crew_used = "researcher"
            def handle(self, text, sender, attachments):
                return f"ran: {text}"

        cmd = DummyCommander()

        # 1. Issue "schedule <task> <when>" command
        out = cmd_mod.try_command(
            "schedule send the news weekdays at 7am",
            "+15551112222",
            cmd,
        )
        assert out is not None
        assert "Scheduled" in out
        assert len(first_sched.added) == 1

        job_id = first_sched.added[0]["id"]
        assert job_id.startswith("nl_")

        # 2. Job was persisted to disk
        persisted = cmd_mod._read_nl_jobs()
        assert job_id in persisted
        assert persisted[job_id]["task"] == "send the news"
        assert persisted[job_id]["cron"] == "0 7 * * 1-5"

        # 3. Simulate a restart: new scheduler, use restore_nl_jobs()
        second_sched = FakeSched()
        restored = cmd_mod.restore_nl_jobs(second_sched, cmd)
        assert restored == 1
        assert len(second_sched.added) == 1
        assert second_sched.added[0]["id"] == job_id

        # 4. Cancel removes from both memory AND disk
        cancel_sched = first_sched
        cancel_sched.remove_job = lambda jid: None  # stub
        fake_main.scheduler = cancel_sched
        out = cmd_mod.try_command(f"cancel {job_id}",
                                  "+15551112222", cmd)
        assert "Cancelled" in out
        assert job_id not in cmd_mod._read_nl_jobs()


# ═════════════════════════════════════════════════════════════════════════════
# E2E: prompt caching — hook applies to real message shape
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_PromptCacheHook:

    def test_litellm_monkey_patch_installs_and_injects(self, monkeypatch):
        from app import prompt_cache_hook

        fake_litellm = MagicMock()
        calls = []

        def original(**kwargs):
            calls.append(kwargs)
            return "ok"

        fake_litellm.completion = original
        if hasattr(fake_litellm, "acompletion"):
            delattr(fake_litellm, "acompletion")
        monkeypatch.setitem(__import__("sys").modules, "litellm", fake_litellm)

        # Simulate startup
        prompt_cache_hook._installed = False
        prompt_cache_hook.install_cache_hook()

        # Simulate an agent call with a realistic-ish Claude request
        long_system = ("You are Claude, an AI assistant. " * 300).strip()
        fake_litellm.completion(
            model="anthropic/claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": long_system},
                {"role": "user", "content": "Hi there"},
            ],
            max_tokens=1024,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )

        assert len(calls) == 1
        sys_msg = calls[0]["messages"][0]
        assert isinstance(sys_msg["content"], list), \
            "system message should have been converted to block form"
        assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        # beta header preserved
        assert calls[0]["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"


# ═════════════════════════════════════════════════════════════════════════════
# E2E: SkillRecord conditional activation — search + filter
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_SkillConditionalFlow:

    def test_filter_in_commander_context(self, monkeypatch):
        """_load_relevant_skills should drop skills that fail matches_context."""
        from app.agents.commander import context as ctx_mod
        from app.self_improvement.types import SkillRecord

        # Fake search_skills returns two skills — one matches, one doesn't
        skills = [
            SkillRecord(id="s1", topic="Local Ollama tips",
                        content_markdown="When local mode is active, start Ollama first.",
                        kb="experiential", requires_mode="local"),
            SkillRecord(id="s2", topic="Cloud-only hack",
                        content_markdown="Only works with cloud providers.",
                        kb="experiential", requires_mode="cloud"),
            SkillRecord(id="s3", topic="Universal skill",
                        content_markdown="Always applies.",
                        kb="experiential"),
        ]
        # Loader uses search_skills_scored (May 2026 contamination fix).
        # Patch both the new and the legacy entry point so the test pins
        # the conditional-activation behaviour regardless of which API
        # the loader reaches for. Distance 0.10 keeps every record under
        # the loader's _SKILL_DISTANCE_CEILING gate.
        scored = [(s, 0.10) for s in skills]
        monkeypatch.setattr("app.self_improvement.integrator.search_skills_scored",
                            lambda _task, n=6: scored)
        monkeypatch.setattr("app.self_improvement.integrator.search_skills",
                            lambda _task, n=6: skills)
        monkeypatch.setattr("app.llm_mode.get_mode", lambda: "local")

        out = ctx_mod._load_relevant_skills("anything", n=5)
        # s2 must not appear; s1 and s3 should
        assert "Local Ollama tips" in out
        assert "Universal skill" in out
        assert "Cloud-only hack" not in out


# ═════════════════════════════════════════════════════════════════════════════
# E2E: plugin registry + Agent patch — tools reach every Agent
# ═════════════════════════════════════════════════════════════════════════════

class TestE2E_PluginRegistryPatchesAgent:
    """Verify pre-init plugin injection extends kwargs['tools'] for Agent().

    Uses a stand-in Agent class to avoid crewai's strict LLM/role validators
    in unit-test environments.
    """

    def test_patched_agent_gets_plugin_tools(self, monkeypatch):
        from app.crews import base_crew

        # Reset state
        base_crew._tool_plugins.clear()
        base_crew._plugin_tools_cache = None
        base_crew._agent_patched = False

        class FakeTool:
            def __init__(self, name):
                self.name = name

        class FakeAgent:
            def __init__(self, *, role=None, tools=None, **kw):
                self.role = role
                self.tools = list(tools) if tools else []

        # Substitute a FakeAgent in the "crewai" module before applying the patch
        import sys, types
        fake_mod = types.ModuleType("crewai_test_module")
        fake_mod.Agent = FakeAgent
        original = sys.modules.get("crewai")
        sys.modules["crewai"] = fake_mod
        try:
            base_crew.register_tool_plugin(lambda: [FakeTool("plugin_a"), FakeTool("plugin_b")])
            base_crew._patch_agent_for_plugins()

            agent = FakeAgent(role="Tester", tools=[FakeTool("builtin")])
            names = {t.name for t in agent.tools}
            assert "builtin" in names
            assert "plugin_a" in names
            assert "plugin_b" in names
        finally:
            if original is None:
                sys.modules.pop("crewai", None)
            else:
                sys.modules["crewai"] = original
            base_crew._agent_patched = False

    def test_patch_deduplicates_by_name(self, monkeypatch):
        from app.crews import base_crew

        base_crew._tool_plugins.clear()
        base_crew._plugin_tools_cache = None
        base_crew._agent_patched = False

        class FakeTool:
            def __init__(self, name):
                self.name = name

        class FakeAgent:
            def __init__(self, *, role=None, tools=None, **kw):
                self.role = role
                self.tools = list(tools) if tools else []

        import sys, types
        fake_mod = types.ModuleType("crewai_test_dedupe")
        fake_mod.Agent = FakeAgent
        original = sys.modules.get("crewai")
        sys.modules["crewai"] = fake_mod
        try:
            base_crew.register_tool_plugin(lambda: [FakeTool("shared_name")])
            base_crew._patch_agent_for_plugins()

            agent = FakeAgent(role="T", tools=[FakeTool("shared_name")])
            names = [t.name for t in agent.tools]
            assert names.count("shared_name") == 1
        finally:
            if original is None:
                sys.modules.pop("crewai", None)
            else:
                sys.modules["crewai"] = original
            base_crew._agent_patched = False

    def test_patch_preserves_agent_without_tools_kwarg(self):
        """An Agent created with no `tools=` still gets plugin tools injected."""
        from app.crews import base_crew

        base_crew._tool_plugins.clear()
        base_crew._plugin_tools_cache = None
        base_crew._agent_patched = False

        class FakeTool:
            def __init__(self, name):
                self.name = name

        class FakeAgent:
            def __init__(self, *, role=None, tools=None, **kw):
                self.tools = list(tools) if tools else []

        import sys, types
        fake_mod = types.ModuleType("crewai_test_notools")
        fake_mod.Agent = FakeAgent
        original = sys.modules.get("crewai")
        sys.modules["crewai"] = fake_mod
        try:
            base_crew.register_tool_plugin(lambda: [FakeTool("only_plugin")])
            base_crew._patch_agent_for_plugins()

            agent = FakeAgent(role="T")  # no tools=
            assert len(agent.tools) == 1
            assert agent.tools[0].name == "only_plugin"
        finally:
            if original is None:
                sys.modules.pop("crewai", None)
            else:
                sys.modules["crewai"] = original
            base_crew._agent_patched = False
