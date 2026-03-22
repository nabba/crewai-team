"""
Integration tests for the self-aware, proactively cooperative AI agent system.

Tests all four phases:
  Phase 1: Functional self-awareness (self-models, self-report, reflection)
  Phase 2: Shared memory + cooperation (scoped memory, belief states, critic)
  Phase 3: Proactive cooperation (trigger scanner, proactive behaviors)
  Phase 4: Meta-cognitive loop (policies, benchmarks, retrospective)

These tests run WITHOUT live LLM/ChromaDB by stubbing heavy dependencies,
matching the pattern in test_security.py.
"""

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# ── Stub heavy dependencies ─────────────────────────────────────────────────
_STUBS = [
    "crewai", "crewai.tools", "langchain_anthropic", "docker",
    "chromadb", "sentence_transformers", "trafilatura",
    "youtube_transcript_api", "brave_search", "apscheduler",
    "apscheduler.schedulers.asyncio", "apscheduler.triggers.cron",
    "fastapi", "fastapi.middleware.cors", "uvicorn",
    "firebase_admin", "firebase_admin.credentials", "firebase_admin.firestore",
    "pypdf", "docx", "openpyxl", "PIL",
    "litellm", "bs4",
]

for mod in _STUBS:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)
    m = sys.modules[mod]
    # Ensure required attributes exist on each stub
    if mod == "crewai":
        if not hasattr(m, "Agent") or not callable(m.Agent):
            m.Agent = MagicMock
        if not hasattr(m, "Task"):
            m.Task = MagicMock
        if not hasattr(m, "Crew"):
            m.Crew = MagicMock
        if not hasattr(m, "LLM"):
            m.LLM = MagicMock
        if not hasattr(m, "Process"):
            m.Process = MagicMock()
            m.Process.sequential = "sequential"
    if mod == "crewai.tools":
        if not hasattr(m, "BaseTool") or not isinstance(m.BaseTool, type):
            class _BaseTool:
                name: str = ""
                description: str = ""
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                def _run(self, *a, **kw):
                    return ""
            m.BaseTool = _BaseTool
        if not hasattr(m, "tool"):
            m.tool = lambda name: (lambda fn: fn)
    if mod == "sentence_transformers":
        if not hasattr(m, "SentenceTransformer") or not callable(getattr(m, "SentenceTransformer", None)):
            m.SentenceTransformer = MagicMock
    if mod == "chromadb":
        if not hasattr(m, "PersistentClient"):
            m.PersistentClient = MagicMock

# Stub pydantic if not installed
for _pmod in ["pydantic", "pydantic.functional_validators", "pydantic_settings"]:
    if _pmod not in sys.modules:
        m = types.ModuleType(_pmod)
        if _pmod == "pydantic":
            m.SecretStr = str
            m.Field = lambda **kw: kw.get("default")
            m.field_validator = lambda *a, **kw: (lambda fn: fn)
        if _pmod == "pydantic_settings":
            class _BS:
                class Config:
                    env_file = ".env"
            m.BaseSettings = _BS
        sys.modules[_pmod] = m

# Set required env vars for config
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-key")
os.environ.setdefault("BRAVE_API_KEY", "brave-test-key")
os.environ.setdefault("SIGNAL_BOT_NUMBER", "+1234567890")
os.environ.setdefault("SIGNAL_OWNER_NUMBER", "+0987654321")
os.environ.setdefault("GATEWAY_SECRET", "test-secret")

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Pre-import modules that we'll patch so they resolve correctly
import app.memory.chromadb_manager  # noqa: E402
import app.memory.scoped_memory  # noqa: E402
import app.memory.belief_state  # noqa: E402
import app.tools.self_report_tool  # noqa: E402
import app.tools.reflection_tool  # noqa: E402
import app.tools.scoped_memory_tool  # noqa: E402
import app.proactive.trigger_scanner  # noqa: E402
import app.policies.policy_loader  # noqa: E402
import app.crews.retrospective_crew  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Functional Self-Awareness
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelfModel(unittest.TestCase):
    """Test structured self-model definitions."""

    def test_get_self_model_known_role(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("researcher")
        self.assertIn("capabilities", model)
        self.assertIn("limitations", model)
        self.assertIn("operating_principles", model)
        self.assertIn("tools_available", model)
        self.assertIn("typical_failure_modes", model)
        self.assertTrue(len(model["capabilities"]) > 0)

    def test_get_self_model_unknown_role(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("nonexistent")
        self.assertEqual(model["capabilities"], [])
        self.assertEqual(model["limitations"], [])

    def test_format_self_model_block_contains_sections(self):
        from app.self_awareness.self_model import format_self_model_block
        block = format_self_model_block("researcher")
        self.assertIn("## Self-Model", block)
        self.assertIn("### My Capabilities", block)
        self.assertIn("### My Limitations", block)
        self.assertIn("### My Operating Principles", block)
        self.assertIn("### My Known Failure Modes", block)

    def test_format_self_model_block_empty_for_unknown(self):
        from app.self_awareness.self_model import format_self_model_block
        block = format_self_model_block("nonexistent")
        self.assertEqual(block, "")

    def test_all_roles_defined(self):
        from app.self_awareness.self_model import SELF_MODELS
        expected_roles = {"researcher", "coder", "writer", "commander", "critic", "introspector", "self_improver", "media_analyst"}
        self.assertEqual(set(SELF_MODELS.keys()), expected_roles)

    def test_critic_model_has_review_capabilities(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("critic")
        caps_text = " ".join(model["capabilities"]).lower()
        self.assertIn("review", caps_text)

    def test_introspector_model_has_policy_capabilities(self):
        from app.self_awareness.self_model import get_self_model
        model = get_self_model("introspector")
        caps_text = " ".join(model["capabilities"]).lower()
        self.assertIn("polic", caps_text)


class TestSelfReportTool(unittest.TestCase):
    """Test the SelfReportTool."""

    @patch("app.tools.self_report_tool.store")
    def test_run_returns_json(self, mock_store):
        from app.tools.self_report_tool import SelfReportTool
        tool = SelfReportTool(agent_role="researcher")
        result = tool._run(
            task_summary="Researched AI trends",
            confidence="high",
            completeness="complete",
        )
        self.assertIn("researcher", result)
        self.assertIn("high", result)
        self.assertIn("complete", result)
        # Verify it stored in ChromaDB
        mock_store.assert_called_once()
        args = mock_store.call_args
        self.assertEqual(args[0][0], "self_reports")  # collection name

    @patch("app.tools.self_report_tool.store")
    def test_normalizes_invalid_confidence(self, mock_store):
        from app.tools.self_report_tool import SelfReportTool
        tool = SelfReportTool(agent_role="coder")
        result = tool._run(
            task_summary="Wrote code",
            confidence="INVALID",
            completeness="done",
        )
        parsed = json.loads(result.replace("Self-report recorded: ", ""))
        self.assertEqual(parsed["confidence"], "medium")
        self.assertEqual(parsed["completeness"], "partial")

    @patch("app.tools.self_report_tool.store")
    def test_includes_blockers_and_needs(self, mock_store):
        from app.tools.self_report_tool import SelfReportTool
        tool = SelfReportTool(agent_role="researcher")
        result = tool._run(
            task_summary="Research task",
            confidence="low",
            completeness="partial",
            blockers="Cannot access paywalled content",
            needs_from_team="Need coder to scrape data",
        )
        self.assertIn("paywalled", result)
        self.assertIn("coder", result)

    @patch("app.tools.self_report_tool.store")
    def test_factory_function(self, mock_store):
        from app.tools.self_report_tool import create_self_report_tool
        tool = create_self_report_tool("writer")
        self.assertEqual(tool.agent_role, "writer")


class TestReflectionTool(unittest.TestCase):
    """Test the ReflectionTool."""

    @patch("app.tools.reflection_tool.store_team")
    @patch("app.tools.reflection_tool.store")
    def test_stores_in_both_collections(self, mock_store, mock_team):
        from app.tools.reflection_tool import ReflectionTool
        tool = ReflectionTool(agent_role="coder")
        result = tool._run(
            task_description="Built a REST API",
            what_went_well="Clean code structure",
            what_went_wrong="Missed edge cases",
            lesson_learned="Always test edge cases first",
        )
        # Should store in agent-specific collection
        mock_store.assert_called_once()
        self.assertEqual(mock_store.call_args[0][0], "reflections_coder")
        # Should also store in team memory
        mock_team.assert_called_once()
        # Result should confirm storage
        self.assertIn("Reflection stored", result)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Shared Memory + Cooperation
# ═══════════════════════════════════════════════════════════════════════════════

class TestScopedMemory(unittest.TestCase):
    """Test hierarchical scoped memory."""

    @patch("app.memory.scoped_memory.store")
    def test_store_scoped_adds_metadata(self, mock_store):
        from app.memory.scoped_memory import store_scoped
        store_scoped("scope_team", "Test decision", importance="high")
        mock_store.assert_called_once()
        args = mock_store.call_args
        self.assertEqual(args[0][0], "scope_team")
        meta = args[0][2]
        self.assertEqual(meta["importance"], "high")
        self.assertIn("ts", meta)

    @patch("app.memory.scoped_memory.store")
    def test_store_team_decision(self, mock_store):
        from app.memory.scoped_memory import store_team_decision
        store_team_decision("Use approach A", importance="critical")
        args = mock_store.call_args
        meta = args[0][2]
        self.assertEqual(meta["importance"], "critical")
        self.assertEqual(meta["type"], "decision")

    @patch("app.memory.scoped_memory.store")
    def test_store_agent_memory(self, mock_store):
        from app.memory.scoped_memory import store_agent_memory
        store_agent_memory("researcher", "Working notes")
        args = mock_store.call_args
        self.assertEqual(args[0][0], "scope_agent_researcher")

    @patch("app.memory.scoped_memory.retrieve_with_metadata")
    def test_retrieve_operational_recency_boost(self, mock_retrieve):
        from app.memory.scoped_memory import retrieve_operational
        now = datetime.now(timezone.utc)
        recent_ts = now.isoformat()
        old_ts = (now - timedelta(days=7)).isoformat()

        mock_retrieve.return_value = [
            {"document": "old doc", "metadata": {"ts": old_ts}, "distance": 0.1},
            {"document": "recent doc", "metadata": {"ts": recent_ts}, "distance": 0.3},
        ]
        results = retrieve_operational("scope_team", "test", n=2)
        # Recent doc should be boosted despite higher distance
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "recent doc")

    @patch("app.memory.scoped_memory.retrieve_filtered")
    def test_retrieve_strategic_filters_by_importance(self, mock_filtered):
        from app.memory.scoped_memory import retrieve_strategic
        mock_filtered.return_value = ["important policy"]
        results = retrieve_strategic("scope_policies", "test")
        self.assertEqual(results, ["important policy"])
        mock_filtered.assert_called_once()


class TestBeliefState(unittest.TestCase):
    """Test ProAgent-style belief state tracking."""

    @patch("app.memory.belief_state.store")
    def test_update_belief(self, mock_store):
        from app.memory.belief_state import update_belief
        update_belief(
            agent_name="researcher",
            state="working",
            current_task="Researching AI",
            confidence="high",
            needs=["data from coder"],
        )
        mock_store.assert_called_once()
        args = mock_store.call_args
        self.assertEqual(args[0][0], "scope_beliefs")
        stored_doc = json.loads(args[0][1])
        self.assertEqual(stored_doc["agent"], "researcher")
        self.assertEqual(stored_doc["state"], "working")
        self.assertEqual(stored_doc["needs"], ["data from coder"])

    @patch("app.memory.belief_state.retrieve_with_metadata")
    def test_get_beliefs(self, mock_retrieve):
        from app.memory.belief_state import get_beliefs
        mock_retrieve.return_value = [
            {
                "document": json.dumps({
                    "agent": "researcher",
                    "state": "working",
                    "current_task": "AI research",
                    "confidence": "high",
                }),
                "metadata": {"agent": "researcher"},
                "distance": 0.1,
            },
        ]
        beliefs = get_beliefs("researcher")
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0]["agent"], "researcher")
        self.assertEqual(beliefs[0]["state"], "working")

    @patch("app.memory.belief_state.retrieve_with_metadata")
    def test_get_team_state_summary(self, mock_retrieve):
        from app.memory.belief_state import get_team_state_summary
        mock_retrieve.return_value = [
            {
                "document": json.dumps({
                    "agent": "researcher",
                    "state": "completed",
                    "current_task": "Done researching",
                    "confidence": "high",
                    "needs": [],
                }),
                "metadata": {},
                "distance": 0.1,
            },
        ]
        summary = get_team_state_summary()
        self.assertIn("TEAM STATE:", summary)
        self.assertIn("researcher", summary)
        self.assertIn("completed", summary)

    @patch("app.memory.belief_state.retrieve_with_metadata")
    def test_empty_state_returns_empty(self, mock_retrieve):
        from app.memory.belief_state import get_team_state_summary
        mock_retrieve.return_value = []
        summary = get_team_state_summary()
        self.assertEqual(summary, "")


class TestChromaDBExtensions(unittest.TestCase):
    """Test the new ChromaDB manager functions."""

    @patch("app.memory.chromadb_manager.get_client")
    @patch("app.memory.chromadb_manager._model")
    def test_retrieve_with_metadata(self, mock_model, mock_client):
        from app.memory.chromadb_manager import retrieve_with_metadata
        mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2])
        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"role": "researcher"}, {"role": "coder"}]],
            "distances": [[0.1, 0.5]],
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_col

        results = retrieve_with_metadata("test_col", "query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["document"], "doc1")
        self.assertEqual(results[0]["metadata"]["role"], "researcher")
        self.assertEqual(results[0]["distance"], 0.1)

    @patch("app.memory.chromadb_manager.get_client")
    @patch("app.memory.chromadb_manager._model")
    def test_retrieve_with_metadata_empty(self, mock_model, mock_client):
        from app.memory.chromadb_manager import retrieve_with_metadata
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_client.return_value.get_or_create_collection.return_value = mock_col
        results = retrieve_with_metadata("empty_col", "query")
        self.assertEqual(results, [])


class TestScopedMemoryTools(unittest.TestCase):
    """Test the scoped memory tool wrappers."""

    def test_factory_creates_five_tools(self):
        from app.tools.scoped_memory_tool import create_scoped_memory_tools
        tools = create_scoped_memory_tools("researcher")
        self.assertEqual(len(tools), 5)
        names = {t.name for t in tools}
        self.assertIn("scoped_memory_store", names)
        self.assertIn("scoped_memory_retrieve", names)
        self.assertIn("team_decision", names)
        self.assertIn("update_team_belief", names)
        self.assertIn("team_state", names)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Proactive Cooperation + Reasoning
# ═══════════════════════════════════════════════════════════════════════════════

class TestProactiveBehaviors(unittest.TestCase):
    """Test proactive behavior definitions."""

    def test_all_roles_defined(self):
        from app.proactive.proactive_behaviors import PROACTIVE_BEHAVIORS
        self.assertIn("commander", PROACTIVE_BEHAVIORS)
        self.assertIn("researcher", PROACTIVE_BEHAVIORS)
        self.assertIn("critic", PROACTIVE_BEHAVIORS)
        self.assertIn("coder", PROACTIVE_BEHAVIORS)

    def test_get_proactive_prompt(self):
        from app.proactive.proactive_behaviors import get_proactive_prompt
        prompt = get_proactive_prompt("researcher", "low_confidence")
        self.assertIn("PROACTIVE ACTION", prompt)
        self.assertIn("low_confidence", prompt)

    def test_get_proactive_prompt_unknown_returns_empty(self):
        from app.proactive.proactive_behaviors import get_proactive_prompt
        prompt = get_proactive_prompt("nonexistent", "unknown")
        self.assertEqual(prompt, "")


class TestTriggerScanner(unittest.TestCase):
    """Test proactive trigger detection."""

    @patch("app.proactive.trigger_scanner.retrieve_with_metadata")
    def test_low_confidence_trigger(self, mock_retrieve):
        from app.proactive.trigger_scanner import _check_low_confidence
        now = datetime.now(timezone.utc)
        mock_retrieve.return_value = [
            {
                "document": json.dumps({
                    "role": "researcher",
                    "task_summary": "AI research",
                    "blockers": "Cannot verify claims",
                }),
                "metadata": {
                    "confidence": "low",
                    "ts": now.isoformat(),
                },
                "distance": 0.1,
            },
        ]
        trigger = _check_low_confidence()
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger["trigger_type"], "low_confidence")
        self.assertIn("researcher", trigger["description"])

    @patch("app.proactive.trigger_scanner.retrieve_with_metadata")
    def test_no_low_confidence_returns_none(self, mock_retrieve):
        from app.proactive.trigger_scanner import _check_low_confidence
        mock_retrieve.return_value = [
            {
                "document": "{}",
                "metadata": {"confidence": "high", "ts": datetime.now(timezone.utc).isoformat()},
                "distance": 0.1,
            },
        ]
        trigger = _check_low_confidence()
        self.assertIsNone(trigger)

    @patch("app.proactive.trigger_scanner.get_beliefs")
    def test_unfulfilled_needs_trigger(self, mock_beliefs):
        from app.proactive.trigger_scanner import _check_unfulfilled_needs
        mock_beliefs.return_value = [
            {
                "agent": "researcher",
                "state": "blocked",
                "needs": ["data validation from coder"],
            },
        ]
        trigger = _check_unfulfilled_needs()
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger["trigger_type"], "unfulfilled_needs")
        self.assertIn("researcher", trigger["description"])

    @patch("app.proactive.trigger_scanner.get_beliefs")
    def test_no_needs_returns_none(self, mock_beliefs):
        from app.proactive.trigger_scanner import _check_unfulfilled_needs
        mock_beliefs.return_value = [
            {"agent": "researcher", "state": "completed", "needs": []},
        ]
        trigger = _check_unfulfilled_needs()
        self.assertIsNone(trigger)

    @patch("app.proactive.trigger_scanner.store_scoped")
    def test_execute_proactive_action_low_confidence(self, mock_store):
        from app.proactive.trigger_scanner import execute_proactive_action
        trigger = {
            "trigger_type": "low_confidence",
            "description": "researcher low confidence on AI research",
            "suggested_action": "Verify with additional sources",
        }
        result = execute_proactive_action(trigger, "original result")
        self.assertIsNotNone(result)
        self.assertIn("Low confidence", result)
        mock_store.assert_called_once()

    @patch("app.proactive.trigger_scanner.retrieve_with_metadata")
    @patch("app.proactive.trigger_scanner.get_beliefs")
    def test_scan_for_triggers_empty_state(self, mock_beliefs, mock_retrieve):
        from app.proactive.trigger_scanner import scan_for_triggers
        mock_retrieve.return_value = []
        mock_beliefs.return_value = []
        triggers = scan_for_triggers(
            {"result": "test", "crews": "research"},
            "test task",
        )
        self.assertEqual(triggers, [])


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Meta-Cognitive Self-Improvement Loop
# ═══════════════════════════════════════════════════════════════════════════════

class TestPolicyLoader(unittest.TestCase):
    """Test policy storage and loading."""

    @patch("app.policies.policy_loader.store_scoped")
    def test_store_policy(self, mock_store):
        from app.policies.policy_loader import store_policy
        store_policy(
            trigger="When confidence is low",
            action="Verify with 2+ additional sources",
            evidence="Low confidence correlated with quality drops",
        )
        mock_store.assert_called_once()
        args = mock_store.call_args
        self.assertEqual(args[0][0], "scope_policies")

    @patch("app.policies.policy_loader.retrieve_strategic")
    def test_load_relevant_policies_formats_output(self, mock_retrieve):
        from app.policies.policy_loader import load_relevant_policies
        mock_retrieve.return_value = [
            json.dumps({
                "trigger": "When research confidence is low",
                "action": "Verify claims with 2+ sources",
            }),
        ]
        result = load_relevant_policies("Research AI", "researcher")
        self.assertIn("ACTIVE POLICIES", result)
        self.assertIn("TRIGGER:", result)
        self.assertIn("ACTION:", result)

    @patch("app.policies.policy_loader.retrieve_strategic")
    def test_load_no_policies_returns_empty(self, mock_retrieve):
        from app.policies.policy_loader import load_relevant_policies
        mock_retrieve.return_value = []
        result = load_relevant_policies("test", "researcher")
        self.assertEqual(result, "")

    @patch("app.policies.policy_loader.retrieve_with_metadata")
    def test_get_all_policies(self, mock_retrieve):
        from app.policies.policy_loader import get_all_policies
        mock_retrieve.return_value = [
            {
                "document": json.dumps({"trigger": "T", "action": "A", "evidence": "E"}),
                "metadata": {},
                "distance": 0.1,
            },
        ]
        policies = get_all_policies()
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["trigger"], "T")


class TestBenchmarks(unittest.TestCase):
    """Test the benchmarking system."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.journal_path = Path(self.tmpdir) / "benchmarks.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    @patch("app.benchmarks.BENCHMARK_PATH")
    def test_record_and_summarize(self, mock_path):
        mock_path.__class__ = Path
        mock_path.parent = Path(self.tmpdir)
        mock_path.exists = lambda: self.journal_path.exists()
        mock_path.read_text = lambda: self.journal_path.read_text()
        mock_path.write_text = lambda x: self.journal_path.write_text(x)

        from app.benchmarks import record_metric, get_benchmark_summary

        # Record some metrics
        record_metric("task_completion_time", 5.2, {"crew": "research"})
        record_metric("task_completion_time", 4.8, {"crew": "coding"})
        record_metric("quality_score", 8.0)

        summary = get_benchmark_summary()
        self.assertIn("task_completion_time", summary)
        self.assertIn("quality_score", summary)

    @patch("app.benchmarks.BENCHMARK_PATH")
    def test_empty_summary(self, mock_path):
        mock_path.exists = lambda: False
        from app.benchmarks import get_benchmark_summary
        summary = get_benchmark_summary()
        self.assertIn("status", summary)

    @patch("app.benchmarks.BENCHMARK_PATH")
    def test_trend(self, mock_path):
        mock_path.__class__ = Path
        mock_path.parent = Path(self.tmpdir)
        mock_path.exists = lambda: self.journal_path.exists()
        mock_path.read_text = lambda: self.journal_path.read_text()
        mock_path.write_text = lambda x: self.journal_path.write_text(x)

        from app.benchmarks import record_metric, get_benchmark_trend

        for i in range(5):
            record_metric("task_completion_time", 10.0 - i)

        trend = get_benchmark_trend("task_completion_time")
        self.assertEqual(len(trend), 5)
        self.assertEqual(trend[0], 10.0)
        self.assertEqual(trend[-1], 6.0)

    @patch("app.benchmarks.BENCHMARK_PATH")
    def test_compare_benchmarks(self, mock_path):
        mock_path.__class__ = Path
        mock_path.parent = Path(self.tmpdir)
        mock_path.exists = lambda: self.journal_path.exists()
        mock_path.read_text = lambda: self.journal_path.read_text()
        mock_path.write_text = lambda x: self.journal_path.write_text(x)

        from app.benchmarks import record_metric, compare_benchmarks

        # Period 1 (older, lower quality)
        for _ in range(5):
            record_metric("quality_score", 5.0)
        # Period 2 (recent, higher quality)
        for _ in range(5):
            record_metric("quality_score", 8.0)

        comp = compare_benchmarks(5, 5)
        self.assertIn("quality_score", comp)
        self.assertEqual(comp["quality_score"]["direction"], "improved")
        self.assertGreater(comp["quality_score"]["change_pct"], 0)


class TestRetrospectiveCrew(unittest.TestCase):
    """Test the retrospective crew trace gathering."""

    @patch("app.crews.retrospective_crew.retrieve_with_metadata")
    def test_gather_traces_with_data(self, mock_retrieve):
        from app.crews.retrospective_crew import RetrospectiveCrew

        mock_retrieve.side_effect = lambda col, query, n=5: [
            {
                "document": json.dumps({
                    "role": "researcher",
                    "task_summary": "AI research",
                    "confidence": "high",
                    "completeness": "complete",
                }),
                "metadata": {"confidence": "high"},
                "distance": 0.1,
            },
        ] if col == "self_reports" else []

        crew = RetrospectiveCrew()
        traces = crew._gather_traces()
        self.assertIn("Self-Reports", traces)
        self.assertIn("researcher", traces)

    @patch("app.crews.retrospective_crew.retrieve_with_metadata")
    def test_gather_traces_empty(self, mock_retrieve):
        from app.crews.retrospective_crew import RetrospectiveCrew
        mock_retrieve.return_value = []
        crew = RetrospectiveCrew()
        traces = crew._gather_traces()
        self.assertEqual(traces, "")

    def test_parse_and_store_policies(self):
        from app.crews.retrospective_crew import RetrospectiveCrew
        crew = RetrospectiveCrew()

        raw = json.dumps([
            {
                "trigger": "When research confidence is low",
                "action": "Verify with 2+ additional sources",
                "evidence": "Observed in 3 recent runs",
            },
            {
                "trigger": "When code fails first execution",
                "action": "Add input validation before main logic",
                "evidence": "60% of code failures were input-related",
            },
        ])

        with patch("app.crews.retrospective_crew.store_policy") as mock_store:
            count = crew._parse_and_store_policies(raw)
            self.assertEqual(count, 2)
            self.assertEqual(mock_store.call_count, 2)

    def test_parse_invalid_json(self):
        from app.crews.retrospective_crew import RetrospectiveCrew
        crew = RetrospectiveCrew()
        count = crew._parse_and_store_policies("not valid json")
        self.assertEqual(count, 0)

    def test_parse_caps_at_five(self):
        from app.crews.retrospective_crew import RetrospectiveCrew
        crew = RetrospectiveCrew()
        raw = json.dumps([
            {"trigger": f"T{i}", "action": f"A{i}", "evidence": f"E{i}"}
            for i in range(10)
        ])
        with patch("app.crews.retrospective_crew.store_policy"):
            count = crew._parse_and_store_policies(raw)
            self.assertEqual(count, 5)


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-PHASE: Integration Checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewConfigFields(unittest.TestCase):
    """Test that config.py has all new fields."""

    def test_new_settings_exist(self):
        # Read config source and check for new fields
        config_path = Path(__file__).parent.parent / "app" / "config.py"
        source = config_path.read_text()
        self.assertIn("retrospective_cron", source)
        self.assertIn("benchmark_cron", source)


class TestNewSignalCommands(unittest.TestCase):
    """Test that commander.py has all new Signal commands."""

    def test_new_commands_exist(self):
        cmd_path = Path(__file__).parent.parent / "app" / "agents" / "commander.py"
        source = cmd_path.read_text()
        self.assertIn("retrospective", source)
        self.assertIn("benchmarks", source)
        self.assertIn("show policies", source)


class TestMainLifespanJobs(unittest.TestCase):
    """Test that main.py has all new scheduled jobs."""

    def test_new_jobs_exist(self):
        main_path = Path(__file__).parent.parent / "app" / "main.py"
        source = main_path.read_text()
        self.assertIn("RetrospectiveCrew", source)
        self.assertIn("get_benchmark_summary", source)
        self.assertIn("retrospective", source)
        self.assertIn("benchmark_snapshot", source)


class TestCrewsHaveBeliefUpdates(unittest.TestCase):
    """Test that all crews update belief states."""

    def test_research_crew_has_beliefs(self):
        source = (Path(__file__).parent.parent / "app/crews/research_crew.py").read_text()
        self.assertIn("update_belief", source)
        self.assertIn('"researcher"', source)
        self.assertIn('"working"', source)
        self.assertIn('"completed"', source)
        self.assertIn('"failed"', source)

    def test_coding_crew_has_beliefs(self):
        source = (Path(__file__).parent.parent / "app/crews/coding_crew.py").read_text()
        self.assertIn("update_belief", source)
        self.assertIn('"working"', source)

    def test_writing_crew_has_beliefs(self):
        source = (Path(__file__).parent.parent / "app/crews/writing_crew.py").read_text()
        self.assertIn("update_belief", source)
        self.assertIn('"working"', source)


class TestCrewsHavePolicies(unittest.TestCase):
    """Test that policies are loaded — now centralized in commander._run_crew()."""

    def test_commander_loads_policies_for_crews(self):
        # S6: Policy loading moved to commander._run_crew() parallel context fetch
        source = (Path(__file__).parent.parent / "app/agents/commander.py").read_text()
        self.assertIn("_load_policies_for_crew", source)
        self.assertIn("load_relevant_policies", source)

    def test_crews_have_benchmarks(self):
        for crew_file in ["research_crew.py", "coding_crew.py", "writing_crew.py"]:
            source = (Path(__file__).parent.parent / f"app/crews/{crew_file}").read_text()
            self.assertIn("record_metric", source, f"{crew_file} missing record_metric")


class TestAgentsHaveSelfAwareness(unittest.TestCase):
    """Test that all agents have self-models and awareness tools."""

    def _check_agent_source(self, filepath):
        source = (Path(__file__).parent.parent / filepath).read_text()
        # Self-model is loaded via compose_backstory (which calls format_self_model_block internally)
        self.assertTrue(
            "format_self_model_block" in source or "compose_backstory" in source,
            f"Expected format_self_model_block or compose_backstory in {filepath}"
        )
        # R5: Self-report and reflection tools moved to post-crew hook.
        # Agents still have scoped memory tools for operational memory.
        self.assertIn("create_scoped_memory_tools", source)

    def test_researcher(self):
        self._check_agent_source("app/agents/researcher.py")

    def test_coder(self):
        self._check_agent_source("app/agents/coder.py")

    def test_writer(self):
        self._check_agent_source("app/agents/writer.py")


if __name__ == "__main__":
    unittest.main()
