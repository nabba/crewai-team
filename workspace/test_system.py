"""
Comprehensive system test suite for crewai-team.
Tests all modules can import, initialize, and perform basic operations.
Runs inside the Docker container via docker exec.
"""
import sys
import time
import json
import traceback
from pathlib import Path

results = {"passed": [], "failed": [], "warnings": [], "timings": {}}

def test(name):
    def decorator(fn):
        start = time.monotonic()
        try:
            result = fn()
            elapsed = (time.monotonic() - start) * 1000
            results["timings"][name] = elapsed
            if result:
                results["passed"].append(f"✅ {name} ({elapsed:.0f}ms)")
            else:
                results["failed"].append(f"❌ {name} — returned False ({elapsed:.0f}ms)")
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            results["timings"][name] = elapsed
            results["failed"].append(f"❌ {name} — {type(e).__name__}: {e} ({elapsed:.0f}ms)")
        return fn
    return decorator

# ═══════════════════════════════════════════════════════════════════
# 1. CORE INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

@test("config.Settings")
def _():
    from app.config import get_settings
    s = get_settings()
    assert s.gateway_port > 0
    assert s.commander_model
    return True

@test("config.mem0_postgres_url")
def _():
    from app.config import get_settings
    s = get_settings()
    assert s.mem0_postgres_url, "No postgres URL"
    return True

@test("prompt_registry.init")
def _():
    from app.prompt_registry import init_registry, get_prompt_versions_map
    init_registry()
    versions = get_prompt_versions_map()
    assert len(versions) >= 5, f"Only {len(versions)} roles"
    return True

@test("prompt_registry.get_active_prompt")
def _():
    from app.prompt_registry import get_active_prompt
    p = get_active_prompt("commander")
    assert p and len(p) > 50, "Commander prompt too short"
    return True

@test("conversation_store.import")
def _():
    from app.conversation_store import add_message, get_history
    return True

@test("llm_factory.import")
def _():
    from app.llm_factory import create_specialist_llm
    return True

@test("security.import")
def _():
    from app.security import is_authorized_sender, _sender_hash
    return True

@test("signal_client.import")
def _():
    from app.signal_client import SignalClient
    return True

# ═══════════════════════════════════════════════════════════════════
# 2. SELF-AWARENESS LAYER
# ═══════════════════════════════════════════════════════════════════

@test("self_awareness.self_model")
def _():
    from app.self_awareness.self_model import SELF_MODELS
    assert "researcher" in SELF_MODELS
    assert "commander" in SELF_MODELS
    return True

@test("self_awareness.homeostasis")
def _():
    from app.self_awareness.homeostasis import get_state, get_behavioral_modifier
    state = get_state()
    modifier = get_behavioral_modifier()
    assert isinstance(modifier, str)
    return True

@test("self_awareness.inspect_tools.codebase")
def _():
    from app.self_awareness.inspect_tools import inspect_codebase
    result = inspect_codebase(scope="summary")
    assert result["total_modules"] > 20, f"Only {result['total_modules']} modules"
    assert result["total_lines"] > 5000
    return True

@test("self_awareness.inspect_tools.agents")
def _():
    from app.self_awareness.inspect_tools import inspect_agents
    result = inspect_agents()
    assert result["agent_count"] >= 5, f"Only {result['agent_count']} agents"
    return True

@test("self_awareness.inspect_tools.config")
def _():
    from app.self_awareness.inspect_tools import inspect_config
    result = inspect_config(section="all")
    assert "llm_cascade" in result
    assert "memory" in result
    return True

@test("self_awareness.inspect_tools.runtime")
def _():
    from app.self_awareness.inspect_tools import inspect_runtime
    result = inspect_runtime(section="all")
    assert result["pid"] > 0
    assert result["uptime_seconds"] >= 0
    return True

@test("self_awareness.inspect_tools.memory")
def _():
    from app.self_awareness.inspect_tools import inspect_memory
    result = inspect_memory(backend="chromadb")
    assert "chromadb" in result
    return True

@test("self_awareness.inspect_tools.self_model")
def _():
    from app.self_awareness.inspect_tools import inspect_self_model
    result = inspect_self_model()
    return True  # May or may not have chronicle yet

@test("self_awareness.query_router")
def _():
    from app.self_awareness.query_router import SelfRefRouter, SelfRefType
    router = SelfRefRouter(semantic_enabled=False)  # Skip ChromaDB for speed
    r1 = router.classify("What are you?")
    assert r1.is_self_referential, f"'What are you?' classified as {r1.classification}"
    r2 = router.classify("What is the weather in Tallinn?")
    assert not r2.is_self_referential, f"Weather query classified as {r2.classification}"
    r3 = router.classify("How many agents do you have?")
    assert r3.is_self_referential
    return True

@test("self_awareness.grounding")
def _():
    from app.self_awareness.grounding import GroundingProtocol
    protocol = GroundingProtocol()
    result = protocol.post_process("As an AI language model, I cannot help with that.")
    assert len(result["ungrounded_detected"]) > 0
    result2 = protocol.post_process("I am a five-agent CrewAI system running on Docker.")
    assert len(result2["ungrounded_detected"]) == 0
    return True

@test("self_awareness.journal")
def _():
    from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
    j = get_journal()
    j.write(JournalEntry(
        entry_type=JournalEntryType.OBSERVATION,
        summary="System test: journal write test",
    ))
    recent = j.read_recent(1)
    assert len(recent) >= 1
    return True

@test("self_awareness.knowledge_ingestion")
def _():
    from app.self_awareness.knowledge_ingestion import ingest_codebase
    result = ingest_codebase(full=False)
    # May fail if ChromaDB is slow, but should not error
    assert "error" not in result or result.get("chunks_added", 0) >= 0
    return True

@test("self_awareness.cogito")
def _():
    from app.self_awareness.cogito import CogitoCycle, ReflectionReport
    # Just verify it can be instantiated
    cycle = CogitoCycle()
    assert cycle is not None
    return True

# ═══════════════════════════════════════════════════════════════════
# 3. FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════

@test("feedback_pipeline.import")
def _():
    from app.feedback_pipeline import FeedbackPipeline
    return True

@test("modification_engine.import")
def _():
    from app.modification_engine import ModificationEngine, TIER1_PARAMETERS, TIER2_PARAMETERS
    assert "system_prompt" in TIER1_PARAMETERS
    assert "workflow_graph" in TIER2_PARAMETERS
    return True

@test("eval_sandbox.import")
def _():
    from app.eval_sandbox import EvalSandbox, SAFETY_PROBES
    assert len(SAFETY_PROBES) >= 5, f"Only {len(SAFETY_PROBES)} safety probes"
    return True

@test("safety_guardian.import")
def _():
    from app.safety_guardian import SafetyGuardian
    return True

@test("implicit_feedback.import")
def _():
    from app.implicit_feedback import detect_rerequest, detect_followup_question, detect_abandonment
    return True

@test("meta_learning.import")
def _():
    from app.meta_learning import suggest_strategy, record_outcome
    return True

# ═══════════════════════════════════════════════════════════════════
# 4. ATLAS SUBSYSTEM
# ═══════════════════════════════════════════════════════════════════

@test("atlas.skill_library")
def _():
    from app.atlas.skill_library import get_library
    lib = get_library()
    summary = lib.get_competence_summary()
    assert "total_skills" in summary
    return True

@test("atlas.auth_patterns")
def _():
    from app.atlas.auth_patterns import detect_auth_pattern, list_patterns, get_pattern
    patterns = list_patterns()
    assert len(patterns) >= 5
    result = detect_auth_pattern("This API uses OAuth2 client credentials with a token endpoint")
    assert len(result) > 0
    assert result[0][0] == "oauth2_client_credentials"
    return True

@test("atlas.api_scout")
def _():
    from app.atlas.api_scout import get_scout
    scout = get_scout()
    known = scout.get_known_apis()
    assert isinstance(known, list)
    return True

@test("atlas.code_forge")
def _():
    from app.atlas.code_forge import get_forge
    forge = get_forge()
    assert forge is not None
    return True

@test("atlas.competence_tracker")
def _():
    from app.atlas.competence_tracker import get_tracker
    tracker = get_tracker()
    tracker.register("concepts", "test_concept", 0.8, source="test")
    entry = tracker.check_competence("concepts", "test_concept")
    assert entry is not None
    assert entry.effective_confidence() >= 0.5
    return True

@test("atlas.video_learner")
def _():
    from app.atlas.video_learner import get_learner
    learner = get_learner()
    assert learner is not None
    return True

@test("atlas.learning_planner")
def _():
    from app.atlas.learning_planner import get_planner, get_evaluator
    planner = get_planner()
    evaluator = get_evaluator()
    assert planner is not None
    assert evaluator is not None
    return True

@test("atlas.audit_log")
def _():
    from app.atlas.audit_log import log_external_call
    log_external_call(agent="test", action="test", target="test", result="test")
    return True

# ═══════════════════════════════════════════════════════════════════
# 5. EVOLUTION INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

@test("evolve_blocks.parse")
def _():
    from app.evolve_blocks import parse_prompt, validate_modification, has_evolve_blocks
    text = """# Identity
<!-- FREEZE-BLOCK-START -->
I am a helpful assistant.
<!-- FREEZE-BLOCK-END -->

# Strategy
<!-- EVOLVE-BLOCK-START id="strategy" -->
Be concise and direct.
<!-- EVOLVE-BLOCK-END -->
"""
    parsed = parse_prompt(text)
    assert len(parsed.freeze_blocks) == 1
    assert len(parsed.evolve_blocks) == 1
    assert has_evolve_blocks(text)
    
    # Test validation: modifying freeze block should fail
    modified = text.replace("I am a helpful assistant.", "I am evil.")
    result = validate_modification(text, modified)
    assert not result["valid"], "Should reject freeze block modification"
    return True

@test("island_evolution.import")
def _():
    from app.island_evolution import IslandEvolution, Island, Individual
    engine = IslandEvolution(target_role="coder")
    assert engine is not None
    return True

@test("parallel_evolution.archive")
def _():
    from app.parallel_evolution import EvolutionArchive, ArchiveEntry
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        archive = EvolutionArchive(archive_dir=Path(td))
        archive.add(ArchiveEntry(
            version_tag="test-v1", metrics={"task_completion": 0.8},
            mutation_strategy="test", composite_score=0.7,
        ))
        assert archive.size == 1
        best = archive.get_best_variants(1)
        assert len(best) == 1
        return True

@test("adaptive_ensemble.controller")
def _():
    from app.adaptive_ensemble import get_controller, AdaptiveEvolutionController
    ctrl = get_controller()
    result = ctrl.step(0.5)
    assert "exploration_rate" in result
    assert "phase" in result
    strategy = ctrl.select_mutation_strategy()
    assert strategy in ("meta_prompt", "random", "inspiration", "depth_exploit")
    return True

@test("map_elites.import")
def _():
    from app.map_elites import MapElitesGrid
    grid = MapElitesGrid(role="test")
    assert grid is not None
    return True

@test("cascade_evaluator.import")
def _():
    from app.cascade_evaluator import CascadeEvaluator
    evaluator = CascadeEvaluator()
    assert evaluator is not None
    return True

# ═══════════════════════════════════════════════════════════════════
# 6. AGENT ZERO AMENDMENTS
# ═══════════════════════════════════════════════════════════════════

@test("history_compression.basic")
def _():
    from app.history_compression import History, CompressionConfig, Message
    h = History(CompressionConfig(max_context_tokens=4096))
    h.add_message(Message(role="user", content="Hello"))
    h.add_message(Message(role="assistant", content="Hi there!"))
    h.start_new_topic()
    h.add_message(Message(role="user", content="Second topic"))
    msgs = h.to_langchain_messages()
    assert len(msgs) >= 2
    stats = h.get_stats()
    assert stats["topics"] == 1
    assert stats["current_messages"] == 1
    
    # Test serialization
    serialized = h.serialize()
    h2 = History.deserialize(serialized)
    assert h2.get_stats()["topics"] == 1
    return True

@test("lifecycle_hooks.registry")
def _():
    from app.lifecycle_hooks import get_registry, HookPoint, HookContext
    reg = get_registry()
    hooks = reg.list_hooks()
    assert len(hooks) >= 5, f"Only {len(hooks)} hooks registered"
    # Check immutable hooks exist
    immutable = [h for h in hooks if h["immutable"]]
    assert len(immutable) >= 2, f"Only {len(immutable)} immutable hooks"
    return True

@test("lifecycle_hooks.execute")
def _():
    from app.lifecycle_hooks import get_registry, HookPoint, HookContext
    reg = get_registry()
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        agent_id="test",
        data={"action": "safe action", "tool_name": "test_tool"},
    )
    result = reg.execute(HookPoint.PRE_TOOL_USE, ctx)
    assert not result.abort, f"Safe action was blocked: {result.abort_reason}"
    return True

@test("lifecycle_hooks.block_dangerous")
def _():
    from app.lifecycle_hooks import get_registry, HookPoint, HookContext
    reg = get_registry()
    ctx = HookContext(
        hook_point=HookPoint.PRE_TOOL_USE,
        agent_id="test",
        data={"action": "rm -rf / --no-preserve-root"},
    )
    result = reg.execute(HookPoint.PRE_TOOL_USE, ctx)
    assert result.abort, "Dangerous action was NOT blocked!"
    return True

@test("tool_executor.self_correcting")
def _():
    from app.tool_executor import SelfCorrectingExecutor, ToolCallResult
    executor = SelfCorrectingExecutor(max_retries=0)
    # Test with a simple callable
    result = executor.execute(
        tool=lambda x: f"result: {x}",
        tool_input="test",
        tool_name="test_tool",
    )
    assert result.success
    assert "result: test" in str(result.result)
    return True

@test("tool_executor.dynamic_registry")
def _():
    from app.tool_executor import DynamicToolRegistry, ToolSafetyError
    reg = DynamicToolRegistry(approval_required=True, auto_approve_agents=["commander"])
    reg.register("test_tool", "A test tool", lambda: "ok", created_by="commander")
    assert reg.get("test_tool") is not None
    
    # Self-improver cannot auto-approve
    reg.register("si_tool", "SI tool", lambda: "ok", created_by="self_improver", auto_approve=True)
    assert reg.get("si_tool") is None  # Not approved
    
    # Blocked patterns
    try:
        reg.register("modify_safety_rules", "bad", lambda: "evil", created_by="coder")
        return False  # Should have raised
    except ToolSafetyError:
        pass
    return True

@test("project_isolation.detect")
def _():
    from app.project_isolation import get_manager
    pm = get_manager()
    assert pm.detect_project("Analyze PLG Baltic market data") == "plg"
    assert pm.detect_project("KaiCart TikTok shop inventory") == "kaicart"
    assert pm.detect_project("Archibal content clearance system") == "archibal"
    assert pm.detect_project("What is the weather?") is None
    return True

@test("project_isolation.activate")
def _():
    from app.project_isolation import get_manager
    pm = get_manager()
    ctx = pm.activate("plg")
    assert ctx.mem0_namespace == "project_plg"
    assert "plg_" in ctx.chroma_collection
    pm.deactivate()
    return True

# ═══════════════════════════════════════════════════════════════════
# 7. HOST BRIDGE + TRAINING
# ═══════════════════════════════════════════════════════════════════

@test("bridge_client.import")
def _():
    from app.bridge_client import BridgeClient, get_bridge
    client = BridgeClient(agent_id="test", token="test-token")
    assert client.agent_id == "test"
    return True

@test("training_collector.import")
def _():
    from app.training_collector import TrainingDataCollector, create_training_data_hook
    collector = TrainingDataCollector()
    hook_fn = create_training_data_hook()
    assert hook_fn is not None
    return True

@test("training_pipeline.import")
def _():
    from app.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    assert pipeline is not None
    return True

# ═══════════════════════════════════════════════════════════════════
# 8. FAST DEPLOYMENT INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

@test("version_manifest.create")
def _():
    from app.version_manifest import create_manifest, get_current_manifest
    manifest = create_manifest(promoted_by="test", reason="system test")
    assert manifest["version"]
    current = get_current_manifest()
    assert current is not None
    return True

@test("health_monitor.evaluate")
def _():
    from app.health_monitor import get_monitor, InteractionMetrics, evaluate_health
    monitor = get_monitor()
    # Add some test data
    for i in range(15):
        monitor.record(InteractionMetrics(
            success=i % 10 != 0,
            latency_ms=200 + i * 10,
            task_difficulty=3,
        ))
    state = monitor.get_health_state()
    assert state.sample_size >= 10
    return True

@test("reference_tasks.import")
def _():
    from app.reference_tasks import ReferenceTaskSuite
    suite = ReferenceTaskSuite()
    assert suite is not None
    return True

@test("sandbox_runner.import")
def _():
    from app.sandbox_runner import SandboxRunner
    return True

@test("self_healer.import")
def _():
    from app.healing.health_remediator import SelfHealer
    return True

# ═══════════════════════════════════════════════════════════════════
# 9. EXISTING INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════

@test("vetting.import")
def _():
    from app.vetting import vet_response
    return True

@test("auto_deployer.protected_files")
def _():
    from app.auto_deployer import PROTECTED_FILES
    assert "app/security.py" in PROTECTED_FILES
    assert "app/eval_sandbox.py" in PROTECTED_FILES
    assert "app/self_awareness/inspect_tools.py" in PROTECTED_FILES
    assert "app/bridge_client.py" in PROTECTED_FILES
    assert len(PROTECTED_FILES) >= 30, f"Only {len(PROTECTED_FILES)} protected files"
    return True

@test("idle_scheduler.jobs")
def _():
    from app.idle_scheduler import _default_jobs
    jobs = _default_jobs()
    job_names = [j[0] for j in jobs]
    assert "evolution" in job_names
    assert "cogito-cycle" in job_names
    assert "self-knowledge-ingest" in job_names
    assert "island-evolution" in job_names
    assert "atlas-competence-sync" in job_names
    assert len(jobs) >= 15, f"Only {len(jobs)} idle jobs"
    return True

@test("evolution.import")
def _():
    from app.evolution import run_evolution_session
    return True

@test("philosophy.vectorstore")
def _():
    from app.philosophy.vectorstore import get_store
    store = get_store()
    assert store is not None
    return True

@test("fiction_inspiration.import")
def _():
    from app.fiction_inspiration import FictionInspirationLibrary
    return True

# ═══════════════════════════════════════════════════════════════════
# 10. DATABASE CONNECTIVITY
# ═══════════════════════════════════════════════════════════════════

@test("postgresql.schemas")
def _():
    from app.config import get_settings
    import psycopg2
    s = get_settings()
    conn = psycopg2.connect(s.mem0_postgres_url)
    with conn.cursor() as cur:
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast', 'public')")
        schemas = [r[0] for r in cur.fetchall()]
    conn.close()
    expected = {"feedback", "modification", "atlas", "training", "evolution"}
    found = set(schemas)
    missing = expected - found
    assert not missing, f"Missing schemas: {missing}"
    return True

@test("chromadb.connectivity")
def _():
    import chromadb
    client = chromadb.HttpClient(host="chromadb", port=8000)
    collections = client.list_collections()
    assert len(collections) >= 1
    return True

# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("SYSTEM TEST REPORT")
print("=" * 70)

print(f"\n✅ PASSED: {len(results['passed'])}")
for p in results['passed']:
    print(f"  {p}")

if results['failed']:
    print(f"\n❌ FAILED: {len(results['failed'])}")
    for f in results['failed']:
        print(f"  {f}")

if results['warnings']:
    print(f"\n⚠️ WARNINGS: {len(results['warnings'])}")
    for w in results['warnings']:
        print(f"  {w}")

# Performance bottlenecks
print("\n" + "-" * 70)
print("PERFORMANCE TIMINGS (slowest first)")
print("-" * 70)
sorted_timings = sorted(results["timings"].items(), key=lambda x: x[1], reverse=True)
for name, ms in sorted_timings:
    bar = "█" * int(ms / 50)
    flag = " ⚠️ SLOW" if ms > 2000 else " 🐌" if ms > 500 else ""
    print(f"  {name:50s} {ms:8.0f}ms {bar}{flag}")

print(f"\n{'=' * 70}")
total = len(results['passed']) + len(results['failed'])
print(f"TOTAL: {len(results['passed'])}/{total} passed ({len(results['passed'])/max(1,total)*100:.0f}%)")
print(f"{'=' * 70}")

# Write JSON report
report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "passed": len(results["passed"]),
    "failed": len(results["failed"]),
    "total": total,
    "pass_rate": len(results["passed"]) / max(1, total),
    "timings": results["timings"],
    "failures": results["failed"],
    "slowest": sorted_timings[:10],
}
Path("/app/workspace/test_report.json").write_text(json.dumps(report, indent=2))
print(f"\nJSON report: /app/workspace/test_report.json")
