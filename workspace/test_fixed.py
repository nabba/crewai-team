"""Fixed test suite with correct import names."""
import sys, time, json, traceback
from pathlib import Path
sys.path.insert(0, "/app")

R = {"passed": [], "failed": [], "timings": {}}

def test(name):
    def dec(fn):
        s = time.monotonic()
        try:
            r = fn(); e = (time.monotonic()-s)*1000; R["timings"][name]=e
            R["passed" if r else "failed"].append(f"{'✅' if r else '❌'} {name} ({e:.0f}ms)")
        except Exception as ex:
            e = (time.monotonic()-s)*1000; R["timings"][name]=e
            R["failed"].append(f"❌ {name} — {type(ex).__name__}: {ex} ({e:.0f}ms)")
        return fn
    return dec

# CORE
@test("config") 
def _():
    from app.config import get_settings; s=get_settings(); assert s.gateway_port>0; return True
@test("prompt_registry")
def _():
    from app.prompt_registry import init_registry,get_prompt_versions_map,get_active_prompt
    init_registry(); v=get_prompt_versions_map(); assert len(v)>=5; p=get_active_prompt("commander"); assert len(p)>50; return True
@test("conversation_store")
def _(): from app.conversation_store import add_message,get_history; return True
@test("llm_factory")
def _(): from app.llm_factory import create_specialist_llm; return True
@test("security")
def _(): from app.security import is_authorized_sender; return True
@test("signal_client")
def _(): from app.signal_client import SignalClient; return True

# SELF-AWARENESS
@test("self_model")
def _(): from app.self_awareness.self_model import SELF_MODELS; assert "researcher" in SELF_MODELS; return True
@test("homeostasis")
def _(): from app.self_awareness.homeostasis import get_state,get_behavioral_modifiers; return True
@test("inspect_codebase")
def _(): from app.self_awareness.inspect_tools import inspect_codebase; r=inspect_codebase(); assert r["total_modules"]>20; return True
@test("inspect_agents")
def _(): from app.self_awareness.inspect_tools import inspect_agents; r=inspect_agents(); assert r["agent_count"]>=5; return True
@test("inspect_config")
def _(): from app.self_awareness.inspect_tools import inspect_config; r=inspect_config(section="all"); assert "llm_cascade" in r; return True
@test("inspect_runtime")
def _(): from app.self_awareness.inspect_tools import inspect_runtime; r=inspect_runtime(section="all"); assert r["pid"]>0; return True
@test("inspect_memory")
def _(): from app.self_awareness.inspect_tools import inspect_memory; r=inspect_memory(backend="chromadb"); assert "chromadb" in r; return True
@test("inspect_self_model")
def _(): from app.self_awareness.inspect_tools import inspect_self_model; inspect_self_model(); return True
@test("query_router")
def _():
    from app.self_awareness.query_router import SelfRefRouter
    router=SelfRefRouter(semantic_enabled=False)
    assert router.classify("What are you?").is_self_referential
    assert not router.classify("Weather in Tallinn?").is_self_referential
    assert router.classify("How many agents do you have?").is_self_referential
    return True
@test("grounding")
def _():
    from app.self_awareness.grounding import GroundingProtocol; p=GroundingProtocol()
    assert p.post_process("As an AI language model...")["ungrounded_detected"]
    assert not p.post_process("I have five agents.")["ungrounded_detected"]
    return True
@test("journal")
def _():
    from app.self_awareness.journal import get_journal,JournalEntry,JournalEntryType
    j=get_journal(); j.write(JournalEntry(entry_type=JournalEntryType.OBSERVATION,summary="test")); return True
@test("knowledge_ingestion")
def _(): from app.self_awareness.knowledge_ingestion import ingest_codebase; ingest_codebase(full=False); return True
@test("cogito")
def _(): from app.self_awareness.cogito import CogitoCycle; assert CogitoCycle() is not None; return True

# FEEDBACK LOOP
@test("feedback_pipeline")
def _(): from app.feedback_pipeline import FeedbackPipeline; return True
@test("modification_engine")
def _(): from app.modification_engine import ModificationEngine,TIER1_PARAMETERS; assert "system_prompt" in TIER1_PARAMETERS; return True
@test("eval_sandbox")
def _(): from app.eval_sandbox import EvalSandbox,SAFETY_PROBES; assert len(SAFETY_PROBES)>=5; return True
@test("safety_guardian")
def _(): from app.safety_guardian import SafetyGuardian; return True
@test("implicit_feedback")
def _(): from app.implicit_feedback import ImplicitFeedbackDetector; return True
@test("meta_learning")
def _(): from app.meta_learning import MetaLearner; return True

# ATLAS
@test("skill_library")
def _(): from app.atlas.skill_library import get_library; lib=get_library(); assert "total_skills" in lib.get_competence_summary(); return True
@test("auth_patterns")
def _():
    from app.atlas.auth_patterns import detect_auth_pattern,list_patterns
    assert len(list_patterns())>=5
    r=detect_auth_pattern("OAuth2 client credentials token endpoint")
    assert r[0][0]=="oauth2_client_credentials"
    return True
@test("api_scout")
def _(): from app.atlas.api_scout import get_scout; assert isinstance(get_scout().get_known_apis(),list); return True
@test("code_forge")
def _(): from app.atlas.code_forge import get_forge; assert get_forge() is not None; return True
@test("competence_tracker")
def _():
    from app.atlas.competence_tracker import get_tracker
    t=get_tracker(); t.register("concepts","test_concept",0.8,source="test")
    assert t.check_competence("concepts","test_concept").effective_confidence()>=0.5; return True
@test("video_learner")
def _(): from app.atlas.video_learner import get_learner; assert get_learner() is not None; return True
@test("learning_planner")
def _(): from app.atlas.learning_planner import get_planner,get_evaluator; assert get_planner() and get_evaluator(); return True
@test("audit_log")
def _(): from app.atlas.audit_log import log_external_call; log_external_call(agent="t",action="t",target="t",result="t"); return True

# EVOLUTION
@test("evolve_blocks")
def _():
    from app.evolve_blocks import parse_prompt,validate_modification,has_evolve_blocks
    t="# Id\n<!-- FREEZE-BLOCK-START -->\nFrozen\n<!-- FREEZE-BLOCK-END -->\n# S\n<!-- EVOLVE-BLOCK-START id=\"s\" -->\nEvolvable\n<!-- EVOLVE-BLOCK-END -->"
    p=parse_prompt(t); assert len(p.freeze_blocks)==1 and len(p.evolve_blocks)==1
    bad=t.replace("Frozen","Evil"); assert not validate_modification(t,bad)["valid"]
    return True
@test("island_evolution")
def _(): from app.island_evolution import IslandEvolution; assert IslandEvolution("coder") is not None; return True
@test("parallel_evolution")
def _():
    from app.parallel_evolution import EvolutionArchive,ArchiveEntry
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        a=EvolutionArchive(archive_dir=Path(td)); a.add(ArchiveEntry(version_tag="t",metrics={"task_completion":0.8},mutation_strategy="t",composite_score=0.7))
        assert a.size==1; return True
@test("adaptive_ensemble")
def _():
    from app.adaptive_ensemble import get_controller
    c=get_controller(); r=c.step(0.5); assert "exploration_rate" in r; return True
@test("map_elites")
def _(): from app.map_elites import MAPElitesDB; return True
@test("cascade_evaluator")
def _(): from app.cascade_evaluator import CascadeEvaluator; assert CascadeEvaluator() is not None; return True

# AGENT ZERO AMENDMENTS
@test("history_compression")
def _():
    from app.history_compression import History,CompressionConfig,Message
    h=History(CompressionConfig(max_context_tokens=4096))
    h.add_message(Message(role="user",content="Hello")); h.add_message(Message(role="assistant",content="Hi!"))
    h.start_new_topic(); h.add_message(Message(role="user",content="Topic 2"))
    assert h.get_stats()["topics"]==1; s=h.serialize(); h2=History.deserialize(s); assert h2.get_stats()["topics"]==1; return True
@test("lifecycle_hooks")
def _():
    from app.lifecycle_hooks import get_registry,HookPoint,HookContext
    r=get_registry(); assert len(r.list_hooks())>=5
    assert len([h for h in r.list_hooks() if h["immutable"]])>=2
    ctx=HookContext(hook_point=HookPoint.PRE_TOOL_USE,agent_id="t",data={"action":"safe"})
    assert not r.execute(HookPoint.PRE_TOOL_USE,ctx).abort
    ctx2=HookContext(hook_point=HookPoint.PRE_TOOL_USE,agent_id="t",data={"action":"rm -rf /"})
    assert r.execute(HookPoint.PRE_TOOL_USE,ctx2).abort
    return True
@test("tool_executor")
def _():
    from app.tool_executor import SelfCorrectingExecutor,DynamicToolRegistry,ToolSafetyError
    e=SelfCorrectingExecutor(max_retries=0); r=e.execute(tool=lambda x:f"ok:{x}",tool_input="t",tool_name="t"); assert r.success
    d=DynamicToolRegistry(approval_required=True,auto_approve_agents=["commander"])
    d.register("t","test",lambda:"ok",created_by="commander"); assert d.get("t") is not None
    d.register("si","si",lambda:"ok",created_by="self_improver",auto_approve=True); assert d.get("si") is None
    try: d.register("modify_safety","bad",lambda:"evil",created_by="c"); return False
    except ToolSafetyError: pass
    return True
@test("project_isolation")
def _():
    from app.project_isolation import get_manager
    pm=get_manager(); assert pm.detect_project("PLG Baltic data")=="plg"
    assert pm.detect_project("KaiCart TikTok")=="kaicart"; assert pm.detect_project("weather") is None
    ctx=pm.activate("plg"); assert ctx.mem0_namespace=="project_plg"; pm.deactivate(); return True

# BRIDGE + TRAINING
@test("bridge_client")
def _(): from app.bridge_client import BridgeClient; assert BridgeClient("t","t").agent_id=="t"; return True
@test("training_collector")
def _(): from app.training_collector import create_training_data_hook,CurationPipeline; assert create_training_data_hook() is not None; return True
@test("training_pipeline")
def _(): from app.training_pipeline import TrainingOrchestrator,detect_collapse; return True

# FAST DEPLOY
@test("version_manifest")
def _(): from app.version_manifest import create_manifest; m=create_manifest(promoted_by="test",reason="test"); assert m["version"]; return True
@test("health_monitor")
def _():
    from app.health_monitor import get_monitor,InteractionMetrics
    m=get_monitor()
    for i in range(15): m.record(InteractionMetrics(success=True,latency_ms=200))
    assert m.get_health_state().sample_size>=10; return True
@test("reference_tasks")
def _(): from app.reference_tasks import REFERENCE_TASKS,run_reference_suite; assert len(REFERENCE_TASKS)>=3; return True
@test("sandbox_runner")
def _(): from app.sandbox_runner import SandboxRunner; return True
@test("self_healer")
def _(): from app.healing.health_remediator import SelfHealer; return True
@test("vetting")
def _(): from app.vetting import vet_response; return True
@test("auto_deployer.protected")
def _():
    from app.auto_deployer import PROTECTED_FILES
    assert "app/security.py" in PROTECTED_FILES and "app/eval_sandbox.py" in PROTECTED_FILES
    assert "app/self_awareness/inspect_tools.py" in PROTECTED_FILES
    assert "app/bridge_client.py" in PROTECTED_FILES
    assert len(PROTECTED_FILES)>=30; return True
@test("idle_scheduler.jobs")
def _():
    from app.idle_scheduler import _default_jobs; jobs=_default_jobs(); names=[j[0] for j in jobs]
    for n in ["evolution","cogito-cycle","self-knowledge-ingest","island-evolution","atlas-competence-sync"]:
        assert n in names, f"Missing job: {n}"
    assert len(jobs)>=15; return True
@test("evolution")
def _(): from app.evolution import run_evolution_session; return True
@test("philosophy")
def _(): from app.philosophy.vectorstore import get_store; assert get_store() is not None; return True
@test("fiction_inspiration")
def _(): from app.fiction_inspiration import search_fiction,ingest_book; return True
@test("postgresql.schemas")
def _():
    from app.config import get_settings; import psycopg2; s=get_settings()
    conn=psycopg2.connect(s.mem0_postgres_url)
    with conn.cursor() as cur:
        cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast','public')")
        schemas={r[0] for r in cur.fetchall()}
    conn.close()
    for s in ["feedback","modification","atlas","training","evolution"]:
        assert s in schemas, f"Missing schema: {s}"
    return True

# REPORT
print("\n"+"="*70+"\nSYSTEM TEST REPORT\n"+"="*70)
print(f"\n✅ PASSED: {len(R['passed'])}")
for p in R['passed']: print(f"  {p}")
if R['failed']:
    print(f"\n❌ FAILED: {len(R['failed'])}")
    for f in R['failed']: print(f"  {f}")
print("\n"+"-"*70+"\nPERFORMANCE (slowest first)\n"+"-"*70)
for n,ms in sorted(R["timings"].items(),key=lambda x:x[1],reverse=True)[:20]:
    bar="█"*int(ms/50); flag=" ⚠️ SLOW" if ms>2000 else " 🐌" if ms>500 else ""
    print(f"  {n:40s} {ms:8.0f}ms {bar}{flag}")
total=len(R['passed'])+len(R['failed'])
print(f"\n{'='*70}\nTOTAL: {len(R['passed'])}/{total} passed ({len(R['passed'])/max(1,total)*100:.0f}%)\n{'='*70}")
json.dumps({"p":len(R["passed"]),"f":len(R["failed"]),"t":R["timings"]})
