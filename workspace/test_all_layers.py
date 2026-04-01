import sys, json, os
from pathlib import Path

P = 0; F = 0; E = []

def t(n, fn):
    global P, F, E
    try:
        r = fn()
        if r:
            P += 1; print(f"  OK {n}", flush=True)
        else:
            F += 1; E.append(n); print(f"  FAIL {n}", flush=True)
    except Exception as e:
        F += 1; E.append(f"{n}: {e}"); print(f"  ERR {n}: {e}", flush=True)

for f in ["/app/workspace/agent_state.json", "/app/workspace/homeostasis.json", "/app/workspace/journal.jsonl"]:
    try:
        os.unlink(f)
    except Exception:
        pass

print("=== L1: SELF-MODEL ===", flush=True)
from app.self_awareness.agent_state import record_task, get_agent_stats, get_all_stats
record_task("research", True, "high", 6, 40.0)
t("Record success", lambda: get_agent_stats("research")["tasks_completed"] == 1)
record_task("research", False, "low", 8, 90.0)
t("Record failure", lambda: get_agent_stats("research")["tasks_failed"] == 1)
for _ in range(3):
    record_task("coding", True, "high", 5, 30.0)
t("Streak=3", lambda: get_agent_stats("coding")["streak"] == 3)
record_task("coding", False, "low", 8, 90.0)
t("Streak reset", lambda: get_agent_stats("coding")["streak"] == 0)
t("All stats >=2 crews", lambda: len(get_all_stats()) >= 2)
from app.souls.loader import _backstory_cache
_backstory_cache.clear()
from app.self_awareness.self_model import format_self_model_block
t("Self-model has runtime stats", lambda: "Runtime Statistics" in format_self_model_block("researcher"))
t("Role mapping: coder", lambda: "Runtime Statistics" in format_self_model_block("coder"))

print("\n=== L2: METACOGNITIVE ===", flush=True)
from app.tools.self_report_tool import SelfReportTool
from app.tools.reflection_tool import ReflectionTool
t("SelfReportTool exists", lambda: hasattr(SelfReportTool, "_run"))
t("ReflectionTool exists", lambda: hasattr(ReflectionTool, "_run"))
_backstory_cache.clear()
from app.souls.loader import compose_backstory
t("Metacog preamble in backstory", lambda: "Self-Awareness Protocol" in compose_backstory("writer"))

print("\n=== L3: REFLEXION ===", flush=True)
from app.agents.commander import _passes_quality_gate, _generate_reflection
from app.agents.commander import _store_reflexion_success, _load_past_reflexion_lessons
t("Quality gate callable", lambda: callable(_passes_quality_gate))
t("Reflection gen non-empty", lambda: len(_generate_reflection("task", "", "research", 1)) > 0)
t("Lesson store no error", lambda: _store_reflexion_success("task", 2, ["try harder"]) or True)
t("Lesson retrieve is list", lambda: isinstance(_load_past_reflexion_lessons("task"), list))

print("\n=== L4: AUTOBIOGRAPHY ===", flush=True)
j = Path("/app/workspace/journal.jsonl")
with open(j, "a") as fj:
    fj.write(json.dumps({"ts": "now", "crew": "test"}) + "\n")
t("Journal file writable", lambda: j.stat().st_size > 0)
from app.memory.system_chronicle import generate_and_save, load_chronicle
generate_and_save()
c = load_chronicle()
t("Chronicle generated (>500 chars)", lambda: len(c) > 500)
t("Chronicle has Who I Am", lambda: "Who I Am" in c)
t("Chronicle has Memory Architecture", lambda: "Memory Architecture" in c)
t("Chronicle has Homeostatic", lambda: "Homeostatic" in c)
t("Skills directory has 100+ files", lambda: len(list(Path("/app/workspace/skills").glob("*.md"))) > 100)

print("\n=== L5: THEORY OF MIND ===", flush=True)
from app.memory.belief_state import get_team_state_summary
t("Team state summary is str", lambda: isinstance(get_team_state_summary(), str))

print("\n=== L6: HOMEOSTASIS ===", flush=True)
try:
    os.unlink("/app/workspace/homeostasis.json")
except Exception:
    pass
from app.self_awareness.homeostasis import update_state, get_state
from app.self_awareness.homeostasis import get_behavioral_modifiers, get_state_summary, TARGETS
update_state("task_complete", "r", True, 5)
t("Energy in range after success", lambda: 0.5 <= get_state()["cognitive_energy"] <= 1.0)
b4 = get_state()["frustration"]
update_state("task_complete", "r", False, 8)
t("Frustration rises on failure", lambda: get_state()["frustration"] > b4)
try:
    os.unlink("/app/workspace/homeostasis.json")
except Exception:
    pass
for _ in range(4):
    update_state("task_complete", "c", False, 7)
t("4 failures trigger CAUTION drive", lambda: get_behavioral_modifiers().get("tier_boost", 0) >= 2)
t("State summary has SYSTEM STATE", lambda: "SYSTEM STATE:" in get_state_summary())
t("TARGETS are correct constants", lambda: TARGETS == {
    "cognitive_energy": 0.7, "frustration": 0.1, "confidence": 0.65, "curiosity": 0.5
})

print("\n=== L7: GLOBAL WORKSPACE ===", flush=True)
from app.agents.commander import _load_homeostatic_context
t("Homeostatic context loader", lambda: isinstance(_load_homeostatic_context(), str))

print("\n=== EXISTING FEATURES ===", flush=True)
from app.agents.commander import _is_introspective, _recover_truncated_routing, Commander
t("Introspective gate positive", lambda: _is_introspective("do you have memory?"))
t("Introspective gate negative", lambda: not _is_introspective("what is the weather?"))
t("Typo tolerance (meory)", lambda: _is_introspective("do you have meory?"))
cmd = Commander()
t("Self-desc has persistent", lambda: "persistent" in cmd._generate_self_description("do you have memory?").lower())
from app.philosophy.rag_tool import PhilosophyRAGTool
t("Philosophy RAG returns passages", lambda: "Passage" in PhilosophyRAGTool()._run("What is virtue?", n_results=2))
from app.souls.loader import load_constitution
t("Constitution has DIGNITY", lambda: "DIGNITY" in load_constitution())
t("Constitution has Humanist", lambda: "Humanist" in load_constitution())
trunc_json = '{"crews": [{"crew": "research", "task": "Find data", "difficulty": 7}, {"crew": "coding", "task": "Cre'
t("Truncated JSON recovery", lambda: _recover_truncated_routing(trunc_json)[0]["crew"] == "research")
import urllib.request as ur
t("Health API", lambda: json.loads(ur.urlopen("http://localhost:8765/health").read())["status"] == "ok")
t("Philosophy API >600 chunks", lambda: json.loads(ur.urlopen("http://localhost:8765/philosophy/status").read())["total_chunks"] > 600)
t("KB API >0 chunks", lambda: json.loads(ur.urlopen("http://localhost:8765/kb/status").read())["total_chunks"] > 0)

print("\n=== E2E: Commander handle() ===", flush=True)
t("E2E: memory question", lambda: "persistent" in Commander().handle("do you have memory?", "t", []).lower())
t("E2E: describe yourself", lambda: "multi-agent" in Commander().handle("describe yourself", "t", []).lower())
t("E2E: kb status command", lambda: "chunk" in Commander().handle("kb status", "t", []).lower())

print("\n=== SAFETY ===", flush=True)
t("Constitution in Docker image", lambda: Path("/app/app/souls/constitution.md").exists())
t("TARGETS immutable", lambda: TARGETS["cognitive_energy"] == 0.7)
t("Phil tool is read-only", lambda: not hasattr(PhilosophyRAGTool(), "_write"))

for f in ["/app/workspace/agent_state.json", "/app/workspace/homeostasis.json", "/app/workspace/journal.jsonl"]:
    try:
        os.unlink(f)
    except Exception:
        pass

print(f"\n{'=' * 50}", flush=True)
print(f"RESULTS: {P}/{P + F} passed, {F} failed", flush=True)
if E:
    print("Failed:", flush=True)
    for e in E:
        print(f"  - {e}", flush=True)
else:
    print("ALL TESTS PASSED", flush=True)
