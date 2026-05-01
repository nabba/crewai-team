"""
subia.tsal.inspect_tools — eight read-only self-inspection tools.

Canonical home (Phase 13). The previous location
`app.subia.tsal.inspect_tools` is preserved as a graceful shim that
re-exports from this module via sys.modules aliasing (same pattern
Phase 1 used for the consciousness/ and self_awareness/ migrations).

Tools:
    1. inspect_codebase        — AST-based project structure
    2. inspect_agents          — Agents from code + soul files
    3. inspect_config          — LLM cascade, memory backends (redacted)
    4. inspect_runtime         — Process info, uptime, task history
    5. inspect_memory          — ChromaDB, PostgreSQL, Neo4j, Mem0 stats
    6. inspect_self_model      — Self-model + chronicle freshness
    7. inspect_beliefs         — HOT-3 belief store query
    8. inspect_attention_state — AST-1 attention schema state

These are the LOW-LEVEL inspectors. The TSAL HostProber, ResourceMonitor,
CodeAnalyst, and ComponentDiscovery wrap and extend them with structured
profiles that feed wiki-native page generation and SubIA closed-loop
wiring. See `app.subia.tsal` package docstring.

IMMUTABLE — infrastructure-level module (Tier-3 protected).
"""

import ast
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR = Path("/app/app")
WORKSPACE = Path("/app/workspace")
SOULS_DIR = APP_DIR / "souls"
_startup_time = time.monotonic()

# TTL cache for inspect_codebase (avoids re-parsing 100+ files every call)
_codebase_cache: dict = {}
_codebase_cache_time: float = 0.0
_CODEBASE_CACHE_TTL = 300  # 5 minutes


# ── Tool 1: Codebase Inspection ──────────────────────────────────────────────

def inspect_codebase(scope: str = "summary") -> dict:
    """AST-based project structure analysis.

    scope: "summary" (module list) or "full" (with classes/functions per module)
    Results cached for 5 minutes to avoid re-parsing 100+ files on repeated calls.
    """
    global _codebase_cache, _codebase_cache_time
    cache_key = scope
    if cache_key in _codebase_cache and (time.monotonic() - _codebase_cache_time) < _CODEBASE_CACHE_TTL:
        return _codebase_cache[cache_key]

    if not APP_DIR.exists():
        return {"error": "App directory not found"}

    modules = []
    total_lines = 0
    total_classes = 0
    total_functions = 0

    for py_file in sorted(APP_DIR.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        rel = str(py_file.relative_to(APP_DIR.parent))
        try:
            source = py_file.read_text(encoding="utf-8", errors="ignore")
            lines = len(source.splitlines())
            total_lines += lines

            module_info = {"path": rel, "lines": lines}

            if scope == "full":
                try:
                    tree = ast.parse(source)
                    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                    functions = [n.name for n in ast.walk(tree)
                                 if isinstance(n, ast.FunctionDef) and not isinstance(n, ast.AsyncFunctionDef)]
                    async_fns = [n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
                    module_info["classes"] = classes
                    module_info["functions"] = (functions + async_fns)[:20]
                    total_classes += len(classes)
                    total_functions += len(functions) + len(async_fns)
                except SyntaxError:
                    module_info["parse_error"] = True

            modules.append(module_info)
        except Exception:
            modules.append({"path": rel, "error": "unreadable"})

    # Package inventory
    packages = set()
    for m in modules:
        parts = m["path"].split("/")
        if len(parts) > 2:
            packages.add(parts[1])

    result = {
        "total_modules": len(modules),
        "total_lines": total_lines,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "packages": sorted(packages),
        "modules": modules if scope == "full" else [m["path"] for m in modules],
    }

    # Cache result
    _codebase_cache[cache_key] = result
    _codebase_cache_time = time.monotonic()
    return result


# ── Tool 2: Agent Discovery ──────────────────────────────────────────────────

def inspect_agents() -> dict:
    """Discover all agents from soul files and self-model definitions."""
    agents = []

    # From soul files
    if SOULS_DIR.exists():
        for md in sorted(SOULS_DIR.glob("*.md")):
            name = md.stem
            if name in ("loader", "__init__"):
                continue
            content = md.read_text(encoding="utf-8", errors="ignore")
            first_line = content.strip().split("\n")[0] if content.strip() else ""
            agents.append({
                "name": name,
                "source": "soul_file",
                "path": str(md.relative_to(APP_DIR.parent)),
                "description": first_line[:200],
                "size_chars": len(content),
            })

    # From self-model definitions
    try:
        from app.subia.self.model import SELF_MODELS
        for role, model in SELF_MODELS.items():
            existing = next((a for a in agents if a["name"] == role), None)
            if existing:
                existing["capabilities"] = model.get("capabilities", [])[:5]
                existing["limitations"] = model.get("limitations", [])[:3]
            else:
                agents.append({
                    "name": role,
                    "source": "self_model",
                    "capabilities": model.get("capabilities", [])[:5],
                    "limitations": model.get("limitations", [])[:3],
                })
    except Exception:
        pass

    # From prompt registry
    try:
        from app.prompt_registry import get_prompt_versions_map
        versions = get_prompt_versions_map()
        for role, version in versions.items():
            existing = next((a for a in agents if a["name"] == role), None)
            if existing:
                existing["prompt_version"] = version
            else:
                agents.append({"name": role, "source": "prompt_registry", "prompt_version": version})
    except Exception:
        pass

    return {"agent_count": len(agents), "agents": agents}


# ── Tool 3: Configuration Inspection ─────────────────────────────────────────

def inspect_config(section: str = "summary") -> dict:
    """Read system configuration with secrets redacted."""
    try:
        from app.config import get_settings
        s = get_settings()
    except Exception as e:
        return {"error": f"Config unavailable: {e}"}

    # Redact secrets
    def _safe(val):
        s_val = str(val)
        if any(k in s_val.lower() for k in ("key", "secret", "password", "token")):
            return "***REDACTED***"
        return s_val

    config = {
        "llm_cascade": {
            "cost_mode": s.cost_mode,
            "llm_mode": s.llm_mode,
            "commander_model": s.commander_model,
            "specialist_model": s.specialist_model,
            "local_model_default": s.local_model_default,
            "local_llm_enabled": s.local_llm_enabled,
            "vetting_enabled": s.vetting_enabled,
            "vetting_model": s.vetting_model,
        },
        "memory": {
            "mem0_enabled": s.mem0_enabled,
            "mem0_postgres_host": s.mem0_postgres_host,
        },
        "evolution": {
            "evolution_iterations": s.evolution_iterations,
            "evolution_auto_deploy": s.evolution_auto_deploy,
            "feedback_enabled": s.feedback_enabled,
            "modification_enabled": s.modification_enabled,
        },
        "parallelism": {
            "max_parallel_crews": s.max_parallel_crews,
            "max_sub_agents": s.max_sub_agents,
            "thread_pool_size": s.thread_pool_size,
        },
    }

    if section == "all" or section == "full":
        config["sandbox"] = {
            "sandbox_image": s.sandbox_image,
            "sandbox_timeout": s.sandbox_timeout_seconds,
        }
        config["atlas"] = {
            "atlas_enabled": getattr(s, "atlas_enabled", False),
            "api_scout_enabled": getattr(s, "atlas_api_scout_enabled", False),
        }
        config["bridge"] = {
            "bridge_enabled": getattr(s, "bridge_enabled", False),
            "bridge_port": getattr(s, "bridge_port", 9100),
        }

    return config


# ── Tool 4: Runtime Inspection ────────────────────────────────────────────────

def inspect_runtime(section: str = "process") -> dict:
    """Process info, uptime, resource usage."""
    import platform

    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "uptime_seconds": round(time.monotonic() - _startup_time),
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }

    if section in ("all", "resources"):
        try:
            import psutil
            proc = psutil.Process()
            info["memory_rss_mb"] = round(proc.memory_info().rss / 1024 / 1024, 1)
            info["cpu_percent"] = proc.cpu_percent(interval=0.1)
            info["threads"] = proc.num_threads()
        except ImportError:
            info["memory_note"] = "psutil not available"

    if section in ("all", "tasks"):
        try:
            from app.conversation_store import get_task_history
            tasks = get_task_history(limit=10)
            info["recent_tasks"] = len(tasks)
        except Exception:
            pass

    if section in ("all", "idle"):
        try:
            from app.idle_scheduler import is_idle, is_enabled
            info["idle"] = is_idle()
            info["background_enabled"] = is_enabled()
        except Exception:
            pass

    return info


# ── Tool 5: Memory Backend Inspection ─────────────────────────────────────────

def inspect_memory(backend: str = "all") -> dict:
    """Statistics from ChromaDB, PostgreSQL, Neo4j, Mem0."""
    stats = {}

    if backend in ("all", "chromadb"):
        try:
            from app.memory.chromadb_manager import get_client
            client = get_client()
            collections = client.list_collections()
            stats["chromadb"] = {
                "collections": len(collections),
                "details": {
                    col.name: {"count": col.count()}
                    for col in collections
                },
            }
        except Exception as e:
            stats["chromadb"] = {"error": str(e)[:200]}

    if backend in ("all", "postgresql"):
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if s.mem0_postgres_url:
                conn = psycopg2.connect(s.mem0_postgres_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT schemaname, tablename
                        FROM pg_tables
                        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                        ORDER BY schemaname, tablename
                    """)
                    tables = [(r[0], r[1]) for r in cur.fetchall()]
                conn.close()
                stats["postgresql"] = {
                    "schemas": list(set(t[0] for t in tables)),
                    "tables": len(tables),
                    "table_list": [f"{t[0]}.{t[1]}" for t in tables],
                }
        except Exception as e:
            stats["postgresql"] = {"error": str(e)[:200]}

    if backend in ("all", "neo4j"):
        try:
            from neo4j import GraphDatabase
            from app.config import get_settings
            s = get_settings()
            pw = s.mem0_neo4j_password.get_secret_value()
            if pw:
                driver = GraphDatabase.driver(s.mem0_neo4j_url, auth=("neo4j", pw))
                with driver.session() as session:
                    nodes = session.run("MATCH (n) RETURN count(n) as c").single()["c"]
                    rels = session.run("MATCH ()-[r]->() RETURN count(r) as c").single()["c"]
                driver.close()
                stats["neo4j"] = {"nodes": nodes, "relationships": rels}
        except Exception as e:
            stats["neo4j"] = {"error": str(e)[:200]}

    return stats


# ── Tool 6: Self-Model Inspection ─────────────────────────────────────────────

def inspect_self_model() -> dict:
    """Read the system chronicle + check freshness."""
    chronicle_path = WORKSPACE / "system_chronicle.md"
    result = {"exists": False, "content": "", "stale": True}

    if chronicle_path.exists():
        result["exists"] = True
        content = chronicle_path.read_text(encoding="utf-8", errors="ignore")
        result["content"] = content[:3000]
        result["size_chars"] = len(content)

        # Check freshness: stale if older than 24 hours
        mtime = chronicle_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        result["age_hours"] = round(age_hours, 1)
        result["stale"] = age_hours > 24

    # Also include homeostatic state
    try:
        from app.subia.homeostasis.state import get_state
        result["homeostasis"] = get_state()
    except Exception:
        pass

    # Agent count from self-model
    try:
        from app.subia.self.model import SELF_MODELS
        result["agents_modeled"] = len(SELF_MODELS)
    except Exception:
        pass

    return result


# ── Tool 7: Belief Inspection (HOT-3) ──────────────────────────────────────────

def inspect_beliefs(domain: str | None = None, min_confidence: float = 0.0,
                    status: str = "ACTIVE", limit: int = 20) -> dict:
    """Retrieve beliefs from the epistemic belief store (HOT-3).

    Allows agents to introspect on their own belief state — a higher-order
    operation connecting HOT-2 (metacognitive monitoring) to HOT-3 (beliefs).

    Args:
        domain: Filter by domain (task_strategy, user_model, self_model, etc.)
        min_confidence: Minimum confidence threshold
        status: Belief status filter (ACTIVE, SUSPENDED, RETRACTED)
        limit: Maximum beliefs to return
    """
    result = {"beliefs": [], "total": 0, "domains": {}}
    try:
        from app.subia.belief.store import get_belief_store
        store = get_belief_store()
        beliefs = store.query_relevant(
            query="",  # Empty query returns all
            domain=domain,
            n=limit,
            min_confidence=min_confidence,
        )
        result["total"] = len(beliefs)
        for b in beliefs[:limit]:
            result["beliefs"].append({
                "belief_id": b.belief_id[:8],
                "content": b.content[:200],
                "domain": b.domain,
                "confidence": round(b.confidence, 3),
                "status": b.belief_status,
                "evidence_count": len(b.evidence_sources) if b.evidence_sources else 0,
            })
            result["domains"][b.domain] = result["domains"].get(b.domain, 0) + 1
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


# ── Tool 8: Attention State Inspection (AST-1) ────────────────────────────────

def inspect_attention_state() -> dict:
    """Inspect the current attention schema state (AST-1).

    Returns:
        current_focus: what items are in the workspace now
        attending_because: why these items won the competition
        is_stuck: boolean — same items for 5+ cycles
        is_captured: boolean — one item dominates >70% salience
        prediction_accuracy: running accuracy of attention predictions
        social_attention: Theory of Mind summary for other agents
    """
    result = {
        "current_focus": [],
        "attending_because": "",
        "is_stuck": False,
        "is_captured": False,
        "prediction_accuracy": 0.0,
        "shifts_this_period": 0,
        "workspace_size": 0,
        "social_attention": {},
    }
    try:
        from app.subia.scene.attention_schema import get_attention_schema
        schema = get_attention_schema()
        summary = schema.get_state_summary()
        result.update({
            "is_stuck": summary.get("is_stuck", False),
            "is_captured": summary.get("is_captured", False),
            "prediction_accuracy": summary.get("prediction_accuracy", 0.0),
            "shifts_this_period": summary.get("shifts_this_period", 0),
            "workspace_size": summary.get("workspace_size", 0),
        })
        if schema._current:
            result["attending_because"] = schema._current.attending_because[:200]
            result["current_focus"] = list(schema._current.workspace_item_ids)[:5]
    except Exception as e:
        result["error"] = str(e)[:200]

    # Social attention (Theory of Mind)
    try:
        from app.subia.scene.attention_schema import get_social_attention_model
        social = get_social_attention_model()
        result["social_attention"] = social.get_summary()
    except Exception:
        pass

    return result


# ── All tools as a dict for easy access ───────────────────────────────────────

ALL_INSPECT_TOOLS = {
    "inspect_codebase": inspect_codebase,
    "inspect_agents": inspect_agents,
    "inspect_config": inspect_config,
    "inspect_runtime": inspect_runtime,
    "inspect_memory": inspect_memory,
    "inspect_self_model": inspect_self_model,
    "inspect_beliefs": inspect_beliefs,
    "inspect_attention_state": inspect_attention_state,
}


def run_all_inspections() -> dict:
    """Run all 8 inspection tools and return combined results."""
    results = {}
    for name, fn in ALL_INSPECT_TOOLS.items():
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
    return results
