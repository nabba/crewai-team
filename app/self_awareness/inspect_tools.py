"""
inspect_tools.py — Six read-only self-inspection tools.

Gives the system grounded self-knowledge by inspecting its own code,
configuration, runtime state, memory backends, and self-model.

All tools are READ-ONLY. They observe, never modify.

Tools:
    1. inspect_codebase  — AST-based project structure, module inventory
    2. inspect_agents    — Discovers agents from code + soul files
    3. inspect_config    — LLM cascade, memory backends (secrets redacted)
    4. inspect_runtime   — Process info, uptime, task history
    5. inspect_memory    — ChromaDB, PostgreSQL, Neo4j, Mem0 statistics
    6. inspect_self_model — Reads self-model + checks freshness

IMMUTABLE — infrastructure-level module.
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


# ── Tool 1: Codebase Inspection ──────────────────────────────────────────────

def inspect_codebase(scope: str = "summary") -> dict:
    """AST-based project structure analysis.

    scope: "summary" (module list) or "full" (with classes/functions per module)
    """
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

    return {
        "total_modules": len(modules),
        "total_lines": total_lines,
        "total_classes": total_classes,
        "total_functions": total_functions,
        "packages": sorted(packages),
        "modules": modules if scope == "full" else [m["path"] for m in modules],
    }


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
        from app.self_awareness.self_model import SELF_MODELS
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
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
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
        from app.self_awareness.homeostasis import get_state
        result["homeostasis"] = get_state()
    except Exception:
        pass

    # Agent count from self-model
    try:
        from app.self_awareness.self_model import SELF_MODELS
        result["agents_modeled"] = len(SELF_MODELS)
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
}


def run_all_inspections() -> dict:
    """Run all 6 inspection tools and return combined results."""
    results = {}
    for name, fn in ALL_INSPECT_TOOLS.items():
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
    return results
