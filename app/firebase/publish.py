"""
firebase.publish — All report_*() functions that WRITE to Firestore documents.

These push dashboard data (metrics, status, inventories, alerts, chat, etc.)
to Firestore so the monitoring UI can display live state.
"""

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.firebase.infra import _get_db, _fire, _now_iso, _add_activity

logger = logging.getLogger(__name__)


# ── System status ─────────────────────────────────────────────────────────────

def report_system_online(version: str = "1.0") -> None:
    """Called once at startup to mark the system as online."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").set({
                "state": "online",
                "version": version,
                "started_at": _now_iso(),
                "last_seen": _now_iso(),
                "crews": {
                    "commander": "idle",
                    "research":  "idle",
                    "coding":    "idle",
                    "writing":   "idle",
                    "self_improvement": "idle",
                },
            })
        except Exception:
            logger.debug("firebase.publish: system online write failed", exc_info=True)
    _fire(_write)


def report_system_offline() -> None:
    """Called on graceful shutdown."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").update({
                "state": "offline",
                "last_seen": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: system offline write failed", exc_info=True)
    _fire(_write)


def heartbeat() -> None:
    """Update last_seen timestamp + all status data — call every 60 s."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("system").update({
                "last_seen": _now_iso(),
                "state": "online",
            })
        except Exception:
            logger.debug("firebase.publish: heartbeat write failed", exc_info=True)

        # Push fleet status + benchmarks
        try:
            from app.ollama_native import get_fleet_status
            fleet = get_fleet_status()
            benchmarks = []
            try:
                from app.llm_benchmarks import get_scores
                for task_type in ["coding", "architecture", "research", "writing"]:
                    scores = get_scores(task_type)
                    for model, score in scores.items():
                        benchmarks.append({"model": model, "task": task_type, "score": round(score, 2)})
            except Exception:
                pass
            report_fleet_status(fleet, benchmarks)
        except Exception:
            pass

        # Push token usage stats
        try:
            report_token_stats()
        except Exception:
            pass

        # Push composite metrics
        try:
            report_metrics()
        except Exception:
            pass

        # Push circuit breaker states
        try:
            report_circuit_breakers()
        except Exception:
            pass

        # Push error journal summary
        try:
            report_errors()
        except Exception:
            pass

        # Push evolution/experiment data
        try:
            report_evolution()
        except Exception:
            pass

        # Push request cost stats
        try:
            report_request_costs()
        except Exception:
            pass

        # Push model catalog (only once or on changes, but cheap enough per heartbeat)
        try:
            report_catalog()
        except Exception:
            pass

        # Push knowledge base stats
        try:
            report_knowledge_base()
        except Exception:
            pass

        # L5: Push ecological awareness stats
        try:
            report_ecological_stats()
        except Exception:
            pass

        # Push learned skills inventory
        try:
            report_skills()
        except Exception:
            pass

        # Push sentience internal state stats
        try:
            report_internal_state()
        except Exception:
            pass
    _fire(_write)


# ── Skills inventory ─────────────────────────────────────────────────────────

_SKILLS_DIR = Path("/app/workspace/skills")


def report_skills() -> None:
    """Push learned skills inventory to Firestore at status/skills."""
    db = _get_db()
    if not db:
        return
    try:
        skills = []
        if _SKILLS_DIR.exists():
            for f in sorted(_SKILLS_DIR.glob("*.md")):
                if f.name == "learning_queue.md":
                    continue
                name = f.stem
                stat = f.stat()
                # Extract first line as description
                description = ""
                try:
                    first_line = f.read_text(errors="replace").split("\n", 1)[0].strip()
                    if first_line.startswith("#"):
                        first_line = first_line.lstrip("#").strip()
                    description = first_line
                except Exception:
                    pass
                skills.append({
                    "name": name,
                    "description": description[:200],
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": stat.st_size,
                })
        db.collection("status").document("skills").set({
            "skills": skills,
            "total": len(skills),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: skills write failed", exc_info=True)


# ── Signal connection health ─────────────────────────────────────────────────

def report_signal_status(connected: bool, last_message_at: Optional[str] = None,
                         message_count: int = 0) -> None:
    """Push Signal connection health to Firestore at status/signal."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("signal").set({
                "connected": connected,
                "last_message_at": last_message_at,
                "message_count": message_count,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: signal status write failed", exc_info=True)
    _fire(_write)


# ── Proposals ─────────────────────────────────────────────────────────────────

def report_proposals(proposals: list[dict]) -> None:
    """Push current proposal list to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("proposals").set({
                "proposals": proposals[:20],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: proposals write failed", exc_info=True)
    _fire(_write)


# ── Fleet status ─────────────────────────────────────────────────────────

def report_fleet_status(fleet_data: list[dict], benchmarks: list[dict] = None) -> None:
    """Push LLM fleet container status to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("fleet").set({
                "containers": fleet_data[:10],
                "benchmarks": (benchmarks or [])[:20],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: fleet write failed", exc_info=True)
    _fire(_write)


# ── Scheduled jobs ────────────────────────────────────────────────────────────

def report_schedule(jobs: list[dict]) -> None:
    """Publish upcoming scheduled jobs.

    Each job dict: {"id": str, "name": str, "next_run": ISO str, "cron": str}
    """
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("schedule").document("jobs").set({"jobs": jobs, "updated_at": _now_iso()})
        except Exception:
            logger.debug("firebase.publish: schedule write failed", exc_info=True)
    _fire(_write)


# ── LLM mode control ────────────────────────────────────────────────────────

def report_llm_mode(mode: str) -> None:
    """Push current LLM mode to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("config").document("llm").set({
                "mode": mode,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: llm mode write failed", exc_info=True)
    _fire(_write)


# ── Metrics ──────────────────────────────────────────────────────────────────

def report_metrics() -> None:
    """Push composite metrics to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.metrics import compute_metrics
        metrics = compute_metrics()
        db.collection("status").document("metrics").set({
            **metrics,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: metrics write failed", exc_info=True)


# ── Circuit breakers ─────────────────────────────────────────────────────────

def report_circuit_breakers() -> None:
    """Push circuit breaker states to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.circuit_breaker import get_all_states
        states = get_all_states()
        db.collection("status").document("circuit_breakers").set({
            "providers": states,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: circuit breaker write failed", exc_info=True)


# ── Error journal ────────────────────────────────────────────────────────────

def report_errors() -> None:
    """Push recent errors to Firestore for dashboard display."""
    db = _get_db()
    if not db:
        return
    try:
        from app.self_heal import get_recent_errors, get_error_patterns
        errors = get_recent_errors(20)
        patterns = get_error_patterns()
        db.collection("status").document("errors").set({
            "recent": errors[:20],
            "patterns": patterns,
            "total_recent": len(errors),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: errors write failed", exc_info=True)


# ── Evolution / experiments ──────────────────────────────────────────────────

def report_evolution() -> None:
    """Push evolution experiment history to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.results_ledger import get_recent_results, get_best_score, get_improvement_trend
        results = get_recent_results(20)
        best = get_best_score()
        trend = get_improvement_trend(20)
        db.collection("status").document("evolution").set({
            "recent_experiments": results[:20],
            "best_score": best,
            "trend": trend[:20],
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: evolution write failed", exc_info=True)


# ── Request cost stats ───────────────────────────────────────────────────────

def report_request_costs() -> None:
    """Push per-request cost aggregates and per-crew breakdown to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.llm_benchmarks import get_request_cost_stats, get_crew_cost_stats
        costs = {}
        crew_costs = {}
        for period in ("hour", "day", "week", "month"):
            costs[period] = get_request_cost_stats(period)
            crew_costs[period] = get_crew_cost_stats(period)
        db.collection("status").document("request_costs").set({
            "stats": costs,
            "by_crew": crew_costs,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: request costs write failed", exc_info=True)


# ── Model catalog ────────────────────────────────────────────────────────────

def report_catalog() -> None:
    """Push model catalog and role assignments to Firestore."""
    db = _get_db()
    if not db:
        return
    try:
        from app.llm_catalog import CATALOG, ROLE_DEFAULTS
        from app.config import get_settings
        settings = get_settings()
        # Compact catalog for dashboard
        models = []
        for name, info in CATALOG.items():
            models.append({
                "name": name,
                "tier": info["tier"],
                "provider": info["provider"],
                "cost_input": info.get("cost_input_per_m", 0),
                "cost_output": info.get("cost_output_per_m", 0),
                "context": info.get("context", 0),
                "multimodal": info.get("multimodal", False),
                "tool_reliability": info.get("tool_use_reliability", 0),
                "description": info.get("description", ""),
            })
        # Current role assignments
        cost_mode = settings.cost_mode
        assignments = ROLE_DEFAULTS.get(cost_mode, ROLE_DEFAULTS.get("balanced", {}))
        db.collection("status").document("catalog").set({
            "models": models,
            "role_assignments": assignments,
            "cost_mode": cost_mode,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: catalog write failed", exc_info=True)


def report_knowledge_base() -> None:
    """Push knowledge base stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.knowledge_base.vectorstore import KnowledgeStore
        store = KnowledgeStore()
        stats = store.stats()

        # Build a mapping from source_path -> original filename from kb_queue history
        filename_map = {}
        try:
            queue_docs = (
                db.collection("kb_queue")
                .where("status", "==", "done")
                .limit(100)
                .get()
            )
            for qd in queue_docs:
                qdata = qd.to_dict()
                sp = qdata.get("source_path", "")
                orig = qdata.get("original_filename", "")
                if sp and orig:
                    filename_map[sp] = orig
        except Exception:
            pass

        # Compact document list for dashboard — include original filename + source_path
        docs = []
        for d in stats.get("documents", [])[:50]:
            source_path = d.get("source_path", "")
            source_name = d.get("source", "unknown")
            # Try to resolve original filename
            original_name = filename_map.get(source_path, "")
            if not original_name:
                # Fallback: strip temp prefixes like "kb_MyDoc_abc123.md" -> "MyDoc"
                import re as _re
                cleaned = _re.sub(r'^kb_', '', source_name)
                cleaned = _re.sub(r'_[a-z0-9]{6,}\.(md|txt|pdf|docx|pptx|xlsx|csv|html|json)$', r'.\1', cleaned)
                if cleaned != source_name:
                    original_name = cleaned
                else:
                    original_name = source_name
            docs.append({
                "source": source_name,
                "source_path": source_path,
                "original_name": original_name,
                "format": d.get("format", "?"),
                "category": d.get("category", "general"),
                "chunks": d.get("total_chunks", 0),
                "ingested_at": d.get("ingested_at", ""),
            })
        db.collection("status").document("knowledge_base").set({
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "total_characters": stats.get("total_characters", 0),
            "estimated_tokens": stats.get("estimated_tokens", 0),
            "categories": stats.get("categories", {}),
            "documents": docs,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: knowledge_base write failed", exc_info=True)


def report_evolution_stats() -> None:
    """Push evolution DGM-DB stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        import os
        if os.environ.get("EVOLUTION_USE_DGM_DB", "false").lower() != "true":
            return
        from app.evolution_db.archive_db import get_evolution_stats
        stats = get_evolution_stats()
        # Serialize recent variants for Firestore (UUIDs -> strings, datetimes -> ISO)
        recent = []
        for v in stats.get("recent", []):
            recent.append({
                "id": str(v.get("id", "")),
                "agent_name": v.get("agent_name", ""),
                "generation": v.get("generation", 0),
                "composite_score": v.get("composite_score") or 0.0,
                "passed": v.get("passed_threshold", False),
                "reasoning": (v.get("modification_reasoning") or "")[:100],
            })
        db.collection("status").document("evolution").set({
            "total_variants": stats.get("total_variants", 0),
            "passed_variants": stats.get("passed_variants", 0),
            "best_score": stats.get("best_score", 0.0),
            "active_runs": stats.get("active_runs", 0),
            "recent": recent,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: evolution stats write failed", exc_info=True)


def report_philosophy_kb() -> None:
    """Push philosophy knowledge base stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.philosophy.vectorstore import get_store
        store = get_store()
        stats = store.get_stats()

        # Build texts list from ChromaDB metadata (source of truth)
        texts_list = store.list_texts()

        db.collection("status").document("philosophy_kb").set({
            "total_chunks": stats.get("total_chunks", 0),
            "total_texts": stats.get("total_texts", 0),
            "traditions": stats.get("traditions", []),
            "authors": stats.get("authors", []),
            "titles": stats.get("titles", []),
            "texts": texts_list,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: philosophy_kb write failed", exc_info=True)


def report_fiction_library() -> None:
    """Push fiction library stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.fiction_inspiration import _get_collection, FICTION_LIBRARY_DIR
        import json as _json

        collection_obj = _get_collection()
        total = collection_obj.count()

        # Build books list from ChromaDB metadata
        books_map = {}
        if total > 0:
            all_meta = collection_obj.get(limit=min(total, 5000), include=["metadatas"])
            for meta in all_meta.get("metadatas", []):
                title = meta.get("book_title", "Unknown")
                if title not in books_map:
                    themes_raw = meta.get("themes", "[]")
                    try:
                        themes = _json.loads(themes_raw)
                    except (ValueError, TypeError):
                        themes = []
                    books_map[title] = {
                        "title": title,
                        "author": meta.get("author", "Unknown"),
                        "genre": meta.get("genre", ""),
                        "themes": themes,
                        "filename": meta.get("source_file", ""),
                        "chunks": 0,
                    }
                books_map[title]["chunks"] += 1

        all_authors = list(set(b["author"] for b in books_map.values()))
        all_themes = list(set(t for b in books_map.values() for t in b["themes"]))

        db.collection("status").document("fiction_library").set({
            "total_chunks": total,
            "total_books": len(books_map),
            "authors": all_authors,
            "themes": all_themes,
            "books": list(books_map.values()),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: fiction_library write failed", exc_info=True)


# ── L5: Ecological awareness stats ────────────────────────────────────────────

def report_ecological_stats() -> None:
    """Push ecological awareness stats to Firestore for the dashboard (L5)."""
    db = _get_db()
    if not db:
        return
    try:
        from app.memory.scoped_memory import retrieve_operational
        recent = retrieve_operational("scope_ecology", "crew execution", n=20)
        if not recent:
            return

        # Parse ecological reports for aggregate stats
        crew_stats: dict[str, list[float]] = {}
        for entry in recent:
            # Extract crew name and duration from "ECOLOGICAL: crew=X, ..."
            parts = {}
            for segment in entry.split(","):
                segment = segment.strip()
                if "=" in segment:
                    key, val = segment.split("=", 1)
                    key = key.replace("ECOLOGICAL: ", "").strip()
                    parts[key] = val.strip()
            crew_name = parts.get("crew", "unknown")
            try:
                duration = float(parts.get("duration", "0").rstrip("s"))
            except (ValueError, TypeError):
                duration = 0
            crew_stats.setdefault(crew_name, []).append(duration)

        # Build summary
        summary = {}
        for crew, durations in crew_stats.items():
            summary[crew] = {
                "recent_executions": len(durations),
                "avg_duration_s": round(sum(durations) / len(durations), 1) if durations else 0,
                "max_duration_s": round(max(durations), 1) if durations else 0,
            }

        db.collection("status").document("ecology").set({
            "crew_stats": summary,
            "total_recent_executions": sum(len(d) for d in crew_stats.values()),
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: ecology write failed", exc_info=True)


# ── Token usage stats ────────────────────────────────────────────────────────

def report_internal_state() -> None:
    """Push sentience internal state stats to Firestore for the dashboard."""
    db = _get_db()
    if not db:
        return
    try:
        from app.control_plane.db import execute

        # Aggregate recent internal states
        rows = execute(
            """
            SELECT
                agent_id,
                AVG(certainty_factual_grounding) AS avg_factual,
                AVG(certainty_tool_confidence) AS avg_tools,
                AVG(certainty_coherence) AS avg_coherence,
                AVG(certainty_meta) AS avg_meta,
                AVG(somatic_valence) AS avg_valence,
                COUNT(*) FILTER (WHERE action_disposition = 'proceed') AS proceed_count,
                COUNT(*) FILTER (WHERE action_disposition = 'cautious') AS cautious_count,
                COUNT(*) FILTER (WHERE action_disposition = 'pause') AS pause_count,
                COUNT(*) FILTER (WHERE action_disposition = 'escalate') AS escalate_count,
                COUNT(*) AS total_steps
            FROM internal_states
            WHERE created_at > NOW() - INTERVAL '1 hour'
            GROUP BY agent_id
            """,
            fetch=True,
        )

        per_agent = {}
        total_steps = 0
        disposition_totals = {"proceed": 0, "cautious": 0, "pause": 0, "escalate": 0}

        for row in (rows or []):
            agent = row.get("agent_id", "unknown") if isinstance(row, dict) else row[0]
            r = row if isinstance(row, dict) else dict(zip(
                ["agent_id", "avg_factual", "avg_tools", "avg_coherence", "avg_meta",
                 "avg_valence", "proceed_count", "cautious_count", "pause_count",
                 "escalate_count", "total_steps"], row))
            per_agent[agent] = {
                "certainty": {
                    "factual": round(float(r.get("avg_factual", 0.5)), 2),
                    "tools": round(float(r.get("avg_tools", 0.5)), 2),
                    "coherence": round(float(r.get("avg_coherence", 0.5)), 2),
                    "meta": round(float(r.get("avg_meta", 0.5)), 2),
                },
                "avg_valence": round(float(r.get("avg_valence", 0.0)), 2),
                "steps": int(r.get("total_steps", 0)),
            }
            total_steps += int(r.get("total_steps", 0))
            disposition_totals["proceed"] += int(r.get("proceed_count", 0))
            disposition_totals["cautious"] += int(r.get("cautious_count", 0))
            disposition_totals["pause"] += int(r.get("pause_count", 0))
            disposition_totals["escalate"] += int(r.get("escalate_count", 0))

        # Homeostatic state
        homeostasis_data = {}
        try:
            from app.self_awareness.homeostasis import get_state
            hs = get_state()
            homeostasis_data = {
                "cognitive_energy": round(hs.get("cognitive_energy", 0.7), 2),
                "frustration": round(hs.get("frustration", 0.1), 2),
                "confidence": round(hs.get("confidence", 0.65), 2),
                "curiosity": round(hs.get("curiosity", 0.5), 2),
            }
        except Exception:
            pass

        db.collection("status").document("internal_state").set({
            "per_agent": per_agent,
            "disposition_totals": disposition_totals,
            "total_steps_1h": total_steps,
            "homeostasis": homeostasis_data,
            "updated_at": _now_iso(),
        })
    except Exception:
        logger.debug("firebase.publish: internal_state write failed", exc_info=True)


def report_token_stats() -> None:
    """Push aggregated token usage to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.llm_benchmarks import get_token_stats
            stats = {}
            for period in ("hour", "day", "week", "month", "quarter", "year"):
                stats[period] = get_token_stats(period)
            db.collection("status").document("tokens").set({
                "stats": stats,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: token stats write failed", exc_info=True)
    _fire(_write)


# ── Self-Healing/Evolving Dashboard Data ─────────────────────────────────────

def report_anomalies() -> None:
    """Push recent anomaly alerts to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.anomaly_detector import get_recent_alerts
            alerts = get_recent_alerts(20)
            db.collection("status").document("anomalies").set({
                "recent_alerts": alerts,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: anomalies write failed", exc_info=True)
    _fire(_write)


def report_variants() -> None:
    """Push variant archive summary to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.variant_archive import get_recent_variants, get_drift_score
            recent = get_recent_variants(20)
            drift = get_drift_score()
            max_gen = max((v.get("generation", 0) for v in recent), default=0) if recent else 0
            db.collection("status").document("variants").set({
                "recent": recent,
                "drift_score": drift,
                "max_generation": max_gen,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: variants write failed", exc_info=True)
    _fire(_write)


def report_tech_radar() -> None:
    """Push tech radar discoveries to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            from app.memory.scoped_memory import retrieve_operational
            items = retrieve_operational("scope_tech_radar", "technology discovery", n=20)
            discoveries = []
            for item in (items or []):
                # Parse stored format: [category] title: summary. Action: ...
                import re as _re
                m = _re.match(r'\[(\w+)\]\s*(.+?):\s*(.+?)(?:\.\s*Action:\s*(.+))?$', item, _re.DOTALL)
                if m:
                    discoveries.append({
                        "category": m.group(1),
                        "title": m.group(2).strip(),
                        "summary": m.group(3).strip(),
                        "action": (m.group(4) or "").strip(),
                    })
                else:
                    discoveries.append({"category": "unknown", "title": item[:80], "summary": item[:200]})
            db.collection("status").document("tech_radar").set({
                "discoveries": discoveries[:15],
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: tech_radar write failed", exc_info=True)
    _fire(_write)


def report_deploys() -> None:
    """Push recent deploy log to Firestore for the dashboard."""
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            import json as _json
            from pathlib import Path as _Path
            deploy_log = _Path("/app/workspace/deploy_log.json")
            if deploy_log.exists():
                entries = _json.loads(deploy_log.read_text())[-10:]
            else:
                entries = []
            db.collection("status").document("deploys").set({
                "recent": entries,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: deploys write failed", exc_info=True)
    _fire(_write)


def report_proposal_actions() -> None:
    """Poll proposal_actions collection for dashboard-initiated approve/reject."""
    db = _get_db()
    if not db:
        return
    try:
        docs = db.collection("proposal_actions").where("status", "==", "pending").limit(5).get()
        for snap in docs:
            data = snap.to_dict()
            pid = data.get("proposal_id")
            action = data.get("action")
            if not pid or not action:
                snap.reference.update({"status": "invalid"})
                continue
            try:
                if action == "approve":
                    from app.proposals import approve_proposal
                    result = approve_proposal(pid)
                elif action == "reject":
                    from app.proposals import reject_proposal
                    result = reject_proposal(pid)
                elif action == "rollback":
                    result = f"Rollback #{pid} — use Signal for rollback"
                else:
                    result = f"Unknown action: {action}"
                snap.reference.update({"status": "done", "result": result[:200]})
                logger.info(f"firebase.publish: proposal action {action} #{pid}: {result[:100]}")
            except Exception as e:
                snap.reference.update({"status": "error", "error": str(e)[:200]})
    except Exception:
        logger.debug("firebase.publish: proposal actions poll failed", exc_info=True)


# ── Credit / Billing Alerts ────────────────────────────────────────────────

# Provider -> purchase URL mapping
_CREDIT_URLS = {
    "openrouter": "https://openrouter.ai/settings/credits",
    "anthropic":  "https://console.anthropic.com/settings/billing",
    "google":     "https://console.cloud.google.com/billing",
    "openai":     "https://platform.openai.com/settings/organization/billing",
}

# Patterns that indicate credit/billing exhaustion in error messages
_CREDIT_PATTERNS = [
    "402", "payment required", "insufficient", "credits", "quota",
    "billing", "afford", "exceeded", "budget", "out of credits",
    "rate_limit_exceeded", "insufficient_quota", "plan_limit",
]

_active_alerts: dict[str, dict] = {}  # provider -> alert dict


def detect_credit_error(error: Exception | str) -> str | None:
    """Check if an error indicates credit/billing exhaustion.

    Returns the provider name if a credit error is detected, else None.
    """
    err = str(error).lower()
    if not any(p in err for p in _CREDIT_PATTERNS):
        return None
    # Identify provider from error context
    if "openrouter" in err or "afford" in err:
        return "openrouter"
    if "anthropic" in err or "claude" in err:
        return "anthropic"
    if "google" in err or "gemini" in err or "vertex" in err:
        return "google"
    if "openai" in err:
        return "openai"
    # 402 without clear provider — check the numeric patterns
    if "402" in err:
        return "openrouter"  # most common 402 source
    return None


def report_credit_alert(provider: str, error_msg: str = "") -> None:
    """Report a credit exhaustion alert to Firestore for dashboard display."""
    url = _CREDIT_URLS.get(provider, "")
    alert = {
        "provider": provider,
        "error": error_msg[:300],
        "url": url,
        "ts": _now_iso(),
        "resolved": False,
    }
    _active_alerts[provider] = alert
    logger.warning(f"CREDIT ALERT: {provider} — {error_msg[:100]}")

    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("status").document("credit_alerts").set({
                "alerts": _active_alerts,
                "updated_at": _now_iso(),
            })
        except Exception:
            logger.debug("firebase.publish: credit alert write failed", exc_info=True)
    _fire(_write)


def resolve_credit_alert(provider: str) -> None:
    """Mark a provider's credit alert as resolved (successful call after error)."""
    if provider in _active_alerts:
        del _active_alerts[provider]
        def _write():
            db = _get_db()
            if not db:
                return
            try:
                db.collection("status").document("credit_alerts").set({
                    "alerts": _active_alerts,
                    "updated_at": _now_iso(),
                })
            except Exception:
                pass
        _fire(_write)


# ── Bidirectional chat (dashboard <-> Signal) ────────────────────────────────

def report_chat_message(role: str, text: str, source: str = "signal") -> None:
    """Write a chat message to Firestore so dashboard sees it in real time.

    Args:
        role:   "user" or "assistant"
        text:   message content
        source: "signal" or "dashboard" (where it originated)
    """
    def _write():
        db = _get_db()
        if not db:
            return
        try:
            db.collection("chat_messages").add({
                "role": role,
                "text": text[:4000],
                "source": source,
                "ts": _now_iso(),
            })
            # Trim to last 100 messages to prevent unbounded growth
            _trim_chat_messages(db)
        except Exception:
            logger.debug("firebase.publish: chat message write failed", exc_info=True)
    _fire(_write)


def report_system_monitor() -> None:
    """Report comprehensive system architecture status to Firestore.

    Aggregates status from ALL subsystems into a single document
    for the dashboard System Architecture Monitor section.
    """
    def _write():
        db = _get_db()
        if not db:
            return

        monitor = {"updated_at": _now_iso(), "subsystems": {}, "summary": {}}

        # ── 1. Core Infrastructure ──────────────────────────────────
        try:
            from app.config import get_settings
            s = get_settings()
            monitor["subsystems"]["core"] = {
                "status": "ok", "label": "Core Infrastructure",
                "modules": ["config", "main", "commander", "conversation_store",
                             "llm_factory", "signal_client", "security"],
                "details": {
                    "llm_mode": s.llm_mode, "cost_mode": s.cost_mode,
                    "vetting": s.vetting_enabled,
                },
            }
        except Exception as e:
            monitor["subsystems"]["core"] = {"status": "error", "error": str(e)[:100]}

        # ── 2. Self-Awareness ───────────────────────────────────────
        try:
            from app.self_awareness.inspect_tools import inspect_self_model, inspect_runtime
            sm = inspect_self_model()
            rt = inspect_runtime(section="process")
            from app.self_awareness.journal import get_journal
            j = get_journal()
            journal_count = j.count()
            monitor["subsystems"]["self_awareness"] = {
                "status": "ok", "label": "Self-Awareness",
                "modules": ["inspect_tools", "query_router", "grounding",
                             "knowledge_ingestion", "cogito", "journal",
                             "homeostasis", "self_model", "world_model"],
                "details": {
                    "chronicle_age_hours": sm.get("age_hours", "?"),
                    "chronicle_stale": sm.get("stale", True),
                    "uptime_seconds": rt.get("uptime_seconds", 0),
                    "journal_entries": sum(journal_count.values()),
                    "journal_breakdown": journal_count,
                },
            }
        except Exception as e:
            monitor["subsystems"]["self_awareness"] = {"status": "error", "error": str(e)[:100]}

        # ── 3. Feedback Loop ────────────────────────────────────────
        try:
            from app.prompt_registry import get_prompt_versions_map
            versions = get_prompt_versions_map()
            monitor["subsystems"]["feedback_loop"] = {
                "status": "ok", "label": "Feedback Loop",
                "modules": ["feedback_pipeline", "modification_engine", "eval_sandbox",
                             "safety_guardian", "implicit_feedback", "meta_learning",
                             "prompt_registry"],
                "details": {
                    "prompt_versions": {k: f"v{v:03d}" for k, v in versions.items()},
                    "roles_tracked": len(versions),
                    "feedback_enabled": s.feedback_enabled if 's' in dir() else True,
                    "modification_enabled": s.modification_enabled if 's' in dir() else True,
                },
            }
        except Exception as e:
            monitor["subsystems"]["feedback_loop"] = {"status": "error", "error": str(e)[:100]}

        # ── 4. ATLAS ────────────────────────────────────────────────
        try:
            from app.atlas.skill_library import get_library
            from app.atlas.competence_tracker import get_tracker
            from app.atlas.api_scout import get_scout
            lib = get_library()
            tracker = get_tracker()
            scout = get_scout()
            skill_summary = lib.get_competence_summary()
            monitor["subsystems"]["atlas"] = {
                "status": "ok", "label": "ATLAS",
                "modules": ["skill_library", "auth_patterns", "api_scout",
                             "code_forge", "competence_tracker", "video_learner",
                             "learning_planner", "audit_log"],
                "details": {
                    "total_skills": skill_summary.get("total_skills", 0),
                    "high_confidence": skill_summary.get("high_confidence", 0),
                    "stale_skills": skill_summary.get("stale", 0),
                    "known_apis": len(scout.get_known_apis()),
                    "competence_strengths": len(tracker.get_strengths()),
                    "competence_gaps": len(tracker.get_gaps()),
                },
            }
        except Exception as e:
            monitor["subsystems"]["atlas"] = {"status": "error", "error": str(e)[:100]}

        # ── 5. Evolution ────────────────────────────────────────────
        try:
            from app.adaptive_ensemble import get_controller
            ctrl = get_controller()
            stats = ctrl.get_stats()
            from app.auto_deployer import PROTECTED_FILES
            monitor["subsystems"]["evolution"] = {
                "status": "ok", "label": "Evolution",
                "modules": ["evolution", "island_evolution", "parallel_evolution",
                             "evolve_blocks", "adaptive_ensemble", "map_elites",
                             "cascade_evaluator"],
                "details": {
                    "ensemble_phase": stats["ensemble"]["phase"],
                    "exploration_rate": round(stats["exploration_rate"], 2),
                    "epoch": stats["epoch"],
                },
            }
        except Exception as e:
            monitor["subsystems"]["evolution"] = {"status": "error", "error": str(e)[:100]}

        # ── 6. Agent Zero Amendments ────────────────────────────────
        try:
            from app.lifecycle_hooks import get_registry
            hooks = get_registry().list_hooks()
            immutable_count = sum(1 for h in hooks if h["immutable"])
            from app.project_isolation import get_manager
            pm = get_manager()
            active_project = pm.active
            monitor["subsystems"]["amendments"] = {
                "status": "ok", "label": "Agent Zero Amendments",
                "modules": ["history_compression", "lifecycle_hooks",
                             "project_isolation", "control_plane"],
                "details": {
                    "hooks_total": len(hooks),
                    "hooks_immutable": immutable_count,
                    "hooks_list": [{"name": h["name"], "point": h["hook_point"],
                                    "priority": h["priority"], "immutable": h["immutable"]}
                                   for h in hooks],
                    "active_project": active_project.name if active_project else None,
                    "projects": [p["name"] for p in pm.list_projects()],
                },
            }
        except Exception as e:
            monitor["subsystems"]["amendments"] = {"status": "error", "error": str(e)[:100]}

        # ── 7. Fast Deploy ──────────────────────────────────────────
        try:
            from app.version_manifest import get_current_manifest, list_manifests
            from app.health_monitor import get_monitor
            manifest = get_current_manifest()
            health = get_monitor().get_health_state()
            monitor["subsystems"]["fast_deploy"] = {
                "status": "ok", "label": "Fast Deploy",
                "modules": ["version_manifest", "health_monitor", "self_healer",
                             "reference_tasks", "sandbox_runner"],
                "details": {
                    "current_version": manifest.get("version", "?") if manifest else "none",
                    "manifests_count": len(list_manifests()),
                    "health_sample_size": health.sample_size,
                    "error_rate": round(health.error_rate, 3),
                },
            }
        except Exception as e:
            monitor["subsystems"]["fast_deploy"] = {"status": "error", "error": str(e)[:100]}

        # ── 8. Training Pipeline ────────────────────────────────────
        try:
            monitor["subsystems"]["training"] = {
                "status": "ok", "label": "Self-Training LLM",
                "modules": ["training_collector", "training_pipeline"],
                "details": {},
            }
            # Try to get stats from DB
            try:
                from app.config import get_settings as _gs
                _s = _gs()
                if _s.mem0_postgres_url:
                    import psycopg2
                    conn = psycopg2.connect(_s.mem0_postgres_url)
                    with conn.cursor() as cur:
                        cur.execute("SELECT count(*) FROM training.interactions")
                        count = cur.fetchone()[0]
                        cur.execute("SELECT count(*) FROM training.runs")
                        runs = cur.fetchone()[0]
                    conn.close()
                    monitor["subsystems"]["training"]["details"] = {
                        "interactions_collected": count,
                        "training_runs": runs,
                    }
            except Exception:
                pass
        except Exception as e:
            monitor["subsystems"]["training"] = {"status": "error", "error": str(e)[:100]}

        # ── 9. Host Bridge ──────────────────────────────────────────
        try:
            bridge_available = False
            try:
                import httpx
                resp = httpx.get("http://host.docker.internal:9100/health", timeout=3)
                if resp.status_code == 200:
                    bridge_data = resp.json()
                    bridge_available = bridge_data.get("status") == "ok"
            except Exception:
                pass
            monitor["subsystems"]["bridge"] = {
                "status": "ok" if bridge_available else "offline",
                "label": "Host Bridge",
                "modules": ["bridge_client", "bridge_tools"],
                "details": {"bridge_reachable": bridge_available},
            }
        except Exception as e:
            monitor["subsystems"]["bridge"] = {"status": "error", "error": str(e)[:100]}

        # ── 10. Knowledge Stores ────────────────────────────────────
        try:
            monitor["subsystems"]["knowledge"] = {
                "status": "ok", "label": "Knowledge Stores",
                "modules": ["philosophy_vectorstore", "fiction_inspiration"],
                "details": {},
            }
        except Exception:
            pass

        # ── 11. Safety & Vetting ────────────────────────────────────
        try:
            from app.auto_deployer import PROTECTED_FILES
            monitor["subsystems"]["safety"] = {
                "status": "ok", "label": "Safety & Vetting",
                "modules": ["vetting", "auto_deployer", "security", "rate_throttle"],
                "details": {
                    "protected_files": len(PROTECTED_FILES),
                    "protected_list": sorted(list(PROTECTED_FILES))[:20],
                },
            }
        except Exception as e:
            monitor["subsystems"]["safety"] = {"status": "error", "error": str(e)[:100]}

        # ── 12. Idle Scheduler ──────────────────────────────────────
        try:
            from app.idle_scheduler import _default_jobs, is_enabled, is_idle
            jobs = _default_jobs()
            monitor["subsystems"]["scheduler"] = {
                "status": "ok", "label": "Idle Scheduler",
                "modules": ["idle_scheduler"],
                "details": {
                    "total_jobs": len(jobs),
                    "enabled": is_enabled(),
                    "idle": is_idle(),
                    "job_names": [j[0] for j in jobs],
                },
            }
        except Exception as e:
            monitor["subsystems"]["scheduler"] = {"status": "error", "error": str(e)[:100]}

        # ── 13. Database ────────────────────────────────────────────
        try:
            from app.config import get_settings as _gs2
            _s2 = _gs2()
            if _s2.mem0_postgres_url:
                import psycopg2
                conn = psycopg2.connect(_s2.mem0_postgres_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT schemaname, count(tablename)
                        FROM pg_tables
                        WHERE schemaname NOT IN ('pg_catalog','information_schema','pg_toast','public')
                        GROUP BY schemaname
                    """)
                    schemas = {r[0]: r[1] for r in cur.fetchall()}
                conn.close()
                monitor["subsystems"]["database"] = {
                    "status": "ok", "label": "Database",
                    "modules": ["postgresql", "chromadb", "neo4j"],
                    "details": {"schemas": schemas},
                }
        except Exception as e:
            monitor["subsystems"]["database"] = {"status": "error", "error": str(e)[:100]}

        # ── Summary ─────────────────────────────────────────────────
        total = len(monitor["subsystems"])
        ok = sum(1 for s in monitor["subsystems"].values() if s.get("status") == "ok")
        errors = sum(1 for s in monitor["subsystems"].values() if s.get("status") == "error")
        offline = sum(1 for s in monitor["subsystems"].values() if s.get("status") == "offline")
        all_modules = []
        for ss in monitor["subsystems"].values():
            all_modules.extend(ss.get("modules", []))
        monitor["summary"] = {
            "subsystems_total": total, "subsystems_ok": ok,
            "subsystems_error": errors, "subsystems_offline": offline,
            "total_modules": len(all_modules),
        }

        db.collection("status").document("system_monitor").set(monitor)
        logger.debug(f"firebase.publish: system_monitor updated ({ok}/{total} subsystems ok, {len(all_modules)} modules)")

    _fire(_write)


def _trim_chat_messages(db, max_messages: int = 100) -> None:
    """Keep only the most recent max_messages in chat_messages."""
    try:
        from google.cloud.firestore_v1 import Query
        docs = (
            db.collection("chat_messages")
            .order_by("ts", direction=Query.DESCENDING)
            .offset(max_messages)
            .limit(50)
            .stream()
        )
        for doc in docs:
            doc.reference.delete()
    except Exception:
        pass


# ── Skills inventory (alias) ─────────────────────────────────────────────────
# report_skills_inventory is an alias used in the user's spec
report_skills_inventory = report_skills
