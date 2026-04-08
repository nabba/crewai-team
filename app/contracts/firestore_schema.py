"""
firestore_schema.py — Documented Firestore collection contracts (ARCHITECTURAL REFERENCE).

NOTE: This file is documentation-as-code. It is not imported at runtime
by any module. It serves as the single source of truth for Firestore
schema documentation, referenced by developers and the dashboard.

This file serves as the single source of truth for all Firestore
collections, their field types, and which subsystems read/write them.
It replaces the implicit protocol that was previously spread across
code and Firestore collection conventions.

This is DOCUMENTATION, not enforcement — but it provides a reference
for any module that interacts with Firestore.
"""

# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS — Who writes, who reads, field types
# ═══════════════════════════════════════════════════════════════════════════

SCHEMA = {
    # ── System Status ─────────────────────────────────────────────────
    "status/system": {
        "writer": "firebase_reporter.report_system_online/offline/heartbeat",
        "reader": "dashboard (onSnapshot)",
        "fields": {
            "status": "str (online/offline)",
            "health_score": "float (0-1)",
            "version": "str",
            "started_at": "str (ISO)",
            "last_heartbeat": "str (ISO)",
            "uptime_seconds": "int",
        },
    },

    # ── Crew Status ───────────────────────────────────────────────────
    "crews/{name}": {
        "writer": "firebase_reporter.crew_started/completed/failed",
        "reader": "dashboard (onSnapshot per crew)",
        "fields": {
            "name": "str",
            "state": "str (idle/active)",
            "current_task": "str (max 300 chars)",
            "task_id": "str (uuid hex)",
            "started_at": "str (ISO)",
            "eta": "str (ISO) or null",
            "model": "str (LLM model name)",
            "last_updated": "str (ISO)",
        },
    },

    # ── Tasks ─────────────────────────────────────────────────────────
    "tasks/{id}": {
        "writer": "firebase_reporter.crew_started/completed/failed",
        "reader": "dashboard (onSnapshot, ordered by started_at desc, limit 20)",
        "fields": {
            "id": "str (uuid hex)",
            "crew": "str",
            "summary": "str (max 4000 chars)",
            "state": "str (running/completed/failed)",
            "started_at": "str (ISO)",
            "completed_at": "str (ISO) or null",
            "result_preview": "str (max 4000 chars) or null",
            "tokens_used": "int",
            "cost_usd": "float",
            "model": "str",
            "parent_task_id": "str or null",
            "is_sub_agent": "bool",
        },
    },

    # ── Config (Dashboard ↔ Runtime) ──────────────────────────────────
    "config/llm": {
        "writer": "dashboard toggle / firebase_reporter.report_llm_mode",
        "reader": "firebase_reporter.start_mode_listener",
        "fields": {"mode": "str (local/hybrid/cloud/insane)"},
    },
    "config/background_tasks": {
        "writer": "dashboard toggle",
        "reader": "idle_scheduler.start_background_listener",
        "fields": {"enabled": "bool"},
    },

    # ── Queues (Dashboard → Backend) ──────────────────────────────────
    "kb_queue/{id}": {
        "writer": "dashboard upload",
        "reader": "firebase_reporter.start_kb_queue_poller (10s interval)",
        "fields": {
            "filename": "str",
            "content": "str (base64 or text, max 2MB)",
            "category": "str",
            "status": "str (pending/processing/done/error)",
        },
    },
    "phil_queue/{id}": {
        "writer": "dashboard upload",
        "reader": "firebase_reporter.start_phil_queue_poller (10s interval)",
        "fields": {
            "filename": "str",
            "content": "str (text)",
            "status": "str (pending/processing/done/error)",
        },
    },
    "fiction_queue/{id}": {
        "writer": "dashboard upload",
        "reader": "firebase_reporter.start_fiction_queue_poller (10s interval)",
        "fields": {
            "filename": "str",
            "content": "str (text)",
            "author": "str",
            "title": "str",
            "themes": "list[str]",
            "status": "str (pending/processing/done/error)",
        },
    },
    "chat_inbox/{id}": {
        "writer": "dashboard chat input",
        "reader": "firebase_reporter.start_chat_inbox_poller (3s interval)",
        "fields": {
            "text": "str (max 4000 chars)",
            "status": "str (pending/processing/done/error)",
            "ts": "str (server timestamp)",
        },
    },

    # ── Dashboard Data (Backend → Dashboard) ──────────────────────────
    "status/metrics": {"writer": "heartbeat", "reader": "dashboard"},
    "status/circuit_breakers": {"writer": "heartbeat", "reader": "dashboard"},
    "status/errors": {"writer": "heartbeat", "reader": "dashboard"},
    "status/evolution": {"writer": "heartbeat", "reader": "dashboard"},
    "status/request_costs": {"writer": "heartbeat", "reader": "dashboard"},
    "status/catalog": {"writer": "heartbeat", "reader": "dashboard"},
    "status/skills": {"writer": "heartbeat", "reader": "dashboard"},
    "status/knowledge_base": {"writer": "heartbeat + KB poller", "reader": "dashboard"},
    "status/philosophy_kb": {"writer": "philosophy poller", "reader": "dashboard"},
    "status/fiction_library": {"writer": "fiction poller", "reader": "dashboard"},
    "status/token_stats": {"writer": "heartbeat", "reader": "dashboard"},
    "status/anomalies": {"writer": "heartbeat", "reader": "dashboard"},
    "status/variants": {"writer": "heartbeat", "reader": "dashboard"},
    "status/tech_radar": {"writer": "heartbeat", "reader": "dashboard"},
    "status/deploys": {"writer": "heartbeat", "reader": "dashboard"},
    "status/proposals": {"writer": "heartbeat", "reader": "dashboard"},
    "status/credit_alerts": {"writer": "rate_throttle / commander", "reader": "dashboard"},
    "status/system_monitor": {"writer": "report_system_monitor", "reader": "dashboard"},

    # ── Activity Feed ─────────────────────────────────────────────────
    "activities/{id}": {
        "writer": "firebase_reporter._add_activity",
        "reader": "dashboard (onSnapshot, ordered by ts desc, limit 50)",
        "fields": {
            "type": "str (task_started/task_completed/task_failed/etc.)",
            "crew": "str",
            "detail": "str (max 300 chars)",
            "task_id": "str",
            "ts": "str (ISO)",
        },
    },

    # ── Chat Messages ─────────────────────────────────────────────────
    "chat_messages/{id}": {
        "writer": "firebase_reporter.report_chat_message",
        "reader": "dashboard (onSnapshot)",
        "fields": {
            "role": "str (user/assistant)",
            "text": "str (max 4000 chars)",
            "source": "str (signal/dashboard)",
            "ts": "str (ISO)",
        },
    },
}
