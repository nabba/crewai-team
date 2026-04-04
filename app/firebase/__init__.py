"""
app.firebase — Firebase/Firestore integration package.

Re-exports all public names so that:
  from app.firebase import X
works for any name that was previously available via:
  from app.firebase_reporter import X
"""

# ── Infrastructure ───────────────────────────────────────────────────────────
from app.firebase.infra import (
    _get_db,
    _fire,
    _now_iso,
    _add_activity,
    _prune_activities,
    _executor,
)

# ── Publish (report_* functions, credit alerts, chat) ────────────────────────
from app.firebase.publish import (
    report_system_online,
    report_system_offline,
    heartbeat,
    report_skills,
    report_skills_inventory,
    report_signal_status,
    report_proposals,
    report_fleet_status,
    report_schedule,
    report_llm_mode,
    report_metrics,
    report_circuit_breakers,
    report_errors,
    report_evolution,
    report_request_costs,
    report_catalog,
    report_knowledge_base,
    report_evolution_stats,
    report_philosophy_kb,
    report_fiction_library,
    report_ecological_stats,
    report_token_stats,
    report_anomalies,
    report_variants,
    report_tech_radar,
    report_deploys,
    report_proposal_actions,
    report_chat_message,
    report_system_monitor,
    report_credit_alert,
    resolve_credit_alert,
    detect_credit_error,
    _active_alerts,
    _trim_chat_messages,
    _CREDIT_URLS,
    _CREDIT_PATTERNS,
)

# ── Listeners (pollers, mode listener, Firestore reads) ─────────────────────
from app.firebase.listeners import (
    read_llm_mode_from_firestore,
    start_mode_listener,
    start_kb_queue_poller,
    start_phil_queue_poller,
    start_fiction_queue_poller,
    start_chat_inbox_poller,
)

# ── Crew tracking (lifecycle events) ────────────────────────────────────────
from app.firebase.crew_tracking import (
    crew_started,
    crew_completed,
    crew_failed,
    update_eta,
    task_delegated,
    update_sub_agent_progress,
    cleanup_stale_tasks,
)
