"""Proactive monitors — close the silent-failure gaps for years-of-uptime.

Monitors run periodically in a daemon thread started at import time.
None of them mutate code or schedule (with the exception of opt-in
gated runners like ``db_backup`` and ``db_vacuum``); they observe and
alert on conditions the existing reactive runbooks can't see (because
nothing throws).

Currently registered monitors:

  * ``disk_quota``               — free disk below threshold → Signal.
  * ``listener_heartbeat``       — Firestore polling threads gone silent.
  * ``cron_liveness``            — scheduled jobs that haven't fired on time.
  * ``vendor_sunset``            — provider models flagged deprecated; files CRs.
  * ``idle_cooldown``            — idle-scheduler job stuck in cooldown >24 h.
  * ``audit_chain_check``        — re-verify the hash-chained audit JSONL daily.
  * ``lock_housekeeper``         — sweep stale lock files weekly.
  * ``adapter_lifecycle``        — orphan / dead-pointer / bloat in workspace/lora.
  * ``retention_chromadb`` etc.  — bounded growth on KB indices, worktrees, attachments.
  * ``signal_heartbeat``         — multi-channel escalation if Signal is wedged.
  * ``db_vacuum``                — monthly conversations.db VACUUM (Wave 0/1 #A6).
  * ``log_archival``             — daily errors.jsonl + audit_journal rotate (Wave 0/1 #A5).
  * ``db_backup``                — opt-in weekly Postgres+Neo4j+ChromaDB (Wave 0/1 #A1).
  * ``crypto_rotation_drill``    — weekly probe; missing/stale pins + readiness drill (§2.1).
  * ``chromadb_hygiene``         — quarterly SQLite VACUUM on every chroma.sqlite3; PROGRAM §40 Item 10.

The driver runs each monitor on its own cadence inside a single daemon
thread. Failure in one monitor never breaks the others — every step is
wrapped in try/except. The thread waits for a generous warm-up after
process start so it doesn't fight boot.

Master switch: ``HEALING_MONITORS_ENABLED`` (defaults ON; set ``false``
to disable the entire driver). The runbook framework's master switch
(``ERROR_RUNBOOKS_ENABLED``) is independent — monitors can run without
runbooks being on.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("HEALING_MONITORS_ENABLED", "true").lower() in (
        "true", "1", "yes",
    )


# Per-monitor cadence. Slow defaults: monitors are alerts, not real-time.
_DEFAULT_CADENCE_S = {
    "disk_quota": 300,           # 5 min — disk fills slow
    "listener_heartbeat": 600,   # 10 min — match the dashboard refresh
    "cron_liveness": 1800,       # 30 min — cron jobs are O(hour) cadence
    "vendor_sunset": 7 * 86400,  # weekly
    "idle_cooldown": 3600,       # hourly — cooldowns last hours
    "audit_chain_check": 23 * 3600,    # ~daily — Wave 1 (#5)
    "lock_housekeeper": 6 * 3600,      # 6h — Wave 1 (#9)
    "adapter_lifecycle": 24 * 3600,    # daily probe — internal cadence 30 d; Wave 2 (#4)
    "retention_chromadb": 24 * 3600,   # daily probe — internal cadence weekly; Wave 2 (#8)
    "retention_worktrees": 12 * 3600,  # twice daily probe — internal cadence daily; Wave 2 (#8)
    "retention_attachments": 12 * 3600,  # twice daily probe — internal cadence daily; Wave 2 (#8)
    "signal_heartbeat": 12 * 3600,     # twice daily probe — internal cadence daily; Wave 2 (#3)
    "db_vacuum": 24 * 3600,            # daily probe — internal cadence 30 d; Wave 0/1 (#A6)
    "log_archival": 6 * 3600,          # 6 h probe — internal cadence daily; Wave 0/1 (#A5)
    "db_backup": 6 * 3600,             # 6 h probe — internal cadence weekly; Wave 0/1 (#A1)
    "silent_regression_detector": 4 * 3600,  # 4 h probe — Phase C #2 (2026-05-09)
    "pattern_learner": 24 * 3600,            # daily probe — Phase C #4 (2026-05-09)
    "llm_output_drift": 24 * 3600,           # daily probe — Phase D #6 (2026-05-09)
    "signal_keepalive": 24 * 3600,           # daily probe — internal 30-day gate; Phase H #2
    "restore_drill": 24 * 3600,              # daily probe — alerts at 100d stale; Phase H #1
    "version_upgrade_drill": 24 * 3600,      # daily probe — alerts at 100d stale; §2.5
    "provider_contract_drift": 7 * 24 * 3600,  # weekly probe; §2.7
    "crypto_rotation_drill": 7 * 24 * 3600,    # weekly probe; §2.1
    "chromadb_hygiene": 24 * 3600,             # daily probe — internal 90-day cadence; PROGRAM §40 Item 10
    "notify_suppression_review": 6 * 3600,     # 6h probe — internal 7d cadence; PROGRAM §41 Item 17
    "drill_staleness": 24 * 3600,              # daily probe; alerts when any drill past cadence+grace; PROGRAM §44.2 Q6.2
    "backup_freshness": 24 * 3600,             # daily probe; alerts when local DR tarball > 14d old; PROGRAM §44.5 Q6.5 P2#3
    "architecture_adoption": 24 * 3600,        # daily probe; proposes rollback CR for unused subsystems; PROGRAM §45.1 Q7.1
    "migration_drill": 24 * 3600,              # daily probe; alerts at 100d stale; PROGRAM §48 Q13.1
    "tz_drift": 24 * 3600,                     # daily probe; alerts on hand-rolled vs zoneinfo divergence; PROGRAM §48 Q13.3
    "identity_drift_digest": 24 * 3600,        # daily probe; internal monthly cadence; PROGRAM §49 Q14.1
    "feedback_loop_drift": 24 * 3600,          # daily probe; internal weekly cadence; PROGRAM §49 Q14.2
    "embedding_drift": 24 * 3600,              # daily probe; internal weekly cadence; PROGRAM §49 Q14.4
    "interest_ossification": 24 * 3600,        # daily probe; internal weekly cadence; PROGRAM §49 Q14.5
    "lock_contention": 24 * 3600,              # daily probe; internal weekly cadence; PROGRAM §49 Q14.6
    "host_substrate_health": 24 * 3600,        # daily probe; internal weekly trend; PROGRAM §51 Q16 Theme 1
    "oauth_token_freshness": 24 * 3600,        # daily probe; internal weekly cadence; PROGRAM §51 Q16 Theme 2
    "operator_anomaly": 24 * 3600,             # daily probe; internal weekly cadence; PROGRAM §51 Q16 Theme 3
    "wiki_staleness": 24 * 3600,               # daily probe; internal weekly cadence; PROGRAM §51 Q16 Theme 5
    "claude_md_compaction": 24 * 3600,         # daily probe; internal annual cadence; PROGRAM §51 Q16 Theme 5
    "latency_slo": 24 * 3600,                  # daily probe; internal weekly cadence; PROGRAM §51 Q16 Theme 6.1
    "answer_regression": 24 * 3600,            # daily probe; internal 90-day cadence; PROGRAM §51 Q16 Theme 6.2
    "goal_progress": 24 * 3600,                # daily probe; PROGRAM §51 Q16 Theme 7.2
    "annual_privacy_review": 24 * 3600,        # daily probe; internal 330-day cadence; PROGRAM §51 Q16 Theme 7.3
    "philosophy_digest": 24 * 3600,            # daily probe; internal quarterly cadence; PROGRAM §51 Q16 Theme 8.2
    "hot1_outcome_reconciler": 24 * 3600,      # daily probe; internal weekly cadence; PROGRAM §51 Q16.1 Item 2
    "velocity_digest": 24 * 3600,              # daily probe; internal quarterly cadence; PROGRAM §51 Q16.1 Item 9
    # PROGRAM §52 Q17 monitors.
    "bit_rot_scan": 24 * 3600,                 # daily probe; internal weekly cadence; Q17.3
    "kb_contradiction": 24 * 3600,             # daily probe; internal weekly cadence; Q17.6
    # PROGRAM §55 — ChromaDB integrity (35th monitor, 2026-05-17).
    "chromadb_integrity": 24 * 3600,           # daily probe; internal 23h cadence
    # 2026-05-18 — schema drift visibility (closes the gap behind the deliberate
    # "no auto-apply migrations" policy). Daily probe; internal weekly cadence.
    "schema_drift": 24 * 3600,
}

_WARMUP_S = 120  # don't run anything in the first 2 min after import.

_driver_started = False
_driver_lock = threading.Lock()
_stop_event = threading.Event()


def _run_one(name: str, fn: Callable[[], None]) -> None:
    started = time.monotonic()
    try:
        fn()
    except Exception:
        logger.debug("healing.monitors[%s]: raised", name, exc_info=True)
    elapsed = time.monotonic() - started
    if elapsed > 30:
        logger.info("healing.monitors[%s]: slow run %.1fs", name, elapsed)


def _driver() -> None:
    """Single daemon loop. Each monitor runs at its own cadence; the loop
    sleeps 30 s between checks so any individual monitor can fire on time
    without the driver being a tight CPU spinner.
    """
    # Warm-up so monitors don't compete with boot.
    if _stop_event.wait(_WARMUP_S):
        return

    # Lazy import each monitor so a broken one never prevents the others
    # from running. The handler bodies themselves are pure-Python with no
    # framework dependencies past os/pathlib/json/time — they should
    # always import successfully.
    monitors: list[tuple[str, Callable[[], None], int, float]] = []
    try:
        from app.healing.monitors import disk_quota
        monitors.append(("disk_quota", disk_quota.run, _DEFAULT_CADENCE_S["disk_quota"], 0.0))
    except Exception:
        logger.debug("monitors: disk_quota import failed", exc_info=True)
    try:
        from app.healing.monitors import listener_heartbeat
        monitors.append(("listener_heartbeat", listener_heartbeat.run, _DEFAULT_CADENCE_S["listener_heartbeat"], 0.0))
    except Exception:
        logger.debug("monitors: listener_heartbeat import failed", exc_info=True)
    try:
        from app.healing.monitors import cron_liveness
        monitors.append(("cron_liveness", cron_liveness.run, _DEFAULT_CADENCE_S["cron_liveness"], 0.0))
    except Exception:
        logger.debug("monitors: cron_liveness import failed", exc_info=True)
    try:
        from app.healing.monitors import vendor_sunset
        monitors.append(("vendor_sunset", vendor_sunset.run, _DEFAULT_CADENCE_S["vendor_sunset"], 0.0))
    except Exception:
        logger.debug("monitors: vendor_sunset import failed", exc_info=True)
    try:
        from app.healing.monitors import idle_cooldown
        monitors.append(("idle_cooldown", idle_cooldown.run, _DEFAULT_CADENCE_S["idle_cooldown"], 0.0))
    except Exception:
        logger.debug("monitors: idle_cooldown import failed", exc_info=True)
    try:
        from app.healing.monitors import audit_chain_check
        monitors.append(("audit_chain_check", audit_chain_check.run, _DEFAULT_CADENCE_S["audit_chain_check"], 0.0))
    except Exception:
        logger.debug("monitors: audit_chain_check import failed", exc_info=True)
    try:
        from app.healing.monitors import lock_housekeeper
        monitors.append(("lock_housekeeper", lock_housekeeper.run, _DEFAULT_CADENCE_S["lock_housekeeper"], 0.0))
    except Exception:
        logger.debug("monitors: lock_housekeeper import failed", exc_info=True)
    try:
        from app.training import adapter_lifecycle
        monitors.append(("adapter_lifecycle", adapter_lifecycle.run, _DEFAULT_CADENCE_S["adapter_lifecycle"], 0.0))
    except Exception:
        logger.debug("monitors: adapter_lifecycle import failed", exc_info=True)
    try:
        from app.healing.monitors import retention
        monitors.append(("retention_chromadb", retention.run_chromadb, _DEFAULT_CADENCE_S["retention_chromadb"], 0.0))
        monitors.append(("retention_worktrees", retention.run_worktrees, _DEFAULT_CADENCE_S["retention_worktrees"], 0.0))
        monitors.append(("retention_attachments", retention.run_attachments, _DEFAULT_CADENCE_S["retention_attachments"], 0.0))
    except Exception:
        logger.debug("monitors: retention import failed", exc_info=True)
    try:
        from app.healing.monitors import signal_heartbeat
        monitors.append(("signal_heartbeat", signal_heartbeat.run, _DEFAULT_CADENCE_S["signal_heartbeat"], 0.0))
    except Exception:
        logger.debug("monitors: signal_heartbeat import failed", exc_info=True)
    try:
        from app.healing.monitors import db_vacuum
        monitors.append(("db_vacuum", db_vacuum.run, _DEFAULT_CADENCE_S["db_vacuum"], 0.0))
    except Exception:
        logger.debug("monitors: db_vacuum import failed", exc_info=True)
    try:
        from app.healing.monitors import log_archival
        monitors.append(("log_archival", log_archival.run, _DEFAULT_CADENCE_S["log_archival"], 0.0))
    except Exception:
        logger.debug("monitors: log_archival import failed", exc_info=True)
    try:
        from app.healing.monitors import db_backup
        monitors.append(("db_backup", db_backup.run, _DEFAULT_CADENCE_S["db_backup"], 0.0))
    except Exception:
        logger.debug("monitors: db_backup import failed", exc_info=True)
    try:
        from app.healing import silent_regression_detector
        monitors.append((
            "silent_regression_detector", silent_regression_detector.run,
            _DEFAULT_CADENCE_S["silent_regression_detector"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: silent_regression_detector import failed", exc_info=True)
    try:
        from app.healing import pattern_learner
        monitors.append((
            "pattern_learner", pattern_learner.run,
            _DEFAULT_CADENCE_S["pattern_learner"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: pattern_learner import failed", exc_info=True)
    try:
        from app.healing import llm_output_drift
        monitors.append((
            "llm_output_drift", llm_output_drift.run,
            _DEFAULT_CADENCE_S["llm_output_drift"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: llm_output_drift import failed", exc_info=True)
    try:
        from app.healing.monitors import signal_keepalive
        monitors.append((
            "signal_keepalive", signal_keepalive.run,
            _DEFAULT_CADENCE_S["signal_keepalive"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: signal_keepalive import failed", exc_info=True)
    try:
        from app.healing.monitors import restore_drill
        monitors.append((
            "restore_drill", restore_drill.run,
            _DEFAULT_CADENCE_S["restore_drill"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: restore_drill import failed", exc_info=True)
    try:
        from app.healing.monitors import version_upgrade_drill
        monitors.append((
            "version_upgrade_drill",
            lambda: version_upgrade_drill.run(),  # accept default args
            _DEFAULT_CADENCE_S["version_upgrade_drill"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: version_upgrade_drill import failed", exc_info=True)
    try:
        from app.healing.monitors import provider_contract_drift
        monitors.append((
            "provider_contract_drift", provider_contract_drift.run,
            _DEFAULT_CADENCE_S["provider_contract_drift"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: provider_contract_drift import failed", exc_info=True,
        )
    try:
        from app.healing.monitors import crypto_rotation_drill
        monitors.append((
            "crypto_rotation_drill", crypto_rotation_drill.run,
            _DEFAULT_CADENCE_S["crypto_rotation_drill"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: crypto_rotation_drill import failed", exc_info=True,
        )
    try:
        from app.healing.monitors import chromadb_hygiene
        monitors.append((
            "chromadb_hygiene", chromadb_hygiene.run,
            _DEFAULT_CADENCE_S["chromadb_hygiene"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: chromadb_hygiene import failed", exc_info=True,
        )
    # PROGRAM §55 (2026-05-17) — chromadb_integrity (35th monitor). Daily
    # integrity_check + atomic snapshot; quarantine + auto-replay on damage.
    try:
        from app.healing.monitors import chromadb_integrity
        monitors.append((
            "chromadb_integrity", chromadb_integrity.run,
            _DEFAULT_CADENCE_S["chromadb_integrity"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: chromadb_integrity import failed", exc_info=True,
        )
    try:
        from app.healing.monitors import notify_suppression_review
        monitors.append((
            "notify_suppression_review", notify_suppression_review.run,
            _DEFAULT_CADENCE_S["notify_suppression_review"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: notify_suppression_review import failed", exc_info=True,
        )
    # PROGRAM §44.2 Q6.2 — alerts when any resilience drill is past
    # its cadence + grace window. Bridges Q6 drill subsystem into the
    # healing-monitor cadence.
    try:
        from app.healing.monitors import drill_staleness
        monitors.append((
            "drill_staleness", drill_staleness.run,
            _DEFAULT_CADENCE_S["drill_staleness"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: drill_staleness import failed", exc_info=True,
        )
    # PROGRAM §44.5 Q6.5 P2#3 — alerts when the local DR tarball is
    # stale (proxy for "backup-sync script died"). Catches the most
    # common failure mode without needing cloud SDKs.
    try:
        from app.healing.monitors import backup_freshness
        monitors.append((
            "backup_freshness", backup_freshness.run,
            _DEFAULT_CADENCE_S["backup_freshness"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: backup_freshness import failed", exc_info=True,
        )
    # PROGRAM §45.1 Q7.1 — proposes rollback CRs for architecture
    # requests that have been COMPLETED for 30+ days but show
    # zero/low adoption signal. Never auto-applies — operator gate
    # intact through the normal CR review flow.
    try:
        from app.healing.monitors import architecture_adoption
        monitors.append((
            "architecture_adoption", architecture_adoption.run,
            _DEFAULT_CADENCE_S["architecture_adoption"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: architecture_adoption import failed", exc_info=True,
        )
    # Q2 §39: structured-diagnosis confidence-threshold auto-tuner.
    # The function has its own 24h gate inside; hourly cadence here
    # just gives us responsiveness on operator-driven settings flips.
    try:
        from app.healing import diagnosis_auto_tune
        monitors.append((
            "diagnosis_auto_tune", diagnosis_auto_tune.maybe_adjust_threshold,
            3600, 0.0,  # hourly probe; internal 24h cadence
        ))
    except Exception:
        logger.debug(
            "monitors: diagnosis_auto_tune import failed", exc_info=True,
        )
    # PROGRAM §48 Q13.1 — alerts when bash deploy/scripts/migration-drill.sh
    # has not been run on quarterly cadence. Catches the silent failure
    # mode "today's code can't read a 6-month-old backup".
    try:
        from app.healing.monitors import migration_drill
        monitors.append((
            "migration_drill", migration_drill.run,
            _DEFAULT_CADENCE_S["migration_drill"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: migration_drill import failed", exc_info=True)
    # PROGRAM §48 Q13.3 — daily probe comparing hand-rolled
    # _helsinki_tz() vs ZoneInfo("Europe/Helsinki"). Catches EU
    # DST abolition + stale host tzdata. On first material
    # divergence files a regular CR proposing consolidation.
    try:
        from app.healing.monitors import tz_drift
        monitors.append((
            "tz_drift", tz_drift.run,
            _DEFAULT_CADENCE_S["tz_drift"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: tz_drift import failed", exc_info=True)
    # PROGRAM §49 Q14.1 — rolling identity-drift digest. Daily
    # probe, internal monthly cadence. Alerts when 30d amendment
    # count exceeds 2× the annualised baseline.
    try:
        from app.identity import drift_digest
        monitors.append((
            "identity_drift_digest", drift_digest.run,
            _DEFAULT_CADENCE_S["identity_drift_digest"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: identity_drift_digest import failed", exc_info=True,
        )
    # PROGRAM §49 Q14.2 — meta-agent recipe-selection Gini probe.
    # Daily probe, internal weekly cadence. Alerts when Gini trends
    # monotonically up over 4+ weeks (closed-loop convergence).
    try:
        from app.healing.monitors import feedback_loop_drift
        monitors.append((
            "feedback_loop_drift", feedback_loop_drift.run,
            _DEFAULT_CADENCE_S["feedback_loop_drift"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: feedback_loop_drift import failed", exc_info=True,
        )
    # PROGRAM §49 Q14.4 — embedding-model fingerprint drift. Daily
    # probe, internal weekly cadence. Re-embeds 20 anchor queries +
    # compares to baseline; alerts on silent vendor model swap.
    try:
        from app.healing.monitors import embedding_drift
        monitors.append((
            "embedding_drift", embedding_drift.run,
            _DEFAULT_CADENCE_S["embedding_drift"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: embedding_drift import failed", exc_info=True,
        )
    # PROGRAM §49 Q14.5 — interest-model ossification. Daily probe,
    # internal weekly cadence. Shannon entropy + Jaccard churn on
    # the interest_model top-30.
    try:
        from app.healing.monitors import interest_ossification
        monitors.append((
            "interest_ossification", interest_ossification.run,
            _DEFAULT_CADENCE_S["interest_ossification"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: interest_ossification import failed", exc_info=True,
        )
    # PROGRAM §49 Q14.6 — live-process lock contention. Daily
    # probe, internal weekly cadence. Reads workspace/healing/
    # lock_waits.jsonl (recorded passively by safe_io); computes
    # p99 per resource.
    try:
        from app.healing.monitors import lock_contention
        monitors.append((
            "lock_contention", lock_contention.run,
            _DEFAULT_CADENCE_S["lock_contention"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: lock_contention import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 1 — host-substrate health probe.
    # Daily probe, internal weekly trend computation. Watches the
    # host itself: free-space trend (days-until-full projection),
    # workspace bytes growth, gateway restart bursts, uptime
    # staleness, Linux memory headroom. Reads-only; surfaces an
    # optional out-of-band host-side telemetry file.
    try:
        from app.healing.monitors import host_substrate_health
        monitors.append((
            "host_substrate_health", host_substrate_health.run,
            _DEFAULT_CADENCE_S["host_substrate_health"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: host_substrate_health import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 2 — vendor-credential freshness probe.
    # Daily probe, internal weekly cadence. Pure file inspection:
    # Google Workspace refresh token age, vendor API key format
    # patterns (Anthropic / OpenAI / OpenRouter / Groq), VAPID
    # keypair completeness. Never issues an external API call.
    try:
        from app.healing.monitors import oauth_token_freshness
        monitors.append((
            "oauth_token_freshness", oauth_token_freshness.run,
            _DEFAULT_CADENCE_S["oauth_token_freshness"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: oauth_token_freshness import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 3 (partial) — operator-pattern anomaly
    # detector. Reads workspace/audit.log request_received rows
    # (timestamps + lengths only, no message content); surfaces
    # hour-of-day shifts, cadence spikes/quiet, message-length
    # shifts, new-authorized-sender events. Observational only —
    # never blocks or refuses any action.
    try:
        from app.healing.monitors import operator_anomaly
        monitors.append((
            "operator_anomaly", operator_anomaly.run,
            _DEFAULT_CADENCE_S["operator_anomaly"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: operator_anomaly import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 5 — wiki staleness digest. Daily probe,
    # internal weekly cadence. Walks ``wiki/`` for markdown files
    # past 365-day mtime; surfaces 10 stalest in a Signal digest
    # per cycle, with per-file 90-day dedup. Excludes auto-archive
    # dirs (legacy / value_reflections / quarterly_reviews /
    # governance / archive).
    try:
        from app.healing.monitors import wiki_staleness
        monitors.append((
            "wiki_staleness", wiki_staleness.run,
            _DEFAULT_CADENCE_S["wiki_staleness"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: wiki_staleness import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 5 — CLAUDE.md compaction proposer.
    # Daily probe; internal annual cadence via _already_proposed_
    # this_year guard. Composes a structural-only KEEP/ARCHIVE split
    # into workspace/self_improvement/claude_md_compaction/<year>/
    # for operator review. Never auto-applies (CLAUDE.md often sits
    # outside the git repo).
    try:
        from app.self_improvement import claude_md_compaction
        monitors.append((
            "claude_md_compaction", claude_md_compaction.run_once,
            _DEFAULT_CADENCE_S["claude_md_compaction"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: claude_md_compaction import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 6.1 — latency SLO. Daily probe, weekly
    # internal cadence. Computes p50/p95/p99 from audit.log + alerts
    # on ≥2× baseline regressions.
    try:
        from app.healing.monitors import latency_slo
        monitors.append((
            "latency_slo", latency_slo.run,
            _DEFAULT_CADENCE_S["latency_slo"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: latency_slo import failed", exc_info=True)
    # PROGRAM §51 Q16 Theme 6.2 — answer regression suite. Daily
    # probe, internal 90-day cadence. Runs frozen Q-A pairs through
    # the cascade + (optionally) an LLM judge.
    try:
        from app.qos import answer_regression
        monitors.append((
            "answer_regression",
            lambda: answer_regression.run_regression() or None,
            _DEFAULT_CADENCE_S["answer_regression"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: answer_regression import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 7.2 — goal-progress probe.
    try:
        from app.companion import goal_progress
        monitors.append((
            "goal_progress", goal_progress.evaluate,
            _DEFAULT_CADENCE_S["goal_progress"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: goal_progress import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 7.3 — annual privacy review.
    try:
        from app.privacy import annual_review
        monitors.append((
            "annual_privacy_review", annual_review.run_once,
            _DEFAULT_CADENCE_S["annual_privacy_review"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: annual_privacy_review import failed", exc_info=True,
        )
    # PROGRAM §51 Q16 Theme 8.2 — philosophy panel quarterly digest.
    try:
        from app.philosophy import panel_digest
        monitors.append((
            "philosophy_digest", panel_digest.run_once,
            _DEFAULT_CADENCE_S["philosophy_digest"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: philosophy_digest import failed", exc_info=True,
        )
    # PROGRAM §51 Q16.1 Item 2 — HOT-1 outcome reconciler. Walks CR
    # audit log for terminal events on error_diagnosis CRs; writes
    # outcomes to a side overlay the consultation reader merges at
    # read time. Closes the "we emit but never read back" loop.
    try:
        from app.healing import hot1_outcome_reconciler
        monitors.append((
            "hot1_outcome_reconciler", hot1_outcome_reconciler.run,
            _DEFAULT_CADENCE_S["hot1_outcome_reconciler"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: hot1_outcome_reconciler import failed", exc_info=True,
        )
    # PROGRAM §51 Q16.1 Item 9 — velocity digest. Daily probe;
    # quarterly internal cadence. Surfaces Theme 4 velocity changes
    # in a Signal digest so operator doesn't have to poll the REST
    # endpoint manually.
    try:
        from app.self_improvement import velocity_digest
        monitors.append((
            "velocity_digest", velocity_digest.run_once,
            _DEFAULT_CADENCE_S["velocity_digest"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: velocity_digest import failed", exc_info=True,
        )
    # PROGRAM §52 Q17.3 — bit-rot scan over identity-critical JSONL
    # files. Daily probe; internal weekly cadence.
    try:
        from app.healing.monitors import bit_rot_scan
        monitors.append((
            "bit_rot_scan", bit_rot_scan.run,
            _DEFAULT_CADENCE_S["bit_rot_scan"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: bit_rot_scan import failed", exc_info=True,
        )
    # PROGRAM §52 Q17.6 — KB contradiction probe over epistemic
    # claims. Daily probe; internal weekly cadence.
    try:
        from app.healing.monitors import kb_contradiction
        monitors.append((
            "kb_contradiction", kb_contradiction.run,
            _DEFAULT_CADENCE_S["kb_contradiction"], 0.0,
        ))
    except Exception:
        logger.debug(
            "monitors: kb_contradiction import failed", exc_info=True,
        )
    # 2026-05-18 — schema_drift. Detects migrations/*.sql declarations
    # not reflected in information_schema. Visibility-only — the
    # change_request validator forbids auto-applying migrations/, so
    # this surfaces drift to the operator who runs psql manually.
    try:
        from app.healing.monitors import schema_drift
        monitors.append((
            "schema_drift", schema_drift.run,
            _DEFAULT_CADENCE_S["schema_drift"], 0.0,
        ))
    except Exception:
        logger.debug("monitors: schema_drift import failed", exc_info=True)

    if not monitors:
        logger.warning("healing.monitors: no monitors loaded; driver exiting")
        return

    logger.info("healing.monitors: driver running %d monitors", len(monitors))

    # Mutable cadence + last-run state. Each tuple is replaced on each tick.
    state = [
        {"name": n, "fn": fn, "cadence": cadence, "last_run": 0.0}
        for n, fn, cadence, _ in monitors
    ]

    while not _stop_event.is_set():
        now = time.monotonic()
        for entry in state:
            if now - entry["last_run"] >= entry["cadence"]:
                _run_one(entry["name"], entry["fn"])
                entry["last_run"] = time.monotonic()
        # Sleep 30 s; lets cadences finer than 30 s run jittery but fine
        # for the slowest cadence (weekly) without burning CPU.
        if _stop_event.wait(30):
            return


_DAEMON_THREAD_NAME = "healing-monitors"


def _is_running() -> bool:
    """True iff a thread named ``_DAEMON_THREAD_NAME`` is currently alive."""
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


def start() -> None:
    """Start the daemon driver. Truly idempotent — checks thread liveness
    on every call, so the watchdog can call this to re-spawn after death
    and the call is safe even if another thread already restarted us.

    The previous implementation gated on a ``_driver_started`` flag that
    drifted out of sync when the thread died (flag stayed True, no restart
    was possible). The new path detects death directly via
    ``threading.enumerate()``.
    """
    global _driver_started
    if not _enabled():
        logger.info("healing.monitors: disabled via HEALING_MONITORS_ENABLED")
        return
    with _driver_lock:
        if _is_running():
            return  # already alive — nothing to do
        if _driver_started:
            logger.warning(
                "healing.monitors: previous daemon thread is dead, re-spawning"
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info("healing.monitors: daemon started (warm-up=%ds)", _WARMUP_S)


def stop() -> None:
    """Signal the driver to exit. Mostly used in tests."""
    _stop_event.set()


# Eager start on import. The warm-up + thread isolation make this safe
# even when the surrounding process is still booting.
start()
