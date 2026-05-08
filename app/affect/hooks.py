"""
hooks.py — Affect lifecycle hook handlers.

Registers handlers on the existing app.lifecycle_hooks bus. Phase 1
handlers:

    POST_LLM_CALL @ priority 9 (immutable)
        Compute affect from the just-produced internal_state, persist
        trace, run welfare check, audit any breaches.

    ON_COMPLETE @ priority 62
        Final affect snapshot for the episode + trace flush.

The handler runs AFTER the existing internal_state computation (priority 8)
so it sees the freshly populated state.somatic / state.certainty. It does
NOT mutate the state — it only observes, records, and audits.

Phase 2 will add PRE_LLM_CALL @ priority 7 to inject affect-modulated
sampling kwargs (in concert with llm_sampling.build_llm_kwargs extension).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_INSTALLED = False


def install() -> None:
    """Register affect handlers on the lifecycle hook bus + scheduled jobs.

    Idempotent. Safe to call multiple times.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    try:
        from app.lifecycle_hooks import HookContext, HookPoint, get_registry
    except Exception:
        logger.error("affect.hooks: lifecycle_hooks unavailable; affect not wired")
        return

    registry = get_registry()

    # ── POST_LLM_CALL: compute and persist affect ─────────────────────────
    def _affect_post_llm(ctx: HookContext) -> HookContext:
        try:
            from app.affect.core import compute_affect, recent_affect
            from app.affect.welfare import audit, check

            internal_state = ctx.metadata.get("_internal_state")
            affect_state, viability_frame = compute_affect(
                internal_state=internal_state,
                persist=True,
            )
            ctx.metadata["_affect_state"] = affect_state.to_dict()
            ctx.metadata["_viability_frame"] = viability_frame.to_dict()

            # Hard-envelope welfare check
            window = recent_affect(n=64)
            for breach in check(affect_state, viability_frame, window):
                audit(breach)
                # On a critical breach, surface as ctx error but DO NOT abort the
                # response — that would compound suffering. Logging + audit only.
                if breach.severity == "critical":
                    ctx.errors.append(f"welfare_breach: {breach.kind}")
        except Exception:
            logger.debug("affect.hooks: POST_LLM_CALL handler failed", exc_info=True)
        return ctx

    registry.register(
        "affect_post_llm", HookPoint.POST_LLM_CALL,
        _affect_post_llm,
        priority=9,                # runs right after internal_state (priority 8)
        immutable=True,            # infrastructure-level
        description="Compute affect (V/A/C), persist trace, welfare-check",
    )

    # ── PRE_TASK: runtime-state task-start tracking + decision-logged ─────
    def _affect_pre_task(ctx: HookContext) -> HookContext:
        try:
            from app.affect.runtime_state import task_started, decision_logged
            task_id = ctx.get("task_id", "") or ctx.metadata.get("task_id", "")
            role = ctx.metadata.get("agent_role", "") or ctx.agent_id or ""
            if task_id:
                task_started(str(task_id), role=str(role))
            # Default: every PRE_TASK is an agent-decided step. ON_DELEGATION
            # below overrides this for cases that get delegated.
            decision_logged(delegated=False)

            # Phase 3: if this is a user-initiated task, update the user OtherModel.
            sender_id = ctx.get("sender_id", "") or ctx.metadata.get("sender_id", "")
            if sender_id:
                try:
                    from app.affect.attachment import primary_user_identity, update_from_interaction
                    # Phase 3: a configured primary user gets its canonical identity;
                    # other senders fall back to a per-sender identity.
                    ident = (primary_user_identity()
                             if str(sender_id).lower() in (primary_user_identity().split(":")[1], "andrus")
                             else f"user:{str(sender_id)[:40]}")
                    update_from_interaction(
                        ident,
                        observed_valence=None,    # learn valence at episode end
                        note=(ctx.task_description or "")[:120],
                        interaction_kind="task_received",
                    )
                except Exception:
                    logger.debug("affect.hooks: user attachment update failed", exc_info=True)
        except Exception:
            logger.debug("affect.hooks: PRE_TASK handler failed", exc_info=True)
        return ctx

    registry.register(
        "affect_pre_task", HookPoint.PRE_TASK,
        _affect_pre_task,
        priority=29,    # after meta_cognitive (15) and other early hooks
        description="Affect runtime-state: task_started + autonomy bump",
    )

    # ── ON_DELEGATION: invert the autonomy signal + update peer OtherModel ─
    def _affect_on_delegation(ctx: HookContext) -> HookContext:
        try:
            from app.affect.runtime_state import decision_logged
            decision_logged(delegated=True)
            # Phase 3: log peer-agent interaction. Crew name approximates role.
            crew = ctx.metadata.get("crew", "") or ctx.metadata.get("delegated_to", "")
            if crew:
                try:
                    from app.affect.attachment import update_from_interaction
                    update_from_interaction(
                        f"peer:{crew}",
                        observed_valence=None,
                        note=f"delegated: {(ctx.task_description or '')[:80]}",
                        interaction_kind="delegation",
                    )
                except Exception:
                    logger.debug("affect.hooks: peer attachment update failed", exc_info=True)
        except Exception:
            logger.debug("affect.hooks: ON_DELEGATION handler failed", exc_info=True)
        return ctx

    registry.register(
        "affect_on_delegation", HookPoint.ON_DELEGATION,
        _affect_on_delegation,
        priority=72,
        description="Affect runtime-state: log delegation as not-agent-decided",
    )

    # ── ON_COMPLETE: episode-end snapshot + task-completed timing + KB tag ─
    def _affect_on_complete(ctx: HookContext) -> HookContext:
        try:
            from app.affect.core import latest_affect
            from app.affect.runtime_state import task_completed
            s = latest_affect()
            if s is not None:
                ctx.metadata["_affect_terminal"] = s.to_dict()

            task_id = ctx.get("task_id", "") or ctx.metadata.get("task_id", "")
            if task_id:
                task_completed(str(task_id))

            # Append affect metadata to experiential / tensions KBs.
            try:
                from app.affect.kb_metadata import tag_episode_with_affect
                tag_episode_with_affect(ctx, terminal_affect=s)
            except Exception:
                logger.debug("affect.hooks: kb_metadata write failed", exc_info=True)

            # Phase 3: feed the episode's terminal valence back into the user
            # OtherModel as observed sentiment, IF this was a user-initiated task.
            sender_id = ctx.get("sender_id", "") or ctx.metadata.get("sender_id", "")
            if sender_id and s is not None:
                try:
                    from app.affect.attachment import primary_user_identity, update_from_interaction
                    ident = (primary_user_identity()
                             if str(sender_id).lower() in (primary_user_identity().split(":")[1], "andrus")
                             else f"user:{str(sender_id)[:40]}")
                    update_from_interaction(
                        ident,
                        observed_valence=float(s.valence),
                        note=f"episode close · attractor={s.attractor}",
                        interaction_kind="episode_close",
                    )
                except Exception:
                    logger.debug("affect.hooks: user attachment close-update failed", exc_info=True)
        except Exception:
            logger.debug("affect.hooks: ON_COMPLETE handler failed", exc_info=True)
        return ctx

    registry.register(
        "affect_on_complete", HookPoint.ON_COMPLETE,
        _affect_on_complete,
        priority=62,
        description="Affect: terminal snapshot + task_completed + KB metadata",
    )

    # ── Scheduled daily reflection cycle (04:30 Helsinki) ─────────────────
    _install_reflection_schedule()

    # ── Narrative-Self schedule: Loop 2 quiet-flush + Loop 3 daily chapter ──
    _install_narrative_schedule()

    # ── One-shot 2-week health check (2026-05-12 04:45 Helsinki) ──────────
    _install_one_shot_health_check()

    _INSTALLED = True
    logger.info(
        "affect.hooks: installed (POST_LLM_CALL@9 immutable, ON_COMPLETE@62, "
        "daily reflection, narrative-self loops 2+3, one-shot health check)"
    )


def _install_reflection_schedule() -> None:
    """Hook the daily reflection cycle + L9 snapshot into the existing APScheduler.

    04:30 EET/EEST — daily reflection (replays last 24h, runs reference panel,
    proposes calibration delta under 6-guardrail flow).
    04:35 EET/EEST — L9 daily homeostasis snapshot (rolled-up affect stats +
    viability + welfare breach counts).

    APScheduler in main.py is already initialized at import time; this adds two
    jobs. Times are local; the temporal_context module handles DST shifts.
    """
    try:
        from app.main import scheduler
        from apscheduler.triggers.cron import CronTrigger
        from app.affect.calibration import run_reflection_cycle
        from app.affect.l9_snapshots import write_daily_snapshot

        scheduler.add_job(
            run_reflection_cycle,
            trigger=CronTrigger(hour=4, minute=30, timezone="Europe/Helsinki"),
            id="affect_reflection_cycle",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )
        scheduler.add_job(
            write_daily_snapshot,
            trigger=CronTrigger(hour=4, minute=35, timezone="Europe/Helsinki"),
            id="affect_l9_snapshot",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )
        logger.info(
            "affect.hooks: daily reflection scheduled at 04:30, L9 snapshot at 04:35 Europe/Helsinki"
        )
    except Exception:
        logger.debug("affect.hooks: scheduler not available; reflection unscheduled", exc_info=True)


def _install_narrative_schedule() -> None:
    """Schedule the Narrative-Self pipeline jobs.

    04:40 EET/EEST — daily chapter consolidation (Loop 3). Runs after the
    welfare reflection (04:30) and L9 snapshot (04:35) so the drift signal
    from the latest reflection report is available when chapters are written.

    Every 5 min — episode quiet-flush check (Loop 2). No-op when there are
    no pending salience events; otherwise flushes if the last salience event
    is older than QUIET_THRESHOLD_S (default 15 min).
    """
    try:
        from app.main import scheduler
        from apscheduler.triggers.cron import CronTrigger
        from app.affect.episodes import maybe_flush_quiet
        from app.affect.narrative import run_chapter_consolidation

        def _chapter_with_workspace_publish() -> None:
            """Wrap run_chapter_consolidation to publish a workspace summary
            on completion (consciousness-roadmap §3.G5).

            The chapter consolidator is read-only-by-Self-Improver per its
            docstring; the publish hook lives here in `hooks.py` rather than
            in `narrative.py` to preserve that boundary.
            """
            chapter = None
            try:
                chapter = run_chapter_consolidation()
            finally:
                try:
                    from app.workspace_publish import publish_to_workspace
                    if chapter is not None:
                        n_episodes = chapter.get("n_episodes", 0)
                        identity_claims = chapter.get("identity_claims") or []
                        publish_to_workspace(
                            source="narrative-chapter",
                            content=(
                                f"Daily chapter consolidated: "
                                f"{n_episodes} episode(s), "
                                f"{len(identity_claims)} active identity claim(s)"
                            ),
                            # Daily-rhythm signal: meaningful but not critical.
                            salience=0.55,
                            signal_type="disposition",
                        )
                except Exception:
                    logger.debug(
                        "affect.hooks: workspace publish for narrative-chapter failed",
                        exc_info=True,
                    )

        scheduler.add_job(
            _chapter_with_workspace_publish,
            trigger=CronTrigger(hour=4, minute=40, timezone="Europe/Helsinki"),
            id="affect_chapter_consolidation",
            replace_existing=True,
            misfire_grace_time=3600,
            coalesce=True,
        )
        scheduler.add_job(
            maybe_flush_quiet,
            trigger="interval", minutes=5,
            id="affect_episode_quiet_flush",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info(
            "affect.hooks: chapter consolidation scheduled at 04:40 Europe/Helsinki, "
            "episode quiet-flush every 5 min"
        )
    except Exception:
        logger.debug(
            "affect.hooks: scheduler not available; narrative-self unscheduled",
            exc_info=True,
        )


def _install_one_shot_health_check() -> None:
    """Register a one-shot DateTrigger for the 2-week narrative-self health check.

    Fires 2026-05-12 04:45 Europe/Helsinki — 5 min after the daily chapter
    consolidator (04:40) so the freshest chapter is included in the audit.

    Self-disabling: if the run date is already past at install time, the job
    is skipped (re-imports across restarts must not retroactively trigger
    a misfire).
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        logger.debug("affect.hooks: zoneinfo unavailable; health check unscheduled")
        return

    helsinki = ZoneInfo("Europe/Helsinki")
    run_at = datetime(2026, 5, 12, 4, 45, tzinfo=helsinki)
    if datetime.now(helsinki) >= run_at:
        return  # past the scheduled time

    try:
        from app.main import scheduler
        from apscheduler.triggers.date import DateTrigger
        from app.affect.health_check import run_health_check

        scheduler.add_job(
            run_health_check,
            trigger=DateTrigger(run_date=run_at),
            id="affect_health_check_2026_05_12",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info(
            "affect.hooks: 2-week health check scheduled at 2026-05-12 04:45 Europe/Helsinki"
        )
    except Exception:
        logger.debug(
            "affect.hooks: scheduler not available; one-shot health check unscheduled",
            exc_info=True,
        )
