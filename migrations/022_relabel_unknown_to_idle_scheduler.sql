-- Relabel the 'unknown' budget bucket to 'idle_scheduler'.
--
-- Historical context: every LLM call made outside a crew lifecycle —
-- which is overwhelmingly idle-scheduler jobs (llm-discovery, fiction
-- ingestion, training-collector, atlas competence sync, …) — landed in
-- a budgets row keyed 'unknown' because the agent_role ContextVar was
-- unset. That bucket grew to ~$60 across projects.
--
-- Going forward, app/idle_scheduler.py:_run_single_job wraps every
-- job in agent_scope(job_name), so new spend lands in honest per-job
-- rows. The historical aggregate moves into a single
-- 'idle_scheduler' row per project (preserves the running total).

DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT project_id, period, spent_usd, spent_tokens, is_paused
        FROM control_plane.budgets
        WHERE agent_role = 'unknown'
    LOOP
        -- Upsert into idle_scheduler row, merging the spend.
        INSERT INTO control_plane.budgets
            (project_id, agent_role, period, limit_usd, spent_usd,
             spent_tokens, is_paused)
        VALUES (r.project_id, 'idle_scheduler', r.period, 50.0,
                r.spent_usd, r.spent_tokens, r.is_paused)
        ON CONFLICT (project_id, agent_role, period) DO UPDATE
        SET spent_usd    = control_plane.budgets.spent_usd
                          + EXCLUDED.spent_usd,
            spent_tokens = control_plane.budgets.spent_tokens
                          + EXCLUDED.spent_tokens,
            updated_at   = NOW();
    END LOOP;

    -- Drop the now-merged unknown rows.
    DELETE FROM control_plane.budgets WHERE agent_role = 'unknown';
END $$;
