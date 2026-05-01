-- ============================================================
-- MIGRATION 029: Epistemic incidents
--
-- One row per IncidentReport produced by the post-mortem pipeline.
-- Stores the full report JSONB (timeline, root cause, enabling
-- factors, behavioral changes) plus indexed top-level fields for
-- the React /incidents view and the Self-Improver flush flag.
--
-- Why fields-as-columns + JSONB instead of pure JSONB:
--   * The recent-feed query orders by ``created_at``, filters by
--     ``severity`` — both are top-level enough to deserve real columns.
--   * ``self_improver_emitted`` is the operational flag for "did we
--     flush this to the Self-Improver loop yet?" — partial-index
--     scans for the unflushed pile must be O(1).
--   * Everything else (timeline, behavioral_changes) lives in the
--     ``report`` JSONB blob — these are read once for drill-in and
--     never queried by content.
--
-- Retention: same as claims (CASCADE from crew_tasks).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_incidents (
    incident_id     TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,

    root_cause_bias_id TEXT NOT NULL,
    severity        TEXT NOT NULL
                    CHECK (severity IN ('low','medium','high','critical')),

    -- Full IncidentReport.as_jsonable() — timeline, enabling factors,
    -- missed signals, behavioral changes, cost. Read once for drill-in.
    report          JSONB NOT NULL DEFAULT '{}'::jsonb,

    self_improver_emitted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me recent incidents"
CREATE INDEX IF NOT EXISTS idx_epistemic_incidents_created
    ON control_plane.epistemic_incidents (created_at DESC);

-- Per-task: "what incidents fired during task X"
CREATE INDEX IF NOT EXISTS idx_epistemic_incidents_task
    ON control_plane.epistemic_incidents (task_id, created_at);

-- Library stats: "how often is each bias the root cause"
CREATE INDEX IF NOT EXISTS idx_epistemic_incidents_root_cause
    ON control_plane.epistemic_incidents (root_cause_bias_id, severity, created_at DESC);

-- Operational flag: "incidents not yet flushed to Self-Improver"
-- Partial index keeps it tiny (most rows have emitted=true once flushed).
CREATE INDEX IF NOT EXISTS idx_epistemic_incidents_unflushed
    ON control_plane.epistemic_incidents (created_at)
    WHERE self_improver_emitted = FALSE;
