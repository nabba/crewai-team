-- ============================================================
-- MIGRATION 031: Epistemic gate overrides
--
-- One row per user override of an epistemic-gate verdict. The
-- override IS the strongest available learning signal — the user
-- told us our gate was wrong (or right, but the user has unseen
-- context) — so each row is also flushed to the Self-Improver as
-- a USER_CORRECTION LearningGap (signal_strength=0.9).
--
-- Powers /epistemic/overrides/recent and the React OverridesPanel.
--
-- Retention: same as everything else (CASCADE from crew_tasks).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_overrides (
    override_id     TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,
    -- nullable because not every override traces back to a peer review
    -- (some are pure calibration revise/block)
    peer_review_id  BIGINT,

    blocked_action  TEXT NOT NULL                 -- "block" | "revise"
                    CHECK (blocked_action IN ('block','revise')),
    user_action     TEXT NOT NULL
                    CHECK (user_action IN ('force_proceed','use_revision','abandon')),
    user_reasoning  TEXT NOT NULL DEFAULT '',
    overridden_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me recent overrides"
CREATE INDEX IF NOT EXISTS idx_epistemic_overrides_overridden
    ON control_plane.epistemic_overrides (overridden_at DESC);

-- Per-task drill-in
CREATE INDEX IF NOT EXISTS idx_epistemic_overrides_task
    ON control_plane.epistemic_overrides (task_id, overridden_at);

-- Action-stratified: "how often does the user force-proceed vs abandon?"
CREATE INDEX IF NOT EXISTS idx_epistemic_overrides_action
    ON control_plane.epistemic_overrides (user_action, overridden_at DESC);

-- Peer-review correlation: "which veto did this override target"
CREATE INDEX IF NOT EXISTS idx_epistemic_overrides_peer_review
    ON control_plane.epistemic_overrides (peer_review_id)
    WHERE peer_review_id IS NOT NULL;
