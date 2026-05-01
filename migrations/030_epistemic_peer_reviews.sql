-- ============================================================
-- MIGRATION 030: Epistemic peer reviews
--
-- One row per peer-review escalation triggered by the calibration
-- gate when ``destructive_without_recheck`` (or any future critical
-- bias requesting peer_review) fires. Each row captures the
-- proposal, the verdict (allow/revise/veto), and the rationale.
--
-- Powers:
--   * /epistemic/peer-reviews/recent for the React panel
--   * Post-mortem cross-reference when an incident's behavioral
--     change cites a peer-review escalation
--
-- Retention: same as everything else (CASCADE from crew_tasks).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_peer_reviews (
    id              BIGSERIAL PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,
    triggering_claim_id TEXT,                          -- nullable: not every review traces to one claim

    proposal_excerpt TEXT NOT NULL,                    -- short snippet of the destructive proposal
    decision        TEXT NOT NULL
                    CHECK (decision IN ('allow','revise','veto')),
    rationale       TEXT NOT NULL DEFAULT '',
    suggested_revision TEXT,
    reviewers       JSONB NOT NULL DEFAULT '[]'::jsonb,    -- list of role strings
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    requested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me recent peer reviews"
CREATE INDEX IF NOT EXISTS idx_epistemic_peer_reviews_requested
    ON control_plane.epistemic_peer_reviews (requested_at DESC);

-- Per-task drill-in
CREATE INDEX IF NOT EXISTS idx_epistemic_peer_reviews_task
    ON control_plane.epistemic_peer_reviews (task_id, requested_at);

-- Decision-stratified stats: "how often do we veto?"
CREATE INDEX IF NOT EXISTS idx_epistemic_peer_reviews_decision
    ON control_plane.epistemic_peer_reviews (decision, requested_at DESC);
