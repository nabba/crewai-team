-- ============================================================
-- MIGRATION 027: Epistemic bias matches
--
-- One row per realtime or post-hoc bias detection. Multi-claim
-- biases (coherence_bias, defending_periphery) reference the
-- triggering claim in ``claim_id`` and the full participating
-- set in ``matched_claim_ids`` JSONB.
--
-- Why a separate table (vs a JSONB column on epistemic_claims):
--   * One claim can fire multiple biases — relation, not attribute.
--   * The realtime feed (React BiasFeed) wants "give me the last N
--     matches across all tasks" with a single index hit.
--   * Severity-stratified queries (critical vs medium) are normal
--     SQL, not JSON path digs.
--
-- Retention: same as claims (CASCADE from crew_tasks).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_bias_matches (
    id              BIGSERIAL PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,
    claim_id        TEXT NOT NULL
                    REFERENCES control_plane.epistemic_claims(claim_id) ON DELETE CASCADE,

    bias_id         TEXT NOT NULL,                       -- e.g. "inference_as_fact"
    severity        TEXT NOT NULL
                    CHECK (severity IN ('low','medium','high','critical')),

    -- Multi-claim biases list every participating claim_id here. For
    -- single-claim biases (the Phase 1 starter), this is just [claim_id].
    matched_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,

    detail          JSONB NOT NULL DEFAULT '{}'::jsonb,
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me the bias feed for the last hour"
CREATE INDEX IF NOT EXISTS idx_epistemic_bias_matches_detected
    ON control_plane.epistemic_bias_matches (detected_at DESC);

-- Per-task drill-in: "what biases fired during task X"
CREATE INDEX IF NOT EXISTS idx_epistemic_bias_matches_task
    ON control_plane.epistemic_bias_matches (task_id, detected_at);

-- Per-claim lookup: "what biases fired on this specific claim"
CREATE INDEX IF NOT EXISTS idx_epistemic_bias_matches_claim
    ON control_plane.epistemic_bias_matches (claim_id);

-- Bias-library stats: "how often is inference_as_fact firing?"
CREATE INDEX IF NOT EXISTS idx_epistemic_bias_matches_bias_severity
    ON control_plane.epistemic_bias_matches (bias_id, severity, detected_at);
