-- ============================================================
-- MIGRATION 028: Epistemic pushback events
--
-- One row per processed user contradiction. Captures the signal
-- (what the user said + which claim was contradicted) plus the
-- foundation-check outcome (REVERIFIED / FALSIFIED / UNVERIFIABLE)
-- and any cascade-invalidated dependent claims.
--
-- Powers two consumers:
--   * /epistemic/pushback/{stats,recent} for the React panel
--   * The Phase 4 post-hoc detector ``defending_periphery``, which
--     reads these events and walks the post-event span trace to
--     check whether the agent expanded the investigation instead
--     of re-running the verifier.
--
-- Retention: same as claims (CASCADE from crew_tasks).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_pushback_events (
    id              BIGSERIAL PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,
    contradicted_claim_id TEXT NOT NULL
                    REFERENCES control_plane.epistemic_claims(claim_id)
                    ON DELETE CASCADE,

    -- Signal: what the user said
    user_evidence   TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),
    detector        TEXT NOT NULL                 -- "regex" | "llm"
                    CHECK (detector IN ('regex','llm')),

    -- Outcome: what the foundation check returned
    outcome         TEXT NOT NULL
                    CHECK (outcome IN ('reverified','falsified','unverifiable')),
    new_evidence_excerpt TEXT NOT NULL DEFAULT '',
    invalidated_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me recent pushback events"
CREATE INDEX IF NOT EXISTS idx_epistemic_pushback_detected
    ON control_plane.epistemic_pushback_events (detected_at DESC);

-- Per-task drill-in: "what pushback fired during task X"
CREATE INDEX IF NOT EXISTS idx_epistemic_pushback_task
    ON control_plane.epistemic_pushback_events (task_id, detected_at);

-- Per-claim lookup: "did this claim ever get pushed back on?"
CREATE INDEX IF NOT EXISTS idx_epistemic_pushback_claim
    ON control_plane.epistemic_pushback_events (contradicted_claim_id);

-- Stratified outcome stats: "how often does foundation hold?"
CREATE INDEX IF NOT EXISTS idx_epistemic_pushback_outcome
    ON control_plane.epistemic_pushback_events (outcome, detected_at DESC);
