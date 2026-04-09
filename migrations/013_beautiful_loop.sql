-- Migration 013: Beautiful Loop additions (Phases 3R, 7, 8)

-- Add Beautiful Loop columns to internal_states
ALTER TABLE internal_states
    ADD COLUMN IF NOT EXISTS hyper_predicted_certainty REAL,
    ADD COLUMN IF NOT EXISTS hyper_actual_certainty REAL,
    ADD COLUMN IF NOT EXISTS hyper_prediction_error REAL,
    ADD COLUMN IF NOT EXISTS free_energy_proxy REAL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS free_energy_trend VARCHAR(15) DEFAULT 'stable',
    ADD COLUMN IF NOT EXISTS precision_weighted_certainty REAL,
    ADD COLUMN IF NOT EXISTS competition_winner TEXT,
    ADD COLUMN IF NOT EXISTS competition_candidates JSONB,
    ADD COLUMN IF NOT EXISTS reality_model JSONB;

-- Behavioral scorecards (Phase 8)
CREATE TABLE IF NOT EXISTS behavioral_scorecards (
    scorecard_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    step_count INTEGER,
    scores JSONB,
    composite_score REAL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scorecards_agent_time
    ON behavioral_scorecards (agent_id, created_at DESC);

-- Reality model snapshots
CREATE TABLE IF NOT EXISTS reality_model_snapshots (
    snapshot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    step_number INTEGER NOT NULL,
    elements JSONB NOT NULL,
    global_coherence REAL,
    mean_precision REAL,
    total_prediction_error REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rm_snapshots_agent_time
    ON reality_model_snapshots (agent_id, created_at DESC);
