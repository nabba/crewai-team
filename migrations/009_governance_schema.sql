-- Unified promotion governance — audit trail across all improvement systems.
-- Every promotion decision (evolution, modification, training, ATLAS)
-- is recorded here with consistent metrics, gates, and outcomes.

CREATE SCHEMA IF NOT EXISTS governance;

CREATE TABLE IF NOT EXISTS governance.promotions (
    id UUID PRIMARY KEY,
    system TEXT NOT NULL,            -- evolution, modification, training, atlas
    target TEXT NOT NULL,            -- what was promoted (role, adapter, skill, etc.)
    proposed_by TEXT,                -- which subsystem/agent proposed this
    quality_score FLOAT,            -- normalized 0.0-1.0
    safety_score FLOAT,             -- normalized 0.0-1.0
    approved BOOLEAN NOT NULL,
    reason TEXT,                     -- human-readable explanation
    gate_results JSONB,             -- which gates passed/failed
    metrics JSONB,                  -- system-specific scores
    baseline_scores JSONB,          -- previous version metrics for regression check
    artifacts JSONB,                -- system-specific (prompt text, adapter path, etc.)
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_gov_system_ts ON governance.promotions(system, created_at);
CREATE INDEX IF NOT EXISTS idx_gov_approved ON governance.promotions(approved, created_at);
