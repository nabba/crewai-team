-- Migration 012: Internal states for sentience architecture
-- Adds: internal_states table, agent_experiences table, self_certainty column
-- Requires: pgvector extension (already enabled for Mem0)

-- Ensure pgvector is available
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Table: internal_states ──────────────────────────────────────────────────
-- One row per reasoning step per agent

CREATE TABLE IF NOT EXISTS internal_states (
    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    crew_id VARCHAR(100),
    venture VARCHAR(20) CHECK (venture IN ('plg', 'archibal', 'kaicart', 'system', '')),
    step_number INTEGER NOT NULL DEFAULT 0,
    decision_context TEXT,

    -- Certainty vector (6 dimensions)
    certainty_factual_grounding REAL DEFAULT 0.5,
    certainty_tool_confidence REAL DEFAULT 0.5,
    certainty_coherence REAL DEFAULT 0.5,
    certainty_task_understanding REAL DEFAULT 0.5,
    certainty_value_alignment REAL DEFAULT 0.5,
    certainty_meta REAL DEFAULT 0.5,

    -- Somatic marker
    somatic_valence REAL DEFAULT 0.0,
    somatic_intensity REAL DEFAULT 0.0,
    somatic_source TEXT,
    somatic_match_count INTEGER DEFAULT 0,

    -- Meta-cognitive state
    meta_strategy_assessment VARCHAR(20) DEFAULT 'not_assessed',
    meta_modification_proposed BOOLEAN DEFAULT FALSE,
    meta_modification_description TEXT,
    meta_compute_phase VARCHAR(10) DEFAULT 'early',
    meta_compute_budget_remaining REAL DEFAULT 1.0,
    meta_reassessment_triggered BOOLEAN DEFAULT FALSE,

    -- Derived
    certainty_trend VARCHAR(10) DEFAULT 'stable',
    action_disposition VARCHAR(10) DEFAULT 'proceed',
    risk_tier INTEGER DEFAULT 1 CHECK (risk_tier BETWEEN 1 AND 4),

    -- Full JSONB for flexible querying
    full_state JSONB,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_disposition CHECK (action_disposition IN ('proceed', 'cautious', 'pause', 'escalate')),
    CONSTRAINT valid_trend CHECK (certainty_trend IN ('rising', 'stable', 'falling'))
);

CREATE INDEX IF NOT EXISTS idx_internal_states_agent_time
    ON internal_states (agent_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_internal_states_venture_time
    ON internal_states (venture, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_internal_states_disposition
    ON internal_states (action_disposition, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_internal_states_risk_tier
    ON internal_states (risk_tier, created_at DESC)
    WHERE risk_tier >= 3;

CREATE INDEX IF NOT EXISTS idx_internal_states_full_state
    ON internal_states USING GIN (full_state);

-- ── Table: agent_experiences ────────────────────────────────────────────────
-- Outcome-tagged experiences for somatic marker lookups
-- Embedding dimension: 768 (Ollama nomic-embed-text via Metal GPU)

CREATE TABLE IF NOT EXISTS agent_experiences (
    experience_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    venture VARCHAR(20),
    context_summary TEXT NOT NULL,
    context_embedding vector(768),
    outcome_score REAL NOT NULL CHECK (outcome_score BETWEEN -1.0 AND 1.0),
    outcome_description TEXT,
    task_type VARCHAR(100),
    tools_used TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- IVFFlat index for fast cosine similarity search
-- Note: requires at least 100 rows before index is effective
CREATE INDEX IF NOT EXISTS idx_agent_experiences_embedding
    ON agent_experiences USING ivfflat (context_embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_agent_experiences_agent_time
    ON agent_experiences (agent_id, created_at DESC);

-- ── View: certainty trends per agent ────────────────────────────────────────

CREATE OR REPLACE VIEW agent_certainty_trends AS
SELECT
    agent_id,
    venture,
    DATE_TRUNC('hour', created_at) AS hour,
    AVG(certainty_factual_grounding) AS avg_factual,
    AVG(certainty_tool_confidence) AS avg_tools,
    AVG(certainty_coherence) AS avg_coherence,
    AVG(certainty_meta) AS avg_meta,
    AVG(somatic_valence) AS avg_valence,
    COUNT(*) FILTER (WHERE action_disposition = 'escalate') AS escalation_count,
    COUNT(*) AS step_count
FROM internal_states
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY agent_id, venture, DATE_TRUNC('hour', created_at)
ORDER BY hour DESC;

-- ── Add self_certainty_score to training.interactions ────────────────────────

ALTER TABLE training.interactions
    ADD COLUMN IF NOT EXISTS self_certainty_score REAL;

CREATE INDEX IF NOT EXISTS idx_interactions_sc_score
    ON training.interactions (self_certainty_score)
    WHERE self_certainty_score IS NOT NULL;
