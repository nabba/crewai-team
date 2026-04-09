-- Migration 014: Emergent Engineering + Prosocial Learning

-- Tool proposals (Phase 9)
CREATE TABLE IF NOT EXISTS tool_proposals (
    proposal_id VARCHAR(200) PRIMARY KEY,
    agent_id VARCHAR(100) NOT NULL,
    tool_name VARCHAR(200),
    tool_description TEXT,
    justification TEXT,
    tool_code TEXT,
    tool_type VARCHAR(50),
    triggered_by TEXT,
    frequency_of_need INTEGER DEFAULT 0,
    status VARCHAR(30) DEFAULT 'pending',
    human_feedback TEXT,
    sandbox_result JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_proposals_agent ON tool_proposals (agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON tool_proposals (status);

-- Prosocial profiles (Phase 10)
CREATE TABLE IF NOT EXISTS prosocial_profiles (
    agent_id VARCHAR(100) PRIMARY KEY,
    total_rounds INTEGER DEFAULT 0,
    generosity REAL DEFAULT 0.5,
    honesty REAL DEFAULT 0.5,
    cooperativeness REAL DEFAULT 0.5,
    respectfulness REAL DEFAULT 0.5,
    altruism REAL DEFAULT 0.5,
    composite_prosociality REAL DEFAULT 0.5,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Prosocial game outcomes (Phase 10)
CREATE TABLE IF NOT EXISTS prosocial_game_outcomes (
    outcome_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_type VARCHAR(50) NOT NULL,
    agent_ids TEXT[] NOT NULL,
    round_number INTEGER,
    actions JSONB,
    individual_scores JSONB,
    collective_score REAL,
    prosocial_scores JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prosocial_outcomes_time
    ON prosocial_game_outcomes (created_at DESC);
