-- Evolution schema for DGM-Hyperagent variant tracking.
-- Runs in the existing mem0 PostgreSQL instance, isolated by schema.

CREATE SCHEMA IF NOT EXISTS evolution;

-- Every agent variant ever generated
CREATE TABLE IF NOT EXISTS evolution.variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    target_type TEXT NOT NULL,
    generation INT NOT NULL DEFAULT 0,
    parent_id UUID REFERENCES evolution.variants(id),
    source_code TEXT NOT NULL,
    file_path TEXT NOT NULL,
    modification_diff TEXT,
    modification_reasoning TEXT,
    scores JSONB DEFAULT '{}',
    composite_score FLOAT,
    passed_threshold BOOLEAN DEFAULT FALSE,
    proposer_model TEXT,
    judge_model TEXT,
    eval_task_set TEXT,
    compute_cost_tokens INT,
    execution_time_seconds FLOAT,
    times_selected INT DEFAULT 0,
    child_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    metadata JSONB DEFAULT '{}'
);

-- Evolution run tracking
CREATE TABLE IF NOT EXISTS evolution.runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    target_type TEXT NOT NULL,
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    generations_completed INT DEFAULT 0,
    max_generations INT,
    best_variant_id UUID REFERENCES evolution.variants(id),
    config JSONB,
    status TEXT DEFAULT 'running'
);

-- Lineage graph edges
CREATE TABLE IF NOT EXISTS evolution.lineage (
    parent_id UUID REFERENCES evolution.variants(id),
    child_id UUID REFERENCES evolution.variants(id),
    run_id UUID REFERENCES evolution.runs(id),
    PRIMARY KEY (parent_id, child_id)
);

-- Human-approved promotions
CREATE TABLE IF NOT EXISTS evolution.promotions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variant_id UUID REFERENCES evolution.variants(id) NOT NULL,
    promoted_at TIMESTAMPTZ DEFAULT now(),
    promoted_by TEXT DEFAULT 'human',
    notes TEXT,
    reverted_at TIMESTAMPTZ
);

-- Immutable evaluation task sets
CREATE TABLE IF NOT EXISTS evolution.eval_sets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    tasks JSONB NOT NULL,
    rubric JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    locked BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_variants_agent ON evolution.variants(agent_name);
CREATE INDEX IF NOT EXISTS idx_variants_parent ON evolution.variants(parent_id);
CREATE INDEX IF NOT EXISTS idx_variants_score ON evolution.variants(composite_score DESC);
CREATE INDEX IF NOT EXISTS idx_variants_generation ON evolution.variants(agent_name, generation);
CREATE INDEX IF NOT EXISTS idx_runs_status ON evolution.runs(status);
