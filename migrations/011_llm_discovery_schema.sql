-- ============================================================
-- MIGRATION 011: LLM Discovery Pipeline
-- Tracks discovered models, benchmark results, and promotions.
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.discovered_models (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        TEXT NOT NULL UNIQUE,       -- e.g. 'openrouter/google/gemma-4-31b-it'
    provider        TEXT NOT NULL,              -- 'openrouter', 'ollama', 'anthropic'
    display_name    TEXT NOT NULL,
    context_window  INT DEFAULT 0,
    cost_input_per_m  NUMERIC(10,6) DEFAULT 0,
    cost_output_per_m NUMERIC(10,6) DEFAULT 0,
    multimodal      BOOLEAN DEFAULT FALSE,
    tool_calling    BOOLEAN DEFAULT FALSE,
    source          TEXT DEFAULT 'openrouter_api', -- 'openrouter_api', 'tech_radar', 'manual'
    raw_metadata    JSONB DEFAULT '{}',

    -- Evaluation
    benchmark_score NUMERIC(6,4),               -- 0.0-1.0 from eval_set
    benchmark_role  TEXT,                        -- which role was benchmarked
    benchmarked_at  TIMESTAMPTZ,

    -- Promotion
    status          TEXT DEFAULT 'discovered'
                    CHECK (status IN ('discovered','benchmarking','approved','rejected','promoted','retired')),
    promoted_tier   TEXT,                       -- 'local', 'free', 'budget', 'mid', 'premium'
    promoted_roles  TEXT[],                     -- roles assigned to
    promoted_at     TIMESTAMPTZ,
    reviewed_by     TEXT,                       -- 'user', 'system', 'auto'

    discovered_at   TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_discovered_models_status
    ON control_plane.discovered_models(status);
CREATE INDEX IF NOT EXISTS idx_discovered_models_provider
    ON control_plane.discovered_models(provider);
