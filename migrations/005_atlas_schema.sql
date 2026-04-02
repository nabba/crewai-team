-- ATLAS schema: Autonomous Tool-Learning & Adaptive Skills System
-- Tables for skill tracking, competence map, and audit log.

CREATE SCHEMA IF NOT EXISTS atlas;

-- Audit log for all external API calls, code executions, credential access
CREATE TABLE IF NOT EXISTS atlas.audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    method TEXT DEFAULT '',
    credential_used TEXT DEFAULT '',
    sandbox_id TEXT DEFAULT '',
    result TEXT DEFAULT 'success',
    response_code INTEGER DEFAULT 0,
    execution_time_ms REAL DEFAULT 0,
    tokens_consumed JSONB DEFAULT '{}',
    cost_usd REAL DEFAULT 0,
    approval TEXT DEFAULT 'auto'
);

CREATE INDEX IF NOT EXISTS idx_atlas_audit_ts ON atlas.audit_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_atlas_audit_agent ON atlas.audit_log(agent);

-- Skill usage tracking (supplements filesystem manifests with DB-queryable stats)
CREATE TABLE IF NOT EXISTS atlas.skill_usage (
    id SERIAL PRIMARY KEY,
    skill_id TEXT NOT NULL,
    used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    context TEXT DEFAULT '',
    latency_ms REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_atlas_skill_usage ON atlas.skill_usage(skill_id, used_at DESC);

-- Learning session tracking
CREATE TABLE IF NOT EXISTS atlas.learning_sessions (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    task_description TEXT NOT NULL,
    steps_total INTEGER DEFAULT 0,
    steps_completed INTEGER DEFAULT 0,
    steps_failed INTEGER DEFAULT 0,
    plan_json JSONB DEFAULT '{}',
    status TEXT DEFAULT 'pending'
);

-- API knowledge cache (supplements filesystem with DB-queryable API registry)
CREATE TABLE IF NOT EXISTS atlas.api_knowledge (
    id SERIAL PRIMARY KEY,
    api_name TEXT NOT NULL UNIQUE,
    base_url TEXT DEFAULT '',
    auth_type TEXT DEFAULT '',
    endpoints_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_verified TIMESTAMPTZ,
    knowledge_json JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_atlas_api_name ON atlas.api_knowledge(api_name);
