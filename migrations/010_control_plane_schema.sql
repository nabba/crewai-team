-- ============================================================
-- MIGRATION 010: Control Plane Schema
-- Extends the existing Mem0 PostgreSQL instance with organizational
-- capabilities: projects, budgets, tickets, audit trail, governance,
-- org chart, and heartbeats.
--
-- All audit tables use INSERT-only access (no UPDATE/DELETE grants).
-- ============================================================

CREATE SCHEMA IF NOT EXISTS control_plane;

-- ── Projects (multi-company isolation) ──────────────────────

CREATE TABLE IF NOT EXISTS control_plane.projects (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    mission         TEXT,
    config_json     JSONB DEFAULT '{}',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO control_plane.projects (name, description, mission)
VALUES ('default', 'Default project', 'General-purpose AndrusAI operations')
ON CONFLICT (name) DO NOTHING;

INSERT INTO control_plane.projects (name, description, mission)
VALUES ('PLG', 'PLG Ticketing', 'Build and grow PLG ticketing platform for live events')
ON CONFLICT (name) DO NOTHING;

INSERT INTO control_plane.projects (name, description, mission)
VALUES ('Archibal', 'Archibal Content Authenticity', 'Content authenticity verification and provenance tracking')
ON CONFLICT (name) DO NOTHING;

INSERT INTO control_plane.projects (name, description, mission)
VALUES ('KaiCart', 'KaiCart TikTok Commerce', 'TikTok-first commerce platform for Thai SMB sellers')
ON CONFLICT (name) DO NOTHING;

-- ── Org Chart (agent hierarchy) ─────────────────────────────

CREATE TABLE IF NOT EXISTS control_plane.org_chart (
    agent_role      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    reports_to      TEXT REFERENCES control_plane.org_chart(agent_role),
    job_description TEXT,
    soul_file       TEXT,
    default_model   TEXT,
    sort_order      INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO control_plane.org_chart (agent_role, display_name, reports_to, job_description, soul_file, sort_order) VALUES
('commander',        'Commander',        NULL,          'Routes requests, orchestrates crews, manages system', 'souls/commander.md', 0),
('researcher',       'Researcher',       'commander',   'Web research, data gathering, fact synthesis', 'souls/researcher.md', 1),
('coder',            'Coder',            'commander',   'Code generation, debugging, architecture', 'souls/coder.md', 2),
('writer',           'Writer',           'commander',   'Documents, reports, summaries, communication', 'souls/writer.md', 3),
('media_analyst',    'Media Analyst',    'commander',   'YouTube, image, audio, video analysis', 'souls/media_analyst.md', 4),
('critic',           'Critic',           'commander',   'Quality review, vetting, safety checks', 'souls/critic.md', 5),
('self_improver',    'Self-Improver',    'commander',   'Learning, evolution, skill acquisition', 'souls/self_improver.md', 6),
('introspector',     'Introspector',     'commander',   'Self-awareness, metacognition, reflection', NULL, 7)
ON CONFLICT (agent_role) DO NOTHING;

-- ── Budgets (per-agent, per-project, per-month) ─────────────

CREATE TABLE IF NOT EXISTS control_plane.budgets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES control_plane.projects(id),
    agent_role      TEXT,
    period          TEXT NOT NULL,
    limit_usd       NUMERIC(10,4) NOT NULL,
    spent_usd       NUMERIC(10,4) DEFAULT 0,
    limit_tokens    BIGINT,
    spent_tokens    BIGINT DEFAULT 0,
    is_paused       BOOLEAN DEFAULT FALSE,
    warning_pct     INT DEFAULT 80,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (project_id, agent_role, period)
);

-- Atomic spend function — row-level lock prevents race conditions
CREATE OR REPLACE FUNCTION control_plane.record_spend(
    p_project_id UUID,
    p_agent_role TEXT,
    p_period TEXT,
    p_cost_usd NUMERIC,
    p_tokens BIGINT
) RETURNS BOOLEAN AS $$
DECLARE
    v_budget RECORD;
BEGIN
    SELECT * INTO v_budget
    FROM control_plane.budgets
    WHERE project_id = p_project_id
      AND (agent_role = p_agent_role OR agent_role IS NULL)
      AND period = p_period
    ORDER BY agent_role NULLS LAST
    LIMIT 1
    FOR UPDATE;

    IF NOT FOUND THEN RETURN TRUE; END IF;  -- no budget = unlimited
    IF v_budget.is_paused THEN RETURN FALSE; END IF;
    IF v_budget.spent_usd + p_cost_usd > v_budget.limit_usd THEN
        UPDATE control_plane.budgets
        SET is_paused = TRUE, updated_at = NOW()
        WHERE id = v_budget.id;
        RETURN FALSE;
    END IF;

    UPDATE control_plane.budgets
    SET spent_usd = spent_usd + p_cost_usd,
        spent_tokens = spent_tokens + COALESCE(p_tokens, 0),
        updated_at = NOW()
    WHERE id = v_budget.id;
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ── Tickets (persistent task tracking) ──────────────────────

CREATE TABLE IF NOT EXISTS control_plane.tickets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES control_plane.projects(id),
    title           TEXT NOT NULL,
    description     TEXT,
    status          TEXT DEFAULT 'todo'
                    CHECK (status IN ('todo','in_progress','review','done','failed','blocked')),
    priority        INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),
    assigned_agent  TEXT,
    assigned_crew   TEXT,
    source          TEXT DEFAULT 'signal',
    parent_id       UUID REFERENCES control_plane.tickets(id),
    difficulty      INT CHECK (difficulty BETWEEN 1 AND 10),
    cost_usd        NUMERIC(10,6) DEFAULT 0,
    tokens_used     BIGINT DEFAULT 0,
    result_summary  TEXT,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tickets_project_status
    ON control_plane.tickets(project_id, status);
CREATE INDEX IF NOT EXISTS idx_tickets_assigned
    ON control_plane.tickets(assigned_agent);
CREATE INDEX IF NOT EXISTS idx_tickets_created
    ON control_plane.tickets(created_at DESC);

-- ── Ticket Comments ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS control_plane.ticket_comments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id       UUID NOT NULL REFERENCES control_plane.tickets(id),
    author          TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata_json   JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comments_ticket
    ON control_plane.ticket_comments(ticket_id, created_at);

-- ── Audit Log (IMMUTABLE — append only) ─────────────────────

CREATE TABLE IF NOT EXISTS control_plane.audit_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    project_id      UUID,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    resource_type   TEXT,
    resource_id     TEXT,
    detail_json     JSONB DEFAULT '{}',
    cost_usd        NUMERIC(10,6),
    tokens          BIGINT
);

CREATE INDEX IF NOT EXISTS idx_audit_project_time
    ON control_plane.audit_log(project_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_actor
    ON control_plane.audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_audit_action
    ON control_plane.audit_log(action);

-- ── Governance Requests (approval queue) ────────────────────

CREATE TABLE IF NOT EXISTS control_plane.governance_requests (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id      UUID REFERENCES control_plane.projects(id),
    request_type    TEXT NOT NULL,
    requested_by    TEXT NOT NULL,
    title           TEXT NOT NULL,
    detail_json     JSONB DEFAULT '{}',
    status          TEXT DEFAULT 'pending'
                    CHECK (status IN ('pending','approved','rejected','expired')),
    reviewed_by     TEXT,
    reviewed_at     TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_governance_status
    ON control_plane.governance_requests(status, created_at DESC);

-- ── Heartbeat Log ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS control_plane.heartbeats (
    id              BIGSERIAL PRIMARY KEY,
    agent_role      TEXT NOT NULL,
    project_id      UUID,
    trigger_type    TEXT NOT NULL,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    tickets_processed INT DEFAULT 0,
    cost_usd        NUMERIC(10,6) DEFAULT 0,
    status          TEXT DEFAULT 'running'
                    CHECK (status IN ('running','completed','failed','skipped'))
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_agent
    ON control_plane.heartbeats(agent_role, started_at DESC);
