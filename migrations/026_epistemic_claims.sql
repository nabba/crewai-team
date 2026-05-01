-- ============================================================
-- MIGRATION 026: Epistemic claims
--
-- One row per assertion the system makes. The foundational
-- data structure for the Epistemic Integrity Layer
-- (see crewai-team/docs/EPISTEMIC_INTEGRITY.md).
--
-- Why a dedicated table (vs spans.detail JSONB):
--   * crew_task_spans.detail is capped at ~8 KB/row by
--     the persistence layer; long claim statements would
--     get lossily truncated.
--   * Claims have first-class queries — "all unverified
--     load-bearing claims for task X", "did any claim get
--     superseded by Y" — that need real indexes, not JSON
--     path traversal.
--   * Claims have their own lifecycle (supersession, decay)
--     that is cleaner as a relation than as opaque JSON.
--
-- Correlation:
--   * task_id FKs into control_plane.crew_tasks (CASCADE on
--     delete — claims belong to their task).
--   * span_id FKs into control_plane.crew_task_spans (SET NULL
--     on delete — claims survive span pruning).
--   * superseded_by is a self-FK; CASCADE would be wrong because
--     supersession is a fact we want to preserve historically.
--
-- Retention: same as spans (7 days), via crew_tasks CASCADE.
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_claims (
    claim_id        TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL
                    REFERENCES control_plane.crew_tasks(id) ON DELETE CASCADE,
    span_id         BIGINT
                    REFERENCES control_plane.crew_task_spans(id) ON DELETE SET NULL,

    agent_role      TEXT NOT NULL,                       -- commander | researcher | coder | writer | self_improver
    statement       TEXT NOT NULL,                        -- the assertion in the agent's words

    status          TEXT NOT NULL
                    CHECK (status IN ('verified','inferred','assumed','contradicted')),
    register        TEXT NOT NULL
                    CHECK (register IN ('declarative','hedged','unverified','internal')),

    -- Evidence: list of {kind, source_ref, excerpt, confidence}.
    -- Verifying action: optional {tool, args, expected_signal, estimated_seconds, safety}.
    evidence            JSONB NOT NULL DEFAULT '[]'::jsonb,
    verifying_action    JSONB,

    load_bearing    BOOLEAN NOT NULL DEFAULT FALSE,
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,   -- list[str]

    superseded_by   TEXT
                    REFERENCES control_plane.epistemic_claims(claim_id) ON DELETE SET NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "give me the ledger for task X, ordered by emission time"
CREATE INDEX IF NOT EXISTS idx_epistemic_claims_task_created
    ON control_plane.epistemic_claims (task_id, created_at);

-- Span lookup: which claims came out of span Y
CREATE INDEX IF NOT EXISTS idx_epistemic_claims_span
    ON control_plane.epistemic_claims (span_id)
    WHERE span_id IS NOT NULL;

-- Calibration hot path: "any unverified load-bearing claims for task X?"
-- Partial index keeps it tiny (most claims are not load-bearing).
CREATE INDEX IF NOT EXISTS idx_epistemic_claims_unverified_load_bearing
    ON control_plane.epistemic_claims (task_id, status)
    WHERE load_bearing = TRUE
      AND status IN ('inferred','assumed');

-- Supersession traversal: "what did claim X get replaced by"
CREATE INDEX IF NOT EXISTS idx_epistemic_claims_superseded_by
    ON control_plane.epistemic_claims (superseded_by)
    WHERE superseded_by IS NOT NULL;
