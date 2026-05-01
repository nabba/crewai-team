-- ============================================================
-- MIGRATION 032: Epistemic tuning proposals
--
-- One row per autotune-generated proposal for the bias library or
-- verifier registry. Proposals are surfaced to the dashboard and
-- await operator accept/reject decisions; the YAML change itself
-- happens via a CODEOWNERS PR — never auto-applied at runtime.
--
-- Idempotent re-runs: ``content_hash`` is a stable hash of the
-- (target_kind, target_id, kind, yaml_patch) tuple. Re-running the
-- analyzer over the same evidence produces the same hash; the
-- UPSERT semantics refresh ``metric_evidence`` and ``confidence``
-- without duplicating rows.
--
-- Status lifecycle: proposed → accepted | rejected | superseded.
-- "superseded" is reserved for future evidence that overturns a
-- prior proposal (e.g., a downgrade proposal that's now stale
-- because the bias's force-proceed rate dropped).
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.epistemic_tuning_proposals (
    proposal_id     TEXT PRIMARY KEY,
    content_hash    TEXT NOT NULL UNIQUE,           -- stable across reruns
    target_kind     TEXT NOT NULL
                    CHECK (target_kind IN ('bias','verifier')),
    target_id       TEXT NOT NULL,                   -- bias_id or verifier_id

    kind            TEXT NOT NULL                    -- ProposalKind
                    CHECK (kind IN (
                        'severity_downgrade',
                        'severity_upgrade',
                        'retirement_candidate',
                        'verifier_retirement'
                    )),
    rationale       TEXT NOT NULL DEFAULT '',
    metric_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    yaml_patch      TEXT NOT NULL DEFAULT '',
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0.5
                    CHECK (confidence >= 0.0 AND confidence <= 1.0),

    status          TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed','accepted','rejected','superseded')),
    operator_note   TEXT NOT NULL DEFAULT '',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot path: "show me open proposals"
CREATE INDEX IF NOT EXISTS idx_epistemic_tuning_proposals_status
    ON control_plane.epistemic_tuning_proposals (status, created_at DESC);

-- Per-target drill-in: "what proposals exist for this bias"
CREATE INDEX IF NOT EXISTS idx_epistemic_tuning_proposals_target
    ON control_plane.epistemic_tuning_proposals (target_kind, target_id, created_at DESC);

-- Recent feed for the React panel
CREATE INDEX IF NOT EXISTS idx_epistemic_tuning_proposals_created
    ON control_plane.epistemic_tuning_proposals (created_at DESC);
