-- ============================================================
-- MIGRATION 035: Pearl Causal Hierarchy layer on claims
--
-- Adds two columns to control_plane.epistemic_claims:
--
--   * pch_layer — the layer of the Pearl Causal Hierarchy that
--     the claim's content sits at:
--         L1 observational ("X correlates with Y")
--         L2 interventional ("doing X changes Y")
--         L3 counterfactual ("if X had been different, Y would …")
--     NULL = "not a causal claim". Existing rows stay NULL — the
--     7-day retention via crew_tasks CASCADE rolls them out
--     naturally; no backfill is performed.
--
--   * causal_evidence_kinds — which species of L2-grade evidence
--     back the claim, if any (ablation, ab_test, do_intervention,
--     controlled_experiment). The CausalLayerOverreachDetector in
--     app.epistemic.detectors.realtime consults this to decide
--     whether an L2/L3-tagged claim is grounded in real
--     intervention or is overreach off observational data.
--
-- A partial index on (task_id, pch_layer) where layer is L2 or L3
-- supports the post-hoc audit query "give me every L2/L3 claim
-- this task made" — the typical scope when reviewing whether the
-- Self-Improver's recommendations were grounded.
-- ============================================================

ALTER TABLE control_plane.epistemic_claims
    ADD COLUMN pch_layer TEXT
        CHECK (pch_layer IS NULL OR pch_layer IN ('L1','L2','L3')),
    ADD COLUMN causal_evidence_kinds JSONB NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_epistemic_claims_pch_overreach
    ON control_plane.epistemic_claims (task_id, pch_layer)
    WHERE pch_layer IN ('L2','L3');
