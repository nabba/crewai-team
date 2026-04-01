-- Modification engine schema — tracks prompt change proposals, evaluations, and promotions.

CREATE SCHEMA IF NOT EXISTS modification;

-- Every modification attempt (proposed, evaluated, promoted, rejected, rolled_back)
CREATE TABLE modification.attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_id UUID,                  -- FK to feedback.patterns that triggered this
    tier TEXT NOT NULL,                -- 'tier1' or 'tier2'
    target_role TEXT NOT NULL,
    target_parameter TEXT NOT NULL,
    strategy TEXT,                     -- additive_instruction, example_injection,
                                      -- instruction_refinement, constraint_addition,
                                      -- persona_calibration

    current_version INT,
    proposed_version INT,
    proposed_content TEXT,
    explanation TEXT,

    status TEXT DEFAULT 'pending',     -- pending, evaluating, approved, rejected,
                                      -- promoted, rolled_back, awaiting_approval
    eval_result JSONB,

    created_at TIMESTAMPTZ DEFAULT now(),
    evaluated_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ,
    cooldown_until TIMESTAMPTZ
);

CREATE INDEX idx_mod_attempts_status ON modification.attempts(status);
CREATE INDEX idx_mod_attempts_role ON modification.attempts(target_role, created_at);

-- Audit log for all modification actions
CREATE TABLE modification.log (
    id SERIAL PRIMARY KEY,
    attempt_id UUID,
    tier TEXT,
    target_role TEXT,
    action TEXT,                       -- proposed, evaluated, promoted, rejected, rolled_back
    detail TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_mod_log_tier_ts ON modification.log(tier, created_at);
CREATE INDEX idx_mod_log_attempt ON modification.log(attempt_id);
