-- Feedback pipeline schema — stores user satisfaction signals and aggregated patterns.
-- Part of the self-improving feedback loop (Tier 1 Adaptive layer).

CREATE SCHEMA IF NOT EXISTS feedback;

-- Individual feedback events (explicit reactions, corrections, implicit signals)
CREATE TABLE feedback.events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id TEXT,              -- conversation_store task row ID
    sender_id TEXT NOT NULL,          -- HMAC-hashed phone number (privacy)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Signal source
    feedback_type TEXT NOT NULL,      -- explicit_positive, explicit_negative, explicit_correction,
                                     -- explicit_instruction, implicit_rerequest, implicit_abandonment,
                                     -- implicit_followup
    raw_signal TEXT,                  -- emoji, correction text, etc.

    -- Diagnosis (filled by classifier LLM)
    category TEXT,                    -- accuracy, style, completeness, relevance, tool_choice, speed, safety
    severity TEXT,                    -- critical, moderate, minor
    target_layer TEXT,                -- adaptive, protected
    target_parameter TEXT,            -- system_prompt, few_shot_examples, style_params, workflow_graph, etc.
    target_role TEXT,                 -- commander, researcher, coder, writer, etc.
    direction TEXT,                   -- natural language: what should change
    confidence FLOAT DEFAULT 0.5,    -- 0.0-1.0

    -- Context from the original interaction
    original_task TEXT,               -- user's original request
    original_response TEXT,           -- bot's response (truncated to 2000 chars)
    crew_used TEXT,                   -- which crew handled the task
    prompt_version INT,               -- which prompt version was active for target_role
    model_used TEXT,                  -- which LLM produced the response

    -- Processing state
    processed BOOLEAN DEFAULT FALSE,
    pattern_id UUID                   -- FK to patterns table once aggregated
);

CREATE INDEX idx_feedback_unprocessed ON feedback.events(processed, timestamp);
CREATE INDEX idx_feedback_category ON feedback.events(category, target_parameter);
CREATE INDEX idx_feedback_sender ON feedback.events(sender_id, timestamp);

-- Aggregated feedback patterns — trigger modification attempts
CREATE TABLE feedback.patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    target_parameter TEXT NOT NULL,
    target_role TEXT,
    direction TEXT NOT NULL,           -- aggregated direction from constituent events
    event_count INT DEFAULT 0,
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,
    status TEXT DEFAULT 'pending',     -- pending, triggered, resolved, conflicting
    triggered_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    modification_id UUID              -- FK to modification.attempts once acted on
);

CREATE INDEX idx_patterns_status ON feedback.patterns(status);

-- Metadata about every bot response (for correlating feedback back to context)
CREATE TABLE feedback.response_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    msg_timestamp BIGINT NOT NULL,    -- Signal message timestamp (ms) of the BOT's reply
    sender_id TEXT NOT NULL,
    task_text TEXT,                    -- user's original request (truncated)
    response_text TEXT,               -- bot's response (truncated)
    crew_used TEXT,
    prompt_versions JSONB,            -- {"commander": 3, "researcher": 2, ...}
    model_used TEXT,
    task_id INT,                      -- conversation_store task row ID
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_response_meta_ts ON feedback.response_metadata(msg_timestamp);
CREATE INDEX idx_response_meta_sender ON feedback.response_metadata(sender_id, created_at);
