-- ============================================================
-- MIGRATION 034: Error anomaly history for the permanent error monitor
--
-- The monitor (app/observability/error_monitor.py) scans
-- workspace/logs/errors.jsonl every 5 minutes, groups errors by
-- normalized signature, and persists detected anomalies here so
-- the React dashboard can display + acknowledge them across
-- gateway restarts.
--
-- Anomaly types:
--   * new_pattern  — a signature first seen in the last 60 min
--                    AND occurring > 5× in that window
--   * rate_spike   — a known signature whose last-hour rate
--                    exceeds 3× its 24h rolling average
--                    AND > 5/hour absolute
--   * total_rate   — total errors/hour > 2σ from 24h rolling mean
--                    (delegated to existing app/anomaly_detector.py)
--
-- Status lifecycle:
--   open → acknowledged (manual silence; preserves history)
--        → resolved     (auto-resolved when rate falls < 50% of
--                         detection threshold for 2 consecutive hours)
-- ============================================================

CREATE TABLE IF NOT EXISTS control_plane.error_anomalies (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_signature  TEXT NOT NULL,
    pattern_sample     TEXT,
    anomaly_type       TEXT NOT NULL
                       CHECK (anomaly_type IN ('new_pattern','rate_spike','total_rate')),
    severity           TEXT NOT NULL DEFAULT 'warning'
                       CHECK (severity IN ('info','warning','critical')),
    hourly_rate        NUMERIC(14,4),
    baseline_rate      NUMERIC(14,4),
    detected_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status             TEXT NOT NULL DEFAULT 'open'
                       CHECK (status IN ('open','acknowledged','resolved')),
    resolved_at        TIMESTAMPTZ,
    notes              TEXT
);

-- Hot-path index: dashboard polls "open anomalies, newest first" every 30s.
CREATE INDEX IF NOT EXISTS idx_error_anomalies_open
    ON control_plane.error_anomalies(detected_at DESC)
    WHERE status = 'open';

-- Lookup index: monitor checks "do I already have an open anomaly for this signature?"
-- before inserting a duplicate.
CREATE INDEX IF NOT EXISTS idx_error_anomalies_signature
    ON control_plane.error_anomalies(pattern_signature, detected_at DESC);
