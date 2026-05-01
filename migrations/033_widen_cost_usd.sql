-- ============================================================
-- MIGRATION 033: Widen control_plane.*.cost_usd from NUMERIC(10,6) to NUMERIC(14,6)
--
-- Three tables created in migration 010 declared cost_usd as
-- NUMERIC(10,6), which caps at $9 999.999999. Long-running
-- self-improvement / evolution sessions accumulate
-- ``RequestCostTracker.total_cost_usd`` past that ceiling across
-- many LLM calls, and the final ``UPDATE control_plane.tickets``
-- (with mirror INSERT into ``control_plane.audit_log``) is
-- rejected by the column constraint:
--
--     numeric field overflow
--     DETAIL: A field with precision 10, scale 6 must round to
--             an absolute value less than 10^4.
--
-- Errors:  327 occurrences over 2026-04-15 → 2026-05-01.
--
-- Migration 021 already used NUMERIC(12,6) for
-- ``control_plane.crew_tasks.cost_usd``; we go to (14,6) here for
-- additional headroom (max ~$99,999,999.999999), well above any
-- plausible session cost.
--
-- This is a non-destructive widening:
--   - All currently-stored values fit (max observed: $8.26)
--   - No indexes reference these columns → no rebuild
--   - Brief ACCESS EXCLUSIVE lock per ALTER on small tables
--     (291 / 1 635 / 3 245 rows) — sub-second
--   - Idempotent in spirit: re-running is safe (Postgres simply
--     re-validates that the new precision is at least as wide)
-- ============================================================

ALTER TABLE control_plane.tickets    ALTER COLUMN cost_usd TYPE NUMERIC(14,6);
ALTER TABLE control_plane.audit_log  ALTER COLUMN cost_usd TYPE NUMERIC(14,6);
ALTER TABLE control_plane.heartbeats ALTER COLUMN cost_usd TYPE NUMERIC(14,6);
