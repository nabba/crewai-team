# Audit — Healing monitors with destructive recommendations (2026-05-16)

## Context

Two incidents on 2026-05-16 followed the same pattern: a healing monitor
emitted an alert pointing at a destructive action against load-bearing
data, the operator acted on it, and only an ad-hoc tar snapshot saved
the data:

1. `migration_drill` alert pointed at running `deploy/scripts/migration-drill.sh`.
   The drill script had a docker-compose bind-mount sharing flaw that
   corrupted the live postgres database when run against a live stack.
   Restored from the 13:58 UTC backup; ~70 min of writes lost.
2. `chromadb_hygiene` orphan-segment detection compared on-disk UUID
   dir names against `collections.id` in `chroma.sqlite3`. ChromaDB
   names dirs after `segments.id` (different IDs). The monitor
   mis-classified 43 live VECTOR segment dirs as orphans across 5 KBs;
   acting on the recommendation deleted them. Restored from a
   pre-delete tar snapshot.

Both incidents shared a structural failure mode:
*filesystem-or-SQLite filter → automatic-or-recommended destructive
action → no operator gate, no snapshot, no schema verification step.*

This audit walks every monitor under `app/healing/monitors/` (n=35) and
classifies them under three verdicts to surface the same shape
elsewhere before it bites a third time.

## Summary

| Verdict | Count |
|---|---|
| SAFE — no destructive recommendation, OR destructive but operator-gated AND snapshot-first | 18 |
| RISK — destructive recommendation OR auto-action whose schema query is heuristic + worth a second look | 14 |
| CONFIRMED-BUG — schema mismatch OR destructive auto-action without snapshot | 3 |

The three CONFIRMED-BUGs are all in `retention.py`. One is fixed in
this PR (the chromadb timestamp-fallback bug — same shape as the
chromadb_hygiene incident). The other two (`run_worktrees` rmtree from
session JSON, `run_attachments` filesystem walk) are documented below
for a follow-up.

## What this PR ships

1. `app/healing/destructive_advisory.py` — guardrail. A monitor whose
   alert points at a destructive action must construct a
   `DestructiveAdvisory` instance with all five discipline fields
   (snapshot_path, apply_command, undo_command, verify_command,
   schema_assumption). The dataclass refuses construction if any field
   is missing or if the snapshot file is not on disk. The alert text
   is formatted with the verify step BEFORE the apply step.
2. `chromadb_hygiene._alert_orphan_segments` converted to use the
   helper. The orphan dirs are now snapshotted into
   `workspace/.snapshots/chromadb_orphans_<ts>.tar.gz` BEFORE the alert
   is emitted, with an inline `xargs -a <targets-file> rm -rf` apply
   command and a `tar -xzf` undo command.
3. `retention._oldest_ids` returns `(ids, stats)` instead of `ids`.
   Records lacking a parseable timestamp metadata are EXCLUDED from
   the deletion candidate pool (the prior `ts = 0.0` fallback caused
   them to be classified as "oldest" and deleted preferentially).
   When most records lack timestamps, retention does nothing this pass
   and the audit row + log warning surface it.

## Status of the follow-up items

All seven RISK-class items + the one CONFIRMED-BUG were closed in
follow-up PRs the same day. The items are listed below with the
shipped fix and the PR reference.

> **Note on this section's shape.** Originally written as
> "What this PR leaves for follow-up." Rewritten 2026-05-16 to
> reflect that the deferred work shipped — kept as a living record
> of what each fix actually addressed so the next maintainer can
> trace decisions back to the audit. See PROGRAM.md §53 for the
> full change log.

### retention.run_worktrees — operator-unattended rmtree from session JSON  ✅ shipped in #116

**Before:** `shutil.rmtree(data.get("worktree_path"))` with only an
`if wt and wt.exists()` guard.

**Shipped:** new `_validate_worktree_path(wt, *, expected_session_id)`
refuses unless the path is absolute, inside `worktree_root() + '/'`
(defends against suffix attacks like `/tmp/agent-sessions-evil`),
basename matches the session ID (defends against cross-session
collision via corrupted JSON), and exactly one segment under root.
Refusals counted in audit row; session JSON left in place for
operator inspection. See `retention.py:_validate_worktree_path`.

### retention.run_attachments — operator-unattended file unlink  ✅ shipped in #116

**Before:** `SIGNAL_ATTACHMENTS_DIR` env var honored verbatim.
`SIGNAL_ATTACHMENTS_DIR=/` would have nuked everything writable
from the container.

**Shipped:** new `_is_attachments_dir_safe(p)` refuses unless the
resolved path is inside `/app/attachments` (the compose mount),
`/app/workspace/`, or `/tmp/`. Suffix-attack defense via explicit
prefix + `/` check. Anything else: refusal recorded in state, no
files touched. See `retention.py:_is_attachments_dir_safe`.

### lock_housekeeper — `*.lock` filename heuristic + auto-delete  ✅ shipped in #116

**Before:** any file ending in `.lock` in the three watched dirs was
auto-deleted if fcntl-uncontested and >1h old.

**Shipped:** `_LOCK_RULES` declares per-directory basename patterns
matching the three known lock-using subsystems:

| Dir | Pattern | Producer |
|---|---|---|
| `/app/workspace` | `.workspace.lock` | `workspace_versioning` |
| `/app/workspace/locks` | `*.lock` | `wiki_tools` (per-slug) |
| `/app/workspace/dreams` | `.wiki_index_reconciler.lock` | `memory/wiki_index_reconciler` |

Files NOT matching their dir's pattern: surfaced via a 24h-deduped
Signal alert, never auto-deleted. Pile-up alert counts only
rule-matching candidates. See `lock_housekeeper.py:_LOCK_RULES`.

### log_archival._purge_old_archives — final retention delete with no snapshot  ✅ shipped in #116

**Before:** `LOG_ARCHIVE_RETENTION_DAYS` had a 7-day floor; per-pass
deletions unbounded.

**Shipped:** floor raised to 30 days (`_MIN_RETENTION_DAYS = 30`);
`_PURGE_PER_PASS_CAP = 100` limits damage from a misset retention —
files past retention purge oldest-first within the cap; deferred
deletions surface as `deferred_due_to_cap`; backlog still drains
naturally over multiple passes. See `log_archival.py:_purge_old_archives`.

### chromadb_hygiene — auto-VACUUM during concurrent ChromaDB writes  ✅ shipped in #117

**Before:** VACUUM accepted `database is locked` silently; a
chronically-locked KB would never reclaim disk and the operator
would never know.

**Shipped:** per-path consecutive-VACUUM-failure tracking. When a
single chroma.sqlite3 file fails VACUUM 4 quarters in a row, a
one-shot Signal alert fires with that specific path. Successful
VACUUM resets the counter. Recovery is silent (`logger.info` only)
because operator already saw the failure alert. State tracking
prevents Signal spam during a sustained streak. See
`chromadb_hygiene.py:_FAILURE_ALERT_THRESHOLD` + the new
`consecutive_failures` block in `run()`.

### db_vacuum — auto-VACUUM on conversations.db  ✅ shipped in #117

**Shipped:** same chronic-failure-tracking shape as
chromadb_hygiene. conversations.db has a single writer (Signal
client) so chronic failure here is more surprising — usually means
the writer is holding the connection through the probe window,
which is itself a bug worth alerting on. See
`db_vacuum.py:_FAILURE_ALERT_THRESHOLD`.

### tz_drift — auto-CR with empty diff  ✅ shipped in #117

**Before:** the CR filed on tz divergence passed
`new_content=src, old_content=src` (zero actual diff; operator read
the prose body and hand-wrote the fix on approval).

**Shipped:** new `_synthesize_zoneinfo_patch(src)` locates the
hand-rolled `_helsinki_tz()` function by exact signature, replaces
it with a one-liner using `ZoneInfo("Europe/Helsinki")`, and adds
`from zoneinfo import ZoneInfo` at the canonical datetime-import
anchor. Output is `ast.parse`-verified. If the source shape doesn't
match (e.g., temporal_context.py refactored before this monitor's
CR lands), the helper returns None and the caller falls back to the
informational-only shape — better than shipping a corrupted patch.
See `tz_drift.py:_synthesize_zoneinfo_patch`.

## Monitors verified SAFE

The following 18 monitors have no destructive recommendation, OR have
destructive recommendation backed by snapshot-first discipline AND
verified schema match:

`architecture_adoption`, `audit_chain_check`, `backup_freshness`,
`bit_rot_scan`, `cron_liveness`, `crypto_rotation_drill`, `db_backup`,
`drill_staleness`, `embedding_drift`, `feedback_loop_drift`,
`host_substrate_health`, `idle_cooldown`, `interest_ossification`,
`kb_contradiction`, `latency_slo`, `listener_heartbeat`,
`lock_contention`, `migration_drill` (the monitor — not the drill
script), `notify_suppression_review`, `oauth_token_freshness`,
`operator_anomaly`, `provider_contract_drift`, `restore_drill` (the
monitor), `signal_heartbeat`, `signal_keepalive`, `vendor_sunset`,
`version_upgrade_drill` (the monitor), `wiki_staleness`.

## The discipline (post-2026-05-16)

For future monitors that emit destructive recommendations:

1. **Snapshot first.** Take a tar snapshot of the targets BEFORE the
   alert is emitted. The operator should see the snapshot path in the
   alert text and have a one-line undo command.
2. **Verify-before-act.** Include a concrete shell command the
   operator can run to validate the monitor's classification against
   the current state of the world. The chromadb_hygiene bug would
   have been caught at this step (the verify command runs the same
   query the monitor used; the operator sees no orphans).
3. **Declare the assumption.** One sentence on what schema assumption
   the classification depends on. The next maintainer reads it before
   relying on the monitor's verdict.
4. **Route through `DestructiveAdvisory`.** The dataclass refuses
   construction if any of (1)-(3) is missing — discipline-enforced
   at compile time, not relied on memory of past incidents.
