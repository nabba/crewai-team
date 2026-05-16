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

## What this PR leaves for follow-up

Listed in descending order of recommended priority.

### retention.run_worktrees — operator-unattended rmtree from session JSON

```python
shutil.rmtree(data.get("worktree_path"))
```

`worktree_path` is read from a per-session JSON file with no
validation. If a malformed session ever wrote a relative path or a
stale path, `rmtree` would hit the wrong target. Saved today by
`if wt and wt.exists()` guard.

Suggested fix: validate `worktree_path` is absolute, inside
`WORKSPACE_ROOT/.coding_sessions/`, and the session's `status` is in
the terminal-set BEFORE calling rmtree. Refuse if any check fails.

### retention.run_attachments — operator-unattended file unlink

Filesystem walk under `SIGNAL_ATTACHMENTS_DIR` deletes files >30d old
OR over a 1 GB total cap. Lower-risk than the other two because the
attachments dir is purpose-built for this. But still: no
operator gate, no snapshot, no dry-run by default. The
`SIGNAL_ATTACHMENTS_DIR` env-tunable is a hazard if anyone ever
points it at a wider path.

Suggested fix: refuse to operate unless `SIGNAL_ATTACHMENTS_DIR` is a
subdirectory of `WORKSPACE_ROOT`. Default `RETENTION_DRY_RUN=true` for
the first month after enabling a new attachments dir.

### lock_housekeeper — `*.lock` filename heuristic + auto-delete

Deletes `*.lock` files in three curated dirs every 6h via an fcntl
contention probe + 1h age check. The "what is a lock file" filter is
`name.endswith('.lock')`. If a future subsystem writes a non-lock
state file with `.lock` suffix, it gets deleted on the first pass.

Suggested fix: maintain an explicit allow-list of lock-file
basenames the housekeeper is allowed to delete. New lock-file
authors must opt in.

### log_archival._purge_old_archives — final retention delete with no snapshot

`workspace/healing/log_archive/*` files past N days (default 90,
env-tunable down to 7) are unlinked. The first-tier rotation IS
snapshot-first by construction (gzip-and-keep), but the final purge
isn't.

Suggested fix: route the final purge through `destructive_advisory`
so the alert shows the operator what would be purged before doing it.
Or shorten the gzip retention so the rotation IS the final stop.

### chromadb_hygiene — auto-VACUUM during concurrent ChromaDB writes

The orphan-segment side of this monitor is fixed in this PR. The
auto-VACUUM side is mature SQLite plumbing but runs unprompted every
90 days against every live `chroma.sqlite3`. ChromaDB writes through
SQLite, so a concurrent write during VACUUM could corrupt. Code uses
`timeout=30.0` and accepts lock-fail silently. Lock-fail is
recoverable but a brief gateway pause during quarterly VACUUM would
be safer.

Suggested fix: optional pre-VACUUM `docker compose stop chromadb` +
post-VACUUM `start chromadb` toggle. Default OFF so existing behavior
preserved.

### db_vacuum — auto-VACUUM on conversations.db

Similar shape: monthly VACUUM on the SQLite file the Signal client
writes to. Less risk than chromadb_hygiene because conversations.db
has a single writer.

Suggested fix: same shape — optional brief Signal-client pause around
the VACUUM, default OFF.

### tz_drift — auto-CR with empty diff

When divergence is detected, files a change request with
`new_content=src, old_content=src` (zero actual diff; the operator
must read the body and hand-write the fix). Not destructive on its
own but produces noise in the CR queue.

Suggested fix: generate a real proposed diff (e.g., a one-line
import addition that uses ZoneInfo) so the CR is meaningful. Or
demote to a Signal alert instead of a no-op CR.

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
