# ChromaDB integrity protection

**PROGRAM §55 (2026-05-17)** — defense-in-depth layer added after the
dual-writer SQLite corruption events of **2026-04-25** and
**2026-05-17** wiped the gateway's `memory/` KB. This doc covers what
was done, what's in place now, and the operator runbook for the
remaining failure modes.

## Root cause (closed)

`docker-compose.yml` defined two services that both wrote to
`workspace/memory/chroma.sqlite3`:

| Container | Mount | chromadb version | Writer |
|---|---|---|---|
| `chromadb` (HTTP server) | `./workspace/memory:/chroma/chroma` | 0.5.23 | server-internal PersistentClient |
| `gateway` (Python) | `./workspace:/app/workspace` | 1.5.9 | `chromadb.PersistentClient(path=".../memory")` |

The chromadb service was an orphan from before commit
`f240d54e` ("Fix 7 modules using broken `chromadb.HttpClient` →
`get_client()`"), which migrated the gateway off the HTTP API and
onto the embedded PersistentClient. Nothing in `app/` has used the
HTTP API since; in fact `tests/test_subsystem_wiring.py:385` asserts
`chromadb.HttpClient(` must not appear in production code. The
container kept running with the same bind-mount and silently
corrupted the SQLite file roughly once every three weeks.

**Fix:** remove the chromadb service from `docker-compose.yml`. Done
in this PR.

**Pin:** `tests/test_chromadb_integrity.py::test_docker_compose_has_no_chromadb_service`
fails CI if the service or bind-mount is ever restored.

## Defense layers

Removing the dual-writer eliminates the specific bug. The protection
layer below catches the next class of damage (unclean restart, journal
recovery anomaly, silent btree damage).

### 1. WAL journaling enforcement (boot)

Every `workspace/<kb>/chroma.sqlite3` is set to
`journal_mode=WAL; synchronous=FULL` at gateway boot. WAL survives
crashes significantly better than the default rollback-journal mode
that every chromadb KB ships with. Persistent across connections, so
this only needs to run once per boot.

Master switch: `chromadb_wal_enforcement_enabled` (default ON).

### 2. Boot integrity scan (`app/main.py` lifespan)

After WAL enforcement, runs `PRAGMA integrity_check` on every KB.
Any non-`ok` result triggers:

1. **Quarantine**: `workspace/<kb>/` → `workspace/<kb>.corrupt_<ts>/`
   (matches the manual rename precedent on disk).
2. **Fresh slate**: empty replacement directory created so chromadb
   can start a new database on the next open.
3. **Signal alert**: tag-keyed `chromadb_integrity:<kb>` for arbiter
   dedup.
4. **Identity-continuity ledger event**: new `chromadb_corruption`
   event kind, automatically surfaced via `summarise_drift` Counter
   into the annual reflection.
5. **Auto-replay (memory KB only)**: re-embed rows from postgres
   source-of-truth tables `beliefs` and `crewai_memories` back into
   the freshly-created collections. Idempotent on `(table, row_id)`.

Master switch: `chromadb_boot_integrity_check_enabled` (default ON).
Auto-replay sub-gate: `chromadb_auto_replay_enabled` (default ON).

### 3. Daily integrity monitor + snapshot (35th healing monitor)

`app/healing/monitors/chromadb_integrity.py` runs every 23 h (daily
probe with internal cadence guard). Two branches per probe:

- **Integrity branch** — same routine as the boot scan; catches damage
  that appeared during a long-running session.
- **Snapshot branch** — `sqlite3.Connection.backup()` to
  `workspace/<kb>/.sqlite_snapshots/<ts>.db`. Keeps the last 7 days.

Snapshot is **never taken on a known-bad file** (pinned by
`test_monitor_does_not_snapshot_corrupt_files`).

Master switches:
- `chromadb_integrity_monitor_enabled` (default ON, integrity branch)
- `chromadb_daily_snapshot_enabled` (default ON, snapshot branch)
- `HEALING_MONITORS_ENABLED` (umbrella for all 35 monitors)

### 4. Snapshot retention

7-day rolling window. Pruning is mtime-based (survives clock skew and
manual file copies). Disk impact: `memory/` chroma.sqlite3 is ~2 MB,
so 7 snapshots = ~14 MB worst case per KB. Other KBs scale to ~150 MB
across the full set.

## Operator runbook

### "ChromaDB integrity failure on `<kb>`" Signal alert

What it means: `PRAGMA integrity_check` flagged structural damage on
that KB. The KB has already been quarantined to
`workspace/<kb>.corrupt_<ts>/` and replaced with a fresh empty one.

If `<kb> == memory`:
- Auto-replay has fired. Check `workspace/healing/chromadb_integrity.json`
  for `last_summary.per_kb[memory].replay`.
- `total_added` rows were re-embedded from postgres. The rest is gone
  unless you have a recent snapshot (see below).

If `<kb> != memory`:
- The replacement is empty. The KB's normal repopulation pipeline
  (wiki crawl for `episteme`, `philosophy`, `knowledge`; aesthetic-KB
  populator for `aesthetics`; etc.) will rebuild on its next pass.

### Restore from a daily snapshot

```bash
# List available snapshots for the KB
ls -la workspace/<kb>/.sqlite_snapshots/

# Stop the gateway briefly (snapshot restore needs an exclusive lock)
docker compose stop gateway

# Restore (creates a new chroma.sqlite3 from the snapshot — the
# HNSW segment dirs aren't snapshotted, chromadb rebuilds them on
# first query)
cp workspace/<kb>/.sqlite_snapshots/<ts>.db workspace/<kb>/chroma.sqlite3

# Restart
docker compose up -d gateway
```

### Manually run an integrity check

```bash
# Inside the gateway container, or with PYTHONPATH set:
python -c "
from pathlib import Path
from app.memory.chromadb_integrity import chromadb_kbs, integrity_check
for db in chromadb_kbs():
    print(db.parent.name, integrity_check(db))
"
```

### Read the current state

```bash
cat workspace/healing/chromadb_integrity.json
```

Shows last run timestamp, per-KB verdict, snapshot info, and the
running per-KB alert dedup state.

### Trigger an immediate replay from postgres

```bash
python -m app.memory.chromadb_integrity replay memory
# or:
python -c "from app.memory.chromadb_integrity import replay_from_postgres; print(replay_from_postgres('memory'))"
```

## What did NOT change

- chromadb 1.1.1 stays as the gateway's embedded library — same
  process, same SQLite-on-disk format.
- Existing `chromadb_hygiene` monitor (quarterly VACUUM) still runs.
  It composes with the new monitor — integrity catches damage,
  hygiene reclaims freelist space.
- HNSW segment dirs (per-collection UUID subdirs) are not
  snapshotted. They're rebuildable on first query from the
  metadata DB.
- Postgres mirror completeness — `beliefs` (6 rows) and
  `crewai_memories` (160 rows) are the only re-embeddable sources.
  Collections like `team_shared`, `commander`, `learning_gaps`,
  `scope_team`, `result_cache` that were in the 347 MB corrupt
  memory KB and have no postgres mirror are not auto-recoverable
  — they rebuild over time from agent activity.

## Re-open conditions

The protection layer is observational and shouldn't need
re-evaluation routinely. Re-audit only if any of these occur:

1. A third `memory.corrupt_*` quarantine event despite this layer.
   → Investigate whether some other dual-writer pattern slipped in
   (e.g. a script running outside the container against the same DB).
2. Boot scan starts failing for non-corruption reasons.
   → The failure-isolated try/except in `app/main.py` should keep
   boot going, but the alert chain may need tuning.
3. The 7-day snapshot retention proves too short (operator wanted
   to restore an older state). Reconsider with operator input.
4. Postgres mirror schema changes break `replay_from_postgres`.
   → The replay function fails-soft per-row; revisit the column
   list in `_replay_beliefs` / `_replay_crewai_memories`.

## Test inventory

`tests/test_chromadb_integrity.py` — 19 tests (16 pass + 3 skipped
gateway-deps):

| Test | What it pins |
|---|---|
| `test_chromadb_kbs_finds_live_and_skips_quarantined` | discovery skips `*.corrupt_*` dirs |
| `test_chromadb_kbs_skips_bak_and_backup_directories` | also skips `*.bak_*` / `_backup` / `.backup` |
| `test_enforce_wal_mode_sets_wal_and_full_sync` | WAL + synchronous=FULL applied + persistent |
| `test_enforce_wal_mode_idempotent` | re-running on a WAL DB stays WAL |
| `test_integrity_check_passes_on_healthy_db` | clean DB returns "ok" |
| `test_integrity_check_catches_corruption` | trashed btree page → non-"ok" verdict |
| `test_integrity_check_reports_missing` | absent file → "missing" |
| `test_daily_snapshot_creates_atomic_backup` | snapshot is openable + has data |
| `test_daily_snapshot_prunes_old_snapshots` | mtime-based retention works |
| `test_daily_snapshot_missing_source` | absent source → graceful error |
| `test_quarantine_kb_renames_and_creates_fresh` | rename + ledger emit + fresh dir |
| `test_quarantine_kb_handles_missing_dir` | absent dir → graceful error |
| `test_boot_integrity_scan_clean_workspace` | healthy KBs → no quarantines |
| `test_boot_integrity_scan_quarantines_corrupt` | damaged KB → quarantined + replay attempted |
| `test_boot_integrity_scan_disabled_no_op` | master switch OFF → scan skipped |
| `test_monitor_respects_cadence` | second run within 23h is a no-op |
| `test_monitor_does_not_snapshot_corrupt_files` | snapshot never taken on a damaged file |
| `test_runtime_settings_chromadb_keys_default_on` | failure-OPEN posture pinned |
| `test_docker_compose_has_no_chromadb_service` | no regression of dual-writer config |
