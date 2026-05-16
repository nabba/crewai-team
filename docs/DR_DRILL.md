# DR Drill — portable export + import + boot drill

> PROGRAM §40 (2026-05-10) — Q3 Item 13.

## What this is

Every system claims it has "backups". Few systems prove it. This is
the drill that proves it: a fully-automatic exercise that takes the
latest backup, restores it into a sandbox, and runs sanity queries
to verify the restore is functional.

The drill answers a single question: **"Could we rebuild from a
backup right now?"** It runs without touching the live workspace.

## Layered backup story

The system has two complementary backup paths:

| Layer | Producer | Format | Best for |
|------|----------|--------|----------|
| Container-resident | Host LaunchAgent + `app/healing/db_backup.py` | `pg_dump` + `neo4j-admin database dump` + chromadb tarball | Same-cluster restore (e.g. K8s rollback) |
| Portable | `app/dr/export_kbs.py` | Self-contained `.tar.gz` with JSONL exports of every KB | Cold-restore on a fresh laptop with NO running cluster |

The container-resident layer is split (2026-05-16): the host launchd
LaunchAgent `org.andrus.botarmy.db-backup` (installed via
`scripts/install_db_backup.sh`) handles Postgres + Neo4j via
`deploy/scripts/backup.sh`, while the gateway-side
`app/healing/db_backup.py` handles ChromaDB. Both write to the same
`workspace/backups/manifest.json` with per-component `ok` + `skipped`
flags.

The DR drill (this doc) validates the **portable** layer. The
binary backup is verified by the existing `restore_drill` healing
monitor.

## What's in the portable tarball

```
dr_<timestamp>.tar.gz
├── manifest.json                                # rows + bytes per artifact
├── chromadb/
│   ├── memory/team_shared.jsonl.gz              # one row per line
│   ├── philosophy/<collection>.jsonl.gz
│   ├── …
├── postgres/
│   ├── control_plane__audit_log.jsonl.gz
│   ├── control_plane__budgets.jsonl.gz
│   ├── …
└── workspace_ledgers/
    ├── affect/trace.jsonl
    ├── affect/archive/2026-04_trace.jsonl       # archive-rotated history
    ├── affect/welfare_audit.jsonl
    ├── identity/continuity_ledger.jsonl
    ├── audit_journal/…
    └── …
```

### What's deliberately EXCLUDED

The export aggressively excludes anything that could be a secret,
even if it lives under an allowed root:

* `.env` and any `.env.*`
* `secrets/` (any directory under that name)
* `google_token.json` (OAuth refresh)
* `vapid_*.pem` (Web Push private keys)
* Any path containing `token`, `credential`, `private_key`,
  `client_secret`

The exclusion list is enforced in
[`app/dr/export_kbs.py`](../app/dr/export_kbs.py)
(`_is_secret_path`). Operators verifying a tarball should check
`manifest.excluded_secret_paths` to confirm the exclusions fired.

Also out of scope (re-fillable or huge binary blobs):

* `cache/` — LLM response cache; rebuilds itself.
* `coding_sessions/.worktrees/` — ephemeral.
* `training_adapters/` — multi-GB LoRA blobs; have their own cadence.

## Operator commands

```bash
# Run an export by hand (writes to workspace/backups/dr/)
python -m app.dr.export_kbs

# Run the drill on the latest export
scripts/dr_boot_drill.sh

# Run a fresh export + drill
scripts/dr_boot_drill.sh --export-fresh

# Keep the sandbox restored dir on disk for inspection
scripts/dr_boot_drill.sh --keep-target

# Restore one KB into a sandbox for forensic inspection
python -m app.dr.import_kbs \
  --tarball workspace/backups/dr/dr_2026-05-11T060000Z.tar.gz \
  --target-dir /tmp/dr_inspect
```

The drill writes a JSON report to `workspace/dr/drill_<timestamp>.json`
and emits a Signal alert summarising the outcome.

## What the drill verifies

For each restored ChromaDB collection:

1. **`count()` matches the manifest** (within ±1 — tiny race during
   export is fine).
2. **`peek(1)`** returns a row with the pinned embedding dimension
   (catches dimension drift that `chromadb_hygiene` would miss).
3. **One smoke `query()`** with the peeked embedding returns ≥1 row
   (catches HNSW segment corruption).

For the workspace ledgers: file count + total bytes restored,
recorded in the drill report.

## Cadence

The portable export does not yet run on its own cadence. **Operator
runs it manually before destructive work** (e.g. before a Tier-3
amendment, before a substrate migration like the Item 12 embedding
swap). Future iteration may schedule it weekly via the healing
daemon, paired to the existing `restore_drill` monitor.

The binary backup runs weekly (when `HEALING_DB_BACKUP_ENABLED=true`).

## Failure modes + recovery

The drill never raises — it always writes a report. Three failure
classes:

* **No tarball available** — drill exits with `overall_ok=false`,
  one error: `"no tarball available under …"`. Run with
  `--export-fresh` to fix.
* **Import error** — the tarball is corrupt or truncated.
  `import_summary.errors` lists the files that failed. The drill
  still verifies whatever did make it across.
* **Query error** — collection restored but ChromaDB can't query
  it. Most common cause: embedding dimension mismatch (you imported
  a 384-dim collection into a 768-dim runtime). Run
  `python -m app.memory.chromadb_rebuild --kb <kb> --collection <name>`
  on the live workspace to remediate.

## Auditing a tarball before restore

Always inspect the manifest before doing anything destructive:

```bash
TARBALL=workspace/backups/dr/dr_2026-05-11T060000Z.tar.gz
tar -xOf $TARBALL manifest.json | jq '{
    duration_s, total_rows_chromadb, total_rows_postgres,
    chromadb_count: (.chromadb | length),
    postgres_count: (.postgres | length),
    ledger_count: (.ledgers | length),
    excluded_secret_paths: (.excluded_secret_paths | length),
    errors: .errors
}'
```

If `excluded_secret_paths` is `0` and you expect secrets to be
present in the workspace, that's a red flag — the path-fragment
denylist may have missed them, or they may not exist on disk.
Investigate before publishing the tarball.

## Why this composes with the existing layers

* `db_backup.py` covers same-cluster K8s restore. It produces fast
  binary dumps but **assumes the destination cluster has the same
  Postgres major version, the same Neo4j version, etc.** That's
  often wrong after a long enough time horizon.
* `app/dr/` produces a **container-independent** tarball driven only
  by application APIs. The boot drill validates that path. Together
  they answer both "fast restore in production" and "cold restore
  from a tarball on a fresh laptop."
* The drill's Signal alert lives outside the existing alert noise —
  separate `tag="dr_drill"` so the operator can filter it for a
  weekly review.
