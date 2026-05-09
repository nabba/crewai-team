# Database Restore Runbook

Disaster-recovery procedure for the AndrusAI memory stack: **Postgres**
(authoritative ledgers — beliefs, change-requests, tickets, audit_log,
control_plane state), **Neo4j** (Mem0 graph store), **ChromaDB** (RAG
indices for KB / philosophy / fiction / episteme / experiential /
aesthetics / tensions).

Backups are produced by either:

- `deploy/scripts/backup.sh` (host-side, operator-runnable)
- `app/healing/db_backup.py` (gateway-driven; opt-in via
  `HEALING_DB_BACKUP_ENABLED=1`)

Both write to `workspace/backups/{postgres,neo4j,chromadb}/` and update
`workspace/backups/manifest.json`.

> **Read this end-to-end before touching anything in production.**
> Restore is one-way; a botched run can lose the data the running
> system is still writing. The "stop, restore, start" order matters.

## 0. Decide what you actually need

`workspace/backups/manifest.json` lists every run with per-component
`ok` flags. Identify the freshest archive set where Postgres, Neo4j,
and ChromaDB are all `ok: true`. Mismatched timestamps across stores
mean foreign-key references in Postgres can dangle into Neo4j /
ChromaDB; for normal operation, restore from the same `started_at`.

```bash
cd <repo-root>
python3 -c "
import json
m = json.load(open('workspace/backups/manifest.json'))
for r in reversed(m.get('runs', [])):
    if r.get('all_ok'):
        print(r['completed_at'])
        for k in ('postgres','neo4j','chromadb'):
            print(' ', k, r[k]['path'])
        break
"
```

If no `all_ok: true` run exists, you'll have to mix-and-match. Pick the
freshest `ok: true` per component but expect divergence: Mem0's graph
node IDs must agree with the Postgres `belief.id` they reference;
ChromaDB chunks reference document hashes that the Postgres KB table
also stores.

## 1. Stop writers

The order matters: stop everything writing to the stores before
overwriting them.

```bash
# Stop the gateway and any side-process writers.
docker compose stop gateway
# Optional but recommended — pause the cron-like idle scheduler too.
# (Already gated by gateway being down, but explicit is safer.)
```

Leave the `postgres`, `neo4j`, and `chromadb` containers **running** —
the restore commands below execute against live containers. If you
stop them, you'll need to start them back up to run the restores.

## 2. Restore Postgres

`pg_dump --clean --if-exists` was used at backup time, so the dump
already contains `DROP IF EXISTS` for every object. Direct restore
into the running database:

```bash
ARCHIVE=workspace/backups/postgres/postgres-20260509T120000Z.sql.gz
PG_CT=$(docker compose ps -q postgres)
PG_USER="${MEM0_PG_USER:-mem0}"
PG_DB="${MEM0_PG_DB:-mem0}"

gunzip -c "$ARCHIVE" \
  | docker exec -i -e PGPASSWORD="${MEM0_PG_PASSWORD:?required}" \
      "$PG_CT" psql --username "$PG_USER" --dbname "$PG_DB" \
      --set ON_ERROR_STOP=1
```

Verify:

```bash
docker exec -e PGPASSWORD="${MEM0_PG_PASSWORD}" "$PG_CT" \
  psql --username "$PG_USER" --dbname "$PG_DB" -c "
    SELECT schemaname, count(*) AS tables
      FROM pg_catalog.pg_tables
     WHERE schemaname IN ('public', 'control_plane', 'mem0')
     GROUP BY schemaname
     ORDER BY schemaname;"

# Spot-check a hot table:
docker exec -e PGPASSWORD="${MEM0_PG_PASSWORD}" "$PG_CT" \
  psql --username "$PG_USER" --dbname "$PG_DB" -c "
    SELECT count(*), max(created_at) FROM control_plane.audit_log;"
```

If `psql --set ON_ERROR_STOP=1` errored out partway through, the
database is in a half-restored state. **Drop the database and re-run
from scratch:**

```bash
# CAUTION — destroys the current database.
docker exec -e PGPASSWORD="${MEM0_PG_PASSWORD}" "$PG_CT" \
  psql --username "$PG_USER" --dbname postgres -c \
  "DROP DATABASE IF EXISTS \"$PG_DB\"; CREATE DATABASE \"$PG_DB\" OWNER \"$PG_USER\";"
gunzip -c "$ARCHIVE" \
  | docker exec -i -e PGPASSWORD="${MEM0_PG_PASSWORD}" "$PG_CT" \
      psql --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1
```

## 3. Restore Neo4j

Neo4j community-edition load requires the database to be **stopped**.
Two options.

### 3a. Online dump → offline load (recommended for Mem0 graph only)

```bash
ARCHIVE=workspace/backups/neo4j/neo4j-20260509T120000Z.dump
NEO_CT=$(docker compose ps -q neo4j)

# Neo4j data lives in /data inside the container, mounted from
# ./workspace/mem0_neo4j on the host. We swap the .dump into the
# container, stop neo4j, run the offline load, restart.
docker cp "$ARCHIVE" "$NEO_CT:/tmp/neo4j.dump"

# Stop the database (not the container — neo4j stays alive but DB stops).
docker exec -u neo4j "$NEO_CT" cypher-shell \
  -u neo4j -p "${MEM0_NEO4J_PASSWORD:?required}" \
  "STOP DATABASE neo4j WAIT;"

# Load the dump (overwrites in-place).
docker exec -u neo4j "$NEO_CT" \
  neo4j-admin database load neo4j --from-stdin --overwrite-destination=true \
  < /dev/null  # actually: pipe the dump
docker exec -i -u neo4j "$NEO_CT" \
  neo4j-admin database load neo4j --from-stdin --overwrite-destination=true \
  < "$ARCHIVE"

# Restart the database.
docker exec -u neo4j "$NEO_CT" cypher-shell \
  -u neo4j -p "${MEM0_NEO4J_PASSWORD}" \
  "START DATABASE neo4j WAIT;"
```

### 3b. Full container reset (when 3a fails)

```bash
# Stop neo4j entirely, wipe its data dir, replace, start.
docker compose stop neo4j

# Caution — destroys current Mem0 graph state.
sudo rm -rf workspace/mem0_neo4j/databases/neo4j/*
sudo rm -rf workspace/mem0_neo4j/transactions/neo4j/*

docker compose start neo4j
sleep 10  # let neo4j init the empty DB

NEO_CT=$(docker compose ps -q neo4j)
docker exec -u neo4j "$NEO_CT" cypher-shell \
  -u neo4j -p "${MEM0_NEO4J_PASSWORD}" "STOP DATABASE neo4j WAIT;"
docker exec -i -u neo4j "$NEO_CT" \
  neo4j-admin database load neo4j --from-stdin --overwrite-destination=true \
  < "$ARCHIVE"
docker exec -u neo4j "$NEO_CT" cypher-shell \
  -u neo4j -p "${MEM0_NEO4J_PASSWORD}" "START DATABASE neo4j WAIT;"
```

Verify:

```bash
docker exec -u neo4j "$NEO_CT" cypher-shell \
  -u neo4j -p "${MEM0_NEO4J_PASSWORD}" \
  "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n
     ORDER BY n DESC LIMIT 10;"
```

## 4. Restore ChromaDB

ChromaDB's persistence is just files in `workspace/memory/`. Stop the
container, swap the directory, restart.

```bash
ARCHIVE=workspace/backups/chromadb/chromadb-20260509T120000Z.tar.gz

docker compose stop chromadb

# Move (not delete) the current state — recover-the-recovery is the
# difference between an outage and an incident.
mv workspace/memory workspace/memory.pre-restore.$(date -u +%Y%m%dT%H%M%SZ)

# The archive contains a top-level `memory/` directory.
tar -xzf "$ARCHIVE" -C workspace/

docker compose start chromadb
```

Verify:

```bash
sleep 5  # chroma init
curl -fs http://localhost:8002/api/v1/heartbeat
curl -fs http://localhost:8002/api/v1/collections | python3 -m json.tool
```

If healthy, eventually delete `workspace/memory.pre-restore.*` to
reclaim disk. Keep it for at least a week — chroma's behavior on
stale embeddings can be subtle.

## 5. Bring the gateway back

```bash
docker compose start gateway

# Tail the logs for the first 60 s — schema-drift handlers and the
# auditor cron run early; if anything was ingested out-of-order it
# surfaces here.
docker compose logs -f gateway 2>&1 | head -200
```

Sanity-check from the host:

```bash
# Health endpoint must come back GREEN.
curl -fs http://localhost:8001/health

# Authenticated cp endpoint — proves DB connectivity end-to-end.
curl -fs -H "Authorization: Bearer $GATEWAY_SECRET" \
  http://localhost:8001/api/cp/audit/recent | python3 -m json.tool | head -40
```

## 6. Reconciliation

The belief outboxes (Postgres → Neo4j, Postgres → ChromaDB) and the
DLQ drainer will re-converge stores that drifted during the restore
window. Watch for residual divergence over the next 24 h:

```bash
# Outbox lag — should trend to 0.
curl -fs -H "Authorization: Bearer $GATEWAY_SECRET" \
  http://localhost:8001/api/cp/outbox/status | python3 -m json.tool

# DLQ depth.
curl -fs -H "Authorization: Bearer $GATEWAY_SECRET" \
  http://localhost:8001/api/cp/dlq/status | python3 -m json.tool
```

If outbox lag stays positive for >2 hours, run the manual reconciler
(see `crewai-team/docs/MEMORY_ARCHITECTURE.md` §belief-outbox).

## 7. Post-restore audit

Once stable, write an incident record:

```bash
python3 -c "
from app.life_companion._common import audit_event
audit_event(
    'db_restore',
    archive_set='20260509T120000Z',
    postgres_ok=True, neo4j_ok=True, chromadb_ok=True,
    operator='you@example.com',
    notes='Restore from <reason>; full set restored per RESTORE.md',
)
"
```

This entry lives in the hash-chained Postgres `control_plane.audit_log`
and is the source of truth for "what happened."

## Drill cadence

Restore is a procedure, not a fire-drill. Run a **dry restore on a
scratch compose stack** every quarter:

```bash
# In a temporary worktree:
git worktree add /tmp/restore-drill main
cd /tmp/restore-drill
cp /path/to/.env .
cp -r /path/to/workspace/backups workspace/backups
docker compose up -d postgres neo4j chromadb
bash deploy/scripts/restore-drill.sh   # not yet shipped — TODO
```

A clean drill = procedure works. A failed drill = fix the procedure
**before** you need it for real.

## Failure modes seen in practice

| Symptom | Cause | Recovery |
| --- | --- | --- |
| `psql: FATAL: role "mem0" does not exist` | Drop-and-recreate path (`§2`) skipped role grant | Re-run `CREATE ROLE mem0 WITH LOGIN PASSWORD '...'` then retry |
| `Unable to connect to neo4j` after 3a | DB still STOPPED after load | Run the explicit `START DATABASE neo4j WAIT;` |
| Chroma returns empty results post-restore | tar excluded `*.lock` (intentional) but chroma re-init didn't complete | Restart the chromadb container; check logs |
| Mem0 errors with "cannot find belief X" | Postgres restored but Neo4j is older — mismatched archive set | Restore Neo4j from the matching archive set; or accept the loss and let outbox re-emit |
