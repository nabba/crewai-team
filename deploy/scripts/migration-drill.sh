#!/usr/bin/env bash
# AndrusAI quarterly schema-migration drill (PROGRAM §48 — Q13.1).
#
# Distinct from siblings:
#   - restore-drill.sh         restores into CURRENT versions, smokes
#   - version-upgrade-drill.sh restores into NEWER versions of PG/Neo/Chroma
#   - THIS                     restores into CURRENT versions, then APPLIES
#                              every pending migrations/*.sql, then runs
#                              startup_migrations.apply_all, then runs
#                              schema-smoke queries that newer migrations
#                              created. Catches "today's code can't read
#                              a 6-month-old backup" — the user's exact
#                              §2.2 concern.
#
# Run quarterly from cron / launchd. Updates a separate manifest at
# workspace/backups/migration_drill_manifest.json so the
# `migration_drill` healing monitor can alert on staleness.
#
# Exits 0 on success, non-zero on failure. Always tears down on exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f docker-compose.yml ]]; then
    echo "ERROR: not a crewai-team checkout (docker-compose.yml missing)" >&2
    exit 2
fi

# --- Compose detection ----------------------------------------------------
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        COMPOSE="docker compose"
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE="docker-compose"
    else
        echo "ERROR: docker compose not found" >&2
        exit 2
    fi
else
    echo "ERROR: docker not found" >&2
    exit 2
fi

DRILL_PROJECT="andrusai-migration-drill"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DRILL_LOG="${REPO_ROOT}/workspace/backups/migration-drill-${TS}.log"
MANIFEST_PATH="${REPO_ROOT}/workspace/backups/migration_drill_manifest.json"
mkdir -p "${REPO_ROOT}/workspace/backups"

cleanup() {
    echo ">> Tearing down drill stack" | tee -a "$DRILL_LOG"
    $COMPOSE -p "$DRILL_PROJECT" down -v --remove-orphans \
        >>"$DRILL_LOG" 2>&1 || true
}
trap cleanup EXIT INT TERM

echo ">> Migration drill ${TS}" | tee "$DRILL_LOG"
echo ">> Log file: $DRILL_LOG"

# --- Find the freshest backup -------------------------------------------
BACKUP_BASE="${REPO_ROOT}/workspace/backups/dr"
if [[ ! -d "$BACKUP_BASE" ]]; then
    echo "ERROR: no DR backup directory at $BACKUP_BASE" | tee -a "$DRILL_LOG" >&2
    exit 3
fi
LATEST_TARBALL="$(ls -t "$BACKUP_BASE"/*.tar.gz 2>/dev/null | head -1 || true)"
if [[ -z "$LATEST_TARBALL" ]]; then
    echo "ERROR: no .tar.gz under $BACKUP_BASE" | tee -a "$DRILL_LOG" >&2
    exit 3
fi
SET_TS="$(basename "$LATEST_TARBALL" .tar.gz)"
echo ">> Restoring set: $SET_TS" | tee -a "$DRILL_LOG"

# Extract into a workdir.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/migration-drill-XXXXXX")"
trap 'rm -rf "$WORK_DIR"; cleanup' EXIT INT TERM
tar -xzf "$LATEST_TARBALL" -C "$WORK_DIR" 2>>"$DRILL_LOG"

PG_ARCHIVE="$(find "$WORK_DIR" -name "postgres*.sql.gz" | head -1)"
if [[ -z "$PG_ARCHIVE" ]]; then
    echo "ERROR: no postgres dump in $LATEST_TARBALL" | tee -a "$DRILL_LOG" >&2
    exit 3
fi

# --- Spin up scratch Postgres (CURRENT version) -------------------------
echo ">> Starting scratch Postgres (current version)" | tee -a "$DRILL_LOG"
PG_USER="${MEM0_PG_USER:-mem0}"
PG_DB="${MEM0_PG_DB:-mem0}"
PG_PASS="${MEM0_PG_PASSWORD:-${POSTGRES_PASSWORD:-test_password}}"

$COMPOSE -p "$DRILL_PROJECT" up -d postgres >>"$DRILL_LOG" 2>&1
PG_CT="$($COMPOSE -p "$DRILL_PROJECT" ps -q postgres)"
sleep 10

# Restore the backup.
echo ">> Restoring Postgres dump" | tee -a "$DRILL_LOG"
gunzip -c "$PG_ARCHIVE" | docker exec -i \
    -e PGPASSWORD="$PG_PASS" "$PG_CT" \
    psql --username "$PG_USER" --dbname "$PG_DB" \
    --set ON_ERROR_STOP=1 2>>"$DRILL_LOG"

# --- Apply pending migrations -------------------------------------------
echo ">> Walking migrations/*.sql and applying pending" | tee -a "$DRILL_LOG"

# Track which migrations have been applied. Match the codebase pattern
# from conversation_store.py:128 (a _schema_version table) so the same
# convention applies whether the snapshot recorded it or not. Idempotent:
# already-applied migrations have CREATE TABLE IF NOT EXISTS guards.
docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
    --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1 \
    -c "CREATE SCHEMA IF NOT EXISTS control_plane;" 2>>"$DRILL_LOG"
docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
    --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1 \
    -c "CREATE TABLE IF NOT EXISTS control_plane._schema_migrations (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());" 2>>"$DRILL_LOG"

MIGRATIONS_APPLIED=0
MIGRATIONS_SKIPPED=0
MIGRATIONS_FAILED=0
for sql_file in "$REPO_ROOT"/migrations/0*_*.sql; do
    [[ ! -f "$sql_file" ]] && continue
    name="$(basename "$sql_file" .sql)"
    # Skip if already applied (the snapshot may have recorded it).
    applied="$(docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
        --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1 \
        -At -c "SELECT 1 FROM control_plane._schema_migrations WHERE name='$name';" \
        2>/dev/null || echo "")"
    if [[ "$applied" == "1" ]]; then
        MIGRATIONS_SKIPPED=$((MIGRATIONS_SKIPPED + 1))
        continue
    fi
    echo "  applying $name..." | tee -a "$DRILL_LOG"
    if cat "$sql_file" | docker exec -i -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
            --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1 \
            >>"$DRILL_LOG" 2>&1; then
        docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
            --username "$PG_USER" --dbname "$PG_DB" --set ON_ERROR_STOP=1 \
            -c "INSERT INTO control_plane._schema_migrations(name) VALUES ('$name') ON CONFLICT DO NOTHING;" \
            2>>"$DRILL_LOG"
        MIGRATIONS_APPLIED=$((MIGRATIONS_APPLIED + 1))
    else
        echo "  ✗ FAILED: $name (see log)" | tee -a "$DRILL_LOG"
        MIGRATIONS_FAILED=$((MIGRATIONS_FAILED + 1))
    fi
done
echo ">> Migrations: applied=$MIGRATIONS_APPLIED skipped=$MIGRATIONS_SKIPPED failed=$MIGRATIONS_FAILED" | tee -a "$DRILL_LOG"

# --- Apply startup migrations (inlined pgvector indexes etc.) -----------
echo ">> Running startup_migrations.apply_all against drill DB" | tee -a "$DRILL_LOG"
STARTUP_OK=true
if ! MEM0_POSTGRES_URL="postgresql://${PG_USER}:${PG_PASS}@localhost:$(docker port "$PG_CT" 5432 | head -1 | cut -d: -f2)/${PG_DB}" \
        python3 -c "from app.memory.startup_migrations import apply_all; apply_all()" \
        >>"$DRILL_LOG" 2>&1; then
    STARTUP_OK=false
    echo "  ✗ startup_migrations.apply_all failed (see log)" | tee -a "$DRILL_LOG"
fi

# --- Schema-smoke queries ------------------------------------------------
# Probe tables created by recent migrations (030+) to confirm
# today's code would find them. Each query returns 0 rows on a
# fresh schema — what matters is that the table EXISTS without error.
echo ">> Schema-smoke queries against migrated DB" | tee -a "$DRILL_LOG"
SMOKE_OK=true
SMOKE_PROBES=""
for probe in \
    "control_plane.audit_log" \
    "control_plane.crew_tasks" \
    "control_plane.tickets" \
    "control_plane.epistemic_peer_reviews" \
    "control_plane.epistemic_overrides" \
    "control_plane.error_anomalies" \
; do
    rows="$(docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" psql \
        --username "$PG_USER" --dbname "$PG_DB" -At \
        -c "SELECT count(*) FROM $probe;" 2>>"$DRILL_LOG" || echo "FAIL")"
    if [[ "$rows" == "FAIL" ]] || [[ ! "$rows" =~ ^[0-9]+$ ]]; then
        SMOKE_OK=false
        echo "  ✗ $probe: query failed" | tee -a "$DRILL_LOG"
    else
        echo "  ✓ $probe: $rows rows" | tee -a "$DRILL_LOG"
    fi
    SMOKE_PROBES="${SMOKE_PROBES}${probe}=${rows};"
done

# --- Manifest update ------------------------------------------------------
COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
ALL_OK=$([[ "$MIGRATIONS_FAILED" == "0" ]] && \
    [[ "$STARTUP_OK" == "true" ]] && \
    [[ "$SMOKE_OK" == "true" ]] && \
    echo true || echo false)

python3 - "$MANIFEST_PATH" "$COMPLETED_AT" "$SET_TS" "$ALL_OK" \
    "$MIGRATIONS_APPLIED" "$MIGRATIONS_SKIPPED" "$MIGRATIONS_FAILED" \
    "$STARTUP_OK" "$SMOKE_OK" "$SMOKE_PROBES" "$DRILL_LOG" <<'PYEOF'
import json
import sys
from pathlib import Path

(manifest_path, completed_at, set_ts, all_ok,
 mig_applied, mig_skipped, mig_failed, startup_ok,
 smoke_ok, smoke_probes, log_path) = sys.argv[1:12]

new_entry = {
    "ts": completed_at,
    "drilled_set_ts": set_ts,
    "all_ok": all_ok == "true",
    "migrations": {
        "applied": int(mig_applied),
        "skipped": int(mig_skipped),
        "failed": int(mig_failed),
    },
    "startup_migrations_ok": startup_ok == "true",
    "smoke_ok": smoke_ok == "true",
    "smoke_probes": dict(
        kv.split("=", 1) for kv in smoke_probes.split(";") if "=" in kv
    ),
    "log": log_path,
}

p = Path(manifest_path)
if p.exists():
    try:
        manifest = json.loads(p.read_text())
    except Exception:
        manifest = {"runs": []}
else:
    manifest = {"runs": []}
manifest.setdefault("runs", []).append(new_entry)
manifest["last_drill_at"] = completed_at
manifest["last_drill_ok"] = new_entry["all_ok"]

manifest["runs"] = manifest["runs"][-100:]

tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(p)
PYEOF

if [[ "$ALL_OK" == "true" ]]; then
    echo "MIGRATION DRILL OK ($SET_TS, applied=$MIGRATIONS_APPLIED)" | tee -a "$DRILL_LOG"
    exit 0
else
    echo "MIGRATION DRILL FAILED ($SET_TS) — see $DRILL_LOG" | tee -a "$DRILL_LOG"
    exit 1
fi
