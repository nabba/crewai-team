#!/usr/bin/env bash
# AndrusAI quarterly schema-migration drill (PROGRAM §48 — Q13.1).
#
# Distinct from siblings:
#   - restore-drill.sh         restores into CURRENT versions, smokes
#   - version-upgrade-drill.sh restores into NEWER versions of PG/Neo/Chroma
#   - THIS                     restores into CURRENT versions, runs
#                              app.memory.startup_migrations.apply_all
#                              (the SAME code production runs at boot),
#                              then runs schema-smoke queries that today's
#                              code would issue. Catches "today's code
#                              can't read a 6-month-old backup" — the
#                              user's exact §2.2 concern.
#
# Design note (2026-05-16): the earlier version of this script walked
# migrations/0*_*.sql and tracked applied ones in a fabricated table
# under the control_plane schema. That tracking table was NOT a
# production artifact — no production code reads or writes it — so the
# tracking was always empty and every drill re-applied every migration
# end-to-end. Several migrations (002, 003, 016, 019) are not idempotent
# on a fully-migrated schema, so those re-applications failed and the
# drill permanently recorded last_drill_ok=false.
#
# The honest test is what production actually does:
#   1. Restore the backup.
#   2. Run startup_migrations.apply_all (the production code path).
#   3. Run the smoke queries today's code would issue.
# A backup taken AFTER a migration was applied in production contains
# the resulting schema, so the smoke queries pass. A backup taken
# BEFORE a never-shipped-in-production migration won't have the new
# tables, so the smoke query for that table FAILS — which is exactly
# the operator-visible signal the drill is supposed to produce.
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

# --- Locate the freshest postgres-OK backup set -------------------------
# This drill only restores Postgres + exercises today's code against it.
# Neo4j / ChromaDB are not touched, so we filter on r.postgres.ok rather
# than r.all_ok (sibling restore-drill / version-upgrade-drill which
# restore all three correctly require all_ok).
echo ">> Locating freshest postgres-ok backup set" | tee -a "$DRILL_LOG"
SET_TS="$(python3 - <<'PYEOF'
import json
import sys
from pathlib import Path

manifest = Path("workspace/backups/manifest.json")
if not manifest.exists():
    sys.exit(0)
try:
    data = json.loads(manifest.read_text())
except Exception:
    sys.exit(1)
for r in reversed(data.get("runs", [])):
    pg = r.get("postgres") or {}
    if pg.get("ok") and pg.get("path"):
        stamp = pg["path"].rsplit("postgres-", 1)[-1].rsplit(".sql.gz", 1)[0]
        print(stamp)
        sys.exit(0)
PYEOF
)"

if [[ -z "$SET_TS" ]]; then
    echo "ERROR: no postgres-ok backup set in manifest — run deploy/scripts/backup.sh first" | tee -a "$DRILL_LOG" >&2
    exit 3
fi

PG_ARCHIVE="${REPO_ROOT}/workspace/backups/postgres/postgres-${SET_TS}.sql.gz"
if [[ ! -f "$PG_ARCHIVE" ]]; then
    echo "ERROR: postgres archive missing: $PG_ARCHIVE" | tee -a "$DRILL_LOG" >&2
    exit 3
fi
echo ">> Restoring set: $SET_TS" | tee -a "$DRILL_LOG"

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

# --- Run production startup migrations against the restored DB ----------
# This is the production code path. apply_all is idempotent (every
# operation uses IF NOT EXISTS) so it's safe to invoke against a
# fully-restored schema. If apply_all evolves to apply more than
# pgvector HNSW indexes, this drill picks up the new behaviour for
# free — that's the point: drill what production does.
#
# Run from a sidecar gateway container on the drill stack's internal
# compose network. The drill's postgres is NOT published to the host
# (publishing 5432:5432 would conflict with the live stack), so a
# host-side python3 can't reach it. Reaching `postgres:5432` from inside
# the drill's network mirrors how apply_all runs on real gateway boot.
# --no-deps keeps us from pulling chromadb/neo4j up; --entrypoint python
# bypasses /entrypoint.sh's uvicorn launch.
echo ">> Running startup_migrations.apply_all against drill DB (sidecar)" | tee -a "$DRILL_LOG"
STARTUP_OK=true
if ! $COMPOSE -p "$DRILL_PROJECT" run --rm --no-deps \
        -e MEM0_POSTGRES_HOST=postgres \
        -e MEM0_POSTGRES_PORT=5432 \
        -e MEM0_POSTGRES_USER="$PG_USER" \
        -e MEM0_POSTGRES_PASSWORD="$PG_PASS" \
        -e MEM0_POSTGRES_DB="$PG_DB" \
        -e SKIP_STARTUP_MIGRATIONS=0 \
        --entrypoint python \
        gateway -c "from app.memory.startup_migrations import apply_all; apply_all()" \
        >>"$DRILL_LOG" 2>&1; then
    STARTUP_OK=false
    echo "  ✗ startup_migrations.apply_all failed (see log)" | tee -a "$DRILL_LOG"
fi

# --- Schema-smoke queries ------------------------------------------------
# Probe tables today's code expects to find. Each query returns 0 rows
# on a fresh schema — what matters is that the table EXISTS without
# error. If the backup is from a time before a migration was applied
# in production AND that table is now expected, the SELECT errors out
# and last_drill_ok=false — which is the alert operators need.
echo ">> Schema-smoke queries against restored DB" | tee -a "$DRILL_LOG"
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
ALL_OK=$([[ "$STARTUP_OK" == "true" ]] && \
    [[ "$SMOKE_OK" == "true" ]] && \
    echo true || echo false)

python3 - "$MANIFEST_PATH" "$COMPLETED_AT" "$SET_TS" "$ALL_OK" \
    "$STARTUP_OK" "$SMOKE_OK" "$SMOKE_PROBES" "$DRILL_LOG" <<'PYEOF'
import json
import sys
from pathlib import Path

(manifest_path, completed_at, set_ts, all_ok,
 startup_ok, smoke_ok, smoke_probes, log_path) = sys.argv[1:9]

new_entry = {
    "ts": completed_at,
    "drilled_set_ts": set_ts,
    "all_ok": all_ok == "true",
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
    echo "MIGRATION DRILL OK ($SET_TS)" | tee -a "$DRILL_LOG"
    exit 0
else
    echo "MIGRATION DRILL FAILED ($SET_TS) — see $DRILL_LOG" | tee -a "$DRILL_LOG"
    exit 1
fi
