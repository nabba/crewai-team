#!/usr/bin/env bash
# AndrusAI quarterly restore drill (Phase H #1, 2026-05-10).
#
# Spins up scratch Postgres + Neo4j + ChromaDB containers in a
# separate compose project, restores the freshest "all_ok" backup
# set into them, runs smoke checks, tears down. Updates the
# manifest at workspace/backups/restore_drill_manifest.json so the
# `restore_drill` healing monitor can alert on stale drills.
#
# Run quarterly from cron / launchd (or manually):
#     bash deploy/scripts/restore-drill.sh
#
# Exits 0 on success, non-zero on any failure. Tears down on both
# paths (signal trap).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f docker-compose.yml ]]; then
    echo "ERROR: not a crewai-team checkout (docker-compose.yml missing)" >&2
    exit 2
fi

# --- Isolation overlay (required) ----------------------------------------
# Without this overlay the drill stack mounts the same host paths as
# the live stack (./workspace/mem0_pgdata, ./workspace/mem0_neo4j,
# ./workspace/memory) and corrupts the live databases when both run
# at once. See header comment in the overlay file for the full story.
DRILL_OVERLAY="docker-compose.drill-isolation.yml"
if [[ ! -f "$DRILL_OVERLAY" ]]; then
    echo "ERROR: drill overlay missing: $DRILL_OVERLAY" >&2
    echo "       Without it the drill would corrupt the live databases." >&2
    exit 2
fi

# --- Compose detection ----------------------------------------------------
if command -v docker >/dev/null 2>&1; then
    if docker compose version >/dev/null 2>&1; then
        COMPOSE="docker compose -f docker-compose.yml -f $DRILL_OVERLAY"
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE="docker-compose -f docker-compose.yml -f $DRILL_OVERLAY"
    else
        echo "ERROR: docker compose not found" >&2
        exit 2
    fi
else
    echo "ERROR: docker not found" >&2
    exit 2
fi

# --- Pre-flight: refuse if any live container exists ---------------------
# Belt-and-suspenders with the overlay. If a future operator removes or
# breaks the overlay, the bind-mount race is back; refuse to start if
# any of the three live containers exists in any state so the same
# 2026-05-16 corruption incident can't recur.
for live in crewai-team-postgres-1 crewai-team-neo4j-1 crewai-team-chromadb-1; do
    if docker inspect "$live" >/dev/null 2>&1; then
        live_state="$(docker inspect "$live" \
            --format '{{.State.Status}}' 2>/dev/null || echo unknown)"
        echo "ERROR: live container $live exists, state=$live_state." >&2
        echo "       The drill stack and live stack share docker-compose.yml;" >&2
        echo "       running both concurrently has caused live-database" >&2
        echo "       corruption in the past." >&2
        echo "       Bring the live stack down first:" >&2
        echo "         docker compose down" >&2
        echo "       Then re-run this script." >&2
        exit 4
    fi
done

# Use a separate compose project so we don't touch the live stack.
DRILL_PROJECT="andrusai-restore-drill"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DRILL_LOG="${REPO_ROOT}/workspace/backups/drill-${TS}.log"
MANIFEST_PATH="${REPO_ROOT}/workspace/backups/restore_drill_manifest.json"
mkdir -p "${REPO_ROOT}/workspace/backups"

cleanup() {
    echo ">> Tearing down drill stack" | tee -a "$DRILL_LOG"
    $COMPOSE -p "$DRILL_PROJECT" down -v 2>>"$DRILL_LOG" || true
}
trap cleanup EXIT

# --- Locate the freshest all_ok backup set --------------------------------
echo ">> Locating freshest all_ok backup set" | tee -a "$DRILL_LOG"
SET_TS="$(python3 - <<'PYEOF'
import json
import sys
from pathlib import Path

manifest = Path("workspace/backups/manifest.json")
if not manifest.exists():
    print("", end="")
    sys.exit(0)
try:
    data = json.loads(manifest.read_text())
except Exception:
    sys.exit(1)
for r in reversed(data.get("runs", [])):
    if r.get("all_ok"):
        # Prefer the run's started_at timestamp's compact form.
        ts = r.get("started_at", "").replace(":", "").replace("-", "")
        # postgres-XXXX.sql.gz file lives at r["postgres"]["path"]
        # — extract the bare timestamp.
        pg = (r.get("postgres") or {}).get("path", "")
        if pg:
            # path is "backups/postgres/postgres-YYYYMMDDTHHMMSSZ.sql.gz"
            stamp = pg.rsplit("postgres-", 1)[-1].rsplit(".sql.gz", 1)[0]
            print(stamp)
            sys.exit(0)
print("")
PYEOF
)"

if [[ -z "$SET_TS" ]]; then
    echo "ERROR: no all_ok backup set in manifest — run backup.sh first" | tee -a "$DRILL_LOG"
    exit 3
fi

PG_ARCHIVE="${REPO_ROOT}/workspace/backups/postgres/postgres-${SET_TS}.sql.gz"
NEO_ARCHIVE="${REPO_ROOT}/workspace/backups/neo4j/neo4j-${SET_TS}.dump"
CHR_ARCHIVE="${REPO_ROOT}/workspace/backups/chromadb/chromadb-${SET_TS}.tar.gz"

for archive in "$PG_ARCHIVE" "$NEO_ARCHIVE" "$CHR_ARCHIVE"; do
    if [[ ! -f "$archive" ]]; then
        echo "ERROR: archive missing: $archive" | tee -a "$DRILL_LOG"
        exit 4
    fi
done
echo "Drilling backup set: $SET_TS" | tee -a "$DRILL_LOG"

# --- Bring up drill stack -------------------------------------------------
echo ">> Bringing up drill stack" | tee -a "$DRILL_LOG"
$COMPOSE -p "$DRILL_PROJECT" up -d postgres neo4j chromadb 2>>"$DRILL_LOG"
sleep 15  # let the services finish init

PG_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q postgres)
NEO_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q neo4j)
CHR_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q chromadb)

if [[ -z "$PG_CT" || -z "$NEO_CT" || -z "$CHR_CT" ]]; then
    echo "ERROR: drill stack failed to start" | tee -a "$DRILL_LOG"
    exit 5
fi

# --- Restore Postgres -----------------------------------------------------
echo ">> Restoring Postgres" | tee -a "$DRILL_LOG"
PG_USER="${MEM0_PG_USER:-mem0}"
PG_DB="${MEM0_PG_DB:-mem0}"
PG_PASS="${MEM0_PG_PASSWORD:-${POSTGRES_PASSWORD:-}}"
gunzip -c "$PG_ARCHIVE" | docker exec -i \
    -e PGPASSWORD="$PG_PASS" "$PG_CT" \
    psql --username "$PG_USER" --dbname "$PG_DB" \
    --set ON_ERROR_STOP=1 2>>"$DRILL_LOG"

# Smoke: one count from a hot table.
PG_ROWS=$(docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" \
    psql --username "$PG_USER" --dbname "$PG_DB" -At -c \
    "SELECT count(*) FROM control_plane.audit_log;" 2>>"$DRILL_LOG" || echo "0")
echo "  Postgres audit_log rows: $PG_ROWS" | tee -a "$DRILL_LOG"

# --- Restore Neo4j --------------------------------------------------------
echo ">> Restoring Neo4j" | tee -a "$DRILL_LOG"
docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "${MEM0_NEO4J_PASSWORD:?required}" \
    "STOP DATABASE neo4j WAIT;" 2>>"$DRILL_LOG"
docker exec -i -u neo4j "$NEO_CT" \
    neo4j-admin database load neo4j --from-stdin --overwrite-destination=true \
    < "$NEO_ARCHIVE" 2>>"$DRILL_LOG"
docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "$MEM0_NEO4J_PASSWORD" \
    "START DATABASE neo4j WAIT;" 2>>"$DRILL_LOG"
NEO_NODES=$(docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "$MEM0_NEO4J_PASSWORD" --format plain \
    "MATCH (n) RETURN count(n) AS n;" 2>>"$DRILL_LOG" | tail -1 || echo "0")
echo "  Neo4j node count: $NEO_NODES" | tee -a "$DRILL_LOG"

# --- Restore ChromaDB -----------------------------------------------------
# Two corrections vs the original implementation:
#
#   1. We resolve the chroma volume's actual name from the running
#      container instead of hardcoding `${DRILL_PROJECT}_chroma`. The
#      isolation overlay (docker-compose.drill-isolation.yml) declares
#      `drill_chroma`, which compose namespaces to
#      `${DRILL_PROJECT}_drill_chroma`. Asking the container is the
#      robust way — survives future overlay renames.
#
#   2. The backup tar is created with `tar -czf ... -C ./workspace memory`
#      so it has a leading `memory/` directory. Extracting without
#      --strip-components=1 produced `/chroma/chroma/memory/<uuid>/...`
#      and chroma silently came up empty. We strip the prefix and
#      clear the volume first so the restore is deterministic.
echo ">> Restoring ChromaDB" | tee -a "$DRILL_LOG"
CHR_VOL=$(docker inspect "$CHR_CT" --format \
    '{{range .Mounts}}{{if eq .Destination "/chroma/chroma"}}{{.Name}}{{end}}{{end}}' \
    2>>"$DRILL_LOG")
if [[ -z "$CHR_VOL" ]]; then
    echo "ERROR: could not resolve chromadb volume from container" | tee -a "$DRILL_LOG" >&2
    exit 6
fi
# Stop chroma so we can swap its persistent volume contents.
$COMPOSE -p "$DRILL_PROJECT" stop chromadb 2>>"$DRILL_LOG"
docker run --rm \
    -v "${CHR_VOL}:/chroma/chroma" \
    -v "$(dirname "$CHR_ARCHIVE"):/backup:ro" \
    alpine:3 sh -c "cd /chroma/chroma && find . -mindepth 1 -delete && tar -xzf /backup/$(basename "$CHR_ARCHIVE") --strip-components=1" \
    2>>"$DRILL_LOG" || true
$COMPOSE -p "$DRILL_PROJECT" start chromadb 2>>"$DRILL_LOG"
sleep 5
CHR_PORT=$(docker port "$CHR_CT" 8000 2>/dev/null | head -1 | cut -d: -f2 || echo "")
if [[ -n "$CHR_PORT" ]]; then
    CHR_HEALTH=$(curl -fs "http://localhost:${CHR_PORT}/api/v1/heartbeat" 2>/dev/null || echo "FAIL")
else
    CHR_HEALTH="(no port mapping)"
fi
echo "  ChromaDB heartbeat: $CHR_HEALTH" | tee -a "$DRILL_LOG"

# --- Manifest update ------------------------------------------------------
COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
ALL_OK=$([[ "$PG_ROWS" =~ ^[0-9]+$ ]] && [[ "$PG_ROWS" -gt 0 ]] \
    && [[ "$CHR_HEALTH" != "FAIL" ]] \
    && echo true || echo false)

python3 - "$MANIFEST_PATH" "$COMPLETED_AT" "$SET_TS" "$ALL_OK" \
    "$PG_ROWS" "$NEO_NODES" "$CHR_HEALTH" "$DRILL_LOG" <<'PYEOF'
import json
import sys
from pathlib import Path

manifest_path, completed_at, set_ts, all_ok, pg_rows, neo_nodes, chr_health, log_path = sys.argv[1:9]
new_entry = {
    "ts": completed_at,
    "drilled_set_ts": set_ts,
    "all_ok": all_ok == "true",
    "smoke": {
        "postgres_audit_log_rows": pg_rows,
        "neo4j_node_count": neo_nodes,
        "chromadb_heartbeat": chr_health,
    },
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

# Cap history at 100 runs.
manifest["runs"] = manifest["runs"][-100:]

tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(p)
PYEOF

if [[ "$ALL_OK" == "true" ]]; then
    echo "RESTORE DRILL OK ($SET_TS)" | tee -a "$DRILL_LOG"
    exit 0
else
    echo "RESTORE DRILL FAILED ($SET_TS) — check $DRILL_LOG" | tee -a "$DRILL_LOG"
    exit 1
fi
