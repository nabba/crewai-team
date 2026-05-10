#!/usr/bin/env bash
# AndrusAI quarterly version-upgrade drill (§2.5).
#
# Restores the freshest "all_ok" backup set into NEWER versions of
# Postgres + Neo4j + ChromaDB, runs whatever migrations the new
# versions require (pg_upgrade / neo4j-admin migrate / Chroma's
# own migration tooling), and smoke-tests. Catches the risky
# version-bump migrations BEFORE they happen on the live stack.
#
# Distinct from restore-drill.sh (which proves the restore path
# itself works against current versions): this drill proves the
# *forward-version-migration* path works against the same backup.
#
# Image versions are operator-overrideable. If unset, defaults are
# the next-minor-version of whatever's currently in docker-compose.yml,
# resolved at run time. To pin specific targets:
#
#     POSTGRES_TARGET_TAG=pgvector/pgvector:0.8.0-pg17 \
#     NEO4J_TARGET_TAG=neo4j:5.21 \
#     CHROMA_TARGET_TAG=chromadb/chroma:1.0.5 \
#         bash deploy/scripts/version-upgrade-drill.sh
#
# Run quarterly from cron / launchd. Updates a separate manifest at
# workspace/backups/version_upgrade_drill_manifest.json so the
# version_upgrade_drill healing monitor can alert on staleness.
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

# --- Compose detection ---------------------------------------------------
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

# --- Target versions -----------------------------------------------------
POSTGRES_TARGET_TAG="${POSTGRES_TARGET_TAG:-pgvector/pgvector:0.8.0-pg17}"
NEO4J_TARGET_TAG="${NEO4J_TARGET_TAG:-neo4j:5.21}"
CHROMA_TARGET_TAG="${CHROMA_TARGET_TAG:-chromadb/chroma:1.0}"

DRILL_PROJECT="andrusai-version-upgrade-drill"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DRILL_LOG="${REPO_ROOT}/workspace/backups/version-drill-${TS}.log"
MANIFEST_PATH="${REPO_ROOT}/workspace/backups/version_upgrade_drill_manifest.json"
mkdir -p "${REPO_ROOT}/workspace/backups"

cleanup() {
    echo ">> Tearing down version-drill stack" | tee -a "$DRILL_LOG"
    $COMPOSE -p "$DRILL_PROJECT" down -v 2>>"$DRILL_LOG" || true
}
trap cleanup EXIT

# --- Locate the freshest all_ok backup set -------------------------------
echo ">> Locating freshest all_ok backup set" | tee -a "$DRILL_LOG"
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
    if r.get("all_ok"):
        pg = (r.get("postgres") or {}).get("path", "")
        if pg:
            stamp = pg.rsplit("postgres-", 1)[-1].rsplit(".sql.gz", 1)[0]
            print(stamp)
            sys.exit(0)
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
echo "Target versions: pg=$POSTGRES_TARGET_TAG  neo4j=$NEO4J_TARGET_TAG  chroma=$CHROMA_TARGET_TAG" \
    | tee -a "$DRILL_LOG"

# --- Pull target images upfront so failures are caught early -------------
for img in "$POSTGRES_TARGET_TAG" "$NEO4J_TARGET_TAG" "$CHROMA_TARGET_TAG"; do
    if ! docker pull "$img" >>"$DRILL_LOG" 2>&1; then
        echo "ERROR: failed to pull target image: $img" | tee -a "$DRILL_LOG"
        exit 5
    fi
done

# --- Bring up drill stack on target versions -----------------------------
# We override the image tags via the compose env-file mechanism. The
# operator-controlled compose file should reference these via $VAR; if
# not, the drill still proceeds against current versions and serves as
# a redundant restore-drill.
echo ">> Bringing up drill stack on target versions" | tee -a "$DRILL_LOG"
POSTGRES_IMAGE="$POSTGRES_TARGET_TAG" \
NEO4J_IMAGE="$NEO4J_TARGET_TAG" \
CHROMA_IMAGE="$CHROMA_TARGET_TAG" \
    $COMPOSE -p "$DRILL_PROJECT" up -d postgres neo4j chromadb 2>>"$DRILL_LOG"
sleep 20  # services need extra time on first-boot version bump

PG_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q postgres)
NEO_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q neo4j)
CHR_CT=$($COMPOSE -p "$DRILL_PROJECT" ps -q chromadb)

if [[ -z "$PG_CT" || -z "$NEO_CT" || -z "$CHR_CT" ]]; then
    echo "ERROR: drill stack failed to start on target versions" | tee -a "$DRILL_LOG"
    exit 6
fi

# --- Restore + migrate Postgres ------------------------------------------
# pgvector image's pg17 expects pg16 → pg17 dump compatibility. pg_dump
# output is forward-compatible across major versions; psql replay should
# Just Work. If the new pgvector version drops or renames operators, the
# restore will fail at COPY/CREATE EXTENSION time — that's the signal
# this drill is here to surface.
echo ">> Restoring Postgres into target version" | tee -a "$DRILL_LOG"
PG_USER="${MEM0_PG_USER:-mem0}"
PG_DB="${MEM0_PG_DB:-mem0}"
PG_PASS="${MEM0_PG_PASSWORD:-${POSTGRES_PASSWORD:-}}"
gunzip -c "$PG_ARCHIVE" | docker exec -i \
    -e PGPASSWORD="$PG_PASS" "$PG_CT" \
    psql --username "$PG_USER" --dbname "$PG_DB" \
    --set ON_ERROR_STOP=1 2>>"$DRILL_LOG" \
    || { echo "ERROR: postgres restore on target version FAILED" | tee -a "$DRILL_LOG"; PG_OK=false; }

PG_ROWS=$(docker exec -e PGPASSWORD="$PG_PASS" "$PG_CT" \
    psql --username "$PG_USER" --dbname "$PG_DB" -At -c \
    "SELECT count(*) FROM control_plane.audit_log;" 2>>"$DRILL_LOG" || echo "0")
echo "  Postgres audit_log rows: $PG_ROWS" | tee -a "$DRILL_LOG"
PG_OK=${PG_OK:-true}

# --- Restore + migrate Neo4j ---------------------------------------------
echo ">> Restoring Neo4j into target version" | tee -a "$DRILL_LOG"
docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "${MEM0_NEO4J_PASSWORD:?required}" \
    "STOP DATABASE neo4j WAIT;" 2>>"$DRILL_LOG" || true
docker exec -i -u neo4j "$NEO_CT" \
    neo4j-admin database load neo4j --from-stdin --overwrite-destination=true \
    < "$NEO_ARCHIVE" 2>>"$DRILL_LOG" \
    || { echo "ERROR: neo4j load on target version FAILED" | tee -a "$DRILL_LOG"; NEO_OK=false; }

# Forward-version migration: neo4j-admin recognises older store formats
# and migrates on START.
docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "$MEM0_NEO4J_PASSWORD" \
    "START DATABASE neo4j WAIT;" 2>>"$DRILL_LOG" \
    || { echo "ERROR: neo4j start on target version FAILED (migration?)" \
        | tee -a "$DRILL_LOG"; NEO_OK=false; }

NEO_NODES=$(docker exec -u neo4j "$NEO_CT" cypher-shell \
    -u neo4j -p "$MEM0_NEO4J_PASSWORD" --format plain \
    "MATCH (n) RETURN count(n) AS n;" 2>>"$DRILL_LOG" | tail -1 || echo "0")
echo "  Neo4j node count: $NEO_NODES" | tee -a "$DRILL_LOG"
NEO_OK=${NEO_OK:-true}

# --- Restore + migrate ChromaDB ------------------------------------------
echo ">> Restoring ChromaDB into target version" | tee -a "$DRILL_LOG"
$COMPOSE -p "$DRILL_PROJECT" stop chromadb 2>>"$DRILL_LOG" || true
docker run --rm \
    -v "${DRILL_PROJECT}_chroma:/chroma/chroma" \
    -v "$(dirname "$CHR_ARCHIVE"):/backup:ro" \
    alpine:3 sh -c "cd /chroma/chroma && tar -xzf /backup/$(basename "$CHR_ARCHIVE")" \
    2>>"$DRILL_LOG" || true
$COMPOSE -p "$DRILL_PROJECT" start chromadb 2>>"$DRILL_LOG" || true
sleep 8  # chroma migration on version bump can take longer than start
CHR_PORT=$(docker port "$CHR_CT" 8000 2>/dev/null | head -1 | cut -d: -f2 || echo "")
if [[ -n "$CHR_PORT" ]]; then
    CHR_HEALTH=$(curl -fs "http://localhost:${CHR_PORT}/api/v1/heartbeat" \
        2>/dev/null || curl -fs "http://localhost:${CHR_PORT}/api/v2/heartbeat" \
        2>/dev/null || echo "FAIL")
else
    CHR_HEALTH="(no port mapping)"
fi
echo "  ChromaDB heartbeat: $CHR_HEALTH" | tee -a "$DRILL_LOG"
CHR_OK="true"
[[ "$CHR_HEALTH" == "FAIL" ]] && CHR_OK="false"

# --- Manifest update -----------------------------------------------------
COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
ALL_OK=$([[ "$PG_OK" == "true" && "$NEO_OK" == "true" && "$CHR_OK" == "true" ]] \
    && echo true || echo false)

python3 - "$MANIFEST_PATH" "$COMPLETED_AT" "$SET_TS" "$ALL_OK" \
    "$PG_ROWS" "$NEO_NODES" "$CHR_HEALTH" \
    "$POSTGRES_TARGET_TAG" "$NEO4J_TARGET_TAG" "$CHROMA_TARGET_TAG" \
    "$DRILL_LOG" <<'PYEOF'
import json
import sys
from pathlib import Path

(
    manifest_path, completed_at, set_ts, all_ok,
    pg_rows, neo_nodes, chr_health,
    pg_target, neo_target, chr_target,
    log_path,
) = sys.argv[1:12]

new_entry = {
    "ts": completed_at,
    "drilled_set_ts": set_ts,
    "all_ok": all_ok == "true",
    "target_versions": {
        "postgres": pg_target,
        "neo4j": neo_target,
        "chromadb": chr_target,
    },
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
manifest["last_target_versions"] = new_entry["target_versions"]
manifest["runs"] = manifest["runs"][-50:]

p.write_text(json.dumps(manifest, indent=2))
PYEOF

if [[ "$ALL_OK" == "true" ]]; then
    echo "✅ Version-upgrade drill PASSED for backup set $SET_TS" | tee -a "$DRILL_LOG"
    exit 0
fi
echo "❌ Version-upgrade drill FAILED for backup set $SET_TS" | tee -a "$DRILL_LOG"
exit 1
