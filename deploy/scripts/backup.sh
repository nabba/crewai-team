#!/usr/bin/env bash
# AndrusAI host-side DB backup script.
# Wave 0/1 closure (#A1, 2026-05-09).
#
# Backs up Postgres + Neo4j + ChromaDB to ./workspace/backups/.
# Run from the repo root: `bash deploy/scripts/backup.sh`.
#
# This is the operator-runnable equivalent of app/healing/db_backup.py.
# Use this when the gateway is down or you'd rather not let the gateway
# do the backups itself. In production K8s, prefer a CronJob; this is
# the laptop-dev path.
#
# Exits non-zero if any of the three steps fails. Manifest entry is
# still written so the freshness monitor sees the partial run.

set -euo pipefail

# Resolve repo root (script is at deploy/scripts/backup.sh).
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

# --- Container resolution ------------------------------------------------
PG_CONTAINER="$($COMPOSE ps -q postgres 2>/dev/null || true)"
NEO_CONTAINER="$($COMPOSE ps -q neo4j 2>/dev/null || true)"

# --- Output paths --------------------------------------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_ROOT="${REPO_ROOT}/workspace/backups"
mkdir -p "${BACKUP_ROOT}/postgres" "${BACKUP_ROOT}/neo4j" "${BACKUP_ROOT}/chromadb"

PG_OUT="${BACKUP_ROOT}/postgres/postgres-${TS}.sql.gz"
NEO_OUT="${BACKUP_ROOT}/neo4j/neo4j-${TS}.dump"
CHR_OUT="${BACKUP_ROOT}/chromadb/chromadb-${TS}.tar.gz"
MANIFEST="${BACKUP_ROOT}/manifest.json"

PG_OK=0
NEO_OK=0
CHR_OK=0
PG_ERR=""
NEO_ERR=""
CHR_ERR=""
PG_BYTES=0
NEO_BYTES=0
CHR_BYTES=0

# --- Postgres ------------------------------------------------------------
echo ">> Postgres"
if [[ -z "$PG_CONTAINER" ]]; then
    PG_ERR="postgres container not running"
    echo "  SKIP: $PG_ERR" >&2
else
    PG_DB="${MEM0_PG_DB:-mem0}"
    PG_USER="${MEM0_PG_USER:-mem0}"
    PG_PASS="${MEM0_PG_PASSWORD:-${POSTGRES_PASSWORD:-}}"
    if docker exec -e PGPASSWORD="$PG_PASS" "$PG_CONTAINER" \
        pg_dump --username "$PG_USER" --dbname "$PG_DB" \
        --clean --if-exists --no-owner --no-privileges \
        2>/tmp/pg_dump_err | gzip > "$PG_OUT"; then
        PG_BYTES=$(stat -c%s "$PG_OUT" 2>/dev/null || stat -f%z "$PG_OUT")
        PG_OK=1
        echo "  OK ($PG_BYTES bytes -> $PG_OUT)"
    else
        PG_ERR="$(head -c 500 /tmp/pg_dump_err 2>/dev/null || echo 'pg_dump failed')"
        rm -f "$PG_OUT"
        echo "  FAIL: $PG_ERR" >&2
    fi
fi

# --- Neo4j ---------------------------------------------------------------
echo ">> Neo4j"
if [[ -z "$NEO_CONTAINER" ]]; then
    NEO_ERR="neo4j container not running"
    echo "  SKIP: $NEO_ERR" >&2
else
    if docker exec -u neo4j "$NEO_CONTAINER" \
        neo4j-admin database dump neo4j --to-stdout \
        > "$NEO_OUT" 2>/tmp/neo4j_err; then
        NEO_BYTES=$(stat -c%s "$NEO_OUT" 2>/dev/null || stat -f%z "$NEO_OUT")
        NEO_OK=1
        echo "  OK ($NEO_BYTES bytes -> $NEO_OUT)"
    else
        NEO_ERR="$(head -c 500 /tmp/neo4j_err 2>/dev/null || echo 'neo4j-admin failed')"
        rm -f "$NEO_OUT"
        echo "  FAIL: $NEO_ERR" >&2
    fi
fi

# --- ChromaDB ------------------------------------------------------------
echo ">> ChromaDB"
CHR_SRC="${REPO_ROOT}/workspace/memory"
if [[ ! -d "$CHR_SRC" ]]; then
    CHR_ERR="chroma source dir not found: $CHR_SRC"
    echo "  SKIP: $CHR_ERR" >&2
else
    if tar --exclude='*.lock' --exclude='*-journal' \
        -czf "$CHR_OUT" -C "${REPO_ROOT}/workspace" memory \
        2>/tmp/chr_err; then
        CHR_BYTES=$(stat -c%s "$CHR_OUT" 2>/dev/null || stat -f%z "$CHR_OUT")
        CHR_OK=1
        echo "  OK ($CHR_BYTES bytes -> $CHR_OUT)"
    else
        CHR_ERR="$(head -c 500 /tmp/chr_err 2>/dev/null || echo 'tar failed')"
        rm -f "$CHR_OUT"
        echo "  FAIL: $CHR_ERR" >&2
    fi
fi

# --- Manifest update -----------------------------------------------------
ALL_OK=$([[ $PG_OK -eq 1 && $NEO_OK -eq 1 && $CHR_OK -eq 1 ]] && echo true || echo false)
COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"

# Build entry JSON. python3 if present (preserves manifest history),
# else jq, else fall back to plain overwrite (loses history).
NEW_ENTRY=$(cat <<EOF
{
  "started_at": "${COMPLETED_AT}",
  "completed_at": "${COMPLETED_AT}",
  "all_ok": ${ALL_OK},
  "source": "operator_script",
  "postgres": {"ok": $([[ $PG_OK -eq 1 ]] && echo true || echo false), "path": "backups/postgres/postgres-${TS}.sql.gz", "bytes": ${PG_BYTES}, "error": $([[ -z "$PG_ERR" ]] && echo null || python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$PG_ERR" 2>/dev/null || echo \"\"$PG_ERR\"\" )},
  "neo4j":    {"ok": $([[ $NEO_OK -eq 1 ]] && echo true || echo false), "path": "backups/neo4j/neo4j-${TS}.dump", "bytes": ${NEO_BYTES}, "error": $([[ -z "$NEO_ERR" ]] && echo null || python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$NEO_ERR" 2>/dev/null || echo \"\"$NEO_ERR\"\" )},
  "chromadb": {"ok": $([[ $CHR_OK -eq 1 ]] && echo true || echo false), "path": "backups/chromadb/chromadb-${TS}.tar.gz", "bytes": ${CHR_BYTES}, "error": $([[ -z "$CHR_ERR" ]] && echo null || python3 -c "import json,sys;print(json.dumps(sys.argv[1]))" "$CHR_ERR" 2>/dev/null || echo \"\"$CHR_ERR\"\" )}
}
EOF
)

if command -v python3 >/dev/null 2>&1; then
    python3 - "$MANIFEST" "$NEW_ENTRY" "$COMPLETED_AT" <<'PYEOF'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
new_entry_str = sys.argv[2]
completed_at = sys.argv[3]

try:
    new_entry = json.loads(new_entry_str)
except Exception as exc:
    print(f"ERROR: bad new-entry JSON: {exc}", file=sys.stderr)
    sys.exit(1)

if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {"runs": []}
else:
    manifest = {"runs": []}

runs = manifest.setdefault("runs", [])
runs.append(new_entry)
if len(runs) > 200:
    manifest["runs"] = runs[-200:]
manifest["last_updated_at"] = completed_at

tmp = manifest_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
tmp.replace(manifest_path)
PYEOF
else
    echo "WARN: python3 not found; manifest history not preserved" >&2
    echo "{\"runs\": [$NEW_ENTRY], \"last_updated_at\": \"$COMPLETED_AT\"}" > "$MANIFEST"
fi

# --- Retention purge -----------------------------------------------------
RETENTION_DAYS="${DB_BACKUP_RETENTION_DAYS:-30}"
echo ">> Purging archives older than ${RETENTION_DAYS} days"
PURGED=0
for sub in postgres neo4j chromadb; do
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        rm -f "$f" && PURGED=$((PURGED + 1))
    done < <(find "${BACKUP_ROOT}/${sub}" -type f -mtime +${RETENTION_DAYS} 2>/dev/null)
done
echo "  Purged: $PURGED file(s)"

# --- Summary -------------------------------------------------------------
echo
echo "Summary:"
echo "  Postgres:  $([[ $PG_OK -eq 1 ]] && echo OK || echo FAIL)"
echo "  Neo4j:     $([[ $NEO_OK -eq 1 ]] && echo OK || echo FAIL)"
echo "  ChromaDB:  $([[ $CHR_OK -eq 1 ]] && echo OK || echo FAIL)"
echo "  Manifest:  $MANIFEST"

if [[ "$ALL_OK" == "true" ]]; then
    exit 0
else
    exit 1
fi
