#!/usr/bin/env bash
# Warm-spare rsync wrapper (Q17.1).
#
# Reads enabled + partner target from workspace/warm_spare/activation.json
# (preferred — operator-owned, gateway never overwrites). Falls back to
# runtime_settings.json keys for the React /cp/settings flow.
#
# Exit codes:
#   0   sync completed (or warm_spare disabled — no-op)
#   1   missing dependency
#   2   partner target unset
#   3   rsync failed
#
# Idempotent and crash-safe — --partial resumes interrupted transfers,
# --update only overwrites destination files when source is newer.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/Users/andrus/BotArmy/crewai-team/workspace}"
ACTIVATION_FILE="$WORKSPACE_ROOT/warm_spare/activation.json"
SETTINGS_FILE="$WORKSPACE_ROOT/runtime_settings.json"
LOG_DIR="$WORKSPACE_ROOT/warm_spare"
mkdir -p "$LOG_DIR"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

if ! command -v rsync >/dev/null 2>&1; then
  log "ERROR: rsync not in PATH"
  exit 1
fi

# Reader prefers activation.json (operator-owned); falls back to
# runtime_settings.json.
read_setting() {
  python3 -c "
import json, sys
try:
    s = json.load(open('$ACTIVATION_FILE'))
    v = s.get('$1')
    if v is not None:
        print(v if not isinstance(v, bool) else ('True' if v else 'False'))
        sys.exit(0)
except (FileNotFoundError, json.JSONDecodeError):
    pass
try:
    s = json.load(open('$SETTINGS_FILE'))
    v = s.get('$2', '')
    print(v if not isinstance(v, bool) else ('True' if v else 'False'))
except Exception:
    sys.exit(1)
"
}

ENABLED="$(read_setting enabled warm_spare_enabled || echo False)"
TARGET="$(read_setting partner_target warm_spare_partner_target || echo '')"

if [[ "$ENABLED" != "True" ]]; then
  log "warm_spare disabled (enabled=$ENABLED) — skipping"
  exit 0
fi

if [[ -z "$TARGET" ]]; then
  log "ERROR: partner target unset"
  exit 2
fi

log "syncing $WORKSPACE_ROOT/ → $TARGET"

EXCLUDES=(
  --exclude=__pycache__/
  --exclude=.cache/
  --exclude=tmp/
  --exclude='*.pyc'
  --exclude='*.tmp'
  --exclude=.env
  --exclude=.env.local
  --exclude=secrets/
  --exclude='*.crt'
  --exclude='*.key'
  --exclude='*.pem'
  --exclude=.DS_Store
  --exclude=.Spotlight-V100
  --exclude=.fseventsd
  --exclude=.Trashes
  --exclude=.TemporaryItems
  --exclude=warm_spare/.sync.log
)

set +e
rsync -av --update --partial --human-readable \
  "${EXCLUDES[@]}" \
  "$WORKSPACE_ROOT/" "$TARGET"
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
  log "ERROR: rsync exited with $RC"
  exit 3
fi

# Write a fresh canonical heartbeat so the partner sees we're alive.
HEARTBEAT="$LOG_DIR/canonical_heartbeat.json"
TS="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
HOSTNAME="$(hostname -s 2>/dev/null || hostname)"
cat > "$HEARTBEAT.tmp" <<EOF
{"ts": "$TS", "hostname": "$HOSTNAME", "pid": $$}
EOF
mv "$HEARTBEAT.tmp" "$HEARTBEAT"
log "heartbeat updated"

rsync -av --update --partial \
  "$HEARTBEAT" "$TARGET/warm_spare/" 2>/dev/null || true

log "done"
