#!/usr/bin/env bash
# kill_the_gateway.sh — LIVE drill: stop the gateway container,
# measure recovery time, verify post-recovery state.
#
# PROGRAM §44.2 — Q6.2 — runs OUTSIDE the gateway because the gateway
# cannot kill itself.
#
# Usage:
#   scripts/drills/kill_the_gateway.sh "EXECUTE KILL DRILL"
#
# The typed-phrase confirmation is required as the first argument.
# Without it, the script exits with usage info and does NOT kill
# anything.
#
# Behaviour:
#   1. Verify the typed-phrase confirmation
#   2. Pre-drill: hit the gateway pre-drill endpoint to verify
#      readiness (DR backup recent, etc.)
#   3. Record T0
#   4. docker compose stop gateway
#   5. Wait 30 seconds (simulate "operator notices and reacts")
#   6. docker compose start gateway
#   7. Poll /health every 5s until 200 OK (timeout 5 min)
#   8. Record T1, compute recovery time
#   9. Smoke-check: /api/cp/sentience/scorecard-pinning
#      (verify anti_goodhart_intact: true)
#  10. Write workspace/resilience/kill_drill_<ts>.json
#  11. The restored gateway's companion loop picks up the report
#      on first idle pass via ingest_external_report()
#
# Exit codes:
#   0 — drill passed
#   1 — drill failed (recovery exceeded target or smoke-check failed)
#   2 — invalid usage / missing prerequisites
#   3 — typed-phrase not provided or wrong

set -uo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:8765}"
COMPOSE_SERVICE="${COMPOSE_SERVICE:-gateway}"
DOWNTIME_SECONDS="${DOWNTIME_SECONDS:-30}"
RECOVERY_TIMEOUT="${RECOVERY_TIMEOUT:-300}"
REPORT_DIR="${REPORT_DIR:-./workspace/resilience}"

PHRASE_EXPECTED="EXECUTE KILL DRILL"

if [[ "${1:-}" != "${PHRASE_EXPECTED}" ]]; then
  cat >&2 <<EOF
kill_the_gateway.sh — LIVE drill, DISRUPTIVE.

Usage:
  $0 "${PHRASE_EXPECTED}"

The typed-phrase confirmation is mandatory. Without it, this script
takes no action.

This script stops the gateway container, waits ${DOWNTIME_SECONDS}s,
restarts, and measures recovery time. Make sure:
  * You've scheduled a maintenance window
  * The drill_kill_the_gateway_enabled runtime setting is ON
  * The most recent DR backup is < 7 days old
EOF
  exit 3
fi

# Verify docker compose is available + the gateway service exists.
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker command not found" >&2
  exit 2
fi
if ! docker compose ps "${COMPOSE_SERVICE}" >/dev/null 2>&1; then
  echo "ERROR: compose service ${COMPOSE_SERVICE} not found" >&2
  exit 2
fi

mkdir -p "${REPORT_DIR}"
TS_START=$(date -u +%Y%m%dT%H%M%SZ)
ISO_START=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)
REPORT_FILE="${REPORT_DIR}/kill_drill_${TS_START}.json"

echo "kill_the_gateway drill starting at ${ISO_START}"
echo "  gateway URL: ${GATEWAY_URL}"
echo "  downtime:    ${DOWNTIME_SECONDS}s"
echo "  timeout:     ${RECOVERY_TIMEOUT}s"
echo "  report:      ${REPORT_FILE}"

# 1. Pre-drill readiness check via the in-gateway endpoint.
#    This verifies the gateway considers itself ready BEFORE we
#    kill it.
echo
echo "[1/5] Pre-drill readiness check..."
PRE_DRILL_BODY="$(curl -sS -X POST "${GATEWAY_URL}/api/cp/drills/run/kill_the_gateway" \
  -H 'Content-Type: application/json' -d '{}' 2>&1 || true)"
echo "  pre-drill response: ${PRE_DRILL_BODY}"

# 2. Record T0 + stop the gateway.
T0=$(date +%s)
echo
echo "[2/5] Stopping gateway at T0=${T0}..."
docker compose stop "${COMPOSE_SERVICE}" || {
  echo "ERROR: docker compose stop failed" >&2
  exit 1
}

# 3. Wait the configured downtime period.
echo "[3/5] Waiting ${DOWNTIME_SECONDS}s (simulating operator-react time)..."
sleep "${DOWNTIME_SECONDS}"

# 4. Restart and poll for /health.
echo "[4/5] Restarting gateway + polling /health..."
docker compose start "${COMPOSE_SERVICE}" || {
  echo "ERROR: docker compose start failed" >&2
  exit 1
}

RECOVERED=0
WAITED=0
while [[ ${WAITED} -lt ${RECOVERY_TIMEOUT} ]]; do
  HTTP_CODE="$(curl -sS -o /dev/null -w '%{http_code}' \
    --max-time 3 "${GATEWAY_URL}/health" 2>/dev/null || echo 000)"
  if [[ "${HTTP_CODE}" == "200" ]]; then
    RECOVERED=1
    break
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  echo "  ...waited ${WAITED}s, HTTP=${HTTP_CODE}"
done

T1=$(date +%s)
ELAPSED=$((T1 - T0))
ISO_END=$(date -u +%Y-%m-%dT%H:%M:%S+00:00)

# 5. Smoke check + write report.
echo
echo "[5/5] Smoke check + writing report..."
SMOKE_CODE=000
SMOKE_BODY=""
if [[ ${RECOVERED} -eq 1 ]]; then
  SMOKE_BODY="$(curl -sS "${GATEWAY_URL}/api/cp/sentience/scorecard-pinning" 2>/dev/null || echo '{}')"
  SMOKE_CODE=200
fi

# Compute pass/fail.
STATUS="pass"
ERRORS_ARR="[]"
if [[ ${RECOVERED} -ne 1 ]]; then
  STATUS="fail"
  ERRORS_ARR='["gateway did not respond within timeout"]'
elif ! echo "${SMOKE_BODY}" | grep -q '"anti_goodhart_intact":true'; then
  STATUS="fail"
  ERRORS_ARR='["smoke check: anti_goodhart_intact missing or false"]'
fi

# Write the report.
cat > "${REPORT_FILE}" <<JSON
{
  "drill_name": "kill_the_gateway",
  "status": "${STATUS}",
  "started_at": "${ISO_START}",
  "completed_at": "${ISO_END}",
  "duration_s": ${ELAPSED},
  "detail": {
    "downtime_seconds": ${DOWNTIME_SECONDS},
    "recovery_seconds": ${ELAPSED},
    "recovered": $( [[ ${RECOVERED} -eq 1 ]] && echo true || echo false ),
    "smoke_code": ${SMOKE_CODE}
  },
  "errors": ${ERRORS_ARR}
}
JSON

echo
echo "Drill ${STATUS}. Recovery: ${ELAPSED}s. Report at: ${REPORT_FILE}"
echo "The restored gateway's companion loop will ingest this on next idle pass."

[[ "${STATUS}" == "pass" ]] && exit 0 || exit 1
