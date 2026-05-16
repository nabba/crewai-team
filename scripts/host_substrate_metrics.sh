#!/usr/bin/env bash
# Host-side substrate metrics collector — companion to
# app/healing/monitors/host_substrate_health.py.
#
# PROGRAM §51 Q16 Theme 1 follow-on (Q16.1 Item 6). The in-container
# monitor sees workspace volume + restart cadence + memory (on Linux).
# What it CANNOT see from inside Docker: SMART data, macOS version,
# host filesystem health beyond the mount, Docker Desktop VM sizing.
#
# This script runs HOST-SIDE under a launchd LaunchAgent (weekly) and
# writes one JSONL row into the bind-mounted workspace at
# workspace/healing/host_metrics.jsonl. The monitor surfaces the
# latest row in its summary without imposing a strict schema.
#
# Dependencies: `jq` (brew install jq). `smartctl` (brew install
# smartmontools) — optional; on macOS, sudo is needed and SMART may
# not be reported on internal Apple SSDs. The script tolerates
# absence and emits null for the fields it can't fill.
#
# Cap: keeps the last 200 rows.

set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HOME/BotArmy/crewai-team/workspace}"
OUT="$WORKSPACE_ROOT/healing/host_metrics.jsonl"

mkdir -p "$(dirname "$OUT")"

# macOS version (always available).
MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo unknown)"

# SMART (optional; fall back to null on any failure).
SMART_REALLOCATED="null"
SMART_TEMP="null"
if command -v smartctl >/dev/null 2>&1; then
  # Apple Silicon: /dev/disk0 is the internal SSD. On Intel Macs that
  # rejected SMART for the boot drive, this returns no rows.
  SMART_JSON="$(sudo -n smartctl -j -A /dev/disk0 2>/dev/null || echo '{}')"
  if [ -n "$SMART_JSON" ] && [ "$SMART_JSON" != "{}" ]; then
    SMART_REALLOCATED="$(echo "$SMART_JSON" | jq -r '.ata_smart_attributes.table[]? | select(.name=="Reallocated_Sector_Ct") | .raw.value // null' 2>/dev/null | head -1 || echo null)"
    SMART_TEMP="$(echo "$SMART_JSON" | jq -r '.temperature.current // null' 2>/dev/null || echo null)"
    [ -z "$SMART_REALLOCATED" ] && SMART_REALLOCATED="null"
    [ -z "$SMART_TEMP" ] && SMART_TEMP="null"
  fi
fi

# Boot drive total size (GB) — different from in-container disk_usage,
# which sees only the bind-mounted volume.
DISK_TOTAL_GB="null"
if command -v diskutil >/dev/null 2>&1; then
  DISK_TOTAL_GB="$(diskutil info / 2>/dev/null | awk '/Volume Total Space/ {print int($5 / 1024 / 1024 / 1024); exit}' || echo null)"
  [ -z "$DISK_TOTAL_GB" ] && DISK_TOTAL_GB="null"
fi

# Docker Desktop VM allocation (optional).
DOCKER_VM_DISK_GB="null"
if command -v docker >/dev/null 2>&1; then
  # `docker system df` shows the host-allocated VM size when on
  # Docker Desktop. We pull the Volumes total.
  DOCKER_VM_DISK_GB="$(docker system df --format '{{json .}}' 2>/dev/null | jq -r 'select(.Type=="Volumes") | .Size' 2>/dev/null | head -1 || echo null)"
  [ -z "$DOCKER_VM_DISK_GB" ] && DOCKER_VM_DISK_GB="null"
fi

# Uptime in seconds (since the host booted, not the container).
HOST_UPTIME_S="$(sysctl -n kern.boottime 2>/dev/null | awk -F'[= ,]+' '{print systime() - $5}' || echo 0)"

TS="$(date -u +%s)"

# Emit one row. Use printf so we control quoting precisely.
printf '{"ts": %s, "ts_iso": "%s", "macos_version": "%s", "smart_reallocated_sectors": %s, "smart_temperature_c": %s, "host_disk_total_gb": %s, "docker_vm_disk_gb": %s, "host_uptime_s": %s}\n' \
  "$TS" \
  "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" \
  "$MACOS_VERSION" \
  "$SMART_REALLOCATED" \
  "$SMART_TEMP" \
  "$DISK_TOTAL_GB" \
  "$DOCKER_VM_DISK_GB" \
  "$HOST_UPTIME_S" \
  >> "$OUT"

# Cap at 200 rows.
if [ "$(wc -l < "$OUT" 2>/dev/null || echo 0)" -gt 200 ]; then
  tail -n 200 "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
fi

echo "✓ Wrote host metrics to $OUT"
