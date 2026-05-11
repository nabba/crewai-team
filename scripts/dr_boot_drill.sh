#!/usr/bin/env bash
# DR boot drill — verify a fresh checkout could rebuild from the latest
# portable export. PROGRAM §40 (2026-05-10) — Q3 Item 13.
#
# Usage:
#   scripts/dr_boot_drill.sh                 # use latest tarball
#   scripts/dr_boot_drill.sh --export-fresh  # export then drill
#   scripts/dr_boot_drill.sh --keep-target   # leave the sandbox dir on disk
#
# Exit codes:
#   0  drill OK
#   1  drill failed (check workspace/dr/drill_*.json)
#   2  prerequisites missing
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "dr_boot_drill: python3 not found on PATH" >&2
  exit 2
fi

# Python: prefer the project venv if it exists.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

exec "$PY" -m app.dr.boot_drill "$@"
