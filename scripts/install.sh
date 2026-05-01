#!/usr/bin/env bash
# Compatibility shim — the installer moved to ../install.sh in April 2026.
# This file forwards any invocation to the new location so existing docs,
# CI jobs, and muscle memory keep working.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEW="${DIR}/install.sh"

if [[ ! -x "$NEW" ]]; then
    echo "scripts/install.sh: the installer has moved to $NEW but it's missing or not executable." >&2
    exit 1
fi

echo "[i] scripts/install.sh is now a shim — forwarding to ./install.sh" >&2
exec "$NEW" "$@"
