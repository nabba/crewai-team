#!/bin/sh
set -e

# Ensure workspace directories exist and are writable by appuser
mkdir -p /app/workspace/output /app/workspace/skills /app/workspace/proposals /app/workspace/applied_code
chown -R appuser:appuser /app/workspace 2>/dev/null || true

# Ensure the signal-cli socket is accessible by appuser (host-mode only —
# in k8s there's no socket, this loop is a no-op).
SOCK="${SIGNAL_SOCKET_PATH:-/tmp/signal-cli.sock}"
if [ -S "$SOCK" ]; then
    chown appuser:appuser "$SOCK" 2>/dev/null || true
    chmod 660 "$SOCK" 2>/dev/null || true
fi

# Drop privileges if we're root, exec directly otherwise.
#
# Local docker-compose: container starts as root, gosu drops to appuser.
# Kubernetes: chart sets runAsUser=1000 (Autopilot also enforces non-root),
# so we already ARE appuser — gosu would error with "operation not
# permitted" because it needs root to switch users. Just exec.
if [ "$(id -u)" = "0" ]; then
    exec gosu appuser "$@"
else
    exec "$@"
fi
