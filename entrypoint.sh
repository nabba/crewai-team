#!/bin/sh
set -e

# Ensure workspace directories are writable by appuser
chown -R appuser:appuser /app/workspace 2>/dev/null || true

# Ensure the signal-cli socket is accessible by appuser
SOCK="${SIGNAL_SOCKET_PATH:-/tmp/signal-cli.sock}"
if [ -S "$SOCK" ]; then
    chown appuser:appuser "$SOCK" 2>/dev/null || true
    chmod 660 "$SOCK" 2>/dev/null || true
fi

# Drop privileges and exec the main process as appuser
exec gosu appuser "$@"
