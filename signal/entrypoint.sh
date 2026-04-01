#!/bin/bash
set -e

SIGNAL_NUMBER="${SIGNAL_NUMBER:-}"
SIGNAL_CONFIG="/home/.local/share/signal-cli"

if [ -z "$SIGNAL_NUMBER" ]; then
    echo "[entrypoint] ERROR: SIGNAL_NUMBER not set"
    exit 1
fi

echo "[entrypoint] Starting signal-cli daemon for $SIGNAL_NUMBER..."

# Start signal-cli daemon in background with HTTP API on port 7583
signal-cli --config "$SIGNAL_CONFIG" \
    -a "$SIGNAL_NUMBER" \
    daemon \
    --http 127.0.0.1:7583 \
    --receive-mode manual \
    --no-receive-stdout \
    &

SIGNAL_PID=$!
echo "[entrypoint] signal-cli daemon PID: $SIGNAL_PID"

# Wait for signal-cli HTTP API to be ready
echo "[entrypoint] Waiting for signal-cli HTTP API..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:7583/api/v1/rpc -d '{"jsonrpc":"2.0","id":1,"method":"version"}' -H "Content-Type: application/json" > /dev/null 2>&1; then
        echo "[entrypoint] signal-cli HTTP API is ready"
        break
    fi
    sleep 1
done

# Start the forwarder (blocks)
echo "[entrypoint] Starting forwarder..."
exec python -u /app/forwarder.py
