"""
Forwards inbound Signal messages from signal-cli JSON-RPC to the FastAPI gateway.

Uses HTTP polling against signal-cli's JSON-RPC `receive` endpoint.
signal-cli must be running in daemon mode with --receive-mode manual:

    signal-cli -a +NUMBER daemon --socket /tmp/signal-cli.sock \
        --http 127.0.0.1:7583 --receive-mode manual --no-receive-stdout

Environment variables:
    GATEWAY_SECRET      — shared secret for authenticating with the gateway
    SIGNAL_CLI_HTTP_URL — signal-cli HTTP endpoint (default: http://127.0.0.1:7583)
    GATEWAY_URL         — gateway inbound endpoint (default: http://gateway:8765/signal/inbound)
    POLL_INTERVAL       — seconds between polls when idle (default: 1)
"""
import json
import os
import time
import requests

SIGNAL_CLI_URL = os.environ.get("SIGNAL_CLI_HTTP_URL", "http://host.docker.internal:7583")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8765/signal/inbound")
GATEWAY_SECRET = os.environ.get("GATEWAY_SECRET", "")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1"))

# Reusable HTTP sessions for connection pooling
_signal_session = requests.Session()
_signal_session.headers["Content-Type"] = "application/json"

_gateway_session = requests.Session()


def log(msg):
    print(f"[forwarder] {msg}", flush=True)


def _receive_messages() -> list:
    """Poll signal-cli HTTP `receive` endpoint for new messages."""
    try:
        resp = _signal_session.post(
            SIGNAL_CLI_URL.rstrip("/") + "/api/v1/rpc",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "receive",
                "params": {"timeout": 5},
            },
            timeout=15,
        )
        data = resp.json()
        if "error" in data:
            err = data["error"].get("message", "")
            # "already being received" means signal-cli is in on-start mode — not fatal
            if "already being received" not in err:
                log(f"receive error: {err}")
            return []
        return data.get("result", [])
    except requests.exceptions.ConnectionError:
        return []  # signal-cli not running yet
    except Exception as e:
        log(f"receive failed: {e}")
        return []


def _process_envelope(envelope: dict) -> None:
    """Extract message data from an envelope and forward to the gateway."""
    data_msg = envelope.get("dataMessage")
    if not data_msg:
        return
    if not data_msg.get("message") and not data_msg.get("attachments"):
        return

    sender = envelope.get("source") or envelope.get("sourceNumber")
    if not sender:
        return

    message = data_msg.get("message", "")
    timestamp = data_msg.get("timestamp") or envelope.get("timestamp", 0)

    # Extract attachment metadata
    attachments = []
    for att in data_msg.get("attachments", []):
        attachments.append({
            "contentType": att.get("contentType", ""),
            "filename": att.get("filename", ""),
            "id": att.get("id", ""),
            "size": att.get("size", 0),
        })

    att_info = f", {len(attachments)} attachment(s)" if attachments else ""
    log(f"Incoming message from {sender[-4:]} ({len(message)} chars{att_info})")

    # Forward to gateway
    payload = {
        "sender": sender,
        "message": message,
        "timestamp": timestamp,
        "attachments": attachments,
    }
    headers = {}
    if GATEWAY_SECRET:
        headers["Authorization"] = f"Bearer {GATEWAY_SECRET}"

    try:
        resp = _gateway_session.post(
            GATEWAY_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        log(f"Forwarded to gateway: {resp.status_code}")
    except Exception as e:
        log(f"Failed to forward: {e}")


def poll_loop():
    """Continuously poll signal-cli for new messages and forward them."""
    log(f"Polling signal-cli at {SIGNAL_CLI_URL} every {POLL_INTERVAL}s")
    log(f"Forwarding to {GATEWAY_URL}")

    consecutive_errors = 0

    while True:
        messages = _receive_messages()

        if messages:
            consecutive_errors = 0
            for msg in messages:
                envelope = msg.get("envelope", msg)
                try:
                    _process_envelope(envelope)
                except Exception as e:
                    log(f"Error processing envelope: {e}")
        elif messages is not None:
            consecutive_errors = 0

        # Adaptive backoff: if signal-cli is down, slow down polling
        if consecutive_errors > 10:
            time.sleep(min(consecutive_errors, 30))
        else:
            time.sleep(POLL_INTERVAL)


def main():
    if not GATEWAY_SECRET:
        log("WARNING: GATEWAY_SECRET not set — requests will be rejected by gateway")

    # Wait for signal-cli to be ready
    log("Waiting for signal-cli...")
    while True:
        try:
            resp = _signal_session.post(
                SIGNAL_CLI_URL.rstrip("/") + "/api/v1/rpc",
                json={"jsonrpc": "2.0", "id": 1, "method": "version"},
                timeout=5,
            )
            version = resp.json().get("result", {}).get("version", "?")
            log(f"signal-cli v{version} is ready")
            break
        except Exception:
            time.sleep(3)

    poll_loop()


if __name__ == "__main__":
    main()
