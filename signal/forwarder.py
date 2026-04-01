"""
Forwards inbound Signal messages from signal-cli REST API to the FastAPI gateway.

Uses HTTP polling against signal-cli's /v1/receive/{number} REST endpoint.
signal-cli must be running in 'normal' mode (REST API).

Environment variables:
    GATEWAY_SECRET        — shared secret for authenticating with the gateway
    SIGNAL_CLI_HTTP_URL   — signal-cli HTTP base URL (default: http://signal-cli:8080)
    SIGNAL_NUMBER         — Signal account number to receive from (e.g. +46731727774)
    GATEWAY_URL           — gateway inbound endpoint (default: http://gateway:8765/signal/inbound)
    POLL_INTERVAL         — seconds between polls when idle (default: 1)
"""
import json
import os
import time
import requests

SIGNAL_CLI_URL = os.environ.get("SIGNAL_CLI_HTTP_URL", "http://signal-cli:8080")
SIGNAL_NUMBER = os.environ.get("SIGNAL_NUMBER", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://gateway:8765/signal/inbound")
GATEWAY_SECRET = os.environ.get("GATEWAY_SECRET", "")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1"))

# Reusable HTTP sessions for connection pooling
_signal_session = requests.Session()
_gateway_session = requests.Session()


def log(msg):
    print(f"[forwarder] {msg}", flush=True)


def _get_number() -> str:
    """Resolve the Signal number to poll — from env or auto-detected via /v1/accounts."""
    if SIGNAL_NUMBER:
        return SIGNAL_NUMBER
    try:
        resp = _signal_session.get(
            SIGNAL_CLI_URL.rstrip("/") + "/v1/accounts",
            timeout=5,
        )
        accounts = resp.json()
        if accounts:
            number = accounts[0].get("number", "")
            log(f"Auto-detected Signal number: {number}")
            return number
    except Exception as e:
        log(f"Could not auto-detect number: {e}")
    return ""


def _receive_messages(number: str) -> list:
    """Poll signal-cli REST API for new messages."""
    try:
        resp = _signal_session.get(
            SIGNAL_CLI_URL.rstrip("/") + f"/v1/receive/{number}",
            params={"timeout": int(POLL_INTERVAL) + 1},
            timeout=POLL_INTERVAL + 5,
        )
        if resp.status_code == 200:
            return resp.json() or []
        if resp.status_code == 400:
            # No messages or transient error
            return []
        log(f"receive HTTP {resp.status_code}: {resp.text[:200]}")
        return []
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


def poll_loop(number: str):
    """Continuously poll signal-cli for new messages and forward them."""
    log(f"Polling signal-cli at {SIGNAL_CLI_URL}/v1/receive/{number} every {POLL_INTERVAL}s")
    log(f"Forwarding to {GATEWAY_URL}")

    consecutive_errors = 0

    while True:
        messages = _receive_messages(number)

        if messages:
            consecutive_errors = 0
            for envelope in messages:
                try:
                    _process_envelope(envelope)
                except Exception as e:
                    log(f"Error processing envelope: {e}")
        else:
            consecutive_errors = 0

        # Adaptive backoff on connection errors
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
            resp = _signal_session.get(
                SIGNAL_CLI_URL.rstrip("/") + "/v1/about",
                timeout=5,
            )
            about = resp.json()
            mode = about.get("mode", "?")
            version = about.get("version", "?")
            log(f"signal-cli v{version} ({mode} mode) is ready")
            break
        except Exception:
            time.sleep(3)

    number = _get_number()
    if not number:
        log("ERROR: No Signal number found. Set SIGNAL_NUMBER env var.")
        raise SystemExit(1)

    poll_loop(number)


if __name__ == "__main__":
    main()
