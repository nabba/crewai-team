"""
Forwards inbound Signal messages from signal-cli JSON-RPC to the FastAPI gateway.

signal-cli must be running in daemon mode with --receive-mode manual:
    signal-cli -a +NUMBER daemon --http 127.0.0.1:7583 --receive-mode manual

Environment variables:
    GATEWAY_SECRET        — shared secret for authenticating with the gateway
    SIGNAL_CLI_HTTP_URL   — signal-cli HTTP endpoint (default: http://127.0.0.1:7583)
    GATEWAY_URL           — gateway inbound endpoint (default: http://127.0.0.1:8765/signal/inbound)
    FORWARDER_OUTBOX_DB   — SQLite path for the durable outbox (default: ~/.crewai-bridge/signal_outbox.sqlite)

Durable outbox: every payload pulled from signal-cli is persisted to a local
SQLite WAL before the gateway POST is attempted. Failed deliveries stay queued
with exponential backoff (cap 600s) and drain on the next loop iteration,
through gateway restarts or hangs. signal-cli's manual receive mode means once
the forwarder reads a message it's the only durable copy until the gateway
acks — the outbox closes that loss window.
"""
import json
import os
import sqlite3
import sys
import time
import requests

SIGNAL_CLI_URL = os.environ.get("SIGNAL_CLI_HTTP_URL", "http://127.0.0.1:7583")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:8765/signal/inbound")
GATEWAY_SECRET = os.environ.get("GATEWAY_SECRET", "")
OUTBOX_DB = os.environ.get(
    "FORWARDER_OUTBOX_DB",
    os.path.expanduser("~/.crewai-bridge/signal_outbox.sqlite"),
)

_signal_session = requests.Session()
_signal_session.headers["Content-Type"] = "application/json"
_gateway_session = requests.Session()

# Backoff ladder for redelivery attempts; capped so a long gateway outage
# doesn't push the next retry hours into the future.
_BACKOFF_SECONDS = [2, 5, 15, 60, 120, 300, 600]
_DRAIN_BATCH = 25  # cap per loop iteration so a large backlog doesn't starve the receive path
_OUTBOX_REPORT_INTERVAL = 60.0  # log queue depth at most once a minute

# Terminal cap. A permanently-broken GATEWAY_URL would otherwise grow
# the outbox forever. After this many attempts the row moves to a
# dead-letter table so the operator can inspect / replay manually.
#
# 4320 × 600 s (the 600 s tail of the backoff ladder) ≈ 30 days. Sized
# for "operator on vacation, gateway crashed, watchdog also down" —
# the worst plausible compound failure for an unattended laptop. Any
# row in outbox_dead is the unambiguous signal that something is
# permanently broken, not just temporarily unreachable. Configurable
# via FORWARDER_MAX_ATTEMPTS env var for operator override.
_MAX_ATTEMPTS = int(os.environ.get("FORWARDER_MAX_ATTEMPTS", "4320"))

# Per-attempt POST timeout. Bumped 30→60 (2026-05-18) so a transient
# gateway slow-moment (post-boot idle-scheduler burst, etc.) doesn't
# trip the per-attempt deadline and force an unnecessary backoff. The
# durable outbox still owns the retry semantics — this just widens
# the window for each attempt before the backoff ladder kicks in.
_GATEWAY_POST_TIMEOUT_S = 60


def log(msg):
    print(f"[forwarder] {msg}", flush=True)


# ---- Durable outbox ---------------------------------------------------------

def _outbox_conn():
    """Open the outbox SQLite, creating the dir + schema if needed.

    WAL mode + busy_timeout keeps concurrent readers (e.g. an operator running
    sqlite3 on the file) from blocking the forwarder.
    """
    os.makedirs(os.path.dirname(OUTBOX_DB), exist_ok=True)
    conn = sqlite3.connect(OUTBOX_DB, timeout=10, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at REAL NOT NULL,
            created_at REAL NOT NULL,
            last_error TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_outbox_next ON outbox(next_attempt_at)"
    )
    # Dead-letter table for rows that exceed _MAX_ATTEMPTS. Kept on disk
    # rather than dropped so the operator can salvage payloads after
    # diagnosing the underlying gateway problem.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outbox_dead (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            created_at REAL NOT NULL,
            killed_at REAL NOT NULL,
            last_error TEXT
        )
        """
    )
    return conn


def _enqueue(kind: str, payload: dict) -> None:
    """Persist a payload to the outbox so it survives crashes and gateway hangs."""
    try:
        conn = _outbox_conn()
        try:
            now = time.time()
            conn.execute(
                "INSERT INTO outbox (kind, payload, attempts, next_attempt_at, created_at) "
                "VALUES (?, ?, 0, ?, ?)",
                (kind, json.dumps(payload), now, now),
            )
        finally:
            conn.close()
    except Exception as e:
        # Catastrophic: outbox unwritable. Log loudly but don't drop the
        # message — fall back to a best-effort direct POST so we degrade to
        # pre-durability behaviour rather than silently losing data.
        log(f"OUTBOX WRITE FAILED ({e}); falling back to direct POST")
        _direct_post(kind, payload)


def _direct_post(kind: str, payload: dict) -> bool:
    """Last-resort direct POST when the outbox itself can't be written.

    Returns True on success. Single attempt, no retries — we don't have
    durable state to track them.
    """
    headers = {}
    if GATEWAY_SECRET:
        headers["Authorization"] = f"Bearer {GATEWAY_SECRET}"
    try:
        resp = _gateway_session.post(GATEWAY_URL, json=payload, headers=headers, timeout=_GATEWAY_POST_TIMEOUT_S)
        log(f"Direct {kind} POST: {resp.status_code}")
        return 200 <= resp.status_code < 300
    except Exception as e:
        log(f"Direct {kind} POST failed: {e}")
        return False


_last_outbox_report = 0.0


def _drain_outbox() -> None:
    """Attempt redelivery of any pending rows whose backoff has elapsed.

    Bounded by _DRAIN_BATCH per call so the receive loop keeps running even
    with a large backlog. Returns nothing; failures stay queued.
    """
    global _last_outbox_report
    try:
        conn = _outbox_conn()
    except Exception as e:
        log(f"OUTBOX OPEN FAILED ({e}); skipping drain this cycle")
        return

    try:
        now = time.time()
        rows = conn.execute(
            "SELECT id, kind, payload, attempts FROM outbox "
            "WHERE next_attempt_at <= ? ORDER BY id ASC LIMIT ?",
            (now, _DRAIN_BATCH),
        ).fetchall()

        headers = {}
        if GATEWAY_SECRET:
            headers["Authorization"] = f"Bearer {GATEWAY_SECRET}"

        for row_id, kind, payload_json, attempts in rows:
            try:
                payload = json.loads(payload_json)
            except Exception as e:
                log(f"Outbox row {row_id} unparseable ({e}); dropping")
                conn.execute("DELETE FROM outbox WHERE id = ?", (row_id,))
                continue

            try:
                resp = _gateway_session.post(
                    GATEWAY_URL, json=payload, headers=headers, timeout=_GATEWAY_POST_TIMEOUT_S,
                )
                if 200 <= resp.status_code < 300:
                    conn.execute("DELETE FROM outbox WHERE id = ?", (row_id,))
                    if attempts > 0:
                        log(f"Outbox row {row_id} delivered after {attempts + 1} attempts ({kind})")
                    else:
                        log(f"Forwarded {kind} to gateway: {resp.status_code}")
                else:
                    _reschedule(conn, row_id, attempts, f"HTTP {resp.status_code}")
            except requests.exceptions.RequestException as e:
                _reschedule(conn, row_id, attempts, str(e)[:200])
            except Exception as e:
                _reschedule(conn, row_id, attempts, f"unexpected: {str(e)[:200]}")

        # Periodic depth report so a slow-growing backlog is visible.
        if (now - _last_outbox_report) >= _OUTBOX_REPORT_INTERVAL:
            pending = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
            if pending:
                oldest = conn.execute(
                    "SELECT MIN(created_at) FROM outbox"
                ).fetchone()[0]
                age = int(now - oldest) if oldest else 0
                log(f"Outbox depth: {pending} pending, oldest {age}s old")
            _last_outbox_report = now
    finally:
        conn.close()


def _reschedule(conn: sqlite3.Connection, row_id: int, attempts: int, err: str) -> None:
    """Bump attempts + push next_attempt_at out by the backoff ladder.

    Once ``attempts + 1 >= _MAX_ATTEMPTS`` the row is moved to the
    ``outbox_dead`` dead-letter table instead of being rescheduled, so a
    permanently broken endpoint cannot grow the live outbox forever.
    """
    new_attempts = attempts + 1
    if new_attempts >= _MAX_ATTEMPTS:
        try:
            row = conn.execute(
                "SELECT kind, payload, created_at FROM outbox WHERE id = ?",
                (row_id,),
            ).fetchone()
            if row is not None:
                kind, payload, created_at = row
                conn.execute(
                    "INSERT INTO outbox_dead "
                    "(kind, payload, attempts, created_at, killed_at, last_error) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (kind, payload, new_attempts, created_at, time.time(), err),
                )
            conn.execute("DELETE FROM outbox WHERE id = ?", (row_id,))
            log(
                f"OUTBOX GIVE-UP row {row_id} after {new_attempts} attempts ({err}); "
                f"moved to outbox_dead. Inspect with: "
                f"sqlite3 {OUTBOX_DB} 'SELECT * FROM outbox_dead;'"
            )
        except Exception as move_err:
            # If we can't move it, leave the row alone — the next pass
            # will try again. Better to over-retry than silently drop.
            log(f"OUTBOX GIVE-UP move failed for row {row_id}: {move_err}")
        return

    idx = min(attempts, len(_BACKOFF_SECONDS) - 1)
    delay = _BACKOFF_SECONDS[idx]
    next_at = time.time() + delay
    conn.execute(
        "UPDATE outbox SET attempts = attempts + 1, next_attempt_at = ?, last_error = ? "
        "WHERE id = ?",
        (next_at, err, row_id),
    )
    # First failure logs at INFO; sustained failures only every 5 attempts
    # to keep the log readable during long outages.
    if attempts == 0 or attempts % 5 == 0:
        log(f"Outbox row {row_id} attempt {new_attempts} failed ({err}); retry in {delay}s")


def _receive_messages():
    """Single receive call with short timeout.

    Returns: list of messages, or None on connection error (distinct from empty []).
    """
    try:
        resp = _signal_session.post(
            SIGNAL_CLI_URL.rstrip("/") + "/api/v1/rpc",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "receive",
                "params": {"timeout": 1},
            },
            timeout=10,
        )
        data = resp.json()
        if "error" in data:
            err = data["error"].get("message", "")
            if "already being received" in err:
                return []
            log(f"receive error: {err}")
            return []
        return data.get("result", [])
    except requests.exceptions.ConnectionError:
        return None  # Signal connection error — distinct from empty
    except requests.exceptions.ReadTimeout:
        return []
    except Exception as e:
        log(f"receive failed: {e}")
        return None


def _check_signal_cli_alive() -> bool:
    """Quick health check on signal-cli."""
    try:
        resp = _signal_session.post(
            SIGNAL_CLI_URL.rstrip("/") + "/api/v1/rpc",
            json={"jsonrpc": "2.0", "id": 1, "method": "version"},
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _wait_for_signal_cli():
    """Block until signal-cli responds, with exponential backoff.

    Starts at 5s, backs off to max 60s. After 5 minutes of failure,
    reports to Firebase dashboard so the owner has visibility.
    """
    log("Waiting for signal-cli to come back...")
    wait = 5
    max_wait = 60
    start = time.time()
    alerted = False

    while True:
        if _check_signal_cli_alive():
            elapsed = time.time() - start
            log(f"signal-cli reconnected after {elapsed:.0f}s")
            return

        # Alert after 5 minutes of continuous failure
        if not alerted and (time.time() - start) > 300:
            alerted = True
            log("WARNING: signal-cli down for >5 minutes — reporting to dashboard")
            try:
                _gateway_session.post(
                    GATEWAY_URL.replace("/signal/inbound", "/location"),
                    json={"signal_cli_down": True, "down_since": start},
                    timeout=5,
                )
            except Exception:
                pass

        time.sleep(wait)
        wait = min(max_wait, wait * 1.5)  # Exponential backoff, cap at 60s


def _process_envelope(envelope: dict) -> None:
    """Extract message data from an envelope and forward to the gateway."""
    data_msg = envelope.get("dataMessage")
    if not data_msg:
        return

    # Handle emoji reactions (feedback signals)
    reaction = data_msg.get("reaction")
    if reaction:
        sender = envelope.get("source") or envelope.get("sourceNumber")
        if not sender:
            return
        emoji = reaction.get("emoji", "")
        target_ts = reaction.get("targetSentTimestamp", 0)
        is_remove = reaction.get("isRemove", False)
        log(f"Reaction from {sender[-4:]}: {emoji} on ts={target_ts} (remove={is_remove})")

        _enqueue("reaction", {
            "type": "reaction_feedback",
            "sender": sender,
            "emoji": emoji,
            "target_timestamp": target_ts,
            "is_remove": is_remove,
        })
        return

    if not data_msg.get("message") and not data_msg.get("attachments"):
        return

    sender = envelope.get("source") or envelope.get("sourceNumber")
    if not sender:
        return

    message = data_msg.get("message", "")
    timestamp = data_msg.get("timestamp") or envelope.get("timestamp", 0)

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

    _enqueue("message", {
        "sender": sender,
        "message": message,
        "timestamp": timestamp,
        "attachments": attachments,
    })


_LOCATION_FILE = "/tmp/botarmy-location.json"
_LOCATION_HELPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "location-helper")
_LOCATION_INTERVAL = 1800  # 30 minutes
_last_location_probe = 0.0


def _probe_location():
    """Try to get location via CoreLocation helper and write to shared file.

    Best-effort: if helper binary doesn't exist or fails, silently skip.
    """
    global _last_location_probe
    now = time.time()
    if now - _last_location_probe < _LOCATION_INTERVAL:
        return
    _last_location_probe = now

    if not os.path.exists(_LOCATION_HELPER):
        return

    try:
        import subprocess
        result = subprocess.run(
            [_LOCATION_HELPER],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            if "lat" in data and "lon" in data:
                with open(_LOCATION_FILE, "w") as f:
                    json.dump(data, f)
                log(f"Location updated: {data.get('lat', '?'):.4f}, {data.get('lon', '?'):.4f} "
                    f"(±{data.get('accuracy', '?')}m)")
    except Exception as e:
        log(f"Location probe failed (non-fatal): {e}")


def poll_loop():
    """Poll signal-cli for messages and forward them.

    Each iteration also drains the durable outbox so any queued payload from
    a prior gateway hang gets retried as soon as the gateway is healthy again.
    """
    log(f"Polling signal-cli at {SIGNAL_CLI_URL} every ~1.5s")
    log(f"Forwarding to {GATEWAY_URL}")
    log(f"Outbox at {OUTBOX_DB}")
    _consecutive_errors = 0
    _MAX_ERRORS = 60  # ~30s of consecutive connection failures triggers reconnect

    # Drain anything left over from a prior process so an early crash doesn't
    # delay redelivery until the next inbound message arrives.
    _drain_outbox()

    while True:
        # Periodic location probe (every 30 min, non-blocking)
        try:
            _probe_location()
        except Exception:
            pass

        messages = _receive_messages()
        if messages is None:
            # Connection error — signal-cli may be down
            _consecutive_errors += 1
        elif messages:
            _consecutive_errors = 0
            for msg in messages:
                envelope = msg.get("envelope", msg)
                try:
                    _process_envelope(envelope)
                except Exception as e:
                    log(f"Error processing envelope: {e}")
        else:
            # Empty list — normal, no new messages
            _consecutive_errors = 0

        # Reconnect if signal-cli has been unresponsive for ~30s
        if _consecutive_errors >= _MAX_ERRORS:
            log(f"signal-cli unresponsive ({_consecutive_errors} consecutive errors)")
            _wait_for_signal_cli()
            _consecutive_errors = 0

        # Drain the outbox every cycle so retries fire on the backoff cadence
        # rather than waiting for the next inbound message.
        try:
            _drain_outbox()
        except Exception as e:
            log(f"Drain pass raised: {e}")

        # Gap between polls — signal-cli needs a brief pause to release the lock
        time.sleep(0.5)


def main():
    if not GATEWAY_SECRET:
        log("WARNING: GATEWAY_SECRET not set — requests will be rejected by gateway")

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
