"""
Host-side watchdog for the BotArmy gateway container.

The gateway runs background work on the same event loop that serves HTTP,
so a heavy idle-job burst (training scorer, ONNX download, sentience probes)
can starve /signal/inbound for minutes after boot. The forwarder drops
messages whose retry budget elapses inside that window. The in-container
healing layer can't recover from a hung event loop because it lives inside
the same process.

This watchdog runs on the host, polls /health, and restarts the gateway
container when it stays unresponsive past a threshold. It also alerts the
operator via signal-cli directly (bypassing the gateway, since by definition
that's the thing that's broken).

Environment:
    HEALTH_URL              gateway endpoint to poll (default http://127.0.0.1:8765/health)
    POLL_INTERVAL_SECONDS   gap between probes (default 20)
    HEALTH_TIMEOUT_SECONDS  per-probe timeout (default 5)
    FAILURE_THRESHOLD       consecutive failures before restart (default 6 → ~2 min)
    RESTART_COOLDOWN_SECONDS  refuse a second restart inside this window (default 300)
    RESTART_GRACE_SECONDS   skip probes for this long after restart kicks off (default 90)
    COMPOSE_PROJECT_DIR     dir containing docker-compose.yml (default /Users/andrus/BotArmy/crewai-team)
    GATEWAY_SERVICE         compose service name (default gateway)
    SIGNAL_CLI_HTTP_URL     signal-cli JSON-RPC endpoint (default http://127.0.0.1:7583)
    SIGNAL_OWNER_NUMBER     recipient for watchdog alerts (required for alerts to fire)
    DOCKER_BIN              docker binary path (default /usr/local/bin/docker)
    LOG_PATH                file to mirror stdout into (default /tmp/gateway-watchdog.log)
    STATE_PATH              JSON file persisting cooldown state across watchdog
                            crashes (default ~/.crewai-bridge/gateway_watchdog_state.json)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Optional

import requests

HEALTH_URL = os.environ.get("HEALTH_URL", "http://127.0.0.1:8765/health")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_SECONDS", "20"))
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT_SECONDS", "5"))
FAILURE_THRESHOLD = int(os.environ.get("FAILURE_THRESHOLD", "6"))
RESTART_COOLDOWN = float(os.environ.get("RESTART_COOLDOWN_SECONDS", "300"))
RESTART_GRACE = float(os.environ.get("RESTART_GRACE_SECONDS", "90"))
COMPOSE_PROJECT_DIR = os.environ.get(
    "COMPOSE_PROJECT_DIR", "/Users/andrus/BotArmy/crewai-team"
)
GATEWAY_SERVICE = os.environ.get("GATEWAY_SERVICE", "gateway")
SIGNAL_CLI_URL = os.environ.get("SIGNAL_CLI_HTTP_URL", "http://127.0.0.1:7583")
SIGNAL_OWNER = os.environ.get("SIGNAL_OWNER_NUMBER", "")
DOCKER_BIN = os.environ.get("DOCKER_BIN", "/usr/local/bin/docker")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/gateway-watchdog.log")
STATE_PATH = os.environ.get(
    "STATE_PATH",
    os.path.expanduser("~/.crewai-bridge/gateway_watchdog_state.json"),
)

_session = requests.Session()


def log(msg: str) -> None:
    # launchd routes both stdout and stderr to LOG_PATH via the plist, so
    # printing once is enough — explicit file writes would duplicate every
    # entry. LOG_PATH is kept as an env var for documentation + future use
    # (e.g. an out-of-launchd run via `python -u gateway_watchdog.py`).
    print(f"[watchdog] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}", flush=True)


# ── Cooldown-state persistence (C3, 2026-05-18) ──────────────────────────
#
# Pre-fix: last_restart_at was a local variable in main(). A watchdog
# crash + launchd respawn (ThrottleInterval=30) reset the cooldown
# state, so the new watchdog could immediately restart the gateway on
# the next failure threshold — defeating the cooldown's anti-thrashing
# purpose. ThrottleInterval + grace + threshold combined gave a natural
# ~3 min backoff so the bug never bit in practice, but the cooldown's
# stated contract is now properly enforced across crashes.


def _load_restart_state() -> Optional[float]:
    """Restore last_restart_at from disk on watchdog startup.

    Returns None on first run or any failure — caller treats that as
    'no prior restart on record'. Stale state (older than the cooldown
    window) is treated as None too, since it can no longer gate
    anything. Never crashes the watchdog.
    """
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = data.get("last_restart_at")
        if not isinstance(ts, (int, float)):
            return None
        if (time.time() - float(ts)) >= RESTART_COOLDOWN:
            # State is stale — cooldown already elapsed.
            return None
        return float(ts)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_restart_state(ts: float) -> None:
    """Persist last_restart_at to disk. Best-effort — a write failure
    just means the cooldown won't survive a watchdog crash, which is
    the pre-C3 behavior we still degrade to gracefully."""
    try:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"last_restart_at": float(ts)}, f)
        os.replace(tmp, STATE_PATH)
    except OSError as e:
        log(f"Failed to persist restart state to {STATE_PATH}: {e}")


def probe_health() -> bool:
    try:
        resp = _session.get(HEALTH_URL, timeout=HEALTH_TIMEOUT)
        return 200 <= resp.status_code < 300
    except requests.exceptions.RequestException:
        return False
    except Exception as e:
        log(f"probe raised unexpected: {e}")
        return False


def signal_alert(text: str) -> None:
    """Send a Signal alert via signal-cli JSON-RPC directly.

    Skips silently if no recipient is configured — alerts are nice-to-have,
    the recovery action is the load-bearing part.
    """
    if not SIGNAL_OWNER:
        return
    try:
        resp = requests.post(
            SIGNAL_CLI_URL.rstrip("/") + "/api/v1/rpc",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "send",
                "params": {"recipient": [SIGNAL_OWNER], "message": text},
            },
            timeout=10,
        )
        if resp.status_code != 200:
            log(f"signal alert HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        log(f"signal alert failed: {e}")


def restart_gateway() -> bool:
    log(f"Restarting compose service '{GATEWAY_SERVICE}' in {COMPOSE_PROJECT_DIR}")
    try:
        result = subprocess.run(
            [DOCKER_BIN, "compose", "restart", GATEWAY_SERVICE],
            cwd=COMPOSE_PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            log("Restart command returned cleanly")
            return True
        log(f"Restart failed (rc={result.returncode}): {result.stderr.strip()[:500]}")
        return False
    except subprocess.TimeoutExpired:
        log("Restart command timed out after 120s")
        return False
    except FileNotFoundError:
        log(f"docker binary not found at {DOCKER_BIN}; install or set DOCKER_BIN")
        return False
    except Exception as e:
        log(f"Restart raised: {e}")
        return False


def main() -> int:
    log(f"Starting — poll {HEALTH_URL} every {POLL_INTERVAL:.0f}s "
        f"(timeout {HEALTH_TIMEOUT:.0f}s), restart after {FAILURE_THRESHOLD} consecutive failures, "
        f"cooldown {RESTART_COOLDOWN:.0f}s, grace {RESTART_GRACE:.0f}s")
    if not SIGNAL_OWNER:
        log("SIGNAL_OWNER_NUMBER not set; alerts disabled (recovery still runs)")

    consecutive_failures = 0
    last_restart_at: Optional[float] = _load_restart_state()
    if last_restart_at is not None:
        remaining = int(RESTART_COOLDOWN - (time.time() - last_restart_at))
        log(
            f"Restored prior restart state — cooldown {remaining}s remaining "
            f"(state file: {STATE_PATH})"
        )
    grace_until: float = 0.0

    while True:
        now = time.time()
        if now < grace_until:
            # Inside post-restart grace; don't even probe yet.
            time.sleep(min(POLL_INTERVAL, grace_until - now))
            continue

        ok = probe_health()
        if ok:
            if consecutive_failures:
                log(f"Recovered after {consecutive_failures} failed probe(s)")
                consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                log("First failed probe — watching")
            elif consecutive_failures % max(1, FAILURE_THRESHOLD // 2) == 0:
                log(f"Failed probe {consecutive_failures}/{FAILURE_THRESHOLD}")

            if consecutive_failures >= FAILURE_THRESHOLD:
                in_cooldown = (
                    last_restart_at is not None
                    and (now - last_restart_at) < RESTART_COOLDOWN
                )
                if in_cooldown:
                    remaining = int(RESTART_COOLDOWN - (now - last_restart_at))
                    log(f"Threshold breached but cooldown active ({remaining}s remaining)")
                else:
                    elapsed_hung = int(consecutive_failures * POLL_INTERVAL)
                    log(f"Threshold breached — gateway hung ~{elapsed_hung}s; restarting")
                    signal_alert(
                        f"⚠️ Gateway watchdog: /health unresponsive for ~{elapsed_hung}s. "
                        f"Restarting {GATEWAY_SERVICE} now."
                    )
                    success = restart_gateway()
                    last_restart_at = time.time()
                    _save_restart_state(last_restart_at)
                    grace_until = last_restart_at + RESTART_GRACE
                    consecutive_failures = 0
                    if success:
                        signal_alert(
                            f"♻️ Gateway restart issued. Probing again in {int(RESTART_GRACE)}s."
                        )
                    else:
                        signal_alert(
                            "❌ Gateway restart FAILED — manual intervention needed."
                        )

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted; exiting")
        sys.exit(0)
