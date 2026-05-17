# Signal Resilience (2026-05-17)

Two host-side layers that keep Signal flowing through gateway hangs and
restarts. Both live *outside* the gateway container by design: an in-process
watchdog can't recover a hung event loop, and an in-process queue can't
survive a container restart that strands its in-memory state.

## Symptom this closes

On 2026-05-17 the gateway HTTP path stopped responding for ~9 minutes after
a restart while the asyncio event loop was saturated by the idle scheduler's
boot-time burst (training_collector running sequential Ollama scoring calls,
the resilience-drill ONNX bundled-MiniLM download, sentience probes, etc.).
The signal-forwarder retried every queued message 4 times with 30s timeouts,
then gave up — dropping 3 messages permanently. `signal-cli`'s
`--receive-mode manual` means once the forwarder has read a message it's the
only durable copy until the gateway acks: signal-cli will never redeliver.

The existing in-container `signal_heartbeat` healing monitor, the `watchdog.py`
daemon-thread reaper, and the runbook dispatcher all live inside the gateway
process. None of them can detect or recover a hung event loop — the patient
can't diagnose its own coma. This document describes the two host-side
primitives that close that gap.

## Layer 1 — Forwarder durable outbox

`signal/forwarder.py` was a fire-and-forget HTTP client with a 4-attempt
in-memory retry budget. After 4 attempts (≈47s elapsed with 2/5/15s backoff
+ four 30s timeouts) it logged "Failed to forward after 4 attempts" and moved
on. No persistence.

The new outbox is a local SQLite file at
`~/.crewai-bridge/signal_outbox.sqlite` (override with
`FORWARDER_OUTBOX_DB`). Every payload pulled from signal-cli is INSERT-ed
*before* any POST is attempted. The schema is:

```sql
CREATE TABLE outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,            -- 'message' | 'reaction'
    payload TEXT NOT NULL,         -- JSON
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL,
    created_at REAL NOT NULL,
    last_error TEXT
);
CREATE INDEX idx_outbox_next ON outbox(next_attempt_at);
```

`_drain_outbox()` runs once per poll loop iteration (every ~0.5–1.5s).
Failed POSTs reschedule on a capped exponential backoff (2s → 5s → 15s → 60s
→ 120s → 300s → 600s); successes DELETE the row. The drain is bounded to 25
rows per call so a large backlog doesn't starve the receive path.

WAL mode + `synchronous=NORMAL` keeps writes fast while still durable across
the forwarder's own crashes. The forwarder also drains on startup so an
orphaned outbox from a previous process flushes immediately, not on next
inbound message.

Belt and suspenders: if the SQLite write itself fails (disk full, permissions),
the forwarder falls back to a single direct POST. We degrade to
pre-durability behaviour rather than silently losing data.

## Layer 2 — Host watchdog

`scripts/gateway_watchdog.py` polls `http://127.0.0.1:8765/health` every
20 s. After 6 consecutive failures (~2 min hung), it runs
`docker compose restart gateway` and sends a Signal alert via signal-cli
JSON-RPC directly (bypassing the gateway — by definition the broken thing).

Knobs (`scripts/gateway_watchdog.plist` env block):

| Var | Default | What |
|---|---|---|
| `HEALTH_URL` | `http://127.0.0.1:8765/health` | endpoint to probe |
| `POLL_INTERVAL_SECONDS` | `20` | gap between probes |
| `HEALTH_TIMEOUT_SECONDS` | `5` | per-probe timeout |
| `FAILURE_THRESHOLD` | `6` | consecutive failures before restart |
| `RESTART_COOLDOWN_SECONDS` | `300` | refuse a second restart inside this window |
| `RESTART_GRACE_SECONDS` | `90` | skip probes for this long after a restart |

The grace window stops the watchdog from re-firing during the gateway's
own boot (~30–90s on this host before HTTP listens). The cooldown stops
restart-thrashing if the gateway keeps hanging — at the bound the operator
has to intervene.

## Installation

```bash
cd /Users/andrus/BotArmy/crewai-team

# Forwarder durability: pick up the new code (signal/forwarder.py)
launchctl bootout gui/$(id -u)/com.botarmy.signal-forwarder
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.botarmy.signal-forwarder.plist

# Watchdog: install + start
./scripts/install_gateway_watchdog.sh install
./scripts/install_gateway_watchdog.sh status      # is it loaded?
tail -F workspace/healing/.gateway_watchdog.log   # observe
```

The outbox SQLite is auto-created at first write.

## Operator verification

After install, expected log signatures:

```
# /tmp/signal-forwarder.log
[forwarder] Outbox at /Users/andrus/.crewai-bridge/signal_outbox.sqlite
[forwarder] Forwarded message to gateway: 200
```

```
# workspace/healing/.gateway_watchdog.log
[watchdog] Starting — poll http://127.0.0.1:8765/health every 20s (timeout 5s), …
[watchdog] Recovered after N failed probe(s)    # only when gateway self-heals
[watchdog] Threshold breached — gateway hung ~120s; restarting
[watchdog] Restart command returned cleanly
```

Inspect the outbox at any time:

```bash
sqlite3 ~/.crewai-bridge/signal_outbox.sqlite \
  "SELECT id, kind, attempts, datetime(created_at,'unixepoch','localtime') AS created, \
   datetime(next_attempt_at,'unixepoch','localtime') AS next, last_error \
   FROM outbox ORDER BY id;"
```

A healthy steady state is 0 rows. A few rows during a gateway hang are
expected and clear out automatically once the watchdog restart completes.
Rows older than a few hours with double-digit attempt counts indicate the
gateway is chronically refusing to accept inbound — that's an underlying
gateway bug, not an outbox problem.

## What this does NOT fix

- The gateway's startup HTTP starvation itself. This work captures it as a
  separate follow-up — the real fix is staggering `app/idle_scheduler.py`'s
  boot-time burst so the event loop stays responsive while the training
  scorer, ONNX downloads, and sentience probes run. Until that lands, the
  watchdog will keep firing periodic restarts and the gateway can spend a
  significant fraction of wall time in boot rather than steady state.
- Outbound message delivery (gateway → Signal). That path goes through the
  separate `host_bridge/main.py` on port 9100 and has its own concerns.
- Anything that requires the gateway to actually finish executing a request.
  The outbox guarantees DELIVERY of the inbound POST; what Commander does
  with it afterward depends on the gateway being healthy enough to run the
  crew.

## Cross-references

- `signal/forwarder.py` — durable outbox implementation
- `scripts/gateway_watchdog.py` — health probe + auto-restart
- `scripts/gateway_watchdog.plist` — launchd LaunchAgent definition
- `scripts/install_gateway_watchdog.sh` — install/start/stop/status helper
- Parallel host-side LaunchAgents using the same pattern:
  `scripts/db_backup_host.plist`, `scripts/warm_spare_host.plist`,
  `scripts/host_substrate_metrics.plist`, `scripts/browse_host_collector.plist`
- In-container layers that this composes with (but does not replace):
  `app/healing/monitors/signal_heartbeat.py`,
  `app/healing/watchdog.py` (daemon-thread reaper),
  `app/conversation_store.py` inbound queue replay at gateway startup.
