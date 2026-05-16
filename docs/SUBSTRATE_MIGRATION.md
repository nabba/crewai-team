# Substrate Migration Playbook

**Status (2026-05-16):** Q16 Theme 1 — decade-resilience hardening.
Companion to the `host_substrate_health` monitor (PROGRAM §51,
the 35th healing monitor). Sibling Q16 artifacts:
`oauth_token_freshness` (36th monitor — vendor independence),
`vendor_independence` drill (5th drill — cascade routing),
`operator_anomaly` (37th monitor — operator-pattern shift).

## Why this exists

The Q6 posture decision (`docs/RESILIENCE_POSTURE.md`) committed to
"good backup + fast bare-metal recovery, not HA." That decision has
a quiet assumption baked into it: **the operator's primary host
stays accessible**. Over a 5-to-10-year horizon, the failure modes
that erode this assumption are not the ones the existing 34
monitors watch:

  * **Consumer M-series SSDs** sometimes hit TBW exhaustion at
    year 4-5. Drive starts reallocating sectors; performance
    degrades; eventual write-failure.
  * **macOS major version transitions** sometimes drop Docker
    Desktop support, break Python compatibility, or change the
    permission model in ways that require operator action.
  * **Laptop replacement** happens. The DR drill restores the
    workspace tarball, but the transition window — `git clone` +
    install dependencies + restore secrets + verify health — is
    unscripted.
  * **24/7 thermal cycling** wears every solder joint over years.

None of this is imminent on a healthy system. All of it is in-scope
for "install once and run for many years."

## What the monitor sees (and what it can't)

`app/healing/monitors/host_substrate_health.py` runs inside the
gateway container. From inside, it can see:

  * Workspace volume free-space **trend** (slope, not just
    threshold). Projects "days until full" at the current burn
    rate; alerts at <60d horizon long before `disk_quota` trips.
  * Workspace bytes growth — sustained week-over-week >10% for
    4+ weeks → "what's growing?" alert.
  * Gateway restart bursts (>3 in 24h) — instability signal.
  * Process uptime >180d — likely running stale code that won't
    pick up self-applied amendments until a restart.
  * Linux memory headroom (when /proc/meminfo available).

What it **cannot** see from inside the container:

  * SMART data (reallocated sectors, temperature, TBW).
  * macOS version + EOL distance.
  * Host filesystem health beyond the workspace mount.
  * Docker Desktop VM sizing vs. host capacity.

These need a host-side companion. See the optional companion below.

## Optional host-side companion (parallels Q15 browse split)

If you want SMART + macOS visibility, run a periodic host-side
shell script that writes one JSONL row per cadence into
`workspace/healing/host_metrics.jsonl`. The monitor surfaces the
last row in its summary without imposing a schema — anything you
write is visible.

Suggested cadence: weekly via launchd LaunchAgent (parallels the
Q15 browse host-collector pattern).

### Minimal collector example

```bash
#!/bin/bash
# Save as ~/Library/Application Scripts/com.andrusai.host-metrics.sh
# Schedule via a launchd LaunchAgent (weekly).

WORKSPACE="/Users/andrus/BotArmy/workspace"
OUT="$WORKSPACE/healing/host_metrics.jsonl"
mkdir -p "$(dirname "$OUT")"

# SMART data — needs `brew install smartmontools`. APFS containers on
# Apple Silicon show up as /dev/disk0; verify with `diskutil list`.
SMART_JSON=$(sudo smartctl -j -A /dev/disk0 2>/dev/null || echo '{}')
SMART_REALLOCATED=$(echo "$SMART_JSON" | jq -r '.ata_smart_attributes.table[]? | select(.name=="Reallocated_Sector_Ct") | .raw.value // 0' 2>/dev/null || echo 0)
SMART_TEMP=$(echo "$SMART_JSON" | jq -r '.temperature.current // 0' 2>/dev/null || echo 0)

# macOS version.
MACOS_VERSION=$(sw_vers -productVersion)

# Docker VM allocation (Docker Desktop only).
DOCKER_DISK=$(docker system df --format json 2>/dev/null | jq -r '.[]? | select(.Type=="Volumes") | .Size' || echo "")

# Emit one row.
echo "{\"ts\": $(date +%s), \"smart_reallocated_sectors\": ${SMART_REALLOCATED:-0}, \"smart_temperature_c\": ${SMART_TEMP:-0}, \"macos_version\": \"$MACOS_VERSION\", \"docker_volume_size\": \"$DOCKER_DISK\"}" >> "$OUT"

# Cap at 200 rows.
tail -n 200 "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"
```

### LaunchAgent skeleton

`~/Library/LaunchAgents/com.andrusai.host-metrics.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.andrusai.host-metrics</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>/Users/andrus/Library/Application Scripts/com.andrusai.host-metrics.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key><integer>0</integer>
        <key>Hour</key><integer>3</integer>
    </dict>
    <key>StandardOutPath</key><string>/tmp/host-metrics.log</string>
    <key>StandardErrorPath</key><string>/tmp/host-metrics.err</string>
</dict>
</plist>
```

Load with `launchctl load ~/Library/LaunchAgents/com.andrusai.host-metrics.plist`.

The companion is **deliberately optional**. The in-container monitor
gives useful signal on its own; SMART data adds early warning on a
slow failure mode.

## Q16 Theme 2 — vendor independence (composes with Theme 1)

The substrate-longevity story isn't complete without the cascade
that runs on it. Two artifacts ship alongside `host_substrate_health`:

  * **`oauth_token_freshness`** (36th healing monitor). Daily probe;
    pure file inspection. Watches:
      - Google Workspace refresh token at `workspace/google_token.json`
        (file mtime as freshness proxy; Google invalidates refresh
        tokens after ~180d of inactivity).
      - Vendor API key formats: Anthropic `sk-ant-…`, OpenAI `sk-…`,
        OpenRouter `sk-or-…`, Groq `gsk_…`. Mismatch → warn (probably
        a typo or vendor format change).
      - VAPID keypair completeness at `workspace/vapid_*.pem`. Half-
        present pair → warn (botched rotation).
    Never calls an external API. Never logs the secret value itself
    (only a 4-hex-char SHA-256 prefix when a stable identifier is
    needed).

  * **`vendor_independence`** drill (5th in the Q6 registry, LOW
    risk). Quarterly auto-run. Verifies the LLM cascade can route
    past the dominant providers (Anthropic + OpenRouter) without
    an outage. Structural checks only:
      - At least 2 non-dominant fallback options are configured
        (Ollama, Groq, Gemini, DeepSeek direct, MiniMax direct).
      - Ollama port reachable (TCP probe, no model call).
      - `llm_selector` source has a blocklist mechanism (so a
        misbehaving dominant provider CAN be routed around).
      - Documented chain in `docs/LLM_SUBSYSTEM.md` matches
        structural presence.
    Master switch `drill_vendor_independence_enabled` (default ON).

## Q16 Theme 3 (partial) — operator anomaly

The 37th monitor `operator_anomaly` is an observational companion
to the existing `unauthorized_sender` audit event. The hard
boundary (allow-listed senders) stays where it is; this monitor
softer-signals dramatic *behavior* shifts in the authorized sender
pattern:

  * Hour-of-day distribution shift (recent 7d ≥3× baseline 30d in
    any 6h bucket).
  * Cadence spike (≥2×) or quiet (≤0.5×).
  * Message-length median shift (≥4×).
  * New authorized sender appearing in last 7d but not in prior 90d.
    Fires as **critical** so it bypasses arbiter suppression.

Reads `workspace/audit.log` `request_received` rows. The audit log
records timestamps + sender + message-length only — never content.
Privacy boundary preserved.

**This monitor never blocks or refuses anything.** It surfaces
information; the operator decides. The defense lever — vacation
mode with allowlisted auto-apply — is deferred to a separate Q16
follow-on so its security contract gets its own review.

## Migration procedure: moving to a new host

If `host_substrate_health` alerts on disk-horizon-warn or
restart-burst, or you want to proactively move (new laptop, year-3
hardware refresh, going Linux), the operationally-correct sequence:

### 1. Pre-migration on the OLD host (1-2 hours)

  * Verify the latest DR drill passed: check
    `workspace/resilience/drill_audit.jsonl` for a recent
    `backup_restore` PASS.
  * Run an extra DR export manually:
    `scripts/dr_boot_drill.sh export`. This writes a tarball to
    `workspace/backups/dr/`.
  * Tag the git HEAD with `git tag substrate-migration-<date>`.
  * Make a fresh `git bundle` of the repo:
    `git bundle create /tmp/botarmy.bundle --all`.
  * Note which secrets need transferring (see secrets checklist
    below).
  * Drain idle jobs cleanly: stop the gateway with
    `docker compose down --timeout 60` so any in-flight
    coding-session worktrees serialize.

### 2. Transfer (manual; depends on circumstance)

  * Copy the DR tarball + git bundle + secrets manifest to the
    NEW host via a trusted channel. SSH over Tailscale is the
    standard path.
  * **Do not** transfer the docker volumes directly — they're
    not portable across host architectures. The DR tarball
    contains the structured exports.

### 3. Bootstrap on the NEW host (4-6 hours)

  * Install prerequisites: Docker Desktop (or Docker on Linux),
    Python 3.11+, git.
  * `git clone <bundle>` → restore the repo.
  * Restore secrets (see checklist).
  * `docker compose up -d` to bring up PostgreSQL, Neo4j, ChromaDB.
  * Wait for the three KB containers to become healthy
    (`docker compose ps`).
  * Import the DR tarball:
    `scripts/dr_boot_drill.sh import --from workspace/backups/dr/<tarball>`.
  * Start the gateway: `docker compose up -d gateway`.
  * Verify health: open `https://<host>/cp/monitor` — all three
    KB containers should be green.

### 4. Post-migration verification (1 hour)

  * Run the `backup_restore` drill explicitly via
    `python -m app.resilience_drills run backup_restore` —
    confirms the import round-trips cleanly.
  * Re-run the `migration_drill` against today's schema:
    `bash deploy/scripts/migration-drill.sh`.
  * Run the new `vendor_independence` drill:
    `python -m app.resilience_drills run vendor_independence`
    — confirms the cascade has its fallback options on the new
    host.
  * Open a Signal chat and run a few smoke queries — `/status`,
    `/threads list`, `/skill list` — to confirm the gateway
    answers normally.
  * Check `host_substrate_health` is alive in the daemon driver
    log (look for `healing.monitors: driver running N monitors`
    on startup, then a probe within the first 30 minutes).
  * Check `oauth_token_freshness` sees the freshly-bootstrapped
    Google token and vendor keys.
  * Tag the new host's HEAD: `git tag substrate-migrated-<date>`.

### Secrets checklist

The DR tarball deliberately excludes these (it's the operator's
responsibility to transport them via a safe channel):

  * `.env` — provider API keys (Anthropic, OpenAI, OpenRouter,
    Groq, Aviationstack, etc.)
  * `google_token.json` — Google Workspace OAuth refresh token
  * `vapid_*.pem` — Web Push signing keys
  * `secrets/*` — any operator-managed secret material
  * Tailscale auth keys, if applicable
  * Signal-CLI registration state (if the migration includes
    moving the Signal number — usually it does NOT; the new host
    talks to the existing Signal-CLI container or re-registers)

### Dry-run cadence

The migration procedure above is exercised by the existing
`backup_restore` drill (Q6.2) — that drill validates the
import-from-tarball path quarterly. It does **not** validate
moving to a *different* host shape.

Recommendation: once a year, do a no-pressure dry-run by
importing the DR tarball into a scratch Linux VM (Multipass /
Lima / a cloud VPS). Run the smoke queries. Confirm dependencies
install. Discard the VM. This catches host-shape-specific failures
(libc divergence, missing kernel features, broken docker compose
profiles) in a low-stakes setting before you ever need them in a
high-stakes setting.

## When the monitor alerts: triage

### `disk_horizon_warn` (<60d to full)

  1. `du -sh workspace/*` to find the dominant directory.
  2. If `workspace/chromadb/`: run the `chromadb_hygiene` monitor
     immediately + consider a manual rebuild via
     `python -m app.memory.chromadb_rebuild --dry-run` first.
  3. If `workspace/affect/` or `workspace/welfare/`: the JSONL
     rotation should have caught this — check
     `workspace/<name>/archive/` for accumulating monthly
     buckets and consider pruning the oldest.
  4. If none of the above: expand the volume (Docker Desktop
     resource settings) or move workspace to a larger disk.

### `workspace_growth_burst` (≥10% WoW for 4+ weeks)

  1. Sort `workspace/*` by recent mtime — look for new
     directories or rapidly-growing files.
  2. Check `workspace/healing/` and `workspace/audit/` for
     log writers that are accumulating without rotation.
  3. If the growth is `workspace/coding_sessions/` or
     `workspace/change_requests/`: check the relevant retention
     monitor is actually firing (cron_liveness for the daemon).

### `restart_burst` (>3 restarts/24h)

  1. `docker logs <gateway>` for the latest exit reason.
  2. Check `workspace/errors.jsonl` for the pattern signature.
  3. If OOM: increase Docker Desktop memory allocation OR find
     the leak via the `pattern_learner` digest.
  4. If a clean Python crash: an unhandled exception in a daemon
     thread that the watchdog can't catch. Check the auditor-
     bridge for an `error_fix_proposed` CR.

### `uptime_stale` (>180d)

  1. Confirm there's a recent DR backup.
  2. `docker compose restart gateway` — workspace state persists.
  3. Check the post-restart self-check ran cleanly (see
     `_process_post_amendment_restart_claims` log line).

### `memory_pressure` (<10% headroom for 4+ weeks)

  1. On Mac: check Activity Monitor — if the host has plenty of
     free memory but the container is starved, the Docker
     Desktop VM allocation is the bottleneck.
  2. On Linux: `docker stats` shows per-container memory; find
     the one consuming the most.
  3. If the gateway itself: look for unbounded caches.

### `oauth_token_freshness` Google warn/crit

  1. Trigger any Google Workspace tool to force a refresh —
     `/gmail recent`, `/calendar today`, etc. The
     googleapiclient writes a new token on every refresh.
  2. If still failing, re-run
     `python -m app.google_workspace.bootstrap` to re-authorize
     against Google's OAuth flow.

### `oauth_token_freshness` vendor critical/warn

  1. Missing Anthropic: set `ANTHROPIC_API_KEY` and restart
     (sudden absence triggers the cascade's failover, but the
     missing key is operator-actionable).
  2. Format mismatch: verify against the vendor's current docs.
     If the vendor changed key shape, update
     `app/healing/monitors/oauth_token_freshness.py`
     `_VENDOR_KEY_PATTERNS` to track the new format.

### `operator_anomaly` new_sender

  1. Confirm you actually added the new sender to the allow-list
     (if surprised, the allow-list itself may have been edited
     without operator intent — investigate `audit.log` for
     `runtime_settings_change` rows touching the sender list).
  2. If intentional, the alert will dedup for 14 days.

### `operator_anomaly` hour_shift / length_shift

  1. If you've changed your schedule (travel, new job, voice
     mode flip), the alert is informational.
  2. If you have NOT changed your pattern but the system thinks
     so, treat as a potential impersonation signal: rotate Signal
     keys (and re-do Signal registration), check for unauthorized
     access to your Signal device, audit recent `request_received`
     rows manually.

## Composing with the rest of the system

This monitor and playbook are deliberately additive:

  * Composes with `disk_quota` (threshold) and `backup_freshness`
    (DR tarball age) but doesn't replace them.
  * Composes with the Q6 resilience drills — substrate migration
    is the human-paced procedure the drills practice in
    miniature; `vendor_independence` is the new 5th drill.
  * Composes with the continuity ledger — when migration
    happens, the operator can emit an `identity_substrate_change`
    event manually so the annual reflection picks it up. (Adding
    this as an automated event is deferred to a future Q16 sub-
    item; for now, manual emission via the ledger API is the
    operator's option.)
  * Composes with `unauthorized_sender` — that's the hard
    boundary; `operator_anomaly` is the softer pattern-shift
    signal on authorized senders.

## Future work (deferred)

  * **Vacation mode**: master switch with operator-set duration,
    pre-staged allowlist, low-risk CRs auto-apply within the
    window. Most invasive piece of Theme 3 — touches the change-
    request lifecycle and has security implications. Shipping
    separately so the contract gets explicit review.
  * Automated emission of `substrate_migration` continuity-
    ledger event on first probe of a new host (heuristic:
    workspace mount UUID changes; OS hostname changes by class).
  * Dry-run migration as an actual Pattern-A drill in the Q6
    registry. Today the playbook is procedural; making it a
    drill requires a scratch-VM provisioner the gateway can
    invoke.
  * Live cascade-fitness evaluation in `vendor_independence` —
    the current drill checks structural mechanics only. A
    follow-on could run a small fixed eval set through the
    cascade with dominant providers blocked and measure quality
    degradation. Cost-bounded ($1/quarter), behind its own
    `drill_vendor_independence_live_enabled` switch.
