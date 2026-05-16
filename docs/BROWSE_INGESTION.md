# Browser-history ingestion (PROGRAM §50) — operator guide

A 6th interest-signal modality joining email, calendar, conversations,
tickets, health, and inbox. Reads Safari + Chromium-family + Firefox
history at idle, drops blocklisted domains, clusters titles into
themes via a daily LLM batch, and feeds the resulting topics into the
companion's interest model and daily briefing.

Sister docs:
[HEALTH_INGESTION.md](HEALTH_INGESTION.md),
[INBOX_INGESTION.md](INBOX_INGESTION.md),
[PERSON_CORRELATION.md](PERSON_CORRELATION.md).

## What this is, briefly

1. A host-native daemon reads each browser's history SQLite file
   read-only with `mode=ro&immutable=1` URI flags (so the WAL doesn't
   block when the browser is running).
2. URLs are canonicalised to `scheme://domain/path` — query strings
   and fragments are stripped at the type boundary.
3. A per-domain blocklist (banking, health portals, auth endpoints,
   operator-added) drops matching rows before they hit disk.
4. Canonical events are appended to per-day JSONL at
   `workspace/browse/events/<YYYY-MM-DD>.jsonl`.
5. Once a day the gateway clusters the day's titles into topical
   labels via Anthropic Haiku 4.5; output lands at
   `workspace/browse/topics/<YYYY-MM-DD>.json`.
6. The companion's interest model picks up `browse` as a 6th source.
   Daily briefing surfaces a `🌐 Browsing themes (7d)` section.

Private/incognito browsing is **automatically excluded** because
browsers don't persist it to disk.

## The two-process split

The gateway runs in Docker. Docker on macOS can't see `~/Library/`
unless every browser directory is bind-mounted (brittle) **and**
Docker Desktop itself has Full Disk Access (must be re-granted on
every Docker upgrade). So the subsystem splits by file boundary:

| Process | Responsibility | Lives in |
|---|---|---|
| **Host collector** (launchd LaunchAgent on macOS) | Reads `~/Library/...` SQLite, canonicalises, applies blocklist, writes events JSONL | `app/browse/host_collector.py` |
| **Gateway** (Docker) | Runs the daily LLM topic batch; serves `/api/cp/browse/*`; handles `/browse` Signal command; surfaces React card + briefing section | `app/browse/topic_extraction.py`, `app/control_plane/browse_api.py`, etc. |

Both sides share the same `workspace/browse/` directory via the
gateway's bind-mount, so events flow end-to-end with no extra glue.

## Enabling it

Three steps.

### 1. Flip the env switches

In `crewai-team/.env`:

```
BROWSE_INGESTION_ENABLED=true
BROWSE_LLM_TOPICS_ENABLED=true   # optional, default ON when first is on
```

`BROWSE_INGESTION_ENABLED` is the master switch. When off, every
write path short-circuits before touching disk.

`BROWSE_LLM_TOPICS_ENABLED` is the second-layer switch for the
"titles leave the host" decision. Default ON when the first switch
is on; flip just this one off to keep event collection running
without any text reaching Anthropic.

Restart the gateway:

```bash
docker compose up -d gateway
```

### 2. Install the host LaunchAgent

```bash
./scripts/install_browse_collector.sh install
```

That copies the plist to `~/Library/LaunchAgents/`, runs `launchctl
bootstrap`, and starts the agent at 30-min cadence (matching the
gateway's idle-tick cadence). The agent will start passing
immediately but produce zero events until step 3.

Other subcommands: `start` (kick one pass), `restart`, `stop`,
`uninstall`, `status`.

### 3. Grant Full Disk Access (macOS only)

The Python interpreter the LaunchAgent invokes needs FDA to read
`~/Library/Safari/History.db`. Chrome, Arc, Brave, and Edge's
history files are also under `~/Library/Application Support/` —
same FDA requirement.

1. **System Settings → Privacy & Security → Full Disk Access**.
2. Click **+**, authenticate, then navigate (⇧⌘G) to the Python
   path the install script printed when you ran `install`. On most
   macOS installs that's the homebrew Python:
   `/opt/homebrew/Cellar/python@3.13/3.13.13/Frameworks/Python.framework/Versions/3.13/bin/`.
3. Select `python3.13` → **Open**.
4. Toggle the new entry **on**.
5. Restart the agent so the running process inherits the grant:
   `./scripts/install_browse_collector.sh restart`.

Verify the next pass collects events:

```bash
tail -5 workspace/browse/.host_collector.log
```

Look for `events=N written=N` with N > 0, instead of
`open failed: unable to open database file`.

## What lands when

| Timeline (after step 3) | Behavior |
|---|---|
| Next 30-min pass | Events flow into `workspace/browse/events/<today>.jsonl` |
| First post-midnight pass | Yesterday's events get the LLM topic batch → `workspace/browse/topics/<yesterday>.json` |
| Next interest-model cycle (12h cadence) | `browse` becomes the 6th source in your top interests |
| Next morning Signal briefing | `🌐 Browsing themes (7d)` section appears (empty until the first LLM batch) |

## Privacy contract — structural guarantees

Pinned by tests in [tests/browse/](../tests/browse/) so the
guarantees survive refactors:

1. **Query strings + fragments stripped at canonicalisation.** No
   `?q=...` or `#section` ever reaches the JSONL. Pinned by
   [`test_canonicalize_strips_query_string`](../tests/browse/test_url_canon.py).
2. **Blocked domains never reach disk.** Filtered at the reader's
   edge before `append_events`. Pinned by reader-level tests across
   all three browser families.
3. **Disabled → zero disk activity.** Master switch off short-
   circuits both readers and topic extraction. Pinned by
   `test_disabled_short_circuit` in multiple modules.
4. **Raw URLs never reach Anthropic.** The LLM batch sends only
   `(title, count, domain)` triples — never paths. Pinned by
   [`test_raw_urls_never_in_llm_batch`](../tests/browse/test_topic_extraction.py).
5. **Blocklisted titles never reach Anthropic.** Their source events
   never made it to disk. Pinned by
   [`test_blocklisted_titles_never_in_llm_batch`](../tests/browse/test_topic_extraction.py).
6. **PII redaction before LLM.** Email-shaped and phone-shaped
   tokens in titles replaced with `<email>` / `<phone>` placeholders.
   Pinned by [`test_titles_with_pii_are_redacted_in_prompt`](../tests/browse/test_topic_extraction.py).
7. **Browse-policy events emitted to the continuity ledger.** Master
   switch flips, blocklist edits, and forget paths all emit
   `browse_ingestion_policy` events surfaced in annual reflection.

## Seeded blocklist

The default seeded entries cover three categories the operator
explicitly flagged:

| Category | Examples |
|---|---|
| Banking & payments | `paypal.com`, `stripe.com`, `revolut.com`, `wise.com`, `n26.com` |
| Nordic banking | `op.fi`, `nordea.fi`, `danskebank.fi`, `s-pankki.fi`, `aktia.fi`, `swedbank.ee`, `seb.ee`, `lhv.ee`, `coop.ee` |
| Finnish/Estonian health | `kanta.fi`, `omaolo.fi`, `terveysasema.fi`, `mehilainen.fi`, `terveystalo.com`, `pihlajalinna.fi`, `digilugu.ee` |
| Authentication endpoints | `accounts.google.com`, `login.microsoftonline.com`, `appleid.apple.com`, `auth0.com`, `okta.com` |
| US patient portals | `mychart.com` |

Full list: [`_SEEDED_BLOCKLIST`](../app/browse/blocklist.py). Pattern
match is suffix-with-dot-boundary, so `example.com` blocks
`example.com` and `foo.example.com` but NOT `notexample.com`.

Operator-added entries go in `workspace/browse/blocklist.txt` (one
domain per line, `#` comments allowed). The file is mtime-cached;
adding entries takes effect within seconds.

## Operator surfaces

### React `/cp/settings` → "Browser-history ingestion"

The [BrowseIngestionCard](../dashboard-react/src/components/BrowseIngestionCard.tsx)
shows enabled/disabled status, 7-day stats, top topic clusters, the
blocklist (seeded + operator), a mute-domain input, and a typed-
phrase **FORGET BROWSE HISTORY** confirmation for the nuclear option.

### Signal `/browse` slash command

```
/browse                          window stats
/browse categories               top topic clusters (7d)
/browse domains                  top domains (7d)
/browse mute <domain>            add to operator blocklist
/browse forget <domain>          clear history for one domain
/browse forget-day <YYYY-MM-DD>  clear one day's events
/browse forget-all               clear EVERYTHING (preserves blocklist)
```

### REST endpoints (`/api/cp/browse/*`)

```
GET  /api/cp/browse/state             enabled + 7-day stats
GET  /api/cp/browse/recent            last N days of events (limited)
GET  /api/cp/browse/categories        top topic clusters
GET  /api/cp/browse/blocklist         seeded + operator entries
POST /api/cp/browse/mute              add a domain to operator file
POST /api/cp/browse/forget            scope: all | domain | day
```

All require `Authorization: Bearer <GATEWAY_SECRET>`.

### Daily briefing

`🌐 Browsing themes (7d)` section appears in the morning and weekly
composers; auto-hides when the master switch is off or no topic
files exist yet.

### Identity continuity ledger

Master-switch flips, blocklist edits, and forget actions emit
`browse_ingestion_policy` events. Year-over-year drift surfaces
through the annual reflection's [`summarise_drift`](../app/identity/continuity_ledger.py)
Counter without code changes.

## Storage layout

```
workspace/browse/
├── events/
│   ├── 2026-05-16.jsonl        # one BrowseEvent per line, append-only
│   └── 2026-05-17.jsonl
├── topics/
│   └── 2026-05-16.json         # daily LLM cluster output
├── state.json                  # per-browser-profile cursor map
├── blocklist.txt               # operator-managed entries
├── .last_pass_at               # cadence marker (host collector)
├── .last_topics_at             # cadence marker (gateway batch)
└── .host_collector.log         # launchd stdout/stderr
```

Day-bucketed events are deliberate — they make the LLM-batch step
trivially "yesterday's events.jsonl" and let an operator forget a
specific day with a single unlink.

## Forget paths

Three scopes:

| Command | Removes | Preserves |
|---|---|---|
| `/browse forget <domain>` | All events for that domain across all days | Blocklist, cursors, other domains |
| `/browse forget-day <YYYY-MM-DD>` | All events from that day | Blocklist, cursors, other days |
| `/browse forget-all` | Every events file + cursor state | Blocklist, topics |

Each emits a `browse_ingestion_policy` event into the continuity
ledger so multi-year identity drift carries a record that you exercised
the forget path.

`mute_domain(d)` (Signal: `/browse mute <d>` / React: mute input)
adds to the blocklist but **does not clear past events** — the
typical pair is `mute` + `forget` for a domain.

## Cost ceiling

The daily LLM batch sends at most 250 deduplicated `(title, count,
domain)` rows to Anthropic Haiku 4.5 with a 1500-token cap. Worst-
case spend is well under $0.001/day.

## Disabling

Two paths:

| To stop... | Do |
|---|---|
| ...just the LLM batch (titles leaving the host) | `BROWSE_LLM_TOPICS_ENABLED=false` + restart gateway |
| ...event collection entirely | `BROWSE_INGESTION_ENABLED=false` + restart gateway + `./scripts/install_browse_collector.sh stop` |
| ...uninstall the LaunchAgent | `./scripts/install_browse_collector.sh uninstall` |
| ...also revoke FDA | System Settings → Privacy & Security → Full Disk Access → toggle the python3.13 entry off |

`forget-all` is independent of the master switch — you can disable
the subsystem and still clear past history (or vice versa).

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `open failed: unable to open database file` (only Safari) | Python lacks FDA | Grant FDA → restart agent |
| `OSError: [Errno 30] Read-only file system: '/app'` in the log | `WORKSPACE_ROOT` env var wasn't set before `python -m app.browse.host_collector` ran. `app.paths` reads it at module-import time, so the runtime fallback in `_setup_workspace` is too late under `-m`. | Set `WORKSPACE_ROOT=/Users/.../crewai-team/workspace` in the launchd plist's `EnvironmentVariables` (or export it before invoking from a shell). The defensive check in `host_collector.py` will refuse to proceed with a clear diagnosis from this point on. |
| Agent shows `last_exit_code` other than 0 | Python path stale (e.g. homebrew Python upgrade) | Re-edit `scripts/browse_host_collector.plist` with the new path → `install` |
| First-pass event count maxes at ~5000 then stops mid-history | Per-pass `LIMIT 5000` in each reader — intentional to bound a single pass. Cursor advances; the next 30-min pass picks up where it left off. | No action needed. Three to four passes are enough to back-fill a year of history on a busy account. |
| No `🌐 Browsing themes` in briefing | Master switch off, OR no topic files yet (LLM batch runs once per day) | Wait a day; check `workspace/browse/topics/` |
| All four readers produce 0 events | No browsers installed at expected paths; check `~/Library/Application Support/` | Most users only have Safari + Chrome — that's normal |
| `chrome` reader silently produces 0 events | Chrome's `Default/History` file doesn't exist (browser installed but unused) | Expected; the readers no-op when DBs are missing |

## Why `WORKSPACE_ROOT` must be set in the env, not just `--workspace`

`python -m app.browse.host_collector` triggers Python's package-import
machinery on `app.browse` BEFORE `host_collector.main()` runs.
`app/browse/__init__.py` imports `app.browse.idle_job → app.browse.store
→ app.paths`, and `app.paths` evaluates `WORKSPACE_ROOT` from the env
at module-import time:

```python
# app/paths.py
WORKSPACE_ROOT: Path = Path(
    os.environ.get("WORKSPACE_ROOT", "/app/workspace"),
)
```

So if `WORKSPACE_ROOT` isn't already set when Python starts, the
package binds `WORKSPACE_ROOT = /app/workspace` (the container path)
into `app.paths` permanently for the process lifetime, and the
runtime `os.environ["WORKSPACE_ROOT"] = ...` in `_setup_workspace`
can't undo that. Events then try to land under `/app/workspace/...`
on macOS — which is the read-only system root — and the host
collector crashes with `OSError: [Errno 30]`.

The launchd plist sets it correctly in `EnvironmentVariables`. The
defensive check in `_setup_workspace` cross-checks `app.paths.WORKSPACE_ROOT`
against the `--workspace` argument and refuses to proceed with a
specific diagnosis if they mismatch.

## What this is NOT

- **Not a productivity tracker.** Dwell time isn't recorded. The
  surface is "what themes are you reading about", not "how much
  time did you spend on X".
- **Not a browsing replayer.** Canonical URLs (domain + path) are
  stored; raw URLs with query strings are not. You can see "you
  visited en.wikipedia.org/wiki/Helsinki", not what search led
  there.
- **Not cross-device.** Each Mac runs its own collector. iCloud
  history sync isn't read.
- **Not an extension.** Real-time dwell-time tracking would require
  a per-browser extension; that's explicitly deferred (Phase E in
  the original plan) until and unless theme quality from titles
  proves too coarse.
