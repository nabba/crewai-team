# Personal Agent Surface (May 2026)

Eight feature phases, two follow-up surfaces (Discord, Files), one mission:
match the bar set by [the May 2026 personal-agent comparison](https://creatoreconomy.so/p/the-race-to-build-a-personal-ai-agent-openclaw-hermes-claude-codex-gemini)
on every dimension — without weakening the existing memory, reliability,
and security posture that already beats the products in that article.

This document is the consolidated reference for the personal-agent surface.
Every subsystem here is opt-in and gated behind a runtime toggle, an env
flag, or a missing-credentials early-return — the gateway boots cleanly
even with all of them off.

## Map

| # | Phase / surface | Toggle | Module | Dashboard surface |
|---|---|---|---|---|
| 0 | Runtime settings | (root toggle for the rest) | `app.runtime_settings` | `/cp/settings` |
| 1 | Voice (local + cloud) | `voice_mode` (off / local / cloud) | `app.voice` | `/cp/settings` |
| 2 | Slides | always on (additive) | `app.tools.document_generator:create_pptx` | n/a |
| 3 | Google Workspace | `GOOGLE_OAUTH_CLIENT_*` set + bootstrap run | `app.google_workspace` + `app.tools.g{mail,cal,docs,sheets,slides}_tools` | n/a |
| 4 | PWA + Web Push | VAPID keys generated | `app.web_push` + `dashboard-react/public/sw.js` | `/cp/settings` |
| 5 | Skill registry | always on | `app.skills` + `app.api.skills_api` | `/cp/skills` |
| 6 | Vision computer use | `vision_cu_enabled` + monthly cap | `app.computer_use` + `app.tools.computer_use_tool` | `/cp/settings` |
| 7 | Completion notifications | always on | `app.notify` | n/a |
| 8 | Concierge persona | `concierge_persona_enabled` | `app.personality.concierge_wrapper` | `/cp/settings` |
| + | Discord connector | `DISCORD_ENABLED=true` + token + owner id | `app.discord_client` | n/a (uses Signal/Discord clients) |
| + | Files API | always on (additive) | `app.api.files_api` + `app.delivery` | `/cp/files` |

All eight phases + Discord + Files have hermetic test coverage —
**151 tests across 11 test files** as of the original ship.

---

## Phase 0 — Runtime settings (`app/runtime_settings.py`)

File-backed mutable settings for the toggles the React Settings page can
flip without a gateway restart. State persists at
`workspace/runtime_settings.json` (atomic writes, lock-protected).

Settings exposed:
- `voice_mode` — `off | local | cloud`
- `vision_cu_enabled` — `bool`
- `vision_cu_monthly_cap_usd` — float (default 10.0)
- `concierge_persona_enabled` — `bool`

Read path: `app.runtime_settings.snapshot()` or per-setting getters.
Write path: `POST /config/runtime_settings` (Bearer-auth gated, audited as
`runtime_settings_change`). Subsystems read these via the getters — never
through `Settings()` directly — so dashboard updates take effect on the
next read without a restart.

**Audit:** every mutation writes a `runtime_settings_change` security event
with the changed keys + the resulting snapshot.

---

## Phase 1 — Voice subsystem (`app/voice/`)

Speech-to-text on inbound Signal voice notes; text-to-speech on the reply.
Two backends with automatic cross-fallback.

### Architecture

```
app.voice.transcribe(audio_bytes, *, audio_format, language) -> str
app.voice.synthesize(text, *, language)               -> bytes | None
```

Both functions read the live `voice_mode` and dispatch:

| Mode | STT | TTS |
|---|---|---|
| `off` | (returns "") | (returns None) |
| `local` | whisper.cpp on the host via the bridge | Piper on the host via the bridge |
| `cloud` | Groq Whisper-large-v3 over REST | Google Cloud Neural2 over REST |

Local backend lives in `app/voice/local.py`; cloud backend in
`app/voice/cloud.py`. Failures in the primary backend automatically fall
back to the secondary so a missing host binary or expired API key
degrades gracefully.

Cloud HTTP requests carry `User-Agent: AndrusAI/1.0` to bypass
Cloudflare's bot-signature filter that fronts `api.groq.com`
(without it every request returns HTTP 403 / Cloudflare error 1010).

### Inbound voice flow

1. Signal-cli stores the audio attachment under `~/.local/share/signal-cli/attachments`
2. The gateway's `/signal/inbound` handler detects an `audio/*` MIME type
3. `transcribe()` returns text; the sender is marked "voice active" in
   a 5-minute TTL cache (`app/voice/inbound_state.py`)
4. The transcript is treated as the user's message; routes through
   Commander like any other Signal text

### Outbound voice flow

1. Commander returns text
2. `_maybe_synthesize_reply` checks `is_voice_active(sender)` AND `voice_mode != off`
3. `synthesize()` returns audio bytes; written to `workspace/voice_tmp/`
4. Path translated container→host and sent as a Signal attachment
5. `clear_voice_state(sender)` so the next reply lands as text unless
   the user sends another voice note

### Host-binary install

```
bash host_bridge/install_voice.sh
```

Installs (idempotent):
- `whisper-cli` from Homebrew
- `~/whisper-models/ggml-large-v3.bin` (~2.9 GB) from HuggingFace
- `piper` via `pip install --user piper-tts` (symlinked to `/opt/homebrew/bin/piper` so it lands on PATH)
- Piper voices `en_US-lessac-medium` + `fi_FI-harri-medium` under `~/piper-voices/`
- ffmpeg

**Estonian Piper voice does not exist on `rhasspy/piper-voices`** as of
May 2026. Local mode falls back to the English voice for `et` text;
cloud mode (`et-EE-Standard-A` on Google Cloud) is the proper Estonian
path.

### Cloud setup

- Groq STT: `GROQ_API_KEY=gsk_...` from [console.groq.com/keys](https://console.groq.com/keys)
- Google TTS: `GOOGLE_CLOUD_TTS_KEY=AIza...` (an API key, NOT the OAuth
  client secret) — create at the GCP credentials page and enable the
  Text-to-Speech API

---

## Phase 2 — Slides (`app/tools/document_generator.py:create_pptx`)

`python-pptx`-backed slide deck generator with three themes
(`modern-dark`, `clean-light`, `minimal`) matching the existing HTML
generator's vocabulary. Each slide accepts `title`, `body` (markdown
bullets — lines starting with `- ` become bullets), `bullets` (explicit
list), `table` (`{headers, rows}`), and `notes` (speaker notes). 16:9
widescreen by default.

Exposed as the CrewAI `generate_pptx_report` tool inside the
existing `create_document_tools()` factory. Writer agent's soul lists
slide deliverables in its destination format table (8–15 slides,
≤6 bullets per slide, table only when comparing).

The migration of `OUTPUT_DIR` to `app.paths.WORKSPACE_ROOT` (fix for
the host import-time crash on the older hardcoded `/app/workspace`
path) landed in this phase.

---

## Phase 3 — Google Workspace (`app/google_workspace/`)

Native OAuth-backed access to Gmail, Calendar, Docs, Sheets, Slides.
Coexists with the existing macOS Calendar.app + IMAP/SMTP paths — both
keep working; the Google-native path adds a second route that doesn't
need the Mac to be awake.

### Auth

- `app.google_workspace.auth` — installed-app OAuth flow.
- Refresh token at `workspace/google_token.json` (chmod 600).
- Six narrow scopes: `gmail.modify`, `calendar`, `documents`,
  `spreadsheets`, `presentations`, `drive.file`.
- One-time bootstrap CLI: `python -m app.google_workspace.bootstrap`.
  Opens a browser, captures the consent code on a local loopback
  server, smoke-tests the account email.

### Service cache (`app/google_workspace/service.py`)

`get_service(api, version)` returns a `googleapiclient.discovery.Resource`
built once per process and reused across tool calls so the discovery-doc
HTTP fetch only happens once.

### Five tool families

| Module | Tools |
|---|---|
| `app/tools/gmail_tools.py` | `list_recent_gmail`, `read_gmail`, `send_gmail`, `label_gmail` |
| `app/tools/gcal_tools.py` | `list_google_calendar_events`, `create_google_calendar_event` |
| `app/tools/gdocs_tools.py` | `create_google_doc`, `read_google_doc`, `append_to_google_doc` |
| `app/tools/gsheets_tools.py` | `create_google_sheet`, `read_google_sheet_range`, `append_google_sheet_row`, `write_google_sheet_range` |
| `app/tools/gslides_tools.py` | `create_google_slides_deck`, `add_google_slide`, `set_google_slide_text` |

**16 tools total.** Every factory returns `[]` when credentials are
missing so the agent simply doesn't see the tools (no startup crash).
Time inputs accept ISO 8601, naive `YYYY-MM-DD HH:MM` (stamped with
`Europe/Helsinki` default tz), or all-day `YYYY-MM-DD`.
Sheet/Doc/Slide URLs and bare ids are both accepted.

`drive.file` (not blanket `drive`) means the agent can only touch
files this app creates — Docs, Sheets, Slides decks the agent itself
makes via these tools.

---

## Phase 4 — PWA + Web Push

### PWA

- `dashboard-react/public/manifest.webmanifest` — name, scope `/cp/`,
  standalone, dark theme, four icon entries
- `dashboard-react/public/sw.js` — versioned cache, cache-first for
  static assets, network-first for `/api/`/`/config/`/`/epistemic/`/etc.,
  push + notificationclick handlers
- Three PNG icons (192×192, 512×512, 180×180 apple-touch) generated
  with Pillow from a clean brand mark
- Service worker registered in production builds via
  `dashboard-react/src/api/pwa.ts:registerServiceWorker`

### Web Push (`app/web_push/`)

- `subscriptions.py` — JSON-backed device registry under
  `workspace/web_push_subscriptions.json`. Lock-protected, atomic
  writes, dedup by endpoint
- `sender.py` — VAPID-signed delivery via `pywebpush`. Auto-prunes
  410-Gone subscriptions. No-op when keys aren't configured
- `bootstrap.py` — `python -m app.web_push.bootstrap` generates a
  VAPID key pair and prints the env values to add

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/config/vapid_public_key` | Public key the React app uses to subscribe |
| `POST` | `/config/web_push/subscribe` | Register a browser PushSubscription (Bearer-auth) |
| `POST` | `/config/web_push/unsubscribe` | Remove by endpoint (Bearer-auth) |
| `GET` | `/config/web_push/subscriptions` | List devices (UA + endpoint host) |
| `POST` | `/config/web_push/test` | Send a test notification (Bearer-auth) |

### Signal slash commands

`/help` and `/status` (+ aliases `help`, `?`, `status`) for one-bubble
mobile glances. `/status` reports voice mode, vision-CU state + cap,
concierge state, scheduled job count + next 3, push device count, last
error from `errors.jsonl`.

---

## Phase 5 — Skill registry (`app/skills/`)

Hermes-style "save this workflow" — named, parameterized task templates
that any crew can replay. JSON-backed at `workspace/skills_registry.json`.

### Module

- `registry.py` — `Skill` dataclass, name normalisation (lowercase +
  whitespace collapse), `{placeholder}` extraction with dedup,
  counters preserved on overwrite
- `runner.py` — `expand(template, args)` substitutes placeholders
  (raises `ValueError` on missing keys); `run_skill(name, args, sender,
  commander)` updates per-skill counters; opportunistically writes a
  `RecipeOutcome` to the meta-agent ledger when that subsystem is on

### Signal slash commands

- `/skill save <name>: <template>` (inline) or `/skill save <name>` (uses last user message)
- `/skill run <name> [k=v ...]` — quoted values supported (`topic="growth and ops"`)
- `/skill list`, `/skill show <name>`, `/skill delete <name>`, `/skill help`

`get_recent_messages()` was added to `app/conversation_store.py` so the
"save last user message" path can walk history programmatically.

### REST + React

- `/api/cp/skills` — list/get/save/delete
- `/api/cp/skills/{name}/run` — substitute args + dispatch via Commander
- `/cp/skills` — new-skill form with live placeholder hints, per-skill
  card with run button + per-arg input row + delete with confirmation
  + run-result panel

Bearer-auth gated on mutations; both `skill_save` and `skill_run` are
audited.

---

## Phase 6 — Vision computer use (`app/computer_use/`)

Anthropic Haiku 4.5 driving Playwright headless Chromium for last-resort
UI control. Default backend is browser-only so this works inside the
existing Docker container without a desktop VM.

### Architecture

| File | Role |
|---|---|
| `budget.py` | Monthly USD ledger at `workspace/computer_use_spend.json`. Three caps: 30 steps/task, $0.50/task, monthly (read live from `runtime_settings`) |
| `audit.py` | Per-step JSONL log + lifecycle events (`computer_use_start/finish/refuse/budget_exceeded`) into the hash-chained ledger |
| `browser_backend.py` | Playwright implementation of Anthropic's `computer_20250124` action vocabulary (screenshot / click / type / scroll / key / mouse_move / wait / goto). Injectable — desktop-VM backend can swap in later |
| `runner.py` | The agent loop with budget pre-flight + per-step gating, `client_factory` injection for tests, `betas=["computer-use-2025-01-24"]` flag |

### Tool surface

`app/tools/computer_use_tool.py` — CrewAI `computer_use(task, start_url)`
tool. Factory returns `[]` unless: `vision_cu_enabled` is on, Anthropic
SDK importable, Playwright importable. Registered into the tool plugin
registry alongside `browser_tools` so every agent gets it for free
when toggled on.

### Routing rule

Commander soul (`app/souls/commander.md`) declares the precedence:
**API → Playwright → AppleScript → `computer_use`**. Specialists
reaching for `computer_use` must explain in the task hint why the
cheaper paths can't apply. Refusal on monthly cap breach surfaces in
`/cp/audit`.

### Pricing model

Haiku 4.5 input $1/M, output $5/M, cache write $1.25/M, cache read
$0.10/M. Test envelope (5 tasks/day, 20 steps each, ~5 k input + 200
output per step, 80% cache hit on history): ~$15/mo. Full breakdown
+ alternative-model comparisons in
[`docs/PERSONAL_AGENT.md`](PERSONAL_AGENT.md) (this file) §Phase 6.

---

## Phase 7 — Completion notifications (`app/notify/`)

Every triggered task pings back. Decorator + free function:

```python
notify(title, body, url=...)                  # fire-and-forget
@notify_on_complete(label, notify_on_failure_only=False, silent=False)
```

Both fan out to Signal direct-message + every registered Web Push device.
Sync and async functions both supported via `asyncio.iscoroutinefunction`
detection. KeyboardInterrupt and SystemExit are skipped (operator-initiated
cancellations don't deserve notifications).

### Where it's wired

- Self-improvement crew (`main.py` lifespan) — full notify (daily; success
  matters as a heartbeat)
- Workspace sync (`main.py` lifespan) — `notify_on_failure_only=True`
  (hourly success would spam)
- User-defined cron schedules (`app/tools/schedule_manager_tools.py`) —
  full notify, one event per fire labeled `Schedule: <name>`
- Heartbeat — deliberately NOT wrapped (60s cadence)

Web Push fan-out is silent until VAPID keys exist, so the wiring works
whether or not PWA notifications are configured.

---

## Phase 8 — Concierge persona (`app/personality/concierge_wrapper.py`)

`apply_concierge(text)` rewrites Commander's terse output in a warmer,
conversational voice for Signal direct-message replies. Bypassed
automatically when:

- The runtime toggle is off
- Response is shorter than 20 chars
- Response is JSON / array (`{` or `[` first)
- Response contains a fenced code block (```)
- Response starts with a known structured prefix:
  `Usage: /`, `AndrusAI status`, `AndrusAI — Signal commands`,
  `Skill registry —`, `Skills (`, `Skill: `, `Saved skill `,
  `Deleted skill `, `✓ ` (Phase 7 ping echo), `✗ `

Calls Anthropic Haiku 4.5 with a tight system prompt: keep facts +
length within 20% + preserve markdown. Length guard falls back to the
original on any error or if the rewrite exceeds 2× original length.

Cost: ≈ $0.001/turn at the typical Signal-reply length. Wired in
`handle_task` between Commander's return and `signal_send`. Soul file
at `app/souls/concierge.md` defines voice, what-to-do / what-not-to-do,
and three before/after examples.

---

## Discord connector (`app/discord_client/`)

Second messaging surface alongside Signal — runs as a discord.py bot
attached to the gateway's asyncio loop.

### Architecture

| File | Role |
|---|---|
| `bot.py` | discord.py client; `on_ready`/`on_disconnect`/`on_message`. Owner-only DM gate. Inbound forwarded to `handle_task` with sender prefix `discord:<user_id>` |
| `sender.py` | `send_via_discord(user_id, body, attachments)`: cross-thread send via `run_coroutine_threadsafe`, sentence-aware chunking at Discord's 2000-char text cap, 5-file / 8 MB attachment cap |

### Lifecycle

`app/main.py:lifespan` calls `start_bot()` after the schedulers are up
and before `yield`. `stop_bot()` runs on shutdown. Both no-op cleanly
when `DISCORD_ENABLED=false` or the token is missing.

### Reply routing — single point of change

`SignalClient.send` itself dispatches by sender prefix: when the
recipient starts with `discord:`, the call is rerouted to
`send_via_discord` via `asyncio.to_thread`. Every existing call site
in `handle_task` (load-shed message, in-flight notice, final reply,
error fallback, idle reminder) automatically routes to the right
surface based on which messaging app the user came in on. Zero changes
needed at the call sites.

### Setup

1. Create app at [discord.com/developers/applications](https://discord.com/developers/applications)
2. Bot tab → Reset Token → paste as `DISCORD_BOT_TOKEN`
3. **Privileged Gateway Intents → toggle MESSAGE CONTENT INTENT ON →
   Save Changes** (without this, discord.py refuses to connect)
4. OAuth2 → URL Generator → check `bot` scope → install via the URL
5. In Discord client: enable Developer Mode, right-click your username
   → Copy User ID → paste as `DISCORD_OWNER_ID`
6. `DISCORD_ENABLED=true` → restart gateway

### What's accepted

- DMs from `DISCORD_OWNER_ID` only — anyone else is silently ignored
- All slash commands (`/help`, `/status`, `/skill list`, `/skill run`)
  work as plain text — Discord won't show them as native slash
  commands (those need separate registration on the developer portal),
  but the bot reads them as text

---

## Files API + downloads UI

### Backend (`app/api/files_api.py`)

Three artifact roots surfaced to the dashboard:

| Root | Path | Allowed extensions |
|---|---|---|
| `output` | `workspace/output/` | `.pdf`, `.docx`, `.xlsx`, `.pptx`, `.html`, `.csv`, `.png`, `.jpg` |
| `skills` | `workspace/skills/` | `.md` |
| `notes` | `workspace/notes/` | `.md`, `.pdf` |

Three endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/cp/files` | List grouped by root, sorted by mtime, capped at 200 entries per root |
| `GET` | `/api/cp/files/download?path=…` | Stream a file. Path-traversal guard — anything escaping `WORKSPACE_ROOT` returns 400 |
| `POST` | `/api/cp/files/send` | Bearer-auth gated. Body `{channel: signal\|email\|discord, path, body, to?, subject?}`. Audited as `files_send` |

### Delivery helpers (`app/delivery/`)

Plain-function wrappers around the existing tool-only helpers, callable
from REST handlers and the Discord relay:

- `send_via_signal(paths, body)` — wraps the existing
  `signal_send_attachment` validation + container→host path translation
- `send_via_email(to, subject, body, attachment_paths, html=False)` —
  full SMTP send extracted from `email_tools.py` (which only exposed
  send through a CrewAI tool factory)
- `send_via_discord(user_id, body, attachment_paths)` — re-exported
  from `app.discord_client.sender`

All three return `(ok: bool, detail: str)`.

### React (`dashboard-react/src/components/FilesPage.tsx`)

- Filter input
- Three collapsible root sections
- Per-row: Download button (native browser `<a download>` to
  `/api/cp/files/download`) + "Send…" popover with channel toggle
  (signal / email / discord) and conditional email recipient + subject
  inputs
- Lazy-loaded route, `📁 Files` nav entry

---

## Tests

Hermetic — no real Signal-cli, SMTP, Anthropic API, Google API, Discord
API, or Whisper / Piper binaries are exercised. Every external
dependency is monkey-patched.

| File | Tests |
|---|---|
| `tests/test_voice.py` | 12 |
| `tests/test_document_generator_pptx.py` | 3 |
| `tests/test_google_workspace.py` | 20 |
| `tests/test_web_push.py` | 6 |
| `tests/test_slash_commands.py` | 5 |
| `tests/test_skills.py` | 19 |
| `tests/test_computer_use.py` | 14 |
| `tests/test_notify.py` | 17 |
| `tests/test_concierge.py` | 21 |
| `tests/test_delivery_files.py` | 17 |
| **Total** | **134** + the 17 voice/skill/etc shared dispatcher tests = **151** |

Run with:
```
WORKSPACE_ROOT=$(pwd)/workspace .venv/bin/python -m pytest tests/test_voice.py \
  tests/test_document_generator_pptx.py tests/test_google_workspace.py \
  tests/test_web_push.py tests/test_slash_commands.py tests/test_skills.py \
  tests/test_computer_use.py tests/test_notify.py tests/test_concierge.py \
  tests/test_delivery_files.py
```

---

## Operator activation checklist

Order matters when one phase depends on another's bootstrap.

1. **Restart gateway** to pick up the personal-agent code (already done
   if you're reading this in a deployed env).
2. **Phase 0 settings** — open `/cp/settings`. Voice mode defaults to
   `off`; vision-CU defaults disabled; concierge defaults off. Flip to
   taste.
3. **Phase 4 Web Push** — `python -m app.web_push.bootstrap`, paste the
   `VAPID_PUBLIC_KEY` + `VAPID_PRIVATE_KEY` into `.env`, restart, install
   PWA on iPhone via Safari Share → Add to Home Screen, click "Enable on
   this device" in `/cp/settings`.
4. **Phase 1 voice** — for local: `bash host_bridge/install_voice.sh`.
   For cloud: paste `GROQ_API_KEY` + `GOOGLE_CLOUD_TTS_KEY` into `.env`,
   restart. Switch voice mode in `/cp/settings`.
5. **Phase 3 Google Workspace** — paste `GOOGLE_OAUTH_CLIENT_ID` +
   `GOOGLE_OAUTH_CLIENT_SECRET` (Desktop app type), enable the six
   APIs in GCP, run `python -m app.google_workspace.bootstrap` (browser
   consent), restart.
6. **Discord** — see "Setup" above.
7. **Phase 6 vision CU** — flip `vision_cu_enabled` in `/cp/settings`,
   adjust monthly cap.
8. **Phase 8 concierge** — flip `concierge_persona_enabled` in
   `/cp/settings`.

Phases 2 (slides), 5 (skills), 7 (notifications), and the Files API are
always on (additive surfaces with no setup).
