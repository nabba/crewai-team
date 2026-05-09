# Life Companion — proactive personal-life surface

> Shipped 2026-05-09 (Wave 2 of the resilience-gap closure plan).
> Three proactive features that watch the operator's life rather
> than the system's health.
>
> See PROGRAM.md §24 for the chronological change-log.

Distinct from `app/healing/` (system-health observability) and
`app/companion/` (per-workspace ideation). Lives at the user-life
abstraction layer.

---

## What this ships

```
app/life_companion/
├── __init__.py            # get_idle_jobs() — registers the three jobs
├── _common.py             # shared helpers (signal alerts, audit, state)
├── email_monitor.py       # 10-min cadence: top-3 urgent unread → Signal
├── daily_briefing.py      # 07:00 / 18:00 / Mon 09:00 cron-style digests
└── routine_detector.py    # nightly: weekday × hour-bucket clustering
```

All three ride the existing idle scheduler via
`app.companion.loop.get_idle_jobs()` — no TIER_IMMUTABLE / TIER_GATED
files modified for the wiring.

---

## Email monitor

`app/life_companion/email_monitor.py` — proactive triage of unread
inbox.

### How it works

1. Cadence guard (10 min default; tunable via
   `LIFE_COMPANION_EMAIL_CHECK_MIN`).
2. Fetches up to 25 unread inbox messages via
   `app.tools.gmail_tools._list_recent(query="in:inbox is:unread")`.
3. Builds `EmailHeaders` per stub; calls
   `app.tools.email_importance.score_email`.
4. Sorts by score; takes top-3 above
   `LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD` (default 1.0).
5. Dedup against `alerted_ids` (FIFO 500-cap) — repeat alerts are
   impossible.
6. Single Signal message with the three bullets:

   ```
   📬 Email triage — 3 urgent unread:

     • Boss <boss@example.com>
       URGENT: server down — please respond
       score=8.2

     • …
   ```

### State

`workspace/life_companion/email_monitor.json` — `{alerted_ids,
last_run_at, last_top}`.

### Scoring decoupled from this module

The `score_email` heuristic lives in `app/tools/email_importance.py`
and is shared with the on-demand "rank my emails" agent tool.
Signals: bulk markers (List-Unsubscribe / List-ID / auto-submitted /
precedence / noreply-style sender / marketing keywords), personal
markers (direct To: / threading / human From / action keywords),
allowlist hits, unread state, recency. Output is explainable —
`EmailScore.reasons` lists every signal that fired.

### Tests

`tests/life_companion/test_email_monitor.py` — alerts on first pass,
dedups same id, respects master + kill switches, cadence guard
short-circuits.

---

## Daily briefing

`app/life_companion/daily_briefing.py` — three flavours of synthesised
digest at fixed local-clock times.

### Flavours

| Flavour | Default time | Idempotency key |
|---|---|---|
| `morning` | `07:00` (env `LIFE_COMPANION_BRIEFING_MORNING`) | per-day |
| `evening` | `18:00` (env `LIFE_COMPANION_BRIEFING_EVENING`) | per-day |
| `weekly` | Mon `09:00` (env `LIFE_COMPANION_BRIEFING_WEEKLY_DOW` / `_TIME`) | per-ISO-week |

The cadence guard fires within ±15 min of the configured time.
Once-per-window idempotency: each flavour stores its
`last_<flavour>_at` key and refuses to send twice.

### Content per flavour

* **Morning** — 📅 next-24h calendar events from `gcal_tools` +
  📬 top-3 urgent unread + 🎯 open project tickets.
* **Evening** — 📅 tomorrow's calendar + 📬 still-flagged unread +
  💡 companion-surfaced ideas.
* **Weekly** — 📅 next-24h + 🎯 open tickets (n=8) +
  💡 companion-surfaced last week.

Every collector fails soft — a missing Calendar token, an empty
inbox, or a down ticket DB just gets a `(none)` line in the digest.

### State

`workspace/life_companion/daily_briefing.json` — `{last_morning_at,
last_evening_at, last_weekly_at}`.

### Tests

`tests/life_companion/test_daily_briefing.py` — morning window
fires, idempotent within window, skips outside windows, weekly
takes priority on its DOW.

---

## Routine detector

`app/life_companion/routine_detector.py` — surface day-of-week +
time-of-day patterns.

### How it works

1. Reads `workspace/affect/episode_affect_tags.jsonl` (the affect
   trace — every completed task is one line with `ts`, `crew`,
   `agent_id`, `task_preview`, `terminal_affect`).
2. Lookback window: 8 weeks (long enough for ≥8 samples per weekly
   routine without dragging in stale behaviour).
3. Clusters episodes by `(weekday, hour-bucket, crew)`. Hour buckets
   are 4-hour windows: night / early-morning / morning / afternoon /
   evening / late-night.
4. A cluster is flagged as a **routine** when:
   * `≥4 occurrences` in the window, AND
   * `≥60 % concentration` (of all this crew's episodes in any
     bucket, this fraction landed on this weekday + bucket), AND
   * `≥2 distinct ISO weeks` represented (one-off flurries don't
     count).
5. Picks the most common `task_preview` prefix as the routine label.

### Two surfacing paths

**Detection alert** — when a NEW routine first appears (`run_id`
not in prior state):

```
🔁 Detected 1 new routine(s):

  • Fri 16:00–20:00: PR review (crew=coder, n=6, weeks=6, concentration=85%)
```

**Nudge** — when a routine's window is approaching today
(within 30 min, deduped per-routine-per-day):

```
⏰ Upcoming routine(s) in the next 30 min:

  • 16:00–20:00: PR review (crew=coder, normally 6× over 6 weeks)
```

### State

`workspace/life_companion/routines.json` — `{last_detect_at,
routines: {id → routine_dict}, last_reminder_at: {id → date}}`.

### Tests

`tests/life_companion/test_routine_detector.py` — finds a 6-week
Friday-afternoon pattern, doesn't over-fire on random scattered
data, dedups against existing routines.

---

## Wiring

`app/companion/loop.py:get_idle_jobs()` already returned the
4 per-workspace ideation jobs (`companion-tick`, `companion-ingest`,
`companion-grand-task`, `companion-xworkspace`). It now also pulls in
the 3 life-companion jobs:

```python
from app.life_companion import get_idle_jobs as _lc_get_idle_jobs
jobs.extend(_lc_get_idle_jobs())
```

Wrapped in try/except so a broken life-companion module never breaks
the rest of the companion pipeline.

The idle scheduler runs each as a LIGHT job (60 s wall-clock cap).
Internal cadence guards in each module ensure heavy work happens at
the right cadence (10 min for email, scheduled-window for briefing,
~daily for routine detection) regardless of how chatty the idle
scheduler is.

---

## Master switches

| Variable | Default | Purpose |
|---|---|---|
| `LIFE_COMPANION_ENABLED` | `true` | Master gate. Set `false` to disable all three features. |
| `LIFE_COMPANION_EMAIL_ENABLED` | `true` | Email monitor on/off. |
| `LIFE_COMPANION_BRIEFING_ENABLED` | `true` | Daily briefing on/off. |
| `LIFE_COMPANION_ROUTINES_ENABLED` | `true` | Routine detector on/off. |
| `LIFE_COMPANION_EMAIL_CHECK_MIN` | `10` | Email cadence (minutes). |
| `LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD` | `1.0` | Score threshold. |
| `LIFE_COMPANION_BRIEFING_MORNING` | `07:00` | Morning time. |
| `LIFE_COMPANION_BRIEFING_EVENING` | `18:00` | Evening time. |
| `LIFE_COMPANION_BRIEFING_WEEKLY_DOW` | `MON` | Weekly DOW. |
| `LIFE_COMPANION_BRIEFING_WEEKLY_TIME` | `09:00` | Weekly time. |
| `EMAIL_IMPORTANT_SENDERS` | (empty) | Comma-separated allowlist for email scorer. |
| `USER_EMAIL` | inferred | The operator's email address — used by the scorer to detect direct-to-user vs cc-only. |

All three components honour the global
`idle_scheduler.is_enabled()` kill switch — the dashboard toggle
that pauses every background job pauses these too.

---

## Why "life companion" rather than "companion"

The existing `app/companion/` package is per-workspace ideation
(brainstorming, document maturation, cross-workspace transfer). Its
abstraction is "what should we build next for project X." The new
package's abstraction is "what does Andrus need today" — operator's
inbox, calendar, routines. Different concern. Same idle-scheduler
plumbing.

---

## Tests

12 tests in `tests/life_companion/` — all passing. Combined with the
other three test directories (healing, governance_amendment,
governance_ratchet), the resilience surface has 148 regression tests.
