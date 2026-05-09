# Life Companion — proactive personal-life surface

> Original three-module package shipped 2026-05-09 (Wave 2 of the
> resilience-gap closure plan). Extended over Phases B–G in the
> 2026-05-09→10 sweep with six more proactive modules.
>
> See PROGRAM.md §24 for the original Wave 2 change-log and §28 for
> the extended sweep.

Distinct from `app/healing/` (system-health observability) and
`app/companion/` (per-workspace ideation). Lives at the user-life
abstraction layer.

---

## What this ships

```
app/life_companion/
├── __init__.py                # get_idle_jobs() — registers all jobs
├── _common.py                 # shared helpers (signal, audit, state)
│
├── email_monitor.py           # 10-min: top-3 urgent unread → Signal     [Wave 2]
├── daily_briefing.py          # 07:00 / 18:00 / Mon 09:00 digests        [Wave 2]
├── routine_detector.py        # nightly: weekday × hour clustering       [Wave 2]
│
├── calendar_prep.py           # 5-min: prep 30 min before each event     [Phase B #2]
├── calendar_horizon.py        # daily 08:00: 72 h scan + conflicts       [Phase G #1]
├── long_arc_follow_up.py      # daily: SelfState.active_commitments      [Phase B #4]
├── personalized_digest.py     # Friday 09:00: RSS/GH/arXiv/news          [Phase D #5]
├── topic_dormancy.py          # daily: "deep on X 6 mo ago — still?"     [Phase G #3]
└── seasonal_nudges.py         # daily: Finland-seasonal triggers          [Phase G #4]
```

All nine ride the existing idle scheduler via
`app.companion.loop.get_idle_jobs()` — no TIER_IMMUTABLE / TIER_GATED
files modified for the wiring.

The Phase B–G additions form a closed loop with the per-workspace
ideation in `app/companion/`: feedback flows into `feedback_router`,
`feedback_weights` (workspace), and `topic_weights` (topic) — see
`docs/COMPANION_FEEDBACK_LOOP.md`.

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
| `LIFE_COMPANION_ENABLED` | `true` | Top-level gate; honored by every module. |
| `LIFE_COMPANION_EMAIL_ENABLED` | `true` | Email monitor on/off. |
| `LIFE_COMPANION_BRIEFING_ENABLED` | `true` | Daily briefing on/off. |
| `LIFE_COMPANION_ROUTINES_ENABLED` | `true` | Routine detector on/off. |
| `LIFE_COMPANION_EMAIL_CHECK_MIN` | `10` | Email cadence (minutes). |
| `LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD` | `1.0` | Email-score threshold. |
| `LIFE_COMPANION_BRIEFING_MORNING` | `07:00` | Morning time. |
| `LIFE_COMPANION_BRIEFING_EVENING` | `18:00` | Evening time. |
| `LIFE_COMPANION_BRIEFING_WEEKLY_DOW` | `MON` | Weekly DOW. |
| `LIFE_COMPANION_BRIEFING_WEEKLY_TIME` | `09:00` | Weekly time. |
| `EMAIL_IMPORTANT_SENDERS` | (empty) | Email scorer allowlist (CSV). |
| `USER_EMAIL` | inferred | Operator email — direct-to-user vs cc-only detection. |
| `CALENDAR_HORIZON_ENABLED` | `true` | G1: 72 h scan + conflict alerts. |
| `CALENDAR_HORIZON_HOUR` | `8` | Local hour the daily scan fires. |
| `PERSONALIZED_DIGEST_ENABLED` | `true` | D5: weekly RSS/GH/arXiv digest. |
| `PERSONALIZED_DIGEST_WEEKDAY` | `4` (Fri) | Weekly send-day (0=Mon). |
| `PERSONALIZED_DIGEST_HOUR` | `9` | Weekly send-hour (local). |
| `GITHUB_FOLLOWING_USER` | (empty) | GitHub username for D5 events feed. |
| `ARXIV_FOLLOWING_AUTHORS` | (empty) | CSV of author names for D5. |
| `VENTURES` | `PLG,Archibal,KaiCart` | Google News query terms. |
| `TOPIC_DORMANCY_ENABLED` | `true` | G3: dormancy nudges. |
| `SEASONAL_NUDGES_ENABLED` | `true` | G4: Finland-seasonal triggers. |

All modules honour the global `idle_scheduler.is_enabled()` kill
switch — the dashboard toggle that pauses every background job
pauses these too. Module-level switches (`LIFE_COMPANION_*` and the
phase-specific ones) layer on top.

---

## Calendar prep — 30 min before each event (Phase B #2)

`app/life_companion/calendar_prep.py` — sends a Signal prep message
30 min before each calendar event.

### How it works

1. Cadence guard (5 min); event window [28, 32] min ahead absorbs
   cadence drift.
2. Lists upcoming events via `app.tools.gcal_tools._service()`
   (calls Calendar API directly — `_list_events` drops the
   `description` field, which we want for agenda).
3. For each event in the trigger window not already prepped (dedup
   over last 200 event IDs):
   - Title, start time (local), location, attendees.
   - Event description (capped 280 chars) as agenda.
   - Per-attendee enrichment (first 2 attendees only):
     `_recent_inbox_from(attendee)` — last-7d Gmail subjects via
     `gmail_tools._list_recent`.
     `_mem0_facts_about(attendee)` — top-2 facts via
     `mem0_manager.search_memory`.
4. Signal alert with `tag=calendar_prep:<event_id>`.

### State

`workspace/life_companion/calendar_prep.json` — `{prepped_event_ids,
last_run_at}`. `prepped_event_ids` capped at 200.

---

## Calendar horizon — 72 h scan + conflicts (Phase G #1)

`app/life_companion/calendar_horizon.py` — daily 08:00 scan over
the next 72 h; surfaces overlapping events and dense clusters.

### How it works

1. Hourly probe; once-per-day gate at `CALENDAR_HORIZON_HOUR`
   (default 08:00 local). Dedup by date.
2. Lists events in next 72 h.
3. Conflict detection: pairs of overlapping timed events. All-day
   events excluded (would overlap everything).
4. Density detection: 3+ consecutive events with <15 min buffer
   between.
5. Signal alert when conflicts OR clusters exist; quiet otherwise.

### State

`workspace/life_companion/calendar_horizon.json` — `{last_run_at,
last_sent_date}`.

### Why a separate module from `calendar_prep`

Different scope, different cadence, different action. `calendar_prep`
is tactical (30-min context for one event); `calendar_horizon` is
strategic (72-h shape, conflict awareness).

---

## Long-arc commitment follow-up (Phase B #4)

`app/life_companion/long_arc_follow_up.py` — daily walk over
`SelfState.active_commitments`; nudges based on age + deadline.

### Cadence

Per-commitment, by age from `created_at`:

* 0–7 days: silent (commitment is too fresh)
* 7+ days, no prior check-in: first nudge
* 8–30 days, last check-in ≥ 14 d: weekly cadence
* 31+ days, last check-in ≥ 30 d: monthly cadence
* Within 7 d of `deadline`: daily nudge (max 7 total)
* Past `deadline`: one final nudge then mute

### Operator commands

`/commitment list` — show all active commitments with id, status,
age, deadline.
`/commitment fulfilled <id>` — mark fulfilled (terminal; mutes nudges).
`/commitment broken <id>` — mark broken (terminal; mutes).
`/commitment deferred <id>` — defer + mute nudges.
`/commitment unmute <id>` — resume nudges.

### State

`workspace/life_companion/long_arc_follow_up.json` —
`{by_commitment: {<id>: {last_check_in_at, deadline_nudges_sent,
muted}}}`.

---

## Personalized weekly digest (Phase D #5)

`app/life_companion/personalized_digest.py` — Friday 09:00 local
Signal digest from external feeds.

### Sources

| Source | How |
|--------|-----|
| RSS feeds | Operator-curated list at `workspace/companion/personalized_feeds.json` (`{"rss": [...]}`). |
| GitHub user events | `https://api.github.com/users/<user>/events/public` (no auth needed). |
| arXiv by author | `http://export.arxiv.org/api/query` with `au:"<name>"` query. |
| Venture news | Google News RSS for each venture term. |

Stdlib-only feed parser (`app/utils/feed_parser.py`) — no
feedparser/lxml deps.

### Cadence

Hourly probe; once-per-ISO-week gate (operator can miss the Friday
9 AM window without losing the digest).

---

## Topic-dormancy nudge (Phase G #3)

`app/life_companion/topic_dormancy.py` — detects topics the operator
was deeply into months ago but hasn't touched in weeks.

### Algorithm

1. Source: `workspace/companion/interest_history.jsonl` — appended
   by `interest_model` on every pass (capped 20k via
   `jsonl_retention`).
2. Per topic over last 365 days:
   - `peak_score_old` — max score in [60, 365] day window.
   - `avg_score_recent` — mean over last 14 days.
3. Dormancy criterion: `peak_score_old > 1.0` AND
   `avg_score_recent < 0.3` AND ≥4 old observations.
4. Per-topic 30-day dedup. `/topic mute <name>` silences a topic
   permanently.

### Operator commands

`/topic mute <name>` — silence dormancy nudges for that topic.
`/topic unmute <name>` — resume nudges.

---

## Finland-seasonal nudges (Phase G #4)

`app/life_companion/seasonal_nudges.py` — once-per-year proactive
nudges aligned with the Helsinki annual cycle.

### Trigger calendar

| Trigger | Window | Why |
|---------|--------|-----|
| First-frost watch | Oct 15 → Nov 5 | Winter tyres, plant covers, outdoor stowage. |
| Kaamos onset | Nov 22 ± 3 | Light therapy, vitamin D, lighter calendar. |
| Winter solstice | Dec 21–22 | Symbolic turning point. |
| Polar-night ends | Jan 18 ± 2 | Energy returns; heavier work realistic again. |
| Vappu | Apr 30 → May 1 | Public holiday, network offline. |
| Juhannus warning | 10 days before Saturday Jun 20–26 | Two-week absence pattern starts. |

Location-gated to Finland via `app.spatial_context.get_location()`.
If operator travels (Bali in February), nudges skip silently.

### Per-(year, key) dedup

Each trigger fires at most once per year. State at
`workspace/life_companion/seasonal_nudges.json`.

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
