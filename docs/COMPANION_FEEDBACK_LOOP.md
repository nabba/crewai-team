# Companion feedback loop

> Built across Phases B → G of the 2026-05-09→10 sweep
> (PROGRAM.md §28). The closed-loop turns the operator's 👍 / 👎
> reactions and explicit slash commands into bias on the next
> selection cycle — workspaces, topics, and skills.

The loop is the single biggest piece of "system that learns from
its operator without retraining a model." It's also the piece that
shipped with the most silent-failure bugs and required the most
audit-cycles to make actually work — see PROGRAM.md §28.4 (Phase E)
and §28.5 (Phase F) for the bug post-mortems.

---

## Architecture

```
                                   ┌─────────────────────┐
 operator's Signal client          │ feedback_pipeline   │
    👍 reaction ────────►──────────┤  (TIER_IMMUTABLE)   │
                                   │  writes feedback.   │
                                   │  events PG table     │
                                   └─────────┬───────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────┐
                          │ feedback_router (10-min idle)│
                          │  LEFT JOIN feedback.events   │
                          │   ⨝ feedback.response_meta   │
                          │   ⨝ notify_meta sidechannel  │
                          └────┬──────┬──────┬──────┬────┘
                               │      │      │      │
                  ┌────────────┘      │      │      └────────────┐
                  ▼                   ▼      ▼                   ▼
          skill counter      recipe ledger   companion          job_feedback
       (skills.registry)   (meta_agent.recorder)  event log     (jsonl)
                                                  ↓
                                           workspace_id
                                                ↓
                            ┌──────────────────────────────────┐
                            │  feedback_weights (workspace-level)│  ◄── D4
                            │  topic_weights    (topic-level)    │  ◄── G2
                            └──────────────────────────────────┘
                                                ↓
                            scheduler.collect_candidates       ◄── biases pick
                            interest_model._score_terms        ◄── biases scoring
```

---

## Pieces, in delivery order

### B3 — `feedback_router` + `notify_meta` (the drain)

Without B3, the IMMUTABLE `feedback_pipeline` writes events to a
PG table that nobody reads. The router runs as a 10-min idle job
under `app.companion.loop.get_idle_jobs()`:

1. **Fetch.** `_fetch_new_events(since_id)` does a single LEFT JOIN
   between `feedback.events` and `feedback.response_metadata` on
   `response_text` (Phase F #8 — was N+1 SQL before; bytes-equal
   join on the 2000-char prefix the IMMUTABLE pipeline truncates
   to). Returns up to 200 rows in oldest→newest order so the cursor
   advances cleanly.

2. **Resolve.** `_resolve_send_ts(event)` reads the JOIN'd
   `msg_timestamp` field — the pre-Phase F implementation made a
   second SQL query per event.

3. **Lookup.** `notify_meta.lookup(send_ts)` finds the
   `(send_ts, metadata)` pair that the original `notify(...)`
   call recorded. The metadata schema is informal but
   convention-driven:
   ```
   {"skill_id":   "...",
    "recipe_id":  "...",
    "task_id":    "...",
    "idea_id":    "...",
    "workspace_id": "...",
    "job_id":     "..."}
   ```

4. **Dispatch.** `_dispatch(event, metadata)` fans the event out:
   * `skill_id`     → `skills.registry.record_run_result(name, success)`
   * `recipe_id`    → `meta_agent.recorder.record_outcome(...)`
   * `idea_id`      → `companion.feedback.record(idea, ws, polarity)`
   * `workspace_id` → `feedback_weights.record_(positive|negative)(ws)`
   * `comment text` → `topic_weights.record_negative_from_comment(text)`
   * `job_id`       → `workspace/companion/job_feedback.jsonl` append

Each sink is best-effort — a single sink failure logs at DEBUG
and leaves the others to run.

### D4 — `feedback_weights` (workspace-level downweight)

`app/companion/feedback_weights.py` keeps a per-workspace
multiplier in `[0.4, 1.0]`:

```
multiplier(ws) = max(MIN, base + (1 - base) × (1 - decay))
   where base  = 1 - 0.2 × down_count
         decay = 0.5 ** (age_days / 3)
```

Each 👎 subtracts 0.2 from base; the 3-day halflife means a single
👎 from 6 days ago is essentially decayed away. 👍 partially
counteracts (decrements down_count, doesn't replace it).

`scheduler.collect_candidates` multiplies the existing
`_affect_weight()` by the workspace multiplier — workspaces with
recent 👎 get less of the next cycle's budget.

### G2 — `topic_weights` (topic-level downweight)

`app/companion/topic_weights.py` is the same shape but bound to
**topics**, not workspaces. The trigger is different:

```
record_negative_from_comment(comment)
  → tokenize
  → match against current interest_profile topic names
  → record_negative(topic) for each match
```

So a 👎 on "I really dislike forest carbon" extracts "forest carbon"
(if it's currently in the interest profile) and downweights only
that topic — not the whole workspace.

`interest_model._score_terms` multiplies each term's accumulated
score by `current_multiplier(term)` before topic ranking. Halflife
7 days (longer than workspace; topic taste moves slower).

### F3 — `notify_on_complete` metadata

`app/notify/api.py:notify_on_complete` accepts a `metadata` dict;
when the wrapped fn finishes, the completion ping calls
`notify(... metadata=metadata)` which records `(send_ts, metadata)`
in `notify_meta`. So when the operator reacts to "✓ Schedule:
weekly_review done", the router can look up `{job_id:
"schedule:weekly_review"}` and route to the job-feedback sink.

Wired so far in `app/tools/schedule_manager_tools.py` (every user
schedule). Other notify users (self_improve, brainstorm completion,
etc.) can add `metadata={...}` over time without changing the
decorator.

### G3 — `interest_history.jsonl` timeseries

`interest_model.run()` appends each pass's top-30 to
`workspace/companion/interest_history.jsonl` (capped 20 000 via
`jsonl_retention`). `topic_dormancy` reads this to detect "deep on
X 6 months ago, silent now" patterns — a different feedback signal
than 👎: implicit dormancy rather than explicit dissatisfaction.

---

## Operator commands surfaced via Signal

* `/commitment list|fulfilled|broken|deferred|unmute <id>` —
  `SelfState.active_commitments` (Phase F #10).
* `/topic mute|unmute <name>` — silence dormancy nudges per topic
  (Phase G #3).

Both write through the same channels the proactive modules read,
so a `/commitment fulfilled X` immediately stops `long_arc_follow_up`
nudges; `/topic mute Y` immediately stops `topic_dormancy` alerts
for Y.

---

## Master switches

| Variable | Default | Effect |
|---|---|---|
| `FEEDBACK_ROUTER_ENABLED` | `true` | Stop the drain (events still write; nothing fans out). |
| `COMPANION_FEEDBACK_WEIGHTS_ENABLED` | `true` | `current_multiplier(ws)` returns 1.0 — scheduler bias off. |
| `COMPANION_TOPIC_WEIGHTS_ENABLED` | `true` | `current_multiplier(topic)` returns 1.0 — interest_model bias off. |
| `TOPIC_DORMANCY_ENABLED` | `true` | Stop dormancy nudges (history still writes). |

Disabling at any layer is safe: the upstream writers don't depend
on consumers, and the consumers each return identity (1.0) when
disabled.

---

## Why the loop took six commits to make actually work

1. **Phase B #3 shipped without entry callers** — `notify(metadata=)`
   had no upstream caller, so the sidechannel stayed empty forever.
   Fixed in Phase F #3.
2. **Phase B #3 SQL queried wrong table** — `feedback.responses` doesn't
   exist; actual is `feedback.response_metadata`. Fixed in Phase E #1.
3. **Phase B #3 SQL was N+1** — one query per event. Fixed in Phase F #8.
4. **Phase B #1 read wrong path** — `events.jsonl` doesn't exist (per-
   workspace at `events/<ws>.jsonl`). Fixed in Phase F #1.
5. **Phase D #4 only worked at workspace level** — per-topic filter
   was the missing piece. Added in Phase G #2.

These were all silent-failure bugs that passed unit tests (mocks
hid the real integration shape). The audits in Phase E + Phase F
flushed them out — see PROGRAM.md §28.4 + §28.5 for the post-mortems.

---

## Tests

* `tests/companion/test_feedback_router.py` — 7 tests, dispatch
  per sink + cursor advance + dedup + disabled.
* `tests/companion/test_feedback_weights.py` — 7 tests, decay +
  thumbs-up counteract + isolation.
* `tests/companion/test_notify_meta.py` — 6 tests, record + lookup
  + window + prune.
* `tests/healing/test_phase_e_followups.py` — 12 tests pinning
  the audit-fix shapes.
* `tests/healing/test_phase_f_followups.py` — 21 tests pinning the
  audit-fix shapes.
* `tests/healing/test_phase_g_followups.py` — 26 tests including
  the topic-weight × interest_model integration round-trip.
