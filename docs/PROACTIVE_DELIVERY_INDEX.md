# 2026-05-09→10 Proactive-companion + self-evolution sweep

> Index for the eight-phase delivery (Phases A → H) over a 24-hour
> window. ~14 000 lines of Python + ~3 500 lines of tests + 394
> green cross-suite tests.
>
> Per-phase change-log lives in PROGRAM.md §27 (Phase A), §28
> (B–G), §29 (H). This file is the reverse map: by-feature →
> docs that explain it.

---

## What got shipped, by user-facing capability

### "Watch my life and surface what matters"

| Capability | Doc | Phase |
|---|---|---|
| 10-min email triage to Signal | LIFE_COMPANION.md → "Email monitor" | Wave 2 |
| 7am morning / 6pm evening / Mon weekly briefings | LIFE_COMPANION.md → "Daily briefing" | Wave 2 |
| Routine detection (DOW × hour clustering) | LIFE_COMPANION.md → "Routine detector" | Wave 2 |
| 30-min pre-meeting prep with email + Mem0 enrichment | LIFE_COMPANION.md → "Calendar prep" | B #2 |
| 72 h calendar horizon scan + conflicts + density warnings | LIFE_COMPANION.md → "Calendar horizon" | G #1 |
| Long-arc commitment follow-up (`SelfState.active_commitments`) | LIFE_COMPANION.md → "Long-arc commitment" | B #4, F #10 |
| Topic-dormancy nudges ("deep on X 6 mo ago — still blocked?") | LIFE_COMPANION.md → "Topic-dormancy" | G #3 |
| Friday 09:00 personalized digest (RSS + GH + arXiv-by-author + venture news) | LIFE_COMPANION.md → "Personalized digest" | D #5 |
| Finland-seasonal nudges (frost / kaamos / Vappu / Juhannus / etc.) | LIFE_COMPANION.md → "Seasonal nudges" | G #4 |

### "Learn from my reactions"

| Capability | Doc | Phase |
|---|---|---|
| 👍 / 👎 reactions drain into skill counter / recipe ledger / companion event log | COMPANION_FEEDBACK_LOOP.md → "B3" | B #3, F #3, F #8, E #1 |
| 👎 on workspace's idea downweights that workspace in next idle cycle | COMPANION_FEEDBACK_LOOP.md → "D4" | D #4 |
| 👎 on topic comment downweights that topic in next idle cycle | COMPANION_FEEDBACK_LOOP.md → "G2" | G #2 |
| Job-completion reactions logged for satisfaction trend | COMPANION_FEEDBACK_LOOP.md → "F3" | F #3 |
| Concierge persona consults mood + time + top interests | LIFE_COMPANION.md (cross-link) — `signal_context.py` | B #5, F #4 |

### "What does Andrus care about right now?"

| Capability | Doc | Phase |
|---|---|---|
| Interest profile from conversations + email + calendar + 👍 + affect tags | COMPANION_FEEDBACK_LOOP.md → context | B #1, E #4, F #1 |
| Interest profile timeseries (jsonl, capped 20k) | COMPANION_FEEDBACK_LOOP.md → "G3" | G #3 |
| Interest profile feeds top-3 into weekly briefing's "🧭 Topics" section | LIFE_COMPANION.md (Daily briefing) | F #6 |
| Interest profile drives paper_pipeline arXiv search terms | SELF_HEAL_V3.md → "Paper-to-experiment pipeline" | C #3, F #11 |

### "Watch the system and self-improve"

| Capability | Doc | Phase |
|---|---|---|
| DB backup engine (PG + Neo4j + Chroma) + freshness monitor + RESTORE.md | SELF_HEAL_V3.md → "DB backup engine" | A #A1 |
| conversations.db monthly VACUUM | SELF_HEAL_V3.md → "conversations.db monthly VACUUM" | A #A6 |
| Log archival (errors.jsonl + audit_journal + evolution-runs) | SELF_HEAL_V3.md → "Log archival" | A #A5, D #2 |
| Per-listener heartbeat with missing-listener alert | SELF_HEAL_V3.md → "Listener-heartbeat monitor" | A #A3, F #9 |
| PG startup circuit breaker | (PROGRAM.md §28.3) | D #1 |
| Vendor-sunset detector files CR (not just alert) | SELF_HEAL_V3.md → "Vendor sunset" | A #A4 |
| Boot-time stale-cooldown reset | SELF_HEAL_V3.md → boot_reset note | A #A7 |
| Silent-regression detector (cron throughput vs baseline) | SELF_HEAL_V3.md → "Silent-regression detector" | C #2 |
| Failure-pattern learner (proposes new runbooks for un-handled signatures) | SELF_HEAL_V3.md → "Failure-pattern learner" | C #4 |
| Semantic LLM-output drift (golden-probes weekly) | SELF_HEAL_V3.md → "Semantic LLM-output drift detector" | D #6 |
| Adapter retirement performance arm | (PROGRAM.md §28.2) | C #1, F #2 |
| Paper-to-experiment pipeline | (PROGRAM.md §28.2) | C #3, F #11 |
| Rejected-hypothesis lessons KB | COMPANION_FEEDBACK_LOOP.md (cross-link) | D #7, F #5 |
| Auto-propose ratchet UP based on rolling avg | GOVERNANCE_RATCHET.md → "Auto-proposers" | C #5 |
| Auto-propose Goodhart Advisory → Enforcing flip | GOVERNANCE_RATCHET.md → "Auto-proposers" | D #3 |
| Restore-drill freshness monitor + quarterly drill script | SELF_HEAL_V3.md → "Restore-drill freshness monitor" | H #1 |
| Signal 120-day re-registration keepalive | SELF_HEAL_V3.md → "Signal 120-day re-registration keepalive" | H #2 |
| Idle-scheduler half-open retry on cooldowns | (PROGRAM.md §29.3) | H #3 |
| Postgres-down on boot bounded retry for mem0 | (PROGRAM.md §29.4) | H #4 |

### "Operator commands"

| Command | What | Phase |
|---|---|---|
| `/help` | List all slash commands | (pre-existing) |
| `/status` | Live system summary | (pre-existing) |
| `/skill save\|list\|show\|run\|delete\|help` | Skill registry | (pre-existing) |
| `/brainstorm` | Brainstorm subsystem | (pre-existing) |
| `/commitment list\|fulfilled\|broken\|deferred\|unmute <id>` | Long-arc commitment status | F #10 |
| `/topic mute\|unmute <name>` | Topic-dormancy nudge silence | G #3 |

---

## Shared utilities introduced

| Module | What | Why extracted |
|---|---|---|
| `app/utils/feed_parser.py` | Stdlib RSS + Atom parser | E6 — `paper_pipeline` had regex parsing, `personalized_digest` had ET parsing — same job, two implementations. |
| `app/utils/hash_embedding.py` | Deterministic 256-d hashing-trick embedding + cosine | E7 — `lessons_learned` and `llm_output_drift` shipped identical 256-d implementations. |
| `app/utils/jsonl_retention.py` | `cap_jsonl(path, max_lines)` + `append_with_cap` | F7 — 5 unbounded JSONLs grew without retention. |

These are the only utilities general enough that two unrelated
subsystems would otherwise duplicate them. Resist the urge to add
more — keep `app/utils/` boring.

---

## Master switches

See PROGRAM.md §28.10 + §29.5 for the full table of 22 new env
switches. Defaults are ON for everything except
`HEALING_DB_BACKUP_ENABLED` (operator decides whether the gateway is
the backup runner).

The runtime-toggleable switches surface on the React `/cp/settings`
page; env-var-only switches require a gateway restart.

---

## Tests

* 394 cross-suite tests pass (healing + companion + concierge + fts5
  + change_requests).
* The Phase E / F / G / H targeted tests (`tests/healing/test_phase_*_
  followups.py`) pin the bug-fix shapes specifically — a future
  rename of a real-system symbol surfaces here BEFORE production
  silently breaks.

---

## Operator action remaining

1. **Goodhart hard gate.** Currently in Advisory mode. The Phase
   D #3 auto-proposer will surface a recommendation in
   `workspace/governance_proposals.jsonl` once conditions warrant.
   Wait at least 14 days of Advisory observation before flipping
   via React `/cp/settings`.
2. **Personalized digest curation.**
   `workspace/companion/personalized_feeds.json` is empty by
   default. Add RSS feeds, GitHub username, arXiv author list,
   ventures override. One-time setup.
3. **DB backup runner.** Decide whether the gateway runs backups
   (`HEALING_DB_BACKUP_ENABLED=1`) or you run them host-side via
   `deploy/scripts/backup.sh` from cron / launchd. Both write to
   `workspace/backups/` with the same manifest format.
4. **Restore drill scheduling.** Add to cron / launchd quarterly:
   ```
   @quarterly cd /path/to/crewai-team && bash deploy/scripts/restore-drill.sh
   ```
   The H1 monitor will Signal-alert at day 100 if no drill has run,
   so the system is self-reminding even if the cron entry is
   forgotten.

---

## Per-phase post-mortems

* PROGRAM.md §27 — Phase A (Wave 0/1 self-heal closure)
* PROGRAM.md §28.1 — Phase B (5 personal-life features)
* PROGRAM.md §28.2 — Phase C (5 self-improvement / observability)
* PROGRAM.md §28.3 — Phase D (7 audit-finding gaps)
* PROGRAM.md §28.4 — Phase E (Phase D audit cleanup; 14 fixes)
* PROGRAM.md §28.5 — Phase F (Phase A-E delivery audit; 11 fixes)
* PROGRAM.md §28.6 — Phase G (4 final companion gaps)
* PROGRAM.md §29 — Phase H (4 silent-failure modes from years-of-uptime audit)

---

## 8 silent-failure modes — final accounting

The original audit identified 8 silent-failure modes for years-of-
uptime risk. After Phase H, all 8 are closed:

| # | Item | Status | Where |
|---|------|--------|-------|
| 1 | Restore-from-backup untested | ✅ DONE | A1 backup engine + H1 quarterly drill |
| 2 | Google OAuth refresh persist | ✅ DONE | pre-existing (refuted in audit) |
| 3 | Signal 120-day re-registration | ✅ DONE | H2 keepalive |
| 4 | Firestore on_snapshot drops silently | ✅ DONE | A3 polling-fallback heartbeat + F9 missing-listener alert |
| 5 | Sticky 1h cooldowns | ✅ DONE | H3 half-open probes at 1/4, 1/2, 3/4 |
| 6 | Disk growth unbounded | ✅ DONE | A5/A6/A7 + D2 + F7 + Wave 2 retention monitors |
| 7 | Postgres-down on boot hangs gateway | ✅ DONE | D1 control_plane + H4 mem0_manager |
| 8 | Vendor model sunsets blind | ✅ DONE | A4 weekly /v1/models diff + CR filing |
