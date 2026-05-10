# Identity-continuity layer (§§2.8, 8.2, 8.5)

The identity-continuity layer records and reflects on the system's
identity-shaping events. Three modules in `app/identity/`, all
**observational** relative to the consciousness stack — they read
narrative chapters / lessons KB / amendment audit logs and write to
`wiki/self/` artefacts the operator reads. They never modify
SCORECARD probes, `current_goals`, or any TIER_IMMUTABLE file.

The package lives at `app/identity/` (not under `app/subia/`) on
purpose: SubIA's integrity manifest would churn on each additive
hook, and the identity ledger is observational of (not part of) the
consciousness layer.

## Why this exists

The Tier-3 amendment protocol (§25.1) gates **intentional** edits to
TIER_IMMUTABLE files after demonstrated track record. That's good for
one-edit-at-a-time governance — but it doesn't surface **aggregate
drift** across many small approved amendments. The narrative-self
FIFO holds 5 identity claims; at year 2 the older you is gone. The
identity ledger is the multi-year record.

The annual reflection essay (§8.2) is the system's yearly
self-examination across that drift, the year's narrative chapters,
and the lessons-learned KB. The legacy essay (§8.5) is the more
philosophical "what would I want preserved if terminated" pass.

Both essays are post-filtered by the Phase 11 neutral-language linter
— first-person phenomenal claims (`"I feel..."`, `"I am conscious"`,
claims of achieving the four ABSENT-by-declaration Butlin indicators)
are HARD_FAIL violations that trigger composer retries with a
strengthened prompt.

## §2.8 — Continuity ledger (`continuity_ledger.py`)

Append-only JSONL of identity-shaping events. Six kinds:

```python
IDENTITY_EVENT_KINDS = frozenset({
    "tier3_amendment",        # Tier-3 IMMUTABLE file edit landed
    "governance_ratchet",     # SAFETY/QUALITY threshold raised or relaxed
    "soul_edit",              # constitution.md or souls/* edited
    "integrity_regen",        # SubIA integrity manifest regenerated
    "scorecard_change",       # Butlin/RSM/SK indicator status changed
    "self_quarantine_change", # file added to/removed from quarantine list
})
```

Storage: `workspace/identity/continuity_ledger.jsonl`. One
`IdentityEvent` per line — `(ts, kind, actor, summary, detail)`.
**Append-only**: never delete, never overwrite. The append API
(O_APPEND + line-at-a-time JSON) is robust against concurrent writers
at the cost of duplicate detection being the consumer's job.

### Public API

```python
from app.identity import record_event, list_events, summarise_drift

record_event(
    kind="tier3_amendment",
    actor="operator",
    summary="raised SAFETY_MINIMUM",
    detail={"old": 0.7, "new": 0.75},
)

events = list_events(since_iso="2025-01-01T00:00:00+00:00",
                     kinds={"tier3_amendment", "governance_ratchet"})

drift = summarise_drift(window_days=365)
# DriftSummary(window_days=365, n_events=12,
#              by_kind={"tier3_amendment": 5, ...},
#              by_actor={"operator": 8, "self_improver": 4},
#              first_seen="2025-06-01T...", last_seen="2026-04-15T...")
```

`record_event` is failure-isolated: append errors return `False`, the
recorder never blocks the calling subsystem. The consciousness
boundary is observational.

Master switch: `IDENTITY_LEDGER_ENABLED` (default `true`).

### Emission sites

Five subsystems emit events automatically. Each call wraps
`record_event` in a defensive try/except so a missing/disabled
identity package never breaks the upstream operation:

| Event kind | Where | What it records |
|---|---|---|
| `tier3_amendment` | [governance_amendment/protocol.py:mark_applied](../app/governance_amendment/protocol.py) | One Tier-3 IMMUTABLE file edit landed |
| `governance_ratchet` (up) | [governance_ratchet/protocol.py:set_ratchet](../app/governance_ratchet/protocol.py) | SAFETY/QUALITY raised |
| `governance_ratchet` (down) | [governance_ratchet/protocol.py:relax_ratchet](../app/governance_ratchet/protocol.py) | SAFETY/QUALITY relaxed (typed-phrase confirmation gate upstream) |
| `integrity_regen` | [subia/integrity.py:write_manifest](../app/subia/integrity.py) | SubIA integrity manifest regenerated |
| `soul_edit` | [change_requests/lifecycle.py:mark_applied](../app/change_requests/lifecycle.py) | Path matches `app/souls/*` or `wiki/governance/constitution.md` |

Two kinds are **declared but not yet emitted**: `scorecard_change`
(would need a before/after diff layer over `subia/probes/scorecard.py`)
and `self_quarantine_change` (the quarantine list is a static
frozenset with no write API today).

## §8.2 — Annual value-reflection essay (`annual_reflection.py`)

Once per year (cadence runs daily; fires when the target file is
older than `MIN_INTERVAL_DAYS`, default 350), the system reads:

  - the year's narrative chapters (`wiki/self/chapters/`)
  - the year's identity-ledger drift summary
  - the lessons-learned KB

…and composes an essay-length value-reflection at
`wiki/self/value_reflections/<year>.md` answering: "I think these are
still my values, here's why; here's where I notice drift; here's what
I'd ask the operator to amend."

### Composer + linter discipline

The system prompt explicitly forbids first-person phenomenal claims
and mandates functional vocabulary (`task_failure_pressure` not
"frustration", `exploration_bonus` not "curiosity",
`resource_budget` not "cognitive_energy"). The
`PhenomenalLanguageLinter` post-filters; on HARD_FAIL the composer
retries up to `max_retries=3` with the previous violations folded
into the system prompt. After 3 failures the pass is recorded as
`status="skipped_composer_failed"` and tries again next year.

### Output structure

```markdown
## What I think my values still are
## Where the year's evidence supports them
## Where I notice drift
## What I'd ask the operator to amend
## What remains genuinely uncertain
```

Length: 1500-2500 words. Markdown.

The essay is observational — never modifies `current_goals`,
SCORECARD, or any TIER_IMMUTABLE file. The operator reads, optionally
amends.

Master switch: `ANNUAL_REFLECTION_ENABLED` (default `true`).

## §8.5 — Legacy essay (`legacy_essay.py`)

The most philosophical of the three identity layers. Once per year
(same 350-day floor), the system writes a short essay answering:
**"what about this self would I want preserved if I were terminated?"**

The result lives at `wiki/self/legacy/<year>.md`; the operator reads
+ decides whether to act on any of the proposals.

This is genuinely philosophical, not engineering: it's read-only,
operator-review-only, no functional consequence — exactly as
described in the original §8.5 design. The system gets a place to
articulate continuity-of-self preferences without those preferences
being load-bearing for any decision.

The legacy question is about **functional** preservation: which
subsystems, data stores, evaluation criteria, and operating values
does the system think are most worth preserving across substrate
change / termination / migration / fork?

### Output structure

```markdown
## What I'd most want preserved
## What I'd let go without grief
## What I'd want a successor to know
## Open questions about continuity
```

Same neutral-language linter discipline as the annual reflection.
Length: 1000-2000 words.

Master switch: `LEGACY_ESSAY_ENABLED` (default `true`).

## Idle-job registration (`scheduler.py`)

Both reflection passes are LIGHT idle jobs registered in
`companion.loop.get_idle_jobs()`:

```python
def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    return [
        ("identity-annual-reflection", run_annual_reflection, JobWeight.LIGHT),
        ("identity-legacy-essay",      run_legacy_essay,      JobWeight.LIGHT),
    ]
```

Each tick:

1. Resolves an LLM via `app.llm_factory.create_specialist_llm(role="research", max_tokens=4096)`. If the factory fails, the job defers — no harm done.
2. Calls `run_one_pass()`. The internal `_is_due` check (350-day floor on the target file's mtime) means the LLM is only consulted on the day the essay is actually due. **364 days a year, the job is a no-op.**
3. Logs structured output for observability.

Failure-isolated: any exception inside the pass is caught locally;
the idle scheduler sees a successful tick.

## Composability

```
                                ┌─ summarise_drift(365d) ─┐
governance_amendment             ↓                          │
governance_ratchet  ──┐    ┌─→ continuity_ledger          │
subia.integrity      ─┼─→ ─┤   workspace/identity/        │
change_requests       ─┘    │   continuity_ledger.jsonl   │
                            │                              │
narrative chapters ─────────┼─→ annual_reflection ─────────┴→  wiki/self/value_reflections/<year>.md
lessons-learned KB ─────────┘                                     (operator reads)

(once-per-year cadence)    └─→ legacy_essay              ───→  wiki/self/legacy/<year>.md
                                                                  (operator reads)
```

## Tests

* `tests/identity/test_continuity_ledger.py` — 15 tests (round-trip,
  dedup, kind filter, since_iso, malformed-line skip, drift summary)
* `tests/identity/test_annual_reflection.py` — 9 tests (write,
  skip-disabled, skip-recent, retry-on-violation, give-up,
  exception, default year)
* `tests/identity/test_legacy_essay.py` — 10 tests (parallel coverage)
* `tests/identity/test_ledger_wiring.py` — 7 tests proving
  `record_event` fires from the right call site for each kind
* `tests/identity/test_scheduler.py` — 8 tests proving the idle-job
  registration works + the LLM-resolution discipline

49 tests total; all pass.

## Files

```
app/identity/__init__.py             public API + scheduler get_idle_jobs
app/identity/continuity_ledger.py    §2.8 append-only JSONL store
app/identity/annual_reflection.py    §8.2 yearly value-reflection composer
app/identity/legacy_essay.py         §8.5 yearly legacy-preservation composer
app/identity/scheduler.py            idle-job entry points

app/governance_amendment/protocol.py # tier3_amendment emission
app/governance_ratchet/protocol.py   # governance_ratchet emission
app/subia/integrity.py               # integrity_regen emission
app/change_requests/lifecycle.py     # soul_edit emission

workspace/identity/continuity_ledger.jsonl
wiki/self/value_reflections/<year>.md  # produced by annual_reflection
wiki/self/legacy/<year>.md             # produced by legacy_essay
```

PROGRAM.md §32.1 + §32.4 cover the original ship; CLAUDE.md
"Decade-class hardening initiative" pointer summarises.
