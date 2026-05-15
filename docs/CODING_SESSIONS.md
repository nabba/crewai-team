# Coding Sessions — Phase 5.4 design (proposal)

**Status:** plan-only PR. No code lands until this design is reviewed and
approved. The implementation is gated on the merge of #54 and #55
(Phase 5.3a backend + 5.3b React UI).

**Why this exists:** the change-request system from #54 lets a coding
agent *deploy* a fix through a human gate. It does not let the agent
*develop* a fix — read, write, test, iterate. Today the agent submits
blind: it can read files via the bridge, but it can't run pytest, can't
verify imports resolve, can't check that a rename touched every call
site. The result is the failure mode that triggered the whole
post-PIM-incident program: agent submits a "fix" that introduces a new
NameError, Commander hallucinates "still broken" for three turns,
operator manually intervenes.

The fix isn't more bridge tools. It's a missing primitive: an ephemeral
worktree where the agent has fast feedback before any change leaves the
sandbox.

---

## 1. The primitive

```
┌─────────────────────────────────────────────────────────────┐
│  Coder agent (gateway container)                            │
│   tools: session_start, session_read, session_write,        │
│          session_run, session_diff, session_submit,         │
│          session_discard                                    │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  app.coding_session.manager                                 │
│   - lifecycle (start / list / get / expire)                 │
│   - quotas (disk, time, count, idle)                        │
│   - bounded subprocess runner                               │
└─────────────────┬───────────────────────────────────────────┘
                  │ via host bridge (read + git worktree)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  Host                                                       │
│   /Users/andrus/BotArmy/crewai-team/  (main repo)           │
│   /tmp/agent-sessions/<session_id>/    (worktree)           │
└─────────────────┬───────────────────────────────────────────┘
                  │ on session_submit:
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  app.change_requests  (Phase 5.3a — already shipped)        │
│   one ChangeRequest per touched file → Signal ASK / React   │
└─────────────────────────────────────────────────────────────┘
```

The worktree is the agent's scratch space. It holds a real `git
worktree add` from `main` (or any base branch). The agent reads, writes,
and runs commands inside it freely — no human gate, because nothing in
the worktree affects production. When the agent is done, `session_submit`
computes the diff, splits it per file, and files one or more change
requests through the existing Phase 5.3a lifecycle. From there, the
human gate, hot-apply, auto-PR, and rollback all work as already shipped.

Three properties the primitive holds:

1. **Sandbox isolation** — writes inside the worktree never reach the
   live tree. The agent can run pytest against its in-progress code
   without polluting anything.
2. **Single escape hatch** — the only way out of the worktree is
   `session_submit`, which routes through the validator (`TIER_IMMUTABLE`
   absolute) and the human gate (Signal/React).
3. **Self-cleaning** — quotas + idle reconciler ensure abandoned
   sessions are garbage-collected; nothing leaks indefinitely.

---

## 2. Tool surface

Six tools, all registered through the existing `@register_tool` decorator
(see `app/tool_registry/`). All capability tags are new — see § 7.

### `coding_session_start(base="main", purpose: str) → session_id`

Creates a fresh worktree from `<base>` at `/tmp/agent-sessions/<uuid>/`.
`purpose` is a one-paragraph statement of what the agent intends to do
— surfaces in the audit log and in the eventual change-request
description.

Constraints:
- Per-agent: max 3 active sessions
- System-wide: max 20 active worktrees, 5 GB total
- `base` must be a real ref reachable from `origin` (rejects garbage
  branch names with a clear error)

Returns: `{session_id, worktree_path, base_sha, expires_at}`.

### `coding_session_read(session_id, path) → content`

Reads any file inside the worktree. Path is relative to the worktree
root. The agent can read both files it has modified AND unchanged files
— useful for "what does the helper this calls actually look like" mid-fix.

### `coding_session_write(session_id, path, content)`

Writes inside the worktree. Two checks fire on every call:

1. **Path validator** — same `validate(path, content)` as
   `app.change_requests.validator`. TIER_IMMUTABLE refused at write time
   so the agent learns the rule early, not at submit. Outside-roots
   refused. Path traversal refused.
2. **Disk quota** — per-session 100 MB cap. Total worktree bytes checked
   on every write.

The change-request validator's TIER_IMMUTABLE rule is the
*authoritative* check. Refusing in `session_write` is just a fast-fail
nicety — re-checked on submit.

### `coding_session_run(session_id, argv: list[str], timeout_s=120) → {stdout, stderr, exit_code, elapsed_ms, truncated}`

Runs a command inside the worktree's working directory. The most important
tool of the seven — this is the iteration loop.

**Sandbox** (non-negotiable):
- `argv` is a list, not a shell string — no `bash -c`, no shell expansion
- Allowlist of executables: `pytest`, `python`, `python3`, `ruff`, `mypy`,
  `eslint`, `npx`, `node`, `git` (read-only subcommands only:
  `status diff log show ls-files`), `gh` (read-only: `pr view issue view`),
  `cat`, `ls`, `wc`, `head`, `tail`, `grep`, `rg`. Anything outside
  rejects with a clear error.
- No network. Implementation: subprocess inside a netns with no default
  route, OR sandboxed via firejail/bwrap with `--net=none`. (Concrete
  approach picked at implementation time — see § 11 question 4.)
- CPU bounded by setrlimit (RLIMIT_CPU) to 4× wallclock
- Wallclock bounded by `timeout_s`, hard-killed at expiry
- stdout/stderr captured to memory, truncated at 64 KB each with a
  `truncated=True` flag in the response
- Working directory is locked to the worktree root — `chdir` traversal
  prevented

Why this list: enough to verify a Python/JS change (test, lint,
typecheck) and read git/PR context, but no install commands, no network,
no arbitrary shell. If the agent needs `pip install <pkg>`, that's a
manifest change that goes through a different review path.

### `coding_session_diff(session_id) → unified_diff`

Returns the cumulative `git diff` for the worktree against its base.
Useful for the agent to self-review before submission ("does my diff
actually do what I intended?"), and required for the test suite to
verify mid-session state.

### `coding_session_submit(session_id, reason: str) → SubmitResult`

Ends the iteration loop, files change requests, destroys the worktree.

```python
SubmitResult = {
    "session_id": str,
    "results": [
        {"path": str, "change_request_id": str | None, "status": str,
         "refusal_reason": str | None},
        ...
    ],
    "summary": {"submitted": int, "refused_tier_immutable": int,
                "refused_validator": int}
}
```

Per-file behaviour:
- TIER_IMMUTABLE → no change request created; result includes
  refusal reason. The session can still submit other files; this one
  is rejected at the boundary.
- Validator failure → no change request created; result includes
  the validator's reason.
- Otherwise → `app.change_requests.create_request(...)` runs;
  `app.change_requests.send_ask(...)` sends the Signal ASK with the diff.
  The change_request_id flows back so the agent can reference it.

Worktree is destroyed after submit regardless of outcome. The agent
cannot continue iterating in a session it has already submitted —
clean state machine, no rebase semantics.

The reason text becomes the prefix for every change request's
`reason` field, with the per-file diff appended automatically.

### `coding_session_discard(session_id, reason: str)`

Destroys the worktree without filing change requests. Audit-logged.
The reason is captured for the postmortem ("agent gave up after
3 failed test runs" is useful debugging signal).

---

## 3. Module layout

```
app/coding_session/
├── __init__.py        # public API + Status enum
├── manager.py         # lifecycle (start, get, list_active, expire)
├── store.py           # session metadata persistence (JSONL + audit log,
│                      #   mirrors app/change_requests/store.py pattern,
│                      #   RLock for save/index reentrancy)
├── runner.py          # bounded subprocess execution
├── quotas.py          # disk / time / count enforcement
├── submit.py          # diff bundling + change-request fan-out
└── reconciler.py      # idle job: kill expired sessions, free disk

app/tools/
└── coding_session_tools.py    # the 7 tools above

app/control_plane/
└── coding_sessions_api.py     # GET /api/cp/coding-sessions (list)
                               # GET /api/cp/coding-sessions/{id} (detail)
                               # No POST — the agent owns lifecycle, not
                               # the operator. List is operator-visibility
                               # only.
```

No new tables in Postgres. Sessions are JSONL on disk like
change_requests, scoped to `workspace/coding_sessions/`. Cheap, durable
across restart, auditable.

---

## 4. State machine

```
ACTIVE ─┬─→ EXPIRED  (TTL or idle timeout — reconciler kills it)
        ├─→ DISCARDED (agent called session_discard)
        └─→ SUBMITTED (agent called session_submit)
```

Every transition appends to the audit log. The audit chain mirrors the
change_requests one (`prev_hash + canonical_json(payload) → entry_hash`).

A session in any non-ACTIVE state is read-only. The agent cannot resume
it, and the operator's React view shows it as historical.

---

## 5. Quotas & sandboxing

| Limit | Default | Why |
|---|---|---|
| Per-agent active sessions | 3 | Stops a runaway agent from hoarding worktrees |
| System active sessions | 20 | Stops a runaway *system* from filling /tmp |
| Per-session disk | 100 MB | A code change is small; this catches accidental log dumps |
| Total worktree disk | 5 GB | Hard ceiling on /tmp usage |
| Session TTL | 30 min | After this, idle reconciler kills it; the agent has to start a fresh session if it needs more time |
| Idle timeout | 10 min | Between any two tool calls; closes leaks if an agent crashes |
| `coding_session_run` wallclock | 120 s default, 600 s max | Long enough for `pytest tests/` on a focused module; short enough that a hung process is killed |

The reconciler runs as an idle job (`reconciler.py`), same pattern as
`belief-outbox-neo4j` and `dlq-drain` already in service. Frequency: every 5 min.

---

## 6. Integration with the change-request system

`session_submit` is the only point of contact. It calls:

```python
from app.change_requests import create_request, send_ask

for (path, new_content, old_content) in diffs_per_file:
    cr = create_request(
        requestor=session.agent_id,
        path=path,
        new_content=new_content,
        old_content=old_content,
        reason=f"{session.purpose}\n\n{submit_reason}\n\n[from coding session {session.id}]",
    )
    if cr.status == Status.PENDING:
        send_ask(cr.id)
    results.append({"path": path, "change_request_id": cr.id, "status": cr.status.value})
```

Three behaviours fall out for free:

- **TIER_IMMUTABLE absolute** — `create_request` returns
  `Status.TIER_IMMUTABLE_REFUSED`; the session can't bypass it.
- **One Signal message per file** — operator sees each file as a
  separate ASK with its own 👍/👎. Big multi-file refactors become
  multi-message; the operator can approve some and reject others.
- **Hot-apply + auto-PR** — already wired in #54. No new code on
  the apply side.

Open question (§ 11.6): should we add an "all-or-nothing" submit mode
where the operator sees one ASK that covers N files and a single 👍
applies all? Useful for refactors. Risky because partial-apply on
failure is messy. Defer to operator feedback after first weeks of use.

---

## 7. Capability vocabulary additions

`app/tool_registry/capabilities.py` is TIER_IMMUTABLE. The new tags
require explicit operator approval to add. Proposed entries:

```python
"runs-coding-session": {
    "description": (
        "Executes commands inside an ephemeral coding-session worktree. "
        "Sandboxed to allowlist; no network; bounded CPU/time. Can read "
        "and execute test runners, linters, type checkers."
    ),
    "tier": "PRODUCTION",
    "category": "code-development",
},
"writes-coding-session": {
    "description": (
        "Writes files inside an ephemeral coding-session worktree. "
        "Worktree is sandboxed; writes never reach the live tree. "
        "Submission via session_submit goes through the change-request "
        "human gate."
    ),
    "tier": "PRODUCTION",
    "category": "code-development",
},
"reads-coding-session": {
    "description": "Reads files inside a coding-session worktree.",
    "tier": "PRODUCTION",
    "category": "code-development",
},
"submits-coding-session": {
    "description": (
        "Bundles a coding-session diff and files change requests through "
        "the human gate. Single escape hatch from sandbox to production."
    ),
    "tier": "PRODUCTION",
    "category": "code-development",
},
```

All four go behind the same governance signoff that other PRODUCTION
capabilities go through.

---

## 8. Coder agent migration

The existing `Coder` agent is the migration target. Two changes:

1. **Tool inventory** — add the 7 `coding_session_*` tools to the
   coder's registry-resolved tool set. Keep `request_restricted_write`
   (one-shot atomic fix path) and the existing `read_host_file`
   (out-of-session reads).
2. **Task description guidance** — update the agent prompt to:
   - Default to `coding_session_start` for any non-trivial fix
     (multi-line, multi-file, or anything that touches a path the agent
     hasn't seen tests for in the same session)
   - Use `request_restricted_write` only for atomic single-file fixes
     where iteration adds no value
   - End every session with `coding_session_run pytest <relevant tests>`
     before `session_submit`
   - On test failure, iterate; on persistent failure, `session_discard`
     and explain to the operator what blocked progress

The prompt change is a code change, but it's NOT in TIER_IMMUTABLE — the
coder agent's persona file sits in `app/agents/`, which is the
Phase 5.3a path. So the very first use of this system can be: the coder
modifies its own task description through the change-request flow.
Recursive but bounded by the same human gate.

---

## 9. Failure modes & defenses

| Failure | Defense |
|---|---|
| Worktree disk fills | Per-session quota check on every write; total cap enforced; oldest expired session evicted first |
| `coding_session_run` runs forever | Wallclock timeout + SIGKILL; CPU rlimit |
| Agent tries to escape sandbox via shell injection | `argv` is a list, no shell; allowlist blocks anything not in the list; chdir locked |
| Network exfil via `coding_session_run` | Subprocess in netns with no route, OR firejail `--net=none` |
| Agent crashes / loop times out, leaves session | Idle reconciler at 5 min interval kills any session past idle/TTL |
| Agent submits TIER_IMMUTABLE | `create_request` refuses; result returned as `refusal_reason` in submit response; agent's session ends in SUBMITTED with N partial successes + M refusals — no special-case path |
| Two agents race on the same file | Each has its own worktree from the same base. Last-merged-wins in the change-request flow. The second-arriving change request will diff against the *current* file content, not the agent's `old_content` — operator sees the up-to-date diff. (Operationally, this should be rare; tracked as a follow-up.) |
| Worktree corrupted | `session_submit` runs `git diff` per-file; if git fails, session goes into a special FAILED state and the operator gets a notification. Worktree retained for forensics, then GC'd after 24 h. |

---

## 10. Test plan

Unit + integration. No new infrastructure needed beyond the existing
test suite.

```
tests/test_coding_session_manager.py
  - start happy path returns session_id + worktree exists
  - start with TIER_IMMUTABLE base ref refused
  - start respects per-agent quota (4th rejected)
  - start respects system quota
  - get / list_active / expire
  - state machine transitions (ACTIVE → SUBMITTED, → DISCARDED, → EXPIRED)
  - audit log hash chain integrity

tests/test_coding_session_runner.py
  - allowlist enforcement (`bash`, `curl`, `pip` rejected)
  - wallclock timeout kills runaway process
  - stdout/stderr truncation
  - exit_code captured correctly
  - argv treated as list, not shell (test injection attempt with `;`)
  - chdir locked to worktree root
  - (manual smoke: network blocked — depends on sandbox impl)

tests/test_coding_session_quotas.py
  - per-session disk cap rejects oversized write
  - total cap enforced across sessions
  - TTL triggers reconciler-side EXPIRE
  - idle timeout triggers EXPIRE

tests/test_coding_session_submit.py
  - happy path: 2 modified files → 2 change requests created, both PENDING
  - TIER_IMMUTABLE in mix: TIER refused, others submitted
  - validator failure in mix: that file refused, others submitted
  - empty diff: returns no change requests, session ends SUBMITTED
  - worktree destroyed after submit
  - signal_ts attached if owner configured

tests/test_coding_session_e2e.py     (single integration test)
  - start session
  - write a file with a deliberate import error
  - run pytest, observe failure in stdout
  - write again with the fix
  - run pytest, observe success
  - submit, assert change request PENDING with the right diff
```

Manual smoke after merge: spin up a coder crew, ask it to fix a small
bug; observe the session lifecycle in the audit log; confirm the
change-request UI surfaces the resulting CR.

---

## 11. Open questions

1. **Sandbox technology for `coding_session_run`** — netns vs firejail
   vs bwrap. Container-internal options likely simplest (gateway
   already runs in a container; we can spawn a child container or use
   `unshare`). Concrete pick at implementation start.

2. **Read-only `gh` and `git` allowlist** — `gh issue view` and `git
   show <sha>` give the agent context that's hard to fetch otherwise.
   But they make network calls (gh) and could pull novel data into the
   loop. Suggested: allow `gh issue view`, `gh pr view`, `git log`,
   `git show`, `git diff`; document as the only network-reaching
   commands the runner permits, justified by their context value.
   Rejected: `gh pr create` (only `change_requests.apply` does that),
   `git push`, `git commit` (only via change-request system).

3. **Reading files outside the worktree?** The agent might want to
   read e.g. `tests/conftest.py` from the host-level repo without
   adding it to its session. Suggestion: `coding_session_read`
   forbids `..`; for cross-tree reads use the existing `read_host_file`
   (already in coder's inventory). Two distinct primitives, clear
   purpose.

4. **`pip install` / `npm install`** — explicitly out of scope. If the
   agent's fix depends on adding a dep, the manifest change goes via
   `request_restricted_write` against `requirements.txt` /
   `package.json`. Manifest changes are reviewed by the operator and
   only then does CI install + test — same path human devs use.

5. **Multi-base / stacked sessions** — `base="feat/foo"` should work,
   not just `base="main"`. Lets the agent rebase its WIP onto an
   in-flight feature branch. Need to think about: what if the base
   branch moves while the session is open? Suggest: lock the worktree
   to the `base_sha` at start; on submit, `git diff` is computed against
   that locked sha; the change-request's auto-PR is opened against
   `main` regardless (gate-2 review still goes through main). The
   `base_sha` is stored on the session and surfaces in audit + in the
   change-request reason.

6. **All-or-nothing submit mode?** Defer; ship per-file submit first,
   measure how often refactors get partial-approved, decide.

7. **Operator visibility** — should `/cp/coding-sessions` be a route in
   the React UI? Suggested: yes, read-only — see what active sessions
   exist, what files they're touching, give the operator preemptive
   visibility before submission. Phase 5.4b if 5.4 itself is too big.

---

## 12. Out of scope (explicitly NOT in 5.4)

- Auto-merge of approved change requests. Gate 2 stays manual.
- CI integration (running the full test suite on submit). The agent's
  `coding_session_run` is enough for fast feedback; CI is the operator's
  durable verification step.
- Multi-agent collaboration on a single session.
- Patches that span the worktree + non-worktree files (e.g., a fix
  that touches `app/foo.py` AND a `workspace/...` skill). The two
  systems remain distinct: `coding_session_*` for repo files,
  `file_manager` for workspace files.
- "Re-open" of a SUBMITTED session. A second iteration is a fresh
  session.
- IDE-style features (LSP, autocomplete, semantic refactor). The agent
  has read+write+test, which is the floor.

---

## 13. Implementation phases (for the eventual code PR)

If this design is accepted, the work breaks into:

| Phase | Scope | Test surface |
|---|---|---|
| **5.4-a** | Manager + store + state machine. No tools yet. | `test_coding_session_manager`, `test_coding_session_quotas` |
| **5.4-b** | Runner with allowlist + sandbox + tests | `test_coding_session_runner` |
| **5.4-c** | Submit + diff bundling + change-request fan-out | `test_coding_session_submit` |
| **5.4-d** | The 7 `coding_session_*` tools + capability registration | smoke test via tool registry |
| **5.4-e** | Coder agent prompt update via the change-request system itself (recursive but bounded) | manual smoke |
| **5.4-f** *(optional)* | `/cp/coding-sessions` React view (read-only) | TestClient-based |

Each lands as its own PR, stacked. Estimated total: ~1400 lines of code +
~600 lines of tests.

---

## 14. The principle

The bridge stays minimal: a filesystem/git tunnel, nothing more. The
change-request system stays minimal: a human-gated deploy lifecycle,
nothing more. The new primitive (coding sessions) is the missing third
piece — fast feedback for development.

Three small primitives, each with a single clear job, composing into
the full coding-agent flow. The system stays elegant because no piece
grows ad-hoc capabilities — when the agent needs something new, we ask
"which primitive does this belong to?" first.

---

## 15. Inline ShinkaEvolve (PROGRAM §45.4 — Q7.4)

When a single-shot fix isn't working, the coder agent can run a
population-based ShinkaEvolve search **inside one coding session**
against ONE target file and ONE evaluator script.

### 15.1 Surface

The agent calls `coding_session_evolve_solution` with:

| Argument             | Meaning                                                                                |
|----------------------|----------------------------------------------------------------------------------------|
| `session_id`         | Active session id (from `coding_session_start`).                                       |
| `initial_path`       | Repo-relative path to the program to evolve. Must exist in the worktree.               |
| `evaluate_path`      | Repo-relative path to a script that scores variants (higher = better).                 |
| `num_generations`    | 1–20 (hard-capped at `MAX_GENERATIONS_INLINE`).                                        |
| `num_islands`        | 1–3 (hard-capped at `MAX_ISLANDS_INLINE`).                                             |
| `max_cost_usd`       | 0.01–5.0 (hard-capped at `MAX_COST_USD_INLINE`).                                       |

Returns a JSON payload with `status`, `baseline_score`, `best_score`,
`delta`, `generations_run`, `variants_evaluated`, `duration_seconds`,
the unified `diff` (only when status is `improved`), and any
`error` / `refusal_reason`.

**Status vocabulary:**

| Status               | Meaning                                                                                |
|----------------------|----------------------------------------------------------------------------------------|
| `improved`           | Best variant scored strictly higher than baseline; diff returned.                      |
| `no_improvement`     | Runner finished but no variant beat baseline; no diff.                                 |
| `refused`            | Pre-validation rejected the request (see §15.3).                                       |
| `disabled`           | Master switch `shinka_inline_evolve_enabled` is OFF.                                   |
| `shinka_unavailable` | The `shinka` package is not installed in the runtime.                                  |
| `error`              | Runner crashed mid-flight (e.g. evaluator errors).                                     |

### 15.2 Hard caps

These are enforced at the bridge layer regardless of caller request:

```python
MAX_GENERATIONS_INLINE = 20
MAX_ISLANDS_INLINE     = 3
MAX_COST_USD_INLINE    = 5.0
```

The agent can request smaller, never larger. Cost is the LLM-proposal
budget for ShinkaEvolve's mutator (not the evaluator's compute, which
runs inside the same sandbox as the regular `coding_session_run`).

### 15.3 Refusal vocabulary

The bridge refuses the request before invoking the runner when ANY of:

* The session is unknown.
* The session is not in `ACTIVE` status.
* The session's worktree directory doesn't exist.
* Either path is absolute or contains `..` traversal.
* Either path resolves outside the worktree.
* Either path doesn't exist in the worktree.
* The repo-relative `initial_path` is in TIER_IMMUTABLE OR under
  `app/subia/` OR is `app/affect/goal_emitter.py` (uses
  `app.architecture_requests.validator.is_protected_path` for the
  shared protection vocabulary). Modifying any of those requires
  Tier-3 amendment, not a coding-session diff.

### 15.4 Master switch

`shinka_inline_evolve_enabled` in `runtime_settings.py` (default ON;
flip in `/cp/settings` → "Inline ShinkaEvolve per coding session"
card). When OFF, the bridge returns `status="disabled"` without
invoking validation or the runner. The bulk ShinkaEvolve subsystem
(`app.shinka_engine`, which the `/cp/evolution` page surfaces) has
its own master switch and is unaffected.

### 15.5 What the bridge does NOT do

* **It does not apply the diff.** When status is `improved`, the
  agent must call `coding_session_write(initial_path, evolved_content)`
  to apply the variant inside the session worktree, then
  `coding_session_submit` to fan it out as a per-file change request.
  Standard CR gate applies — nothing reaches the repo without
  operator approval.
* **It does not modify shared state.** All ShinkaEvolve intermediate
  artifacts go under `<worktree>/.shinka_inline/<timestamp>/`. When
  the session terminates, the worktree (including those artifacts)
  is cleaned up.
* **It does not survive a session boundary.** Each run is scoped to
  ONE session — no resumption, no cross-session population transfer.

### 15.6 Operator visibility

Every call to the bridge — even refused, disabled, or errored ones —
produces an audit row at:

```
workspace/coding_sessions/<session_id>/evolution_audit.jsonl
```

This file lives **outside** the worktree, so the trail survives
session cleanup. Each row carries:

* `ts`, `session_id`, `agent_id`
* `initial_path`, `evaluate_path`
* Requested `num_generations`, `num_islands`, `max_cost_usd`
* Returned `status`, `baseline_score`, `best_score`, `delta`,
  `generations_run`, `variants_evaluated`, `duration_seconds`
* `diff_sha256` and `diff_length` — the full diff is **not** stored
  (it's reconstructable from the post-submit CR audit log if the
  agent submits the variant)
* `error` / `refusal_reason` (whichever applies)

The file is capped at 200 rows per session via
`app.utils.jsonl_retention.append_with_cap`; older rows rotate out.

Operators see this surface in two places:

* **REST**: `GET /api/cp/coding-sessions/<id>/evolution_runs?limit=N`
  returns `{session_id, summary, runs}` where `summary` is a roll-up
  of `n_runs`, `by_status`, `best_delta`, `total_max_cost_usd`,
  `total_duration_seconds`, `last_run_at`.
* **React**: `/cp/coding-sessions` → click a session → the detail
  drawer's "Evolution runs" section renders KPI cards + per-run
  rows with status-coded badges (`IMPROVED` / `NO Δ` / `REFUSED` /
  `DISABLED` / `NO SHINKA` / `ERROR`). The section hides for
  sessions that never invoked evolution (zero noise).

### 15.7 When to use inline evolve

The agent should reach for `coding_session_evolve_solution` only
when ALL of:

1. A single-shot edit + test cycle has failed (typically ≥2 attempts).
2. The target file has a clear scalar fitness function (the
   evaluator script returns a number that's actually meaningful).
3. The variation space is plausibly fruitful (e.g. there's a
   parameter or branch that could plausibly improve under search).

Inline evolve is the *expensive* tool — it burns up to $5 of LLM
proposal budget per run. Most fixes should ride the regular write+run
loop. The audit row exists precisely so the operator can spot
overuse.

### 15.8 Test surface

```
tests/coding_session/test_evolution_bridge.py — 15 tests (refusal
  vocabulary, hard-cap clamping, improvement+diff, runner errors,
  shinka_unavailable, duration recording)

tests/coding_session/test_evolve_tool.py       — agent-tool integration
  (gateway-deps; passes when crewai is installed)

tests/test_q7_4_inline_evolve.py               — 15 tests (audit
  persistence, master-switch OFF path, REST endpoint wiring, React
  surface wiring, default-ON pinning)
```
