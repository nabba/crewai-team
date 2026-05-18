# Change Requests — Phase 5.3a

Agent-callable code-modification workflow with a human gate.

Lets a coding (or any) agent propose a write to a restricted path
(`app/...`, `tests/...`, `dashboard-react/...`, etc.), validates it
against the immutable safety rules, asks the operator for approval
via Signal **and** the React control plane, and on approval hot-applies
the change and opens an auto-PR against `main`.

This is the systemic fix to the failure mode discovered after the
2026-05-04 PIM incident: the coding crew was sandboxed to
`output/`, `skills/`, `proposals/` only and could not patch
production code. See `PROGRAM.md` §16 for the incident write-up.

**Update 2026-05-18 (Q18 / PROGRAM §60):** `create_request()` is now
deduplicated. Identical proposals (same `requestor` + `path` +
`diff`) from a single producer collapse into one canonical CR with
`recurrence_count` instead of accumulating duplicate records.
Closes the 2026-05-16 incident where the `local_only_drill` filed
1204 identical CRs into the operator's review queue. See
`docs/RESILIENCE_DRILLS_V2.md` §60.4 for the dedup semantics +
`app/change_requests/spam_cleanup.py` for the one-shot
consolidator that migrates legacy duplicates into the new model.

---

## 1. Architecture (one diagram)

```
┌──────────────────────────────────────────────────────────────────┐
│                     AGENT (e.g. coder crew)                      │
│   request_restricted_write(path, new_content, reason)            │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
                  ┌───────────────▼───────────────┐
                  │  app.change_requests.lifecycle │
                  │     create_request(...)         │
                  └───────────────┬───────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
        ┌───────────────┐                   ┌──────────────┐
        │   validator   │                   │     store    │
        │ TIER_IMMUTABLE│ ─── refused ───►  │ JSONL +      │
        │ allowed-roots │                   │ hash-chained │
        │ size cap      │                   │ audit log    │
        └───────┬───────┘                   └──────────────┘
                │ ok
                ▼
        ┌───────────────┐         ┌──────────────────┐
        │  send_ask()   │ ──────► │  Signal owner    │
        │  diff + 👍/👎 │         │  (msg ts saved)  │
        └───────────────┘         └────────┬─────────┘
                                           │
                            ┌──────────────┴──────────────┐
                            │                             │
                            ▼                             ▼
                  Signal reaction handler         React /api/cp/changes
                  (main.py /signal/inbound)       (POST .../approve)
                            │                             │
                            └──────────────┬──────────────┘
                                           ▼
                                  ┌────────────────┐
                                  │   approve()    │
                                  │  + apply()     │
                                  └────────┬───────┘
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
          host bridge: write       importlib.reload         git: branch + commit
          (HOST_REPO_PATH)         (best-effort, hot)       + push + gh pr create
```

Two surfaces, one lifecycle. Whoever decides first wins; the loser
sees "already approved" / "already rejected" via the lifecycle's
idempotent transitions.

---

## 2. Public API

```python
from app.change_requests import (
    # data model
    ChangeRequest, Status, DecisionSource,
    # lifecycle (state transitions)
    create_request, approve, reject, attach_signal_ts,
    mark_applied, mark_apply_failed, mark_rolled_back, mark_timeout,
    # apply / rollback (filesystem + git)
    ApplyResult, apply_change, rollback_change,
    # signal
    build_ask_body, send_ask, find_request_by_signal_ts,
    # store / lookup
    get, list_all, find_by_signal_ts,
    # validator
    validate, is_protected,
)
```

---

## 3. State machine

```
PENDING ─┬─→ APPROVED ──→ APPLIED ──→ ROLLED_BACK
         │           ╲
         ├─→ REJECTED ╲─→ APPLY_FAILED ──→ (retry-apply) ──→ APPLIED
         ├─→ TIER_IMMUTABLE_REFUSED   (terminal at request time)
         └─→ TIMEOUT
```

Transitions:

| From | To | Trigger |
| --- | --- | --- |
| PENDING | APPROVED | Signal 👍 OR React `POST /approve` |
| PENDING | REJECTED | Signal 👎 OR React `POST /reject` |
| PENDING | TIMEOUT | no decision in 10 min (idle reconciler — TODO) |
| PENDING | TIER_IMMUTABLE_REFUSED | validator rejected at `create_request` time — never reaches Signal/React |
| APPROVED | APPLIED | hot-apply + git auto-PR succeeded |
| APPROVED | APPLY_FAILED | hot-apply or git operations failed |
| APPLY_FAILED | APPROVED | React `POST /retry-apply` |
| APPLIED | ROLLED_BACK | React `POST /rollback` |

**TIER_IMMUTABLE files are absolute.** Even React-side operator override cannot
bypass them. The list lives in `app/auto_deployer.py`. The validator
returns `is_tier_immutable=True` so the lifecycle records the request
as `TIER_IMMUTABLE_REFUSED` (durable in the audit log) and the agent
gets a clear error: *"operator must edit directly via PR."*

---

## 4. The two gates

| Gate | When | What | Reversible |
| --- | --- | --- | --- |
| **Gate 1** | Signal 👍 / React approve | Hot-apply file + reload module + open PR | Yes — operator clicks Rollback in React |
| **Gate 2** | Operator merges the auto-PR | Change becomes durable in `main` | Manual `git revert` |

The hot-apply lets the running gateway pick up the fix immediately.
The auto-PR is the durable artifact — without merge, the change
disappears on the next gateway redeploy. (CI auto-merge is intentionally
NOT wired up; the operator's merge click is gate 2.)

---

## 5. Validator rules

```python
from app.change_requests import validate, is_protected
result = validate(path="app/agents/pim_agent.py", new_content="...")
# → ValidationResult(ok=True, reason=None, is_tier_immutable=False)
```

Allowed roots: `app/`, `tests/`, `docs/`, `dashboard-react/`,
`deploy/`, `scripts/`, `host_bridge/`.

Rejected:
- TIER_IMMUTABLE paths (security core, eval infra, governance, forge,
  souls, capabilities — see `app/auto_deployer.py`)
- Paths outside the allowed roots (e.g. `workspace/foo.py`)
- Path traversal (`../...`)
- Absolute paths
- Blocked patterns: `.env`, `secrets/`, `.git/`, `__pycache__/`, etc.
- Content > 1 MB

`is_protected(path)` is a quick check used by the API to flag protected
files in list responses.

---

## 6. Storage

```
workspace/change_requests/
├── <id>.json          # one file per request, full ChangeRequest
└── audit.jsonl        # append-only, hash-chained
```

Audit log entry shape:

```json
{
  "ts": "2026-05-04T16:00:00Z",
  "prev_hash": "ab12cd34ef567890",
  "entry_hash": "5d4e3c2b1a098765",
  "payload": {
    "event": "approved",
    "request_id": "f47ac10b...",
    "status": "approved",
    "path": "app/agents/pim_agent.py",
    "requestor": "coder",
    "decided_by": "react-approve"
  }
}
```

The hash chain mirrors the Forge audit-log discipline (see
`app/forge/registry.py`). Tampering shows up immediately on
verification.

The store uses an `RLock` (not `Lock`) because `save()` holds the
lock while calling `_index()` for first-time lazy-load — see
`app/change_requests/store.py:39`.

---

## 7. Agent-callable tool

```python
# Tool name: request_restricted_write
# Capability: registers-tool (Phase 5.4 may add 'requests-code-change')
# Tier: PRODUCTION

# Pattern of use inside an agent task:
current = read_host_file(path="app/agents/pim_agent.py")
new = current.replace(
    "from crewai import Agent\n",
    "from crewai import Agent\n"
    "from app.agents._common import optional_tool_group\n",
)
result = request_restricted_write(
    path="app/agents/pim_agent.py",
    new_content=new,
    old_content=current,
    reason=(
        "PIM crew fails with NameError: optional_tool_group is not defined. "
        "Adding the missing import."
    ),
)
# → "Change request <id> created (PENDING). Signal ASK sent to operator
#    (msg ts=...). The operator will react with 👍 to approve or 👎 to reject."
```

The tool **returns synchronously** after submission. It does NOT block
waiting for the user's reaction — that comes out-of-band via Signal
or React. If the agent's task requires the change to be applied
before continuing, the task description must explicitly say so.

---

## 8. Operator surfaces

### Signal

The ASK body has the diff inline (truncated to 2000 chars), the
reason, and the request id:

```
🔧 CHANGE REQUEST · app/agents/pim_agent.py

From: coder
Reason: PIM crew fails with NameError: optional_tool_group is not defined.
Adding the missing import.

```diff
--- a/app/agents/pim_agent.py
+++ b/app/agents/pim_agent.py
@@ -1,3 +1,4 @@
 from crewai import Agent
+from app.agents._common import optional_tool_group
 ...
```

👍 to approve and apply  ·  👎 to reject
id: f47ac10b-58cc-4372-a567-0e02b2c3d479
```

The reaction handler in `app/main.py` correlates the reaction's
`target_ts` with the saved Signal message ts and dispatches to
`approve()` or `reject()` + `apply_change()`.

### React control plane

```
GET    /api/cp/changes                       # list (status filter, limit)
GET    /api/cp/changes/{id}                  # detail
POST   /api/cp/changes/{id}/approve          # approve + apply
POST   /api/cp/changes/{id}/reject
POST   /api/cp/changes/{id}/rollback         # revert APPLIED change
POST   /api/cp/changes/{id}/retry-apply      # retry after APPLY_FAILED
```

Auth: same `require_gateway_auth` dependency as the rest of `/cp/`.

`is_protected` is included in every list/detail response so the React
UI can flag TIER_IMMUTABLE paths visibly.

### React UI (Phase 5.3b)

The `/cp/changes` route surfaces the operator side:

- **List view** — newest-first table of all change requests, filterable
  by status (`pending`, `approved`, `applied`, `apply_failed`, `rejected`,
  `rolled_back`, `tier_immutable_refused`, `timeout`). Each row shows
  the path, requestor, status badge, `🛑 PROTECTED` flag for
  TIER_IMMUTABLE paths, the reason (truncated), apply error if any,
  and a short id.
- **Detail drawer** — slides in from the right when a row is clicked.
  Shows status, decision metadata (who/when/why), apply metadata
  (branch, commit, PR URL with a click-through link), rollback metadata
  (when/by/revert PR), the **full unified diff** with line-level
  coloring (`+` green, `-` red, hunk headers blue, file headers grey),
  and per-state action buttons:
  - PENDING → `Approve + apply`, `Reject`
  - APPLY_FAILED → `Retry apply` (also a manual reject path)
  - APPLIED → `Roll back…` (with a confirmation step)
  - TIER_IMMUTABLE_REFUSED → no actions; explicit "operator must edit
    directly via PR" notice
- **Polling** — list refetches every 8 s; detail every 5 s. The cache
  invalidates immediately on mutation success.

Files:

```
dashboard-react/src/types/changes.ts            # types matching backend
dashboard-react/src/api/changes.ts              # react-query hooks
dashboard-react/src/components/ChangesPage.tsx  # list + drawer + actions
```

Wired into `App.tsx` (`<Route path="/changes" />`) and `Layout.tsx`
(nav item with ✏️ icon, between Governance and Org Chart).

---

## 9. Apply / rollback details

### Apply path (`app/change_requests/apply.py`)

1. `bridge.write_file(path, new_content)` — host writes the file.
2. `_try_module_reload(file_path)` — `importlib.reload` for `app/...`
   Python files. Best-effort: failures don't fail the apply, just
   log a "restart needed" note.
3. Git operations via the bridge:
   - `git fetch origin main`
   - `git checkout -B auto/<id>-<short-path>`
   - `git add <path>`
   - `git commit -m "auto: <reason summary>"` (Co-Authored-By footer)
   - `git push -u origin <branch>`
   - `gh pr create --base main --head <branch> ...`

The commit SHA, branch, and PR URL are stored on the ChangeRequest
and surfaced in the React detail view.

### Rollback path

1. `bridge.write_file(path, old_content)` — restore captured original.
2. `_try_module_reload(file_path)`.
3. Git: `git revert <commit_sha>` on a `revert/<id>` branch + push +
   `gh pr create`. Operator merges to make the rollback durable
   in `main`.

Only `APPLIED` requests are rollbackable. Other terminal states
(`REJECTED`, `ROLLED_BACK`, `APPLY_FAILED`, `TIER_IMMUTABLE_REFUSED`,
`TIMEOUT`) cannot be rolled back — they were never applied.

---

## 10. Race & idempotency

The lifecycle's transition functions are idempotent on no-op
re-entry:

- `approve(id, source=...)` on an already-APPROVED request is a
  no-op (returns the existing record). The `decided_by` of the
  first decision sticks.
- `reject(id, source=...)` on an already-REJECTED request is a
  no-op.
- `apply_change(id)` on an APPLIED request is a no-op.

If Signal 👍 lands while the operator is hitting React's "Approve"
button, whichever path wins first becomes the recorded decision;
the loser sees the no-op return and the operator's UI shows
"already approved (Signal)."

---

## 11. Tests

`tests/test_change_requests.py` — 40 tests, all pass.

| Class | Coverage |
| --- | --- |
| `TestValidator` | TIER_IMMUTABLE rejection (with flag), allowed paths, traversal, blocked patterns, content size, normalization |
| `TestModels` | `to_dict`/`from_dict` roundtrip, `is_terminal`/`is_rollbackable` predicates |
| `TestStore` | save/get, hash-chained audit log integrity, list filtering, `find_by_signal_ts` |
| `TestLifecycle` | `create_request` happy path + TIER_IMMUTABLE refusal, approve/reject transitions, idempotency, illegal transitions |
| `TestAPI` | list / detail / approve / reject / rollback / retry-apply; `403` on TIER_IMMUTABLE approve; `409` on wrong-state |
| `TestSignalIntegration` | `build_ask_body` shape; diff truncation pointer to React |
| `TestAgentTool` | tool registered; factory; TIER_IMMUTABLE refused; validation rejected; pending when Signal owner unavailable |

Run them:

```bash
docker exec crewai-team-gateway-1 python -m pytest tests/test_change_requests.py -v
```

---

## 12. Files

```
app/change_requests/
├── __init__.py        # public API exports
├── models.py          # ChangeRequest dataclass + Status / DecisionSource enums
├── validator.py       # validate() + is_protected()
├── store.py           # JSONL + hash-chained audit, RLock for reentrancy
├── lifecycle.py       # state transitions
├── apply.py           # hot-apply + git auto-PR + rollback
└── signal.py          # build_ask_body() + send_ask() + ts correlator

app/tools/
└── restricted_write_tool.py   # request_restricted_write BaseTool

app/control_plane/
└── changes_api.py     # /api/cp/changes/*  REST endpoints

app/main.py            # Signal reaction handler dispatch (👍 → approve+apply,
                       #                                   👎 → reject)

tests/
└── test_change_requests.py    # 40 tests

docs/
└── CHANGE_REQUESTS.md         # this file
```

---

## 13. Open follow-ups

- **Phase 5.4 — surface drift cleanup** (delegated coding specialists
  get bridge tools + `request_restricted_write` so they actually have
  the means to fix production code).
- **Idle reconciler — TIMEOUT transition** for PENDING requests older
  than 10 minutes. Currently the timeout state is defined but never
  written.
- **Capability vocabulary** — Phase 5.4 may add a dedicated
  `requests-code-change` capability tag (currently `registers-tool`
  is reused as the closest existing tag).
- **Per-agent requestor identity** — `_run` currently records
  `requestor="agent"`. Phase 5.4 wiring can plumb the actual
  caller's `agent_id` from the BaseTool invocation context.
- **Aggregated counts on the list view** — current `ChangesPage`
  shows status counts of the *currently filtered* slice. A small
  `/api/cp/changes/counts` endpoint would let it show all-status
  counts at the top.

See `PROGRAM.md` §16 for the full incident-driven program.
