# AUTO_APPLY change requests

**Status (2026-05-10): Future-facing capability — shipped, dormant by default.**

The change-request system supports an `AUTO_APPLY` risk class that
bypasses the operator approval gate for narrowly-scoped, additive,
revertible patches from allowlisted self-heal handlers. The
infrastructure is fully built and tested; **no caller is allowed to
use it until an operator explicitly opts the caller and the path
into the validator's allowlists**.

This document is the operator handover for activating the
capability when a clean candidate consumer arrives.

## What auto-apply is for

Standard CRs route through `Signal 👍/👎` or `/cp/changes` operator
review. That gate is the right default — most agent-proposed
changes need eyeballs.

But three kinds of self-heal patches are *operationally noisy* and
*structurally trivial*:

1. New file additions (config injections, marker files).
2. Single-line additions to specific known files (defensive
   defaults, missing imports).
3. Idempotent overwrites of generated config (e.g. a runtime
   override JSON).

For these, every operator approval is a friction tax that delays
recovery. AUTO_APPLY lets specific handlers ship the patch immediately,
with an auto-revert guarantee if the originating error pattern recurs.

## Safety profile

A CR with `risk_class=AUTO_APPLY` is applied without operator approval
ONLY if it passes ALL of:

| Check | Constraint |
|---|---|
| Standard validator | Passes (TIER_IMMUTABLE refused, allowed roots, no path traversal, ≤1 MB, no blocked names like .env) |
| Forbidden prefixes | Path is NOT under `app/memory/`, `app/souls/`, `wiki/governance/`, `migrations/`, `deploy/`, `host_bridge/` |
| Requestor allowlist | `requestor` is in `_AUTO_APPLY_ALLOWED_REQUESTORS` (empty by default) |
| Path allowlist | Path matches an entry in `_AUTO_APPLY_ALLOWED_PATHS` (empty by default) |
| Line cap | Net delta (added − removed) ≤ 20 lines |
| Additive-only | Zero deleted lines from `old_content` |
| Per-pattern rate | ≤ 3 auto-applies per pattern signature per UTC day |
| Global rate | ≤ 10 auto-applies total per UTC day |

If any check fails, the CR is **gracefully downgraded** to STANDARD
risk class — it routes through the operator gate normally, never
gets rejected just for being ineligible for auto-apply.

## Auto-revert guarantee

After a successful auto-apply:

1. The CR + its origin pattern signature is registered with the
   `auto_revert` watcher (`app/change_requests/auto_revert.py`).
2. The watcher polls every 60s for 30 minutes.
3. If the originating error pattern recurs at all (≥1 increase
   from the baseline count at apply time), the watcher
   automatically calls `rollback_change` — git revert + hot-revert
   the file via the host bridge + open a revert PR.
4. After 30 minutes without recurrence, the watch entry is
   unregistered (success).

Operator is loudly notified at every transition (apply, revert,
revert failure) via Signal.

## SubIA / Global Workspace integration

Each auto-apply transition publishes to the Global Workspace
([`app.workspace_publish.publish_to_workspace`](../app/workspace_publish.py)):

| Transition | salience | signal_type |
|---|---|---|
| Apply succeeded | 0.6 | disposition |
| Apply failed | 0.45 | disposition |
| Auto-revert fired | 0.7 | trend_reversal |

Auto-applies that touch `app/souls/*` or `wiki/governance/constitution.md`
also emit a `soul_edit` event to the identity continuity ledger via
the existing `mark_applied` hook (no new wiring needed). This is
also why those paths are in the forbidden-prefix list: a soul edit
should never bypass operator review.

## Activating the capability

The capability ships dormant. To activate it for a specific consumer:

### Step 1 — design the candidate carefully

A clean auto-apply candidate is:

- A handler that runs in response to a specific recurring error pattern
- Produces a deterministic, idempotent patch (same input → same output)
- Touches a *single* file in a non-Tier-3, non-forbidden-prefix root
- The patch is purely additive (≤ 20 lines new, 0 removed)
- The originating error pattern has a stable signature so the
  auto-revert watcher can monitor it

If any of these doesn't hold, route through the standard CR gate.

### Step 2 — opt the requestor + path into the allowlists

Edit [`app/change_requests/validator.py`](../app/change_requests/validator.py):

```python
_AUTO_APPLY_ALLOWED_REQUESTORS: frozenset[str] = frozenset({
    "your_handler_name",
})

_AUTO_APPLY_ALLOWED_PATHS: tuple[str, ...] = (
    "workspace/idle_scheduler_overrides.json",
    "docs/proposed_capabilities/",   # prefix match — trailing slash matters
)
```

The allowlists are tuples/frozensets to make changes auditable in
git history.

### Step 3 — wire the handler

In your handler, call `create_request` with `risk_class=AUTO_APPLY`
and an `origin_pattern_signature`. If the handler's CR passes the
strict validator, call `auto_approve(cr.id)` to fire the apply path:

```python
from app.change_requests import (
    create_request, auto_approve, RiskClass,
)

cr = create_request(
    requestor="your_handler_name",
    path="workspace/idle_scheduler_overrides.json",
    new_content=...,
    old_content=...,
    reason=...,
    risk_class=RiskClass.AUTO_APPLY,
    origin_pattern_signature=anomaly["pattern_signature"],
)

if cr.risk_class == RiskClass.AUTO_APPLY:
    # Passed validator — fire the apply.
    auto_approve(cr.id)
else:
    # Was downgraded — operator approval needed.
    send_ask(cr.id)
```

### Step 4 — verify in production

Watch the gateway logs for:

- `change_requests: auto-approved <cr_id> by <requestor>` — apply fired
- `change_requests: applied <cr_id> — branch=... sha=... pr=...` — git landed
- `auto_revert: registered <cr_id>` — watcher active
- `auto_revert: watch expired for <cr_id>` — 30 min clean → success
- `auto_revert: rolling back <cr_id>` — pattern recurred → rollback

Cross-check via `/cp/changes/<cr_id>` for the full audit trail.

## Master switches

| Flag | Default | Purpose |
|---|---|---|
| `CHANGE_REQUESTS_AUTO_REVERT_ENABLED` | `true` | Disables the auto-revert daemon. Set to `false` to stop the watcher (existing registrations are cleaned up next start). |

## Why dormant by default?

Auto-apply is a step toward "agents modify code without explicit
approval." It must remain default-OFF until a real consumer is
identified, designed, and the operator chooses to opt it in
explicitly. The empty allowlists mean a misconfigured handler cannot
sneak a patch through — every auto-apply requires deliberate operator
action to enable.

## See also

- [`app/change_requests/validator.py`](../app/change_requests/validator.py) — `validate_auto_apply` + allowlists + forbidden prefixes
- [`app/change_requests/lifecycle.py`](../app/change_requests/lifecycle.py) — `auto_approve` + rate limit
- [`app/change_requests/auto_revert.py`](../app/change_requests/auto_revert.py) — watcher daemon
- [`tests/test_change_requests_auto_apply.py`](../tests/test_change_requests_auto_apply.py) — 30 tests covering the surface
- [`docs/CHANGE_REQUESTS.md`](CHANGE_REQUESTS.md) — base CR system
- [`docs/SELF_HEAL_V3.md`](SELF_HEAL_V3.md) — runbook handler architecture (the natural producer of AUTO_APPLY CRs)
