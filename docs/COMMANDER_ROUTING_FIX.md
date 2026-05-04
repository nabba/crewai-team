# Commander Routing Fix — Phase 5.2

Two-layer defense against the failure mode that surfaced 2026-05-04
(see `PROGRAM.md` §15 for the full incident write-up). Stops the
Commander LLM from emitting hallucinated "X crew is broken" routing
decisions when the underlying issue has been fixed but conversation
history still shows the old failure.

---

## 1. The failure mode

```
User:  "what is my calendar tomorrow?"
Bot:   "Crew pim failed: name 'optional_tool_group' is not defined"

[ PR #50 lands; gateway restarted; PIM agent now constructs cleanly ]

User:  "what is my calendar tomorrow?"
Bot:   "The PIM crew is currently broken (a code error: optional_tool_group
        is not defined). I can't fetch your calendar until this is fixed.
        Please ask me to debug and fix the PIM crew, and then I can check
        your schedule for tomorrow."
```

The Commander's routing prompt sees the prior failure message in
conversation history, decides "PIM is still broken," and emits
`{"crew": "direct", "task": "<refusal text>"}`. The system uses the
inline task as the response. The user gets a hallucinated refusal
instead of their calendar.

This happened three times in a row before being caught manually.
The actual PIM agent had been working from attempt #2 onward.

---

## 2. The fix — two layers

### Layer 1: Conversation-history sanitation (`mark_stale_failures`)

Before feeding history to the routing prompt, walk each line. When a
line matches a failure marker AND `system_state` shows at least one
crew has succeeded since the gateway started, prefix the line with:

```
[PRIOR — LIKELY RESOLVED, recent successful runs: ['pim', 'coding']]
```

This makes the failure message visible to the LLM (so it has full
context) but framed as historical, not authoritative. The LLM is
much less likely to extrapolate "still broken" from tagged content
than from raw stack traces.

**Conservative behavior**: failures are tagged ONLY when there's
positive evidence of recent success. If no successful runs exist
(e.g. fresh gateway boot, buffer empty), failures are left as-is —
we don't want to mislabel an unconfirmed failure as resolved.

Failure markers detected:
- `Crew \w+ failed:`
- `NameError:`, `ImportError:`, `AttributeError:`
- `Traceback (most recent call last):`
- `is currently broken`
- `is not working`
- `want me to (debug|fix)`

### Layer 2: Routing-decision validator (`validate_routing_decision`)

After the LLM emits a routing decision, scan each entry. For each
`crew=direct` decision, inspect the inline `task` text. When BOTH:

1. A refusal marker is present (`is broken`, `is currently broken`,
   `i can't fetch`, `want me to debug`, etc.)
2. A valid crew name appears as a whole word (`pim`, `coding`,
   `research`, etc.)

…override the decision. Replace it with:

```python
{"crew": <mentioned_crew>, "task": <original user_input>, "difficulty": <preserved>}
```

The actual dispatch then runs — either succeeds (real answer to the
user's actual question) or fails with the actual current error
(better feedback than a hallucinated refusal).

**Layer 2 fires unconditionally** once the refusal pattern is detected,
regardless of `system_state`. Reasoning: if the crew really IS broken,
a real attempt produces the actual error which the user can act on; if
not broken, the dispatch produces the right answer. Cost of an extra
dispatch is low; benefit of always-fresh failure surface is high.

---

## 3. Defense in depth — why both layers

| Scenario | Layer 1 prevents? | Layer 2 catches? |
|---|---|---|
| Bug fixed, gateway restarted, crew has succeeded once | Yes — failures tagged stale, LLM emits real dispatch | Catches residual hallucination if any |
| Bug fixed, gateway restarted, no successful runs yet (fresh) | No — buffer empty, no positive signal | Yes — pattern match on output |
| Bug NOT actually fixed, conversation has old failure | Maybe — if some other crew has succeeded, tags will fire | Yes — forces real dispatch, real error surfaces |
| User asks a non-PIM question with PIM in conversation history | Layer 1 doesn't tag (no failure marker matches their question); Layer 2 doesn't fire (no refusal markers in routing decision) |

Together they cover the propensity-to-hallucinate (Layer 1) and the
hallucination itself (Layer 2).

---

## 4. What this is NOT

* **Not a routing rewrite.** The LLM still does the heavy lifting of
  understanding the user's intent and picking a crew. We just
  intercept the specific pattern of "refusing because of stale
  context."
* **Not a permanent prompt change.** Layer 1 is a runtime annotation
  of the history feed; Layer 2 is a post-hoc check on the output.
  The Commander's system prompt is unchanged.
* **Not a workaround.** Both layers degrade gracefully:
  - Layer 1: if `system_state` is unreachable or buffer is empty,
    history is fed unchanged (current behavior).
  - Layer 2: if the LLM emits a sensible decision, validator passes
    it through unchanged. Override only fires on the specific
    refusal-as-direct pattern.
* **Not a replacement for fixing actual bugs.** When a crew IS
  broken, Layer 2 forces the dispatch which produces the real error.
  The error then needs an actual fix (e.g. via the Phase 5.3
  change-request workflow). Layer 2 just ensures the user sees the
  real problem instead of a confabulated one.

---

## 5. Components

| Path | Role |
|------|------|
| `app/agents/commander/routing_overrides.py` | The two layers — `mark_stale_failures()` and `validate_routing_decision()` plus the helper `detect_refusal_pattern()`. |
| `app/agents/commander/orchestrator.py` (modified) | Wire-in: Layer 1 is called between history fetch and prompt build (after `_fetch_history`); Layer 2 is called after JSON parsing and crew-name validation. Both wrapped in try/except — Phase 5.2 must not break routing. |
| `tests/test_commander_routing_overrides.py` | 20 tests including the headline `test_pim_incident_replay`. |
| `docs/COMMANDER_ROUTING_FIX.md` (this file) | Full reference. |

---

## 6. Tests

20 tests across:

| Class | Coverage |
|---|---|
| `TestDetectRefusalPattern` | The classifier fires on real PIM-incident text + variants; doesn't fire on legitimate non-dispatch responses; word-boundary check prevents partial-word false positives. |
| `TestValidateRoutingDecision` | Override on the headline incident; passes normal direct responses unchanged; passes correct crew dispatches unchanged; mixed-list override only affects the refusal entry. |
| `TestMarkStaleFailures` | No state → unchanged; no successful runs → unchanged (conservative); successful runs → tags applied; non-failure lines preserved; stack-trace patterns tagged; buffer-unavailable handled; empty input handled. |
| `TestPIMIncidentEndToEnd` | Replay full incident: Layer 1 marks the history correctly; Layer 2 catches residual hallucination; telemetry logging includes recent-success context. |

The headline test (`test_pim_incident_replay`) feeds the actual
routing decision the Commander emitted on 2026-05-04 and asserts the
override fires:

```python
incident_decision = [{
    "crew": "direct",
    "task": (
        "The PIM crew is currently broken (a code error: "
        "optional_tool_group is not defined). I can't fetch your "
        "calendar until this is fixed. Please ask me to debug and "
        "fix the PIM crew."
    ),
    "difficulty": 1,
}]
fixed = validate_routing_decision(incident_decision, "what is my calendar tomorrow?")

assert fixed[0]["crew"] == "pim"
assert fixed[0]["task"] == "what is my calendar tomorrow?"
```

---

## 7. Failure modes (all non-fatal)

| Where | Behavior |
|---|---|
| `system_state.get_system_state()` raises | Layer 1 skipped, history passes through unchanged. Layer 2 still runs (doesn't depend on state). |
| `mark_stale_failures` raises | Logged at debug; history passes through unchanged. |
| `validate_routing_decision` raises | Logged at debug; original decisions used. |
| Refusal pattern misses the actual hallucination | LLM's response goes through; failure surfaces in user-visible response (current behavior). |
| Refusal pattern false-positive on legitimate direct response | Forces a crew dispatch when none was needed. Crew may produce a useful answer or a "I don't have information for this" — either is better than the false-positive being silent. Operator-visible in logs. |

---

## 8. Operator visibility

Every override fires a `WARNING` log line:

```
routing_overrides: detected refusal-as-direct mentioning crew=pim;
overriding to actual dispatch with user_input='what is my calendar tomorrow?'
(last pim success at 2026-05-04T16:09:41Z). Hallucinated response was:
The PIM crew is currently broken...
```

Operators can grep `routing_overrides:` in gateway logs to count
how often Layer 2 fires. If it fires often without real underlying
bugs, the LLM's refusal-propensity is an issue worth investigating
upstream (model/prompt). If it fires rarely, the fix is doing its
quiet work.

---

## 9. What's next

* **Phase 5.3** — Change-request system + Signal/React voting + auto-PR.
  This gives the system a real path to fix bugs that surface (whether
  found via Layer 2's "real dispatch" or any other mechanism). Without
  Phase 5.3, the user still has to manually fix bugs even after Layer 2
  forces them to surface.
* **Phase 5.4** — Surface drift cleanup (delegated coding specialists
  get bridge tools + `request_restricted_write`).

---

## 10. Migration progress (post-PIM-incident program)

| Phase | Status |
|-------|--------|
| 5.1 — system_state foundation (#52) | DONE |
| **5.2 — Commander routing fix** | **THIS PR** |
| 5.3 — Change-request system + React UI | Next |
| 5.4 — Surface drift cleanup | After 5.3 |
