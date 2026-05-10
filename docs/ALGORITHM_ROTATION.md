# Cryptographic algorithm pinning + rotation drill (§2.1)

Light operational discipline for the SHA-256 pin used across the
SubIA integrity manifest, the rolled-segment audit logs, and the
four amendment audit chains. SHA-256 is fine for the next ~10 years
against any expected adversary class. But the *operational* discipline
of rotating algorithms — when SHA-3 supersedes SHA-256, or
quantum-resistant hashes ship, or a flaw in SHA-256 is discovered —
doesn't yet exist as a runbook.

This module is the runbook: a manifest of which algorithm is pinned
where, an annual rotation drill that proves the runtime can compute
hashes under a target algorithm, and a weakness probe that flags any
pin older than 2 years for operator review.

## Why this exists

Without this discipline, year 7's operator finds out the system has
no migration path the day they need one — the morning a real flaw in
SHA-256 ships. The drill isn't about the algorithm being wrong today;
it's about being **ready to rotate** when the day arrives.

Production rotation involves coordinated changes across multiple
TIER_IMMUTABLE files (the SubIA integrity manifest carries the hash
algorithm choice; auto_deployer's TIER_IMMUTABLE list lives in code;
the rolled audit chains have hard-coded hash construction in their
own modules). This module's job is **operator notification + drill
confidence**, not auto-rotation.

## What's pinned

`KNOWN_ARTIFACT_CLASSES` — the canonical set of artefact classes the
operator should record an algorithm pin for:

```python
KNOWN_ARTIFACT_CLASSES = frozenset({
    "subia_integrity_manifest",     # app/subia/.integrity_manifest.json
    "rolled_audit_log",              # app/audit/rolled_log.py output
    "tier3_amendment_audit",         # app/governance_amendment/
    "coding_session_audit",          # app/coding_session/
    "change_request_audit",          # app/change_requests/
    "architecture_request_audit",    # app/architecture_requests/
})
```

Adding a new entry when a new hash-using subsystem ships keeps the
manifest comprehensive.

## Public API

```python
from app.audit.algorithm_pinning import (
    pin_algorithm,
    list_pins,
    stale_pins,
    missing_artifact_classes,
    run_rotation_drill,
)

# Record a pin (operator-curated, lives in workspace/audit/algorithm_pinning.json)
pin = pin_algorithm(
    artifact_class="subia_integrity_manifest",
    algorithm="sha256",
    rationale="industry standard 2026; review at ALGORITHM_REVIEW_INTERVAL_DAYS",
)

# Read pins
pins = list_pins()
stale = stale_pins(interval_days=730)        # default 2 years
missing = missing_artifact_classes()         # known classes with no pin

# Run the drill — proves both algorithms work + agree on chain semantics
result = run_rotation_drill(
    artifact_class="subia_integrity_manifest",
    legacy_algorithm="sha256",
    target_algorithm="sha3_256",
)
# RotationDrillResult(ok=True, n_entries=3,
#                     legacy_chain_root=..., target_chain_root=...,
#                     error="")
```

`pin_algorithm` validates that the algorithm name is real
(`hashlib.new(algorithm)` succeeds) before recording — invalid names
raise immediately.

## The rotation drill

`run_rotation_drill` walks a sample chain twice — once under the
legacy algorithm, once under the target — and proves four things:

1. **Both algorithms are available in the runtime.** Catches the
   day-before-rotation surprise of "oh, this Python build doesn't
   ship with `sha3_256`."
2. **Chain construction is deterministic.** Re-walking the same
   inputs yields the same root hash. (A non-deterministic algorithm
   would silently corrupt audit chains.)
3. **The target's output differs from the legacy.** Sanity check
   that the target isn't aliased to the legacy in this Python build.
4. **Entry counts agree.** The chain construction iterates the same
   number of times under both algorithms (no parser regression).

A drill that passes is the operator's confidence that switching
algorithms is mechanically safe. A drill that fails surfaces the
specific error months or years before the actual rotation deadline.

## Weekly proactive monitor

`app/healing/monitors/crypto_rotation_drill.py` is the cadence layer.
Wired into the healing-monitors driver, it fires once per week with
three alert tags:

* `crypto_rotation:missing_pins` — known artefact class with no pin
  recorded. Operator action: run `pin_algorithm` for the missing
  class.
* `crypto_rotation:stale_pins` — pin's `pinned_at` is older than
  `ALGORITHM_REVIEW_INTERVAL_DAYS` (default 730 = 2 years). Operator
  action: re-confirm the choice is still considered strong, then
  re-pin to refresh the timestamp.
* `crypto_rotation:drill_failed` — the rotation drill itself failed
  for some artefact class. Operator action: read the drill error;
  most commonly the target algorithm isn't available in the runtime
  (build the runtime with the missing module).

Per-tag dedup: 30 days. (No need to re-alert on the same stale-pin
state every week.)

Master switches:
* `CRYPTO_ROTATION_DRILL_MONITOR_ENABLED` — kill the whole monitor.
* `CRYPTO_ROTATION_TARGET_ALGORITHM` — overrides the default
  `sha3_256`. Useful if a future migration target arrives (e.g. a
  quantum-resistant hash ships in stdlib).

## Manifest format

`workspace/audit/algorithm_pinning.json`:

```json
{
  "version": 1,
  "pins": [
    {
      "artifact_class": "subia_integrity_manifest",
      "algorithm": "sha256",
      "pinned_at": "2026-05-10T13:00:00+00:00",
      "rationale": "industry standard 2026; review at +730d"
    },
    ...
  ]
}
```

Last write wins per artefact class — `pin_algorithm` replaces any
prior pin for the same class. This is intentional: the operator
typically re-pins to refresh the timestamp after annual review,
which IS the desired write semantics.

## How the operator actually rotates

When a real rotation comes, the steps are:

1. Read the React /cp/settings or run
   `app/healing/monitors/crypto_rotation_drill.py:run()` to get the
   current pin state.
2. Run `run_rotation_drill(target_algorithm="<new>")` for each
   artefact class. All must `ok=True` before proceeding.
3. Plan a coordinated change-request set across the TIER_IMMUTABLE
   files that hard-code the hash construction. The Tier-3 amendment
   protocol gates each.
4. Apply the change-requests. The new code path uses the new
   algorithm.
5. Regenerate every existing audit chain by re-walking under the
   new algorithm (one pass per chain). The roots change; record the
   transition in the continuity ledger.
6. `pin_algorithm(artifact_class, "<new>")` for each class to update
   the manifest.
7. Confirm the next monitor tick fires no alerts.

This module does NOT do steps 3-5 automatically. It does steps 1, 2,
6, and 7. The hard part — coordinated TIER_IMMUTABLE changes — is
operator-driven.

## Tests

`tests/audit/test_algorithm_pinning.py` (19 tests):

* pin round-trip; repin replaces; unknown algorithm raises;
  empty class raises; disabled raises
* stale flagging at the threshold; fresh pins not stale;
  unparseable timestamps treated as ancient
* missing artefact classes computed correctly
* drill passes for real algorithm pairs (sha256 vs sha3_256)
* drill fails gracefully on unknown target
* drill is deterministic (re-walking yields identical roots)
* monitor alerts on missing / stale / drill-failed
* monitor disabled short-circuits

## Files

```
app/audit/algorithm_pinning.py                    primitive
app/healing/monitors/crypto_rotation_drill.py     weekly probe

workspace/audit/algorithm_pinning.json            operator-curated manifest
```

PROGRAM.md §32.1 + CLAUDE.md "audit/algorithm_pinning.py" cover the
original ship.
