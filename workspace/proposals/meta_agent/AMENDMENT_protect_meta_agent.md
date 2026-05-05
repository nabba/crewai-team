# Operator action: protect the meta-agent layer in TIER_IMMUTABLE

_Generated as part of the meta-agent layer rollout (May 2026). Apply by hand — no code in the meta-agent package can edit `app/auto_deployer.py`._

## Why this is needed

The meta-agent layer (`app/self_improvement/meta_agent/`) selects which augmentation to apply on top of an agent factory output. Its boundaries — selector thresholds, schema, the bounded-augmentation set — are safety-critical: an autonomously modified selector could undo the "bounded variant" guarantee by adding tunable knobs that reach beyond `force_tier`/`extra_tools`/`task_hint`/`max_execution_time`.

The package is already self-described as `IMMUTABLE — infrastructure-level module` in each module's docstring (matching the convention used by the rest of `app/self_improvement/`). This proposal upgrades that from a developer-discipline marker to a `TIER_IMMUTABLE` enforcement entry.

## Suggested edit

In `app/auto_deployer.py`, inside the `TIER_IMMUTABLE` `frozenset`, add the following block (alphabetised under the "Self-improvement" comment, or appended at the end of the set — whichever the maintainer prefers):

```python
    # Self-improvement meta-agent layer — bounded Hyperagents variant.
    # The selector thresholds, schema, and apply boundaries are
    # infrastructure-level safety constraints. Letting agents edit
    # these would defeat the "bounded variant" guarantee documented
    # in the package docstring.
    "app/self_improvement/meta_agent/__init__.py",
    "app/self_improvement/meta_agent/types.py",
    "app/self_improvement/meta_agent/feature_flag.py",
    "app/self_improvement/meta_agent/store.py",
    "app/self_improvement/meta_agent/selector.py",
    "app/self_improvement/meta_agent/apply.py",
    "app/self_improvement/meta_agent/recorder.py",
    "app/self_improvement/meta_agent/policy_gap.py",
    "app/self_improvement/meta_agent/amendment.py",
```

## Risks

1. **Tightens, never loosens.** Each file added becomes "never auto-modify, period." Reverting is hot — just remove the entries.
2. **Operator-only edits going forward.** Future tuning of the selector's `SELECTION_THRESHOLDS`, the recipe schema, etc. will need a manual hand-edit through the GATED canary path or a direct operator commit. That is the intent.
3. **No effect on opt-in dispatch.** The `META_AGENT` env flags continue to work; this proposal only changes who can edit the code that processes them.

## Reversal plan

If for some reason these need to come out of `TIER_IMMUTABLE`:

1. Delete the listed entries from `app/auto_deployer.py`.
2. The next dispatch picks up the loosened protection immediately.
3. Files revert to the tier OPEN default — modification through standard eval + rollback becomes possible again.

## Verification after applying

```sh
pytest tests/test_auto_deployer*.py -v
pytest tests/test_meta_agent*.py -v
```

Both should pass. The first verifies `TIER_IMMUTABLE`'s contents are still well-formed; the second verifies the meta-agent layer continues to function with no behaviour change (its file paths just gained protection).
