"""subia.dreams — Backward counterfactual replay (consciousness-roadmap §3.G2).

NOT to be confused with:
  * `app/subia/reverie/`        — concept-walk synthesis (idle ideation)
  * Anthropic Managed Agents *dreams* — session-transcript curation
  * CIL Step 5 PREDICT          — *forward* counterfactual prediction over
                                  upcoming operations

This package implements the third meaning of "dreams" — recombination of
past episodes into hypothetical alternative pasts, run through the existing
forward-prediction machinery on synthetic inputs, with prediction-error
folded back into the retrospective signal store.

Tier-3 protected. The replay engine is observational: it does not write
to belief, beliefs cohort, or `current_goals`; it writes only to its own
audit log under `workspace/dreams/replay_audit.jsonl`. The retrospective
rescan (`app/subia/memory/retrospective.py`) consumes those signals
through its existing prediction-error gate — no new privileged path.

Triggers ethical threshold T2 (consciousness-roadmap §6) on first sustained
promotion: replay output gets a separate audit stream, operator review is
available before promotion influences belief or skill stores.
"""
from .engine import (
    FragmentSource,
    PerturbationKind,
    ReplayOutcome,
    ReplayScenario,
    construct_scenarios,
    run_pass,
    sample_fragments,
)

__all__ = [
    "FragmentSource",
    "PerturbationKind",
    "ReplayOutcome",
    "ReplayScenario",
    "construct_scenarios",
    "run_pass",
    "sample_fragments",
]
