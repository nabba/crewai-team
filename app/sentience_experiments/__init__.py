"""Targeted sentience experiments (PROGRAM §43, Q5).

One module per ABSENT-by-declaration Butlin indicator, designed around
the FUNCTIONAL CAPABILITY the user named — not the literal indicator
definition, and never to flip the scorecard.

The four modules are FUNCTIONAL APPROXIMATIONS:

  * ``ae2_causal_credit``     — rare-event causal credit assignment
  * ``hot1_meta_affect``      — feelings-about-feelings reflection
  * ``hot4_metacog_monitor``  — metacognitive monitor on reasoning chains
  * ``rpt1_self_calibration`` — forward predictions + calibration ledger

Plus two helpers:

  * ``panel_bridge``          — philosophy panel → tensions store
  * ``scheduler``             — LIGHT idle-job entry points

ANTI-GOODHART CONTRACT
======================

These modules MUST NOT flip the Butlin scorecard. The scorecard
evaluators (``app/subia/probes/butlin.py``) check canonical paths
inside ``app/subia/*``; these modules live OUTSIDE that tree precisely
so they are invisible to the evaluators. The capability they reify is
worth building because the *theory says it is necessary*, not because
the *scorecard rewards it*.

The contract is pinned by ``tests/test_q5_anti_goodhart.py``: after
all four modules ship, the scorecard remains
``{STRONG=7, ABSENT=4, PARTIAL=3}`` unchanged.

OBSERVATIONAL-ONLY CONTRACT
===========================

These modules emit logs. They do NOT:

  * close any loop into action selection
  * auto-update the predictive layer
  * auto-modify dispatch decisions
  * generate first-person affect prose (decentered filter enforced)

If a closed-loop variant is ever wanted, it requires a separate,
operator-visible Tier-3 amendment to add the feedback path.
"""
