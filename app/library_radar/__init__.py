"""Library radar — filter tech-radar discoveries into adoption proposals.

Where :mod:`app.crews.tech_radar_crew` discovers candidate technologies
(7 topical web searches, stores in ChromaDB ``scope_tech_radar``) and
plants stubs for *models* in ``control_plane.discovered_models``,
nothing currently acts on the *framework / tool* discoveries — they
sit in ChromaDB and surface in the React dashboard, but there's no
pipeline that turns them into adoption proposals.

This module is that pipeline (minimal version):

  1. Read the recent ``scope_tech_radar`` ChromaDB entries.
  2. Filter to ``category in {frameworks, tools}``.
  3. Filter out anything already pinned in ``requirements.txt``.
  4. For survivors, write a markdown adoption proposal at
     ``docs/proposed_libraries/<slug>.md``.

The proposal is *advisory*. The operator reads it, evaluates whether
to adopt, and (when adopting) files a change-request for the
``requirements.txt`` edit through the standard human-gated review
path.

Trial + canary (sandboxed install + benchmark + auto-score) — Q10.1
(PROGRAM §46.13) — is now layered on top via :mod:`trial_runner` +
:mod:`trial_state`. The proposer marks new discoveries as pending in
the trial-state ledger; the trial_runner walks those rows on the
same daemon cadence, runs the smoke import test inside a
coding-session sandbox, and on pass files an *adoption CR* for the
``requirements.txt`` edit through the standard operator gate.

Master switches:
  * ``LIBRARY_RADAR_ENABLED`` (default ``true``) — discovery loop
  * ``LIBRARY_TRIAL_RUNNER_ENABLED`` (default ``true``) — trial runner

Daemon thread eager-starts at import time, mirrors
:mod:`app.healing.monitors`.
"""

from app.library_radar.proposer import (
    Discovery,
    run_one_pass,
    start,
    stop,
)
# Q10.1 (PROGRAM §46.13) — trial runner + state surfaces. Side-effect
# imports so the modules are reachable via attribute access from
# tests, REST handlers, and the proposer's daemon loop.
from app.library_radar import trial_runner, trial_state  # noqa: F401

__all__ = [
    "Discovery",
    "run_one_pass",
    "start",
    "stop",
    "trial_runner",
    "trial_state",
]
