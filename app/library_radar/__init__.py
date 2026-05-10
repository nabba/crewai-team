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

Trial + canary (sandboxed install + benchmark + auto-score) is a
follow-on layer on top of this primitive — not in this commit.

Master switch: ``LIBRARY_RADAR_ENABLED`` (default ``true``). Daemon
thread eager-starts at import time, mirrors :mod:`app.healing.monitors`.
"""

from app.library_radar.proposer import (
    Discovery,
    run_one_pass,
    start,
    stop,
)

__all__ = [
    "Discovery",
    "run_one_pass",
    "start",
    "stop",
]
