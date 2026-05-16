"""privacy — yearly data-source audit (PROGRAM §51 Q16 Theme 7).

The system collects increasingly varied operator data over time
(person correlation, browse history, health export, inbox handlers,
calendar / email, voice transcripts). The annual privacy review
enumerates these sources, what they're used for, retention, and
any boundary expansions since the last review.

Public API:
  * ``compose_review(year=None)`` — writes
    ``wiki/privacy/audit_<year>.md`` and returns the path.
  * ``run_once(now=None)`` — idle-job entry point with annual
    cadence guard.
"""
from __future__ import annotations

from app.privacy.annual_review import (
    compose_review,
    run_once,
)

__all__ = ["compose_review", "run_once"]
