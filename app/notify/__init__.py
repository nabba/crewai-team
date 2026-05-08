"""
app.notify — completion notifications fanned out to Signal + Web Push.

Two public surfaces:

  - ``notify(title, body, url=...)`` — fire-and-forget delivery to every
    subscribed channel. Used by ad-hoc callers (e.g. a long-running
    coding session that finishes asynchronously).

  - ``@notify_on_complete(label=..., notify_on_failure_only=False)`` —
    decorator for scheduled jobs. Wraps a sync OR async callable; on
    completion, builds a one-line "✓ <label> done (Xs)" message (or
    "✗ <label> failed: ..." on exception) and dispatches via ``notify``.

Default delivery: Signal direct message to the configured owner +
Web Push fan-out to every registered PWA device. Web Push is silent
when VAPID keys aren't configured (Phase 4).

Phase 7 contract: every TRIGGERED task in the system should ping back
on completion. Apply the decorator at scheduler-registration time so
the wrapped function stays clean.
"""
from __future__ import annotations

from app.notify.api import notify, notify_on_complete

__all__ = ["notify", "notify_on_complete"]
