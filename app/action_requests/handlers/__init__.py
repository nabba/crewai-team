"""Per-action-type handler registry.

Each :class:`ActionHandler` provides validation, application, and
summary-rendering for one ``ActionType``. Adding a new type is a
single ``register_handler`` call from a new handler module.
"""
from __future__ import annotations

from app.action_requests.handlers.base import (
    ActionHandler,
    ApplyResult,
    HandlerRegistry,
)

# Module-level singleton registry. Concrete handlers self-register on
# import below.
_REGISTRY = HandlerRegistry()


def register_handler(handler: ActionHandler) -> None:
    """Register a handler. Idempotent — re-registering the same type
    overwrites (last write wins; production callers should not rely
    on this except for tests)."""
    _REGISTRY.register(handler)


def get_handler(action_type) -> ActionHandler | None:
    return _REGISTRY.get(action_type)


def list_action_types() -> list:
    return _REGISTRY.list_types()


# ── Concrete handlers self-register here ────────────────────────────

from app.action_requests.handlers import email_draft  # noqa: E402,F401
register_handler(email_draft.EmailDraftHandler())

# Q9.5 (PROGRAM §46.8, 2026-05-16) — calendar_invite + signal_send.
# Imports are guarded so a broken handler (e.g. google_workspace not
# bootstrapped in a test/dev env) doesn't take down email_draft.
try:
    from app.action_requests.handlers import calendar_invite  # noqa: E402,F401
    register_handler(calendar_invite.CalendarInviteHandler())
except Exception:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).debug(
        "action_requests: calendar_invite handler registration failed",
        exc_info=True,
    )

try:
    from app.action_requests.handlers import signal_send  # noqa: E402,F401
    register_handler(signal_send.SignalSendHandler())
except Exception:  # noqa: BLE001
    import logging
    logging.getLogger(__name__).debug(
        "action_requests: signal_send handler registration failed",
        exc_info=True,
    )


__all__ = [
    "ActionHandler",
    "ApplyResult",
    "HandlerRegistry",
    "get_handler",
    "list_action_types",
    "register_handler",
]
