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


__all__ = [
    "ActionHandler",
    "ApplyResult",
    "HandlerRegistry",
    "get_handler",
    "list_action_types",
    "register_handler",
]
