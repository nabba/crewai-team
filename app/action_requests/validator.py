"""ActionRequest validation — dispatches to the registered handler.

Two layers:

  1. **Structural**: action_type is supported, summary + reason are
     non-empty, requestor is non-empty, data is a dict.
  2. **Type-specific**: handled by ``handler.validate(data)``.

The result distinguishes ``is_valid=False`` (handler rejected the
data) from a structural failure — both lead to status=INVALID, but
the reason differs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.action_requests.handlers import get_handler, list_action_types
from app.action_requests.models import ActionRequest, ActionType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str | None = None


def is_action_type_supported(action_type) -> bool:
    return action_type in list_action_types()


def validate(req: ActionRequest) -> ValidationResult:
    """Validate an ActionRequest. Returns ``ValidationResult``.

    Structural checks first; if they pass, dispatches to the
    type-specific handler.
    """
    if not req.requestor.strip():
        return ValidationResult(ok=False, reason="requestor must be non-empty")
    if not req.summary.strip():
        return ValidationResult(ok=False, reason="summary must be non-empty")
    if not req.reason.strip():
        return ValidationResult(ok=False, reason="reason must be non-empty")
    if not isinstance(req.data, dict):
        return ValidationResult(ok=False, reason="data must be a dict")

    if not is_action_type_supported(req.action_type):
        supported = [t.value for t in list_action_types()]
        return ValidationResult(
            ok=False,
            reason=(
                f"unsupported action_type {req.action_type.value!r}; "
                f"supported: {supported}"
            ),
        )

    handler = get_handler(req.action_type)
    if handler is None:
        return ValidationResult(
            ok=False,
            reason=f"handler not registered for {req.action_type.value!r}",
        )
    ok, reason = handler.validate(req.data)
    if not ok:
        return ValidationResult(
            ok=False,
            reason=f"handler rejected data: {reason}",
        )
    return ValidationResult(ok=True)
