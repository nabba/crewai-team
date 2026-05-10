"""Action-request primitive — operator-gated NON-CODE actions.

Where :mod:`app.change_requests` mediates agent-proposed *file edits*
(with TIER_IMMUTABLE refusal + git/PR flow), this package mediates
agent-proposed *external actions* — email drafts, calendar invites,
Slack messages, anything where an action affects the world OUTSIDE
the repo. The operator approval surface is the same (Signal 👍 /
React); the underlying lifecycle, validator, and applier are
type-specific.

Design choice (deliberate): action_requests is a *parallel* primitive
to change_requests, not a generalization that subsumes it. The two
systems share the operator's mental model (both gate via 👍/👎) but
keep their internals separate so neither evolves at the speed of the
other. A future commit could lift the shared shape into a base; for
now, parallel is simpler and safer.

Action types are pluggable via a handler registry. Each handler
provides:

  ``validate(data) -> ValidationResult``
      Type-specific validation (e.g. email recipient parsing, calendar
      timezone check).
  ``apply(data) -> ApplyResult``
      Side-effect: send the email, create the calendar event, etc.
  ``render_summary(data) -> str``
      For the Signal ASK message body and React UI.

This commit ships the primitive plus one concrete type
(``email_draft``). Adding new types is one handler module plus one
registration call — no lifecycle / store / signal changes.

State machine::

    PENDING ─┬─→ APPROVED ──→ APPLIED
             │             ╲
             ├─→ REJECTED    ╲─→ APPLY_FAILED
             ├─→ INVALID
             └─→ TIMEOUT

Default-OFF master switch ``ACTION_REQUESTS_ENABLED`` so the operator
opts into this surface.
"""

from app.action_requests.handlers import (
    ActionHandler,
    ApplyResult,
    HandlerRegistry,
    get_handler,
    list_action_types,
)
from app.action_requests.lifecycle import (
    InvalidActionTransition,
    apply,
    approve,
    create_request,
    expire,
    reject,
)
from app.action_requests.models import (
    ActionRequest,
    ActionStatus,
    ActionType,
    DecisionSource,
)
from app.action_requests.store import (
    find_by_signal_ts,
    get,
    list_all,
    reset_for_tests,
)
from app.action_requests.validator import (
    ValidationResult,
    is_action_type_supported,
    validate,
)

__all__ = [
    "ActionHandler",
    "ActionRequest",
    "ActionStatus",
    "ActionType",
    "ApplyResult",
    "DecisionSource",
    "HandlerRegistry",
    "InvalidActionTransition",
    "ValidationResult",
    "apply",
    "approve",
    "create_request",
    "expire",
    "find_by_signal_ts",
    "get",
    "get_handler",
    "is_action_type_supported",
    "list_action_types",
    "list_all",
    "reject",
    "reset_for_tests",
    "validate",
]
