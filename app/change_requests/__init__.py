"""Change-request system — Phase 5.3a.

Agent-callable workflow for writing to restricted paths (e.g.
``app/agents/*.py``) with human approval via Signal 👍/👎 OR
React operator override.

Public API::

    from app.change_requests import (
        create_request, send_ask, approve, reject,
        apply_change, rollback_change,
        get, list_all, find_by_signal_ts,
        Status, DecisionSource,
    )

Architecture (per the post-PIM-incident ultrathink design):

  Agent calls ``request_restricted_write(path, content, reason)``
    → ``create_request(...)``           validates + persists
    → ``send_ask(request_id)``          sends Signal ASK
                                        (also visible in React)
  User 👍 in Signal (or operator approves in React)
    → ``approve(request_id, source=...)``
    → ``apply_change(request_id)``       writes file via bridge,
                                        opens auto-PR against main
  Operator merges PR (gate 2)            durable

  TIER_IMMUTABLE files: rejected at validate() time, never reach
  Signal/React. No human override path.

  Rollback: operator clicks Rollback in React → revert commit pushed
  + hot-revert applied + revert PR opened.

See docs/CHANGE_REQUESTS.md for full reference.
"""
from app.change_requests.lifecycle import (
    approve,
    attach_signal_ts,
    auto_approve,
    create_request,
    mark_applied,
    mark_apply_failed,
    mark_rolled_back,
    mark_timeout,
    reject,
)
from app.change_requests.apply import (
    ApplyResult,
    apply_change,
    rollback_change,
)
from app.change_requests.models import (
    ChangeRequest,
    DecisionSource,
    RiskClass,
    Status,
)
from app.change_requests.signal import (
    build_ask_body,
    find_request_by_signal_ts,
    send_ask,
)
from app.change_requests.store import (
    find_by_signal_ts,
    get,
    list_all,
)
from app.change_requests.validator import (
    is_protected,
    validate,
    validate_auto_apply,
)

__all__ = [
    # models
    "ChangeRequest", "Status", "DecisionSource", "RiskClass",
    # lifecycle
    "create_request", "approve", "auto_approve", "reject", "mark_timeout",
    "mark_applied", "mark_apply_failed", "mark_rolled_back",
    "attach_signal_ts",
    # apply / rollback
    "ApplyResult", "apply_change", "rollback_change",
    # signal
    "send_ask", "find_request_by_signal_ts", "build_ask_body",
    # store
    "get", "list_all", "find_by_signal_ts",
    # validator
    "validate", "validate_auto_apply", "is_protected",
]
