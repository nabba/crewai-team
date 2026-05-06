"""Signal voting for change requests.

Sends an ASK message to the configured owner with the diff and a
👍/👎 prompt; records the message timestamp on the request so the
reaction handler can correlate.

Reaction-handling lives in ``app/main.py`` (Signal endpoint
``/signal/inbound``). When the user reacts on a message timestamp
that matches a pending change request, the handler dispatches to
``app.change_requests.lifecycle.approve()`` or ``reject()``.

The message body is bounded — full diff in Signal can exceed
practical message size. Truncation policy:
  * Header: path + reason (always full)
  * Diff: ≤ 2000 chars; if longer, truncate with a "see React" pointer
  * Footer: 👍/👎 prompt + change-request id
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


_MAX_DIFF_CHARS = 2000
_MAX_REASON_CHARS = 600


def build_ask_body(
    *,
    request_id: str,
    requestor: str,
    path: str,
    reason: str,
    diff: str,
) -> str:
    """Compose the Signal message body for an ASK."""
    short_reason = reason[:_MAX_REASON_CHARS]
    truncated_reason = "" if len(reason) <= _MAX_REASON_CHARS else "…\n[reason truncated]"

    if len(diff) <= _MAX_DIFF_CHARS:
        diff_block = diff
        diff_note = ""
    else:
        diff_block = diff[:_MAX_DIFF_CHARS]
        diff_note = (
            f"\n\n[diff truncated — see /api/cp/changes/{request_id} "
            f"for the full unified diff]"
        )

    body = (
        f"🔧 CHANGE REQUEST · {path}\n\n"
        f"From: {requestor}\n"
        f"Reason: {short_reason}{truncated_reason}\n\n"
        f"```diff\n{diff_block}\n```{diff_note}\n\n"
        f"👍 to approve and apply  ·  👎 to reject\n"
        f"id: {request_id}"
    )
    return body


def send_ask(request_id: str) -> int | None:
    """Look up the request, compose the body, send via Signal,
    record the message timestamp on the request.

    Returns the Signal message timestamp on success, None on failure.
    The caller (lifecycle / tool) decides what to do on None — typically
    log + leave the request in PENDING for manual operator action via
    React.
    """
    try:
        from app.change_requests import lifecycle, store
        from app.config import get_settings
        from app.signal_client import send_message_blocking
    except Exception as exc:
        logger.warning("change_requests.signal: imports failed: %s", exc)
        return None

    cr = store.get(request_id)
    if cr is None:
        logger.warning("change_requests.signal: request %r not found", request_id)
        return None

    settings = get_settings()
    recipient = (getattr(settings, "signal_owner_number", "") or "").strip()
    if not recipient:
        logger.warning(
            "change_requests.signal: SIGNAL_OWNER_NUMBER not set; "
            "ASK not sent for request %s. Operator must approve via React.",
            request_id,
        )
        return None

    body = build_ask_body(
        request_id=cr.id,
        requestor=cr.requestor,
        path=cr.path,
        reason=cr.reason,
        diff=cr.diff,
    )
    try:
        ts = send_message_blocking(recipient, body)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "change_requests.signal: send_message_blocking raised "
            "for %s: %s", request_id, exc,
        )
        return None
    if ts is None:
        logger.warning(
            "change_requests.signal: send_message_blocking returned None "
            "for %s", request_id,
        )
        return None

    try:
        lifecycle.attach_signal_ts(request_id, ts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "change_requests.signal: attach_signal_ts raised for %s: %s",
            request_id, exc,
        )
        # The message went out; the request just lacks the ts on record.
        # The reaction handler won't be able to correlate, but operator
        # can still approve via React.
    return ts


def find_request_by_signal_ts(signal_ts: int) -> str | None:
    """Resolver used by the Signal reaction handler in main.py.
    Returns the request id, or None if no pending request matches."""
    try:
        from app.change_requests import store
    except Exception:
        return None
    return store.find_by_signal_ts(signal_ts)
