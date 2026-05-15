"""Signal-send action handler — PROGRAM §46.8 (Q9.5).

The agent proposes a Signal message to a specific recipient; the
operator approves; this handler dispatches it via
:func:`app.signal_client.send_message_blocking` so we have the
Signal timestamp on the apply artifact (useful for follow-up audit
correlations).

Data payload shape::

    {
        "recipient":   "+358401234567" | "andrus:internal" | "discord:1234567890",
        "text":        "...",
        "attachments": ["/path/1", ...],   # optional
    }

The ``recipient`` can be any addressing the existing ``SignalClient``
understands — the same module re-routes ``discord:<id>`` to Discord
DM and ``email:<addr>`` to email (per the cross-channel dispatch
already shipped). This handler doesn't second-guess the routing.
"""
from __future__ import annotations

import logging
from typing import Any

from app.action_requests.handlers.base import ActionHandler, ApplyResult
from app.action_requests.models import ActionType

logger = logging.getLogger(__name__)


_MAX_TEXT_CHARS = 8000   # generous; Signal handles long bodies fine


class SignalSendHandler(ActionHandler):
    @property
    def action_type(self):
        return ActionType.SIGNAL_SEND

    def validate(self, data: dict[str, Any]) -> tuple[bool, str | None]:
        recipient = data.get("recipient")
        if not isinstance(recipient, str) or not recipient.strip():
            return False, "recipient is required and non-empty"
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return False, "text is required and non-empty"
        if len(text) > _MAX_TEXT_CHARS:
            return False, f"text exceeds {_MAX_TEXT_CHARS} chars"
        attachments = data.get("attachments")
        if attachments is not None:
            if not isinstance(attachments, list):
                return False, "attachments must be a list or omitted"
            for a in attachments:
                if not isinstance(a, str):
                    return False, "each attachment must be a path string"
        return True, None

    def apply(self, data: dict[str, Any]) -> ApplyResult:
        try:
            from app.signal_client import send_message_blocking
        except Exception as exc:  # noqa: BLE001
            return ApplyResult(
                ok=False, error=f"signal_client import failed: {exc}",
            )
        recipient = data["recipient"]
        text = data["text"]
        attachments = data.get("attachments") or None
        try:
            ts = send_message_blocking(
                recipient=recipient,
                text=text,
                attachments=attachments,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "signal_send: send_message_blocking raised: %s",
                exc, exc_info=True,
            )
            return ApplyResult(ok=False, error=f"send raised: {exc}")
        if ts is None:
            return ApplyResult(
                ok=False,
                error="send_message_blocking returned None (Signal dispatch failed)",
            )
        return ApplyResult(
            ok=True,
            artifact={
                "recipient": recipient,
                "signal_timestamp": int(ts),
                "char_count": len(text),
            },
        )

    def render_summary(self, data: dict[str, Any]) -> str:
        recipient = (data.get("recipient") or "")[:60]
        text = (data.get("text") or "")
        preview = text[:60].replace("\n", " ")
        if len(text) > 60:
            preview += "…"
        return f"💬 Signal to {recipient}: “{preview}”"
