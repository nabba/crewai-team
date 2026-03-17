"""
Structured audit logging for all agent actions.
Writes to a dedicated audit log so security events can be monitored
independently of general application logs.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Any


# Dedicated audit logger — configure its handler in main.py
audit_logger = logging.getLogger("crewai.audit")


def _redact(value: str, max_len: int = 200) -> str:
    """Truncate long values and strip potential secrets from log output."""
    if not isinstance(value, str):
        value = str(value)
    value = value[:max_len]
    return value


def log_request_received(sender_redacted: str, message_length: int) -> None:
    """Log an inbound Signal message (sender already redacted by security.py)."""
    _emit("request_received", {
        "sender": sender_redacted,
        "message_length": message_length,
    })


def log_crew_dispatch(crew: str, input_preview: str) -> None:
    """Log when a crew is dispatched with user input."""
    _emit("crew_dispatch", {
        "crew": crew,
        "input_preview": _redact(input_preview, 100),
    })


def log_tool_call(tool: str, agent: str, input_preview: str) -> None:
    """Log when an agent calls a tool."""
    _emit("tool_call", {
        "tool": tool,
        "agent": agent,
        "input_preview": _redact(input_preview, 100),
    })


def log_tool_blocked(tool: str, agent: str, reason: str) -> None:
    """Log when a tool call is blocked (SSRF, path traversal, etc.)."""
    _emit("tool_blocked", {
        "tool": tool,
        "agent": agent,
        "reason": reason,
    })


def log_security_event(event: str, detail: str) -> None:
    """Log a security-relevant event (auth failure, rate limit, injection attempt)."""
    _emit("security_event", {
        "event": event,
        "detail": _redact(detail, 200),
    })


def log_response_sent(sender_redacted: str, response_length: int) -> None:
    """Log when a response is sent back to the owner."""
    _emit("response_sent", {
        "sender": sender_redacted,
        "response_length": response_length,
    })


def _emit(event_type: str, data: dict[str, Any]) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    audit_logger.info(json.dumps(record))
