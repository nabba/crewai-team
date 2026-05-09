"""Append-only audit storage with hash-chain integrity, plus
structured per-event log helpers used across the codebase.

Two surfaces live here:

  1. Legacy event-log helpers (originally ``app/audit.py`` —
     merged into the package on 2026-05-10 to fix the shadow-bug
     described below):
         log_request_received, log_response_sent, log_crew_dispatch,
         log_tool_call, log_tool_blocked, log_security_event

  2. Hash-chain audit storage primitive (this package's original
     reason to exist):
         RolledLogStore / RolledLogReader / RolledLogVerifier
         GENESIS / LEGACY_PREFIX / SegmentInfo / VerificationResult

  The :mod:`rolled_log` submodule exposes ``RolledLogStore``,
  ``RolledLogReader``, ``RolledLogVerifier`` — a decade-class durable
  storage primitive for JSONL audit logs with size-triggered rotation
  and tamper-evident hash chains across segment boundaries.

  The :mod:`migration` submodule converts legacy single-file logs
  (JSON list-of-dicts or plain JSONL) into rolled-segment form,
  preserving forensic continuity via a sentinel boundary entry.

Pre-2026-05-10 history. The module ``app/audit.py`` lived alongside
this package directory. Python prefers packages over modules with
the same name, so ``from app.audit import log_tool_blocked`` failed
at module-load time. Symptom: gateway crashloop on startup
(``ImportError: cannot import name 'log_tool_blocked' from 'app.audit'``)
because ``app/tools/attachment_reader.py`` had a top-level
``from app.audit import log_tool_blocked``. The fix consolidates
the legacy event-log helpers into this ``__init__.py`` and deletes
the shadowed ``app/audit.py``. See sibling fix in ``app/utils/`` for
the same pattern.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.audit.rolled_log import (
    GENESIS,
    LEGACY_PREFIX,
    RolledLogReader,
    RolledLogStore,
    RolledLogVerifier,
    SegmentInfo,
    VerificationResult,
)

# Dedicated audit logger — configure its handler in main.py
audit_logger = logging.getLogger("crewai.audit")


# ── Legacy event-log helpers (was app/audit.py) ──────────────────────────────


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
    try:
        from app.trace import get_trace_id
        _tid = get_trace_id()
    except Exception:
        _tid = ""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": _tid,
        "event": event_type,
        **data,
    }
    audit_logger.info(json.dumps(record))


__all__ = [
    # Hash-chain primitive (original package surface)
    "GENESIS",
    "LEGACY_PREFIX",
    "RolledLogReader",
    "RolledLogStore",
    "RolledLogVerifier",
    "SegmentInfo",
    "VerificationResult",
    # Legacy event-log helpers (was app/audit.py)
    "audit_logger",
    "log_request_received",
    "log_response_sent",
    "log_crew_dispatch",
    "log_tool_call",
    "log_tool_blocked",
    "log_security_event",
]
