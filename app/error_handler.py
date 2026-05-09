"""
error_handler.py — Centralized error handling and structured logging.

Provides:
  - ErrorCategory enum for classifying errors
  - report_error() for structured error logging to JSONL
  - safe_execute() context manager for consistent error handling
  - setup_structured_logging() to add file-based JSON handler

Usage:
    from app.error_handler import safe_execute, ErrorCategory

    with safe_execute("load_skills", category=ErrorCategory.DATA):
        skills = load_from_chromadb()

    # Or for fire-and-forget operations:
    with safe_execute("telemetry_write", log_level="debug"):
        store_metrics()
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Error Categories ────────────────────────────────────────────────────────

class ErrorCategory(str, Enum):
    TRANSIENT = "transient"  # Network, timeout, overloaded — likely recovers
    DATA = "data"            # Parse, corruption, dimension mismatch
    SYSTEM = "system"        # OOM, disk full, process errors
    LOGIC = "logic"          # Assertion failures, invariant violations

# ── Thread-safe Error Counters ──────────────────────────────────────────────

_counters: dict[str, int] = {}
_counter_lock = threading.Lock()

def get_error_counts() -> dict[str, int]:
    """Return a copy of error counters (for dashboard reporting)."""
    with _counter_lock:
        return dict(_counters)

def reset_error_counts() -> None:
    with _counter_lock:
        _counters.clear()

# ── Structured Error Reporting ──────────────────────────────────────────────

_structured_log_path: Path | None = None

def setup_structured_logging(
    log_path: str = "/app/workspace/logs/errors.jsonl",
    max_mb: int = 50,
) -> None:
    """Add a rotating file handler for structured JSON error logs.

    Call once at startup. Errors are written as one JSON object per line.
    """
    global _structured_log_path
    _structured_log_path = Path(log_path)
    _structured_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add rotating file handler to root logger for ERROR+ level
    handler = RotatingFileHandler(
        str(_structured_log_path),
        maxBytes=max_mb * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.WARNING)

    class JsonFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "lineno": record.lineno,
            }, default=str)

    handler.setFormatter(JsonFormatter())

    # Noise filter — drops a fixed allowlist of known third-party
    # WARNs that we already handle (e.g. discord.py PyNaCl, Anthropic
    # credit-balance 400 caught by CreditAwareAnthropicCompletion,
    # OpenRouter Stealth 502 already excluded by provider blocker).
    # Attached to the JSONL handler ONLY — stdout/docker logs still
    # show every WARN for live debugging.
    try:
        from app.logging_filters import JsonlNoiseFilter
        handler.addFilter(JsonlNoiseFilter())
    except Exception:
        # Filter setup is best-effort — never block error logging.
        logger.debug("noise filter not installed", exc_info=True)

    logging.getLogger().addHandler(handler)
    logger.info(f"Structured logging enabled: {log_path}")

def report_error(
    category: ErrorCategory,
    message: str,
    exc: Exception | None = None,
    context: dict | None = None,
) -> None:
    """Log a structured error and increment counter.

    Args:
        category: Error classification (transient, data, system, logic)
        message: Human-readable error description
        exc: Optional exception object
        context: Optional dict with extra context (agent_id, crew, etc.)
    """
    # Increment counter
    key = f"{category.value}"
    with _counter_lock:
        _counters[key] = _counters.get(key, 0) + 1

    # Build structured log entry (with trace_id for correlation)
    try:
        from app.trace import get_trace_id
        _tid = get_trace_id()
    except Exception:
        _tid = ""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "trace_id": _tid,
        "category": category.value,
        "message": message[:500],
        "exception": f"{type(exc).__name__}: {exc}" if exc else None,
        "context": context,
    }

    # Log at appropriate level
    if category in (ErrorCategory.SYSTEM, ErrorCategory.LOGIC):
        logger.error(f"[{category.value}] {message}", exc_info=exc is not None)
    elif category == ErrorCategory.DATA:
        logger.warning(f"[{category.value}] {message}")
    else:
        logger.debug(f"[{category.value}] {message}")

    # Write to structured log file (if configured)
    if _structured_log_path:
        try:
            from app.safe_io import safe_append
            safe_append(_structured_log_path, json.dumps(entry, default=str))
        except Exception:
            pass  # Logging must never crash the caller

    # ── Fix 1: Invoke ON_ERROR lifecycle hook chain ─────────────────────────
    # Activates failure_classifier (MAST), fault_isolator, healing_knowledge,
    # checkpoint_error hooks. Without this, all 4 hooks remain dormant for
    # operationally-logged errors (which is most of them in this system).
    if not _in_hook_dispatch.value:
        _in_hook_dispatch.value = True
        try:
            from app.lifecycle_hooks import get_registry, HookContext, HookPoint
            ctx = HookContext(
                hook_point=HookPoint.ON_ERROR,
                agent_id=(context or {}).get("agent_id", "") or (context or {}).get("crew", ""),
                task_description=(context or {}).get("task_description", "")[:200],
                errors=[message[:500]] + ([f"{type(exc).__name__}: {exc}"] if exc else []),
                metadata={
                    "category": category.value,
                    "trace_id": _tid,
                    "context": context or {},
                    **{k: v for k, v in (context or {}).items()
                       if isinstance(v, (str, int, float, bool))},
                },
            )
            get_registry().execute(HookPoint.ON_ERROR, ctx)
        except Exception:
            pass  # Hook chain must never crash the caller
        finally:
            _in_hook_dispatch.value = False


# Sentinel to prevent re-entrant hook dispatch (a hook reporting an error
# would cause infinite recursion otherwise).
class _ReentryGuard(threading.local):
    def __init__(self):
        self.value = False

_in_hook_dispatch = _ReentryGuard()


# ── Safe Execute Context Manager ────────────────────────────────────────────

@contextmanager
def safe_execute(
    label: str = "operation",
    category: ErrorCategory = ErrorCategory.TRANSIENT,
    log_level: str = "warning",
    context: dict | None = None,
):
    """Context manager for consistent error handling.

    Usage:
        with safe_execute("load_skills", category=ErrorCategory.DATA):
            skills = chromadb.retrieve("skills", query)

        # Exceptions are caught, logged, and swallowed.
        # Code after the with-block continues normally.
    """
    try:
        yield
    except Exception as exc:
        report_error(category, f"{label}: {exc}", exc=exc, context=context)
