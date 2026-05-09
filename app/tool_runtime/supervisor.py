"""tool_runtime/supervisor.py — mid-iteration tool failure repair.

Wraps tool callables passed to CrewAI's ``_handle_native_tool_calls``.
When a wrapped tool raises:

  1. Classify the exception (rate_limit | auth | network | timeout |
     schema | unknown).
  2. For transient classes (rate_limit, network, timeout) retry with
     exponential backoff, up to ``TOOL_SUPERVISOR_MAX_RETRIES``.
  3. If still failing AND the registry has alternative tools with
     overlapping capabilities, swap to the cheapest alternative and
     try once.
  4. If everything fails, return a structured tool-result *string*
     (NOT raise) so the agent's next-thought step sees a usable
     observation, not an exception. This preserves CrewAI loop
     semantics — same shape as the recovery loop's "soft-fail with
     surfaced diagnostic" philosophy.

Off by default — set ``TOOL_SUPERVISOR_ENABLED=true`` to opt in.

Composes with ``app/recovery/loop.py`` — the supervisor handles
in-iteration tool failures (raised exceptions, mid-dispatch); the
recovery loop handles refusal-shaped final answers (post-vetting).
Both can fire on the same task without overlap.

Audit trail: ``actor='tool_supervisor'``, actions ``invocation.failed
| invocation.retried | invocation.substituted | invocation.gave_up``.
Query via ``/api/cp/audit?actor=tool_supervisor``.
"""
from __future__ import annotations

import logging
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    """Master switch. Default off, per the Recovery Loop precedent.

    Runtime-settings wins on a live system (so the React /cp/settings
    toggle takes effect without restart). Env var is the test /
    degraded-boot fallback.
    """
    try:
        from app.runtime_settings import get_tool_supervisor_enabled
        return bool(get_tool_supervisor_enabled())
    except Exception:
        return os.getenv("TOOL_SUPERVISOR_ENABLED", "false").lower() in ("true", "1", "yes")


def _max_retries() -> int:
    try:
        return max(0, min(5, int(os.getenv("TOOL_SUPERVISOR_MAX_RETRIES", "2"))))
    except ValueError:
        return 2


def _backoff_ms() -> int:
    try:
        return max(0, min(60_000, int(os.getenv("TOOL_SUPERVISOR_BACKOFF_MS", "500"))))
    except ValueError:
        return 500


# Categories whose cheapest remediation is "wait a beat and try again."
_TRANSIENT = ("rate_limit", "network", "timeout")

# Class-name fragments we can match without inspecting the message text.
_NAME_FRAGMENTS = ("rate_limit", "network", "timeout", "auth", "schema")

# Best-effort message regexes per category. First match wins.
_MESSAGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("rate_limit", re.compile(
        r"rate.?limit|\b429\b|too many requests|quota exceeded", re.I)),
    ("auth", re.compile(
        r"\b401\b|\b403\b|unauthori[sz]ed|forbidden|invalid api key|"
        r"missing credentials|authentication failed", re.I)),
    ("network", re.compile(
        r"connection (refused|reset|aborted)|name (or service )?not known|"
        r"network is unreachable|temporary failure|\bdns\b", re.I)),
    ("timeout", re.compile(
        r"timed?[\- ]?out|deadline exceeded|read timeout|operation timed", re.I)),
    ("schema", re.compile(
        r"validation error|invalid (argument|input)|missing (required )?field|"
        r"expected .* got|pydantic", re.I)),
]


def classify_exception(exc: BaseException) -> str:
    """Bucket an exception. Falls back to ``"unknown"``."""
    name = type(exc).__name__.lower()
    msg = str(exc)

    for frag in _NAME_FRAGMENTS:
        if frag in name or frag.replace("_", "") in name:
            return frag

    for category, pat in _MESSAGE_PATTERNS:
        if pat.search(msg):
            return category

    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "network"

    return "unknown"


def _is_transient(category: str) -> bool:
    return category in _TRANSIENT


# Mid-substitute calls run unsupervised — mirrors `recovery_loop._in_recovery`.
_in_substitute: ContextVar[bool] = ContextVar(
    "tool_supervisor_in_substitute", default=False
)


def _audit(action: str, **detail: Any) -> None:
    """Best-effort audit. Never raises."""
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(actor="tool_supervisor", action=action, detail=detail)
    except Exception:
        logger.debug("tool_supervisor: audit write failed", exc_info=True)


@dataclass
class _Substitute:
    name: str
    callable: Callable[..., Any]


def _find_substitute(failed_name: str) -> _Substitute | None:
    """Return the first registry alternative with overlapping capabilities.

    Returns None if the failed tool isn't in the registry, has no
    capabilities declared, or no other tool covers any of them.
    """
    try:
        from app.tool_registry.registry import ToolRegistry
        reg = ToolRegistry.instance()
        spec = reg.get(failed_name)
        if spec is None or not spec.capabilities:
            return None

        candidates = reg.filter(
            capabilities=spec.capabilities,
            tier_at_most=spec.tier,
            loadable_only=True,
        )
        for cand in candidates:
            if cand.name == failed_name:
                continue
            try:
                instance = reg.build_instance(cand.name)
                run = getattr(instance, "_run", None) or getattr(instance, "run", None)
                if callable(run):
                    return _Substitute(name=cand.name, callable=run)
                if callable(instance):
                    return _Substitute(name=cand.name, callable=instance)
            except Exception:
                continue
        return None
    except Exception:
        logger.debug("tool_supervisor: substitute lookup failed", exc_info=True)
        return None


def _format_failure(
    name: str, category: str, errors: list[str], substitute: str | None
) -> str:
    """Structured tool-result the agent sees instead of an exception."""
    lines = [
        f"[tool-supervisor] tool '{name}' failed after {len(errors)} attempt(s).",
        f"Failure class: {category}.",
    ]
    if substitute:
        lines.append(f"Tried alternative '{substitute}' — also failed.")
    if errors:
        lines.append(f"Last error: {errors[-1][:200]}")
    lines.append(
        "Action for the agent: try a different approach or surface this "
        "limitation to the user. Do not retry the same tool with the same args."
    )
    return "\n".join(lines)


def wrap_tool_function(
    name: str, fn: Callable[..., Any]
) -> Callable[..., Any]:
    """Return a wrapped version of ``fn`` that supervises failures.

    No-op if the supervisor is disabled — returns the original
    callable unchanged so there's zero overhead in the off path.
    """
    if not is_enabled():
        return fn

    max_retries = _max_retries()
    backoff_s = _backoff_ms() / 1000.0

    def supervised(*args: Any, **kwargs: Any) -> Any:
        # Recursion guard: substitute calls run unsupervised.
        if _in_substitute.get():
            return fn(*args, **kwargs)

        errors: list[str] = []
        last_category = "unknown"

        for attempt in range(max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                if attempt > 0:
                    _audit(
                        "invocation.retried",
                        tool=name,
                        attempts=attempt + 1,
                        success=True,
                        category=last_category,
                    )
                return result
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException as exc:
                category = classify_exception(exc)
                last_category = category
                err_str = f"{type(exc).__name__}: {str(exc)[:300]}"
                errors.append(err_str)
                _audit(
                    "invocation.failed",
                    tool=name,
                    attempt=attempt + 1,
                    category=category,
                    error=err_str,
                )
                if attempt < max_retries and _is_transient(category):
                    time.sleep(backoff_s * (2 ** attempt))
                    continue
                break

        # Substitute path.
        sub = _find_substitute(name)
        if sub is not None:
            token = _in_substitute.set(True)
            try:
                result = sub.callable(*args, **kwargs)
                _audit(
                    "invocation.substituted",
                    failed_tool=name,
                    substitute=sub.name,
                    success=True,
                    category=last_category,
                )
                return result
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except BaseException as sub_exc:
                sub_err = f"{type(sub_exc).__name__}: {str(sub_exc)[:300]}"
                errors.append(f"[substitute {sub.name}] {sub_err}")
                _audit(
                    "invocation.substituted",
                    failed_tool=name,
                    substitute=sub.name,
                    success=False,
                    category=classify_exception(sub_exc),
                    error=sub_err,
                )
            finally:
                _in_substitute.reset(token)

        # Give up — return a structured observation, not an exception.
        _audit(
            "invocation.gave_up",
            tool=name,
            attempts=len(errors),
            category=last_category,
            substitute=sub.name if sub else None,
        )
        return _format_failure(
            name=name,
            category=last_category,
            errors=errors,
            substitute=sub.name if sub else None,
        )

    supervised.__name__ = getattr(fn, "__name__", name)
    supervised.__qualname__ = getattr(fn, "__qualname__", name)
    return supervised


def supervise_available_functions(
    funcs: dict[str, Callable[..., Any]]
) -> dict[str, Callable[..., Any]]:
    """Wrap every entry of an ``available_functions`` dict.

    Convenience used by ``LoadableAgentExecutor`` at each schema
    re-render. No-op when the supervisor is disabled.
    """
    if not is_enabled():
        return funcs
    return {n: wrap_tool_function(n, fn) for n, fn in funcs.items()}
