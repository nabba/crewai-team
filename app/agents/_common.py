"""
agents._common — shared helpers for agent factory functions.

Phase E1 of the remediation plan.

Today every agent (coder, researcher, writer, critic, observer, ...) has
the same shape::

    try:
        from app.tensions.tools import get_tension_tools
        tools.extend(get_tension_tools("coder"))
    except Exception:
        pass

A typo, a config error, or a missing dep silently strips a third of an
agent's tools. Nothing logs. Debugging "why did the coder forget the
file_manager tool?" requires reading every except clause to figure out
which group fell on the floor.

This module provides one tiny helper — :func:`optional_tool_group` — that
swaps ``except: pass`` for categorized logging:

    * ``ModuleNotFoundError`` → debug log (expected — feature not enabled
      in this build, e.g. tensions package not installed).
    * Any other ``Exception`` → warning log with traceback (real bug in
      the tool factory or a runtime error).

Behaviour-wise this is a strict superset of ``try: ... except: pass`` —
nothing that succeeded before will fail now, and nothing that silently
failed will succeed now. The only change is observability.

Usage::

    from app.agents._common import optional_tool_group

    with optional_tool_group("coder", "tensions"):
        from app.tensions.tools import get_tension_tools
        tools.extend(get_tension_tools("coder"))

The helper is intentionally minimal — no new dependencies, no DSL, no
auto-loading of tool groups. Agents stay declarative; only the failure
mode changes.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def optional_tool_group(role: str, group: str) -> Iterator[None]:
    """Wrap an optional tool-group import + load with diagnostic logging.

    Catches:
        ModuleNotFoundError → debug-level log (feature off / not installed).
        Exception           → warning-level log with traceback (real fault).

    The block runs to completion if the import + extend succeed. On
    failure, the partially-built ``tools`` list in the caller's scope is
    preserved up to the point of failure — same as the prior
    ``try: ... except: pass`` pattern.
    """
    try:
        yield
    except ModuleNotFoundError as e:
        # Distinct, low-severity: tool group simply isn't installed.
        logger.debug(
            "agent %s: tool group %r unavailable (%s)",
            role, group, e,
        )
    except Exception as e:  # noqa: BLE001 — deliberate broad catch
        # Real fault — bug in factory, runtime error, etc. Log with
        # traceback so it can be investigated; do not propagate.
        logger.warning(
            "agent %s: tool group %r failed to load: %s",
            role, group, e,
            exc_info=True,
        )
