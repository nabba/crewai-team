"""Locks down the per-tool timeout overrides table.

Regression for the 2026-05-09 forest-age-distribution-over-time
failure: ``gee_run_script`` was timing out at the default 180s on
country-scale Earth Engine compute, the agent then thrashed on broken
MCP code-interpreter servers, and the janitor killed the whole task
at 15 minutes idle. Bumping the gee_run_script budget to 600s
(10 min) lets a single Hansen-GFC reduction across 2000-2024 finish
without panic.

This file isn't a comprehensive test of ``tools_timeout.py`` — that
module is well-tested via real tool calls. It just freezes the
per-tool override table so a future "let me clean up that dict"
refactor doesn't silently revert the gee budget.
"""
from __future__ import annotations

import pytest


def test_gee_run_script_has_long_budget():
    """gee_run_script must have an explicit override above 300s.

    Below 300s, country-scale GEE reductions reliably time out (see
    workspace/tool_failure_context.json — multiple "tool 'gee_run_script'
    timed out after 180s" entries from 2026-05).
    """
    from app.tools_timeout import _PER_TOOL_OVERRIDES, _resolve_timeout

    budget = _resolve_timeout("gee_run_script")
    assert "gee_run_script" in _PER_TOOL_OVERRIDES, (
        "gee_run_script must have an explicit timeout override; "
        "the default 180s is too short for serious EE compute."
    )
    assert budget >= 300, (
        f"gee_run_script timeout = {budget}s; expected >= 300s for "
        "real-world country-scale time-series queries"
    )


def test_default_timeout_unchanged():
    """The default budget stays at 180s — most tools are fine with it,
    only the explicit per-tool overrides should be longer."""
    from app.tools_timeout import _DEFAULT_TIMEOUT_SECONDS

    assert _DEFAULT_TIMEOUT_SECONDS == 180


def test_orchestrator_budgets_unchanged():
    """research_orchestrator carries its own internal budget logic;
    the wrapper should give it the full handle_task window."""
    from app.tools_timeout import _resolve_timeout

    assert _resolve_timeout("research_orchestrator") >= 600


def test_memory_tool_budgets_strict():
    """Memory tools should fail fast — a >60s call on Mem0 means
    Neo4j/pgvector is wedged."""
    from app.tools_timeout import _resolve_timeout

    for tool in (
        "recall_facts", "persist_fact", "persist_conversation",
        "team_memory_retrieve",
    ):
        assert _resolve_timeout(tool) <= 60, (
            f"{tool} timeout should be tight (<= 60s); a slower call "
            "means the memory backend is broken"
        )


def test_unknown_tool_uses_default():
    """The fallback path: any tool NOT in the override table uses
    the 180s default."""
    from app.tools_timeout import _DEFAULT_TIMEOUT_SECONDS, _resolve_timeout

    assert (
        _resolve_timeout("a_tool_that_does_not_exist_anywhere")
        == _DEFAULT_TIMEOUT_SECONDS
    )


def test_empty_tool_name_uses_default():
    from app.tools_timeout import _DEFAULT_TIMEOUT_SECONDS, _resolve_timeout

    assert _resolve_timeout("") == _DEFAULT_TIMEOUT_SECONDS
