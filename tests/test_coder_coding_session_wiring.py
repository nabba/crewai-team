"""Smoke test for Phase 5.4-e: coder agent gets the coding-session
tools + guidance block in its backstory.

The full ``create_coder()`` factory pulls in LLM credentials, KB
connections, mem0, and other heavy dependencies that aren't
appropriate for a pytest unit. Instead this module checks the
two things 5.4-e actually changes:

  1. The ``_CODING_SESSION_GUIDANCE`` block is appended to
     ``CODER_BACKSTORY`` and visible in the assembled prompt.
  2. The module's tool-build paths reference
     ``create_coding_session_tools`` (i.e., the wiring lines
     weren't deleted by a future refactor).

The end-to-end "coder builds and has the seven tools" smoke is
covered by ``tests/test_capability_e2e.py::TestAgentWiring`` once
the integration test surface picks up #60's symbols.
"""
from __future__ import annotations

import inspect


class TestBackstoryGuidance:

    def test_guidance_block_in_backstory(self) -> None:
        from app.agents.coder import CODER_BACKSTORY

        assert "Coding sessions (Phase 5.4)" in CODER_BACKSTORY
        # Key behavioural directives the operator-side review depends on:
        assert "coding_session_start" in CODER_BACKSTORY
        assert "coding_session_submit" in CODER_BACKSTORY
        assert "coding_session_discard" in CODER_BACKSTORY
        assert "TIER_IMMUTABLE" in CODER_BACKSTORY
        assert "QUOTA_EXCEEDED" in CODER_BACKSTORY

    def test_guidance_distinguishes_request_restricted_write_path(self) -> None:
        """The guidance must teach the agent when NOT to use a coding
        session — the one-shot ``request_restricted_write`` path
        remains the right tool for atomic single-file fixes."""
        from app.agents.coder import CODER_BACKSTORY

        assert "request_restricted_write" in CODER_BACKSTORY


class TestCoderToolWiring:
    """Verifies the tool-wiring lines exist in both code paths
    (legacy + LoadableAgent). We don't actually call ``create_coder``
    — too many heavy deps — but the source-level check catches
    accidental deletion of either line."""

    def test_legacy_path_extends_with_coding_session(self) -> None:
        from app.agents import coder

        src = inspect.getsource(coder._legacy_create_coder)
        assert "create_coding_session_tools" in src
        assert "tools.extend(create_coding_session_tools())" in src

    def test_loadable_path_extends_with_coding_session(self) -> None:
        from app.agents import coder

        src = inspect.getsource(coder._build_loadable_coder)
        assert "create_coding_session_tools" in src
        assert "eager.extend(create_coding_session_tools())" in src

    def test_goal_mentions_coding_session(self) -> None:
        from app.agents import coder

        # Goal string is the same in both factories — checking the
        # legacy one is enough.
        src = inspect.getsource(coder._legacy_create_coder)
        assert "coding_session_start" in src

    def test_create_coding_session_tools_is_optional_group(self) -> None:
        """The wiring uses the existing ``optional_tool_group`` pattern
        so a Phase 5.4 build error doesn't take the whole coder
        offline (graceful degradation is the convention)."""
        from app.agents import coder

        src = inspect.getsource(coder._legacy_create_coder)
        # The tools.extend line must live under the
        # optional_tool_group("coder", "coding_session") block.
        assert (
            'optional_tool_group("coder", "coding_session")'
            in src
        )
