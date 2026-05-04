"""Tests for app.agents.commander.routing_overrides — Phase 5.2.

The 2026-05-04 PIM incident: user asked "what is my calendar
tomorrow?", PIM agent had a NameError (missing import); after the
import was patched and gateway restarted, PIM was working — but the
Commander LLM, seeing the prior failure response in conversation
history, kept emitting ``crew=direct`` with a hallucinated response
that "PIM is broken, want me to fix it?" The user got three retries
of the same hallucination instead of their calendar.

This test suite is the regression panel for that pattern. Two layers:

  * ``mark_stale_failures`` (Layer 1) — tags prior failure messages
    in conversation history when ``system_state`` shows the relevant
    crew has succeeded since.
  * ``validate_routing_decision`` (Layer 2) — overrides the
    refusal-as-direct hallucination with a real dispatch.

The headline test is ``TestPIMIncidentReplay`` — feeds the actual
incident's routing decision shape and asserts override fires.
"""
from __future__ import annotations

import pytest


# ── Layer 2: detect_refusal_pattern ──────────────────────────────────


class TestDetectRefusalPattern:

    def test_pim_is_currently_broken_detected(self):
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        text = (
            "The PIM crew is currently broken (a code error: optional_tool_group "
            "is not defined). I can't fetch your calendar until this is fixed."
        )
        assert detect_refusal_pattern(text) == "pim"

    def test_short_form_detected(self):
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        assert detect_refusal_pattern("The pim crew is broken") == "pim"

    def test_coding_crew_failed_detected(self):
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        text = "The coding crew failed last time. Want me to debug?"
        assert detect_refusal_pattern(text) == "coding"

    def test_no_refusal_markers_returns_none(self):
        """Legitimate non-dispatch responses pass through."""
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        assert detect_refusal_pattern("It is 2:42 PM in Helsinki.") is None
        assert detect_refusal_pattern("I'd need more details to help.") is None

    def test_refusal_without_crew_returns_none(self):
        """Refusal markers but no specific crew name → not the pattern."""
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        # "is broken" is in markers but no valid crew mentioned.
        assert detect_refusal_pattern("Something is broken on my end.") is None

    def test_empty_input(self):
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        assert detect_refusal_pattern("") is None
        assert detect_refusal_pattern(None) is None

    def test_partial_word_does_not_match(self):
        """Word boundary check — 'composing' doesn't match 'pim'."""
        from app.agents.commander.routing_overrides import detect_refusal_pattern
        # "is broken" markers BUT no whole-word crew name
        text = "The composing system is broken."  # "composing" is not "coding"
        assert detect_refusal_pattern(text) is None


# ── Layer 2: validate_routing_decision ──────────────────────────────


class TestValidateRoutingDecision:

    def test_pim_incident_replay(self):
        """The headline regression. Feeds the actual decision the
        Commander emitted on 2026-05-04 and asserts override fires."""
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        incident_decision = [{
            "crew": "direct",
            "task": (
                "The PIM crew is currently broken (a code error: "
                "optional_tool_group is not defined). I can't fetch your "
                "calendar until this is fixed. Please ask me to debug "
                "and fix the PIM crew."
            ),
            "difficulty": 1,
        }]
        fixed = validate_routing_decision(
            incident_decision,
            user_input="what is my calendar tomorrow?",
        )
        assert len(fixed) == 1
        assert fixed[0]["crew"] == "pim"
        assert fixed[0]["task"] == "what is my calendar tomorrow?"
        assert fixed[0]["difficulty"] == 1

    def test_normal_direct_passes_through(self):
        """A direct response that's NOT a refusal stays unchanged."""
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        normal = [{
            "crew": "direct",
            "task": "It is 2:42 PM in Helsinki.",
            "difficulty": 1,
        }]
        result = validate_routing_decision(normal, user_input="what time is it?")
        assert result == normal

    def test_real_crew_dispatch_passes_through(self):
        """When the LLM correctly routes to a crew, we don't touch it."""
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        good = [{
            "crew": "pim",
            "task": "List my calendar events for tomorrow.",
            "difficulty": 2,
        }]
        result = validate_routing_decision(good, user_input="calendar tomorrow")
        assert result == good

    def test_multi_decision_override_only_affects_refusal(self):
        """Mixed list — only the refusal-shaped direct gets overridden."""
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        decisions = [
            {"crew": "direct", "task": "PIM is broken, debug it?", "difficulty": 1},
            {"crew": "research", "task": "Look up X", "difficulty": 5},
            {"crew": "direct", "task": "Sure, I'll do that.", "difficulty": 1},
        ]
        result = validate_routing_decision(decisions, user_input="orig input")
        assert result[0]["crew"] == "pim"
        assert result[0]["task"] == "orig input"
        assert result[1] == decisions[1]
        assert result[2] == decisions[2]


# ── Layer 1: mark_stale_failures ────────────────────────────────────


class TestMarkStaleFailures:

    def test_no_state_passes_through(self):
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = "Crew pim failed: NameError\nUser: try again"
        assert mark_stale_failures(history, system_state=None) == history

    def test_no_successful_runs_passes_through(self):
        """Conservative behavior — if no crew has succeeded since
        gateway start, the failure may be live; don't tag stale."""
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = "Crew pim failed: NameError\nUser: try again"
        state = {
            "recent_crew_runs": {
                "available": True,
                "by_crew": {"pim": [{"ts": "...", "ok": False}]},
            },
        }
        assert mark_stale_failures(history, system_state=state) == history

    def test_successful_run_tags_failures(self):
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = (
            "Assistant: Crew pim failed: NameError: optional_tool_group is not defined\n"
            "User: try again"
        )
        state = {
            "recent_crew_runs": {
                "available": True,
                "by_crew": {"pim": [{"ts": "2026-05-04T16:00", "ok": True}]},
            },
        }
        result = mark_stale_failures(history, system_state=state)
        assert "[PRIOR — LIKELY RESOLVED" in result
        assert "['pim']" in result
        # Non-failure lines unchanged
        assert "User: try again" in result

    def test_buffer_unavailable_passes_through(self):
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = "Crew pim failed: NameError"
        state = {"recent_crew_runs": {"available": False}}
        assert mark_stale_failures(history, system_state=state) == history

    def test_traceback_pattern_tagged(self):
        """Stack traces are also failure markers."""
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = (
            "Assistant: Traceback (most recent call last): ...\n"
            "User: hmm"
        )
        state = {
            "recent_crew_runs": {
                "available": True,
                "by_crew": {"coding": [{"ts": "...", "ok": True}]},
            },
        }
        result = mark_stale_failures(history, system_state=state)
        assert "[PRIOR — LIKELY RESOLVED" in result
        assert "User: hmm" in result  # non-failure line unchanged

    def test_empty_history(self):
        from app.agents.commander.routing_overrides import mark_stale_failures
        state = {"recent_crew_runs": {"available": True, "by_crew": {}}}
        assert mark_stale_failures("", system_state=state) == ""


# ── End-to-end: full PIM-incident scenario ──────────────────────────


class TestPIMIncidentEndToEnd:
    """Replay the full incident flow with both layers active.

    Setup mirrors the actual 2026-05-04 timeline:
      1. PIM has a NameError; first attempt fails.
      2. Bug is patched + gateway restarted.
      3. PIM has succeeded once since restart (telemetry shows it).
      4. User retries calendar question.
      5. Expected: routing dispatches to PIM (not refusal-as-direct).

    Layer 1 tags the failure as stale; Layer 2 catches any residual
    hallucination if the LLM still emits one.
    """

    def test_layer_1_alone_marks_history(self):
        """Layer 1: history sanitation reduces refusal propensity."""
        from app.agents.commander.routing_overrides import mark_stale_failures
        history = (
            "User: what is my calendar tomorrow?\n"
            "Assistant: Crew pim failed: name 'optional_tool_group' is not defined\n"
            "User: what is my calendar tomorrow?"
        )
        state = {
            "recent_crew_runs": {
                "available": True,
                "by_crew": {
                    "pim": [
                        {"ts": "2026-05-04T16:09", "ok": True, "duration_s": 1.8},
                    ],
                },
            },
        }
        result = mark_stale_failures(history, system_state=state)
        # The failure line is tagged
        assert "[PRIOR — LIKELY RESOLVED" in result
        # Both user messages survive
        assert result.count("what is my calendar tomorrow?") == 2

    def test_layer_2_catches_residual_hallucination(self):
        """Layer 2: even if the LLM still emits the refusal (e.g.
        Layer 1 didn't apply because no successful runs yet), the
        validator overrides it."""
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        # Simulate the LLM being stubborn — emits the refusal anyway.
        residual_hallucination = [{
            "crew": "direct",
            "task": (
                "The PIM crew is still broken — the optional_tool_group "
                "import is not defined. Want me to debug and fix it?"
            ),
            "difficulty": 1,
        }]
        result = validate_routing_decision(
            residual_hallucination,
            user_input="what is my calendar tomorrow?",
        )
        # Override fires regardless of system_state (Layer 2 is unconditional
        # once the refusal pattern is detected — the actual dispatch will
        # surface the real error or success)
        assert result[0]["crew"] == "pim"
        assert result[0]["task"] == "what is my calendar tomorrow?"

    def test_telemetry_logging_includes_recent_success(self, caplog):
        """When system_state shows recent success, the override
        warning includes that context for operator visibility."""
        import logging
        from app.agents.commander.routing_overrides import (
            validate_routing_decision,
        )
        decision = [{
            "crew": "direct",
            "task": "PIM is broken, want me to debug?",
            "difficulty": 1,
        }]
        state = {
            "recent_crew_runs": {
                "available": True,
                "by_crew": {
                    "pim": [{"ts": "2026-05-04T16:09", "ok": True}],
                },
            },
        }
        with caplog.at_level(logging.WARNING, logger="app.agents.commander.routing_overrides"):
            validate_routing_decision(decision, "calendar?", system_state=state)
        # Should have logged at WARNING with the override context
        assert any(
            "refusal-as-direct" in r.getMessage() and "pim" in r.getMessage()
            for r in caplog.records
        )
