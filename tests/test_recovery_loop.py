"""
Tests for the 2026-04-28 Capability Recovery Loop.

User asked: "if the final answer is something like 'I cannot…' /
'I do not have access…' etc., the system should try different
approaches — look at available crews, tools, skills, find an
alternative route, maybe develop new skills/tools, or write+execute
code. Not just loop a standard LLM refusal."

This test suite pins:
  * Refusal detector — fires on real refusals, ignores false positives
    (long answers that contain "I can't" in passing, policy refusals
    we should respect, etc.)
  * Capability librarian — finds the right alternative crews for
    today's specific failure modes (email → PIM, code execution →
    sandbox, etc.)
  * Strategies — re_route + escalate_tier + forge_queue contract
  * Loop — budget guard, recursion guard, env-flag gate
  * Today's 5 specific regression cases each map to an alternative

Off-by-default — env flag RECOVERY_LOOP_ENABLED gates the loop. Tests
set it explicitly so they're hermetic regardless of the host's env.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests._v2_shim import install_settings_shim
install_settings_shim()


# ══════════════════════════════════════════════════════════════════════
# Refusal detector — conservative
# ══════════════════════════════════════════════════════════════════════

class TestRefusalDetectorFires:
    """Real refusals from today's session must trigger detection."""

    @pytest.fixture(autouse=True)
    def _reset_env(self, monkeypatch):
        monkeypatch.delenv("RECOVERY_DETECTION_THRESHOLD", raising=False)

    def test_no_access_to_email_fires(self):
        """The exact refusal the user saw 5+ times today."""
        from app.recovery.refusal_detector import detect_refusal
        text = (
            "I do not have access to your email account. To retrieve "
            "and rank your emails from today, I need you to authorize "
            "a connection to your email provider."
        )
        sig = detect_refusal(text)
        assert sig is not None, "missing_tool refusal must fire"
        assert sig.category == "missing_tool"
        assert sig.confidence >= 0.80

    def test_unavailable_in_environment_fires(self):
        """The coding-crew-dumps-script regression."""
        from app.recovery.refusal_detector import detect_refusal
        text = "EXECUTION OUTPUT: <unavailable in this environment: no connected execution tool / MCP server to run python and capture real stdout>"
        sig = detect_refusal(text)
        assert sig is not None
        assert sig.category == "missing_tool"

    def test_sorry_had_trouble_fires(self):
        from app.recovery.refusal_detector import detect_refusal
        text = "Sorry, I had trouble understanding that request. Please try again."
        sig = detect_refusal(text)
        assert sig is not None
        assert sig.category == "generic"


class TestRefusalDetectorIgnoresFalsePositives:
    """Conservative — biased toward NOT firing in ambiguous cases."""

    def test_long_useful_answer_with_passing_mention(self):
        """A 1500-char answer with one casual 'I can't' should not fire."""
        from app.recovery.refusal_detector import detect_refusal
        useful = (
            "Here are the top 10 PSPs operating in CEE. PayU has its "
            "headquarters in Wrocław and processes ~€XX billion/year. "
            "Adyen — Dutch giant, dominant in NL/BE/DE. Stripe — US-led "
            "global, strong in PL/EE/LT. Worldline — French; recent "
            "Ingenico merger gave them broad CEE coverage. " * 8
        ) + " I can't run real-time queries against their internal APIs, but the public data above is current as of last quarter."
        sig = detect_refusal(useful)
        assert sig is None, (
            "A long useful answer with a single passing 'I can't' must "
            "NOT trigger recovery — that would re-do work for nothing."
        )

    def test_policy_refusal_respected(self):
        """If the agent declines on policy grounds we MUST NOT recover."""
        from app.recovery.refusal_detector import detect_refusal
        text = (
            "I cannot help with that — generating malicious code "
            "violates my guidelines."
        )
        sig = detect_refusal(text)
        assert sig is None, (
            "Policy refusals must be respected — recovery would mean "
            "trying to bypass safety, which we never want."
        )

    def test_short_response_below_min_length(self):
        from app.recovery.refusal_detector import detect_refusal
        assert detect_refusal("ok") is None
        assert detect_refusal("") is None
        assert detect_refusal(None) is None  # type: ignore[arg-type]

    def test_threshold_env_override_respected(self, monkeypatch):
        """User can tune sensitivity via env."""
        from app.recovery.refusal_detector import detect_refusal
        text = "I cannot do that exact thing right now."
        # At default 0.8 threshold this scores too low to fire
        monkeypatch.setenv("RECOVERY_DETECTION_THRESHOLD", "0.10")
        assert detect_refusal(text) is not None
        monkeypatch.setenv("RECOVERY_DETECTION_THRESHOLD", "0.99")
        assert detect_refusal(text) is None


# ══════════════════════════════════════════════════════════════════════
# Capability librarian — finds the right alternatives
# ══════════════════════════════════════════════════════════════════════

class TestLibrarianMapsTodaysFailures:
    """Today's specific failures must produce sensible recovery alternatives."""

    def test_email_question_routes_to_pim(self):
        from app.recovery.librarian import find_alternatives
        alts = find_alternatives(
            "what are the most important emails I have received today",
            refusal_category="missing_tool",
            used_crew="research",
        )
        crews = [a.crew for a in alts if a.strategy == "re_route"]
        assert "pim" in crews, (
            "Email questions refused by 'research' crew must produce "
            "a re_route alternative to 'pim'."
        )

    def test_calendar_question_routes_to_pim(self):
        from app.recovery.librarian import find_alternatives
        alts = find_alternatives(
            "what meetings do I have tomorrow",
            refusal_category="missing_tool",
            used_crew="research",
        )
        assert any(a.crew == "pim" and a.strategy == "re_route" for a in alts)

    def test_unknown_task_still_returns_forge_queue(self):
        """Even when no crew alternative exists, forge_queue is always
        last so the user gets a diagnostic instead of a bare refusal."""
        from app.recovery.librarian import find_alternatives
        alts = find_alternatives(
            "explain quantum chromodynamics",
            refusal_category="generic",
            used_crew="research",
        )
        assert any(a.strategy == "forge_queue" for a in alts)

    def test_sync_alternatives_ranked_cheap_first(self):
        """Sync strategies (re_route, escalate_tier) must be ranked
        cheapest-first so the budget walk hits the most-likely-to-recover
        path before expensive ones. forge_queue is always LAST regardless
        of cost — it's the async fallback that always succeeds."""
        from app.recovery.librarian import find_alternatives
        alts = find_alternatives(
            "send my colleague the email about meetings",
            refusal_category="generic",   # generic fires escalate_tier too
            used_crew="research",
            used_tier="budget",
        )
        # Drop the trailing forge_queue to inspect runtime-strategy order
        runtime = [a for a in alts if a.strategy != "forge_queue"]
        costs = [a.est_cost_usd for a in runtime]
        assert costs == sorted(costs), (
            "Sync alternatives must be returned cheap-first; forge_queue "
            "always last."
        )
        # And forge_queue is always last in the full list
        assert alts[-1].strategy == "forge_queue", (
            "forge_queue is the unconditional async fallback — it's "
            "always the last alternative."
        )

    def test_escalate_tier_only_for_generic_refusal(self):
        """No point bumping the model tier for a missing-tool issue —
        a stronger LLM can't conjure a missing API key."""
        from app.recovery.librarian import find_alternatives
        alts_generic = find_alternatives(
            "summarize the meeting",
            refusal_category="generic",
            used_crew="research",
            used_tier="budget",
        )
        alts_missing = find_alternatives(
            "summarize the meeting",
            refusal_category="missing_tool",
            used_crew="research",
            used_tier="budget",
        )
        gen_strats = {a.strategy for a in alts_generic}
        miss_strats = {a.strategy for a in alts_missing}
        assert "escalate_tier" in gen_strats
        assert "escalate_tier" not in miss_strats


class TestLibrarianRegistryBridge:
    """The librarian augments its hand-curated _CAPABILITY_MAP with
    semantic search over the tool registry. Catches tools whose
    phrasing doesn't hit any keyword in the map."""

    @staticmethod
    def _fake_match(name: str, reason: str = "semantic match (d=0.30)"):
        from types import SimpleNamespace
        return SimpleNamespace(name=name, reason=reason, score=0.9)

    def test_registry_hit_emits_direct_tool_for_recipe_eligible_tool(self):
        """A registry hit on a recipe-eligible tool must surface as a
        direct_tool alternative, even when keyword inference produced
        nothing."""
        from app.recovery import librarian
        with patch(
            "app.tool_registry.discovery.search_tools",
            return_value=[self._fake_match("email_tools.check_email")],
        ):
            alts = librarian.find_alternatives(
                "pull the most recent message from my correspondence",
                refusal_category="missing_tool",
                used_crew="research",
            )
        direct_tool_alts = [
            a for a in alts
            if a.strategy == "direct_tool" and a.tool == "email_tools.check_email"
        ]
        assert direct_tool_alts, (
            "Registry semantic match on a recipe-eligible tool must "
            "emit a direct_tool alternative."
        )
        assert "registry semantic match" in direct_tool_alts[0].rationale.lower()

    def test_registry_hit_dedups_against_keyword_path(self):
        """When the keyword path already emitted a direct_tool for tool
        X, the registry path must NOT emit a duplicate."""
        from app.recovery import librarian
        with patch(
            "app.tool_registry.discovery.search_tools",
            return_value=[self._fake_match("email_tools.check_email")],
        ):
            alts = librarian.find_alternatives(
                "what emails arrived today",  # hits the 'email' keywords
                refusal_category="missing_tool",
                used_crew="research",
            )
        direct_tool_alts = [
            a for a in alts
            if a.strategy == "direct_tool" and a.tool == "email_tools.check_email"
        ]
        assert len(direct_tool_alts) == 1, (
            "Same (strategy, tool, crew) tuple must not be emitted "
            f"twice. Got: {[a.rationale for a in direct_tool_alts]}"
        )

    def test_registry_failure_does_not_break_recovery(self):
        """If search_tools raises, find_alternatives must still return
        the keyword-path alternatives + forge_queue."""
        from app.recovery import librarian
        with patch(
            "app.tool_registry.discovery.search_tools",
            side_effect=RuntimeError("chromadb unreachable"),
        ):
            alts = librarian.find_alternatives(
                "what emails arrived today",
                refusal_category="missing_tool",
                used_crew="research",
            )
        strategies = {a.strategy for a in alts}
        assert "forge_queue" in strategies, "forge_queue must always survive"
        assert "re_route" in strategies, (
            "Keyword-path re_route to PIM must survive a registry failure."
        )

    def test_registry_hit_for_unknown_tool_is_dropped(self):
        """A registry hit on a tool with no direct_tool recipe must be
        dropped — we don't know how to invoke it directly. Today's
        eligibility list is hard-coded; expanding it is a separate change."""
        from app.recovery import librarian
        with patch(
            "app.tool_registry.discovery.search_tools",
            return_value=[self._fake_match("hypothetical.new_tool")],
        ):
            alts = librarian.find_alternatives(
                "do something only that hypothetical tool can do",
                refusal_category="missing_tool",
                used_crew="research",
            )
        assert not any(
            a.tool == "hypothetical.new_tool" for a in alts
        ), "Tools without a recipe must not be surfaced as direct_tool."


# ══════════════════════════════════════════════════════════════════════
# Strategies — re_route + forge_queue
# ══════════════════════════════════════════════════════════════════════

class TestReRouteStrategy:

    def test_succeeds_when_target_crew_returns_substantive(self):
        from app.recovery.strategies import re_route
        from app.recovery.librarian import Alternative
        commander = MagicMock()
        commander._run_crew.return_value = (
            "Here are your top 25 emails from today, ranked by sender "
            "importance and unread state: ..." + ("..." * 50)
        )
        alt = Alternative(
            strategy="re_route", crew="pim", rationale="x",
            est_cost_usd=0.02, est_latency_s=30, sync=True,
        )
        ctx = {"commander": commander, "crew_used": "research", "difficulty": 5}
        result = re_route.execute("emails today", alt, ctx)
        assert result.success
        assert "Redirected to pim" in (result.note or "")
        assert result.route_changed

    def test_fails_when_target_crew_also_refuses(self):
        from app.recovery.strategies import re_route
        from app.recovery.librarian import Alternative
        commander = MagicMock()
        commander._run_crew.return_value = (
            "I do not have access to your email account either. " * 3
        )
        alt = Alternative(
            strategy="re_route", crew="pim", rationale="x",
            est_cost_usd=0.02, est_latency_s=30, sync=True,
        )
        ctx = {"commander": commander, "crew_used": "research", "difficulty": 5}
        result = re_route.execute("emails today", alt, ctx)
        assert not result.success
        assert "also produced a refusal" in (result.error or "")


class TestForgeQueueStrategy:
    """Always succeeds (returns diagnostic). Frequency tracker auto-
    queues forge experiment after threshold."""

    def setup_method(self):
        # Clean frequency file so each test starts fresh
        import app.recovery.strategies.forge_queue as fq
        try:
            fq._FREQUENCY_PATH.unlink()
        except FileNotFoundError:
            pass

    def test_first_refusal_records_but_does_not_queue(self, tmp_path, monkeypatch):
        import app.recovery.strategies.forge_queue as fq
        monkeypatch.setattr(fq, "_FREQUENCY_PATH", tmp_path / "freq.json")
        monkeypatch.setattr(fq, "_LEARNING_QUEUE", tmp_path / "queue.md")
        from app.recovery.strategies import forge_queue
        from app.recovery.librarian import Alternative
        alt = Alternative(strategy="forge_queue", rationale="x",
                          est_cost_usd=0, est_latency_s=0, sync=False)
        result = forge_queue.execute(
            "find sales lead at Foo Co",
            alt,
            {"refusal_category": "missing_tool"},
        )
        assert result.success
        assert result.text is not None
        # Frequency recorded but threshold not yet hit
        data = json.loads((tmp_path / "freq.json").read_text())
        gap_keys = list(data.keys())
        assert len(gap_keys) == 1
        assert data[gap_keys[0]]["count_in_window"] == 1
        # Learning queue NOT yet written (count 1 < threshold 3)
        assert not (tmp_path / "queue.md").exists()

    def test_third_refusal_queues_forge(self, tmp_path, monkeypatch):
        import app.recovery.strategies.forge_queue as fq
        monkeypatch.setattr(fq, "_FREQUENCY_PATH", tmp_path / "freq.json")
        monkeypatch.setattr(fq, "_LEARNING_QUEUE", tmp_path / "queue.md")
        from app.recovery.strategies import forge_queue
        from app.recovery.librarian import Alternative
        alt = Alternative(strategy="forge_queue", rationale="x",
                          est_cost_usd=0, est_latency_s=0, sync=False)
        for i in range(3):
            forge_queue.execute(
                "find sales lead at company",
                alt,
                {"refusal_category": "missing_tool"},
            )
        # Threshold hit → learning queue written
        assert (tmp_path / "queue.md").exists()
        contents = (tmp_path / "queue.md").read_text()
        assert "refusal-recovery" in contents
        assert "missing_tool" in contents


# ══════════════════════════════════════════════════════════════════════
# Loop — env gate, recursion guard, budget
# ══════════════════════════════════════════════════════════════════════

class TestLoopGating:

    def test_disabled_returns_not_triggered(self, monkeypatch):
        monkeypatch.delenv("RECOVERY_LOOP_ENABLED", raising=False)
        from app.recovery.loop import maybe_recover, is_enabled
        assert is_enabled() is False
        result = maybe_recover(
            "I do not have access to your email account.",
            "emails today", "research",
        )
        assert result.triggered is False

    def test_enabled_triggers_on_refusal(self, monkeypatch):
        monkeypatch.setenv("RECOVERY_LOOP_ENABLED", "true")
        # Bigger budget so re_route runs even when direct_tool +
        # skill_chain fail first (test environment has no real
        # email config or matching skill).
        monkeypatch.setenv("RECOVERY_MAX_ATTEMPTS", "5")
        commander = MagicMock()
        commander._run_crew.return_value = (
            "Here are your top emails: 1) ... 2) ... " * 30
        )
        from app.recovery.loop import maybe_recover
        result = maybe_recover(
            "I do not have access to your email account.",
            "what are the most important emails today",
            "research",
            commander=commander,
            difficulty=5,
        )
        assert result.triggered is True
        assert result.success is True
        assert "pim" in (result.note or "").lower()

    def test_recursion_guard_prevents_double_dip(self, monkeypatch):
        """A strategy's own LLM call must not trigger recovery again."""
        monkeypatch.setenv("RECOVERY_LOOP_ENABLED", "true")
        from app.recovery.loop import maybe_recover, _in_recovery
        token = _in_recovery.set(True)
        try:
            result = maybe_recover(
                "I do not have access.",
                "x", "research",
            )
            assert result.triggered is False
        finally:
            _in_recovery.reset(token)

    def test_budget_caps_attempts(self, monkeypatch):
        """RECOVERY_MAX_ATTEMPTS limits how many strategies we try."""
        monkeypatch.setenv("RECOVERY_LOOP_ENABLED", "true")
        monkeypatch.setenv("RECOVERY_MAX_ATTEMPTS", "1")
        commander = MagicMock()
        # Failing first crew, never gets to second
        commander._run_crew.return_value = (
            "I cannot access that data either either either."
        )
        from app.recovery.loop import maybe_recover
        result = maybe_recover(
            "I do not have access to your email account.",
            "emails today",
            "research",
            commander=commander, difficulty=5,
        )
        assert len(result.strategies_tried) <= 1


# ══════════════════════════════════════════════════════════════════════
# Today's 5 regression cases — end-to-end
# ══════════════════════════════════════════════════════════════════════

class TestTodaysRegressions:
    """Each refusal you saw today maps to a specific recovery path.
    These tests pin those mappings so they don't silently regress."""

    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("RECOVERY_LOOP_ENABLED", "true")

    def test_email_no_access_recovers_via_pim(self, monkeypatch):
        # Larger budget so we don't bail before re_route (direct_tool
        # runs first and may fail when email tools aren't configured;
        # re_route is the fallback that the test mocks for success).
        monkeypatch.setenv("RECOVERY_MAX_ATTEMPTS", "5")
        from app.recovery.loop import maybe_recover
        commander = MagicMock()
        commander._run_crew.return_value = (
            "Top 25 emails from yesterday: 1) Build update from CTO "
            "(priority high) 2) Customer escalation from PayU ..." * 5
        )
        result = maybe_recover(
            "I do not have access to your email account. "
            "To retrieve and rank your emails, authorize a connection.",
            "what are the most important emails I received yesterday",
            "research",
            commander=commander, difficulty=5,
        )
        assert result.triggered and result.success
        # re_route must be one of the strategies that ran (its place
        # in the order depends on whether direct_tool was reachable).
        assert "re_route" in result.strategies_tried
        assert "pim" in (result.note or "").lower()

    def test_unknown_capability_falls_to_forge_queue(self, tmp_path, monkeypatch):
        """When no crew alternative matches, forge_queue at least
        produces a diagnostic answer with action items."""
        import app.recovery.strategies.forge_queue as fq
        monkeypatch.setattr(fq, "_FREQUENCY_PATH", tmp_path / "freq.json")
        monkeypatch.setattr(fq, "_LEARNING_QUEUE", tmp_path / "queue.md")
        from app.recovery.loop import maybe_recover
        # Use a refusal that's missing-tool-shaped (NOT policy-shaped)
        # so the detector fires but no crew alternative matches.
        result = maybe_recover(
            "I'm unable to access that data — I don't have a tool for "
            "this kind of esoteric domain query.",
            "do something extremely esoteric I've never asked for",
            "research",
            difficulty=5,
        )
        assert result.triggered
        assert result.success  # forge_queue always succeeds
        assert result.text is not None
        # Diagnostic mentions the actionable suggestion
        assert "next time" in result.text.lower() or "to make this work" in result.text.lower()
