"""Regression: 6 misclassified WARN sites must log at INFO.

Pre-fix shape (the operator-reported bug):

  pattern_learner reported the following as "uncovered failure
  patterns" requiring runbook scaffolds:

    787a4626 — llm_selector: no candidate meets min_recency
    bd88ef52 — circuit_breaker[anthropic_credits]: HALF_OPEN → OPEN
    e527cceb — Proposal rejected — path violations
    b32c0f0c — Agent('Writer' [anthropic]): capped tools 52 → 25
    9fec32a0 — CreditAwareAnthropicCompletion: credit-exhausted 400 …
               failing over mid-call to OpenRouter

  Each was logging at WARNING but described by-design behavior
  (graceful degradation, validator working, breaker doing its job,
  designed failover). The pattern_learner saw them as new failures
  to remediate — but the right answer is "this isn't an error."

Post-fix: each site logs at INFO. The audit/dashboard channels that
care still subscribe to those loggers' INFO streams; errors.jsonl
stays clean.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP = _REPO_ROOT / "app"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── Source-grep contracts (cheaper than booting each module) ────────


class TestLlmSelectorKeepIncumbent:

    def test_no_candidate_min_recency_logs_at_info(self) -> None:
        src = _read(_APP / "llm_selector.py")
        # The original message string is unchanged; only the level moved.
        # Pin the contract: the f-string lives under logger.info, not
        # logger.warning.
        m = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*\"llm_selector: no candidate meets min_recency",
            src,
        )
        assert m is not None, "could not locate keep-incumbent log site"
        assert m.group(1) == "info", (
            f"keep-incumbent message must log at INFO; found {m.group(1)}"
        )


class TestCircuitBreakerHalfOpenToOpen:

    def test_half_open_to_open_logs_at_info(self) -> None:
        src = _read(_APP / "circuit_breaker.py")
        # The HALF_OPEN→OPEN transition (probe-failed) should be INFO;
        # the CLOSED→OPEN transition (first trip) should still be WARN.
        # Pin both with positional context.
        # 1. HALF_OPEN→OPEN block
        idx = src.index("HALF_OPEN → OPEN")
        # Look BACKWARD for the logger call.
        before = src[max(0, idx - 200):idx]
        assert "logger.info(" in before, (
            "HALF_OPEN→OPEN must log at INFO (probe-fail is by-design)"
        )
        assert "logger.warning(" not in before, (
            "HALF_OPEN→OPEN must NOT log at WARN"
        )

    def test_first_trip_still_warns(self) -> None:
        """The first CLOSED→OPEN transition (consecutive failures
        threshold) is the operator-visible signal — keep WARN."""
        src = _read(_APP / "circuit_breaker.py")
        # The {prev} → OPEN ... consecutive failures format is the
        # threshold-trip path.
        m = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*f\"circuit_breaker\[\{self\.name\}\]: \{prev\} → OPEN",
            src,
        )
        assert m is not None
        assert m.group(1) == "warning", (
            f"first-trip transition must stay WARN; found {m.group(1)}"
        )


class TestProposalsPathViolations:

    def test_path_violation_logs_at_info(self) -> None:
        src = _read(_APP / "proposals.py")
        m = re.search(
            r"logger\.(info|warning)\(f\"Proposal rejected — path violations:",
            src,
        )
        assert m is not None
        assert m.group(1) == "info", (
            f"path-violation rejection must log at INFO; found {m.group(1)}"
        )


class TestBaseCrewToolCap:

    def test_post_init_tool_cap_logs_at_info(self) -> None:
        src = _read(_APP / "crews" / "base_crew.py")
        # Two tool-cap sites:
        #   * "{crew_name} [{provider}]: capped tools {before} → ..."
        #     — post-init cap (around line 850)
        #   * "Agent('{role}' [{provider}]): capped tools ..."
        #     — pre-init cap (around line 1340)
        # Both should be INFO.
        m1 = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*f\"\{crew_name\} \[\{provider\}\]: capped tools",
            src,
        )
        assert m1 is not None, "post-init cap log site moved"
        assert m1.group(1) == "info"

        m2 = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*f\"Agent\('\{kwargs\.get\('role',\s*'\?'\)\}'\s+\[\{provider\}\]\):",
            src,
        )
        assert m2 is not None, "pre-init cap log site moved"
        assert m2.group(1) == "info"


class TestCreditAwareAnthropicFailover:

    def test_sync_failover_logs_at_info(self) -> None:
        src = _read(_APP / "llms" / "credit_aware_anthropic.py")
        # Sync path — message format: "%s from Anthropic — failing over"
        m = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*\"CreditAwareAnthropicCompletion: %s from Anthropic — \"",
            src,
        )
        assert m is not None, "sync-path failover log site moved"
        assert m.group(1) == "info"

    def test_async_failover_logs_at_info(self) -> None:
        src = _read(_APP / "llms" / "credit_aware_anthropic.py")
        m = re.search(
            r"logger\.(info|warning)\(\s*\n?\s*\"CreditAwareAnthropicCompletion: %s from Anthropic \(async\) — \"",
            src,
        )
        assert m is not None, "async-path failover log site moved"
        assert m.group(1) == "info"


# ── Noise filter for third-party WARNs we already handle ────────────


class TestJsonlNoiseFilter:
    """Third-party messages we've explicitly decided are noise must
    be filtered from the JSONL handler."""

    def test_filter_drops_discord_voice_warn(self) -> None:
        import logging
        from app.logging_filters import JsonlNoiseFilter

        f = JsonlNoiseFilter()
        rec = logging.LogRecord(
            name="discord.client", level=logging.WARNING,
            pathname="discord/client.py", lineno=1,
            msg="PyNaCl is not installed, voice will NOT be supported",
            args=None, exc_info=None,
        )
        assert f.filter(rec) is False, "discord voice warn should be dropped"

    def test_filter_drops_anthropic_400(self) -> None:
        import logging
        from app.logging_filters import JsonlNoiseFilter

        f = JsonlNoiseFilter()
        rec = logging.LogRecord(
            name="root", level=logging.WARNING, pathname="x", lineno=1,
            msg=(
                "Anthropic API call failed: Error code: 400 - {'type': "
                "'error', ... 'credit balance is too low'}"
            ),
            args=None, exc_info=None,
        )
        assert f.filter(rec) is False

    def test_filter_drops_openrouter_stealth_502(self) -> None:
        import logging
        from app.logging_filters import JsonlNoiseFilter

        f = JsonlNoiseFilter()
        rec = logging.LogRecord(
            name="root", level=logging.WARNING, pathname="x", lineno=1,
            msg=(
                "OpenAI API call failed: Error code: 502 - {'error': "
                "{'message': 'Invalid URL: '}}"
            ),
            args=None, exc_info=None,
        )
        assert f.filter(rec) is False

    def test_filter_keeps_unrelated_warn(self) -> None:
        """A genuine unrelated warning must pass through."""
        import logging
        from app.logging_filters import JsonlNoiseFilter

        f = JsonlNoiseFilter()
        rec = logging.LogRecord(
            name="my.module", level=logging.WARNING,
            pathname="x", lineno=1,
            msg="Database connection lost, retrying in 5s",
            args=None, exc_info=None,
        )
        assert f.filter(rec) is True, "unrelated warn must pass through"

    def test_filter_handles_format_failure_safely(self) -> None:
        """If record formatting raises, default to keeping (don't
        silently drop real errors)."""
        import logging
        from app.logging_filters import JsonlNoiseFilter

        f = JsonlNoiseFilter()
        # %d expects int but we pass a string — record.getMessage() will
        # raise TypeError.
        rec = logging.LogRecord(
            name="x", level=logging.ERROR, pathname="y", lineno=1,
            msg="error count: %d", args=("not-an-int",), exc_info=None,
        )
        assert f.filter(rec) is True, (
            "filter must default-keep when formatting fails"
        )


class TestNoiseFilterWiredIntoErrorHandler:
    """The setup_structured_logging() must attach the noise filter to
    the JSONL handler. Otherwise the third-party WARNs leak through."""

    def test_wired_into_setup(self) -> None:
        src = _read(_APP / "error_handler.py")
        assert "JsonlNoiseFilter" in src, (
            "setup_structured_logging must import + attach JsonlNoiseFilter"
        )
        assert "handler.addFilter" in src, (
            "filter must be attached to the JSONL handler, not root logger"
        )
