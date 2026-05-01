"""Tests for app.forge.audit.semantic — credit-aware failover path.

Background — 2026-04-30: user reported the Forge dashboard showing
``attack_classes_considered: ["judge_unavailable_fail_closed"]``. Root
cause: the semantic-audit judge called Anthropic directly, swallowed
the 400 "credit balance too low" error, and returned None. The shared
``circuit_breaker['anthropic_credits']`` was never tripped (so the rest
of the system kept hitting Anthropic too) and there was no failover to
OpenRouter — the audit just rejected every tool with risk=10.

Fix: the judge now mirrors the CreditAwareAnthropicCompletion pattern —
on credit-exhausted, trip the shared breaker and route through
OpenRouter Claude.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────


def _credit_exhausted_error():
    """Synthetic Anthropic 400 with the canonical credit-exhausted text."""
    return Exception(
        "Error code: 400 - {'type': 'error', 'error': {'type': "
        "'invalid_request_error', 'message': 'Your credit balance is too "
        "low to access the Anthropic API.'}}"
    )


# ── Failover routing ───────────────────────────────────────────────────


class TestJudgeFailover:

    def test_anthropic_success_returns_directly(self, monkeypatch):
        """Happy path: Anthropic returns text — no failover invoked."""
        from app.forge.audit import semantic
        monkeypatch.setattr(
            semantic, "_call_judge_anthropic_direct",
            lambda p, m: "from-anthropic",
        )
        # Failover should NOT run — sentinel that fails the test if called
        def _fail(*a, **k):
            raise AssertionError("OpenRouter must not be called on happy path")
        monkeypatch.setattr(semantic, "_call_judge_openrouter", _fail)
        # Breaker must be closed for direct path to be taken
        from app import circuit_breaker
        b = circuit_breaker.get_breaker("anthropic_credits")
        for _ in range(b.failure_count + 1):
            circuit_breaker.record_success("anthropic_credits")
        assert semantic._call_judge("hello", "claude-sonnet-4-6") == "from-anthropic"

    def test_credit_exhausted_trips_breaker_and_failovers(self, monkeypatch):
        """The actual fix: 400 credit-exhausted → trip breaker → OpenRouter."""
        from app.forge.audit import semantic
        from app import circuit_breaker

        # Reset breaker
        b = circuit_breaker.get_breaker("anthropic_credits")
        for _ in range(b.failure_count + 1):
            circuit_breaker.record_success("anthropic_credits")
        assert circuit_breaker.is_available("anthropic_credits")

        def _credit_400(prompt, model):
            raise _credit_exhausted_error()
        monkeypatch.setattr(
            semantic, "_call_judge_anthropic_direct", _credit_400,
        )
        monkeypatch.setattr(
            semantic, "_call_judge_openrouter",
            lambda p, m: "from-or",
        )

        result = semantic._call_judge("hello", "claude-sonnet-4-6")
        assert result == "from-or"
        # Breaker must now be open
        assert not circuit_breaker.is_available("anthropic_credits")

    def test_unrelated_exception_does_not_failover(self, monkeypatch):
        """Non-credit errors (e.g. 500, network) should NOT trip the
        credit breaker AND should NOT failover. The auditor will
        produce a fail-closed reject as it always did — but the bug
        being fixed is specifically the credit-exhausted case."""
        from app.forge.audit import semantic
        from app import circuit_breaker

        # Reset breaker
        b = circuit_breaker.get_breaker("anthropic_credits")
        for _ in range(b.failure_count + 1):
            circuit_breaker.record_success("anthropic_credits")

        def _other_error(prompt, model):
            raise Exception("502 bad gateway")
        monkeypatch.setattr(
            semantic, "_call_judge_anthropic_direct", _other_error,
        )

        # OpenRouter should NOT be called
        def _fail(*a, **k):
            raise AssertionError("OpenRouter must not be called on non-credit errors")
        monkeypatch.setattr(semantic, "_call_judge_openrouter", _fail)

        result = semantic._call_judge("hello", "claude-sonnet-4-6")
        assert result is None
        # Breaker must NOT be tripped — this isn't a credit issue
        assert circuit_breaker.is_available("anthropic_credits")

    def test_open_breaker_skips_anthropic_uses_or_directly(self, monkeypatch):
        """If the breaker is already open at entry, don't bother probing
        Anthropic — go straight to OpenRouter. Same semantics as
        CreditAwareAnthropicCompletion's per-call check."""
        from app.forge.audit import semantic
        from app import circuit_breaker

        # Trip the breaker
        circuit_breaker.record_failure("anthropic_credits")
        assert not circuit_breaker.is_available("anthropic_credits")

        # Anthropic direct must NOT be called
        def _fail(*a, **k):
            raise AssertionError("Anthropic direct must not be called when breaker open")
        monkeypatch.setattr(
            semantic, "_call_judge_anthropic_direct", _fail,
        )
        monkeypatch.setattr(
            semantic, "_call_judge_openrouter",
            lambda p, m: "from-or-direct",
        )

        result = semantic._call_judge("hello", "claude-sonnet-4-6")
        assert result == "from-or-direct"


class TestOpenRouterFailover:

    def test_no_openrouter_key_returns_none(self, monkeypatch):
        """Missing OPENROUTER_API_KEY → can't failover → None.

        Caller will then fail closed — but with a clear log line
        explaining the configuration gap, not a silent "judge_unavailable"."""
        from app.forge.audit import semantic
        fake_settings = MagicMock()
        fake_settings.openrouter_api_key.get_secret_value.return_value = ""
        monkeypatch.setattr("app.config.get_settings", lambda: fake_settings)
        # No OpenAI client should even be constructed
        with patch("openai.OpenAI", side_effect=AssertionError("must not be called")):
            result = semantic._call_judge_openrouter("hello", "claude-sonnet-4-6")
        assert result is None


class TestModelMapping:

    def test_known_models_map_directly(self):
        from app.forge.audit.semantic import _model_to_anthropic_id
        assert _model_to_anthropic_id("claude-opus-4-7") == "claude-opus-4-7"
        assert _model_to_anthropic_id("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_haiku_resolves_to_full_id(self):
        from app.forge.audit.semantic import _model_to_anthropic_id
        assert _model_to_anthropic_id("claude-haiku-4-5") == "claude-haiku-4-5-20251001"

    def test_unknown_falls_back_to_sonnet(self):
        from app.forge.audit.semantic import _model_to_anthropic_id
        assert _model_to_anthropic_id("not-a-real-model") == "claude-sonnet-4-6"
