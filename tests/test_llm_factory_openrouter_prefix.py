"""Regression test for the OpenRouter model_id-prefix bug.

Background — 2026-05-02: the Ops anomaly dashboard surfaced a 24x
baseline rate spike of ``Anthropic API call failed: 'str' object has
no attribute 'content'`` errors on coding-role calls to
``claude-opus-4-7``. Root cause: ``_try_api`` was passing
``entry["model_id"]`` (e.g. ``anthropic/claude-opus-4-7``) directly
to CrewAI's LLM factory together with OpenRouter's base_url. CrewAI
routes by model_id prefix — ``anthropic/...`` selects its native
AnthropicCompletion provider, which makes Anthropic-SDK
``messages.create()`` calls. Those calls hit OpenRouter's endpoint
(due to the base_url override) and get back a payload the Anthropic
SDK can't parse — ``response.content`` ends up being a string
instead of the expected list of TextBlocks.

Fix: when serving any model via OpenRouter, prepend ``openrouter/``
to the model_id so CrewAI's dispatcher selects litellm's openrouter
provider (which correctly hits ``/chat/completions`` and parses an
OpenAI-shaped response).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestOpenRouterModelIdPrefix:

    def _stub_settings(self, monkeypatch):
        fake = MagicMock()
        fake.openrouter_api_key.get_secret_value.return_value = "sk-or-fake"
        # Patch via the module-local binding (`app.llm_factory.get_settings`)
        # because llm_factory imports the function at module load time —
        # patching `app.config.get_settings` doesn't affect the already-
        # imported reference, plus the underlying function is `@cache`-d.
        monkeypatch.setattr("app.llm_factory.get_settings", lambda: fake)

    def _reset_breaker(self, name: str):
        from app import circuit_breaker
        b = circuit_breaker.get_breaker(name)
        for _ in range(b.failure_count + 1):
            circuit_breaker.record_success(name)

    def test_anthropic_prefixed_model_gets_openrouter_prepended(self, monkeypatch):
        """An ``anthropic/claude-opus-4-7`` entry routed via OpenRouter
        must reach _cached_llm with ``openrouter/anthropic/claude-opus-4-7``."""
        self._stub_settings(monkeypatch)
        self._reset_breaker("openrouter")

        captured: dict = {}

        def _fake_cached(model_id, max_tokens=4096, **kwargs):
            captured["model_id"] = model_id
            captured["base_url"] = kwargs.get("base_url", "")
            return MagicMock()

        monkeypatch.setattr("app.llm_factory._cached_llm", _fake_cached)

        from app.llm_factory import _try_api
        entry = {
            "model_id": "anthropic/claude-opus-4-7",
            "tier": "premium",
            "cost_output_per_m": 25.0,
        }
        result = _try_api("claude-opus-4-7", entry, 4096, role="coding")
        assert result is not None
        assert captured["model_id"] == "openrouter/anthropic/claude-opus-4-7"
        assert "openrouter.ai" in captured["base_url"]

    def test_already_prefixed_model_id_is_idempotent(self, monkeypatch):
        """If the catalog already stores ``openrouter/...``, don't
        double-prefix it."""
        self._stub_settings(monkeypatch)
        self._reset_breaker("openrouter")

        captured: dict = {}

        def _fake_cached(model_id, max_tokens=4096, **kwargs):
            captured["model_id"] = model_id
            return MagicMock()

        monkeypatch.setattr("app.llm_factory._cached_llm", _fake_cached)

        from app.llm_factory import _try_api
        entry = {
            "model_id": "openrouter/anthropic/claude-sonnet-4-6",
            "tier": "mid",
            "cost_output_per_m": 5.0,
        }
        _try_api("claude-sonnet-4-6", entry, 4096, role="research")
        assert captured["model_id"] == "openrouter/anthropic/claude-sonnet-4-6"
        assert not captured["model_id"].startswith("openrouter/openrouter/")

    def test_non_anthropic_models_also_get_prefixed(self, monkeypatch):
        """Same prefix rule applies to all OpenRouter-served models —
        not just Claude. A bare ``deepseek/deepseek-chat-v3`` entry
        must become ``openrouter/deepseek/deepseek-chat-v3``."""
        self._stub_settings(monkeypatch)
        self._reset_breaker("openrouter")

        captured: dict = {}

        def _fake_cached(model_id, max_tokens=4096, **kwargs):
            captured["model_id"] = model_id
            return MagicMock()

        monkeypatch.setattr("app.llm_factory._cached_llm", _fake_cached)

        from app.llm_factory import _try_api
        entry = {
            "model_id": "deepseek/deepseek-chat-v3-0324",
            "tier": "budget",
            "cost_output_per_m": 0.40,
        }
        _try_api("deepseek-chat-v3", entry, 4096, role="research")
        assert captured["model_id"] == "openrouter/deepseek/deepseek-chat-v3-0324"

    def test_skips_when_breaker_open(self, monkeypatch):
        """If openrouter circuit breaker is OPEN, _try_api short-
        circuits before any model_id manipulation."""
        self._stub_settings(monkeypatch)
        from app import circuit_breaker
        # Trip the breaker
        for _ in range(20):
            circuit_breaker.record_failure("openrouter")
        assert not circuit_breaker.is_available("openrouter")

        called = {"n": 0}
        def _fake_cached(*args, **kwargs):
            called["n"] += 1

        monkeypatch.setattr("app.llm_factory._cached_llm", _fake_cached)

        from app.llm_factory import _try_api
        entry = {
            "model_id": "anthropic/claude-opus-4-7",
            "tier": "premium",
            "cost_output_per_m": 25.0,
        }
        result = _try_api("claude-opus-4-7", entry, 4096, role="coding")
        assert result is None
        assert called["n"] == 0
        # Reset for other tests
        for _ in range(25):
            circuit_breaker.record_success("openrouter")

    def test_skips_when_no_api_key(self, monkeypatch):
        """No OPENROUTER_API_KEY → return None gracefully (logs warning)."""
        fake = MagicMock()
        fake.openrouter_api_key.get_secret_value.return_value = ""
        monkeypatch.setattr("app.llm_factory.get_settings", lambda: fake)
        self._reset_breaker("openrouter")

        called = {"n": 0}
        def _fake_cached(*args, **kwargs):
            called["n"] += 1

        monkeypatch.setattr("app.llm_factory._cached_llm", _fake_cached)

        from app.llm_factory import _try_api
        entry = {
            "model_id": "anthropic/claude-opus-4-7",
            "tier": "premium",
            "cost_output_per_m": 25.0,
        }
        result = _try_api("claude-opus-4-7", entry, 4096, role="coding")
        assert result is None
        assert called["n"] == 0
