"""Regression test for mem0 provider selection — Ollama models go via
the native ``ollama`` provider, not ``litellm``.

Background — 2026-05-02: the Ops anomaly dashboard surfaced repeated
``WARNING: LLM extraction failed: Model 'ollama/qwen3.5:35b-a3b-q4_K_M'
in litellm does not support function calling.`` errors. Root cause:
mem0's litellm provider has an UNCONDITIONAL pre-flight gate at
mem0/llms/litellm.py:70 — it calls ``litellm.supports_function_calling
(model)`` even when the actual extraction call only passes
``response_format={"type":"json_object"}`` and no tools. litellm's
static registry doesn't list any Ollama models as supporting tools,
so every mem0 fact-extraction with our local model failed.

Fix: when the configured model is an Ollama tag, switch to mem0's
native ``ollama`` provider, which has no such gate and uses the
ollama-python client to talk to Ollama's native API. The underlying
model (Qwen3.5) supports tools natively per Ollama's catalog.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


class TestMem0ProviderSelection:

    def _stub_settings(self, monkeypatch, model: str):
        fake = MagicMock()
        fake.mem0_llm_model = model
        fake.mem0_embedder_model = "nomic-ai/nomic-embed-text-v1.5"
        fake.mem0_postgres_url = "postgresql://x@y/z"
        # Patch via the import site
        monkeypatch.setattr(
            "app.config.get_settings", lambda: fake,
        )

    def test_ollama_model_uses_native_provider(self, monkeypatch):
        """``ollama/qwen3.5:35b-a3b-q4_K_M`` → provider ``ollama`` with
        bare model name (no prefix), no litellm pre-flight gate."""
        self._stub_settings(monkeypatch, "ollama/qwen3.5:35b-a3b-q4_K_M")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama:11434")
        from app.memory.mem0_manager import _get_config
        cfg = _get_config()
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["llm"]["config"]["model"] == "qwen3.5:35b-a3b-q4_K_M"
        assert cfg["llm"]["config"]["ollama_base_url"] == "http://test-ollama:11434"

    def test_ollama_chat_prefix_also_routes_to_native(self, monkeypatch):
        """``ollama_chat/...`` is litellm's chat-completions variant —
        same underlying daemon, treat the same way."""
        self._stub_settings(monkeypatch, "ollama_chat/qwen3.5:35b-a3b-q4_K_M")
        from app.memory.mem0_manager import _get_config
        cfg = _get_config()
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["llm"]["config"]["model"] == "qwen3.5:35b-a3b-q4_K_M"

    def test_anthropic_model_keeps_litellm_provider(self, monkeypatch):
        """Cloud models (Claude, OpenAI) still use the litellm provider
        — they're properly registered with function-calling support."""
        self._stub_settings(monkeypatch, "anthropic/claude-sonnet-4-6")
        from app.memory.mem0_manager import _get_config
        cfg = _get_config()
        assert cfg["llm"]["provider"] == "litellm"
        assert cfg["llm"]["config"]["model"] == "anthropic/claude-sonnet-4-6"

    def test_openrouter_model_keeps_litellm_provider(self, monkeypatch):
        self._stub_settings(monkeypatch, "openrouter/qwen/qwen3-coder")
        from app.memory.mem0_manager import _get_config
        cfg = _get_config()
        assert cfg["llm"]["provider"] == "litellm"

    def test_ollama_uses_default_base_url_when_env_unset(self, monkeypatch):
        """No OLLAMA_BASE_URL / OLLAMA_HOST → falls back to
        ``host.docker.internal:11434`` (the prevailing default in this
        project)."""
        self._stub_settings(monkeypatch, "ollama/qwen3.5:35b-a3b-q4_K_M")
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        from app.memory.mem0_manager import _get_config
        cfg = _get_config()
        assert "host.docker.internal" in cfg["llm"]["config"]["ollama_base_url"]
