"""
test_concierge — wrapper toggle, skip heuristics, LLM call, length guard.

The Anthropic SDK is monkey-patched so no real API call happens.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_settings(tmp_path, monkeypatch):
    import app.runtime_settings as rs
    monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
    monkeypatch.setattr(rs, "_cache", None, raising=False)
    yield
    monkeypatch.setattr(rs, "_cache", None, raising=False)


# ── Toggle gating ─────────────────────────────────────────────────────────

def test_passthrough_when_disabled(monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    # Default state (toggle off) — no rewrite, no LLM call.
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail("LLM should not be called when disabled"),
    )
    result = apply_concierge("Some long enough conversational reply that would otherwise be rewritten.")
    assert result.startswith("Some long enough")


# ── Skip heuristics ───────────────────────────────────────────────────────

@pytest.fixture
def _enable_concierge():
    """Turn the toggle on for the duration of the test."""
    import app.runtime_settings as rs
    rs.set_concierge_persona_enabled(True)
    yield
    rs.set_concierge_persona_enabled(False)


def test_skip_too_short(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail("should skip short text"),
    )
    assert apply_concierge("ok") == "ok"


def test_skip_json_payload(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    payload = '{"crews": [{"crew": "research", "task": "summarize"}]}'
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail("should skip JSON"),
    )
    assert apply_concierge(payload) == payload


def test_skip_array_payload(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    payload = '[{"id": 1}, {"id": 2}, {"id": 3}]'
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail("should skip JSON array"),
    )
    assert apply_concierge(payload) == payload


def test_skip_fenced_code(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    code = "Here is your snippet:\n```python\nprint('hi')\n```"
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail("should skip fenced code"),
    )
    assert apply_concierge(code) == code


@pytest.mark.parametrize("prefix", [
    "Usage: /skill run <name>",
    "AndrusAI status\n  voice: off",
    "AndrusAI — Signal commands\n\nStatus & info:",
    "Skill registry — save tasks you run repeatedly.",
    "Skills (3 total):",
    "Skill: weekly status",
    "Saved skill 'weekly'.",
    "Deleted skill 'foo'.",
    "✓ done in 4.2s",
    "✗ failed: RuntimeError: nope",
])
def test_skip_known_structured_prefixes(_enable_concierge, monkeypatch, prefix):
    from app.personality.concierge_wrapper import apply_concierge
    monkeypatch.setattr(
        "app.personality.concierge_wrapper._rewrite_with_llm",
        lambda *a, **kw: pytest.fail(f"should skip {prefix!r}"),
    )
    assert apply_concierge(prefix) == prefix


# ── LLM rewrite path ──────────────────────────────────────────────────────

def _stub_anthropic(monkeypatch, response_text: str):
    """Patch the Anthropic SDK so no real request goes out."""
    captured = {"system": None, "user": None, "model": None, "max_tokens": None}

    class _FakeContentBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class _FakeResponse:
        def __init__(self, text):
            self.content = [_FakeContentBlock(text)]

    class _FakeClient:
        def __init__(self, **kwargs):
            self.messages = MagicMock()
            self.messages.create = self._create

        def _create(self, **kw):
            captured.update(kw)
            return _FakeResponse(response_text)

    monkeypatch.setattr("anthropic.Anthropic", _FakeClient)
    monkeypatch.setattr(
        "app.personality.concierge_wrapper.get_anthropic_api_key",
        lambda: "sk-test",
    )
    return captured


def test_rewrite_replaces_terse_with_warm(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    captured = _stub_anthropic(monkeypatch, "Done — research crew is on it, about 18 seconds.")
    original = "Routed to research crew. ETA 18s. 3 sources will be checked."
    rewritten = apply_concierge(original)
    assert "research crew" in rewritten
    assert rewritten != original
    # Ensure the fake Anthropic call was actually made.
    assert captured["model"]
    assert captured["system"] is not None
    assert "concierge" in (captured["system"] or "").lower()


def test_rewrite_falls_back_when_too_long(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    very_long = "warm " * 200  # ~1000 chars
    _stub_anthropic(monkeypatch, very_long)
    original = "Routed to research crew. ETA 18s."
    # Length guard kicks in; concierge falls back to the original.
    assert apply_concierge(original) == original


def test_rewrite_falls_back_on_empty_response(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    _stub_anthropic(monkeypatch, "")
    original = "Routed to research crew. ETA 18s. 3 sources will be checked."
    assert apply_concierge(original) == original


def test_rewrite_falls_back_when_no_api_key(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge
    monkeypatch.setattr(
        "app.personality.concierge_wrapper.get_anthropic_api_key",
        lambda: "",
    )
    original = "Routed to research crew. ETA 18s. 3 sources will be checked."
    assert apply_concierge(original) == original


def test_rewrite_falls_back_when_anthropic_raises(_enable_concierge, monkeypatch):
    from app.personality.concierge_wrapper import apply_concierge

    class _ExplodingClient:
        def __init__(self, **kw):
            self.messages = MagicMock()
            self.messages.create = self._boom

        def _boom(self, **kw):
            raise RuntimeError("API down")

    monkeypatch.setattr("anthropic.Anthropic", _ExplodingClient)
    monkeypatch.setattr(
        "app.personality.concierge_wrapper.get_anthropic_api_key",
        lambda: "sk-test",
    )
    original = "Routed to research crew. ETA 18s. 3 sources will be checked."
    assert apply_concierge(original) == original  # fallback, no raise


def test_empty_input_passes_through(_enable_concierge):
    from app.personality.concierge_wrapper import apply_concierge
    assert apply_concierge("") == ""
    assert apply_concierge("   ") == "   "
