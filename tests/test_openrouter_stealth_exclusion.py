"""Regression: OpenRouter Stealth-provider exclusion must fire on
prefix-routed calls, not just explicit-base_url calls.

Pre-fix shape (the operator-reported bug):

  pattern_learner reported 630 occurrences/week of:
    "OpenAI API call failed: Error code: 502 -
     {'error': {'message': 'Invalid URL: ', 'code': 502,
                'metadata': {'provider_name': 'Stealth'}}}"

  The provider exclusion in app/llm_factory.py was gated on
  ``"openrouter.ai" in (base_url or "")`` — but the bulk of our
  OpenRouter traffic uses prefix routing
  (``model_id="openrouter/deepseek/deepseek-chat"``) without an
  explicit ``base_url`` kwarg. litellm routes those calls to
  OpenRouter via the ``OPENROUTER_API_KEY`` env var, so the
  trigger never fired and Stealth was never excluded.

Post-fix:
  Trigger fires when EITHER condition is true:
    1. ``"openrouter.ai" in base_url`` (original behavior), OR
    2. ``model_id`` starts with ``openrouter/`` (new — covers
       prefix routing)

  The Stealth filter is still env-var-overrideable
  (``OPENROUTER_IGNORE_PROVIDERS=""`` to disable, or to
  add other provider names).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLM_FACTORY = _REPO_ROOT / "app" / "llm_factory.py"


@pytest.fixture(scope="module")
def factory_src() -> str:
    return _LLM_FACTORY.read_text(encoding="utf-8")


# ── Source-grep contract ────────────────────────────────────────────


class TestStealthFilterTrigger:
    """The trigger condition must check both the base_url AND the
    model_id prefix. A regex is used because we want to catch any
    refactoring that DROPS the prefix check."""

    def test_trigger_checks_openrouter_ai_in_base_url(
        self, factory_src: str,
    ) -> None:
        # The original check is preserved.
        assert '"openrouter.ai" in (base_url or "")' in factory_src

    def test_trigger_checks_openrouter_prefix_in_model_id(
        self, factory_src: str,
    ) -> None:
        """The 2026-05-10 T3.3 fix: prefix-routed calls
        (``openrouter/<vendor>/<model>``) must also trigger the
        Stealth-exclusion filter."""
        # Match either '.startswith("openrouter/")' anywhere in the
        # provider-exclusion block.
        m = re.search(
            r'\(model_id\s+or\s+""\)\.startswith\("openrouter/"\)',
            factory_src,
        )
        assert m is not None, (
            "T3.3 fix requires the trigger to also check "
            'model_id.startswith("openrouter/")'
        )

    def test_trigger_combines_with_OR(self, factory_src: str) -> None:
        """The two conditions must be combined with OR (either one
        triggering the filter is sufficient)."""
        # Find the assignment site, then read a 200-char window.
        idx = factory_src.find("_is_openrouter_call")
        assert idx >= 0, "trigger must be named _is_openrouter_call"
        window = factory_src[idx:idx + 300]
        assert " or " in window, (
            "the two trigger conditions must be OR-combined; "
            f"window:\n{window}"
        )
        # Both branches must appear in the window.
        assert "openrouter.ai" in window
        assert 'startswith("openrouter/")' in window


# ── Functional simulation ───────────────────────────────────────────
#
# We can't easily import llm_factory in the host venv (pydantic_settings
# missing), so the contract above is enforced via source-grep. The
# functional behavior is exercised by the gateway integration test in
# the next class, which only runs in-container.


@pytest.fixture
def in_container_only():
    """Skip outside the gateway container — the test depends on
    fully-installed llm_factory + crewai stack."""
    try:
        from app import llm_factory  # noqa: F401
    except ModuleNotFoundError as exc:
        pytest.skip(f"llm_factory unavailable in this env: {exc}")


class TestPrefixRoutingTriggersFilter:
    """When called with a prefix-routed model_id and NO explicit
    base_url, the filter must inject the provider.ignore list."""

    def test_prefix_routed_call_gets_provider_ignore(
        self, in_container_only, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate a prefix-routed call. Capture the kwargs passed
        to the LLM constructor and assert provider.ignore was
        injected with 'Stealth'."""
        from app import llm_factory

        # Replace the LLM class with a stub that records kwargs.
        captured = {}

        def _stub_llm(*, model, max_tokens, **kwargs):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            captured.update(kwargs)
            return object()  # opaque LLM stand-in

        monkeypatch.setattr(
            llm_factory, "_get_LLM_class",
            lambda: lambda model, max_tokens, **kw: _stub_llm(
                model=model, max_tokens=max_tokens, **kw,
            ),
        )
        # Bust the cache so our stub gets called.
        monkeypatch.setattr(llm_factory, "_llm_cache", {})

        # Prefix-routed call — no base_url kwarg.
        llm_factory._cached_llm(
            model_id="openrouter/deepseek/deepseek-chat",
            max_tokens=1024,
        )

        # The filter must have injected extra_body.provider.ignore.
        extra_body = captured.get("extra_body", {})
        provider = extra_body.get("provider", {})
        ignore = provider.get("ignore", [])
        assert "Stealth" in ignore, (
            f"prefix-routed openrouter call must get Stealth in "
            f"provider.ignore; got extra_body={extra_body!r}"
        )

    def test_explicit_base_url_still_triggers(
        self, in_container_only, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Original behavior preserved — explicit base_url must
        still trigger."""
        from app import llm_factory

        captured = {}

        def _stub_llm(*, model, max_tokens, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            llm_factory, "_get_LLM_class",
            lambda: lambda model, max_tokens, **kw: _stub_llm(
                model=model, max_tokens=max_tokens, **kw,
            ),
        )
        monkeypatch.setattr(llm_factory, "_llm_cache", {})

        llm_factory._cached_llm(
            model_id="some-model",  # no openrouter prefix
            max_tokens=1024,
            base_url="https://openrouter.ai/api/v1",
        )

        extra_body = captured.get("extra_body", {})
        provider = extra_body.get("provider", {})
        assert "Stealth" in provider.get("ignore", []), (
            "explicit openrouter.ai base_url must still trigger "
            "Stealth exclusion"
        )

    def test_non_openrouter_call_unaffected(
        self, in_container_only, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A call with no openrouter prefix and no openrouter base_url
        must NOT receive a provider.ignore — we don't pollute
        non-OpenRouter requests."""
        from app import llm_factory

        captured = {}

        def _stub_llm(*, model, max_tokens, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            llm_factory, "_get_LLM_class",
            lambda: lambda model, max_tokens, **kw: _stub_llm(
                model=model, max_tokens=max_tokens, **kw,
            ),
        )
        monkeypatch.setattr(llm_factory, "_llm_cache", {})

        llm_factory._cached_llm(
            model_id="anthropic/claude-sonnet-4-6",
            max_tokens=1024,
            base_url="https://api.anthropic.com",
        )

        extra_body = captured.get("extra_body", {})
        provider = extra_body.get("provider", {})
        assert provider.get("ignore", []) == [], (
            "non-openrouter call must NOT have Stealth in "
            "provider.ignore; this would be a leak across vendors"
        )

    def test_env_override_disables_filter(
        self, in_container_only, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting OPENROUTER_IGNORE_PROVIDERS="" must turn the filter
        off for operators who want to opt back into Stealth (or any
        provider) explicitly."""
        from app import llm_factory

        monkeypatch.setenv("OPENROUTER_IGNORE_PROVIDERS", "")

        captured = {}

        def _stub_llm(*, model, max_tokens, **kwargs):
            captured.update(kwargs)
            return object()

        monkeypatch.setattr(
            llm_factory, "_get_LLM_class",
            lambda: lambda model, max_tokens, **kw: _stub_llm(
                model=model, max_tokens=max_tokens, **kw,
            ),
        )
        monkeypatch.setattr(llm_factory, "_llm_cache", {})

        llm_factory._cached_llm(
            model_id="openrouter/x/y",
            max_tokens=1024,
        )

        extra_body = captured.get("extra_body", {})
        provider = extra_body.get("provider", {})
        # When env is empty, filter must NOT inject anything.
        assert "Stealth" not in provider.get("ignore", []), (
            "OPENROUTER_IGNORE_PROVIDERS=\"\" must disable Stealth "
            "exclusion entirely"
        )
