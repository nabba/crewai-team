"""
llm_factory.py — Multi-tier LLM provider with cascade routing.

Architecture:
  Commander:     always Claude Opus 4.6 (routing reliability, tiny token volume)
  Specialists:   cascade through tiers based on llm_mode + cost_mode + availability:
                   1. Local Ollama (free, Metal GPU)  — if mode allows and local_llm_enabled
                   2. API tier (budget/mid via OpenRouter) — if mode allows and api_tier_enabled
                   3. Claude Sonnet 4.6 (premium fallback) — always available
  Vetting:       Claude Sonnet 4.6 by default (near-Opus quality, 5x cheaper)
                 Only applied to local Ollama output (API-tier models are frontier quality)
"""

import logging
import time
from crewai import LLM
from app.config import get_settings, get_anthropic_api_key
from app.llm_catalog import (
    get_model, get_model_id, get_provider, get_tier,
    get_default_for_role, CATALOG,
)
from app import circuit_breaker

logger = logging.getLogger(__name__)

_last_model_name: str | None = None
_last_tier: str | None = None


def create_commander_llm() -> LLM:
    """Commander always uses Claude for maximum routing intelligence."""
    settings = get_settings()
    model_name = get_default_for_role("commander", settings.cost_mode)
    entry = get_model(model_name)
    if not entry or entry["provider"] != "anthropic":
        model_name = "claude-opus-4.6"
        entry = get_model(model_name)
    return LLM(
        model=entry["model_id"],
        api_key=get_anthropic_api_key(),
        max_tokens=512,
    )


def create_specialist_llm(
    max_tokens: int = 4096,
    role: str = "default",
    task_hint: str = "",
    force_tier: str | None = None,
) -> LLM:
    """
    Create an LLM for a specialist role using the tier cascade.
    Behavior depends on current llm_mode:
      local:  Ollama only, Claude fallback if Ollama fails
      cloud:  API tier (OpenRouter) or Claude, skip Ollama
      hybrid: Try Ollama first, cascade to API tier, then Claude

    If force_tier is set (e.g. from difficulty-based routing), it overrides
    the default tier selection from llm_selector.
    """
    global _last_model_name, _last_tier
    from app.llm_mode import get_mode
    settings = get_settings()
    mode = get_mode()

    from app.llm_selector import select_model
    model_name = select_model(role, task_hint, force_tier=force_tier)
    entry = get_model(model_name)

    if not entry:
        logger.warning(f"llm_factory: model {model_name!r} not in catalog, falling back")
        return _claude_fallback(role, max_tokens)

    tier = entry["tier"]
    provider = entry["provider"]

    # ── LOCAL mode: only Ollama, Claude fallback ──────────────────────
    if mode == "local":
        if tier == "local" and settings.local_llm_enabled:
            llm = _try_local(model_name, entry, max_tokens, role)
            if llm:
                return llm
        return _claude_fallback(role, max_tokens)

    # ── CLOUD mode: skip Ollama, use API/Anthropic ───────────────────
    if mode == "cloud":
        if tier in ("budget", "mid") and settings.api_tier_enabled:
            llm = _try_api(model_name, entry, max_tokens, role)
            if llm:
                return llm
        if provider == "anthropic":
            return _create_anthropic(model_name, entry, max_tokens, role)
        if tier == "premium" and provider == "openrouter":
            llm = _try_api(model_name, entry, max_tokens, role)
            if llm:
                return llm
        return _claude_fallback(role, max_tokens)

    # ── HYBRID mode: full cascade ────────────────────────────────────
    # Try local Ollama first
    if tier == "local" and settings.local_llm_enabled:
        llm = _try_local(model_name, entry, max_tokens, role)
        if llm:
            return llm
        # Local failed — try API tier
        if settings.api_tier_enabled:
            logger.info(f"llm_factory: local failed for role={role}, trying API tier")
            api_model = get_default_for_role(role, settings.cost_mode)
            api_entry = get_model(api_model)
            if api_entry and api_entry["tier"] in ("budget", "mid"):
                llm = _try_api(api_model, api_entry, max_tokens, role)
                if llm:
                    return llm
        return _claude_fallback(role, max_tokens)

    # Try API tier (OpenRouter)
    if tier in ("budget", "mid") and settings.api_tier_enabled:
        llm = _try_api(model_name, entry, max_tokens, role)
        if llm:
            return llm
        return _claude_fallback(role, max_tokens)

    # Premium tier (Anthropic or OpenRouter)
    if provider == "anthropic":
        return _create_anthropic(model_name, entry, max_tokens, role)
    elif provider == "openrouter":
        llm = _try_api(model_name, entry, max_tokens, role)
        if llm:
            return llm
        return _claude_fallback(role, max_tokens)

    return _claude_fallback(role, max_tokens)


def create_vetting_llm() -> LLM:
    """Vetting gate — Sonnet 4.6 by default (#1 GDPval-AA, 5x cheaper than Opus)."""
    settings = get_settings()
    model_name = settings.vetting_model
    entry = get_model(model_name)
    if entry and entry["provider"] == "anthropic":
        return LLM(
            model=entry["model_id"],
            api_key=get_anthropic_api_key(),
            max_tokens=4096,
        )
    return LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=get_anthropic_api_key(),
        max_tokens=4096,
    )


def create_cheap_vetting_llm() -> LLM:
    """Cheap verification gate — budget model for quick yes/no quality checks.
    Falls back to Sonnet if OpenRouter key is not set."""
    settings = get_settings()
    if settings.api_tier_enabled and settings.openrouter_api_key:
        budget_model = get_model("deepseek-v3.2")
        if budget_model:
            return LLM(
                model=budget_model["model_id"],
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.openrouter_api_key,
                max_tokens=256,
            )
    # Fallback to Sonnet
    return LLM(
        model="anthropic/claude-sonnet-4-6",
        api_key=get_anthropic_api_key(),
        max_tokens=256,
    )


def is_using_local() -> bool:
    return _last_tier == "local"

def is_using_api_tier() -> bool:
    return _last_tier in ("budget", "mid")

def get_last_model() -> str | None:
    return _last_model_name

def get_last_tier() -> str | None:
    return _last_tier


def _try_local(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM | None:
    global _last_model_name, _last_tier
    if not circuit_breaker.is_available("ollama"):
        logger.info(f"llm_factory: skipping Ollama (circuit open)")
        return None
    try:
        from app.ollama_native import spawn_model
        start = time.monotonic()
        url = spawn_model(model_name)
        spawn_ms = int((time.monotonic() - start) * 1000)
        if url:
            _last_model_name = model_name
            _last_tier = "local"
            circuit_breaker.record_success("ollama")
            logger.info(f"llm_factory: role={role} → LOCAL {model_name} at {url} (spawn: {spawn_ms}ms)")
            return LLM(model=entry["model_id"], base_url=url, max_tokens=max_tokens)
        circuit_breaker.record_failure("ollama")
    except Exception as exc:
        circuit_breaker.record_failure("ollama")
        logger.warning(f"llm_factory: local {model_name} failed: {exc}")
    return None


def _try_api(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM | None:
    global _last_model_name, _last_tier
    if not circuit_breaker.is_available("openrouter"):
        logger.info(f"llm_factory: skipping OpenRouter (circuit open)")
        return None
    settings = get_settings()
    api_key = settings.openrouter_api_key
    if not api_key:
        logger.warning("llm_factory: OpenRouter API key not set, skipping API tier")
        return None
    try:
        _last_model_name = model_name
        _last_tier = entry["tier"]
        circuit_breaker.record_success("openrouter")
        logger.info(f"llm_factory: role={role} → API {model_name} (${entry['cost_output_per_m']:.2f}/Mo)")
        return LLM(
            model=entry["model_id"],
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        circuit_breaker.record_failure("openrouter")
        logger.warning(f"llm_factory: API {model_name} failed: {exc}")
        _last_model_name = None
        _last_tier = None
    return None


def _create_anthropic(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM:
    global _last_model_name, _last_tier
    _last_model_name = model_name
    _last_tier = entry["tier"]
    logger.info(f"llm_factory: role={role} → ANTHROPIC {model_name} (${entry['cost_output_per_m']:.2f}/Mo)")
    return LLM(model=entry["model_id"], api_key=get_anthropic_api_key(), max_tokens=max_tokens)


def _claude_fallback(role: str, max_tokens: int) -> LLM:
    global _last_model_name, _last_tier
    _last_model_name = "claude-sonnet-4.6"
    _last_tier = "premium"
    logger.info(f"llm_factory: role={role} → FALLBACK Claude Sonnet 4.6")
    return LLM(model="anthropic/claude-sonnet-4-6", api_key=get_anthropic_api_key(), max_tokens=max_tokens)
