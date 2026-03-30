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
import threading
import time
from crewai import LLM
from app.config import get_settings, get_anthropic_api_key
from app.llm_catalog import (
    get_model, get_model_id, get_provider, get_tier,
    get_default_for_role, CATALOG,
)
from app import circuit_breaker

logger = logging.getLogger(__name__)

# Thread-local storage for last model/tier — prevents race conditions
# when multiple crews process concurrently in the commander thread pool (Q7).
_tls = threading.local()

# B2: Cache LLM objects by (model_id, max_tokens) to avoid re-creating per request.
# LLM objects are stateless — they just wrap a model_id + api_key + params.
# Thread-safe because dict reads are atomic in CPython and LLM() is immutable.
_llm_cache: dict[tuple, LLM] = {}
_llm_cache_lock = threading.Lock()


def _cached_llm(model_id: str, max_tokens: int = 4096, **kwargs) -> LLM:
    """Get or create an LLM object, caching by (model_id, max_tokens).

    LLM objects are stateless wrappers — safe to share across requests.
    Cache eliminates ~50-100ms of object creation per specialist call.
    """
    base_url = kwargs.get("base_url", "")
    key = (model_id, max_tokens, base_url or "default")
    cached = _llm_cache.get(key)
    if cached is not None:
        return cached
    with _llm_cache_lock:
        cached = _llm_cache.get(key)
        if cached is not None:
            return cached
        llm = LLM(model=model_id, max_tokens=max_tokens, **kwargs)
        _llm_cache[key] = llm
        logger.debug(f"llm_cache: new entry for {model_id} max={max_tokens} (cache size: {len(_llm_cache)})")
        return llm


def _get_last(attr: str) -> str | None:
    return getattr(_tls, attr, None)


def _set_last(model: str | None, tier: str | None) -> None:
    _tls.last_model_name = model
    _tls.last_tier = tier


def create_commander_llm() -> LLM:
    """Create the Commander routing LLM.

    Prefers Claude (Anthropic) for maximum routing quality.
    Falls back to the best available OpenRouter model when Anthropic
    credits are exhausted or the API key is missing/invalid.
    """
    from app.config import get_openrouter_api_key

    settings = get_settings()

    # Try Anthropic (Claude) first
    try:
        anthropic_key = get_anthropic_api_key()
        if anthropic_key:
            model_name = get_default_for_role("commander", settings.cost_mode)
            entry = get_model(model_name)
            if not entry or entry["provider"] != "anthropic":
                entry = get_model("claude-sonnet-4.6")
            if entry:
                return _cached_llm(entry["model_id"], max_tokens=1024, api_key=anthropic_key)
    except Exception:
        pass

    # Fallback: use best OpenRouter model for routing
    logger.warning("Commander: Anthropic unavailable, falling back to OpenRouter for routing")
    openrouter_key = get_openrouter_api_key()
    # Use deepseek-v3.2 — strong at JSON routing tasks, very cheap
    fallback_entry = get_model("deepseek-v3.2")
    if fallback_entry:
        return _cached_llm(fallback_entry["model_id"], max_tokens=1024, api_key=openrouter_key)
    # Last resort: any available model
    return _cached_llm("openrouter/deepseek/deepseek-chat", max_tokens=1024, api_key=openrouter_key)


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
      insane: Premium only — Opus for critical roles, Gemini 3.1 Pro / Sonnet for others

    If force_tier is set (e.g. from difficulty-based routing), it overrides
    the default tier selection from llm_selector.
    """
    # Q7: thread-local last model/tier tracking
    from app.llm_mode import get_mode
    settings = get_settings()
    mode = get_mode()

    # ── INSANE mode: premium-only, hardcoded role mapping ─────────────
    if mode == "insane":
        return _insane_mode_select(role, max_tokens)

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
        if tier in ("free", "budget", "mid") and settings.api_tier_enabled:
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
            if api_entry and api_entry["tier"] in ("free", "budget", "mid"):
                llm = _try_api(api_model, api_entry, max_tokens, role)
                if llm:
                    return llm
        return _claude_fallback(role, max_tokens)

    # Try API tier (OpenRouter)
    if tier in ("free", "budget", "mid") and settings.api_tier_enabled:
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
    """Vetting gate — Sonnet 4.6 preferred, falls back to OpenRouter if Anthropic unavailable."""
    from app.config import get_openrouter_api_key
    settings = get_settings()
    model_name = settings.vetting_model
    entry = get_model(model_name)
    anthropic_key = get_anthropic_api_key()
    if entry and entry["provider"] == "anthropic" and anthropic_key:
        return _cached_llm(entry["model_id"], max_tokens=4096, api_key=anthropic_key)
    if anthropic_key:
        return _cached_llm("anthropic/claude-sonnet-4-6", max_tokens=4096, api_key=anthropic_key)
    # Fallback to best OpenRouter model for vetting
    logger.warning("create_vetting_llm: Anthropic unavailable, using deepseek-v3.2 for vetting")
    fallback = get_model("deepseek-v3.2")
    model_id = fallback["model_id"] if fallback else "openrouter/deepseek/deepseek-chat"
    return _cached_llm(model_id, max_tokens=4096, api_key=get_openrouter_api_key())


def create_cheap_vetting_llm() -> LLM:
    """Cheap verification gate — budget model for quick yes/no quality checks.
    Falls back to Sonnet if OpenRouter key is not set."""
    settings = get_settings()
    or_key = settings.openrouter_api_key.get_secret_value()
    if settings.api_tier_enabled and or_key:
        budget_model = get_model("deepseek-v3.2")
        if budget_model:
            return _cached_llm(budget_model["model_id"], max_tokens=256,
                               base_url="https://openrouter.ai/api/v1", api_key=or_key)
    return _cached_llm("anthropic/claude-sonnet-4-6", max_tokens=256, api_key=get_anthropic_api_key())


def is_using_local() -> bool:
    return _get_last("last_tier") == "local"

def is_using_api_tier() -> bool:
    return _get_last("last_tier") in ("budget", "mid")

def get_last_model() -> str | None:
    return _get_last("last_model_name")

def get_last_tier() -> str | None:
    return _get_last("last_tier")


# ── INSANE mode role → model mapping ──────────────────────────────────────
# Critical roles get Opus; heavy-lifting roles get Gemini 3.1 Pro; support roles get Sonnet.
_INSANE_ROLE_MAP = {
    # Critical: Claude Opus 4.6
    "commander":    "claude-opus-4.6",
    "vetting":      "claude-opus-4.6",
    "critic":       "claude-opus-4.6",
    # Heavy-lifting: Gemini 3.1 Pro
    "coding":       "gemini-3.1-pro",
    "research":     "gemini-3.1-pro",
    "architecture": "gemini-3.1-pro",
    "debugging":    "gemini-3.1-pro",
    "planner":      "gemini-3.1-pro",
    "media":        "gemini-3.1-pro",
    # Support: Claude Sonnet 4.6
    "writing":      "claude-sonnet-4.6",
    "synthesis":    "claude-sonnet-4.6",
    "introspector": "claude-sonnet-4.6",
    "self_improve": "claude-sonnet-4.6",
    "default":      "claude-sonnet-4.6",
}


def _insane_mode_select(role: str, max_tokens: int) -> LLM:
    """INSANE mode: premium-only models — Opus, Gemini 3.1 Pro, Sonnet."""
    # Q7: thread-local last model/tier tracking
    model_name = _INSANE_ROLE_MAP.get(role, "claude-sonnet-4.6")
    entry = get_model(model_name)
    if not entry:
        return _claude_fallback(role, max_tokens)

    _set_last(model_name, "premium")

    if entry["provider"] == "anthropic":
        logger.info(f"llm_factory: [INSANE] role={role} → ANTHROPIC {model_name}")
        return _cached_llm(entry["model_id"], max_tokens=max_tokens, api_key=get_anthropic_api_key())

    # Gemini 3.1 Pro via OpenRouter
    settings = get_settings()
    api_key = settings.openrouter_api_key.get_secret_value()
    if api_key and circuit_breaker.is_available("openrouter"):
        gemini_max = max(max_tokens, 16384)
        logger.info(f"llm_factory: [INSANE] role={role} → API {model_name} (${entry['cost_output_per_m']:.2f}/Mo, max_tokens={gemini_max})")
        circuit_breaker.record_success("openrouter")
        return _cached_llm(entry["model_id"], max_tokens=gemini_max,
                           base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Fallback if OpenRouter unavailable
    logger.warning(f"llm_factory: [INSANE] OpenRouter unavailable for {model_name}, falling back to Claude")
    return _claude_fallback(role, max_tokens)


def _try_local(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM | None:
    # Q7: thread-local last model/tier tracking
    if not circuit_breaker.is_available("ollama"):
        logger.info(f"llm_factory: skipping Ollama (circuit open)")
        return None
    try:
        from app.ollama_native import spawn_model
        start = time.monotonic()
        url = spawn_model(model_name)
        spawn_ms = int((time.monotonic() - start) * 1000)
        if url:
            _set_last(model_name, "local")
            circuit_breaker.record_success("ollama")
            logger.info(f"llm_factory: role={role} → LOCAL {model_name} at {url} (spawn: {spawn_ms}ms)")
            return _cached_llm(entry["model_id"], max_tokens=max_tokens, base_url=url)
        circuit_breaker.record_failure("ollama")
    except Exception as exc:
        circuit_breaker.record_failure("ollama")
        logger.warning(f"llm_factory: local {model_name} failed: {exc}")
    return None


def _try_api(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM | None:
    # Q7: thread-local last model/tier tracking
    if not circuit_breaker.is_available("openrouter"):
        logger.info(f"llm_factory: skipping OpenRouter (circuit open)")
        return None
    settings = get_settings()
    api_key = settings.openrouter_api_key.get_secret_value()
    if not api_key:
        logger.warning("llm_factory: OpenRouter API key not set, skipping API tier")
        return None
    try:
        _set_last(model_name, entry["tier"])
        circuit_breaker.record_success("openrouter")
        logger.info(f"llm_factory: role={role} → API {model_name} (${entry['cost_output_per_m']:.2f}/Mo)")
        return _cached_llm(entry["model_id"], max_tokens=max_tokens,
                           base_url="https://openrouter.ai/api/v1", api_key=api_key)
    except Exception as exc:
        circuit_breaker.record_failure("openrouter")
        logger.warning(f"llm_factory: API {model_name} failed: {exc}")
        _set_last(None, None)
    return None


def _create_anthropic(model_name: str, entry: dict, max_tokens: int, role: str) -> LLM:
    # Q7: thread-local last model/tier tracking
    _set_last(model_name, entry["tier"])
    logger.info(f"llm_factory: role={role} → ANTHROPIC {model_name} (${entry['cost_output_per_m']:.2f}/Mo)")
    return _cached_llm(entry["model_id"], max_tokens=max_tokens, api_key=get_anthropic_api_key())


def _claude_fallback(role: str, max_tokens: int) -> LLM:
    """Final fallback: Claude Sonnet if Anthropic is available, else best OpenRouter model."""
    from app.config import get_openrouter_api_key
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        _set_last("claude-sonnet-4.6", "premium")
        logger.info(f"llm_factory: role={role} → FALLBACK Claude Sonnet 4.6")
        return _cached_llm("anthropic/claude-sonnet-4-6", max_tokens=max_tokens, api_key=anthropic_key)
    # Anthropic key missing — use OpenRouter deepseek as ultimate fallback
    _set_last("deepseek-v3.2", "budget")
    logger.warning(f"llm_factory: role={role} → FALLBACK deepseek-v3.2 (Anthropic key missing)")
    return _cached_llm("openrouter/deepseek/deepseek-chat", max_tokens=max_tokens, api_key=get_openrouter_api_key())
