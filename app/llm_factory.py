"""
llm_factory.py — Multi-tier LLM provider with cascade routing.
NOTE: `from __future__ import annotations` makes all type hints strings,
avoiding the need to import crewai.LLM at module load time (~2s saving).

Architecture:
  Commander:     resolver pick (premium-floor role) at current runtime mode
  Specialists:   cascade through tiers based on runtime mode + availability:
                   1. Local Ollama (free, Metal GPU)  — if mode allows local
                      tier and local_llm_enabled
                   2. API tier (budget/mid via OpenRouter) — if mode whitelist
                      includes it and api_tier_enabled
                   3. Claude Sonnet 4.6 (premium fallback) — always available
  Vetting:       Resolver pick for the vetting role at the current runtime mode.

Runtime mode vocabulary (see app.llm_catalog.RUNTIME_MODES):
  free, budget, balanced [default], quality, insane, anthropic
"""
from __future__ import annotations

import functools
import logging
import threading
import time
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from crewai import LLM  # type hints only — no runtime import cost
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
_llm_cache: dict[tuple, "LLM"] = {}
_llm_cache_lock = threading.Lock()

# Lazy-loaded crewai.LLM class — avoids 1.9s import at module load time.
# crewai's import chain pulls in its entire framework including litellm,
# pydantic models, tool registries, etc. Deferring to first use saves ~2s
# on cold boot and makes the module importable in <10ms.
# Uses @functools.cache (Python 3.9+) — thread-safe, no manual global needed.
@functools.cache
def _get_LLM_class():
    """Lazy-load crewai.LLM on first use."""
    from crewai import LLM
    return LLM


def _cached_llm(
    model_id: str,
    max_tokens: int = 4096,
    *,
    sampling_key: str = "",
    llm_builder=None,
    **kwargs,
) -> "LLM":
    """Get or create an LLM object, caching by
    (builder-tag, model_id, max_tokens, base_url, sampling_key).

    LLM objects are stateless wrappers over (model_id, api_key, params)
    — safe to share across requests.  Cache eliminates ~50-100ms of
    object creation per specialist call.

    Parameters
    ----------
    model_id, max_tokens, sampling_key, **kwargs
        Forwarded to the LLM constructor.
    llm_builder : Callable[[str, int, **kwargs], LLM], optional
        Factory for non-default LLM subclasses (e.g.
        ``CreditAwareAnthropicCompletion``).  Called as
        ``llm_builder(model_id, max_tokens, **kwargs)``.  If omitted,
        the default ``crewai.LLM`` constructor is used.

        NOTE: cached instances must behave correctly under every call —
        no sticky per-instance state that would break auto-recovery /
        shared-state contracts.  Our CreditAware subclass satisfies
        this because it consults ``circuit_breaker["anthropic_credits"]``
        on every ``call()``, so a cached instance always routes
        correctly even after credits are restored.

    Cache isolation
    ---------------
    The builder identity is part of the cache key.  Without this, a
    CreditAware entry under ``model_id=claude-sonnet-4-6`` would
    collide with a plain-``crewai.LLM`` entry for the same model id,
    and whichever built first would lock the cache shape.  Tagging by
    ``builder.__qualname__`` keeps the namespaces independent.
    """
    base_url = kwargs.get("base_url", "")
    builder_tag = llm_builder.__qualname__ if llm_builder is not None else "default"
    key = (builder_tag, model_id, max_tokens, base_url or "default", sampling_key)
    cached = _llm_cache.get(key)
    if cached is not None:
        return cached
    with _llm_cache_lock:
        cached = _llm_cache.get(key)
        if cached is not None:
            return cached

        # ── Anthropic prompt caching: enable via extra_headers ──
        # Reduces cost by ~90% on cached prefix tokens (system prompt,
        # constitution, soul files). Only activates for Claude models.
        # litellm passes extra_headers through to the Anthropic SDK.
        if _is_anthropic_model(model_id):
            extra_headers = kwargs.pop("extra_headers", {}) or {}
            extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
            kwargs["extra_headers"] = extra_headers

        # ── OpenRouter provider exclusion ──
        # OpenRouter's anonymous "Stealth" sub-provider class periodically
        # returns 502 `Invalid URL: ''` (3 174/50k errors as of 2026-04-30).
        # We exclude such providers by default via OpenRouter's documented
        # provider-routing API. Active role-assigned models (Claude / Gemma /
        # DeepSeek paid variants) all have non-Stealth routes, so this is a
        # reliability gain with no functional loss. Override via env var
        # OPENROUTER_IGNORE_PROVIDERS (CSV); set it empty to disable filtering.
        if "openrouter.ai" in (base_url or ""):
            import os as _os
            env_ignore = _os.environ.get("OPENROUTER_IGNORE_PROVIDERS", "Stealth")
            ignore_list = [n.strip() for n in env_ignore.split(",") if n.strip()]
            if ignore_list:
                extra_body = dict(kwargs.pop("extra_body", {}) or {})
                provider_pref = dict(extra_body.get("provider", {}) or {})
                existing = list(provider_pref.get("ignore", []) or [])
                for name in ignore_list:
                    if name not in existing:
                        existing.append(name)
                provider_pref["ignore"] = existing
                extra_body["provider"] = provider_pref
                kwargs["extra_body"] = extra_body

        if llm_builder is not None:
            llm = llm_builder(model_id, max_tokens, **kwargs)
        else:
            LLM = _get_LLM_class()
            llm = LLM(model=model_id, max_tokens=max_tokens, **kwargs)

        _llm_cache[key] = llm
        logger.debug(
            "llm_cache: new entry builder=%s model=%s max=%d sampling=%r (cache size: %d)",
            builder_tag, model_id, max_tokens, sampling_key, len(_llm_cache),
        )
        return llm


def _is_anthropic_model(model_id: str) -> bool:
    """Check if a model ID is an Anthropic Claude model."""
    lower = model_id.lower()
    return any(k in lower for k in ("claude-opus", "claude-sonnet", "claude-haiku", "anthropic/claude"))


def _get_promoted_adapter(role: str) -> str | None:
    """Get promoted LoRA adapter path for an agent role, if one exists."""
    try:
        from app.training_pipeline import list_adapters
        from pathlib import Path
        for adapter in list_adapters():
            if adapter.promoted and (role in adapter.agent_roles or "all" in adapter.agent_roles):
                if Path(adapter.adapter_path).exists():
                    return adapter.adapter_path
    except Exception:
        pass
    return None


class _AdapterLLM:
    """LLM wrapper that routes inference through host bridge MLX with a LoRA adapter.

    Drop-in replacement for crewai.LLM — implements the .call() interface.
    Used when a promoted adapter exists for the agent's role AND local mode
    is active (adapter inference only makes sense on the host Metal GPU).
    """

    def __init__(self, model: str, adapter_path: str, max_tokens: int = 4096):
        self.model = f"mlx-adapter/{model}"
        self._base_model = model
        self._adapter = adapter_path
        self._max_tokens = max_tokens

    def call(self, prompt, **kwargs) -> str:
        # _AdapterLLM is the only LLM call path in this codebase that doesn't
        # derive from CrewAI's BaseLLM, so CrewAI's event bus never fires
        # LLMCallCompletedEvent / LLMCallFailedEvent for it.  We emit the
        # activity heartbeat explicitly here so the progressive-timeout stall
        # detector in handle_task sees this path as alive.  (The fallback
        # branch below goes through a real crewai.LLM, which the event bus
        # DOES cover — so no second record is needed in that branch.)
        from app.rate_throttle import record_llm_activity
        try:
            from app.bridge_client import get_bridge
            bridge = get_bridge("specialist")
            if not bridge or not bridge.is_available():
                raise ConnectionError("Host bridge unavailable")
            result = bridge.mlx_generate(
                prompt=str(prompt)[:4000],
                model=self._base_model,
                adapter_path=self._adapter,
                max_tokens=self._max_tokens,
            )
            if "error" in result:
                raise RuntimeError(result["error"])
            record_llm_activity()
            return result.get("response", "")
        except Exception:
            # Record the failure-as-activity BEFORE falling back, so a task
            # that's legitimately in a retry cycle doesn't look silent.
            record_llm_activity()
            # Fall back to Ollama base model (no adapter)
            logger.debug("AdapterLLM falling back to Ollama", exc_info=True)
            from app.config import get_settings
            s = get_settings()
            LLM = _get_LLM_class()
            fallback = LLM(
                model=f"ollama/{s.local_model_default}",
                max_tokens=self._max_tokens,
                base_url=s.local_llm_base_url,
            )
            return str(fallback.call(prompt))

    # CrewAI compatibility — LLM is referenced via getattr in some places
    def __str__(self):
        return self.model


def _get_last(attr: str) -> str | None:
    return getattr(_tls, attr, None)


def _set_last(model: str | None, tier: str | None) -> None:
    _tls.last_model_name = model
    _tls.last_tier = tier


def create_commander_llm() -> LLM:
    """Create the Commander routing LLM using the resolver's pick.

    Previously this function hard-forced an Anthropic model — any
    non-Anthropic pick from the resolver was silently swapped to
    ``claude-sonnet-4.6``. That bypassed the whole point of the
    scoring resolver.

    Now we honour the resolver's choice and route to whichever
    provider owns the chosen model:
      * Anthropic  → Anthropic SDK (requires ANTHROPIC_API_KEY)
      * OpenRouter → OpenRouter API (requires OPENROUTER_API_KEY)
      * Ollama     → local inference
    If the chosen provider's key is missing, we fall through to the
    cheapest API-tier alternative with a valid key, and ultimately
    to the DeepSeek survival bootstrap.
    """
    from app.config import get_openrouter_api_key
    from app.llm_mode import get_mode

    settings = get_settings()
    mode = get_mode()
    model_name = get_default_for_role("commander", mode)
    entry = get_model(model_name) or {}

    provider = entry.get("provider")
    if provider == "anthropic":
        anthropic_key = get_anthropic_api_key()
        if anthropic_key:
            logger.info(f"create_commander_llm: resolved {model_name} (anthropic)")
            return _build_claude_llm(
                model_name, entry["model_id"], max_tokens=1024, role="commander",
                tier=entry.get("tier", "premium"),
                cost_out=entry.get("cost_output_per_m", 15.0),
            )
        logger.warning(
            "create_commander_llm: resolver picked %s but ANTHROPIC_API_KEY is missing",
            model_name,
        )
    elif provider == "openrouter":
        or_key = get_openrouter_api_key()
        if or_key:
            logger.info(f"create_commander_llm: resolved {model_name} (openrouter)")
            return _cached_llm(
                entry["model_id"], max_tokens=1024,
                base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
        logger.warning(
            "create_commander_llm: resolver picked %s but OPENROUTER_API_KEY is missing",
            model_name,
        )
    elif provider == "ollama":
        # Commander via local Ollama — model_id is "ollama_chat/..."
        logger.info(f"create_commander_llm: resolved {model_name} (ollama local)")
        return _cached_llm(entry["model_id"], max_tokens=1024)

    # Fall-through: pick the cheapest API-tier survivor whose key is set.
    logger.warning(
        "create_commander_llm: resolver pick %r unreachable, falling back",
        model_name,
    )
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        sonnet = get_model("claude-sonnet-4.6")
        if sonnet:
            return _build_claude_llm(
                "claude-sonnet-4.6", sonnet["model_id"], max_tokens=1024,
                role="commander",
                tier=sonnet.get("tier", "premium"),
                cost_out=sonnet.get("cost_output_per_m", 15.0),
            )
    or_key = get_openrouter_api_key()
    if or_key:
        deepseek = get_model("deepseek-v3.2")
        if deepseek:
            return _cached_llm(
                deepseek["model_id"], max_tokens=1024,
                base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
    return _cached_llm(
        "openrouter/deepseek/deepseek-chat", max_tokens=1024, api_key=or_key,
    )


def create_specialist_llm(
    max_tokens: int = 4096,
    role: str = "default",
    task_hint: str = "",
    force_tier: str | None = None,
    phase: str | None = None,
) -> LLM:
    """
    Create an LLM for a specialist role using the tier cascade.

    Behavior depends on current runtime mode (see app.llm_mode.get_mode):
      free      Local + OpenRouter-free only, Claude fallback if empty pool
      budget    Cascade local → cheap cloud APIs (~$1.5/M-out ceiling)
      balanced  Default. Cascade every tier, mild cost preference
      quality   Cascade every tier, strong preference for premium
      insane    Premium only, no cost ceiling, no local
      anthropic Anthropic-only (Haiku/Sonnet/Opus) line-up

    If force_tier is set (e.g. from difficulty-based routing), it overrides
    the default tier selection from llm_selector.

    `phase` (creative-mode only) is one of "diverge"/"discuss"/"converge".
    When set, phase-dependent sampling parameters (temperature/top_p/min_p/
    presence_penalty) are applied. When None, legacy behavior is preserved
    byte-for-byte — including LLM cache identity.
    """
    # Q7: thread-local last model/tier tracking
    from app.llm_mode import get_mode
    settings = get_settings()
    mode = get_mode()

    from app.llm_selector import select_model

    # ── Restrictive modes (free / budget / quality / insane / anthropic)
    #    constrain the candidate pool, then run the regular LLM selector
    #    inside that pool. Every role still gets its normal score-driven
    #    pick — the mode only narrows the shortlist.
    #
    #    ``balanced`` (default) gets the unconstrained selector + full
    #    tier-cascading fallback path below.
    if mode != "balanced":
        chosen = _pool_constrained_select(role, task_hint, mode, force_tier)
        if not chosen:
            logger.warning(
                "llm_factory: mode=%s has no usable model for role=%s, falling back to Claude",
                mode, role,
            )
            return _claude_fallback(role, max_tokens, phase=phase)
        model_name, entry = chosen
        return _build_from_entry(
            model_name, entry, max_tokens, role,
            phase=phase, mode=mode, settings=settings,
        )

    # ── BALANCED mode: unconstrained selector + full cascade (default) ──
    model_name = select_model(role, task_hint, force_tier=force_tier)
    entry = get_model(model_name)

    if not entry:
        logger.warning(f"llm_factory: model {model_name!r} not in catalog, falling back")
        return _claude_fallback(role, max_tokens, phase=phase)

    tier = entry["tier"]
    provider = entry["provider"]

    # ── HYBRID mode: full cascade ────────────────────────────────────
    # Try local Ollama first
    if tier == "local" and settings.local_llm_enabled:
        llm = _try_local(model_name, entry, max_tokens, role, phase=phase)
        if llm:
            # Stage 4.3 — race local vs API on short prompts (default OFF).
            return _maybe_race_wrap(llm, role, max_tokens, phase)
        # Local failed — try API tier
        if settings.api_tier_enabled:
            logger.info(f"llm_factory: local failed for role={role}, trying API tier")
            api_model = get_default_for_role(role, mode)
            api_entry = get_model(api_model)
            if api_entry and api_entry["tier"] in ("free", "budget", "mid"):
                llm = _try_api(api_model, api_entry, max_tokens, role, phase=phase)
                if llm:
                    return llm
        return _claude_fallback(role, max_tokens, phase=phase)

    # Try API tier (OpenRouter)
    if tier in ("free", "budget", "mid") and settings.api_tier_enabled:
        llm = _try_api(model_name, entry, max_tokens, role, phase=phase)
        if llm:
            return llm
        return _claude_fallback(role, max_tokens, phase=phase)

    # Premium tier (Anthropic or OpenRouter)
    if provider == "anthropic":
        return _create_anthropic(model_name, entry, max_tokens, role, phase=phase)
    elif provider == "openrouter":
        llm = _try_api(model_name, entry, max_tokens, role, phase=phase)
        if llm:
            return llm
        return _claude_fallback(role, max_tokens, phase=phase)

    return _claude_fallback(role, max_tokens, phase=phase)


def create_vetting_llm() -> LLM:
    """Vetting gate — uses the resolver's pick for the ``vetting`` role.

    The ``VETTING_MODEL`` env var is NOT consulted — it was a piece of
    hand-curation that bypassed the resolver and the overlay. If you
    need to pin vetting to a specific model, install a row in
    ``control_plane.role_assignments`` (via Signal / governance
    approval). The resolver + overlay are the single source of truth.
    """
    from app.config import get_openrouter_api_key
    from app.llm_mode import get_mode
    settings = get_settings()

    model_name = get_default_for_role("vetting", get_mode())
    entry = get_model(model_name) or {}

    provider = entry.get("provider") if entry else None

    if provider == "anthropic":
        anthropic_key = get_anthropic_api_key()
        if anthropic_key:
            logger.info(f"create_vetting_llm: resolved {model_name} (anthropic)")
            return _build_claude_llm(
                model_name, entry["model_id"], max_tokens=4096, role="vetting",
                tier=entry.get("tier", "premium"),
                cost_out=entry.get("cost_output_per_m", 15.0),
            )
    elif provider == "openrouter":
        or_key = get_openrouter_api_key()
        if or_key:
            logger.info(f"create_vetting_llm: resolved {model_name} (openrouter)")
            return _cached_llm(
                entry["model_id"], max_tokens=4096,
                base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
    elif provider == "ollama":
        logger.info(f"create_vetting_llm: resolved {model_name} (ollama local)")
        return _cached_llm(entry["model_id"], max_tokens=4096)

    # Fall-through to bootstrap survivors.
    logger.warning(
        "create_vetting_llm: resolver pick %r unreachable, falling back", model_name,
    )
    anthropic_key = get_anthropic_api_key()
    if anthropic_key:
        return _build_claude_llm(
            "claude-sonnet-4.6", "anthropic/claude-sonnet-4-6",
            max_tokens=4096, role="vetting",
        )
    or_key = get_openrouter_api_key()
    fallback = get_model("deepseek-v3.2") or {}
    model_id = fallback.get("model_id", "openrouter/deepseek/deepseek-chat")
    return _cached_llm(
        model_id, max_tokens=4096,
        base_url="https://openrouter.ai/api/v1", api_key=or_key,
    )


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
    return _build_claude_llm(
        "claude-sonnet-4.6", "anthropic/claude-sonnet-4-6",
        max_tokens=256, role="cheap-vetting",
    )


class _RacingLLM:
    """Stage 4.3 — cascade race wrapper (hybrid mode, short prompts only).

    On `.call(prompt)`:
      * if len(prompt) >= threshold, delegates to primary (cost-safe).
      * otherwise races primary + secondary, returns first non-error.

    Invariant: primary is always the Ollama local LLM; secondary is the
    OpenRouter fallback. Both are crewai.LLM objects (same .call() contract).

    Gated by settings.cascade_race_short — default False.
    """

    def __init__(self, primary, secondary, *, threshold_chars: int, timeout_s: float):
        self._primary = primary
        self._secondary = secondary
        self._threshold = threshold_chars
        self._timeout = timeout_s
        self.model = getattr(primary, "model", "racing-llm")

    def __str__(self):
        return f"race({self._primary}, {self._secondary})"

    def call(self, prompt, **kwargs):
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        if len(prompt_str) >= self._threshold:
            return self._primary.call(prompt, **kwargs)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm-race") as ex:
            f_primary = ex.submit(self._primary.call, prompt, **kwargs)
            f_secondary = ex.submit(self._secondary.call, prompt, **kwargs)
            done, pending = wait(
                {f_primary, f_secondary}, timeout=self._timeout,
                return_when=FIRST_COMPLETED,
            )
            for p in pending:
                p.cancel()
            for f in done:
                try:
                    return f.result(timeout=0.1)
                except Exception:
                    continue
            # Both futures failed or timed out in the first window — give
            # primary a bit more time, then fall through to secondary.
            try:
                return f_primary.result(timeout=2.0)
            except Exception:
                return f_secondary.result(timeout=2.0)

    # Some callers invoke LLM attributes directly; forward to primary.
    def __getattr__(self, name):
        return getattr(self._primary, name)


def _maybe_race_wrap(primary, role: str, max_tokens: int, phase: str | None):
    """If cascade_race_short is enabled, return a _RacingLLM wrapping primary
    with an API-tier secondary. On any failure, returns primary unwrapped.
    """
    try:
        settings = get_settings()
        if not getattr(settings, "cascade_race_short", False):
            return primary
        if not settings.api_tier_enabled:
            return primary
        from app.llm_mode import get_mode
        api_model = get_default_for_role(role, get_mode())
        api_entry = get_model(api_model)
        if not (api_entry and api_entry.get("tier") in ("free", "budget", "mid")):
            return primary
        secondary = _try_api(api_model, api_entry, max_tokens, role, phase=phase)
        if secondary is None:
            return primary
        threshold_chars = int(settings.cascade_race_token_threshold * 4)  # ~4 chars/tok
        timeout_s = float(settings.cascade_race_timeout_s)
        return _RacingLLM(primary, secondary,
                          threshold_chars=threshold_chars, timeout_s=timeout_s)
    except Exception:
        return primary


def is_using_local() -> bool:
    return _get_last("last_tier") == "local"

def is_using_api_tier() -> bool:
    return _get_last("last_tier") in ("budget", "mid")

def get_last_model() -> str | None:
    return _get_last("last_model_name")

def get_last_tier() -> str | None:
    return _get_last("last_tier")


# INSANE mode now delegates to resolve_role_default with cost_mode="quality".
# The resolver already picks the strongest model in the premium tier that
# meets the role's constraints — exactly what INSANE used to hardcode.
# No more static role-map: if Opus 4.8 lands tomorrow it becomes the
# INSANE-mode commander automatically.


def _sampling(phase: str | None, provider: str) -> tuple[dict, str]:
    """Return (llm_kwargs, cache_key) for phase+provider. ({}, '') when phase is None.

    Reads the latest affect snapshot via `app.affect.core.latest_affect()`
    and forwards it to `build_llm_kwargs` so phase-aware
    temperature / top_p modulation actually fires on the LLM hot path.
    Affect import is lazy + exception-safe so the sampling path stays
    byte-identical to legacy behaviour when the affect layer is
    disabled or hasn't yet computed an affect frame.
    """
    if phase is None:
        return {}, ""
    from app.llm_sampling import build_llm_kwargs, sampling_cache_key

    affect_state: dict | None = None
    affect_key_part = ""
    try:
        from app.affect.core import latest_affect
        s = latest_affect()
        if s is not None:
            affect_state = s.to_dict()
            # Coarse cache-key bucket: round V/A to 0.1 so equivalent
            # affect states share kwargs cache entries instead of
            # producing per-call uniques. Attractor name is stable
            # within a band, so include it too.
            v = round(float(affect_state.get("valence", 0.0)), 1)
            a = round(float(affect_state.get("arousal", 0.0)), 1)
            attractor = str(affect_state.get("attractor", "neutral"))[:16]
            affect_key_part = f"|{attractor}|v={v}|a={a}"
    except Exception:
        # Affect layer not installed or first call before any
        # POST_LLM_CALL — fall through to legacy unmodulated path.
        pass

    base_key = sampling_cache_key(phase, provider)
    cache_key = base_key + affect_key_part if base_key else base_key
    return build_llm_kwargs(phase, provider, affect_state), cache_key


# ── Mode pool filters ────────────────────────────────────────────────────────
#
# Every non-hybrid mode narrows the candidate pool the selector scores across.
# Shape: {mode: {"tiers": [...] | None, "provider": str | None}}.
# "tiers=None" means any tier is acceptable.
def _mode_pool(mode: str) -> dict[str, object]:
    """Return ``{"tiers": [...], "provider": str | None}`` for a mode.

    Reads from the unified policy dicts in ``app.llm_catalog``. The
    factory no longer maintains its own shadow table — the policy is
    defined once, in the catalog, and this function is just a thin
    adapter keeping the existing ``_pool_constrained_select`` signature.
    """
    from app.llm_catalog import (
        _MODE_TIER_WHITELIST, _MODE_PROVIDER_WHITELIST, _normalize_mode,
    )
    canon = _normalize_mode(mode)
    tiers = _MODE_TIER_WHITELIST.get(canon)
    provider_set = _MODE_PROVIDER_WHITELIST.get(canon)
    # _pool_constrained_select expects a list (or None) for tiers and a
    # single string (or None) for provider. Anthropic is the only mode
    # with a provider lock today, and its whitelist is a single value.
    provider: str | None = None
    if provider_set is not None and len(provider_set) == 1:
        provider = next(iter(provider_set))
    return {
        "tiers": list(tiers) if tiers is not None else None,
        "provider": provider,
    }


def _entry_in_pool(entry: dict, tiers: list[str] | None, provider: str | None) -> bool:
    """Return True when ``entry`` satisfies a mode's pool filter."""
    if tiers is not None and entry.get("tier") not in tiers:
        return False
    if provider is not None and entry.get("provider") != provider:
        return False
    return True


def _pool_constrained_select(
    role: str,
    task_hint: str,
    mode: str,
    force_tier: str | None,
) -> tuple[str, dict] | None:
    """Run the LLM selector inside the active mode's candidate pool.

    Strategy:
      1. Ask the normal selector for its preferred model. If it already sits
         in the allowed pool, use it.
      2. Otherwise re-run the selector with ``force_tier`` for each allowed
         tier; first match wins.
      3. Last resort: catalog scan scored by the role's ``strengths`` map
         (falling back to ``general``). Ties break on context size.
    Returns (name, entry) or None if nothing in the pool is usable.
    """
    from app.llm_catalog import CATALOG
    from app.llm_selector import select_model as _select

    pool = _mode_pool(mode)
    tiers = pool.get("tiers")  # type: ignore[assignment]
    provider = pool.get("provider")  # type: ignore[assignment]

    # 1. Selector's default pick
    try:
        base_name = _select(role, task_hint, force_tier=force_tier)
    except Exception:
        base_name = None
    if base_name:
        base_entry = get_model(base_name)
        if base_entry and _entry_in_pool(base_entry, tiers, provider):
            return base_name, base_entry

    # 2. Try forcing each allowed tier
    if tiers:
        for forced in tiers:
            try:
                alt = _select(role, task_hint, force_tier=forced)
            except Exception:
                continue
            if not alt:
                continue
            alt_entry = get_model(alt)
            if alt_entry and _entry_in_pool(alt_entry, tiers, provider):
                return alt, alt_entry

    # 3. Catalog walk scored by role strengths
    candidates = [
        (name, dict(entry))
        for name, entry in CATALOG.items()
        if _entry_in_pool(entry, tiers, provider)
    ]
    if not candidates:
        return None

    def _score(name_entry: tuple[str, dict]) -> tuple[float, int]:
        _name, entry = name_entry
        strengths = entry.get("strengths", {}) or {}
        role_score = float(strengths.get(role, 0) or 0.0)
        general = float(strengths.get("general", 0) or 0.0)
        primary = role_score if role_score > 0 else general
        return (primary, int(entry.get("context") or 0))

    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _build_from_entry(
    model_name: str,
    entry: dict,
    max_tokens: int,
    role: str,
    *,
    phase: str | None,
    mode: str,
    settings,
) -> LLM:
    """Instantiate the appropriate LLM client for a (name, entry) pick.

    Routes by tier + provider; local Ollama for tier=="local", Anthropic
    SDK for Anthropic entries, OpenRouter otherwise. When the preferred
    route isn't available the function falls back to Claude so the system
    stays functional rather than raising — a warning is emitted so the
    operator notices.
    """
    tier = entry.get("tier", "")
    provider = entry.get("provider", "")

    if tier == "local" and settings.local_llm_enabled:
        llm = _try_local(model_name, entry, max_tokens, role, phase=phase)
        if llm:
            return llm
        logger.info(
            "llm_factory: mode=%s chose local model %s but Ollama unavailable",
            mode, model_name,
        )

    if provider == "anthropic":
        return _create_anthropic(model_name, entry, max_tokens, role, phase=phase)

    if settings.api_tier_enabled:
        llm = _try_api(model_name, entry, max_tokens, role, phase=phase)
        if llm:
            return llm

    logger.warning(
        "llm_factory: mode=%s could not instantiate %s (tier=%s provider=%s), falling back to Claude",
        mode, model_name, tier, provider,
    )
    return _claude_fallback(role, max_tokens, phase=phase)


# _insane_mode_select was removed when INSANE mode moved to the uniform
# pool-filter path. If any external module imported it, re-add a thin
# shim here that delegates to _pool_constrained_select + _build_from_entry.


def _try_local(model_name: str, entry: dict, max_tokens: int, role: str, phase: str | None = None) -> LLM | None:
    # Q7: thread-local last model/tier tracking
    if not circuit_breaker.is_available("ollama"):
        logger.info(f"llm_factory: skipping Ollama (circuit open)")
        return None

    # ── Adapter-aware inference (T4-14): if a promoted LoRA adapter exists
    #    for this role AND the host bridge's MLX is available, prefer the
    #    _AdapterLLM path which runs on Metal GPU with the fine-tune applied.
    adapter_path = _get_promoted_adapter(role or "default")
    if adapter_path:
        try:
            from app.bridge_client import get_bridge
            bridge = get_bridge("specialist")
            if bridge and bridge.is_available():
                status = bridge.mlx_status()
                if status.get("available"):
                    _set_last(model_name, "local")
                    logger.info(
                        f"llm_factory: role={role} → MLX ADAPTER "
                        f"{adapter_path} (base={model_name})"
                    )
                    return _AdapterLLM(model_name, adapter_path, max_tokens)
        except Exception:
            logger.debug("adapter selection failed, falling back to Ollama",
                         exc_info=True)

    try:
        from app.ollama_native import spawn_model
        start = time.monotonic()
        url = spawn_model(model_name)
        spawn_ms = int((time.monotonic() - start) * 1000)
        if url:
            _set_last(model_name, "local")
            circuit_breaker.record_success("ollama")
            logger.info(f"llm_factory: role={role} → LOCAL {model_name} at {url} (spawn: {spawn_ms}ms)")
            extra, key = _sampling(phase, "ollama")
            return _cached_llm(entry["model_id"], max_tokens=max_tokens,
                               sampling_key=key, base_url=url, **extra)
        circuit_breaker.record_failure("ollama")
    except Exception as exc:
        circuit_breaker.record_failure("ollama")
        logger.warning(f"llm_factory: local {model_name} failed: {exc}")
    return None


def _try_api(model_name: str, entry: dict, max_tokens: int, role: str, phase: str | None = None) -> LLM | None:
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
        extra, key = _sampling(phase, "openrouter")
        return _cached_llm(entry["model_id"], max_tokens=max_tokens,
                           sampling_key=key,
                           base_url="https://openrouter.ai/api/v1", api_key=api_key, **extra)
    except Exception as exc:
        circuit_breaker.record_failure("openrouter")
        logger.warning(f"llm_factory: API {model_name} failed: {exc}")
        _set_last(None, None)
    return None


# ── Anthropic-direct LLM factory with credit-exhausted failover ─────────
#
# When the Anthropic API returns
#     400 invalid_request_error "Your credit balance is too low..."
# we fail over to the same Claude model served via OpenRouter.  Authoritative
# state lives in circuit_breaker["anthropic_credits"] (threshold 1, 3600s
# cooldown) — tripping is idempotent and visible to every LLM factory in the
# process; auto-recovery happens when the breaker transitions to HALF_OPEN
# and the next Anthropic probe succeeds.  No monkey-patching, no global
# mutable flags: just a typed subclass (CreditAwareAnthropicCompletion) and
# the existing circuit-breaker infrastructure.


def _anthropic_to_openrouter_model_id(anthropic_model_id: str) -> str:
    """Translate an Anthropic-SDK model id (dashes in version) into the
    OpenRouter equivalent.

    AA/Anthropic emits  : anthropic/claude-sonnet-4-6
    OpenRouter expects  : openrouter/anthropic/claude-sonnet-4.6
    """
    import re as _re
    slug = anthropic_model_id
    if slug.startswith("anthropic/"):
        slug = slug[len("anthropic/"):]
    # Claude slug pattern: claude-<family>-<major>-<minor>.  Restore the
    # single "-<major>-<minor>" tail to dots so it matches OpenRouter's
    # naming.  e.g. claude-sonnet-4-6 → claude-sonnet-4.6
    slug = _re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", slug)
    return f"openrouter/anthropic/{slug}"


def _build_claude_via_openrouter(
    model_name: str,
    model_id: str,
    max_tokens: int,
    *,
    role: str,
    phase: str | None,
    tier: str = "premium",
    cost_out: float = 15.0,
) -> "LLM":
    """Build a Claude LLM routed through OpenRouter.

    Used in two places:
      * Direct substitute when the anthropic_credits breaker is OPEN
      * Lazy fallback target built by CreditAwareAnthropicCompletion
        on the first mid-call 400 we see
    """
    from app.config import get_openrouter_api_key
    or_key = get_openrouter_api_key()
    if not or_key:
        raise RuntimeError(
            "Anthropic credits exhausted AND OPENROUTER_API_KEY is unset — "
            "cannot serve Claude requests. Top up Anthropic or set "
            "OPENROUTER_API_KEY to enable the failover route."
        )
    or_model_id = _anthropic_to_openrouter_model_id(model_id)
    _set_last(f"{model_name} (via OpenRouter)", tier)
    logger.info(
        "llm_factory: role=%s → OPENROUTER %s (~$%.2f/Mo; anthropic_credits breaker=%s)",
        role, or_model_id, cost_out,
        circuit_breaker.get_breaker("anthropic_credits").state,
    )
    extra, sample_key = _sampling(phase, "openrouter")
    return _cached_llm(
        or_model_id, max_tokens=max_tokens, sampling_key=sample_key,
        base_url="https://openrouter.ai/api/v1", api_key=or_key, **extra,
    )


def _build_claude_llm(
    model_name: str,
    model_id: str,
    max_tokens: int,
    *,
    role: str,
    phase: str | None = None,
    tier: str = "premium",
    cost_out: float = 15.0,
) -> "LLM":
    """The single, elegant Claude factory for this module.

    Routing rule:
      * ``circuit_breaker["anthropic_credits"]`` OPEN
          → direct Anthropic is known-unavailable; build via OpenRouter now.
      * else
          → build a CreditAwareAnthropicCompletion (proper BaseLLM subclass,
            passes Agent Pydantic validation) with an injected fallback
            factory.  If the first call fails with credit-exhausted the
            subclass trips the breaker, builds the OR equivalent, and
            retries transparently.  All subsequent calls on that instance
            use the OR path directly.

    This is the only entry point for Anthropic-direct LLM construction
    in this module.  Every caller (``_create_anthropic``,
    ``_claude_fallback``, ``create_commander_llm``, ``create_vetting_llm``,
    ``create_cheap_vetting_llm``) funnels through here so the failover
    policy is applied uniformly.
    """
    # Lazy import: CreditAwareAnthropicCompletion depends on crewai.LLM
    # which we defer per the module's cold-boot discipline (see
    # `_get_LLM_class`).  Putting the import here keeps the llm_factory
    # import graph flat.
    from app.llms.credit_aware_anthropic import CreditAwareAnthropicCompletion

    def _or_fallback():
        return _build_claude_via_openrouter(
            model_name, model_id, max_tokens,
            role=role, phase=phase, tier=tier, cost_out=cost_out,
        )

    if not circuit_breaker.is_available("anthropic_credits"):
        logger.info(
            "llm_factory: role=%s → OpenRouter Claude (anthropic_credits "
            "breaker OPEN, %0.0fs to reprobe)",
            role,
            circuit_breaker.get_breaker("anthropic_credits").seconds_until_half_open(),
        )
        return _or_fallback()

    _set_last(model_name, tier)
    logger.info(
        "llm_factory: role=%s → ANTHROPIC %s ($%.2f/Mo) + credit-aware failover",
        role, model_name, cost_out,
    )
    extra, sample_key = _sampling(phase, "anthropic")

    # Go through _cached_llm with a CreditAware builder — entries get
    # keyed as (builder=CreditAware, model_id, max_tokens, ...) so they
    # don't collide with default crewai.LLM entries for the same model.
    # Cache-safe because the subclass consults the credit breaker on
    # every call (no sticky per-instance failover state that would
    # break auto-recovery after a shared cached hand-off).
    def _credit_aware_builder(mid: str, mt: int, **kw) -> CreditAwareAnthropicCompletion:
        llm = CreditAwareAnthropicCompletion(model=mid, max_tokens=mt, **kw)
        return llm.set_fallback_factory(_or_fallback)

    return _cached_llm(
        model_id, max_tokens,
        sampling_key=sample_key,
        llm_builder=_credit_aware_builder,
        api_key=get_anthropic_api_key(),
        **extra,
    )


def _create_anthropic(
    model_name: str,
    entry: dict,
    max_tokens: int,
    role: str,
    phase: str | None = None,
) -> "LLM":
    """Build a specialist-tier Anthropic LLM with credit-aware failover."""
    return _build_claude_llm(
        model_name, entry["model_id"], max_tokens,
        role=role, phase=phase,
        tier=entry.get("tier", "premium"),
        cost_out=entry.get("cost_output_per_m", 15.0),
    )


def _claude_fallback(
    role: str,
    max_tokens: int,
    phase: str | None = None,
) -> "LLM":
    """Final fallback: Claude Sonnet if Anthropic is reachable (including via
    OpenRouter), else DeepSeek via OpenRouter as the survival bootstrap.
    """
    from app.config import get_openrouter_api_key

    anthropic_key = get_anthropic_api_key()
    or_key = get_openrouter_api_key()

    # Preferred: Claude (direct or via OR — the subclass picks the right
    # path based on the breaker state).
    if anthropic_key or (or_key and not circuit_breaker.is_available("anthropic_credits")):
        return _build_claude_llm(
            "claude-sonnet-4.6", "anthropic/claude-sonnet-4-6", max_tokens,
            role=role, phase=phase, tier="premium", cost_out=15.0,
        )
    # If OR has Claude-capable routing but we have no Anthropic key, still
    # go through the Claude factory so the breaker logic applies.
    if or_key:
        return _build_claude_llm(
            "claude-sonnet-4.6", "anthropic/claude-sonnet-4-6", max_tokens,
            role=role, phase=phase, tier="premium", cost_out=15.0,
        )

    # Survival bootstrap: no Anthropic key and no OpenRouter key?
    # Something is misconfigured.  Emit a non-Claude model that might
    # work if any OR shadow key is configured elsewhere.
    _set_last("deepseek-v3.2", "budget")
    logger.warning(
        "llm_factory: role=%s → FALLBACK deepseek-v3.2 "
        "(no Claude option available: both ANTHROPIC_API_KEY and "
        "OPENROUTER_API_KEY are unset).", role,
    )
    extra, key = _sampling(phase, "openrouter")
    return _cached_llm(
        "openrouter/deepseek/deepseek-chat",
        max_tokens=max_tokens, sampling_key=key,
        base_url="https://openrouter.ai/api/v1", api_key=or_key, **extra,
    )


# ── Provider health check for graceful degradation ──────────────────────────

_all_providers_exhausted = False
_exhaustion_alerted = False


def check_all_providers_health() -> bool:
    """Return True if at least one LLM provider is available.

    If ALL circuit breakers are OPEN, returns False. The caller (orchestrator)
    is responsible for force-probing and user communication — this function
    does NOT send Signal alerts because circuit-breaker state often reflects
    background-task noise, not actual provider outages.
    """
    global _all_providers_exhausted
    from app.circuit_breaker import is_available

    anthropic_ok = is_available("anthropic")
    openrouter_ok = is_available("openrouter")
    ollama_ok = is_available("ollama")

    any_available = anthropic_ok or openrouter_ok or ollama_ok

    if not any_available and not _all_providers_exhausted:
        _all_providers_exhausted = True
        logger.warning(
            "All LLM circuit breakers OPEN — orchestrator will force-probe "
            "(anthropic=%s, openrouter=%s, ollama=%s)",
            "open", "open", "open",
        )
    elif any_available and _all_providers_exhausted:
        _all_providers_exhausted = False
        logger.info("LLM provider recovered — circuit breakers back to normal")

    return any_available
