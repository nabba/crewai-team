"""
llm_selector.py — Cost-aware, capability-aware model selection.

Selection algorithm:
  1. Check for env override (ROLE_MODEL_RESEARCH=kimi-k2.5)
  2. Get default model for role + cost_mode from catalog
  3. If task_hint present, detect task type and potentially override
  4. Apply special rules (multimodal → Kimi, parallel → budget tier)
  5. Check benchmark history for performance adjustments
  6. Verify model availability (local: Ollama, API: key present)
  7. Return best available model name
"""

import contextvars
import logging
import os

from app.config import get_settings
from app.llm_catalog import (
    CATALOG, TASK_ALIASES, ROLE_DEFAULTS,
    get_model, get_default_for_role, get_candidates_by_tier,
    canonical_task_type,
)
from app.llm_benchmarks import get_scores

logger = logging.getLogger(__name__)


# ── Active task-difficulty tracking (2026-04-26) ──────────────────────────
#
# The orchestrator's _run_crew sets this at the start of a task run.
# select_model reads it as a fallback when force_tier isn't passed
# explicitly — which is exactly the path sub-agents take when CrewAI's
# delegate_work_to_coworker spawns a "Web Research Specialist" inside
# the coordinator. The coordinator gets force_tier from
# difficulty_to_tier; the sub-agents historically did not, and that's
# how research at d=8 ended up calling gemma-4-31b-it (budget tier).
#
# ContextVar instead of threading.local because CrewAI's tool execution
# can hop coroutines / threads through asyncio.Task copies — ContextVar
# values are inherited by copy_context() automatically, threading.local
# isn't.

_active_difficulty: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_active_difficulty", default=None,
)


def set_active_difficulty(difficulty: int | None) -> object:
    """Bind the active task difficulty for the duration of one crew
    dispatch. Returns the reset token — caller MUST pass it to
    ``reset_active_difficulty`` in a ``finally`` block.
    """
    return _active_difficulty.set(difficulty)


def reset_active_difficulty(token: object) -> None:
    try:
        _active_difficulty.reset(token)  # type: ignore[arg-type]
    except (LookupError, ValueError):
        # Token from a different context (rare — defensive cleanup).
        _active_difficulty.set(None)


def get_active_difficulty() -> int | None:
    return _active_difficulty.get()


# Role × difficulty → minimum tier. Empty default means "use whatever
# the cost mode picks". Tighter thresholds for research because real
# research requires multi-step persistence that budget-tier models give
# up on too quickly (the 2026-04-25 gemma-4-31b-it research-at-d8 case).
#
# 2026-05-02 audit (H4): synthesis added.  The orchestrator's vetting
# + critic + reflexion paths and base_crew's auto-skill distillation
# all call create_specialist_llm(role="synthesis") without passing
# force_tier.  Without an explicit floor, synthesis falls to budget-
# tier (gemma-4-31b-it at $0.40/Mo) regardless of how premium the
# parent crew was.  The 2026-05-02 12:12 dispatch ran a force_tier=
# premium coding crew through a budget-tier synthesis LLM that mangled
# the tool-call summary into "malformed tool invocation leakage" —
# vetting then flagged the whole thing as failed and rerouted.  The
# floor below makes synthesis match the crew's quality bar.
_ROLE_DIFFICULTY_TIER_FLOOR: dict[str, list[tuple[int, str]]] = {
    # Sorted descending by difficulty so the first match wins.
    "research":  [(8, "premium"), (7, "mid")],
    "writing":   [(9, "premium")],
    "coding":    [(9, "premium"), (7, "mid")],
    "synthesis": [(8, "premium"), (7, "mid")],
}


def _resolve_difficulty_tier_floor(role: str, difficulty: int | None) -> str | None:
    """Lookup the minimum tier for ``(role, difficulty)`` from the policy
    table. Returns None when there's no floor (most cases)."""
    if difficulty is None:
        return None
    rules = _ROLE_DIFFICULTY_TIER_FLOOR.get(role, [])
    for threshold, tier in rules:
        if difficulty >= threshold:
            return tier
    return None


def difficulty_to_tier(difficulty: int, mode: str) -> str | None:
    """Map task difficulty (1-10) to a preferred model tier.

    Returns None for medium difficulty (4-7) to let the default
    catalog/cost_mode logic decide.

    NOTE: "local" tier is ONLY for sentience hooks and background tasks.
    User-facing crews always use at least "budget" tier because small
    local models (llama3.1:8b) don't handle tool calls properly.
    """
    if mode == "insane":
        return "premium"
    if difficulty <= 3:
        return "budget"  # Even easy tasks need a competent model for tool use
    elif difficulty >= 8:
        return "premium"
    return None  # medium → use default catalog logic

_MULTIMODAL_MODELS = [name for name, info in CATALOG.items() if info.get("multimodal")]


def detect_task_type(role: str, task_hint: str = "") -> str:
    """Thin delegator to :func:`app.llm_catalog.canonical_task_type`.

    Kept for backwards compatibility; new code should call the catalog
    helper directly.
    """
    return canonical_task_type(role=role, task_hint=task_hint)


_cached_ollama_url: str | None = None

def _get_ollama_url() -> str:
    """Get Ollama API URL. Probes actual connectivity (cached after first call)."""
    global _cached_ollama_url
    if _cached_ollama_url:
        return _cached_ollama_url
    import requests
    # Try host.docker.internal first (inside Docker), then localhost (native)
    for host in ("http://host.docker.internal:11434", "http://localhost:11434"):
        try:
            requests.get(f"{host}/api/tags", timeout=2)
            _cached_ollama_url = host
            return host
        except Exception:
            pass
    return "http://host.docker.internal:11434"


def _get_ollama_memory_usage() -> float:
    """Query Ollama for total VRAM used by loaded models (GB)."""
    try:
        import requests
        resp = requests.get(f"{_get_ollama_url()}/api/ps", timeout=2)
        data = resp.json()
        total = sum(m.get("size", 0) for m in data.get("models", []))
        return total / (1024 ** 3)
    except Exception:
        return 0.0


def _get_system_ram_gb() -> float:
    """Get total unified memory (Apple Silicon).

    Inside Docker, cgroup reports only container limit (e.g. 8GB).
    Ollama runs on the HOST, so we need HOST RAM for model selection.
    Use Ollama API to infer: if Ollama can load a 29GB model, host has >=48GB.
    """
    # First: try native sysctl (if running on host)
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2).decode().strip()
        return int(out) / (1024 ** 3)
    except Exception:
        pass
    # Inside Docker: infer from Ollama loaded models
    # If Ollama has 29GB loaded, host must have at least 48GB
    try:
        import requests
        resp = requests.get(f"{_get_ollama_url()}/api/ps", timeout=2)
        total_loaded = sum(m.get("size", 0) for m in resp.json().get("models", []))
        loaded_gb = total_loaded / (1024 ** 3)
        if loaded_gb > 20:
            return 48.0  # M4 Max
        elif loaded_gb > 10:
            return 36.0  # M4 Pro
        else:
            return 24.0  # M4 base
    except Exception:
        pass
    return 48.0  # Default: assume M4 Max


# Headroom: always keep this much RAM free for OS + embeddings + Docker
_RAM_HEADROOM_GB = 16.0


def _select_local_resource_aware(
    tier_candidates: list[tuple[str, float]],
    role: str,
    task_type: str,
    get_model_fn: callable,
    settings,
) -> str | None:
    """Pick the best local model that fits in available memory with headroom.

    Checks actual Ollama VRAM usage, system RAM, and model size_gb from catalog.
    Prefers models already loaded (zero load time) over unloaded ones.
    Respects tool-use requirements.
    """
    total_ram = _get_system_ram_gb()
    current_usage = _get_ollama_memory_usage()
    available = total_ram - _RAM_HEADROOM_GB

    # Check which models are already loaded in Ollama
    loaded_models = set()
    try:
        import requests
        resp = requests.get(f"{_get_ollama_url()}/api/ps", timeout=2)
        loaded_models = {m.get("name", "") for m in resp.json().get("models", [])}
    except Exception:
        pass

    _ROLES_NEEDING_TOOLS = {"coding", "research", "writing", "media", "self_improve", "critic"}
    needs_tools = role in _ROLES_NEEDING_TOOLS

    best_loaded = None      # Best model already in VRAM
    best_fits = None        # Best model that fits in available RAM

    for name, score in tier_candidates:
        entry = get_model_fn(name)
        if not entry:
            continue
        if needs_tools and entry.get("supports_tools") is False:
            continue

        model_size = entry.get("size_gb", 0) * 1.6  # ~1.6x for KV cache overhead
        already_loaded = name in loaded_models or any(name in lm for lm in loaded_models)

        if already_loaded:
            if best_loaded is None or score > tier_candidates[0][1]:
                best_loaded = name
        elif model_size <= available:
            if best_fits is None:
                best_fits = name

    # Prefer already-loaded model (zero load time, no memory pressure change)
    if best_loaded:
        logger.info(f"llm_selector: local resource-aware → {best_loaded} (already loaded, "
                    f"RAM: {current_usage:.1f}/{total_ram:.0f}GB, headroom: {_RAM_HEADROOM_GB}GB)")
        return best_loaded

    # Otherwise pick the best model that fits
    if best_fits:
        entry = get_model_fn(best_fits)
        model_size = entry.get("size_gb", 0) * 1.6
        logger.info(f"llm_selector: local resource-aware → {best_fits} ({model_size:.0f}GB needed, "
                    f"available: {available:.0f}GB)")
        return best_fits

    return None


# Minimum INTERNAL benchmark samples required before a model is
# eligible as a pareto-demotion target.  2026-04-24 outage: the
# selector demoted to ``stepfun/step-3.5-flash`` — zero internal
# observations, only a scraped external rank — and that model
# network-stalled mid-task for 262 s, breaking handle_task.  Below
# this floor, the model is "unknown quality" (which is NOT the
# same as "low quality"), so we refuse to pareto-demote to it.
_MIN_SAMPLES_FOR_DEMOTION = 5


def _pareto_cheaper_alternative(
    default_model: str,
    default_entry: dict,
    default_score: float,
    bench_scores: dict[str, float],
    get_model_fn,
    *,
    quality_gap: float = 0.05,
    task_type: str | None = None,
) -> str | None:
    """Return a catalog key that Pareto-dominates ``default_model`` on
    (cost, quality) — cheaper AND close-or-better in benchmark score.

    Pareto operates inside the selector's outer envelope (tier gating,
    availability): we only consider candidates with bench scores at
    least ``default_score - quality_gap`` and ``cost_output_per_m``
    strictly less than the default. Ties go to whichever is cheapest.
    Returns None when nothing dominates.

    Reliability gates (added 2026-04-25):

    * **Sample count** — the candidate must have at least
      ``_MIN_SAMPLES_FOR_DEMOTION`` internal benchmark records for
      this task type.  Stops the selector from demoting to a model
      we've never actually tried, just because its scraped external
      rank is slightly higher than the default's unknown one.
    * **Circuit breaker** — the candidate's per-model breaker
      (``circuit_breaker["model:<name>"]``) must not be OPEN.  A
      model that has repeatedly network-failed in this session is
      ineligible until its cooldown elapses.
    """
    if not bench_scores:
        return None
    default_cost = float(default_entry.get("cost_output_per_m", 0) or 0)
    floor = default_score - quality_gap

    # Pull sample counts + breaker-open set up front so we make one
    # fast pass through the candidate list instead of N DB hits.
    # ``task_type=None`` means the caller can't feed the sample gate;
    # in that case we preserve legacy behaviour (no sample check) —
    # the breaker gate still applies.
    apply_sample_gate = task_type is not None
    sample_counts: dict[str, int] = {}
    if apply_sample_gate:
        try:
            from app.llm_benchmarks import get_sample_counts
            sample_counts = get_sample_counts(task_type)
        except Exception:
            sample_counts = {}
    try:
        from app.circuit_breaker import get_breaker
    except Exception:
        get_breaker = None

    best: tuple[str, float] | None = None  # (name, cost)
    for name, score in bench_scores.items():
        if name == default_model:
            continue
        if score < floor:
            continue
        # Reliability gate 1: require minimum internal observations
        # when a task type is provided.
        if apply_sample_gate and sample_counts.get(name, 0) < _MIN_SAMPLES_FOR_DEMOTION:
            logger.debug(
                "pareto: skip %r — only %d internal samples (need %d)",
                name, sample_counts.get(name, 0), _MIN_SAMPLES_FOR_DEMOTION,
            )
            continue
        # Reliability gate 2: skip if a per-model circuit breaker is open.
        if get_breaker is not None:
            try:
                if get_breaker(f"model:{name}").is_open():
                    logger.info(
                        "pareto: skip %r — per-model circuit breaker OPEN",
                        name,
                    )
                    continue
            except Exception:
                pass  # breaker lookup failure = fail-open
        entry = get_model_fn(name)
        if not entry:
            continue
        cost = float(entry.get("cost_output_per_m", 0) or 0)
        if cost >= default_cost:
            continue
        if best is None or cost < best[1]:
            best = (name, cost)
    return best[0] if best else None


def select_model(
    role: str,
    task_hint: str = "",
    max_ram_gb: float = 48.0,
    force_tier: str | None = None,
    *,
    expected_input_tokens: int = 2000,
    expected_output_tokens: int = 1500,
    budget_usd: float | None = None,
) -> str:
    """Resolve the catalog key for a role/task given the current cost
    mode, overlay assignments, telemetry, and external ranks.

    Phase 4 additions:
      - ``expected_input_tokens``/``expected_output_tokens`` let callers
        hint the token volume of the upcoming call. Used together with
        ``budget_usd`` to demote premium-tier defaults whose estimated
        cost would exceed the budget, provided a cheaper alternative
        scores within ``quality_gap`` of the default.
      - Cross-tier Pareto kicks in when blended benchmark scores exist
        and the default is API-tier (local/free paths untouched).
    """
    settings = get_settings()

    # Step 1: Environment override
    env_key = f"ROLE_MODEL_{role.upper()}"
    env_override = os.environ.get(env_key)
    if env_override and env_override in CATALOG:
        logger.info(f"llm_selector: {env_key}={env_override} (env override)")
        return env_override

    # Step 1b: Apply difficulty-based tier floor (2026-04-26).
    # When ``force_tier`` isn't passed, but the active task is
    # high-difficulty research (or another role with a floor in the
    # _ROLE_DIFFICULTY_TIER_FLOOR table), promote ``force_tier`` to
    # the minimum required tier so sub-agents inherit the parent's
    # quality bar even though CrewAI doesn't propagate force_tier
    # through delegate_work_to_coworker.
    if not force_tier:
        active_diff = get_active_difficulty()
        floor_tier = _resolve_difficulty_tier_floor(role, active_diff)
        if floor_tier:
            logger.info(
                f"llm_selector: difficulty floor — role={role} d={active_diff} "
                f"→ force_tier={floor_tier} (sub-agent inherited from active context)"
            )
            force_tier = floor_tier

    # Step 2: Default from catalog (consults role_assignments overlay)
    # Reads the live runtime mode so dashboard/Signal switches take effect
    # immediately, without waiting for a config reload.
    from app.llm_mode import get_mode
    mode = get_mode()
    default_model = get_default_for_role(role, mode)

    # Step 2b — Output-ceiling check (2026-05-03 audit cleanup, item 5).
    # When the caller passes expected_output_tokens, swap out a default
    # whose model can't deliver that many tokens.  Pre-fix the H2 clamp
    # (PR #30) caught the ceiling at the LLM constructor and silently
    # truncated; here we catch it at the SELECTOR so a high-output task
    # doesn't get routed to a low-ceiling model in the first place.
    # Cheaper than retry-after-truncation.
    if expected_output_tokens > 4096:
        try:
            from app.llm_factory import model_max_output_tokens
            default_entry_for_cap = get_model(default_model)
            default_mid = (default_entry_for_cap or {}).get("model_id", default_model)
            if model_max_output_tokens(default_mid) < expected_output_tokens:
                # Walk the catalog for a candidate with sufficient ceiling
                # whose tier the current mode allows.  First match wins;
                # caller's tier preferences (force_tier etc.) get applied
                # downstream so this is just a "ceiling sufficiency" filter.
                for cand_name, cand_entry in CATALOG.items():
                    if cand_name == default_model:
                        continue
                    cand_mid = cand_entry.get("model_id", cand_name)
                    if model_max_output_tokens(cand_mid) >= expected_output_tokens \
                            and _tier_allowed(cand_entry.get("tier", ""), settings):
                        logger.info(
                            "llm_selector: output-ceiling swap — role=%s "
                            "expected_output=%d, default %s ceiling=%d, "
                            "swapping to %s ceiling=%d",
                            role, expected_output_tokens, default_model,
                            model_max_output_tokens(default_mid),
                            cand_name, model_max_output_tokens(cand_mid),
                        )
                        default_model = cand_name
                        break
        except Exception:
            # Selector is best-effort; never block the call
            pass

    # Step 3: Task-specific overrides
    task_type = detect_task_type(role, task_hint)

    # Cache model lookups within this call to avoid redundant dict scans
    _model_cache: dict[str, dict | None] = {}

    def _cached_get_model(name: str) -> dict | None:
        if name not in _model_cache:
            _model_cache[name] = get_model(name)
        return _model_cache[name]

    # Multimodal tasks need a multimodal model
    if task_type == "multimodal":
        default_entry = _cached_get_model(default_model)
        if default_entry and not default_entry.get("multimodal"):
            for mm_model in _MULTIMODAL_MODELS:
                mm_entry = _cached_get_model(mm_model)
                if mm_entry and _tier_allowed(mm_entry["tier"], settings):
                    logger.info(f"llm_selector: multimodal → {default_model} → {mm_model}")
                    default_model = mm_model
                    break

    # Force tier if specified
    if force_tier:
        tier_candidates = get_candidates_by_tier(task_type, [force_tier])
        if tier_candidates:
            # Resource-aware selection for local tier: pick the best model
            # that fits in available memory with healthy headroom
            if force_tier == "local":
                selected = _select_local_resource_aware(
                    tier_candidates, role, task_type, _cached_get_model, settings
                )
                if selected:
                    return selected
                # If nothing fits, fall through to API tier
                logger.info(f"llm_selector: no local model fits in memory, falling through")
            else:
                forced = tier_candidates[0][0]
                # Tool-use compatibility check
                _ROLES_NEEDING_TOOLS_EARLY = {"coding", "research", "writing", "media", "self_improve", "critic"}
                if role in _ROLES_NEEDING_TOOLS_EARLY:
                    forced_entry = _cached_get_model(forced)
                    if forced_entry and forced_entry.get("supports_tools") is False:
                        for name, _score in tier_candidates[1:]:
                            alt_entry = _cached_get_model(name)
                            if alt_entry and alt_entry.get("supports_tools") is not False:
                                if _model_available(name, settings, max_ram_gb):
                                    logger.info(f"llm_selector: force_tier={force_tier}, "
                                                f"{forced} no tools → {name}")
                                    return name
                        logger.info(f"llm_selector: force_tier={force_tier} has no tool-capable model")
                    else:
                        if _model_available(forced, settings, max_ram_gb):
                            logger.info(f"llm_selector: force_tier={force_tier} → {forced}")
                            return forced
                else:
                    if _model_available(forced, settings, max_ram_gb):
                        logger.info(f"llm_selector: force_tier={force_tier} → {forced}")
                        return forced

    # Step 4: Benchmark adjustment (blended internal + external — Phase 3)
    bench_scores = get_scores(task_type)
    if bench_scores:
        default_entry = _cached_get_model(default_model)
        default_bench = bench_scores.get(default_model)
        if default_bench is not None and default_entry:
            for name, bench_score in sorted(bench_scores.items(), key=lambda x: -x[1]):
                if name == default_model:
                    break
                candidate = _cached_get_model(name)
                if not candidate:
                    continue
                if candidate["tier"] == default_entry["tier"]:
                    if bench_score > (default_bench + 0.1):
                        logger.info(f"llm_selector: benchmark override {default_model} → {name}")
                        default_model = name
                        break

    # Step 4b: Pareto demotion — when a cheaper model scores close to
    # the default, prefer it unless the caller explicitly asked for a
    # tier via force_tier. Respects the outer envelope; never crosses
    # into local from an API default (local bypasses the cost axis).
    if bench_scores and not force_tier:
        default_entry = _cached_get_model(default_model)
        default_bench = bench_scores.get(default_model, 0.0)
        if default_entry and default_entry.get("tier") in ("budget", "mid", "premium"):
            alt = _pareto_cheaper_alternative(
                default_model, default_entry, default_bench,
                bench_scores, _cached_get_model,
                task_type=task_type,
            )
            if alt:
                alt_entry = _cached_get_model(alt)
                # Only cross-tier when the alt is cheaper AND API-tier.
                if alt_entry and alt_entry.get("tier") in ("budget", "mid", "premium"):
                    logger.info(
                        "llm_selector: pareto demotion %s → %s "
                        "(score %.2f→%.2f, cost %.2f→%.2f)",
                        default_model, alt,
                        default_bench, bench_scores.get(alt, 0.0),
                        float(default_entry.get("cost_output_per_m", 0)),
                        float(alt_entry.get("cost_output_per_m", 0)),
                    )
                    default_model = alt

    # Step 4c: Budget enforcement. If the caller specified a hard USD
    # ceiling for this call and the current default would blow it,
    # demote to the cheapest bench-eligible candidate within budget
    # whose score stays within 0.10 of the default's.
    if budget_usd is not None and bench_scores:
        from app.llm_catalog import estimate_task_cost
        def _fits(name: str) -> bool:
            return estimate_task_cost(
                name, expected_input_tokens, expected_output_tokens,
            ) <= budget_usd

        default_cost = estimate_task_cost(
            default_model, expected_input_tokens, expected_output_tokens,
        )
        if default_cost > budget_usd:
            default_bench = bench_scores.get(default_model, 0.0)
            best: tuple[str, float] | None = None  # (name, score)
            for name, score in bench_scores.items():
                if score < default_bench - 0.10:
                    continue
                if not _fits(name):
                    continue
                if best is None or score > best[1]:
                    best = (name, score)
            if best:
                logger.info(
                    "llm_selector: budget demotion %s → %s (budget=$%.4f, "
                    "default_cost=$%.4f)",
                    default_model, best[0], budget_usd, default_cost,
                )
                default_model = best[0]

    # Step 5: Tool-use compatibility check
    # CrewAI agents need tool calling for most roles (research, coding, writing, media).
    # Models that don't support tools (e.g. codestral via Ollama) will get a 400 error.
    _ROLES_NEEDING_TOOLS = {"coding", "research", "writing", "media", "self_improve", "critic"}
    if role in _ROLES_NEEDING_TOOLS:
        default_entry = _cached_get_model(default_model)
        if default_entry and default_entry.get("supports_tools") is False:
            logger.info(f"llm_selector: {default_model} doesn't support tools, skipping for role={role}")
            default_model = _find_fallback(role, task_type, settings, max_ram_gb)

    # Step 6: Availability check
    if _model_available(default_model, settings, max_ram_gb):
        logger.info(f"llm_selector: role={role} task={task_type} mode={mode} → {default_model}")
        return default_model

    return _find_fallback(role, task_type, settings, max_ram_gb)


def _tier_allowed(tier: str, settings) -> bool:
    if tier == "local":
        return settings.local_llm_enabled
    if tier in ("free", "budget", "mid"):
        return settings.api_tier_enabled
    return True

def _model_available(model_name: str, settings, max_ram_gb: float) -> bool:
    entry = get_model(model_name)
    if not entry:
        return False
    tier = entry["tier"]
    if tier == "local":
        if not settings.local_llm_enabled:
            return False
        return entry.get("ram_gb", 20) <= max_ram_gb
    if tier in ("free", "budget", "mid"):
        return settings.api_tier_enabled and bool(settings.openrouter_api_key.get_secret_value())
    if entry["provider"] == "anthropic":
        return True
    if entry["provider"] == "openrouter":
        return bool(settings.openrouter_api_key.get_secret_value())
    return False

def _find_fallback(role: str, task_type: str, settings, max_ram_gb: float) -> str:
    _needs_tools = role in {"coding", "research", "writing", "media", "self_improve", "critic"}
    for tier in ("free", "budget", "mid", "premium"):
        candidates = get_candidates_by_tier(task_type, [tier])
        for name, _score in candidates:
            if _model_available(name, settings, max_ram_gb):
                # Skip models that don't support tools when role needs them
                if _needs_tools:
                    entry = get_model(name)
                    if entry and entry.get("supports_tools") is False:
                        continue
                logger.info(f"llm_selector: fallback role={role} → {name} (tier={tier})")
                return name
    logger.warning("llm_selector: all tiers failed, using Claude Sonnet 4.6")
    return "claude-sonnet-4.6"
