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

import logging
import os
import re

from app.config import get_settings
from app.llm_catalog import (
    CATALOG, TASK_ALIASES, ROLE_DEFAULTS,
    get_model, get_default_for_role, get_candidates_by_tier,
)
from app.llm_benchmarks import get_scores

logger = logging.getLogger(__name__)


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

_KEYWORD_PATTERNS: list[tuple[str, str]] = [
    (r"\b(debug|traceback|error|fix\s+bug|stacktrace)\b", "debugging"),
    (r"\b(architect|design|plan|system\s+design|review)\b", "architecture"),
    (r"\b(code|implement|function|class|module|script|program)\b", "coding"),
    (r"\b(research|search|find|learn|investigate|analyze)\b", "research"),
    (r"\b(write|summarize|document|report|explain|describe)\b", "writing"),
    (r"\b(reason|think|logic|proof|math)\b", "reasoning"),
    (r"\b(image|photo|screenshot|picture|visual|pdf|scan)\b", "multimodal"),
    (r"\b(video|audio|podcast|youtube|camera|media|voice|music|mp[34])\b", "multimodal"),
]

_MULTIMODAL_MODELS = [name for name, info in CATALOG.items() if info.get("multimodal")]


def detect_task_type(role: str, task_hint: str = "") -> str:
    if task_hint:
        hint_lower = task_hint.lower()
        for pattern, task_type in _KEYWORD_PATTERNS:
            if re.search(pattern, hint_lower):
                return task_type
    role_map = {
        "coding": "coding", "architecture": "architecture",
        "research": "research", "writing": "writing",
        "media": "multimodal", "critic": "reasoning", "introspector": "reasoning",
        "self_improve": "research", "vetting": "vetting",
        "synthesis": "writing", "planner": "research", "default": "general",
    }
    return role_map.get(role, "general")


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


def select_model(
    role: str, task_hint: str = "", max_ram_gb: float = 48.0, force_tier: str | None = None,
) -> str:
    settings = get_settings()

    # Step 1: Environment override
    env_key = f"ROLE_MODEL_{role.upper()}"
    env_override = os.environ.get(env_key)
    if env_override and env_override in CATALOG:
        logger.info(f"llm_selector: {env_key}={env_override} (env override)")
        return env_override

    # Step 2: Default from catalog
    cost_mode = settings.cost_mode
    default_model = get_default_for_role(role, cost_mode)

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

    # Step 4: Benchmark adjustment
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
        logger.info(f"llm_selector: role={role} task={task_type} mode={cost_mode} → {default_model}")
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
