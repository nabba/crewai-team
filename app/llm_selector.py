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
    """
    if difficulty <= 3:
        return "local" if mode != "cloud" else "budget"
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
        "critic": "reasoning", "introspector": "reasoning",
        "self_improve": "research", "vetting": "vetting",
        "synthesis": "writing", "planner": "research", "default": "general",
    }
    return role_map.get(role, "general")


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

    # Multimodal tasks need a multimodal model
    if task_type == "multimodal":
        default_entry = get_model(default_model)
        if default_entry and not default_entry.get("multimodal"):
            for mm_model in _MULTIMODAL_MODELS:
                mm_entry = get_model(mm_model)
                if mm_entry and _tier_allowed(mm_entry["tier"], settings):
                    logger.info(f"llm_selector: multimodal → {default_model} → {mm_model}")
                    default_model = mm_model
                    break

    # Force tier if specified
    if force_tier:
        tier_candidates = get_candidates_by_tier(task_type, [force_tier])
        if tier_candidates:
            forced = tier_candidates[0][0]
            if _model_available(forced, settings, max_ram_gb):
                logger.info(f"llm_selector: force_tier={force_tier} → {forced}")
                return forced

    # Step 4: Benchmark adjustment
    bench_scores = get_scores(task_type)
    if bench_scores:
        default_entry = get_model(default_model)
        default_bench = bench_scores.get(default_model)
        if default_bench is not None and default_entry:
            for name, bench_score in sorted(bench_scores.items(), key=lambda x: -x[1]):
                if name == default_model:
                    break
                candidate = get_model(name)
                if not candidate:
                    continue
                if candidate["tier"] == default_entry["tier"]:
                    if bench_score > (default_bench + 0.1):
                        logger.info(f"llm_selector: benchmark override {default_model} → {name}")
                        default_model = name
                        break

    # Step 5: Availability check
    if _model_available(default_model, settings, max_ram_gb):
        logger.info(f"llm_selector: role={role} task={task_type} mode={cost_mode} → {default_model}")
        return default_model

    return _find_fallback(role, task_type, settings, max_ram_gb)


def _tier_allowed(tier: str, settings) -> bool:
    if tier == "local":
        return settings.local_llm_enabled
    if tier in ("budget", "mid"):
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
    if tier in ("budget", "mid"):
        return settings.api_tier_enabled and bool(settings.openrouter_api_key.get_secret_value())
    if entry["provider"] == "anthropic":
        return True
    if entry["provider"] == "openrouter":
        return bool(settings.openrouter_api_key.get_secret_value())
    return False

def _find_fallback(role: str, task_type: str, settings, max_ram_gb: float) -> str:
    for tier in ("budget", "mid", "premium"):
        candidates = get_candidates_by_tier(task_type, [tier])
        for name, _score in candidates:
            if _model_available(name, settings, max_ram_gb):
                logger.info(f"llm_selector: fallback role={role} → {name} (tier={tier})")
                return name
    logger.warning("llm_selector: all tiers failed, using Claude Sonnet 4.6")
    return "claude-sonnet-4.6"
