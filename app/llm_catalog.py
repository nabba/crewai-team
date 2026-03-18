"""
llm_catalog.py — Curated registry of LLM models with metadata.

Each model entry has:
  - size_gb: approximate disk/download size
  - ram_gb: approximate RAM needed when loaded
  - speed: qualitative rating (very_fast, fast, medium, slow)
  - context: max context window in tokens
  - strengths: dict of task_type → score (0.0-1.0)

Optimized for Apple M4 Max (48GB unified memory, Metal GPU).
qwen3:30b-a3b is the primary model for all roles — MoE architecture
gives excellent quality with very fast inference on Metal GPU.
"""

CATALOG: dict[str, dict] = {
    # ── Coding specialists ────────────────────────────────────────────────
    "qwen3:30b-a3b": {
        "size_gb": 18, "ram_gb": 20, "speed": "very_fast", "context": 32768,
        "description": "MoE model — activates ~3B params/token, excellent all-rounder on Metal GPU",
        "strengths": {
            "coding": 0.92, "architecture": 0.85, "research": 0.85,
            "writing": 0.85, "general": 0.88, "debugging": 0.85,
            "reasoning": 0.85,
        },
    },
    "codestral:22b": {
        "size_gb": 13, "ram_gb": 15, "speed": "fast", "context": 32768,
        "description": "Mistral's code-specialized model",
        "strengths": {
            "coding": 0.95, "debugging": 0.85, "general": 0.50,
        },
    },

    # ── Reasoning / architecture ──────────────────────────────────────────
    "deepseek-r1:32b": {
        "size_gb": 19, "ram_gb": 22, "speed": "medium", "context": 32768,
        "description": "Strong reasoning, architecture, debugging",
        "strengths": {
            "architecture": 0.95, "debugging": 0.90, "coding": 0.80,
            "reasoning": 0.95, "research": 0.75,
        },
    },
    "gemma3:27b": {
        "size_gb": 17, "ram_gb": 19, "speed": "medium", "context": 128000,
        "description": "Google's strong reasoning model, huge context",
        "strengths": {
            "reasoning": 0.85, "writing": 0.80, "research": 0.80,
            "coding": 0.70, "general": 0.80,
        },
    },

    # ── General / writing ─────────────────────────────────────────────────
    "mistral-small:24b": {
        "size_gb": 14, "ram_gb": 16, "speed": "fast", "context": 32768,
        "description": "Mistral's balanced model, good at everything",
        "strengths": {
            "coding": 0.80, "writing": 0.85, "general": 0.80,
            "research": 0.75, "reasoning": 0.75,
        },
    },

    # ── Small / fast fallback ─────────────────────────────────────────────
    "llama3.1:8b": {
        "size_gb": 5, "ram_gb": 6, "speed": "very_fast", "context": 131072,
        "description": "Meta's small fast model, huge context, good fallback",
        "strengths": {
            "general": 0.60, "writing": 0.60, "coding": 0.50,
            "research": 0.55, "reasoning": 0.50,
        },
    },
}

# Task type aliases — maps common keywords to canonical task types
_TASK_ALIASES: dict[str, str] = {
    "code": "coding", "implement": "coding", "program": "coding", "fix": "debugging",
    "debug": "debugging", "review": "architecture", "architect": "architecture",
    "design": "architecture", "plan": "architecture",
    "write": "writing", "summarize": "writing", "document": "writing", "report": "writing",
    "research": "research", "search": "research", "find": "research", "learn": "research",
    "reason": "reasoning", "analyze": "reasoning", "think": "reasoning",
}


def get_candidates(task_type: str) -> list[tuple[str, float]]:
    """
    Return models ranked by strength for a task type.
    Returns list of (model_name, score) sorted descending.
    """
    task_type = _TASK_ALIASES.get(task_type, task_type)
    scored = []
    for name, info in CATALOG.items():
        score = info["strengths"].get(task_type, info["strengths"].get("general", 0.5))
        scored.append((name, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def get_smallest_model() -> str:
    """Return the smallest model in the catalog (fast fallback)."""
    return min(CATALOG, key=lambda m: CATALOG[m]["ram_gb"])


def get_model_info(model: str) -> dict | None:
    """Return catalog entry for a model, or None if not in catalog."""
    return CATALOG.get(model)


def get_ram_requirement(model: str) -> float:
    """Return estimated RAM in GB needed to run a model."""
    info = CATALOG.get(model)
    return info["ram_gb"] if info else 20.0  # assume 20GB for unknown models


def format_catalog() -> str:
    """Format the catalog for display in Signal."""
    lines = ["Available LLM Models:\n"]
    for name, info in sorted(CATALOG.items(), key=lambda x: -x[1]["ram_gb"]):
        top_strength = max(info["strengths"], key=info["strengths"].get)
        lines.append(
            f"  {name} ({info['size_gb']}GB, {info['speed']}) "
            f"— best at: {top_strength} ({info['strengths'][top_strength]:.0%})"
        )
    return "\n".join(lines)
