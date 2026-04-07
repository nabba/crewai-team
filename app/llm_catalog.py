"""
llm_catalog.py — Multi-tier LLM registry with cost, quality, and capability metadata.

Architecture (three tiers + local):
  LOCAL:    Ollama on Metal GPU — free, fast, moderate quality
  BUDGET:   Frontier API models via OpenRouter — cheap, high quality
  MID:      Strong API models via OpenRouter — moderate cost, strong quality
  PREMIUM:  Anthropic Claude + Gemini — expensive, highest reliability

Each model entry includes:
  - tier: local | budget | mid | premium
  - provider: ollama | openrouter | anthropic
  - model_id: string used to create crewai.LLM (provider-prefixed)
  - cost_input_per_m / cost_output_per_m: USD per 1M tokens (0 for local)
  - context: max context window in tokens
  - multimodal: supports image/video input
  - strengths: task_type → score (0.0-1.0)
  - tool_use_reliability: 0.0-1.0 (critical for CrewAI tool-calling loops)

Pricing as of March 2026. Verify at https://openrouter.ai/models before deploying.

Optimized for Apple M4 Max (48GB unified memory) for local tier.
"""

from __future__ import annotations

# ── Model catalog ──────────────────────────────────────────────────────────

CATALOG: dict[str, dict] = {

    # ═══════════════════════════════════════════════════════════════════════
    # LOCAL — Ollama on Metal GPU (free, ~15-25 tok/s on M4 Max)
    # ═══════════════════════════════════════════════════════════════════════

    "qwen3:30b-a3b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/qwen3:30b-a3b",
        "size_gb": 18, "ram_gb": 20, "speed": "very_fast",
        "context": 32_768, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.70,
        "description": "MoE ~3B active — best local all-rounder on Metal GPU",
        "strengths": {
            "coding": 0.82, "architecture": 0.75, "research": 0.75,
            "writing": 0.75, "general": 0.78, "debugging": 0.75,
            "reasoning": 0.75, "routing": 0.55, "vetting": 0.50,
        },
    },
    "deepseek-r1:32b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/deepseek-r1:32b",
        "size_gb": 19, "ram_gb": 22, "speed": "medium",
        "context": 32_768, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.60,
        "description": "Strong local reasoning — architecture, debugging, proofs",
        "strengths": {
            "architecture": 0.85, "debugging": 0.80, "coding": 0.75,
            "reasoning": 0.88, "research": 0.70,
        },
    },
    "codestral:22b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/codestral:22b",
        "size_gb": 13, "ram_gb": 15, "speed": "fast",
        "context": 32_768, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.0,
        "supports_tools": False,  # Ollama codestral does NOT support tool calling
        "description": "Mistral code specialist — code completion only, no tool use",
        "strengths": {"coding": 0.88, "debugging": 0.80, "general": 0.50},
    },
    "gemma4:26b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/gemma4:26b",
        "size_gb": 18, "ram_gb": 20, "speed": "very_fast",
        "context": 256_000, "multimodal": True,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.75,
        "description": "Gemma 4 MoE — 3.8B active params, vision+text, 256K context, function calling",
        "strengths": {
            "coding": 0.82, "architecture": 0.78, "research": 0.80,
            "writing": 0.80, "general": 0.82, "debugging": 0.78,
            "reasoning": 0.80, "routing": 0.60, "vetting": 0.55,
            "multimodal": 0.75,
        },
    },
    "gemma4:31b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/gemma4:31b",
        "size_gb": 20, "ram_gb": 23, "speed": "medium",
        "context": 256_000, "multimodal": True,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.78,
        "description": "Gemma 4 dense 31B — strongest local Google model, vision, 256K context",
        "strengths": {
            "coding": 0.85, "architecture": 0.82, "research": 0.85,
            "writing": 0.85, "general": 0.85, "debugging": 0.82,
            "reasoning": 0.85, "routing": 0.65, "vetting": 0.60,
            "multimodal": 0.80,
        },
    },
    "gemma3:27b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/gemma3:27b",
        "size_gb": 17, "ram_gb": 19, "speed": "medium",
        "context": 128_000, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.62,
        "description": "Google reasoning model — huge local context window (superseded by gemma4)",
        "strengths": {
            "reasoning": 0.80, "writing": 0.75, "research": 0.75,
            "coding": 0.65, "general": 0.75,
        },
    },
    "llama3.1:8b": {
        "tier": "local",
        "provider": "ollama",
        "model_id": "ollama_chat/llama3.1:8b",
        "size_gb": 5, "ram_gb": 6, "speed": "very_fast",
        "context": 131_072, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.45,
        "description": "Fast small fallback — keeps things running when RAM is tight",
        "strengths": {
            "general": 0.55, "writing": 0.55, "coding": 0.45,
            "research": 0.50, "reasoning": 0.45,
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # FREE — $0 API models via OpenRouter (free tier, rate-limited)
    # Best for: bulk parallel work, media analysis, low-priority background
    # ═══════════════════════════════════════════════════════════════════════

    "nemotron-nano-2-vl": {
        "tier": "free",
        "provider": "openrouter",
        "model_id": "openrouter/nvidia/nemotron-nano-12b-v2-vl:free",
        "context": 32_768, "multimodal": True,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.55,
        "description": (
            "NVIDIA Nemotron Nano 2 VL — 12B hybrid Transformer-Mamba. "
            "Multimodal: text + multi-image. Leading OCRBench v2. "
            "Free tier. Video understanding + document intelligence."
        ),
        "strengths": {
            "multimodal": 0.88, "research": 0.65, "writing": 0.60,
            "general": 0.62, "coding": 0.55, "reasoning": 0.60,
        },
    },
    "nemotron-3-super": {
        "tier": "free",
        "provider": "openrouter",
        "model_id": "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "context": 1_000_000, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.65,
        "description": (
            "NVIDIA Nemotron 3 Super — 120B MoE (12B active). "
            "1M context, cross-document reasoning, multi-step planning. "
            "Free tier. Strong for synthesis and long-context tasks."
        ),
        "strengths": {
            "research": 0.78, "reasoning": 0.80, "architecture": 0.75,
            "writing": 0.75, "general": 0.76, "coding": 0.72,
            "synthesis": 0.80,
        },
    },
    "trinity-large": {
        "tier": "free",
        "provider": "openrouter",
        "model_id": "openrouter/arcee-ai/trinity-large-preview:free",
        "context": 128_000, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.60,
        "description": (
            "Arcee Trinity Large — 400B sparse MoE (13B active). "
            "Strong creative writing, agentic tool use. "
            "Free tier. Natively supports 512K (128K in preview)."
        ),
        "strengths": {
            "writing": 0.82, "general": 0.75, "research": 0.72,
            "coding": 0.70, "reasoning": 0.72, "architecture": 0.68,
        },
    },
    "step-3.5-flash": {
        "tier": "free",
        "provider": "openrouter",
        "model_id": "openrouter/stepfun/step-3.5-flash:free",
        "context": 256_000, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.60,
        "description": (
            "StepFun Step 3.5 Flash — 196B sparse MoE (11B active). "
            "Speed-optimized reasoning. 256K context. Free tier."
        ),
        "strengths": {
            "reasoning": 0.78, "research": 0.72, "coding": 0.70,
            "writing": 0.70, "general": 0.72, "architecture": 0.68,
        },
    },
    "minimax-m2.5-free": {
        "tier": "free",
        "provider": "openrouter",
        "model_id": "openrouter/minimax/minimax-m2.5:free",
        "context": 196_608, "multimodal": False,
        "cost_input_per_m": 0.0, "cost_output_per_m": 0.0,
        "tool_use_reliability": 0.75,
        "description": (
            "MiniMax M2.5 free tier — 80.2% SWE-bench, token-efficient. "
            "Strong agentic coding + office document fluency."
        ),
        "strengths": {
            "coding": 0.88, "debugging": 0.85, "reasoning": 0.82,
            "writing": 0.78, "general": 0.80, "research": 0.78,
            "architecture": 0.80,
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # BUDGET — Frontier API at <$1.50/M output (via OpenRouter)
    # Best for: parallel sub-agents, background jobs, high-volume work
    # ═══════════════════════════════════════════════════════════════════════

    "deepseek-v3.2": {
        "tier": "budget",
        "provider": "openrouter",
        "model_id": "openrouter/deepseek/deepseek-chat",
        "context": 128_000, "multimodal": False,
        "cost_input_per_m": 0.28, "cost_output_per_m": 0.42,
        "tool_use_reliability": 0.82,
        "description": (
            "DeepSeek V3.2 — cheapest frontier model. Built for agentic tool use "
            "with Sparse Attention + RL-trained on 1800+ environments. "
            "SWE-bench 73.1%, AIME 91.7%. Text-only."
        ),
        "strengths": {
            "coding": 0.87, "reasoning": 0.90, "research": 0.85,
            "writing": 0.82, "debugging": 0.87, "architecture": 0.85,
            "general": 0.85, "routing": 0.70, "vetting": 0.78,
        },
    },
    "minimax-m2.5": {
        "tier": "budget",
        "provider": "openrouter",
        "model_id": "openrouter/minimax/minimax-m2.5",
        "context": 128_000, "multimodal": False,
        "cost_input_per_m": 0.30, "cost_output_per_m": 1.20,
        "tool_use_reliability": 0.80,
        "description": (
            "MiniMax M2.5 — 80.2% SWE-bench Verified at budget pricing. "
            "Open-weight community favorite. Interleaved thinking."
        ),
        "strengths": {
            "coding": 0.90, "debugging": 0.88, "reasoning": 0.85,
            "writing": 0.80, "general": 0.83, "research": 0.80,
            "architecture": 0.82,
        },
    },

    "gemma-4-26b": {
        "tier": "budget",
        "provider": "openrouter",
        "model_id": "openrouter/google/gemma-4-26b-a4b-it",
        "context": 256_000, "multimodal": True,
        "cost_input_per_m": 0.13, "cost_output_per_m": 0.40,
        "tool_use_reliability": 0.78,
        "description": "Gemma 4 26B MoE via OpenRouter — 3.8B active, vision+text, function calling, 256K context",
        "strengths": {
            "coding": 0.82, "architecture": 0.78, "research": 0.80,
            "writing": 0.80, "general": 0.82, "debugging": 0.78,
            "reasoning": 0.80, "multimodal": 0.75,
        },
    },
    "gemma-4-31b": {
        "tier": "budget",
        "provider": "openrouter",
        "model_id": "openrouter/google/gemma-4-31b-it",
        "context": 256_000, "multimodal": True,
        "cost_input_per_m": 0.14, "cost_output_per_m": 0.40,
        "tool_use_reliability": 0.80,
        "description": "Gemma 4 31B dense via OpenRouter — strongest open Google model, vision, 256K, function calling",
        "strengths": {
            "coding": 0.85, "architecture": 0.82, "research": 0.85,
            "writing": 0.85, "general": 0.85, "debugging": 0.82,
            "reasoning": 0.85, "multimodal": 0.80,
        },
    },

    "mimo-v2-omni": {
        "tier": "budget",
        "provider": "openrouter",
        "model_id": "openrouter/xiaomi/mimo-v2-omni",
        "context": 256_000, "multimodal": True,
        "cost_input_per_m": 0.40, "cost_output_per_m": 2.00,
        "tool_use_reliability": 0.80,
        "description": (
            "Xiaomi MiMo-V2-Omni — frontier omni-modal model. Natively processes "
            "image, video, and audio inputs. Visual grounding, multi-step planning, "
            "tool use, code execution. 256K context."
        ),
        "strengths": {
            "multimodal": 0.92, "research": 0.82, "coding": 0.80,
            "reasoning": 0.82, "writing": 0.78, "general": 0.82,
            "architecture": 0.78, "debugging": 0.78,
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # MID — Strong API at $1-4/M output (via OpenRouter)
    # Best for: quality research, multimodal, agentic coding
    # ═══════════════════════════════════════════════════════════════════════

    "mimo-v2-pro": {
        "tier": "mid",
        "provider": "openrouter",
        "model_id": "openrouter/xiaomi/mimo-v2-pro",
        "context": 1_000_000, "multimodal": False,
        "cost_input_per_m": 1.00, "cost_output_per_m": 3.00,
        "tool_use_reliability": 0.90,
        "description": (
            "Xiaomi MiMo-V2-Pro (ex-Hunter Alpha) — 1T+ params, 1M context. "
            "Top-tier agentic model. #1 on ClawBench, approaches Opus 4.6 on "
            "PinchBench. Designed as agent brain for complex orchestration."
        ),
        "strengths": {
            "coding": 0.92, "reasoning": 0.93, "architecture": 0.92,
            "research": 0.90, "writing": 0.85, "debugging": 0.90,
            "general": 0.90, "routing": 0.85, "vetting": 0.85,
        },
    },
    "kimi-k2.5": {
        "tier": "mid",
        "provider": "openrouter",
        "model_id": "openrouter/moonshotai/kimi-k2.5",
        "context": 262_000, "multimodal": True,
        "cost_input_per_m": 0.60, "cost_output_per_m": 3.00,
        "tool_use_reliability": 0.85,
        "description": (
            "Kimi K2.5 — 1T MoE (32B active). Natively multimodal (MoonViT). "
            "256K context. Agent Swarm capable. AIME 96.1%, GPQA 87.6%. "
            "Best open-weight all-rounder."
        ),
        "strengths": {
            "research": 0.92, "reasoning": 0.90, "coding": 0.88,
            "writing": 0.85, "architecture": 0.88, "debugging": 0.85,
            "general": 0.88, "multimodal": 0.92,
        },
    },
    "glm-5": {
        "tier": "mid",
        "provider": "openrouter",
        "model_id": "openrouter/zhipu/glm-5",
        "context": 200_000, "multimodal": False,
        "cost_input_per_m": 1.00, "cost_output_per_m": 3.20,
        "tool_use_reliability": 0.83,
        "description": (
            "GLM-5 — 744B MoE (40B active). #1 open-weight on Artificial Analysis "
            "and LMArena. 98% frontend build success, 200K context. "
            "Strong agentic engineering."
        ),
        "strengths": {
            "coding": 0.90, "architecture": 0.88, "reasoning": 0.88,
            "writing": 0.83, "research": 0.85, "debugging": 0.87,
            "general": 0.87,
        },
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PREMIUM — Highest quality + tool-use reliability
    # Best for: commander routing, user-facing output, vetting, critic
    # ═══════════════════════════════════════════════════════════════════════

    "gemini-3.1-pro": {
        "tier": "premium",
        "provider": "openrouter",
        "model_id": "openrouter/google/gemini-3.1-pro-preview",
        "context": 1_000_000, "multimodal": True,
        "cost_input_per_m": 2.00, "cost_output_per_m": 12.00,
        "tool_use_reliability": 0.90,
        "description": (
            "Gemini 3.1 Pro — #1 on 13/16 benchmarks. ARC-AGI-2 77.1%, "
            "GPQA 94.3%. 1M context. Same price as predecessor."
        ),
        "strengths": {
            "reasoning": 0.95, "research": 0.92, "coding": 0.90,
            "writing": 0.88, "architecture": 0.90, "debugging": 0.88,
            "general": 0.92, "multimodal": 0.90, "vetting": 0.88,
        },
    },
    "claude-sonnet-4.6": {
        "tier": "premium",
        "provider": "anthropic",
        "model_id": "anthropic/claude-sonnet-4-6",
        "context": 1_000_000, "multimodal": True,
        "cost_input_per_m": 1.00, "cost_output_per_m": 5.00,
        "tool_use_reliability": 0.95,
        "description": (
            "Claude Sonnet 4.6 — #1 GDPval-AA human preference (1633 Elo). "
            "Near-Opus quality at 1/5 the cost. SWE-bench 79.6%. "
            "Default on Claude.ai + GitHub Copilot agent."
        ),
        "strengths": {
            "writing": 0.93, "coding": 0.91, "research": 0.90,
            "reasoning": 0.90, "architecture": 0.88, "debugging": 0.88,
            "general": 0.92, "routing": 0.90, "vetting": 0.92,
        },
    },
    "claude-opus-4.6": {
        "tier": "premium",
        "provider": "anthropic",
        "model_id": "anthropic/claude-opus-4-6",
        "context": 1_000_000, "multimodal": True,
        "cost_input_per_m": 5.00, "cost_output_per_m": 25.00,
        "tool_use_reliability": 0.98,
        "description": (
            "Claude Opus 4.6 — highest tool-use reliability (0.98). "
            "SWE-bench 80.8%. ARC-AGI-2 68.8%. GDPval-AA 1606 Elo. "
            "Reserve for routing and critical-path vetting."
        ),
        "strengths": {
            "routing": 0.98, "vetting": 0.95, "coding": 0.92,
            "writing": 0.92, "research": 0.90, "reasoning": 0.92,
            "architecture": 0.92, "debugging": 0.90, "general": 0.93,
        },
    },
}


# ── Default role → model assignments ───────────────────────────────────────

ROLE_DEFAULTS: dict[str, dict[str, str]] = {
    "budget": {
        "commander":    "claude-sonnet-4.6",
        "research":     "deepseek-v3.2",
        "coding":       "minimax-m2.5",
        "writing":      "deepseek-v3.2",
        "media":        "mimo-v2-omni",
        "critic":       "deepseek-v3.2",
        "introspector": "deepseek-v3.2",
        "self_improve":  "deepseek-v3.2",
        "vetting":      "deepseek-v3.2",
        "synthesis":    "deepseek-v3.2",
        "planner":      "deepseek-v3.2",
        "evo_critic":   "deepseek-v3.2",
        "default":      "deepseek-v3.2",
    },
    "balanced": {
        "commander":    "claude-opus-4.6",
        "research":     "deepseek-v3.2",
        "coding":       "minimax-m2.5",
        "writing":      "claude-sonnet-4.6",
        "media":        "gemma4:26b",         # Vision-capable local model
        "critic":       "gemini-3.1-pro",
        "introspector": "gemma4:26b",         # Local — saves API cost for self-reflection
        "self_improve":  "gemma4:26b",        # Local — background task, no API spend
        "vetting":      "claude-sonnet-4.6",
        "synthesis":    "claude-sonnet-4.6",
        "planner":      "gemma4:26b",         # Local — background task
        "evo_critic":   "gemma4:26b",         # Local — evolution judging
        "default":      "deepseek-v3.2",
    },
    "quality": {
        "commander":    "claude-opus-4.6",
        "research":     "mimo-v2-pro",
        "coding":       "mimo-v2-pro",
        "writing":      "claude-sonnet-4.6",
        "media":        "mimo-v2-omni",
        "critic":       "gemini-3.1-pro",
        "introspector": "mimo-v2-pro",
        "self_improve":  "deepseek-v3.2",
        "vetting":      "claude-opus-4.6",
        "synthesis":    "claude-sonnet-4.6",
        "planner":      "mimo-v2-pro",
        "evo_critic":   "claude-sonnet-4.6",
        "default":      "mimo-v2-pro",
    },
}

# Task type aliases
TASK_ALIASES: dict[str, str] = {
    "code": "coding", "implement": "coding", "program": "coding",
    "fix": "debugging", "debug": "debugging",
    "review": "architecture", "architect": "architecture",
    "design": "architecture", "plan": "architecture",
    "write": "writing", "summarize": "writing", "document": "writing",
    "report": "writing",
    "research": "research", "search": "research", "find": "research",
    "learn": "research",
    "reason": "reasoning", "analyze": "reasoning", "think": "reasoning",
}


# ── Public API ─────────────────────────────────────────────────────────────

def get_model(name: str) -> dict | None:
    return CATALOG.get(name)

def get_model_id(name: str) -> str:
    entry = CATALOG.get(name)
    if not entry:
        raise KeyError(f"Model {name!r} not in catalog")
    return entry["model_id"]

def get_tier(name: str) -> str:
    entry = CATALOG.get(name)
    return entry["tier"] if entry else "unknown"

def get_provider(name: str) -> str:
    entry = CATALOG.get(name)
    return entry["provider"] if entry else "unknown"

def is_multimodal(name: str) -> bool:
    entry = CATALOG.get(name)
    return entry.get("multimodal", False) if entry else False

def get_default_for_role(role: str, cost_mode: str = "balanced") -> str:
    mode_defaults = ROLE_DEFAULTS.get(cost_mode, ROLE_DEFAULTS["balanced"])
    return mode_defaults.get(role, mode_defaults["default"])

def get_candidates(task_type: str) -> list[tuple[str, float]]:
    task_type = TASK_ALIASES.get(task_type, task_type)
    scored = []
    for name, info in CATALOG.items():
        score = info["strengths"].get(task_type, info["strengths"].get("general", 0.5))
        scored.append((name, score))
    scored.sort(key=lambda x: -x[1])
    return scored

def get_candidates_by_tier(task_type: str, tiers: list[str] | None = None) -> list[tuple[str, float]]:
    task_type = TASK_ALIASES.get(task_type, task_type)
    scored = []
    for name, info in CATALOG.items():
        if tiers and info["tier"] not in tiers:
            continue
        score = info["strengths"].get(task_type, info["strengths"].get("general", 0.5))
        scored.append((name, score))
    scored.sort(key=lambda x: -x[1])
    return scored

def get_smallest_model() -> str:
    local = {k: v for k, v in CATALOG.items() if v["tier"] == "local"}
    if not local:
        return "deepseek-v3.2"
    return min(local, key=lambda m: local[m].get("ram_gb", 99))

def get_ram_requirement(model: str) -> float:
    info = CATALOG.get(model)
    return info["ram_gb"] if info and "ram_gb" in info else 20.0

def estimate_task_cost(model_name: str, input_tokens: int = 2000, output_tokens: int = 2000) -> float:
    entry = CATALOG.get(model_name)
    if not entry:
        return 0.0
    return (input_tokens / 1_000_000) * entry["cost_input_per_m"] + (output_tokens / 1_000_000) * entry["cost_output_per_m"]

def format_catalog() -> str:
    lines = ["LLM Model Catalog:\n"]
    for tier_name in ("local", "budget", "mid", "premium"):
        tier_models = {k: v for k, v in CATALOG.items() if v["tier"] == tier_name}
        if not tier_models:
            continue
        lines.append(f"\n  [{tier_name.upper()}]")
        for name, info in sorted(tier_models.items(), key=lambda x: -max(x[1]["strengths"].values())):
            top = max(info["strengths"], key=info["strengths"].get)
            cost = info["cost_output_per_m"]
            cost_str = "free" if cost == 0 else f"${cost:.2f}/Mo"
            lines.append(f"  {name}  ({cost_str}, tool:{info.get('tool_use_reliability', 0):.0%}) — best: {top}")
    return "\n".join(lines)

def format_role_assignments(cost_mode: str = "balanced") -> str:
    defaults = ROLE_DEFAULTS.get(cost_mode, ROLE_DEFAULTS["balanced"])
    lines = [f"Role Assignments [{cost_mode}]:\n"]
    for role, model in sorted(defaults.items()):
        entry = CATALOG.get(model, {})
        cost = entry.get("cost_output_per_m", 0)
        cost_str = "free" if cost == 0 else f"${cost:.2f}/Mo"
        lines.append(f"  {role:<14} → {model} ({cost_str})")
    return "\n".join(lines)
