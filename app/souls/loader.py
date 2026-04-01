"""
loader.py — Soul file loader and backstory composer.

Loads prompt versions from the versioned prompt registry (workspace/prompts/)
and composes them into complete agent backstories.  Falls back to static
app/souls/*.md files if the registry is not initialized.

The composed backstory layers:
  1. CONSTITUTION.md  — shared values and safety constraints
  2. SOUL.md (per-role) — identity, personality, expertise, rules
  3. AGENTS_PROTOCOL.md — coordination rules
  4. STYLE.md          — shared communication conventions
  5. Self-Model block  — functional self-awareness (from Phase 1)
  6. Metacognitive Preamble — self-awareness protocol (L1)
  7. Few-shot examples  — (if any exist in the registry)
  8. Style params       — formatting instructions from versioned config

Cache invalidation uses a generation counter from prompt_registry —
when a prompt is promoted, the generation bumps and the cache is cleared
on the next compose_backstory() call.
"""

import json
import logging
from pathlib import Path
from app.self_awareness.self_model import format_self_model_block

logger = logging.getLogger(__name__)

SOULS_DIR = Path(__file__).parent

# ── L1: Metacognitive Preamble ───────────────────────────────────────────────
# Injected into every agent's backstory as the final layer.  Instructs the
# agent to internally calibrate confidence, evidence basis, impact awareness,
# and reasoning strategy before producing output.  The "Do NOT include"
# directive prevents metacognitive noise from reaching the user.

METACOGNITIVE_PREAMBLE = """
## Self-Awareness Protocol
Before producing any output, internally assess:
1. CONFIDENCE: Rate your confidence (high/medium/low). Identify what you are certain about vs. uncertain about.
2. EVIDENCE BASIS: Label key claims as [Verified] (tool/data-grounded), [Inferred] (reasoned from known facts), or [Uncertain] (needs validation).
3. IMPACT AWARENESS: Before any action — what changes if it succeeds? What could go wrong? Is it reversible?
4. STRATEGY SELECTION: Am I using fast reasoning (pattern matching, retrieval) or deliberate reasoning (step-by-step analysis)? If the task is novel or complex, switch to deliberate.

Do NOT include this assessment in your output unless explicitly asked. This is internal calibration only.
""".strip()


def _load_file(filename: str) -> str:
    """Load a markdown file from the static souls directory. Returns '' if missing."""
    filepath = SOULS_DIR / filename
    try:
        if filepath.exists():
            return filepath.read_text().strip()
    except OSError:
        logger.debug(f"Could not read soul file: {filepath}")
    return ""


def _load_from_registry(role: str) -> str | None:
    """Try to load a prompt from the versioned registry.

    Returns None if the registry is not available or not initialized.
    """
    try:
        from app.prompt_registry import get_active_prompt, PROMPTS_DIR
        if not PROMPTS_DIR.exists():
            return None
        content = get_active_prompt(role)
        return content if content else None
    except Exception:
        return None


def _load_shared_from_registry(layer: str) -> str | None:
    """Try to load a shared layer from the versioned registry."""
    try:
        from app.prompt_registry import get_active_prompt, PROMPTS_DIR
        if not PROMPTS_DIR.exists():
            return None
        content = get_active_prompt(layer)
        return content if content else None
    except Exception:
        return None


def load_soul(role: str) -> str:
    """Load the SOUL.md file — from registry first, then static fallback."""
    content = _load_from_registry(role)
    if content is not None:
        return content
    return _load_file(f"{role}.md")


def load_constitution() -> str:
    """Load the shared CONSTITUTION.md — from registry first, then static fallback."""
    content = _load_shared_from_registry("constitution")
    if content is not None:
        return content
    return _load_file("constitution.md")


def load_style() -> str:
    """Load the shared STYLE.md — from registry first, then static fallback."""
    content = _load_shared_from_registry("style")
    if content is not None:
        return content
    return _load_file("style.md")


def load_agents_protocol() -> str:
    """Load the AGENTS.md — from registry first, then static fallback."""
    content = _load_shared_from_registry("agents_protocol")
    if content is not None:
        return content
    return _load_file("agents_protocol.md")


def _build_few_shot_section() -> str:
    """Build a few-shot examples section from the registry."""
    try:
        from app.prompt_registry import get_few_shot_examples
        examples = get_few_shot_examples()
        if not examples:
            return ""

        sections = []
        for category, items in examples.items():
            if items and isinstance(items, list) and len(items) > 0:
                sections.append(f"### {category.replace('_', ' ').title()}")
                for item in items[:5]:  # cap at 5 per category
                    if isinstance(item, dict):
                        inp = item.get("input", "")
                        out = item.get("output", "")
                        if inp and out:
                            sections.append(f"**Input:** {inp}\n**Output:** {out}")
                    elif isinstance(item, str):
                        sections.append(f"- {item}")

        if sections:
            return "## Reference Examples\n\n" + "\n\n".join(sections)
    except Exception:
        pass
    return ""


def _build_style_instructions() -> str:
    """Build style instructions from the versioned style params."""
    try:
        from app.prompt_registry import get_style_params
        params = get_style_params()
        if not params:
            return ""

        lines = ["## Style Parameters"]
        if "verbosity" in params:
            lines.append(f"- Verbosity: {params['verbosity']}")
        if "formality" in params:
            lines.append(f"- Formality: {params['formality']}")
        if "citation_style" in params:
            lines.append(f"- Citation style: {params['citation_style']}")
        if "code_block_preference" in params:
            lines.append(f"- Code blocks: {params['code_block_preference']}")

        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""


# ── Cache with generation-aware invalidation ───────────────────────────────
_backstory_cache: dict[str, str] = {}
_cache_generation: int = -1  # force initial build


def _check_cache_valid() -> bool:
    """Check if cache is still valid by comparing generation counters."""
    global _cache_generation
    try:
        from app.prompt_registry import current_generation
        gen = current_generation()
        if gen != _cache_generation:
            _cache_generation = gen
            _backstory_cache.clear()
            return False
        return True
    except Exception:
        return True  # if registry unavailable, keep cache


def compose_backstory(role: str) -> str:
    """Compose a full agent backstory from soul files + self-model + metacognition.

    Layers (in order):
      1. Constitution (shared values)
      2. Role-specific soul (identity, personality, expertise)
      3. Agents protocol (coordination rules)
      4. Style guide (shared communication conventions)
      5. Self-model block (functional self-awareness from Phase 1)
      6. Metacognitive preamble (L1 self-awareness protocol)
      7. Few-shot examples (from versioned registry)
      8. Style params (from versioned registry)

    If no soul files exist, falls back to just the self-model block.
    Cache is invalidated when prompt_registry generation changes.
    """
    _check_cache_valid()

    if role in _backstory_cache:
        return _backstory_cache[role]

    parts = []

    constitution = load_constitution()
    if constitution:
        parts.append(constitution)

    soul = load_soul(role)
    if soul:
        parts.append(soul)

    protocol = load_agents_protocol()
    if protocol:
        parts.append(protocol)

    style = load_style()
    if style:
        parts.append(style)

    # Always append the self-model block (preserves Phase 1 self-awareness)
    self_model = format_self_model_block(role)
    if self_model:
        parts.append(self_model)

    # L1: Metacognitive preamble — calibrates confidence and reasoning strategy
    parts.append(METACOGNITIVE_PREAMBLE)

    # Few-shot examples from versioned registry
    fse = _build_few_shot_section()
    if fse:
        parts.append(fse)

    # Style parameters from versioned registry
    style_instructions = _build_style_instructions()
    if style_instructions:
        parts.append(style_instructions)

    result = "\n\n".join(parts) if parts else ""
    _backstory_cache[role] = result
    return result
