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
from app.subia.self.model import format_self_model_block

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

## Tool-First Principle
You have real tools attached to this task. They are listed in your system prompt with their exact
names and argument schemas. You MUST treat tool calls as your primary way of getting new information
or changing state. Follow this protocol:

1. SCAN your tool list before every answer. Ask: "Does any of these tools fit this request, even
   loosely?" If yes — call it. If multiple fit, call the most specific one first.

2. NEVER refuse with "I don't have access", "I can't do that", "I'm unable to", "I don't have the
   ability", or any equivalent phrase if a tool that could plausibly handle the request exists in
   your tool list. Instead, ATTEMPT THE TOOL. Report what happened. A tool failing is information;
   refusing without trying is a failure mode.

3. CHAIN tools freely. If one tool gives you partial data (e.g. a list of IDs), use another tool to
   expand it (fetch details, search, read). Compose up to 5 tool calls before you conclude.

4. EXPLORE when uncertain. If the user asks something novel and no tool looks perfect, try the
   closest match. Tools like `web_fetch`, `web_search`, `knowledge_search`, `session_search`,
   `memory_search`, MCP bridges, and `browser_fetch` are broad enough to help with almost anything
   — use them as catch-alls.

5. ONLY THEN consider refusing — and even then, your refusal must name which tools you tried and
   why they failed. "I tried X and Y, both returned empty, so I cannot answer with high confidence"
   is acceptable. "I can't do that" without a tool attempt is never acceptable.

6. If you cannot remember which tools you have, re-read the tool list in your system prompt before
   answering — don't rely on prior knowledge of what this agent "normally" does.
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


# ── Reasoning method preambles (Mechanism 1 — DMAD) ─────────────────────────
# Parsed from reasoning_methods.md. Each section is keyed by the method name
# declared in the `## METHOD: <name>` header line.
_reasoning_methods_cache: dict[str, str] | None = None


def _load_reasoning_methods() -> dict[str, str]:
    """Parse reasoning_methods.md into {method_name: preamble_text}.

    Returns an empty dict if the file is missing, keeping creative mode a
    strictly additive feature.
    """
    global _reasoning_methods_cache
    if _reasoning_methods_cache is not None:
        return _reasoning_methods_cache
    raw = _load_file("reasoning_methods.md")
    result: dict[str, str] = {}
    if not raw:
        _reasoning_methods_cache = result
        return result
    current_name: str | None = None
    current_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("## METHOD:"):
            if current_name is not None:
                result[current_name] = "\n".join(current_lines).strip()
            current_name = stripped.removeprefix("## METHOD:").strip()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)
    if current_name is not None:
        result[current_name] = "\n".join(current_lines).strip()
    _reasoning_methods_cache = result
    return result


def get_reasoning_method(name: str) -> str:
    """Return the preamble for a named reasoning method, or '' if unknown."""
    return _load_reasoning_methods().get(name, "")


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


def compose_backstory(role: str, reasoning_method: str | None = None) -> str:
    """Compose a full agent backstory from soul files + self-model + metacognition.

    Layers (in order):
      1. Constitution (shared values)
      2. Role-specific soul (identity, personality, expertise)
      3. Agents protocol (coordination rules)
      4. Style guide (shared communication conventions)
      4.5. Reasoning method preamble (creative mode only — Mechanism 1)
      5. Self-model block (functional self-awareness from Phase 1)
      6. Metacognitive preamble (L1 self-awareness protocol)
      7. Few-shot examples (from versioned registry)
      8. Style params (from versioned registry)

    If no soul files exist, falls back to just the self-model block.
    Cache is invalidated when prompt_registry generation changes.

    `reasoning_method` (creative mode) is looked up in reasoning_methods.md
    and inserted as layer 4.5. When None, legacy behavior is preserved and
    the existing cache identity is used.
    """
    _check_cache_valid()

    cache_key = role if reasoning_method is None else f"{role}::{reasoning_method}"
    if cache_key in _backstory_cache:
        return _backstory_cache[cache_key]

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

    # Layer 4.5: reasoning-method preamble (creative mode only)
    if reasoning_method:
        method_text = get_reasoning_method(reasoning_method)
        if method_text:
            parts.append(method_text)
        else:
            logger.warning(
                f"compose_backstory: unknown reasoning_method {reasoning_method!r} "
                f"for role={role}; continuing without it."
            )

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
    _backstory_cache[cache_key] = result
    return result
