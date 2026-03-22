"""
loader.py — Soul file loader and backstory composer.

Loads SOUL.md, CONSTITUTION.md, STYLE.md, and AGENTS.md files from the
app/souls/ directory and composes them into complete agent backstories.

The composed backstory layers:
  1. CONSTITUTION.md  — shared values and safety constraints
  2. SOUL.md (per-role) — identity, personality, expertise, rules
  3. STYLE.md          — shared communication conventions
  4. Self-Model block  — functional self-awareness (from Phase 1)
  5. Metacognitive Preamble — self-awareness protocol (L1)

Falls back gracefully: if soul files don't exist, returns just the
self-model block to preserve Phase 1-4 behavior.
"""

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
    """Load a markdown file from the souls directory. Returns '' if missing."""
    filepath = SOULS_DIR / filename
    try:
        if filepath.exists():
            return filepath.read_text().strip()
    except OSError:
        logger.debug(f"Could not read soul file: {filepath}")
    return ""


def load_soul(role: str) -> str:
    """Load the SOUL.md file for a specific agent role."""
    return _load_file(f"{role}.md")


def load_constitution() -> str:
    """Load the shared CONSTITUTION.md."""
    return _load_file("constitution.md")


def load_style() -> str:
    """Load the shared STYLE.md."""
    return _load_file("style.md")


def load_agents_protocol() -> str:
    """Load the AGENTS.md coordination protocol."""
    return _load_file("agents_protocol.md")


# S14: Cache composed backstories — soul files don't change at runtime.
# Saves disk reads and string concatenation on every agent creation.
_backstory_cache: dict[str, str] = {}


def compose_backstory(role: str) -> str:
    """Compose a full agent backstory from soul files + self-model + metacognition.

    Layers (in order):
      1. Constitution (shared values)
      2. Role-specific soul (identity, personality, expertise)
      3. Style guide (shared communication conventions)
      4. Self-model block (functional self-awareness from Phase 1)
      5. Metacognitive preamble (L1 self-awareness protocol)

    If no soul files exist, falls back to just the self-model block.
    Results are cached at module level (soul files don't change at runtime).
    """
    if role in _backstory_cache:
        return _backstory_cache[role]

    parts = []

    constitution = load_constitution()
    if constitution:
        parts.append(constitution)

    soul = load_soul(role)
    if soul:
        parts.append(soul)

    style = load_style()
    if style:
        parts.append(style)

    # Always append the self-model block (preserves Phase 1 self-awareness)
    self_model = format_self_model_block(role)
    if self_model:
        parts.append(self_model)

    # L1: Metacognitive preamble — calibrates confidence and reasoning strategy
    parts.append(METACOGNITIVE_PREAMBLE)

    result = "\n\n".join(parts) if parts else ""
    _backstory_cache[role] = result
    return result
