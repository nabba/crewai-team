"""
personality.concierge_wrapper — warm front-voice over the agent team's output.

Contract:

    apply_concierge(text) -> str

Returns the original text unchanged when:
  - the runtime toggle (Phase 0) is off
  - the response is empty / very short (already terse enough)
  - the response is structured (JSON, fenced code, slash-command help,
    completion-notification echo, status / skill-registry readouts)

Otherwise, calls a cheap LLM (Anthropic Haiku 4.5 by default) with a system
prompt sourced from ``app/souls/concierge.md`` and asks it to reword the
internal output. The LLM is prompted to keep facts, length, and structure;
the function falls back to the original on any error so a flaky LLM call
never costs the user their reply.

Cost: a single Haiku call per Signal reply when concierge is on. The
default model is ``claude-haiku-4-5-20251001`` at $1 / $5 per 1M tokens
in/out; even verbose replies cost <$0.001/turn.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.config import get_anthropic_api_key

logger = logging.getLogger(__name__)

# Default model for the rewrite. Concierge's job is paraphrasing — Haiku is
# plenty. Override via ``CONCIERGE_MODEL`` env var if a different SKU is
# desired.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Below this character count we don't bother — the response is already
# Signal-bubble-sized and a rewrite adds latency without adding warmth.
_MIN_REWRITE_LEN = 20

# Regex for "looks like JSON" — opens with { or [ on the first non-empty line.
_JSON_OPEN_RE = re.compile(r"^\s*[\{\[]")
_FENCED_CODE_RE = re.compile(r"```")

# Phrase prefixes that mark structured / non-conversational output. Stay
# in sync with the soul file's "When to step aside" list.
_SKIP_PREFIXES = (
    "Usage: /",
    "AndrusAI — Signal commands",       # /help block
    "AndrusAI status",                   # /status block
    "Skill registry —",                  # /skill help
    "Skills (",                          # /skill list
    "Skill: ",                           # /skill show
    "Saved skill ",                      # /skill save confirmation
    "Deleted skill ",                    # /skill delete confirmation
    "✓ ",                                # completion ping echo
    "✗ ",
)


def apply_concierge(text: str, *, model: Optional[str] = None) -> str:
    """Maybe rewrite ``text`` in the concierge voice. Returns ``text``
    unchanged whenever the wrap is disabled or the heuristics suggest
    skipping. Never raises — failures fall back to the input."""
    if not text:
        return text

    try:
        from app.runtime_settings import get_concierge_persona_enabled
        if not get_concierge_persona_enabled():
            return text
    except Exception:
        return text

    if _should_skip(text):
        return text

    return _rewrite_with_llm(text, model=model or DEFAULT_MODEL)


# ── Skip heuristics ────────────────────────────────────────────────────────

def _should_skip(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < _MIN_REWRITE_LEN:
        return True
    # JSON / YAML payload
    if _JSON_OPEN_RE.match(stripped):
        return True
    # Fenced code blocks anywhere — we can't safely rephrase prose around code
    # without risking touching the code itself, so leave it intact.
    if _FENCED_CODE_RE.search(stripped):
        return True
    # Known structured outputs identified by leading line.
    first_line = stripped.split("\n", 1)[0]
    if first_line.startswith(_SKIP_PREFIXES):
        return True
    return False


# ── LLM rewrite ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are AndrusAI's concierge layer.

Take the internal response below and reword it in a warm, conversational tone
suitable for a Signal direct message read on a phone.

Rules — non-negotiable:
- Keep ALL facts, numbers, links, file paths, and proper nouns unchanged.
- Don't add information that isn't in the original.
- Don't soften error messages into vagueness — if something failed, keep it clear.
- Match the original's length within ~20%; don't pad.
- Preserve markdown structure (lists, links, code spans) — don't flatten lists.
- Avoid filler: "Per your request,", "I hope this helps,", "Feel free to,",
  "circling back," "leveraging," "reach out if you have any questions."
- One emoji at most, only if the situation truly calls for one. Default: none.

Output ONLY the rewritten message, no preface, no quotation marks, no
explanation. If the original is already conversational and clear, return
it almost verbatim.
"""


def _rewrite_with_llm(text: str, *, model: str) -> str:
    """Call Anthropic with the concierge prompt. Returns ``text`` on any error."""
    try:
        from anthropic import Anthropic
    except Exception:
        return text

    key = get_anthropic_api_key()
    if not key:
        return text

    try:
        client = Anthropic(api_key=key)
    except Exception as exc:
        logger.debug(f"concierge_wrapper: client init failed: {exc}")
        return text

    try:
        # Keep max_tokens proportional to the input — concierge should match
        # the original within ~20%. Cap at 1024 so a runaway expansion
        # can't blow up cost.
        budget = max(256, min(1024, int(len(text) / 2)))
        resp = client.messages.create(
            model=model,
            max_tokens=budget,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as exc:
        logger.debug(f"concierge_wrapper: LLM call failed: {exc}")
        return text

    try:
        # SDK returns ContentBlock list; concatenate text blocks.
        blocks = getattr(resp, "content", None) or []
        parts = []
        for b in blocks:
            kind = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
            if kind == "text":
                t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else "")
                if t:
                    parts.append(t)
        rewritten = "".join(parts).strip()
    except Exception:
        return text

    if not rewritten:
        return text

    # Guard against a runaway model that ignores the length constraint —
    # if the rewrite is more than 2x the original, fall back. Better the
    # blunt original than a 5-paragraph monologue.
    if len(rewritten) > max(80, len(text) * 2):
        logger.info("concierge_wrapper: rewrite exceeded 2x length, using original")
        return text

    return rewritten
