"""LLM-driven source proposals.

When a workspace has a ``seed_prompt`` (or a synthesised grand task — Phase 11)
this module asks a cheap-tier LLM for 3–5 plausible research sources.
The user accepts/edits/rejects via the React UI; only accepted suggestions
become real Source rows.

Phase 6 ships ``web_search``-typed suggestions only. Phase 6.5+ widens the
allowed types when RSS / URL-poll connectors land.
"""

from __future__ import annotations

import json
import logging
import re

from app.companion import config as _config
from app.companion.sources import ALLOWED_TYPES

logger = logging.getLogger(__name__)

DEFAULT_MAX_SUGGESTIONS = 5


def propose(workspace_id: str, *,
            max_count: int = DEFAULT_MAX_SUGGESTIONS) -> list[dict]:
    """Return up to ``max_count`` source suggestions.

    Each suggestion is a dict ``{type, config, reason}``. Empty list when
    the workspace has no seed yet, or when the LLM call / parse fails.
    Surface in the UI as accept/edit/dismiss cards.
    """
    cfg = _config.load(workspace_id)
    if cfg is None:
        return []
    seed = (cfg.seed_prompt or "").strip()
    if not seed:
        return []

    prompt = _PROMPT_TEMPLATE.format(seed=seed[:1500], n=max_count)
    try:
        raw = _invoke_suggester(prompt)
    except Exception as exc:
        logger.debug("companion.source_suggester: LLM failed: %s", exc)
        return []

    proposals = _parse_proposals(raw, max_count=max_count)
    return [p for p in proposals if p["type"] in ALLOWED_TYPES]


_PROMPT_TEMPLATE = """\
Given this workspace seed: "{seed}"

Propose {n} concrete sources useful for ongoing research.
Output ONLY a JSON array of objects with these fields:
  type    — "web_search"
  config  — for "web_search": {{"query": "<search string>"}}
  reason  — one short sentence on why this source helps

Each query should pull current material on a different angle of the seed.
Return only the JSON array, no markdown fences, no commentary.
"""


def _invoke_suggester(prompt: str) -> str:
    """Indirection over the cheap-tier LLM call, for testability."""
    from app.llm_factory import create_specialist_llm
    llm = create_specialist_llm(max_tokens=600, role="researcher")
    return str(llm.call(prompt))


def _parse_proposals(raw: str, *, max_count: int) -> list[dict]:
    """Best-effort JSON-array extraction. Returns at most ``max_count`` items."""
    if not raw:
        return []
    text = raw.strip()
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```", text,
                     flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(text[start: end + 1])
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items[:max_count]:
        if not isinstance(item, dict):
            continue
        type_ = str(item.get("type", "")).strip()
        cfg = item.get("config")
        reason = str(item.get("reason", ""))[:240]
        if not type_ or not isinstance(cfg, dict):
            continue
        out.append({"type": type_, "config": cfg, "reason": reason})
    return out
