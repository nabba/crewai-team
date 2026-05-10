"""Runbooks B + H — model-capability mismatch handlers (auto-action).

Two recurring families:

  * **B**: ``"<embed-model>" does not support chat`` — an embed-only
    model was routed to a chat endpoint. Signal: ``logger="root"``,
    sample contains ``does not support chat``.
  * **H**: ``Model '<m>' in litellm does not support function calling`` —
    Mem0 LLM extraction sent ``tool_choice`` to a non-tool model.
    Signal: ``logger="mem0.memory.main"``, sample contains
    ``does not support function calling``.

The handlers do TWO things on each anomaly:

  1. **Track + alert** (existing behaviour): record the offending
     model in ``workspace/self_heal/model_capabilities.json`` and
     emit a Signal alert per (model, capability) per 24 h.
  2. **Auto-action** (Q2 2026): write the model name to the matching
     runtime blocklist via ``app.runtime_settings`` so subsystems
     stop routing the failing capability to the model. The LLM
     selector consults ``chat_blocked_models`` at default-tier
     selection (``app.llm_selector.select_model`` Step 5.5);
     consumer code consults ``no_function_calling_models`` to fall
     back to unstructured extraction.

Catch-all pattern (``.*``) is used here because the SHA-1 signature
varies with the model name, which is part of the normalized message.
The handler defends with a sample-substring check so unrelated
anomalies never reach the action path.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.healing.handlers._common import (
    RateLimiter,
    audit_event,
    read_state_json,
    sample_contains,
    send_signal_alert,
    write_state_json,
)
from app.healing.runbooks import RunbookResult, register_runbook

logger = logging.getLogger(__name__)

_STATE_FILE = "model_capabilities.json"

_MODEL_NAME_PATTERNS = [
    # Both single- and double-quoted forms appear in the wild.
    re.compile(r"['\"]([\w./:_+\-]+)['\"]\s+(?:in litellm\s+)?does not support"),
    re.compile(r"Model\s+['\"]([\w./:_+\-]+)['\"]"),
]

# 1 alert per (model, capability) per 24h.
_LIMITER = RateLimiter(cooldown_seconds=24 * 3600)


def _extract_model_name(sample: str) -> str:
    for pat in _MODEL_NAME_PATTERNS:
        m = pat.search(sample)
        if m:
            return m.group(1)
    return ""


def _record_capability_mismatch(model: str, capability: str) -> dict[str, Any]:
    """Append model → unsupported_capability to the state file. Returns the
    updated entry so callers can decide whether to alert (first-seen vs
    already-tracked).
    """
    state = read_state_json(_STATE_FILE, {"models": {}})
    models = state.setdefault("models", {})
    entry = models.setdefault(model, {"unsupported": [], "first_seen": time.time(), "count": 0})
    entry["count"] = int(entry.get("count", 0)) + 1
    entry["last_seen"] = time.time()
    if capability not in entry["unsupported"]:
        entry["unsupported"].append(capability)
        entry["newly_added"] = True
    else:
        entry["newly_added"] = False
    write_state_json(_STATE_FILE, state)
    return entry


def _apply_runtime_block(model: str, capability: str) -> bool:
    """Write the offending model to the matching runtime blocklist.
    Returns ``True`` when the model is newly added, ``False`` when
    already in the list (or on any failure — fail-safe).

    Capability → setting:
      * ``"chat"``             → ``chat_blocked_models``
      * ``"function_calling"`` → ``no_function_calling_models``

    Other capabilities are no-op; only these two have a consumer
    (the LLM selector / Mem0 extractor) that consults the list.
    """
    if model in (None, "", "<unknown>"):
        return False
    try:
        from app.runtime_settings import (
            add_chat_blocked_model,
            add_no_function_calling_model,
        )
    except Exception:
        logger.debug(
            "model_capability: runtime_settings setters unavailable",
            exc_info=True,
        )
        return False
    try:
        if capability == "chat":
            return bool(add_chat_blocked_model(model))
        if capability == "function_calling":
            return bool(add_no_function_calling_model(model))
    except Exception:
        logger.debug(
            "model_capability: runtime block write failed for %s/%s",
            model, capability, exc_info=True,
        )
    return False


def _route(anomaly: dict, capability: str, runbook_name: str) -> RunbookResult:
    sample = anomaly.get("pattern_sample") or anomaly.get("sample") or ""
    model = _extract_model_name(sample) or "<unknown>"

    entry = _record_capability_mismatch(model, capability)
    blocked_now = _apply_runtime_block(model, capability)

    audit_event(
        f"model_capability_mismatch_{capability}",
        model=model,
        capability=capability,
        count=entry.get("count"),
        pattern_signature=anomaly.get("pattern_signature"),
        runbook=runbook_name,
        runtime_block_added=blocked_now,
    )

    # Alert at most once per (model, capability) per 24h. The first
    # alert per model now confirms the runtime block is in effect —
    # operator gets a single actionable summary instead of a stream
    # of "consider removing it" reminders.
    if entry.get("newly_added") or _LIMITER.allow():
        if blocked_now:
            tail = (
                f"Auto-block applied to runtime_settings."
                f"{'chat_blocked_models' if capability == 'chat' else 'no_function_calling_models'} "
                f"— router will stop sending this capability to the model."
            )
        elif capability in ("chat", "function_calling"):
            # Already in the list — nothing to do.
            tail = (
                f"Already in runtime_settings."
                f"{'chat_blocked_models' if capability == 'chat' else 'no_function_calling_models'}; "
                f"investigate why the route is still picking it."
            )
        else:
            tail = (
                f"Tracked in `workspace/self_heal/model_capabilities.json`. "
                f"No runtime auto-block for capability `{capability}` — "
                f"operator action required."
            )
        send_signal_alert(
            f"⚠️  Self-heal: model `{model}` does not support **{capability}** "
            f"(seen {entry.get('count')} times). {tail}",
            tag=runbook_name,
        )

    return RunbookResult(
        name=runbook_name,
        success=True,
        detail=(
            f"recorded {model} as unsupported({capability})"
            + ("; runtime_block_added" if blocked_now else "")
        ),
        extra={
            "model": model,
            "capability": capability,
            "runtime_block_added": blocked_now,
        },
    )


def _handle_embed_misroute(anomaly: dict) -> RunbookResult:
    if not sample_contains(anomaly, "does not support chat"):
        return RunbookResult(
            name="embed_model_misroute_alert",
            success=False,
            detail="sample mismatch",
            error="sample_mismatch",
        )
    return _route(anomaly, "chat", "embed_model_misroute_alert")


def _handle_no_function_calling(anomaly: dict) -> RunbookResult:
    if not sample_contains(anomaly, "does not support function calling"):
        return RunbookResult(
            name="mem0_no_function_calling_alert",
            success=False,
            detail="sample mismatch",
            error="sample_mismatch",
        )
    return _route(anomaly, "function_calling", "mem0_no_function_calling_alert")


# Both B and H are catch-all patterns (model name varies → SHA-1 varies),
# so the dispatcher can only route ONE catch-all per anomaly. Instead of
# registering directly here we expose ``handle_embed_misroute`` /
# ``handle_no_function_calling`` for the multi-router in
# ``handlers/multi_router.py``, which is the single ``.*`` entry that
# dispatches by sample-substring.
handle_embed_misroute = _handle_embed_misroute
handle_no_function_calling = _handle_no_function_calling


def register() -> None:
    """No-op; the router in ``multi_router.py`` registers the catch-all."""
