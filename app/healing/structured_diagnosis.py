"""Structured fix generator for code-class errors (Q2 §39).

The classic ``error_diagnosis`` path produces a *prose proposal*:
"add `from foo import bar` to file X line 12." Operators must read,
parse, and apply manually. Empirically this drove the May 2026
audit's "0 resolved, 1 attempted" finding — diagnoses produced,
never applied.

This module generates ``(path, new_content)`` patches the change-
request system can apply directly. The LLM reads the full current
file, the error message + traceback, and returns a complete
replacement-file content. The CR's diff shows exactly what changed
— operator approves with a single 👍.

Confidence-gated: every generation reports a self-assessed
confidence. Below ``current_threshold()`` (auto-tuned in
``diagnosis_auto_tune.py``), the module declines and returns
``None`` so the caller falls back to the prose path. Telemetry on
every attempt feeds the auto-tuner.

Multi-site bug detection: the prompt instructs the LLM to decline
when the fix would need to touch multiple files. Single-site fixes
only — anything broader requires operator-led work, period.

HOT-1 observation hook: every generation attempt (filed OR
declined) emits a metacognitive-repair event to
``workspace/subia/observations/metacognitive_repair.jsonl``. The
hook is passive — see ``docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md``
for what the future HOT-1 probe will compute from this data.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────


# Files larger than this skip structured diagnosis (LLM context-window
# pressure + cost). Falls back to prose for big files.
_MAX_FILE_BYTES_FOR_STRUCTURED = 50 * 1024  # 50 KB

# Per-pattern rate limit so a fast-firing pattern doesn't burn LLM
# spend before the auto-tuner has a chance to react.
_PER_PATTERN_PER_HOUR = 3


@dataclass(frozen=True)
class StructuredFix:
    """Output of a successful structured diagnosis."""

    path: str
    new_content: str
    old_content: str
    confidence: float
    reasoning: str
    declined: bool = False
    decline_reason: str = ""

    @property
    def is_actionable(self) -> bool:
        return (not self.declined) and bool(self.new_content) and self.path


# ── Public API ────────────────────────────────────────────────────────


def generate_structured_fix(
    *,
    error_message: str,
    error_traceback: str,
    file_path: str,
    file_content: str,
    pattern_signature: str = "",
    error_class: str = "",
) -> Optional[StructuredFix]:
    """Generate a (path, new_content) fix proposal via LLM.

    Returns:
      * ``StructuredFix(declined=False, ...)`` when the LLM produced
        a plausible single-site fix above the confidence threshold.
      * ``StructuredFix(declined=True, decline_reason="...")`` when
        the LLM decided not to fix (multi-site bug, ambiguity, etc).
        Caller should fall back to prose diagnosis.
      * ``None`` on infrastructure failure (LLM unavailable, parse
        error, file too large, rate-limited). Caller should fall
        back to prose.

    All paths emit telemetry + HOT-1 observation events. None of
    these emissions raise — every error path degrades silently into
    "fall back to prose."

    Q16 Theme 8 (PROGRAM §51): consults
    :mod:`app.healing.hot1_consultation` for prior-attempt history
    before generating. When the pattern has chronically failed (≥3
    declines, no successes), short-circuits as Guard 0 to save LLM
    spend. Otherwise the prior-attempt hint is spliced into the LLM
    prompt.
    """
    # Guard 0: HOT-1 consultation — does prior history say skip?
    hot1_hint_for_prompt: Optional[str] = None
    try:
        from app.healing.hot1_consultation import consult as _hot1_consult
        hot1_context = _hot1_consult(
            pattern_signature=pattern_signature,
            file_path=file_path,
        )
        if hot1_context.get("recommendation") == "skip":
            _emit_telemetry_declined(
                pattern_signature=pattern_signature,
                file_path=file_path,
                error_class=error_class,
                confidence=0.0,
                decline_reason="hot1_skip_chronic_failure",
            )
            return None
        hot1_hint_for_prompt = hot1_context.get("hint_for_prompt")
    except Exception:
        logger.debug("structured_diagnosis: hot1 consult failed", exc_info=True)

    # Guard 1: file size cap.
    if len(file_content.encode("utf-8")) > _MAX_FILE_BYTES_FOR_STRUCTURED:
        _emit_telemetry_declined(
            pattern_signature=pattern_signature,
            file_path=file_path,
            error_class=error_class,
            confidence=0.0,
            decline_reason="file_too_large",
        )
        return None

    # Guard 2: per-pattern rate limit (cheap LLM-spend guard).
    if not _within_rate_limit(pattern_signature):
        _emit_telemetry_declined(
            pattern_signature=pattern_signature,
            file_path=file_path,
            error_class=error_class,
            confidence=0.0,
            decline_reason="rate_limited",
        )
        return None

    # Guard 3: LLM call.
    fix = _call_llm_for_fix(
        error_message=error_message,
        error_traceback=error_traceback,
        file_path=file_path,
        file_content=file_content,
        prior_attempts_hint=hot1_hint_for_prompt,
    )
    if fix is None:
        return None

    # Always emit HOT-1 observation BEFORE confidence-gating —
    # the metacognitive event is "the system reasoned about its
    # own error," whether or not the operator-side filing fires.
    _emit_hot1_observation(
        fix=fix,
        pattern_signature=pattern_signature,
        file_path=file_path,
        error_class=error_class,
    )

    # If the LLM declined, return that explicitly so the caller can
    # surface the decline_reason (and the auto-tuner gets a
    # "declined" event).
    if fix.declined:
        _emit_telemetry_declined(
            pattern_signature=pattern_signature,
            file_path=file_path,
            error_class=error_class,
            confidence=fix.confidence,
            decline_reason=fix.decline_reason or "llm_declined",
        )
        return fix

    # Confidence gate.
    threshold = _current_threshold()
    if fix.confidence < threshold:
        _emit_telemetry_declined(
            pattern_signature=pattern_signature,
            file_path=file_path,
            error_class=error_class,
            confidence=fix.confidence,
            decline_reason="below_threshold",
        )
        return StructuredFix(
            path=fix.path,
            new_content=fix.new_content,
            old_content=fix.old_content,
            confidence=fix.confidence,
            reasoning=fix.reasoning,
            declined=True,
            decline_reason=f"below_threshold (active={threshold:.2f})",
        )

    # Filed-eligible. Caller will create_request and emit "filed"
    # telemetry once the CR exists.
    return fix


def current_threshold() -> float:
    """Public read of the active confidence threshold. Caller-friendly
    wrapper around the internal helper that consults runtime_settings
    overrides AND the auto-tune state file."""
    return _current_threshold()


# ── LLM call ──────────────────────────────────────────────────────────


_LLM_SYSTEM_PROMPT = """You are a structural code-fix generator for a Python codebase.

You receive:
  * error_message + error_traceback that fired in production
  * file_path of the most-likely-broken file
  * the FULL current contents of that file

Your job is to produce a complete replacement for that single file
that fixes the root cause of the error. STRICT rules:

  1. SINGLE-SITE ONLY. If the fix would require touching multiple
     files, decline by returning {"declined": true,
     "decline_reason": "multi_site"}. Operator-led work only.

  2. ADDITIVE-PREFERRED. Adding lines (imports, defaults, defensive
     guards) is safer than rewriting logic. If the fix requires
     deleting > 5 lines of existing code, decline with
     {"declined": true, "decline_reason": "destructive"}.

  3. NO INVENTED REFERENCES. Every import / function / class you
     reference must exist in the file content provided OR be a
     standard-library / well-known dependency. If you need a
     symbol whose definition you can't see, decline with
     {"declined": true, "decline_reason": "missing_context"}.

  4. SELF-RATE CONFIDENCE. Output a float 0.0-1.0 for confidence.
     0.9+: you're certain. 0.7-0.9: high confidence with one minor
     uncertainty. 0.5-0.7: plausible but unverified. <0.5:
     speculative — strongly consider declining instead.

Output STRICT JSON. No prose outside the JSON. Schema:

  {
    "declined": false,
    "new_content": "<full replacement file content>",
    "confidence": 0.85,
    "reasoning": "<one paragraph explaining the causal hypothesis>"
  }

OR

  {
    "declined": true,
    "decline_reason": "multi_site|destructive|missing_context|other",
    "confidence": 0.0,
    "reasoning": "<one paragraph explaining why you can't fix>"
  }
"""


def _call_llm_for_fix(
    *,
    error_message: str,
    error_traceback: str,
    file_path: str,
    file_content: str,
    prior_attempts_hint: Optional[str] = None,
) -> Optional[StructuredFix]:
    """Issue the LLM call. Returns None on infrastructure failure
    (caller falls back to prose). Returns StructuredFix on success
    OR decline.

    Q16 Theme 8 (PROGRAM §51): ``prior_attempts_hint`` is the optional
    "what's been tried before" text from
    :mod:`app.healing.hot1_consultation`. When provided, it's
    prepended to the user message so the LLM doesn't propose blind.
    """
    try:
        from anthropic import Anthropic
        from app.config import get_anthropic_api_key
    except Exception:
        return None
    key = get_anthropic_api_key()
    if not key:
        return None
    try:
        client = Anthropic(api_key=key)
    except Exception:
        return None

    hint_block = (
        f"=== prior_attempt_context ===\n{prior_attempts_hint}\n\n"
        if prior_attempts_hint else ""
    )
    user_msg = (
        f"file_path: {file_path}\n\n"
        f"{hint_block}"
        f"=== error_message ===\n{error_message}\n\n"
        f"=== error_traceback ===\n{error_traceback[:2000]}\n\n"
        f"=== current file content ===\n{file_content}"
    )

    try:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception:
        return None

    raw_text = _extract_text(resp)
    if not raw_text:
        return None
    parsed = _parse_strict_json(raw_text)
    if parsed is None:
        return None

    declined = bool(parsed.get("declined"))
    confidence = float(parsed.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(parsed.get("reasoning") or "")[:1000]

    if declined:
        return StructuredFix(
            path=file_path,
            new_content="",
            old_content=file_content,
            confidence=confidence,
            reasoning=reasoning,
            declined=True,
            decline_reason=str(parsed.get("decline_reason") or "other")[:80],
        )

    new_content = parsed.get("new_content")
    if not isinstance(new_content, str) or not new_content.strip():
        return None
    if new_content == file_content:
        # LLM returned the file unchanged — equivalent to declining.
        return StructuredFix(
            path=file_path,
            new_content="",
            old_content=file_content,
            confidence=confidence,
            reasoning=reasoning,
            declined=True,
            decline_reason="no_change_proposed",
        )

    return StructuredFix(
        path=file_path,
        new_content=new_content,
        old_content=file_content,
        confidence=confidence,
        reasoning=reasoning,
    )


def _extract_text(resp) -> str:
    blocks = getattr(resp, "content", None) or []
    parts: list[str] = []
    for b in blocks:
        if getattr(b, "type", None) == "text":
            parts.append(getattr(b, "text", "") or "")
    return "".join(parts).strip()


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_strict_json(text: str) -> Optional[dict]:
    """Tolerate the model wrapping the output in a ```json fence."""
    candidate = text.strip()
    fence_match = _FENCE_RE.match(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# ── Threshold / rate-limit lookup ─────────────────────────────────────


def _current_threshold() -> float:
    """Resolve the active confidence threshold.

    Read precedence:
      1. ``runtime_settings.structured_diagnosis_threshold_override``
         — operator manual pin (overrides everything when set)
      2. auto-tune state file — current value within
         ``[floor, ceiling]`` from runtime_settings
      3. fallback default 0.70 — used when neither source is
         readable (degraded boot)
    """
    try:
        from app.runtime_settings import get_structured_diagnosis_threshold_override
        override = get_structured_diagnosis_threshold_override()
        if override is not None:
            return float(override)
    except Exception:
        pass
    try:
        from app.healing.diagnosis_auto_tune import current_state
        state = current_state()
        return float(state.get("current", 0.70))
    except Exception:
        return 0.70


def _within_rate_limit(pattern_signature: str) -> bool:
    """Per-pattern hourly rate limit. Read recent telemetry and
    count attempts in the last hour."""
    if not pattern_signature:
        return True
    try:
        from app.healing.diagnosis_telemetry import attempts_for_pattern_in_window
        n = attempts_for_pattern_in_window(
            pattern_signature, window_seconds=3600,
        )
        return n < _PER_PATTERN_PER_HOUR
    except Exception:
        return True


# ── Telemetry helpers ─────────────────────────────────────────────────


def _emit_telemetry_declined(*, pattern_signature, file_path, error_class,
                              confidence, decline_reason) -> None:
    try:
        from app.healing.diagnosis_telemetry import record_declined
        record_declined(
            pattern_signature=pattern_signature,
            file_path=file_path,
            error_class=error_class,
            confidence=confidence,
            threshold=_current_threshold(),
            decline_reason=decline_reason,
        )
    except Exception:
        logger.debug("structured_diagnosis: telemetry-declined emit failed",
                     exc_info=True)


# ── HOT-1 metacognitive observation ──────────────────────────────────


def _emit_hot1_observation(
    *,
    fix: StructuredFix,
    pattern_signature: str,
    file_path: str,
    error_class: str,
) -> None:
    """Emit one row to the metacognitive-repair observation log.

    The future HOT-1 probe will read this log to compute its
    indicator score (frequency / diversity / quality / hypothesis
    depth). For now the log is passive collection.

    See ``docs/CONSCIOUSNESS_HOT1_OBSERVATIONS.md`` for the schema
    and the scoring sketch.
    """
    try:
        log_path = Path(
            os.environ.get("HOT1_OBSERVATION_LOG")
            or "/app/workspace/subia/observations/metacognitive_repair.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        delta_added, delta_removed = _delta_lines(fix.old_content, fix.new_content)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "metacognitive_repair_proposal",
            "indicator_relevance": ["HOT-1", "HOT-2"],
            "originating_error": {
                "pattern_signature": pattern_signature,
                "file_path": file_path,
                "error_class": error_class,
            },
            "higher_order_thought": {
                "causal_hypothesis": fix.reasoning[:1000],
                "hypothesis_length_chars": len(fix.reasoning or ""),
                "self_assessed_confidence": fix.confidence,
                "declined": fix.declined,
                "decline_reason": fix.decline_reason if fix.declined else None,
            },
            "proposed_intervention": {
                "kind": "code_patch",
                "target_path": fix.path,
                "delta_additive_only": delta_removed == 0,
                "delta_lines_added": delta_added,
                "delta_lines_removed": delta_removed,
            },
            "outcome": None,  # filled in by outcome reconciler when CR resolves
        }
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(log_path, json.dumps(row, sort_keys=True), max_lines=5000)
    except Exception:
        logger.debug("structured_diagnosis: HOT-1 observation emit failed",
                     exc_info=True)


def _delta_lines(old_content: str, new_content: str) -> tuple[int, int]:
    """Compute (added, removed) line counts via difflib. Same shape
    as ``app.change_requests.validator._net_line_delta``."""
    if not old_content and not new_content:
        return 0, 0
    if not old_content:
        return len(new_content.splitlines()), 0
    if not new_content:
        return 0, len(old_content.splitlines())
    try:
        import difflib
        diff = list(difflib.unified_diff(
            old_content.splitlines(), new_content.splitlines(), lineterm="",
        ))
        added = sum(
            1 for line in diff
            if line.startswith("+") and not line.startswith("+++")
        )
        removed = sum(
            1 for line in diff
            if line.startswith("-") and not line.startswith("---")
        )
        return added, removed
    except Exception:
        return 0, 0
