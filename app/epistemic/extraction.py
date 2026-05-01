"""Path 3 — extract claims from agent output text.

Catches claims that weren't emitted via path 1 (explicit) or path 2
(tool-call boundary). Two strategies:

* **regex_extractor** (default): cheap, deterministic, no LLM call.
  Catches simple declarative shapes like ``X is Y`` or ``X exists``.
  Intentionally conservative — false negatives are cheaper than
  false positives at the calibration gate.

* **llm_extractor** (opt-in via ``EPISTEMIC_PATH3_LLM_EXTRACTION=true``):
  budget-tier LLM call with structured output. Phase 2 ships the
  function as a fallback-to-regex stub; the real LLM wiring lands in
  a later phase. This keeps the dispatcher contract stable so
  callers can always rely on :func:`extract_claims`.

Both extractors return INFERRED claims with low confidence (0.4) — the
text is the agent's own narrative, not exact-answer evidence. The
realtime detectors that consume these claims (notably
:class:`InferenceAsFactDetector`) decide whether to fire based on
register/load_bearing, both of which are caller-supplied at emit time.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from app.epistemic.ledger import VerificationStatus

logger = logging.getLogger(__name__)


#: Maximum claims emitted from a single agent output. Beyond this we
#: stop extracting — a single output emitting >8 claims is a signal of
#: a runaway loop, not productive reasoning.
CAP_PER_OUTPUT: int = 8


@dataclass(frozen=True)
class ExtractedClaim:
    statement: str
    status: VerificationStatus
    confidence: float


# Two narrow regex shapes — anything broader generates noise.
# Pattern 1: "X is Y." / "X is not Y."
# Pattern 2: "X exists." / "X does not exist."
# Sentences end at .!?\n; the captures elide the terminator.
_PATTERN_IS = re.compile(
    r"(?:(?<=^)|(?<=[.!?\n]))\s*"
    r"(?P<sentence>[^.!?\n]{3,200}?\sis(?:\snot)?\s[^.!?\n]{1,200}?)"
    r"(?=[.!?\n])",
    re.IGNORECASE,
)

_PATTERN_EXISTS = re.compile(
    r"(?:(?<=^)|(?<=[.!?\n]))\s*"
    r"(?P<sentence>[^.!?\n]{3,200}?\s(?:exists|does not exist)[^.!?\n]{0,80}?)"
    r"(?=[.!?\n])",
    re.IGNORECASE,
)


def regex_extractor(text: str) -> list[ExtractedClaim]:
    """Cheap regex extraction. Returns up to :data:`CAP_PER_OUTPUT`
    INFERRED claims with confidence 0.4.

    Deduplicates by exact statement string. Skips trivial single-token
    matches (e.g. "It is X.") since those rarely encode a checkable
    assertion.
    """
    if not text:
        return []
    out: list[ExtractedClaim] = []
    seen: set[str] = set()
    for pattern in (_PATTERN_IS, _PATTERN_EXISTS):
        for match in pattern.finditer(text):
            statement = match.group("sentence").strip()
            if not _is_substantive(statement) or statement in seen:
                continue
            seen.add(statement)
            out.append(ExtractedClaim(
                statement=statement,
                status=VerificationStatus.INFERRED,
                confidence=0.4,
            ))
            if len(out) >= CAP_PER_OUTPUT:
                return out
    return out


def llm_extractor(text: str) -> list[ExtractedClaim]:
    """LLM-based extraction. Phase 2: fallback to regex.

    Wired to a budget-tier specialist with structured output in a
    later phase. Keeping the function present (rather than
    NotImplementedError) so the dispatcher contract stays stable.
    """
    logger.debug(
        "epistemic llm_extractor: not yet wired to LLM subsystem; "
        "using regex fallback"
    )
    return regex_extractor(text)


def extract_claims(text: str) -> list[ExtractedClaim]:
    """Dispatch to the configured extractor.

    Reads ``EPISTEMIC_PATH3_LLM_EXTRACTION`` at call time so flipping
    the env var doesn't require a process restart.
    """
    if _llm_enabled():
        try:
            return llm_extractor(text)
        except Exception as exc:
            logger.warning(
                "epistemic extract_claims: llm_extractor raised (%s); "
                "falling back to regex", exc,
            )
            return regex_extractor(text)
    return regex_extractor(text)


def _is_substantive(statement: str) -> bool:
    """Filter out trivial one-pronoun sentences that rarely encode a
    checkable assertion. The 'is/exists' patterns inevitably catch
    things like "It is X" or "This is Y" — those add noise to the
    ledger without contributing detectable bias signal."""
    head = statement.split(maxsplit=1)
    if not head:
        return False
    first_word = head[0].lower().rstrip(",;:")
    return first_word not in {"it", "this", "that", "there", "he", "she", "they"}


def _llm_enabled() -> bool:
    val = os.getenv("EPISTEMIC_PATH3_LLM_EXTRACTION", "").strip().lower()
    return val in ("1", "true", "yes", "on")
