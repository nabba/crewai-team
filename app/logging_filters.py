"""Logging filters for the structured-error JSONL handler.

Background. The JSONL handler in ``app/error_handler.py`` is attached
to the **root logger** at WARNING level, so any third-party library
that emits a WARNING ends up in ``workspace/logs/errors.jsonl``. Most
of those messages are benign and our code already handles the
underlying condition correctly — but pattern_learner sees them as
"new uncovered failure patterns" and proposes runbook scaffolds.

This module provides ``JsonlNoiseFilter``, a ``logging.Filter`` that
drops a fixed allowlist of well-known third-party WARN messages
*from the JSONL handler only*. The same records still flow through
normal log handlers (stdout / docker logs / debug consoles) so
debugging isn't affected.

Why a hardcoded list (not pattern-learner-driven)?

  Each entry here represents an explicit operator decision that the
  message is informational despite being WARN-level. We do NOT want
  the noise filter to grow automatically — every new entry should
  go through code review so we don't silently hide a real regression.

Add new entries with sparing judgment:

  * Confirm the underlying condition is already handled.
  * Confirm the noise volume is high (≥10/week pattern_learner threshold).
  * Confirm there's no actionable signal in the message itself.

For our own code, prefer demoting the log site to INFO instead — that
keeps the filter list small and self-documenting.
"""
from __future__ import annotations

import logging
from typing import Final

# Substrings that, if present in ``record.getMessage()``, mean we drop
# the record from the JSONL handler. Match is substring-only — keep
# patterns specific enough to avoid false positives.
_NOISE_SUBSTRINGS: Final[tuple[str, ...]] = (
    # ── discord.py optional-deps WARNs ──────────────────────────────
    # Both fire once at startup when running in a container that
    # doesn't ship voice libraries. We don't use voice; these are
    # informational. (Patterns e06f8b8f and a037d19a, ~1×/restart but
    # multi-restart noise can accumulate.)
    "voice will NOT be supported",          # PyNaCl + davey variants
    # ── Anthropic SDK credit-balance WARN already handled ─────────
    # Pattern eb829b26: 143 occurrences in 7 days. CreditAware-
    # AnthropicCompletion (app/llms/credit_aware_anthropic.py) catches
    # these 400s and fails over to OpenRouter; the circuit_breaker
    # OPEN transition is the operator-visible signal. The raw SDK
    # WARN is duplicate noise.
    "Anthropic API call failed: Error code: 400",
    # ── OpenRouter Stealth 502 already handled by env-var blocker ─
    # Pattern 30bbb7cd: 630 occurrences in 7 days. The
    # OPENROUTER_IGNORE_PROVIDERS=Stealth route in app/llm_factory.py
    # excludes Stealth at the request layer; legacy in-flight calls
    # that pre-date that filter still surface here. Drop them.
    "OpenAI API call failed: Error code: 502",
)


class JsonlNoiseFilter(logging.Filter):
    """Drops WARN records whose message matches a known-noise pattern.

    Attach to the JSONL ``RotatingFileHandler``, NOT to the root
    logger — we want stdout / debug consoles to keep showing the
    messages.
    """

    def __init__(
        self,
        substrings: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(name="jsonl_noise_filter")
        self._substrings: tuple[str, ...] = substrings or _NOISE_SUBSTRINGS

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True to keep the record, False to drop it."""
        try:
            msg = record.getMessage()
        except Exception:
            # If formatting fails, default to keeping the record so
            # we don't silently drop real errors.
            return True
        for needle in self._substrings:
            if needle in msg:
                return False
        return True


def get_noise_substrings() -> tuple[str, ...]:
    """Read-only accessor for tests and operator-side audits."""
    return _NOISE_SUBSTRINGS
