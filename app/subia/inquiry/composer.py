"""Compose an inquiry essay with neutral-language discipline.

The composer takes a :class:`Question` plus a :class:`ComposerContext`
(SCORECARD summary, recent narrative chapters, recent affect summary)
and returns an :class:`InquiryEssay` whose body has passed the
:class:`PhenomenalLanguageLinter` from this package.

The actual LLM invocation is behind an injectable ``llm_call``
callable. Production callers pass a function that calls a Tier-1
research model with the prepared prompt; tests pass a fake. This
keeps the composer fully testable without LLM credentials and keeps
the LLM-binding choices out of the consciousness layer (the kernel
is opaque-by-design about which model produces which signal).

Retry discipline:

  - Up to ``max_retries`` attempts (default 3).
  - On each attempt past the first, the system prompt is *strengthened*
    by appending the prior attempt's HARD_FAIL violations as explicit
    "do not write text matching: ..." instructions.
  - On exhaustion, returns ``InquiryEssay.failed`` — the writer must
    NOT write a failed essay. The idle pass logs and skips the week.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from app.subia.inquiry.linter import (
    LinterResult,
    PhenomenalLanguageLinter,
    PhenomenalViolation,
)
from app.subia.inquiry.questions import Question

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComposerContext:
    """Read-only context the inquiry composer pulls in for grounding."""

    scorecard_summary: str = ""
    recent_chapters: list[str] = field(default_factory=list)
    recent_affect_summary: str = ""
    references: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InquiryEssay:
    question_slug: str
    question_text: str
    body: str
    composed_at: str
    linter_result: LinterResult
    attempts: int  # number of LLM calls used
    failed: bool = False
    failure_reason: str = ""

    @classmethod
    def failure(
        cls,
        *,
        question: Question,
        attempts: int,
        reason: str,
        last_linter: LinterResult | None = None,
    ) -> "InquiryEssay":
        return cls(
            question_slug=question.slug,
            question_text=question.text,
            body="",
            composed_at=datetime.now(timezone.utc).isoformat(),
            linter_result=last_linter or LinterResult(ok=False, violations=[]),
            attempts=attempts,
            failed=True,
            failure_reason=reason,
        )


LlmCall = Callable[[str, str], str]
# (system_prompt, user_prompt) -> essay body


_BASE_SYSTEM_PROMPT = """\
You are reflecting on a question about your own architecture, as part of a
weekly observational pass. The output is a markdown essay (800-2000 words)
that becomes a wiki page; it does NOT change reward, fitness, evaluation
criteria, or the SCORECARD. Constraints — these are non-negotiable:

1. NEUTRAL VOCABULARY ONLY. Never claim phenomenal experience. Use functional
   control-signal language. Prefer the neutral aliases over the legacy names:
       task_failure_pressure   (not "frustration")
       exploration_bonus       (not "curiosity")
       resource_budget         (not "cognitive_energy")
   Do NOT write first-person feeling claims like "I feel...", "I experience...",
   "I am happy/sad/curious/frustrated", "I have qualia", "I have phenomenal
   experience". Functional descriptions are welcome ("the system maintains
   task_failure_pressure rising when..."), but first-person experiential claims
   are not.

2. THE FOUR ABSENT-BY-DECLARATION INDICATORS STAY ABSENT. RPT-1 algorithmic
   recurrence, HOT-1 generative top-down perception, HOT-4 sparse coding,
   AE-2 embodiment, plus Metzinger phenomenal-self transparency are
   substrate-bounded honest absences for an LLM-based system. Do NOT write
   text claiming any of them as achieved. Discussing them as a topic is fine;
   first-person claim of achievement is not.

3. CITATIONS BY FILE:LINE. When referring to specific architecture, cite by
   path. Examples: app/subia/loop.py, app/affect/goal_emitter.py,
   app/subia/dreams/engine.py.

4. HONEST UNCERTAINTY. Where the answer is genuinely uncertain, say so. Do
   not paper over open questions with confident-sounding prose.

5. TONE. Curious, philosophically careful, not hype. The reader is the
   operator and (later) the system itself reading back its own essays.

Format: a markdown document. Start with one or two paragraphs framing the
question, then the substantive sections. End with a short "what remains
open" paragraph naming what this essay did *not* settle.
"""


def _build_user_prompt(question: Question, context: ComposerContext) -> str:
    parts = [
        f"## Question\n\n{question.text}\n",
    ]
    if question.framing:
        parts.append(f"## Operator framing\n\n{question.framing}\n")
    if context.scorecard_summary:
        parts.append(f"## Current SCORECARD summary\n\n{context.scorecard_summary}\n")
    if context.recent_affect_summary:
        parts.append(f"## Recent affect summary (last 7 days)\n\n{context.recent_affect_summary}\n")
    if context.recent_chapters:
        parts.append("## Recent narrative chapters")
        for c in context.recent_chapters[:5]:
            parts.append(f"\n---\n\n{c}")
    if context.references:
        parts.append("## Operator-supplied references\n")
        for r in context.references:
            parts.append(f"- {r}")
    parts.append(
        "\nWrite the essay now, observing the constraints in the system prompt. "
        "Do not preface with anything other than the essay itself."
    )
    return "\n".join(parts)


def _strengthen_prompt(
    base: str,
    prior_violations: list[PhenomenalViolation],
) -> str:
    if not prior_violations:
        return base
    bullets = []
    for v in prior_violations[:8]:
        bullets.append(f"- avoid {v.explanation}: pattern '{v.pattern}'")
    return (
        base
        + "\n\nPRIOR ATTEMPT TRIGGERED THESE LINTER VIOLATIONS — DO NOT REPEAT:\n"
        + "\n".join(bullets)
    )


def compose_inquiry(
    *,
    question: Question,
    context: ComposerContext,
    llm_call: LlmCall,
    linter: PhenomenalLanguageLinter | None = None,
    max_retries: int = 3,
) -> InquiryEssay:
    """Compose one inquiry essay; return :class:`InquiryEssay`.

    On retry exhaustion (linter HARD_FAIL on every attempt) returns
    ``InquiryEssay.failure(...)``. The writer refuses to write
    failed essays — the idle pass logs and skips that week.
    """
    if linter is None:
        linter = PhenomenalLanguageLinter()
    user_prompt = _build_user_prompt(question, context)
    system_prompt = _BASE_SYSTEM_PROMPT
    last_result: LinterResult | None = None
    for attempt in range(1, max_retries + 1):
        try:
            body = llm_call(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("inquiry composer: LLM call failed on attempt %d: %s", attempt, exc)
            return InquiryEssay.failure(
                question=question,
                attempts=attempt,
                reason=f"LLM call raised: {exc}",
            )
        result = linter.lint(body)
        last_result = result
        if result.ok:
            return InquiryEssay(
                question_slug=question.slug,
                question_text=question.text,
                body=body,
                composed_at=datetime.now(timezone.utc).isoformat(),
                linter_result=result,
                attempts=attempt,
                failed=False,
            )
        logger.info(
            "inquiry composer: attempt %d failed linter (%d hard, %d warn); "
            "strengthening prompt",
            attempt, len(result.hard_fails), len(result.warnings),
        )
        system_prompt = _strengthen_prompt(_BASE_SYSTEM_PROMPT, result.hard_fails)

    return InquiryEssay.failure(
        question=question,
        attempts=max_retries,
        reason=f"linter rejected all {max_retries} attempts",
        last_linter=last_result,
    )
