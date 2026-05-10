"""Weekly philosophical inquiry pass — operator-curated questions, neutral-language essays.

Consciousness-roadmap addition (post-§3 close): the system has phronesis,
shadow-bias mining, and reverie concept-walks; it does *not* yet have a
deliberate place for explicit philosophical inquiry into questions like
"what is the relationship between my goals and Andrus's?" This package
provides that surface as a weekly idle pass.

Subsystem boundary (read-only by design):

  Reads   : ``wiki/self/inquiry_questions.md`` (operator-curated list),
            recent ``wiki/self/inquiries/*.md`` (selector dedup),
            SCORECARD summary, recent narrative chapters, recent
            affect summary.
  Writes  : ``wiki/self/inquiries/<date>-<slug>.md`` ONLY.
            The :class:`InquiryWriter` is path-confined; any attempt
            to write elsewhere raises.
  Forbids : Modifying ``current_goals`` (G1's ``goal_emitter`` is the
            sole writer; AE-1 STRONG anchor),
            modifying SCORECARD probes (TIER_IMMUTABLE),
            modifying any TIER_IMMUTABLE file,
            modifying the curated question list autonomously
            (additions go through change_requests).

Phase 11 discipline: every essay passes through :class:`PhenomenalLanguageLinter`
before being written. The composer prompt explicitly forbids phenomenal-claim
vocabulary; the linter is the mechanical safety net. On linter rejection
the composer retries up to ``max_retries`` with a stricter prompt; if all
retries fail, the idle pass logs and skips that week — *never* writes a
contaminated essay.

Failure isolation: idle-job entry point (:func:`idle_registration.run_inquiry_pass`)
catches any exception and returns a structured result. The pass NEVER raises
into the idle scheduler.

Public surface:
"""

from app.subia.inquiry.composer import (
    ComposerContext,
    InquiryEssay,
    compose_inquiry,
)
from app.subia.inquiry.linter import (
    PhenomenalLanguageLinter,
    LinterResult,
    PhenomenalViolation,
)
from app.subia.inquiry.questions import Question, load_questions
from app.subia.inquiry.selector import select_next_question
from app.subia.inquiry.writer import InquiryWriter, WriteRefused

__all__ = [
    "ComposerContext",
    "InquiryEssay",
    "InquiryWriter",
    "LinterResult",
    "PhenomenalLanguageLinter",
    "PhenomenalViolation",
    "Question",
    "WriteRefused",
    "compose_inquiry",
    "load_questions",
    "select_next_question",
]
