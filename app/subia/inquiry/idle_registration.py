"""Idle-job entry point for the weekly inquiry pass.

Failure-isolated: any exception during the pass is caught and
returned as a structured :class:`PassResult`. The idle scheduler
sees a successful return regardless. This keeps a malfunctioning
inquiry pass from breaking the rest of the system.

Master switch: ``INQUIRY_PASS_ENABLED`` (default ``true`` — the
pass is read-only-additive; the failure mode is "no inquiry written
this week"). Set to ``false`` to disable entirely.

Cadence: once per Saturday in production (the idle scheduler
provides cron-like cadence; this module just exposes ``run_once``
as a callable).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

from app.subia.inquiry.composer import (
    ComposerContext,
    InquiryEssay,
    LlmCall,
    compose_inquiry,
)
from app.subia.inquiry.questions import Question, load_questions
from app.subia.inquiry.selector import select_next_question
from app.subia.inquiry.writer import InquiryWriter, WriteRefused

logger = logging.getLogger(__name__)


def is_enabled() -> bool:
    return os.getenv("INQUIRY_PASS_ENABLED", "true").lower() in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class PassResult:
    """Structured outcome of one inquiry pass run."""

    status: str  # "wrote_essay" | "skipped_disabled" | "skipped_no_questions" |
                 # "skipped_composer_failed" | "skipped_writer_refused" |
                 # "skipped_unexpected_error"
    question_slug: str = ""
    written_to: str = ""
    failure_reason: str = ""


ContextProvider = Callable[[], ComposerContext]


def run_once(
    *,
    llm_call: LlmCall,
    context_provider: ContextProvider | None = None,
    questions_path: str | None = None,
    inquiries_dir: str | None = None,
) -> PassResult:
    """Run one inquiry pass; never raises."""
    if not is_enabled():
        logger.info("inquiry pass: disabled via INQUIRY_PASS_ENABLED")
        return PassResult(status="skipped_disabled")

    try:
        questions = load_questions(questions_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("inquiry pass: load_questions failed: %s", exc)
        return PassResult(
            status="skipped_unexpected_error",
            failure_reason=f"load_questions: {exc}",
        )

    if not questions:
        return PassResult(status="skipped_no_questions")

    try:
        question = select_next_question(questions, inquiries_dir=inquiries_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("inquiry pass: select_next_question failed: %s", exc)
        return PassResult(
            status="skipped_unexpected_error",
            failure_reason=f"select_next_question: {exc}",
        )
    if question is None:
        return PassResult(status="skipped_no_questions")

    context = context_provider() if context_provider else ComposerContext()

    try:
        essay = compose_inquiry(
            question=question,
            context=context,
            llm_call=llm_call,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("inquiry pass: compose_inquiry raised: %s", exc)
        return PassResult(
            status="skipped_unexpected_error",
            question_slug=question.slug,
            failure_reason=f"compose_inquiry: {exc}",
        )

    if essay.failed:
        logger.info(
            "inquiry pass: composer failed for %s (%s)",
            question.slug, essay.failure_reason,
        )
        return PassResult(
            status="skipped_composer_failed",
            question_slug=question.slug,
            failure_reason=essay.failure_reason,
        )

    try:
        writer = InquiryWriter(inquiries_dir=inquiries_dir)
        path = writer.write(essay)
    except WriteRefused as exc:
        logger.warning("inquiry pass: writer refused: %s", exc)
        return PassResult(
            status="skipped_writer_refused",
            question_slug=question.slug,
            failure_reason=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("inquiry pass: writer raised: %s", exc)
        return PassResult(
            status="skipped_unexpected_error",
            question_slug=question.slug,
            failure_reason=f"writer: {exc}",
        )

    return PassResult(
        status="wrote_essay",
        question_slug=question.slug,
        written_to=str(path),
    )
