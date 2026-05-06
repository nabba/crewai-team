"""dossier_crew — Investment-grade company dossier crew.

Wraps :func:`app.dossier.pipeline.build_dossier` in the standard
``SomeCrew().run(task, parent_task_id, difficulty)`` shape that
:mod:`app.crews.registry` expects.

Why a dedicated crew rather than reusing the financial crew
============================================================
The financial crew is an open-ended LLM-driven analyst — it fits
"compute the DCF for AAPL" or "explain Spotify's gross margin trend."
The dossier crew is the *opposite*: a deterministic data pipeline
with the LLM scoped to prose composition only.  The two coexist:
  * Use ``financial`` for ad-hoc analyst chat and bespoke modelling.
  * Use ``company_dossier`` for structured one-shot reports.

The commander dispatches between them via the standard capability-
routing layer (this module just provides the runner).

Streaming
=========
Long-running steps push progress over Signal via the existing
``signal_client`` (same path the research orchestrator uses).  The
user sees "Collecting dossier… Selecting peers… Composing… Rendering
PDF… Done." as the build proceeds.
"""
from __future__ import annotations

import logging

from app.dossier.pipeline import build_dossier
from app.observability.task_progress import (
    current_task_id,
    record_output_progress,
)

logger = logging.getLogger(__name__)


def _stream_progress(message: str) -> None:
    """Push a one-line progress note to the user's Signal thread.

    Mirrors the partial-stream pattern in
    :mod:`app.tools.research_orchestrator` so the stall detector sees
    activity at every major pipeline step.  Fail-soft — a Signal
    hiccup must not abort the build.
    """
    tid = current_task_id.get()
    if not tid:
        return
    try:
        record_output_progress(tid, note=message)
    except Exception:
        logger.debug("dossier_crew: task_progress record failed", exc_info=True)
    try:
        from app import signal_client as _sc_module
        sc = getattr(_sc_module, "signal_client", None)
        if sc is not None:
            sc._send_sync(tid, f"[dossier] {message}")
    except Exception:
        logger.debug("dossier_crew: progress send failed (non-fatal)",
                     exc_info=True)


class DossierCrew:
    """Standard ``Crew().run(...)`` shape for the registry."""

    def run(
        self,
        task_description: str,
        parent_task_id: str | None = None,
        difficulty: int = 5,
    ) -> str:
        """Build a dossier for the company in ``task_description``.

        Returns a Signal-friendly summary string.  The PDF lives in
        ``/app/workspace/output/`` (production) or
        ``$DOSSIER_OUTPUT_DIR`` (dev override) — agents call
        ``signal_send_attachment`` on the path to deliver it.
        """
        try:
            build = build_dossier(
                query=task_description,
                task_id=parent_task_id,
                progress_callback=_stream_progress,
            )
        except ValueError as exc:
            return (
                f"Could not build dossier: {exc}\n\n"
                "Provide the company's name (and ideally its stock "
                "ticker) so the dossier can be built — e.g. "
                "\"Build a dossier for Spotify (SPOT)\"."
            )
        except Exception as exc:
            logger.exception("dossier_crew: build_dossier crashed")
            return (
                f"Dossier build failed: {type(exc).__name__}: {exc}\n\n"
                "The collection pipeline encountered an error.  Run "
                "the request again, or check that the relevant "
                "adapters' API keys are set."
            )

        summary = build.summary()
        # The crew's job is the build; downstream tools (specifically
        # signal_send_attachment) handle PDF delivery.  We surface the
        # path explicitly so an agent or operator can pick it up.
        return summary
