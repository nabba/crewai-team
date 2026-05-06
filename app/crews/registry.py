"""
registry.py — Central registry of named crews the commander can dispatch to.

Motivation
----------
The commander's ``_run_crew`` used to contain an 11-branch
``if crew_name == "research": ... elif crew_name == "coding": ...``
chain with nearly identical bodies:

    if crew_name == "research":
        from app.crews.research_crew import ResearchCrew
        result = ResearchCrew().run(enriched_task, parent_task_id=..., difficulty=...)
    elif crew_name == "coding":
        from app.crews.coding_crew import CodingCrew
        result = CodingCrew().run(enriched_task, parent_task_id=..., difficulty=...)
    ...

Two problems: adding a crew means editing the orchestrator (Open-Closed
violation), and the creative crew's special shape (``run_creative_crew``
function + post-processing of ``aborted_reason``) was a bespoke elif
branch instead of a registered adapter.

The registry replaces the chain with a lazy-import registry of named
*runners* — zero-arg-safe callables that accept the standard
``(task, parent_task_id, difficulty)`` tuple and return a ``str``
result.  The orchestrator becomes a one-line ``registry.dispatch(...)``.

Usage
-----
At startup (see ``install_defaults``)::

    from app.crews import registry
    registry.register("research", registry.class_run_runner(
        "app.crews.research_crew", "ResearchCrew"))
    registry.register("creative", _creative_adapter)   # special shape

In ``orchestrator._run_crew``::

    result = registry.dispatch(
        crew_name, enriched_task,
        parent_task_id=parent_task_id, difficulty=difficulty,
    )
    if result is None:            # unknown crew_name
        return crew_task           # fallback to passing the task through
"""
from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# A *runner* is a callable with the standard crew-invocation signature
# that returns the final string result.  Every registered crew —
# regardless of whether the underlying implementation is a class's
# ``.run()`` method or a free-standing function — is exposed through
# this uniform shape.
CrewRunner = Callable[[str, Optional[str], int], str]


_registry: dict[str, CrewRunner] = {}


def register(name: str, runner: CrewRunner) -> None:
    """Register ``runner`` under ``name`` (replacing any prior entry).

    Idempotent: re-registering the same name is safe and expected
    during hot reloads.
    """
    was_replace = name in _registry
    _registry[name] = runner
    if was_replace:
        logger.debug("crews.registry: replaced '%s'", name)
    else:
        logger.info(
            "crews.registry: registered '%s' (total=%d)",
            name, len(_registry),
        )


def dispatch(
    name: str,
    task: str,
    *,
    parent_task_id: Optional[str] = None,
    difficulty: int = 5,
) -> Optional[str]:
    """Run the crew registered under ``name``.

    Returns the crew's output string, or ``None`` if ``name`` has no
    registered runner — the orchestrator uses ``None`` as the signal
    to fall back to passing the task through unchanged (the old
    ``else: return crew_task`` branch).
    """
    runner = _registry.get(name)
    if runner is None:
        return None
    return runner(task, parent_task_id, difficulty)


def registered_names() -> list[str]:
    """Observability helper: names of every registered crew."""
    return sorted(_registry.keys())


# ── Runner factories ─────────────────────────────────────────────────
#
# Two shapes cover every crew we have today:
#
#   1. ``class_run_runner(module, cls)`` — the common case.  Lazy-
#      imports the module on first call, instantiates ``cls()``, calls
#      ``.run(task, parent_task_id=pid, difficulty=d)``.  Every *_crew
#      module that exposes a ``SomeCrew`` class uses this.
#
#   2. Custom adapters for crews whose entry points don't match that
#      signature (e.g. creative_crew's module-level function that
#      returns an object with ``.final_output`` + ``.aborted_reason``).
#      These are written by hand and registered explicitly.


def class_run_runner(module_path: str, class_name: str) -> CrewRunner:
    """Build a runner that does ``module.class_name().run(...)`` on first
    invocation, importing the module lazily.

    Lazy import matters because importing 10+ crew modules eagerly at
    gateway startup costs ~1-2s and drags in transitive CrewAI / tool
    machinery that most runs will never touch.
    """
    def _run(task: str, parent_task_id: Optional[str], difficulty: int) -> str:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls().run(
            task, parent_task_id=parent_task_id, difficulty=difficulty,
        )
    _run.__name__ = f"{class_name}.run"
    _run.__qualname__ = f"{module_path}:{class_name}.run"
    return _run


# ── Default wiring ──────────────────────────────────────────────────


def install_defaults() -> None:
    """Register every built-in crew.

    Call once at gateway startup.  Safe to call again — registrations
    are idempotent by name.
    """
    # Standard ``SomeCrew().run()`` class-method crews — the large
    # majority.  Each entry is a single data-driven line.
    for name, module, cls in (
        ("research",      "app.crews.research_crew",       "ResearchCrew"),
        ("coding",        "app.crews.coding_crew",         "CodingCrew"),
        ("writing",       "app.crews.writing_crew",        "WritingCrew"),
        ("media",         "app.crews.media_crew",          "MediaCrew"),
        ("pim",           "app.crews.pim_crew",            "PIMCrew"),
        ("financial",     "app.crews.financial_crew",      "FinancialCrew"),
        ("desktop",       "app.crews.desktop_crew",        "DesktopCrew"),
        ("repo_analysis", "app.crews.repo_analysis_crew",  "RepoAnalysisCrew"),
        ("devops",        "app.crews.devops_crew",         "DevOpsCrew"),
        # Investment-grade company dossier — see app/dossier/ for
        # the pipeline.  Distinct from ``financial`` (open-ended
        # analyst) because it is a deterministic data + structured-
        # composition pipeline that always produces the same multi-
        # page report shape.
        ("company_dossier", "app.crews.dossier_crew",      "DossierCrew"),
    ):
        register(name, class_run_runner(module, cls))

    # Non-standard: creative_crew returns a structured result (with an
    # optional ``aborted_reason``) via a module-level function instead
    # of a class's ``.run()``.  Wrap that bespoke shape into the
    # registry's uniform signature here so the orchestrator doesn't
    # have to special-case it.
    register("creative", _creative_runner)


def _creative_runner(
    task: str,
    parent_task_id: Optional[str],
    difficulty: int,
) -> str:
    """Adapter for ``creative_crew.run_creative_crew``, which has a
    different signature (takes ``creativity=`` kwarg, returns a
    structured result rather than a plain string) than the standard
    ``*Crew().run()`` shape.
    """
    from app.crews.creative_crew import run_creative_crew
    run_result = run_creative_crew(
        task,
        creativity="high",
        parent_task_id=parent_task_id,
    )
    out = run_result.final_output
    if getattr(run_result, "aborted_reason", None):
        out = f"{out}\n\n[Note: {run_result.aborted_reason}]"
    return out
