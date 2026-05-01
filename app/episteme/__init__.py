"""
episteme — Metacognitive / Research Knowledge Base.

Contains research papers, architecture decisions, design patterns, and
failed experiments.  Grounds the Self-Improver in theory so it makes
principled improvements, not just hill-climbing.

Epistemic status: theoretical/empirical — more trustworthy than fiction,
but still claims that need validation against the specific system.

──────────────────────────────────────────────────────────────────────
Naming note (Phase A5 disambiguation):

    app.episteme   ← THIS PACKAGE: RAG retrieval over the research KB.
                     "What does the literature say about X?"
                     Vector store + search tools used by agents.

    app.epistemic  ← DIFFERENT PACKAGE: claim ledger / calibration /
                     pushback / overrides — runtime tracking of WHAT
                     this system claims and WHETHER those claims hold up.

Both names are correct in their own context; they intentionally do
not overlap. If you import the wrong one you will get the wrong thing.
──────────────────────────────────────────────────────────────────────

IMMUTABLE — infrastructure-level module.
"""

from app.episteme.vectorstore import EpistemeStore, get_store
from app.episteme.tools import EpistemeSearchTool, get_episteme_tools

__all__ = [
    "EpistemeStore",
    "get_store",
    "EpistemeSearchTool",
    "get_episteme_tools",
]
