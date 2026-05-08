"""
paths.py — Central registry of filesystem locations.

Single source of truth for workspace paths. Replaces ~20 hardcoded
"/app/workspace/..." literals scattered across the codebase.

WORKSPACE_ROOT is read from the environment with a default of
"/app/workspace" so the same code runs in the container and in
dev environments.

Usage:
    from app.paths import WORKSPACE_ROOT, ERROR_JOURNAL, ensure_dirs
    ensure_dirs()
    ERROR_JOURNAL.write_text(...)
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT: Path = Path(
    os.environ.get("WORKSPACE_ROOT", "/app/workspace")
).resolve()

# ── Journals & metric files (top-level workspace) ────────────────────────
ERROR_JOURNAL = WORKSPACE_ROOT / "error_journal.json"
ERROR_TRACKER = WORKSPACE_ROOT / "error_tracker.json"
AUDIT_JOURNAL = WORKSPACE_ROOT / "audit_journal.json"
AGENT_STATE = WORKSPACE_ROOT / "agent_state.json"
VARIANT_ARCHIVE = WORKSPACE_ROOT / "variant_archive.json"
BENCHMARKS = WORKSPACE_ROOT / "benchmarks.json"
SELF_KNOWLEDGE_HASHES = WORKSPACE_ROOT / ".self_knowledge_hashes.json"

# ── Directories ──────────────────────────────────────────────────────────
LOGS_DIR = WORKSPACE_ROOT / "logs"
CREWAI_STORAGE = WORKSPACE_ROOT / "crewai_storage"
ATLAS_DIR = WORKSPACE_ROOT / "atlas"
EVOLUTION_ARCHIVE_DIR = WORKSPACE_ROOT / "evolution_archive"
APPLIED_CODE_DIR = WORKSPACE_ROOT / "applied_code"
KNOWLEDGE_DIR = WORKSPACE_ROOT / "knowledge"
PHILOSOPHY_DIR = WORKSPACE_ROOT / "philosophy"
FICTION_LIBRARY_DIR = WORKSPACE_ROOT / "fiction_library"
LITERATURE_LIBRARY_DIR = WORKSPACE_ROOT / "literature_library"
PROMPTS_DIR = WORKSPACE_ROOT / "prompts"
MEMORY_DIR = WORKSPACE_ROOT / "memory"
SELF_AWARENESS_DATA = WORKSPACE_ROOT / "self_awareness_data"

# ── New knowledge bases (Phase 2) ──────────────────────────────────────────
EPISTEME_DIR = WORKSPACE_ROOT / "episteme"
EPISTEME_TEXTS_DIR = EPISTEME_DIR / "texts"
EXPERIENTIAL_DIR = WORKSPACE_ROOT / "experiential"
EXPERIENTIAL_ENTRIES_DIR = EXPERIENTIAL_DIR / "entries"
AESTHETICS_DIR = WORKSPACE_ROOT / "aesthetics"
AESTHETICS_PATTERNS_DIR = AESTHETICS_DIR / "patterns"
TENSIONS_DIR = WORKSPACE_ROOT / "tensions"
TENSIONS_ENTRIES_DIR = TENSIONS_DIR / "entries"

# ── SubIA paths (created during Phase 1; safe to reference now) ──────────
SUBIA_ROOT = WORKSPACE_ROOT / "subia"
SUBIA_SELF_DIR = SUBIA_ROOT / "self"
SUBIA_WORKSPACE_DIR = SUBIA_ROOT / "workspace"
KERNEL_STATE = SUBIA_SELF_DIR / "kernel-state.md"
HOMEOSTATIC_PROFILE = SUBIA_SELF_DIR / "homeostatic-profile.md"
SOCIAL_MODELS = SUBIA_SELF_DIR / "social-models.md"
PREDICTION_ACCURACY = SUBIA_SELF_DIR / "prediction-accuracy.md"
NARRATIVE_AUDIT = SUBIA_SELF_DIR / "self-narrative-audit.md"
CONSCIOUSNESS_STATE = SUBIA_SELF_DIR / "consciousness-state.md"
HOT_MD = SUBIA_WORKSPACE_DIR / "hot.md"
SUBIA_INTEGRITY_MANIFEST = WORKSPACE_ROOT / ".subia_integrity.json"

# ── Affect Layer paths ──────────────────────────────────────────────────
# Mirror the SubIA pattern: every persistence path is registered here so
# WORKSPACE_ROOT overrides (e.g. dev/test redirection to a tempdir) reach
# the affect layer too. Modules under app/affect/ MUST import these
# constants rather than hardcode `/app/workspace/affect/...`.
AFFECT_ROOT                 = WORKSPACE_ROOT / "affect"
AFFECT_TRACE                = AFFECT_ROOT / "trace.jsonl"
AFFECT_AUDIT                = AFFECT_ROOT / "welfare_audit.jsonl"
AFFECT_SETPOINTS            = AFFECT_ROOT / "setpoints.json"
AFFECT_CALIBRATION          = AFFECT_ROOT / "calibration.json"
AFFECT_REFLECTIONS_DIR      = AFFECT_ROOT / "reflections"
AFFECT_ATTACHMENTS_DIR      = AFFECT_ROOT / "attachments"
AFFECT_PEERS_DIR            = AFFECT_ATTACHMENTS_DIR / "peers"
AFFECT_CARE_LEDGER          = AFFECT_ATTACHMENTS_DIR / "care_ledger.jsonl"
AFFECT_CHECK_INS            = AFFECT_ATTACHMENTS_DIR / "check_in_candidates.jsonl"
AFFECT_PHASE5_GATE          = AFFECT_ROOT / "phase5_gate.jsonl"
AFFECT_PHASE5_PROPOSALS     = AFFECT_ROOT / "phase5_proposals.jsonl"
AFFECT_L9_SNAPSHOTS         = AFFECT_ROOT / "l9_snapshots.jsonl"
AFFECT_SALIENCE             = AFFECT_ROOT / "salience.jsonl"
AFFECT_EPISODES_DIR         = AFFECT_ROOT / "episodes"
AFFECT_LAST_FLUSH           = AFFECT_ROOT / "episodes_last_flush.txt"
AFFECT_CHAPTERS_DIR         = AFFECT_ROOT / "chapters"
AFFECT_HEALTH_CHECKS_DIR    = AFFECT_ROOT / "health_checks"
AFFECT_IDENTITY_CLAIMS      = AFFECT_ROOT / "identity_claims.json"
AFFECT_KB_TAGS              = AFFECT_ROOT / "episode_affect_tags.jsonl"

# ── Dreams / wiki-index reconciler (consciousness-roadmap §4) ───────────
# Lives next to AFFECT_ROOT so consciousness-adjacent shadow artifacts are
# co-located. Adoption flows through the change-request gate; this dir is
# observability + candidate-file scratch space, never the live source of
# truth for `wiki/index.md`.
DREAMS_ROOT                  = WORKSPACE_ROOT / "dreams"
WIKI_INDEX_CANDIDATE         = DREAMS_ROOT / "wiki_index.candidate.md"
WIKI_INDEX_AUDIT             = DREAMS_ROOT / "wiki_index_audit.jsonl"

# ── System-wide files ────────────────────────────────────────────────────
SYSTEM_CHRONICLE = WORKSPACE_ROOT / "system_chronicle.md"
WORKSPACE_LOCK = WORKSPACE_ROOT / ".workspace.lock"
INTEGRITY_CHECKSUMS = WORKSPACE_ROOT / ".integrity_checksums.json"

# Directories that should exist before the app starts writing.
_MANAGED_DIRS = (
    LOGS_DIR,
    CREWAI_STORAGE,
    ATLAS_DIR,
    EVOLUTION_ARCHIVE_DIR,
    APPLIED_CODE_DIR,
    KNOWLEDGE_DIR,
    PHILOSOPHY_DIR,
    FICTION_LIBRARY_DIR,
    PROMPTS_DIR,
    MEMORY_DIR,
    SELF_AWARENESS_DATA,
    SUBIA_ROOT,
    SUBIA_SELF_DIR,
    SUBIA_WORKSPACE_DIR,
    AFFECT_ROOT,
    AFFECT_REFLECTIONS_DIR,
    AFFECT_ATTACHMENTS_DIR,
    AFFECT_PEERS_DIR,
    AFFECT_EPISODES_DIR,
    AFFECT_CHAPTERS_DIR,
    AFFECT_HEALTH_CHECKS_DIR,
    EPISTEME_DIR,
    EPISTEME_TEXTS_DIR,
    EXPERIENTIAL_DIR,
    EXPERIENTIAL_ENTRIES_DIR,
    AESTHETICS_DIR,
    AESTHETICS_PATTERNS_DIR,
    TENSIONS_DIR,
    TENSIONS_ENTRIES_DIR,
    LITERATURE_LIBRARY_DIR,
    DREAMS_ROOT,
)


def ensure_dirs() -> None:
    """Create all managed workspace directories if missing. Idempotent."""
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    for d in _MANAGED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def under_workspace(path: Path | str) -> bool:
    """Return True if the given path is inside WORKSPACE_ROOT.

    Use this to validate paths supplied by untrusted code before
    writing — prevents escapes via '..' or absolute paths outside
    the workspace.
    """
    try:
        Path(path).resolve().relative_to(WORKSPACE_ROOT)
        return True
    except (ValueError, OSError):
        return False
