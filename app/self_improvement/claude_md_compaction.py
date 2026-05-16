"""CLAUDE.md compaction proposer — annual ritual.

PROGRAM §51 — Q16 Theme 5 (knowledge management at decade-scale).
CLAUDE.md grows monotonically as Q-batches ship. Over 5+ years the
file becomes unreadable. The system uses `docs/claude-md-archive.md`
for one-off offload already (per CLAUDE.md itself); this module
makes the offload **cyclic** by composing a yearly compaction
proposal the operator can review.

What this module does:

  * Reads any CLAUDE.md it can locate (project root and/or sub-repos).
  * Identifies "Q-batch entry" bullets by a stable heuristic:
    bullets starting with "- Q<N>" or "- PROGRAM §<N>".
  * Splits entries into KEEP (last ``_KEEP_RECENT_MONTHS`` of dated
    entries) and ARCHIVE (everything older).
  * Composes:
      - Proposed compacted CLAUDE.md (preserves the head section
        + KEEP block + a one-line pointer to the archive file).
      - Companion archive file (``docs/claude-md-archive-<year>.md``)
        with the offloaded entries.
  * Writes BOTH to ``workspace/self_improvement/claude_md_compaction/
    <year>/`` for operator review. The operator applies the diff
    manually (this module deliberately does NOT auto-write to
    CLAUDE.md; the file may live outside any git repo).

What this module **deliberately doesn't** do:

  * Touch CLAUDE.md directly. CLAUDE.md often sits OUTSIDE the
    git repo (e.g. ``/Users/andrus/BotArmy/CLAUDE.md`` is not
    tracked by the ``crewai-team/`` repo). The CR system writes
    into the repo; that's the wrong target.
  * Use LLM-driven rewriting. The compaction is purely structural —
    keep recent, archive older, preserve everything verbatim.
    No risk of paraphrase-drift.
  * File a CR. The proposal lands as plain files; operator picks
    up via Signal alert or directly from workspace/.

Cadence: annual idle job (max one proposal/year per source file).
Failure-isolated: any one source file failing doesn't break others.

Master switch: ``claude_md_compaction_enabled`` (default ON).
"""
from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Default search locations for CLAUDE.md instances. Operator can
# extend via env var ``CLAUDE_MD_PATHS`` (colon-separated absolute
# paths). The probes resolve relative to ``_workspace().parent``
# to find the project root.
_DEFAULT_SOURCES = (
    "CLAUDE.md",                # project root
    "../CLAUDE.md",             # parent of the gateway dir
    "../../CLAUDE.md",          # grand-parent
)

_KEEP_RECENT_MONTHS = 6  # entries from the last 6 months stay in CLAUDE.md
_MIN_BYTES_TO_PROPOSE = 30_000  # ~10 KB threshold; smaller files don't need compaction
_MIN_DAYS_BETWEEN_PROPOSALS = 330  # avoid proposing twice in the same year


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_claude_md_compaction_enabled
        return get_claude_md_compaction_enabled()
    except Exception:
        return os.getenv(
            "CLAUDE_MD_COMPACTION_ENABLED", "true",
        ).lower() in ("true", "1", "yes", "on")


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _output_dir() -> Path:
    return _workspace() / "self_improvement" / "claude_md_compaction"


def _candidate_paths() -> list[Path]:
    """Compose the list of CLAUDE.md paths to consider. Operator-
    overridable via ``CLAUDE_MD_PATHS`` env var (colon-separated
    absolute paths)."""
    override = os.getenv("CLAUDE_MD_PATHS", "").strip()
    if override:
        return [Path(p) for p in override.split(":") if p.strip()]
    repo = _repo_root()
    paths: list[Path] = []
    for relative in _DEFAULT_SOURCES:
        candidate = (repo / relative).resolve()
        if candidate not in paths:
            paths.append(candidate)
    return paths


# ── Q-batch line classification ──────────────────────────────────────────


# Match Q-batch entries: lines starting with "- Q<n>" or
# "- PROGRAM §<n>" (with optional leading spaces, since some bullets
# are nested). Both formats are used by CLAUDE.md authors.
_Q_BATCH_LINE = re.compile(
    r"^\s*-\s+(?:Q\d+|PROGRAM\s+§\d+)",
    re.IGNORECASE,
)

# Optional date inside a Q-batch entry, e.g. "(2026-05-16; PROGRAM…".
_DATE_IN_LINE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _entry_age_days(line: str, now: float) -> Optional[float]:
    """Extract the date from a Q-batch line and compute its age in
    days. Returns None when no date appears."""
    m = _DATE_IN_LINE.search(line)
    if not m:
        return None
    try:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dt = datetime(y, mo, d, tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return None
    age_s = max(0.0, now - dt.timestamp())
    return age_s / 86400


def _split_lines(source_text: str, now: float) -> tuple[list[str], list[str], list[str]]:
    """Walk the CLAUDE.md text line-by-line. Returns three lists:

      * ``head`` — lines BEFORE the first Q-batch entry (preserved verbatim).
      * ``keep`` — Q-batch entries within ``_KEEP_RECENT_MONTHS``.
      * ``archive`` — Q-batch entries older than the cutoff (or undated;
        we conservatively archive undated entries since they're usually
        the oldest part of the doc).
    """
    cutoff_days = _KEEP_RECENT_MONTHS * 30
    head: list[str] = []
    keep: list[str] = []
    archive: list[str] = []
    seen_first_qbatch = False
    current_entry: list[str] = []
    current_is_keep: Optional[bool] = None

    def _flush_entry():
        nonlocal current_entry, current_is_keep
        if not current_entry:
            return
        if current_is_keep:
            keep.extend(current_entry)
        else:
            archive.extend(current_entry)
        current_entry = []
        current_is_keep = None

    for raw_line in source_text.splitlines(keepends=True):
        if _Q_BATCH_LINE.match(raw_line):
            seen_first_qbatch = True
            # Flush prior entry (if any).
            _flush_entry()
            age_days = _entry_age_days(raw_line, now)
            # Decide keep-or-archive based on dated age. Undated lines
            # default to ARCHIVE (they're usually the oldest entries
            # that predate the dated convention).
            current_is_keep = (
                age_days is not None and age_days <= cutoff_days
            )
            current_entry = [raw_line]
        elif seen_first_qbatch:
            # Continuation of the current Q-batch entry (sub-bullets,
            # detail text, etc.) until the next top-level Q-batch
            # bullet OR a non-bullet line. The simplest robust rule:
            # any line that ISN'T a top-level Q-batch line continues
            # the current entry until we see EOF or a new Q-batch.
            current_entry.append(raw_line)
        else:
            head.append(raw_line)
    _flush_entry()
    return head, keep, archive


def _format_archive(
    archive_lines: list[str],
    *,
    year: int,
    source_name: str,
) -> str:
    """Header + verbatim archive block."""
    header = (
        f"# CLAUDE.md archive — entries pre-{year}\n\n"
        f"Offloaded from `{source_name}` by the annual compaction "
        f"ritual on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.\n"
        f"Each entry below appeared in `CLAUDE.md` at the time it was "
        f"written — preserved verbatim, no rewriting.\n\n"
        f"## Q-batch entries\n\n"
    )
    return header + "".join(archive_lines)


def _format_compacted(
    head_lines: list[str],
    keep_lines: list[str],
    *,
    archive_filename: str,
    year: int,
) -> str:
    """Head + KEEP entries + one-line pointer to the archive."""
    pointer = (
        f"\n- (Pre-{year} Q-batch entries archived to "
        f"`docs/{archive_filename}` by the annual compaction ritual; "
        f"see PROGRAM §51 Q16 Theme 5 for the procedure.)\n"
    )
    return "".join(head_lines) + "".join(keep_lines) + pointer


def _already_proposed_this_year(year: int, source_name: str) -> bool:
    """Skip if a proposal for this (year, source) already exists."""
    proposal_dir = _output_dir() / str(year)
    if not proposal_dir.exists():
        return False
    stem = re.sub(r"\W+", "_", source_name).strip("_") or "claude_md"
    return any(proposal_dir.glob(f"{stem}.*"))


def compose_proposal(
    *,
    source_path: Path,
    now: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Compose a compaction proposal for one CLAUDE.md source. Returns
    a result dict on success; None on skip (file too small, missing,
    or already proposed this year). Never raises.
    """
    cur = float(now) if now is not None else time.time()
    if not source_path.exists() or not source_path.is_file():
        return None
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return None
    if len(source_bytes) < _MIN_BYTES_TO_PROPOSE:
        return None
    try:
        source_text = source_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None
    year = datetime.fromtimestamp(cur, tz=timezone.utc).year

    if _already_proposed_this_year(year, source_path.name):
        return None

    head, keep, archive = _split_lines(source_text, cur)
    if not archive:
        # Nothing old enough to archive — file is naturally young.
        return None

    archive_filename = f"claude-md-archive-{year - 1}.md"
    archive_body = _format_archive(
        archive, year=year, source_name=source_path.name,
    )
    compacted_body = _format_compacted(
        head, keep, archive_filename=archive_filename, year=year,
    )

    # Persist under workspace/self_improvement/claude_md_compaction/<year>/.
    out_dir = _output_dir() / str(year)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    stem = re.sub(r"\W+", "_", source_path.name).strip("_") or "claude_md"
    compacted_out = out_dir / f"{stem}.compacted.md"
    archive_out = out_dir / f"{stem}.archive.md"
    notes_out = out_dir / f"{stem}.notes.md"
    try:
        compacted_out.write_text(compacted_body, encoding="utf-8")
        archive_out.write_text(archive_body, encoding="utf-8")
        notes_out.write_text(
            f"# CLAUDE.md compaction proposal — {year}\n\n"
            f"Source: `{source_path}`\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
            f"Original size: {len(source_bytes)} bytes\n"
            f"Compacted size: {len(compacted_body)} bytes\n"
            f"Entries KEPT: {sum(1 for l in keep if _Q_BATCH_LINE.match(l))}\n"
            f"Entries ARCHIVED: {sum(1 for l in archive if _Q_BATCH_LINE.match(l))}\n\n"
            f"## Operator next steps\n\n"
            f"1. Review `{compacted_out.name}` — this is the proposed "
            f"replacement for `{source_path}`.\n"
            f"2. Review `{archive_out.name}` — this is the offloaded "
            f"history; place it at `docs/{archive_filename}` in your "
            f"repo before swapping.\n"
            f"3. If both look right: `cp {archive_out} <repo>/docs/"
            f"{archive_filename} && cp {compacted_out} {source_path}`.\n"
            f"4. If anything looks wrong: delete this directory and "
            f"the next annual pass will retry.\n\n"
            f"## Composition contract\n\n"
            f"* Head section (everything before the first Q-batch "
            f"bullet) is preserved verbatim.\n"
            f"* Each Q-batch entry's date is parsed from the ISO "
            f"date in the bullet text (e.g. `2026-05-16`).\n"
            f"* Entries with age ≤ {_KEEP_RECENT_MONTHS} months: KEPT.\n"
            f"* Entries older OR undated: ARCHIVED.\n"
            f"* No paraphrasing — the proposal is a pure structural "
            f"split.\n",
            encoding="utf-8",
        )
    except OSError:
        return None
    return {
        "source": str(source_path),
        "year": year,
        "compacted_path": str(compacted_out),
        "archive_path": str(archive_out),
        "notes_path": str(notes_out),
        "n_kept": sum(1 for l in keep if _Q_BATCH_LINE.match(l)),
        "n_archived": sum(1 for l in archive if _Q_BATCH_LINE.match(l)),
        "original_bytes": len(source_bytes),
        "compacted_bytes": len(compacted_body),
    }


def run_once(*, now: Optional[float] = None) -> dict[str, Any]:
    """Run one pass across all candidate CLAUDE.md paths. Idle-job
    entry point. Returns a summary dict."""
    summary: dict[str, Any] = {
        "ran": False,
        "candidates": [],
        "proposals": [],
    }
    if not _enabled():
        summary["skipped"] = True
        return summary
    summary["ran"] = True
    for path in _candidate_paths():
        summary["candidates"].append(str(path))
        try:
            proposal = compose_proposal(source_path=path, now=now)
        except Exception:
            logger.debug(
                "claude_md_compaction: compose raised", exc_info=True,
            )
            proposal = None
        if proposal is not None:
            summary["proposals"].append(proposal)
            try:
                from app.notify import notify
                notify(
                    title="📜 CLAUDE.md compaction proposal ready",
                    body=(
                        f"Annual compaction proposed for `{path}`:\n"
                        f"  • kept: {proposal['n_kept']} recent Q-batch entries\n"
                        f"  • archived: {proposal['n_archived']} older entries\n"
                        f"  • original: {proposal['original_bytes']} bytes\n"
                        f"  • compacted: {proposal['compacted_bytes']} bytes\n\n"
                        f"Review the three files in "
                        f"`{proposal['notes_path']}`'s directory."
                    ),
                    url="/cp/files",
                    topic="claude_md_compaction",
                    critical=False,
                    arbitrate=True,
                )
            except Exception:
                logger.debug(
                    "claude_md_compaction: notify failed", exc_info=True,
                )
    return summary
