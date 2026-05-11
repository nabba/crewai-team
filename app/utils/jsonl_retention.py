"""Bounded JSONL retention helper.

Phase F #7 (2026-05-09) — original ``cap_jsonl`` + ``append_with_cap``.
PROGRAM §40 (2026-05-10) — ``append_with_archive_rotate`` +
``read_archive`` for consciousness-relevant logs that should preserve
history forever (affect trace, salience, care ledger, welfare audit).

Multiple modules under ``app/`` shipped append-only JSONL ledgers
without retention policy. Over years they grow unboundedly. Two
discipline patterns exist:

  * ``cap_jsonl`` / ``append_with_cap`` — truncate-on-overflow.
    Right for OPERATIONAL telemetry (paper_pipeline proposals,
    adapter lifecycle history, diagnosis telemetry). Lost lines are
    fine — the consumer reads only recent state.

  * ``append_with_archive_rotate`` / ``read_archive`` — rotate the
    oldest half to ``<path.parent>/archive/<YYYY-MM>_<basename>``.
    Right for CONSCIOUSNESS-RELEVANT data (affect trace, salience,
    care ledger, welfare audit). Future probes (HOT-1, decentered
    reflection, backward counterfactual replay) may want decade-
    scale data. Truncating loses history forever; archiving keeps
    it queryable while bounding the live-hot file.

Both helpers are best-effort + silent on failure — losing a few
rows is preferable to crashing a writer because the disk filled.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def cap_jsonl(path: Path | str, max_lines: int) -> int:
    """Truncate ``path`` to the last ``max_lines`` lines. Returns the
    number of lines dropped (0 when no rewrite needed).

    Idempotent — calling repeatedly is a no-op once the file is
    within bounds.
    """
    p = Path(path)
    if not p.exists() or max_lines <= 0:
        return 0
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        logger.debug("jsonl_retention: read failed for %s", p, exc_info=True)
        return 0
    if len(lines) <= max_lines:
        return 0
    keep = lines[-max_lines:]
    dropped = len(lines) - len(keep)
    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text("".join(keep), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug("jsonl_retention: rewrite failed for %s", p, exc_info=True)
        return 0
    return dropped


def append_with_cap(path: Path | str, json_line: str, max_lines: int) -> None:
    """Append ``json_line`` (no trailing newline) to ``path`` and
    enforce the cap. Best-effort; silent on failure.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with p.open("a", encoding="utf-8") as f:
            if not json_line.endswith("\n"):
                f.write(json_line + "\n")
            else:
                f.write(json_line)
    except OSError:
        logger.debug("jsonl_retention: append failed for %s", p, exc_info=True)
        return
    cap_jsonl(p, max_lines)


# ── Archive rotation (PROGRAM §40, 2026-05-10) ────────────────────────


def _archive_dir_for(path: Path) -> Path:
    """Default archive location for a live JSONL file."""
    return path.parent / "archive"


def _archive_name(now: datetime, basename: str) -> str:
    """``YYYY-MM_<basename>`` so a single month's rotations all share
    the same archive file (subsequent rotations append rather than
    creating timestamped variants)."""
    return f"{now.strftime('%Y-%m')}_{basename}"


def append_with_archive_rotate(
    path: Path | str,
    json_line: str,
    *,
    max_lines: int,
    archive_dir: Path | str | None = None,
    keep_fraction: float = 0.5,
    _now: datetime | None = None,
) -> None:
    """Append ``json_line``. When the live file exceeds ``max_lines``,
    rotate the OLDEST ``(1 - keep_fraction)`` of it into
    ``<archive_dir>/<YYYY-MM>_<basename>`` and keep the newer rest in
    the live file. Preserves history while bounding the read-hot file.

    Defaults: ``archive_dir = <path.parent>/archive/``,
    ``keep_fraction = 0.5`` (rotate oldest half).

    The archive file is OPEN for append — multiple rotations within
    the same UTC month accumulate into one monthly archive file.
    Rolling over month boundaries creates a new file. Old months are
    never auto-deleted (the archive grows forever; operator can run
    ``compact_archive(...)`` if needed).

    Failure modes degrade silently — losing a single rotation is
    preferable to crashing the writer.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Normal append.
    try:
        with p.open("a", encoding="utf-8") as f:
            if not json_line.endswith("\n"):
                f.write(json_line + "\n")
            else:
                f.write(json_line)
    except OSError:
        logger.debug(
            "jsonl_retention: archive-rotate append failed for %s",
            p, exc_info=True,
        )
        return

    # Rotation: only when over cap.
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    if len(lines) <= max_lines:
        return

    # We've passed the cap. Move the oldest (1 - keep_fraction) to
    # archive; keep the newer keep_fraction in-place.
    keep_fraction = max(0.1, min(0.9, float(keep_fraction)))
    keep_count = int(max_lines * keep_fraction)
    rotate_count = len(lines) - keep_count
    if rotate_count <= 0:
        return
    rotate_lines = lines[:rotate_count]
    keep_lines = lines[rotate_count:]

    arch_dir = (
        Path(archive_dir) if archive_dir is not None else _archive_dir_for(p)
    )
    arch_dir.mkdir(parents=True, exist_ok=True)
    now = _now or datetime.now(timezone.utc)
    arch_path = arch_dir / _archive_name(now, p.name)

    # Append-rotate: archive file accumulates OLDEST-first within
    # its month. Then atomic-rewrite the live file with the kept tail.
    try:
        with arch_path.open("a", encoding="utf-8") as f:
            f.writelines(rotate_lines)
    except OSError:
        logger.debug(
            "jsonl_retention: archive write failed for %s → %s",
            p, arch_path, exc_info=True,
        )
        return

    try:
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text("".join(keep_lines), encoding="utf-8")
        tmp.replace(p)
    except OSError:
        logger.debug(
            "jsonl_retention: live-rewrite failed for %s after archive write",
            p, exc_info=True,
        )
        return

    logger.info(
        "jsonl_retention: rotated %d lines from %s → %s (kept %d live)",
        rotate_count, p, arch_path, len(keep_lines),
    )


def read_archive(
    path: Path | str,
    *,
    archive_dir: Path | str | None = None,
    include_live: bool = True,
) -> Iterator[str]:
    """Yield every line in chronological order across the live file
    plus all archived monthly files for ``path``.

    Use case: HOT-1 probe / decentered reflection / backward
    counterfactual replay walking the full multi-year history of a
    log without caring whether it's been rotated.

    Iterates lazily (line-by-line) so even a multi-GB archive is
    OK on memory.
    """
    p = Path(path)
    arch_dir = (
        Path(archive_dir) if archive_dir is not None else _archive_dir_for(p)
    )
    # Walk archives in chronological order. Filename is
    # ``YYYY-MM_<basename>`` so plain string sort is chronological.
    if arch_dir.exists():
        archives = sorted(arch_dir.glob(f"*_{p.name}"))
        for arch_path in archives:
            try:
                with arch_path.open("r", encoding="utf-8") as f:
                    yield from f
            except OSError:
                continue
    if include_live and p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                yield from f
        except OSError:
            return


def archive_stats(
    path: Path | str,
    *,
    archive_dir: Path | str | None = None,
) -> dict:
    """Return summary stats for a path's archive: number of monthly
    files, oldest/newest months, total archived bytes, total archived
    lines (approx — counts via newline scan).

    Cheap to call: O(n_files) stat calls + a per-file line count.
    """
    p = Path(path)
    arch_dir = (
        Path(archive_dir) if archive_dir is not None else _archive_dir_for(p)
    )
    if not arch_dir.exists():
        return {
            "archive_files": 0,
            "oldest_month": None,
            "newest_month": None,
            "archived_bytes": 0,
            "archived_lines": 0,
        }
    archives = sorted(arch_dir.glob(f"*_{p.name}"))
    if not archives:
        return {
            "archive_files": 0,
            "oldest_month": None,
            "newest_month": None,
            "archived_bytes": 0,
            "archived_lines": 0,
        }
    total_bytes = 0
    total_lines = 0
    for arch_path in archives:
        try:
            stat = arch_path.stat()
            total_bytes += stat.st_size
            with arch_path.open("rb") as f:
                total_lines += sum(1 for _ in f)
        except OSError:
            continue
    # Filename: "YYYY-MM_<basename>" — split on first underscore.
    oldest = archives[0].name.split("_", 1)[0]
    newest = archives[-1].name.split("_", 1)[0]
    return {
        "archive_files": len(archives),
        "oldest_month": oldest,
        "newest_month": newest,
        "archived_bytes": total_bytes,
        "archived_lines": total_lines,
    }
