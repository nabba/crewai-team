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

import contextlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


# ── Cross-rotation reader/writer lock (PROGRAM §40.2 Item 3) ─────────────
#
# Without a coordinated lock, a reader chaining
# ``read_archive(include_live=True)`` can briefly see lines BOTH in the
# archive (just-appended) AND in the live file (not-yet-truncated)
# during a rotation pass — yielding duplicates. The window is
# microseconds in single-process gateways, but multi-process readers
# (dashboard poll, decentered probe, healing monitors) DO collide.
#
# We use a POSIX advisory lock on a sidecar ``.rotation_lock`` file
# adjacent to the JSONL. The rotator takes LOCK_EX during the
# archive-write + live-truncate critical section; readers take LOCK_SH
# while iterating across archive + live. The lock is best-effort —
# platforms without fcntl (Windows) silently fall through to the
# pre-Q3.2 behavior. macOS + Linux + Docker Linux all support it.

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False


def _lock_path_for(path: Path) -> Path:
    """Sidecar lock-file path. Adjacent to the JSONL so the same
    filesystem permissions apply. Hidden by ``.`` prefix."""
    return path.parent / f".{path.name}.rotation_lock"


@contextlib.contextmanager
def _rotation_lock(path: Path, exclusive: bool):
    """Acquire LOCK_EX (writer) or LOCK_SH (reader) on the sidecar
    lock file for ``path``. Best-effort: silently degrades to no-op
    when fcntl isn't available, when the lock file can't be opened,
    or when the lock acquisition fails. The rotator IS still
    correctness-isolated by the in-process ``_LOCK`` writers hold;
    this lock additionally guards cross-process readers."""
    if not _HAS_FCNTL:
        yield
        return
    lp = _lock_path_for(path)
    try:
        lp.parent.mkdir(parents=True, exist_ok=True)
        # Open in read+write+create mode; fcntl works on any open fd.
        fd = os.open(str(lp), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        yield
        return
    try:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, op)
        except OSError:
            # Couldn't take the lock — degrade gracefully.
            yield
            return
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


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

    # Q3.2 (PROGRAM §40.2 Item 3) — exclusive lock around archive
    # append + live truncate, so cross-process readers can't observe
    # the lines in BOTH archive and live simultaneously and double-
    # count them.
    with _rotation_lock(p, exclusive=True):
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
    # Q3.2 (PROGRAM §40.2 Item 3) — shared lock around the entire
    # archive+live read so we can't be interleaved with a rotation in
    # progress. The rotator's exclusive lock will serialise us briefly.
    # The lock is best-effort across platforms.
    with _rotation_lock(p, exclusive=False):
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
