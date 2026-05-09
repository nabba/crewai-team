"""Log archival — gzip rotated logs to a cold archive instead of discarding.

Wave 0/1 closure (#A5, 2026-05-09). Two existing log files were rotating
without archival:

  * ``workspace/logs/errors.jsonl`` — RotatingFileHandler keeps 3
    historical files (.1, .2, .3) and discards anything older. With
    ~5,000 errors/day, that's at most a few days of history retained.
  * ``workspace/audit_journal.json`` — the operational audit journal
    (separate from the immutable Postgres ``control_plane.audit_log``).
    Currently no rotation; just a single growing file.

This monitor runs daily:

  1. Walks the rotated errors.jsonl.{1,2,3} files; gzips them into
     ``workspace/logs/archive/<YYYY-MM>.jsonl.gz`` (one per month;
     append-mode for monthly aggregation).
  2. Same for audit_journal.json — but since it's monolithic, we
     check the file size; when it crosses ``_AUDIT_ROTATE_BYTES``
     (default 10 MB), we copy-then-truncate, gzip the copy into
     ``workspace/audit_archive/``.
  3. Retention: deletes archive files older than
     ``LOG_ARCHIVE_RETENTION_DAYS`` (default 90 days).

Failure to archive is non-fatal — the source files stay where they
are, and the next pass tries again.
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "log_archival.json"
_RUN_CADENCE_S = 24 * 3600   # daily

# Sources we archive.
_ERRORS_LOG_DIR = Path("/app/workspace/logs")
_ERRORS_ARCHIVE_DIR = Path("/app/workspace/logs/archive")
_AUDIT_JOURNAL_PATH = Path("/app/workspace/audit_journal.json")
_AUDIT_ARCHIVE_DIR = Path("/app/workspace/audit_archive")
_AUDIT_ROTATE_BYTES = 10 * 1024 * 1024  # 10 MB

# Phase D #2 (2026-05-09): evolution-run archives. Per-run dirs under
# ``workspace/shinka_results/`` accumulate forever; we tarball + gzip
# any older than 90 days into ``workspace/shinka_results/archive/``.
_EVOLUTION_RUNS_DIR = Path("/app/workspace/shinka_results")
_EVOLUTION_ARCHIVE_DIR = Path("/app/workspace/shinka_results/archive")
_EVOLUTION_AGE_DAYS = 90


def _retention_days() -> int:
    raw = os.getenv("LOG_ARCHIVE_RETENTION_DAYS", "90").strip()
    try:
        return max(7, int(raw))
    except ValueError:
        return 90


def _archive_errors_jsonl() -> dict:
    """Find errors.jsonl.{1,2,3} and gzip-append into a monthly archive."""
    summary = {"rotated_files": 0, "bytes_archived": 0}
    if not _ERRORS_LOG_DIR.exists():
        return summary

    _ERRORS_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    month_key = datetime.now(timezone.utc).strftime("%Y-%m")
    archive_path = _ERRORS_ARCHIVE_DIR / f"{month_key}.jsonl.gz"

    for rotated in sorted(_ERRORS_LOG_DIR.glob("errors.jsonl.*")):
        if rotated.name == "errors.jsonl":
            continue  # only the rotated suffixes, not the live file
        try:
            size = rotated.stat().st_size
            with rotated.open("rb") as src, gzip.open(archive_path, "ab") as dst:
                shutil.copyfileobj(src, dst)
            rotated.unlink()
            summary["rotated_files"] += 1
            summary["bytes_archived"] += size
        except OSError:
            logger.debug(
                "log_archival: archive failed for %s", rotated, exc_info=True,
            )
    return summary


def _archive_audit_journal() -> dict:
    """If audit_journal.json is past the size cap, gzip-rotate it."""
    summary = {"rotated": False, "bytes_archived": 0}
    if not _AUDIT_JOURNAL_PATH.exists():
        return summary
    try:
        size = _AUDIT_JOURNAL_PATH.stat().st_size
    except OSError:
        return summary
    if size < _AUDIT_ROTATE_BYTES:
        return summary

    _AUDIT_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = _AUDIT_ARCHIVE_DIR / f"audit_journal_{ts}.json.gz"

    try:
        with _AUDIT_JOURNAL_PATH.open("rb") as src, gzip.open(archive_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        # Truncate the live file rather than delete — concurrent writers
        # already have it open; truncation preserves their FD.
        with _AUDIT_JOURNAL_PATH.open("w") as f:
            f.write("[]")
        summary["rotated"] = True
        summary["bytes_archived"] = size
    except OSError:
        logger.debug(
            "log_archival: audit-journal rotate failed", exc_info=True,
        )
    return summary


def _archive_evolution_runs() -> dict:
    """Tar+gzip per-run dirs older than ``_EVOLUTION_AGE_DAYS`` and delete.

    Phase D #2 (2026-05-09): ShinkaEvolve writes one directory per run
    under ``workspace/shinka_results/run_<ts>/``. Without retention the
    dir grows unbounded — each run is ~100KB-2MB plus per-iteration
    artefacts. Gzipped tarballs land in ``shinka_results/archive/``;
    the original dir is removed after a successful tar.
    """
    summary = {"archived_runs": 0, "bytes_archived": 0}
    if not _EVOLUTION_RUNS_DIR.exists():
        return summary
    _EVOLUTION_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - _EVOLUTION_AGE_DAYS * 86400
    try:
        for entry in _EVOLUTION_RUNS_DIR.iterdir():
            # Skip the archive subdir itself, and any non-run files.
            if not entry.is_dir():
                continue
            if entry.name == "archive":
                continue
            try:
                if entry.stat().st_mtime > cutoff:
                    continue
                size_before = sum(
                    p.stat().st_size for p in entry.rglob("*") if p.is_file()
                )
                target = _EVOLUTION_ARCHIVE_DIR / f"{entry.name}.tar.gz"
                with tarfile.open(target, "w:gz") as tar:
                    tar.add(entry, arcname=entry.name)
                shutil.rmtree(entry)
                summary["archived_runs"] += 1
                summary["bytes_archived"] += size_before
            except OSError:
                logger.debug(
                    "log_archival: evolution archive failed for %s",
                    entry, exc_info=True,
                )
                continue
    except OSError:
        logger.debug(
            "log_archival: evolution dir scan failed", exc_info=True,
        )
    return summary


def _purge_old_archives(retention_days: int) -> dict:
    """Delete archive files older than retention_days. Returns count + bytes."""
    summary = {"deleted_files": 0, "bytes_deleted": 0}
    cutoff = time.time() - retention_days * 24 * 3600
    # The evolution archive runs on its own retention (90 days fixed) —
    # keep the user-tunable retention only for errors + audit archives.
    for d in (_ERRORS_ARCHIVE_DIR, _AUDIT_ARCHIVE_DIR, _EVOLUTION_ARCHIVE_DIR):
        if not d.exists():
            continue
        try:
            for f in d.iterdir():
                if not f.is_file():
                    continue
                try:
                    if f.stat().st_mtime > cutoff:
                        continue
                    size = f.stat().st_size
                    f.unlink()
                    summary["deleted_files"] += 1
                    summary["bytes_deleted"] += size
                except OSError:
                    continue
        except OSError:
            continue
    return summary


def run() -> None:
    """One pass — cadence-guarded internally."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    err_summary = _archive_errors_jsonl()
    audit_summary = _archive_audit_journal()
    evolution_summary = _archive_evolution_runs()
    purge_summary = _purge_old_archives(_retention_days())

    audit_event(
        "log_archival_pass",
        rotated_error_files=err_summary["rotated_files"],
        bytes_archived=(
            err_summary["bytes_archived"]
            + audit_summary["bytes_archived"]
            + evolution_summary["bytes_archived"]
        ),
        audit_journal_rotated=audit_summary["rotated"],
        evolution_runs_archived=evolution_summary["archived_runs"],
        deleted_old_archives=purge_summary["deleted_files"],
        bytes_deleted=purge_summary["bytes_deleted"],
    )

    state["last_summary"] = {
        "errors": err_summary,
        "audit": audit_summary,
        "evolution": evolution_summary,
        "purge": purge_summary,
    }
    write_state_json(_STATE_FILE, state)
