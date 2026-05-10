"""Inbox watcher + router.

One scan tick = walk ``workspace/inbox/`` → classify each file →
dispatch to the per-kind handler → archive on success, leave in place
on failure → write a per-file manifest under ``.processed/``.

Idempotency
-----------

  * ``.processed/<sha256>.json`` records each file by content hash, so
    re-dropping the same bytes (different name, different timestamp)
    is a no-op.
  * Hidden files / dotfiles / files inside ``.processed`` /
    ``.archive`` are skipped.
  * A file currently being written (modified within the last 5 s) is
    deferred to the next tick to avoid partial reads.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.inbox.classifier import FileClassification, classify_file

logger = logging.getLogger(__name__)


_DEFAULT_INBOX = Path("/app/workspace/inbox")
_PROCESSED_DIR = ".processed"
_ARCHIVE_DIR = ".archive"
_QUIET_SECONDS = 5  # don't process a file modified in the last N seconds


# Handler signature: (path, classification, base_dir) -> str
# Return string is a one-line outcome the manifest records (e.g.
# "imported: 12,345 records across 5 kinds"). A handler that fails
# raises; the router catches and records ``status="failed"``.
Handler = Callable[[Path, FileClassification, Path], str]


def _enabled() -> bool:
    return os.getenv("INBOX_INGESTION_ENABLED", "false").lower() in (
        "true", "1", "yes", "on",
    )


def _resolve_inbox() -> Path:
    raw = os.getenv("INBOX_DIR")
    return Path(raw) if raw else _DEFAULT_INBOX


@dataclass
class ScanResult:
    """Outcome of one scan tick."""

    status: str          # "ok" | "skipped_disabled" | "skipped_no_inbox"
    processed: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    skipped_dedup: list[str] = field(default_factory=list)
    skipped_unknown: list[str] = field(default_factory=list)


# ── Handler registry ──────────────────────────────────────────────────


def _handle_apple_health(
    path: Path, classification: FileClassification, base: Path,
) -> str:
    """Route Apple Health zips to the §5.1 importer."""
    from app.health import import_apple_export
    result = import_apple_export(path)
    if result.status != "ok":
        raise RuntimeError(
            f"apple_health import {result.status}: {result.failure_reason}"
        )
    parts = ", ".join(
        f"{k}={n}" for k, n in sorted(result.records_written.items())
    )
    return (
        f"imported {result.total_written} records across "
        f"{len(result.records_written)} kinds ({parts})"
    )


def _handle_text(
    path: Path, classification: FileClassification, base: Path,
) -> str:
    """Drop text/markdown into the canonical notes root the React
    ``/cp/files`` view lists from (``workspace/notes/``).

    The destination is the same root surfaced by ``app/api/files_api.py``,
    so anything copied here shows up in the dashboard's Files tab and
    can be sent via Signal/Email/Discord through the existing
    ``send_via_*`` plumbing. On filename collision we add a numeric
    suffix instead of overwriting (the inbox is meant to be additive).
    """
    from app.paths import WORKSPACE_ROOT
    notes_root = Path(os.getenv("INBOX_NOTES_DIR", str(WORKSPACE_ROOT / "notes")))
    notes_root.mkdir(parents=True, exist_ok=True)
    dest = notes_root / path.name
    if dest.exists():
        stem, suffix = dest.stem, dest.suffix
        i = 1
        while True:
            cand = notes_root / f"{stem}.{i}{suffix}"
            if not cand.exists():
                dest = cand
                break
            i += 1
    shutil.copy2(path, dest)
    return f"copied to {dest}"


def _handle_unsupported(
    path: Path, classification: FileClassification, base: Path,
) -> str:
    """A handler that intentionally refuses. Used for kinds we recognise
    but haven't wired a real handler for yet (audio/image/pdf/csv/
    spreadsheet). The file stays in place; the operator sees an alert."""
    raise RuntimeError(
        f"no handler for kind={classification.kind!r}; "
        f"recognised but not yet routable"
    )


HANDLER_REGISTRY: dict[str, Handler] = {
    "apple_health_export": _handle_apple_health,
    "text": _handle_text,
    # Recognised but unhandled — they sit until a real handler is wired.
    "audio": _handle_unsupported,
    "image": _handle_unsupported,
    "pdf": _handle_unsupported,
    "csv": _handle_unsupported,
    "spreadsheet": _handle_unsupported,
}


# ── Helpers ───────────────────────────────────────────────────────────


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _manifest_path(processed_dir: Path, sha: str) -> Path:
    return processed_dir / f"{sha}.json"


def _is_recently_modified(path: Path, threshold_s: float) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age < threshold_s


def _archive_path(archive_dir: Path, original_name: str) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bucket = archive_dir / today
    bucket.mkdir(parents=True, exist_ok=True)
    target = bucket / original_name
    if not target.exists():
        return target
    # On a name collision, append a numeric suffix.
    stem = target.stem
    suffix = target.suffix
    i = 1
    while True:
        cand = bucket / f"{stem}.{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def _write_manifest(
    processed_dir: Path,
    sha: str,
    *,
    path: Path,
    classification: FileClassification,
    handler: str,
    status: str,
    outcome: str,
    moved_to: str | None,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sha256": sha,
        "original_name": path.name,
        "classification": {
            "kind": classification.kind,
            "confidence": classification.confidence,
            "reason": classification.reason,
        },
        "handler": handler,
        "status": status,
        "outcome": outcome,
        "moved_to": moved_to,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    target = _manifest_path(processed_dir, sha)
    tmp = target.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp.replace(target)
    except OSError as exc:
        logger.warning("inbox.router: manifest write failed for %s: %s", sha, exc)


# ── Main entry point ──────────────────────────────────────────────────


def scan_and_route(
    *,
    inbox_dir: Path | str | None = None,
    handlers: dict[str, Handler] | None = None,
) -> ScanResult:
    """One scan tick. Failure-isolated — a per-file error is captured
    in :attr:`ScanResult.failed`; never raises."""
    if not _enabled():
        return ScanResult(status="skipped_disabled")

    base = Path(inbox_dir) if inbox_dir else _resolve_inbox()
    if not base.exists():
        return ScanResult(status="skipped_no_inbox")

    processed_dir = base / _PROCESSED_DIR
    archive_dir = base / _ARCHIVE_DIR

    registry = handlers if handlers is not None else HANDLER_REGISTRY
    result = ScanResult(status="ok")

    for entry in sorted(base.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if _is_recently_modified(entry, _QUIET_SECONDS):
            result.deferred.append(entry.name)
            continue

        sha = _sha256_of(entry)
        if not sha:
            result.failed.append({
                "name": entry.name,
                "reason": "could not hash file",
            })
            continue

        if _manifest_path(processed_dir, sha).exists():
            result.skipped_dedup.append(entry.name)
            # Move the duplicate out of the inbox so subsequent ticks
            # don't keep tripping the dedup branch.
            try:
                target = _archive_path(archive_dir, entry.name)
                shutil.move(str(entry), str(target))
            except OSError:
                pass
            continue

        classification = classify_file(entry)
        if classification.kind == "unknown":
            result.skipped_unknown.append(entry.name)
            _write_manifest(
                processed_dir, sha,
                path=entry, classification=classification,
                handler="<none>", status="unknown",
                outcome=classification.reason, moved_to=None,
            )
            continue

        handler = registry.get(classification.kind)
        if handler is None:
            result.skipped_unknown.append(entry.name)
            _write_manifest(
                processed_dir, sha,
                path=entry, classification=classification,
                handler="<none>", status="no_handler",
                outcome=f"no handler registered for {classification.kind}",
                moved_to=None,
            )
            continue

        handler_name = getattr(handler, "__name__", str(handler))
        try:
            outcome = handler(entry, classification, base)
        except Exception as exc:  # noqa: BLE001
            result.failed.append({
                "name": entry.name,
                "kind": classification.kind,
                "reason": str(exc),
            })
            _write_manifest(
                processed_dir, sha,
                path=entry, classification=classification,
                handler=handler_name, status="failed",
                outcome=f"handler raised: {exc}", moved_to=None,
            )
            continue

        # Success — archive the file and record the manifest.
        moved_to: str | None = None
        try:
            target = _archive_path(archive_dir, entry.name)
            shutil.move(str(entry), str(target))
            moved_to = str(target)
        except OSError as exc:
            logger.warning(
                "inbox.router: archive move failed for %s: %s", entry, exc,
            )

        _write_manifest(
            processed_dir, sha,
            path=entry, classification=classification,
            handler=handler_name, status="processed",
            outcome=outcome, moved_to=moved_to,
        )
        result.processed.append({
            "name": entry.name,
            "kind": classification.kind,
            "outcome": outcome,
            "moved_to": moved_to,
        })

    return result
