"""One-shot CR-spam cleanup tool.

PROGRAM §57 — Q18 migration helper. Folds N identical CRs from one
requestor into a single canonical record with ``recurrence_count``
and archives the duplicates. Designed to clean up the 2026-05-16
incident (1200 ``local_only_drill`` CRs) without losing audit trail.

Use case
========

The Q3 lifecycle dedup prevents future spam, but the historical
spam files are still on disk cluttering the operator review queue.
This module is the one-shot consolidator that brings the existing
store into the dedup model. Idempotent — re-running on already-
cleaned data is a no-op.

What it does
============

  1. List all PENDING CRs from the named requestor.
  2. Group by ``(requestor, path)`` — the dedup key (no ``content_hash``
     yet on legacy records).
  3. For each group with > 1 CR, pick the oldest as canonical and:
       a. Set canonical.recurrence_count = N - 1
       b. Set canonical.last_recurrence_at = newest CR's created_at
       c. Set canonical.content_hash = freshly-computed
       d. Set canonical.first_seen_at = canonical.created_at
       e. Move the other CRs' JSON files to
          ``workspace/change_requests/archive/<timestamp>_drill_spam/``
       f. Record the consolidation in the audit log.

Safety
------

* Never deletes files — only moves to an archive subdir under the
  same ``workspace/change_requests/`` tree.
* Operates on PENDING only — terminal CRs (REJECTED/APPLIED/...) are
  left alone (their final dispositions matter for audit).
* Best-effort: per-CR failures are logged and skipped; the run
  continues with the next group.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.change_requests import store
from app.change_requests.lifecycle import _compute_content_hash, _compute_diff
from app.change_requests.models import ChangeRequest, Status

logger = logging.getLogger(__name__)


def _archive_dir() -> Path:
    """The archive subdir under change_requests. Honors the
    monkey-patched ``store._STORE_DIR`` so tests don't pollute prod."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_drill_spam")
    return store._STORE_DIR / "archive" / ts


def _record_audit(consolidated_id: str, archived_ids: list[str], requestor: str) -> None:
    """Record the consolidation in the store's existing audit log."""
    try:
        store._append_audit({
            "event": "spam_cleanup_consolidation",
            "consolidated_cr_id": consolidated_id,
            "archived_cr_ids": archived_ids,
            "requestor": requestor,
            "n_archived": len(archived_ids),
        })
    except Exception:
        logger.warning("spam_cleanup: audit append failed", exc_info=True)


def consolidate(requestor: str) -> dict[str, Any]:
    """Consolidate PENDING CRs from ``requestor`` into deduped records.

    Returns a summary dict with ``n_groups``, ``n_consolidated`` (CRs
    that bumped a canonical), ``n_archived`` (CRs moved to archive),
    ``canonical_ids`` (preserved record ids).
    """
    summary: dict[str, Any] = {
        "requestor": requestor,
        "n_groups": 0,
        "n_consolidated": 0,
        "n_archived": 0,
        "canonical_ids": [],
        "groups": [],
    }
    # store.list_all returns newest-first; we need oldest-first for
    # "pick the oldest as canonical".
    all_pending = [
        cr for cr in store.list_all(status=Status.PENDING, limit=10000)
        if cr.requestor == requestor
    ]
    all_pending.sort(key=lambda c: c.created_at)

    # Group by (requestor, path). For drill spam these are usually
    # the same path; for completeness we key by both.
    groups: dict[tuple[str, str], list[ChangeRequest]] = {}
    for cr in all_pending:
        groups.setdefault((cr.requestor, cr.path), []).append(cr)
    summary["n_groups"] = sum(1 for g in groups.values() if len(g) > 1)

    archive_dir = _archive_dir()
    archive_dir.mkdir(parents=True, exist_ok=True)

    for (req, path), members in groups.items():
        if len(members) < 2:
            continue
        canonical = members[0]
        duplicates = members[1:]

        # Backfill Q18 fields on the canonical.
        if not canonical.content_hash:
            canonical.content_hash = _compute_content_hash(
                requestor=canonical.requestor,
                path=canonical.path,
                diff=canonical.diff,
                reason=canonical.reason,
            )
        if not canonical.first_seen_at:
            canonical.first_seen_at = canonical.created_at
        canonical.recurrence_count = len(duplicates)
        canonical.last_recurrence_at = duplicates[-1].created_at

        try:
            store.save(canonical, audit_event="spam_cleanup_canonical")
        except Exception as exc:
            logger.warning("spam_cleanup: canonical save failed for %s: %s",
                            canonical.id, exc)
            continue

        archived_ids: list[str] = []
        for dup in duplicates:
            src = store._record_path(dup.id)
            if not src.exists():
                continue
            try:
                shutil.move(str(src), str(archive_dir / src.name))
                archived_ids.append(dup.id)
            except OSError as exc:
                logger.warning("spam_cleanup: move failed for %s: %s",
                                dup.id, exc)
                continue

        # Drop the archived ids from the in-memory index so list_all
        # stops returning them immediately.
        with store._LOCK:
            idx = store._index()
            for aid in archived_ids:
                idx.pop(aid, None)

        _record_audit(canonical.id, archived_ids, req)

        summary["n_consolidated"] += 1
        summary["n_archived"] += len(archived_ids)
        summary["canonical_ids"].append(canonical.id)
        summary["groups"].append({
            "requestor": req,
            "path": path,
            "canonical_id": canonical.id,
            "n_archived": len(archived_ids),
        })

    summary["archive_dir"] = str(archive_dir)
    return summary


# ── CLI entry point ─────────────────────────────────────────────────────


def _main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="One-shot CR spam cleanup")
    parser.add_argument(
        "--requestor", required=True,
        help="Requestor whose duplicates to consolidate "
             "(e.g. 'local_only_drill').",
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute summary without moving files.")
    args = parser.parse_args()

    if args.dry_run:
        all_pending = [
            cr for cr in store.list_all(status=Status.PENDING, limit=10000)
            if cr.requestor == args.requestor
        ]
        groups: dict[tuple[str, str], int] = {}
        for cr in all_pending:
            key = (cr.requestor, cr.path)
            groups[key] = groups.get(key, 0) + 1
        n_dupes = sum(c - 1 for c in groups.values() if c > 1)
        print(f"Would consolidate {len(all_pending)} PENDING CRs from "
              f"{args.requestor!r} into {sum(1 for c in groups.values() if c > 1)} "
              f"canonical(s); would archive {n_dupes} duplicates.")
        return 0

    summary = consolidate(args.requestor)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
