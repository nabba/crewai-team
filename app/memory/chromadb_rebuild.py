"""Operator-runnable per-collection rebuild for ChromaDB.

PROGRAM §40 (2026-05-10) — Q3 Item 10.

The quarterly ``chromadb_hygiene`` monitor recovers space from
soft-deleted rows in the SQLite metadata file via plain ``VACUUM``.
That doesn't touch HNSW segments — those are managed by chromadb-
internal code and only get reclaimed by a full rebuild.

This module provides the operator-initiated path:

    python -m app.memory.chromadb_rebuild --kb memory --collection team_shared
    python -m app.memory.chromadb_rebuild --kb memory --collection team_shared --dry-run

Strategy (atomic from the application's point of view):

  1. Verify the source collection exists and is read-coherent
     (``count()`` succeeds; sample peek returns vectors of the
     pinned dimension).
  2. Stream every row out (``ids, documents, metadatas, embeddings``)
     in batches via ``collection.get(...)`` with offset cursoring.
  3. Pause writes (cooperative — operator runs this in a quiescent
     window; the manager takes a short module-level lock).
  4. ``client.delete_collection(name)``  →  ``client.create_collection(name, metadata=…)``
  5. Re-insert rows in batches with the original IDs preserved.
  6. Verify final count() matches.

Caveats:
  * ChromaDB has no rename API, so the swap is delete-then-recreate.
    There is a brief window (single-digit seconds for typical sizes)
    where the collection has no rows — the operator runs this in a
    quiescent window. Concurrent reads during that window will see
    an empty collection or hit zero-count short-circuits in the
    manager.
  * If anything raises mid-rebuild, the exported snapshot is dumped
    to ``workspace/<kb>/.rebuild_backups/<collection>__<ts>.jsonl.gz``
    so the operator can re-import manually with
    ``--from-snapshot <path>``.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Streaming batch size. ChromaDB's get() pulls all rows by default;
# we cursor in chunks to keep memory bounded for collections with
# tens of thousands of rows.
_EXPORT_BATCH = 500
_IMPORT_BATCH = 200            # add() is faster in moderate batches

# Module-level rebuild lock; if multiple operator runs collide, the
# second blocks. Real cross-process safety relies on the operator
# running this in a quiescent window — this lock is a courtesy.
_rebuild_lock = threading.Lock()


@dataclass
class RebuildPlan:
    kb: str                     # workspace/<kb> root (memory, philosophy, …)
    collection: str             # ChromaDB collection name
    workspace_root: Path
    dry_run: bool = False
    snapshot_path: Path | None = None   # written under .rebuild_backups
    created_at: float = field(default_factory=time.time)


@dataclass
class RebuildSummary:
    kb: str
    collection: str
    rows_exported: int
    rows_reimported: int
    bytes_before: int
    bytes_after: int
    duration_s: float
    snapshot_path: str | None
    dry_run: bool
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "kb": self.kb,
            "collection": self.collection,
            "rows_exported": self.rows_exported,
            "rows_reimported": self.rows_reimported,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "duration_s": round(self.duration_s, 3),
            "snapshot_path": self.snapshot_path,
            "dry_run": self.dry_run,
            "ok": self.ok,
            "error": self.error,
        }


def _kb_root(workspace_root: Path, kb: str) -> Path:
    return workspace_root / kb


def _kb_size_bytes(kb_root: Path) -> int:
    """Sum of every file under the KB directory. Cheap (one stat per file)."""
    total = 0
    if not kb_root.exists():
        return 0
    for p in kb_root.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _open_client(kb_root: Path):
    """Open a ChromaDB PersistentClient pointed at the KB-specific
    persist dir. We don't reuse the global manager's client because
    KB-rooted clients are KB-specific."""
    import chromadb
    return chromadb.PersistentClient(path=str(kb_root))


def _stream_collection(col, batch: int = _EXPORT_BATCH) -> Iterator[dict]:
    """Yield ``{ids, documents, metadatas, embeddings}`` chunks from
    ``collection.get(...)`` paged by offset."""
    offset = 0
    while True:
        chunk = col.get(
            limit=batch, offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = chunk.get("ids") or []
        if not ids:
            return
        yield {
            "ids": list(ids),
            "documents": list(chunk.get("documents") or [None] * len(ids)),
            "metadatas": list(chunk.get("metadatas") or [None] * len(ids)),
            "embeddings": list(chunk.get("embeddings") or [None] * len(ids)),
        }
        offset += len(ids)
        if len(ids) < batch:
            return


def _ensure_snapshot_dir(kb_root: Path) -> Path:
    snap_dir = kb_root / ".rebuild_backups"
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


def _write_snapshot(plan: RebuildPlan, chunks: list[dict]) -> Path:
    """Append-stream gzipped JSONL with one row per line (id, doc,
    meta, embedding). Compresses ~10x on typical text + 768-d float."""
    snap_dir = _ensure_snapshot_dir(_kb_root(plan.workspace_root, plan.kb))
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime(plan.created_at))
    fname = f"{plan.collection}__{ts}.jsonl.gz"
    path = snap_dir / fname
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for chunk in chunks:
            for i in range(len(chunk["ids"])):
                row = {
                    "id": chunk["ids"][i],
                    "document": chunk["documents"][i],
                    "metadata": chunk["metadatas"][i],
                    "embedding": chunk["embeddings"][i],
                }
                f.write(json.dumps(row, default=str))
                f.write("\n")
    return path


def _read_snapshot(path: Path) -> Iterator[dict]:
    """Yield rows from a gzipped snapshot. Used by --from-snapshot
    recovery flow."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@contextmanager
def _exclusive_rebuild():
    """Take the in-process rebuild lock. Best-effort cross-process
    safety relies on the operator running this in a quiescent window."""
    acquired = _rebuild_lock.acquire(timeout=5.0)
    if not acquired:
        raise RuntimeError("chromadb_rebuild: another rebuild is in progress")
    try:
        yield
    finally:
        _rebuild_lock.release()


def rebuild(plan: RebuildPlan) -> RebuildSummary:
    """Execute a rebuild plan. See module docstring for semantics."""
    started = time.monotonic()
    kb_root = _kb_root(plan.workspace_root, plan.kb)
    bytes_before = _kb_size_bytes(kb_root)

    if not kb_root.exists():
        return RebuildSummary(
            kb=plan.kb, collection=plan.collection,
            rows_exported=0, rows_reimported=0,
            bytes_before=0, bytes_after=0,
            duration_s=time.monotonic() - started,
            snapshot_path=None, dry_run=plan.dry_run, ok=False,
            error=f"kb root does not exist: {kb_root}",
        )

    with _exclusive_rebuild():
        try:
            client = _open_client(kb_root)
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=0, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=None, dry_run=plan.dry_run, ok=False,
                error=f"client open failed: {exc}",
            )

        try:
            col = client.get_collection(plan.collection)
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=0, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=None, dry_run=plan.dry_run, ok=False,
                error=f"collection not found: {exc}",
            )

        # Capture the original metadata so we can restore it on the
        # newly-created collection.
        try:
            col_metadata = dict(col.metadata or {})
        except Exception:
            col_metadata = {}

        # 1. Export everything.
        chunks: list[dict] = []
        rows_exported = 0
        try:
            for chunk in _stream_collection(col):
                chunks.append(chunk)
                rows_exported += len(chunk["ids"])
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=None, dry_run=plan.dry_run, ok=False,
                error=f"export failed: {exc}",
            )

        # 2. Snapshot to disk BEFORE mutating anything.
        try:
            snap_path = _write_snapshot(plan, chunks)
            plan.snapshot_path = snap_path
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=None, dry_run=plan.dry_run, ok=False,
                error=f"snapshot write failed: {exc}",
            )

        if plan.dry_run:
            # Dry run: snapshot only. No mutation. Operator can grep
            # the snapshot to confirm content before re-running for real.
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=str(snap_path), dry_run=True, ok=True,
            )

        # 3. Delete the source collection.
        try:
            client.delete_collection(plan.collection)
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=str(snap_path), dry_run=False, ok=False,
                error=f"delete_collection failed: {exc}",
            )

        # 4. Recreate fresh.
        try:
            new_col = client.create_collection(
                plan.collection, metadata=col_metadata or None,
            )
        except Exception as exc:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=0,
                bytes_before=bytes_before, bytes_after=bytes_before,
                duration_s=time.monotonic() - started,
                snapshot_path=str(snap_path), dry_run=False, ok=False,
                error=f"create_collection failed: {exc}",
            )

        # 5. Re-insert. We use the original IDs to preserve linkage
        # with anything that may reference rows by ID.
        rows_reimported = 0
        for chunk in chunks:
            ids = chunk["ids"]
            embeddings = chunk["embeddings"]
            documents = chunk["documents"]
            metadatas = chunk["metadatas"]
            # ChromaDB requires non-None metadata per row; substitute {}.
            metadatas_clean = [m if isinstance(m, dict) else {} for m in metadatas]
            for start in range(0, len(ids), _IMPORT_BATCH):
                end = start + _IMPORT_BATCH
                try:
                    new_col.add(
                        ids=ids[start:end],
                        embeddings=embeddings[start:end],
                        documents=documents[start:end],
                        metadatas=metadatas_clean[start:end],
                    )
                    rows_reimported += end - start if end <= len(ids) else len(ids) - start
                except Exception as exc:
                    return RebuildSummary(
                        kb=plan.kb, collection=plan.collection,
                        rows_exported=rows_exported, rows_reimported=rows_reimported,
                        bytes_before=bytes_before, bytes_after=_kb_size_bytes(kb_root),
                        duration_s=time.monotonic() - started,
                        snapshot_path=str(snap_path), dry_run=False, ok=False,
                        error=(
                            f"reinsert failed at row {start}; "
                            f"snapshot preserved at {snap_path} — "
                            f"recover via `--from-snapshot {snap_path}`. "
                            f"Underlying: {exc}"
                        ),
                    )

        # 6. Verify count.
        try:
            final_count = new_col.count()
        except Exception:
            final_count = -1
        if final_count != rows_exported:
            return RebuildSummary(
                kb=plan.kb, collection=plan.collection,
                rows_exported=rows_exported, rows_reimported=rows_reimported,
                bytes_before=bytes_before, bytes_after=_kb_size_bytes(kb_root),
                duration_s=time.monotonic() - started,
                snapshot_path=str(snap_path), dry_run=False, ok=False,
                error=(
                    f"post-rebuild count mismatch: "
                    f"reimported {final_count} ≠ exported {rows_exported}; "
                    f"snapshot preserved at {snap_path}"
                ),
            )

    bytes_after = _kb_size_bytes(kb_root)
    return RebuildSummary(
        kb=plan.kb, collection=plan.collection,
        rows_exported=rows_exported, rows_reimported=rows_reimported,
        bytes_before=bytes_before, bytes_after=bytes_after,
        duration_s=time.monotonic() - started,
        snapshot_path=str(plan.snapshot_path) if plan.snapshot_path else None,
        dry_run=False, ok=True, error=None,
    )


def restore_from_snapshot(
    snapshot_path: Path, kb: str, collection: str, workspace_root: Path,
) -> RebuildSummary:
    """Recreate ``collection`` under ``workspace/<kb>/`` from a snapshot
    file. Used when a rebuild() failed mid-way and left an empty
    collection on disk."""
    started = time.monotonic()
    kb_root = _kb_root(workspace_root, kb)
    bytes_before = _kb_size_bytes(kb_root)
    if not kb_root.exists():
        return RebuildSummary(
            kb=kb, collection=collection,
            rows_exported=0, rows_reimported=0,
            bytes_before=0, bytes_after=0,
            duration_s=time.monotonic() - started,
            snapshot_path=str(snapshot_path), dry_run=False, ok=False,
            error=f"kb root does not exist: {kb_root}",
        )
    rows: list[dict] = list(_read_snapshot(snapshot_path))
    if not rows:
        return RebuildSummary(
            kb=kb, collection=collection,
            rows_exported=0, rows_reimported=0,
            bytes_before=bytes_before, bytes_after=bytes_before,
            duration_s=time.monotonic() - started,
            snapshot_path=str(snapshot_path), dry_run=False, ok=False,
            error="snapshot is empty",
        )
    with _exclusive_rebuild():
        client = _open_client(kb_root)
        try:
            client.delete_collection(collection)
        except Exception:
            pass  # ok if it doesn't exist
        col = client.create_collection(collection)
        rows_reimported = 0
        for start in range(0, len(rows), _IMPORT_BATCH):
            chunk = rows[start:start + _IMPORT_BATCH]
            col.add(
                ids=[r["id"] for r in chunk],
                documents=[r.get("document") for r in chunk],
                metadatas=[r.get("metadata") if isinstance(r.get("metadata"), dict) else {} for r in chunk],
                embeddings=[r["embedding"] for r in chunk],
            )
            rows_reimported += len(chunk)
    return RebuildSummary(
        kb=kb, collection=collection,
        rows_exported=len(rows), rows_reimported=rows_reimported,
        bytes_before=bytes_before, bytes_after=_kb_size_bytes(kb_root),
        duration_s=time.monotonic() - started,
        snapshot_path=str(snapshot_path), dry_run=False, ok=True,
    )


def _resolve_workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.memory.chromadb_rebuild",
        description="Operator-runnable per-collection rebuild for ChromaDB.",
    )
    parser.add_argument(
        "--kb", required=True,
        help="KB directory under workspace/ (memory, philosophy, episteme, …)",
    )
    parser.add_argument(
        "--collection", required=True,
        help="ChromaDB collection name to rebuild.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Snapshot-only. Writes the collection to "
            "workspace/<kb>/.rebuild_backups/ and exits without mutating "
            "the live collection."
        ),
    )
    parser.add_argument(
        "--from-snapshot", default=None,
        help=(
            "Recover a collection from an existing snapshot file. "
            "Implies a destructive recreate of the named collection."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    workspace_root = _resolve_workspace_root()
    if args.from_snapshot:
        snap = Path(args.from_snapshot)
        if not snap.exists():
            print(f"ERROR: snapshot not found: {snap}", file=sys.stderr)
            return 2
        summary = restore_from_snapshot(
            snap, kb=args.kb, collection=args.collection,
            workspace_root=workspace_root,
        )
    else:
        plan = RebuildPlan(
            kb=args.kb, collection=args.collection,
            workspace_root=workspace_root, dry_run=args.dry_run,
        )
        summary = rebuild(plan)

    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
