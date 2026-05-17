"""
source_ledger.py — Per-KB append-only canonical source of truth for chromadb.

PROGRAM §56 (2026-05-17) — 10-year resiliency mechanism. Built on top
of §55 (which closed the dual-writer corruption); §56 is the layer
that guarantees **every chromadb KB is reconstructable** from a plain
JSONL file even if the chromadb files are lost entirely.

The big idea
============

ChromaDB stores two things on disk:

  * ``chroma.sqlite3``    — collection metadata + document text + the
                            doc_id index (recoverable, but format-
                            specific and version-coupled).
  * ``<uuid>/``           — HNSW segment files holding the actual
                            embedding vectors (binary, opaque).

Both are **derived** from the input text + the current embedding
model. If we keep an append-only log of every ``(collection, doc_id,
text, metadata)`` tuple ever stored, we can rebuild both from scratch
at any time:

  1. Read the ledger
  2. Re-embed each row with the *current* embedding model
  3. ``col.upsert(ids=..., documents=..., embeddings=..., metadatas=...)``
  4. Done — KB is identical (modulo embedding-model drift, which is
     exactly what we want when models rotate)

This makes the SQLite + HNSW files purely cacheable. Lose them, lose
nothing. Quarantine on corruption (§55) is now a no-impact event:
the integrity layer detects damage, the replay layer reads the
ledger, the KB is reconstructed.

Storage layout
==============

One JSONL file per KB at ``workspace/<kb>/.source_ledger.jsonl``.

Each line is a JSON object with the schema::

    {
      "ts":         1747512345.123,        # epoch seconds, write order
      "collection": "beliefs",             # logical collection within the KB
      "doc_id":     "uuid-string",         # the chromadb id; replay-idempotent
      "text":       "...",                 # the indexed text (verbatim)
      "metadata":   {...},                 # what was passed to col.add
      "prev_hash":  "<64-hex>",            # hash of the prior row; genesis = "0"*64
      "hash":       "<64-hex>"             # sha256 of prev_hash + canonical(payload)
    }

Hash chaining
-------------

The hash field is ``sha256(prev_hash + canonical_json(row_without_hash_fields))``.
"Canonical" = ``json.dumps(..., sort_keys=True, separators=(",", ":"))``
so the bytes are deterministic regardless of insertion order in the
dict. The chain is verifiable: walk forward, recompute, compare. Any
mismatch indicates tampering or bit-rot.

This matches the existing project pattern (audit.log,
coding-session audit, change-request audit, continuity_ledger are
all hash-chained JSONL).

Off-host destinations
=====================

The ledger is automatically replicated by Q17.1's warm-spare manifest
(which already covers ``workspace/``). PROGRAM §56 adds two more
optional destinations: S3 + Google Drive (see
``app/memory/source_ledger_offhost/``). Both off by default; turn on
when operator wires credentials.

Failure modes
=============

Every public function is failure-isolated. If the ledger write fails
during a chromadb store, the chromadb write still succeeds — the
ledger is best-effort. The next bootstrap pass will catch the gap
because the doc_id exists in chromadb but not in the ledger.

Replay is idempotent on doc_id, so re-running it many times is safe.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────


LEDGER_FILENAME = ".source_ledger.jsonl"
GENESIS_HASH = "0" * 64
_REPLAY_BATCH = 32  # rows per chromadb upsert call during replay


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def ledger_path(kb_name: str) -> Path:
    """Where the ledger lives for a given KB."""
    return _workspace_root() / kb_name / LEDGER_FILENAME


# ── Hashing ──────────────────────────────────────────────────────────────


def _canonical_json(payload: dict) -> str:
    """Deterministic JSON encoding — the same Python dict must produce
    the same bytes regardless of insertion order."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _compute_hash(prev_hash: str, payload: dict) -> str:
    """Hash chain link: sha256(prev_hash + canonical_json(payload))."""
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical_json(payload).encode("utf-8"))
    return h.hexdigest()


# ── Row dataclass ────────────────────────────────────────────────────────


# Ledger operation types. Rows without an ``op`` field (pre-§56-iter-2
# data) default to ``add`` — the only operation the original schema
# supported. ``update`` carries new text and/or new metadata; ``delete``
# is a tombstone with no text/metadata.
OP_ADD = "add"
OP_UPDATE = "update"
OP_DELETE = "delete"
_VALID_OPS = (OP_ADD, OP_UPDATE, OP_DELETE)

# §56 iter-3 (2026-05-17) — sentinel that distinguishes "update did
# not touch metadata" from "update cleared metadata to {}". When the
# caller of ``append_update`` passes ``new_metadata=None``, the ledger
# stores this sentinel as the metadata dict; replay/compaction detect
# it and preserve the prior metadata instead of overwriting with empty.
# Using a key with a project-namespaced prefix so it never collides
# with operator-provided metadata.
_META_NO_CHANGE_SENTINEL = {"__sl_no_change__": True}


def _is_no_change_sentinel(meta: Optional[dict]) -> bool:
    """Return True iff ``meta`` is the no-change sentinel.

    Strict equality check (length 1 + the one expected key with True
    value). Avoids matching operator metadata that happens to contain
    the sentinel key alongside others.
    """
    return (
        isinstance(meta, dict)
        and len(meta) == 1
        and meta.get("__sl_no_change__") is True
    )


@dataclass
class LedgerRow:
    ts: float
    collection: str
    doc_id: str
    text: str
    metadata: dict
    prev_hash: str
    hash: str
    op: str = OP_ADD  # backward compat — pre-tombstone rows are adds

    @classmethod
    def from_json(cls, line: str) -> "LedgerRow":
        d = json.loads(line)
        return cls(
            ts=float(d["ts"]),
            collection=str(d["collection"]),
            doc_id=str(d["doc_id"]),
            text=str(d.get("text") or ""),
            metadata=dict(d.get("metadata") or {}),
            prev_hash=str(d.get("prev_hash") or GENESIS_HASH),
            hash=str(d.get("hash") or ""),
            op=str(d.get("op") or OP_ADD),
        )

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "op": self.op,
            "collection": self.collection,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }

    @property
    def payload_for_hash(self) -> dict:
        """The hashed payload excludes the hash itself (chicken/egg) but
        includes everything else including prev_hash — that's how the
        chain stays tamper-evident at every link.

        For backward compatibility, pre-tombstone rows (``op=="add"``
        when ``op`` wasn't in the original JSON) must hash identically
        to the original schema — we omit ``op`` from the hashed payload
        when it's the default. Only update/delete rows include it.
        """
        payload = {
            "ts": self.ts,
            "collection": self.collection,
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
        }
        if self.op != OP_ADD:
            payload["op"] = self.op
        return payload


# ── Append ───────────────────────────────────────────────────────────────


def _last_hash(path: Path) -> str:
    """Read the last line of the ledger and return its hash. Returns
    GENESIS_HASH if the file is missing or empty.

    We seek to the end and walk backward 4 KB at a time looking for a
    newline. Most ledgers are small (<10 MB) so a full read is also
    cheap — we choose the seek strategy because it's O(1) regardless
    of ledger size, which is important for long-term operation.
    """
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH
    try:
        size = path.stat().st_size
        chunk = min(4096, size)
        with path.open("rb") as f:
            f.seek(max(0, size - chunk))
            tail = f.read().decode("utf-8", errors="replace")
        # Find the last non-empty line in tail. If the ledger is larger
        # than chunk and the last line isn't contained in it, we walk
        # back more — rare path; small files just succeed first try.
        lines = [ln for ln in tail.split("\n") if ln.strip()]
        if not lines:
            return GENESIS_HASH
        last = lines[-1]
        try:
            return str(json.loads(last).get("hash") or GENESIS_HASH)
        except Exception:
            # Truncated last line — read the whole file.
            pass
        with path.open("r") as f:
            for line in f:
                pass
            try:
                return str(json.loads(line).get("hash") or GENESIS_HASH)
            except Exception:
                return GENESIS_HASH
    except Exception:
        logger.debug("source_ledger: _last_hash failed for %s", path, exc_info=True)
        return GENESIS_HASH


def _safe_append(path: Path, line: str) -> None:
    """Atomic-append a line. Falls back from app.safe_io.safe_append
    (POSIX flock + fsync) to plain Python open(..., 'a') if safe_io
    isn't importable (tests / stripped builds)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from app.safe_io import safe_append  # type: ignore
        safe_append(path, line)
        return
    except Exception:
        pass
    # Fallback — atomic enough for tests; safe_io adds the flock+fsync
    # for production single-writer guarantees.
    with path.open("a", encoding="utf-8") as f:
        if not line.endswith("\n"):
            line = line + "\n"
        f.write(line)


def _append_op(kb_name: str, op: str, collection: str, doc_id: str,
               text: str = "", metadata: Optional[dict] = None,
               ts: Optional[float] = None) -> Optional[LedgerRow]:
    """Generic op-append. Called by ``append_row`` (add), ``append_update``,
    and ``append_delete``. Hash chain stays valid across all three op
    types because the ``op`` field is part of the hashed payload for
    non-add operations.

    Failure-isolated: any error is logged and the function returns
    ``None`` — callers must not be impacted by ledger I/O failures.
    """
    if op not in _VALID_OPS:
        logger.warning("source_ledger: invalid op %s for kb=%s", op, kb_name)
        return None
    try:
        path = ledger_path(kb_name)
        prev = _last_hash(path)
        ts_v = float(ts) if ts is not None else time.time()
        payload = {
            "ts": ts_v,
            "collection": collection,
            "doc_id": doc_id,
            "text": text,
            "metadata": dict(metadata or {}),
            "prev_hash": prev,
        }
        # Backward-compat: op is included in the hashed payload only
        # when it's not the default ``add``. This way pre-tombstone
        # ``add`` rows hash identically under both the old and new
        # schema. See ``LedgerRow.payload_for_hash`` for the symmetric
        # read path.
        if op != OP_ADD:
            payload["op"] = op
        h = _compute_hash(prev, payload)
        row = LedgerRow(
            ts=ts_v, collection=collection, doc_id=doc_id, text=text,
            metadata=dict(metadata or {}), prev_hash=prev, hash=h, op=op,
        )
        line = _canonical_json(row.to_dict())
        _safe_append(path, line + "\n")
        return row
    except Exception:
        logger.debug(
            "source_ledger: _append_op failed kb=%s op=%s collection=%s",
            kb_name, op, collection, exc_info=True,
        )
        return None


def append_row(kb_name: str, collection: str, doc_id: str, text: str,
               metadata: Optional[dict] = None, ts: Optional[float] = None) -> Optional[LedgerRow]:
    """Append an ``add`` row to the KB's ledger. Default for every
    chromadb store call. Hash-chained from the last row on disk.
    """
    return _append_op(
        kb_name, OP_ADD, collection, doc_id, text, metadata, ts,
    )


def append_update(kb_name: str, collection: str, doc_id: str,
                  new_text: Optional[str] = None,
                  new_metadata: Optional[dict] = None,
                  ts: Optional[float] = None) -> Optional[LedgerRow]:
    """Append an ``update`` row. ``new_text`` and ``new_metadata`` are
    independent — pass only the one being changed.

    Metadata semantics (iter-3, 2026-05-17):
      * ``new_metadata=None``  → ledger stores no-change sentinel;
                                   replay preserves prior metadata.
      * ``new_metadata={}``    → ledger stores ``{}``; replay CLEARS
                                   metadata to empty on the rebuild.
      * ``new_metadata={...}`` → ledger stores the dict; replay uses it
                                   as the new metadata (overwriting).

    Text semantics (unchanged): ``new_text=None`` and ``new_text=""``
    both mean "no text change" — replay uses ``new_text or last_text``.
    Empty string for text isn't a meaningful chromadb-document value,
    so coalescing them is fine.
    """
    if new_metadata is None:
        meta_to_store = _META_NO_CHANGE_SENTINEL
    else:
        meta_to_store = new_metadata
    return _append_op(
        kb_name, OP_UPDATE, collection, doc_id,
        text=new_text or "",
        metadata=meta_to_store,
        ts=ts,
    )


def append_delete(kb_name: str, collection: str, doc_id: str,
                  ts: Optional[float] = None) -> Optional[LedgerRow]:
    """Append a ``delete`` tombstone. Replay treats this as terminal
    for ``(collection, doc_id)`` — the doc is dropped from the rebuilt
    KB even if earlier add/update rows exist for it.

    Text and metadata are intentionally empty: the doc is gone, there's
    nothing to preserve except the fact of deletion.
    """
    return _append_op(
        kb_name, OP_DELETE, collection, doc_id, text="", metadata={}, ts=ts,
    )


# ── Bulk-add hook (for KBs whose vectorstores call col.add directly) ─────


def hook_collection_add(kb_name: str, collection: str,
                       ids: list[str], documents: list[str],
                       metadatas: Optional[list[dict]] = None) -> int:
    """Mirror ``col.add(ids=..., documents=..., metadatas=...)`` into the
    ledger. Used by KB-specific vectorstores (episteme, experiential,
    philosophy) that bypass ``chromadb_manager.store``.

    Returns the number of rows successfully appended. Failure-isolated
    per-row — one bad row doesn't break the rest.
    """
    if not ids:
        return 0
    metas = metadatas or [{}] * len(ids)
    n_appended = 0
    for i, doc_id in enumerate(ids):
        text = documents[i] if i < len(documents) else ""
        meta = metas[i] if i < len(metas) else {}
        if not text:
            continue
        row = append_row(kb_name, collection, str(doc_id), text, meta or {})
        if row is not None:
            n_appended += 1
    return n_appended


def hook_collection_delete(kb_name: str, collection: str,
                           ids: list[str]) -> int:
    """Mirror ``col.delete(ids=[...])`` into the ledger as tombstones.

    Replay treats each tombstone as terminal for ``(collection, doc_id)``
    so deleted rows don't get resurrected on rebuild.

    Failure-isolated per-row.
    """
    if not ids:
        return 0
    n_appended = 0
    for doc_id in ids:
        row = append_delete(kb_name, collection, str(doc_id))
        if row is not None:
            n_appended += 1
    return n_appended


def hook_collection_update(kb_name: str, collection: str,
                           ids: list[str],
                           documents: Optional[list[str]] = None,
                           metadatas: Optional[list[dict]] = None) -> int:
    """Mirror ``col.update(ids=..., documents=..., metadatas=...)``.

    Either ``documents`` or ``metadatas`` may be omitted/None — only
    the supplied facets are recorded. Replay folds updates over the
    last add/update per doc_id.

    Failure-isolated per-row.
    """
    if not ids:
        return 0
    n_appended = 0
    for i, doc_id in enumerate(ids):
        new_text = None
        if documents is not None and i < len(documents):
            new_text = documents[i]
        new_meta = None
        if metadatas is not None and i < len(metadatas):
            new_meta = metadatas[i]
        row = append_update(
            kb_name, collection, str(doc_id),
            new_text=new_text, new_metadata=new_meta,
        )
        if row is not None:
            n_appended += 1
    return n_appended


# ── Read ─────────────────────────────────────────────────────────────────


def read_all(kb_name: str) -> Iterator[LedgerRow]:
    """Yield every row in the KB's ledger, in append order."""
    path = ledger_path(kb_name)
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    yield LedgerRow.from_json(raw)
                except Exception:
                    logger.debug(
                        "source_ledger: malformed row in %s — skipping",
                        path, exc_info=True,
                    )
                    continue
    except OSError:
        logger.debug("source_ledger: read_all failed for %s", path, exc_info=True)
        return


def read_since(kb_name: str, since_ts: float) -> Iterator[LedgerRow]:
    """Yield rows with ``ts > since_ts``. Linear scan; ledgers are
    append-only so the rows are already in time order — we just stream
    until ``ts`` rises above the cutoff and then yield the rest.

    The use case is incremental replay: ``since_ts`` is the latest
    timestamp the live KB already contains; this returns the missing
    suffix.
    """
    for row in read_all(kb_name):
        if row.ts > since_ts:
            yield row


def count_rows(kb_name: str) -> int:
    """Cheap row count (line count of the ledger)."""
    path = ledger_path(kb_name)
    if not path.exists():
        return 0
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ── Verify ───────────────────────────────────────────────────────────────


@dataclass
class VerifyResult:
    ok: bool
    rows_seen: int = 0
    first_bad_row: int = -1
    first_bad_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "rows_seen": self.rows_seen,
            "first_bad_row": self.first_bad_row,
            "first_bad_reason": self.first_bad_reason,
        }


def verify_chain(kb_name: str) -> VerifyResult:
    """Walk the ledger and recompute every hash. Returns OK or the
    first violation. The hash chain detects:

      * Row tampering (any field changed → hash mismatch)
      * Insertion (prev_hash linkage broken)
      * Deletion (chain breaks at the deletion point)
      * Reordering (prev_hash references the wrong row)

    Used by the bit_rot_scan extension (Q17.3) and the quarterly
    replay drill.
    """
    prev = GENESIS_HASH
    for i, row in enumerate(read_all(kb_name)):
        if row.prev_hash != prev:
            return VerifyResult(
                ok=False, rows_seen=i, first_bad_row=i,
                first_bad_reason=f"prev_hash_mismatch: expected={prev[:8]} got={row.prev_hash[:8]}",
            )
        expected_hash = _compute_hash(prev, row.payload_for_hash)
        if row.hash != expected_hash:
            return VerifyResult(
                ok=False, rows_seen=i + 1, first_bad_row=i,
                first_bad_reason=f"hash_mismatch: expected={expected_hash[:8]} got={row.hash[:8]}",
            )
        prev = row.hash
    return VerifyResult(ok=True, rows_seen=_count_iter_from(kb_name))


def _count_iter_from(kb_name: str) -> int:
    """Re-iterate to get the count; we already did one pass in
    verify_chain so this is the second pass. Cheap (small ledgers)."""
    return count_rows(kb_name)


# ── Replay ───────────────────────────────────────────────────────────────


@dataclass
class ReplayResult:
    ok: bool
    kb_name: str
    rows_seen: int = 0
    rows_upserted: int = 0
    rows_skipped: int = 0
    collections_seen: list[str] = field(default_factory=list)
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "kb_name": self.kb_name,
            "rows_seen": self.rows_seen,
            "rows_upserted": self.rows_upserted,
            "rows_skipped": self.rows_skipped,
            "collections_seen": self.collections_seen,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }


def replay_kb(kb_name: str, since_ts: Optional[float] = None,
              target_path: Optional[Path] = None,
              max_rows: Optional[int] = None) -> ReplayResult:
    """Reconstruct a KB from its ledger.

    Args:
        kb_name:   The KB whose ledger we read (e.g. "memory").
        since_ts:  If given, only re-embed rows newer than this — for
                   incremental drift recovery. None = full rebuild.
        target_path: If given, write to that chromadb path instead of
                   the live KB. Used by the quarterly drill which
                   rebuilds to a scratch dir for verification. None =
                   live KB at workspace/<kb_name>/.
        max_rows:  Safety cap; None = unbounded.

    Re-embedding uses the *current* embed model (via
    chromadb_manager.embed). This is intentional: if the embedding
    model has rotated since the rows were written, the replay updates
    the index to use the new model. Embedding-rotation-tolerant.

    Idempotent: doc_id-keyed via chromadb's upsert. Re-running on a
    full KB is a no-op (or near-no-op if some metadata was updated).
    """
    started = time.monotonic()
    result = ReplayResult(ok=False, kb_name=kb_name)

    if not _gate("chromadb_source_ledger_enabled"):
        result.error = "source_ledger_disabled"
        return result

    try:
        from app.memory import chromadb_manager  # late import
    except Exception as exc:
        result.error = f"chromadb_manager_import: {exc}"
        return result

    # Either use the live client (target_path None) or open a fresh
    # client at the requested path. Drill uses the second path so the
    # rebuild is purely additive — live KB never touched.
    try:
        if target_path is None:
            client = chromadb_manager.get_kb_client(kb_name)
        else:
            import chromadb  # type: ignore
            target_path = Path(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(target_path))
    except Exception as exc:
        result.error = f"client_open: {exc}"
        return result

    # Fold add → update → delete chains per (collection, doc_id) so we
    # end up with one "final state" per surviving doc. Deleted docs
    # are skipped entirely. Updates merge text/metadata on top of the
    # latest add for that doc.
    #
    # State shape: state[(collection, doc_id)] = LedgerRow representing
    # the doc's final state after all observed ops. None means "deleted
    # — skip this doc in the replay output."
    state: dict[tuple[str, str], Optional[LedgerRow]] = {}
    rows_iter = read_since(kb_name, since_ts) if since_ts is not None else read_all(kb_name)

    for row in rows_iter:
        result.rows_seen += 1
        if max_rows is not None and result.rows_seen > max_rows:
            break
        key = (row.collection, row.doc_id)
        if row.op == OP_ADD:
            state[key] = row
        elif row.op == OP_UPDATE:
            prior = state.get(key)
            if prior is None:
                # Update with no prior add — skip; we can't reconstruct
                # the original text. This is the legitimate case where
                # the add row was deleted from the ledger or never
                # captured (rare). Counted as skipped.
                result.rows_skipped += 1
                continue
            merged_text = row.text or prior.text
            # Metadata semantics (iter-3):
            #   * no-change sentinel  → keep prior metadata
            #   * explicit {}         → clear metadata
            #   * any other dict      → overwrite with the new dict
            if _is_no_change_sentinel(row.metadata):
                merged_meta = prior.metadata
            else:
                merged_meta = dict(row.metadata or {})
            state[key] = LedgerRow(
                ts=row.ts, op=OP_ADD,  # fold back to add for replay
                collection=row.collection, doc_id=row.doc_id,
                text=merged_text, metadata=merged_meta,
                prev_hash=row.prev_hash, hash=row.hash,
            )
        elif row.op == OP_DELETE:
            # Terminal — doc is gone, drop from rebuild output.
            state[key] = None
        # Unknown op (forward-compat): leave state alone.

    # Build per-collection batches from the folded state.
    by_col: dict[str, list[LedgerRow]] = {}
    for (col_name, _doc_id), row in state.items():
        if row is None:
            continue  # deleted doc — skip
        if not row.text or not row.text.strip():
            result.rows_skipped += 1
            continue
        by_col.setdefault(col_name, []).append(row)

    # Now flush each collection in batches.
    for col_name, rows in by_col.items():
        if col_name not in result.collections_seen:
            result.collections_seen.append(col_name)
        try:
            col = client.get_or_create_collection(col_name)
        except Exception as exc:
            logger.warning(
                "source_ledger.replay_kb: collection %s open failed: %s",
                col_name, exc,
            )
            result.rows_skipped += len(rows)
            continue

        for batch_start in range(0, len(rows), _REPLAY_BATCH):
            batch = rows[batch_start : batch_start + _REPLAY_BATCH]
            try:
                ids = [r.doc_id for r in batch]
                docs = [r.text for r in batch]
                metas = [r.metadata or {} for r in batch]
                # Compute embeddings in a single pass using the live
                # backend. We could let chromadb auto-embed but the
                # gateway has a curated embed function with metal
                # acceleration + caching — use it.
                embeddings = []
                for r in batch:
                    try:
                        embeddings.append(chromadb_manager.embed(r.text))
                    except Exception as exc:
                        # Drop this row from the batch; embed errors
                        # should not nuke the whole replay.
                        logger.debug(
                            "source_ledger.replay_kb: embed failed for doc_id=%s: %s",
                            r.doc_id, exc,
                        )
                        embeddings.append(None)
                # Filter rows where embed failed.
                keep = [(i, e) for i, e in enumerate(embeddings) if e is not None]
                if not keep:
                    result.rows_skipped += len(batch)
                    continue
                kept_ids = [ids[i] for i, _ in keep]
                kept_docs = [docs[i] for i, _ in keep]
                kept_metas = [metas[i] for i, _ in keep]
                kept_embeds = [e for _, e in keep]
                # upsert is idempotent on id — exactly what we want.
                col.upsert(
                    ids=kept_ids,
                    documents=kept_docs,
                    metadatas=kept_metas,
                    embeddings=kept_embeds,
                )
                result.rows_upserted += len(kept_ids)
                result.rows_skipped += len(batch) - len(kept_ids)
            except Exception as exc:
                logger.warning(
                    "source_ledger.replay_kb: batch upsert failed kb=%s col=%s: %s",
                    kb_name, col_name, exc,
                )
                result.rows_skipped += len(batch)
                continue

    result.duration_s = time.monotonic() - started
    result.ok = True
    return result


# ── Bootstrap (back-fill ledger from existing chromadb contents) ─────────


def bootstrap_kb(kb_name: str, max_rows: Optional[int] = None) -> dict:
    """One-time back-fill: walks every collection in the KB, dumps all
    rows to the ledger if not already present.

    This is for migrating existing data into the new ledger primitive.
    After running once per KB, normal dual-writes keep the ledger
    current — bootstrap is then a no-op.

    Idempotent on doc_id. Safe to interrupt and re-run.

    Failure-isolated; returns a dict with stats + any per-collection
    errors.
    """
    info: dict[str, Any] = {
        "kb_name": kb_name,
        "ok": False,
        "rows_added": 0,
        "rows_already_present": 0,
        "collections": {},
        "error": None,
    }

    if not _gate("chromadb_source_ledger_enabled"):
        info["error"] = "source_ledger_disabled"
        return info
    if not _gate("chromadb_ledger_bootstrap_enabled"):
        info["error"] = "bootstrap_disabled"
        return info

    # Build a set of (collection, doc_id) we already have in the
    # ledger so we can skip duplicates fast.
    seen: set[tuple[str, str]] = set()
    for row in read_all(kb_name):
        seen.add((row.collection, row.doc_id))

    try:
        from app.memory import chromadb_manager  # type: ignore
    except Exception as exc:
        info["error"] = f"chromadb_manager_import: {exc}"
        return info

    try:
        client = chromadb_manager.get_kb_client(kb_name)
        collections = client.list_collections()
    except Exception as exc:
        info["error"] = f"list_collections: {exc}"
        return info

    for col_obj in collections:
        col_name = getattr(col_obj, "name", None) or str(col_obj)
        col_info = {"added": 0, "skipped": 0, "error": None}
        info["collections"][col_name] = col_info
        try:
            # Use get() with no filter to fetch all rows. For very
            # large collections this can be memory-heavy; cap at a
            # reasonable maximum per pass (the daemon re-runs daily).
            data = col_obj.get(include=["documents", "metadatas"])
        except Exception as exc:
            col_info["error"] = f"get_failed: {exc}"
            continue
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or [{}] * len(ids)
        for i, doc_id in enumerate(ids):
            if (col_name, doc_id) in seen:
                col_info["skipped"] += 1
                info["rows_already_present"] += 1
                continue
            text = docs[i] if i < len(docs) else ""
            metadata = metas[i] if i < len(metas) else {}
            if not text:
                # No text to re-embed from — skip; the row is
                # unrecoverable anyway.
                col_info["skipped"] += 1
                continue
            row = append_row(
                kb_name, col_name, doc_id, text, metadata or {},
            )
            if row is not None:
                col_info["added"] += 1
                info["rows_added"] += 1
                seen.add((col_name, doc_id))
            if max_rows is not None and info["rows_added"] >= max_rows:
                break
        if max_rows is not None and info["rows_added"] >= max_rows:
            break

    info["ok"] = True
    return info


# ── Drift detection ──────────────────────────────────────────────────────


@dataclass
class DriftResult:
    kb_name: str
    ledger_rows: int = 0
    kb_rows_by_collection: dict[str, int] = field(default_factory=dict)
    kb_rows_total: int = 0
    ledger_rows_by_collection: dict[str, int] = field(default_factory=dict)
    drift_pct: float = 0.0
    needs_replay: bool = False
    direction: str = ""  # "kb_short" (KB missing rows) | "ledger_short" | "in_sync"

    def to_dict(self) -> dict:
        return {
            "kb_name": self.kb_name,
            "ledger_rows": self.ledger_rows,
            "kb_rows_by_collection": self.kb_rows_by_collection,
            "kb_rows_total": self.kb_rows_total,
            "ledger_rows_by_collection": self.ledger_rows_by_collection,
            "drift_pct": round(self.drift_pct, 4),
            "needs_replay": self.needs_replay,
            "direction": self.direction,
        }


# 5% drift threshold — below this we don't take corrective action. Drift
# is normal during a high-write window (writes go to chromadb first,
# ledger slightly after on a slow disk). 5% is also above the noise
# from in-flight transactions.
DRIFT_REPLAY_THRESHOLD = 0.05


def check_drift(kb_name: str) -> DriftResult:
    """Compare KB row count vs ledger row count.

    KB short = chromadb has fewer rows than the ledger does → the
    chromadb file was rebuilt, lost rows; replay to reconstruct.
    Ledger short = chromadb has more rows than the ledger → the
    ledger is missing rows; bootstrap will catch it.

    The drill triggers replay only when KB is short (i.e. data was
    lost). Ledger-short is repaired by bootstrap, not replay.
    """
    result = DriftResult(kb_name=kb_name)
    result.ledger_rows = count_rows(kb_name)

    # Per-collection ledger counts.
    for row in read_all(kb_name):
        result.ledger_rows_by_collection[row.collection] = (
            result.ledger_rows_by_collection.get(row.collection, 0) + 1
        )

    try:
        from app.memory import chromadb_manager  # type: ignore
        client = chromadb_manager.get_kb_client(kb_name)
        for col_obj in client.list_collections():
            try:
                cnt = col_obj.count()
            except Exception:
                cnt = 0
            name = getattr(col_obj, "name", None) or str(col_obj)
            result.kb_rows_by_collection[name] = int(cnt)
            result.kb_rows_total += int(cnt)
    except Exception:
        logger.debug("source_ledger.check_drift: kb introspect failed", exc_info=True)

    if result.ledger_rows == 0 and result.kb_rows_total == 0:
        result.direction = "in_sync"
        return result
    # Compute drift in the direction "ledger has more than KB" — the
    # case where replay would help. Negative drift = KB has more rows
    # than ledger, which means bootstrap (not replay) is the fix.
    if result.ledger_rows == 0:
        result.drift_pct = 0.0
        result.direction = "ledger_short"
        return result
    if result.kb_rows_total > result.ledger_rows:
        result.direction = "ledger_short"
        result.drift_pct = (result.kb_rows_total - result.ledger_rows) / max(1, result.ledger_rows)
        return result
    if result.kb_rows_total < result.ledger_rows:
        result.direction = "kb_short"
        result.drift_pct = (result.ledger_rows - result.kb_rows_total) / max(1, result.ledger_rows)
        if result.drift_pct >= DRIFT_REPLAY_THRESHOLD:
            result.needs_replay = True
        return result
    result.direction = "in_sync"
    return result


# ── KB discovery ─────────────────────────────────────────────────────────


def list_kbs() -> list[str]:
    """All chromadb-managed KBs under the workspace root.

    Same filter as ``chromadb_integrity.chromadb_kbs`` — skips
    quarantined snapshots, derived backup dirs, etc.
    """
    root = _workspace_root()
    if not root.exists():
        return []
    out: list[str] = []
    for p in root.glob("*/chroma.sqlite3"):
        parent = p.parent
        if any(seg in parent.name for seg in ("corrupt_", "bak_", "_backup", ".backup")):
            continue
        out.append(parent.name)
    return sorted(out)


# ── Compaction ───────────────────────────────────────────────────────────


_HISTORY_DIRNAME = ".source_ledger_history"
_COMPACTION_MIN_ROWS = 100
_COMPACTION_MIN_REDUCTION_PCT = 0.20
_COMPACTION_MAX_TAIL_PASSES = 3  # iter-3: race-window tail-stabilization


@dataclass
class CompactionResult:
    ok: bool
    kb_name: str
    rows_before: int = 0
    rows_after: int = 0
    rows_dropped: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    history_path: str = ""
    skipped_reason: str = ""
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "kb_name": self.kb_name,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "rows_dropped": self.rows_dropped,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "history_path": self.history_path,
            "skipped_reason": self.skipped_reason,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }


def _apply_row_to_state(state: dict, row: LedgerRow) -> None:
    """Apply one ledger row to a per-doc folded state in place.

    Same semantics as the inline fold inside ``compact_ledger`` — kept
    as a helper so the tail-stabilization loop can apply tail rows
    DIRECTLY onto the base state (orphan updates can find their prior
    add in the base; deletes propagate correctly).

    Update metadata handling (iter-3): the no-change sentinel preserves
    prior metadata; an explicit ``{}`` clears; anything else overwrites.
    """
    key = (row.collection, row.doc_id)
    if row.op == OP_ADD:
        state[key] = (row.text, dict(row.metadata or {}), row.ts)
    elif row.op == OP_UPDATE:
        prior = state.get(key)
        if prior is None:
            return  # orphan update — no prior add (within full ledger)
        merged_text = row.text or prior[0]
        if _is_no_change_sentinel(row.metadata):
            merged_meta = prior[1]
        else:
            merged_meta = dict(row.metadata or {})
        state[key] = (merged_text, merged_meta, row.ts)
    elif row.op == OP_DELETE:
        state[key] = None


def _fold_ledger_from(ledger: Path, start_offset: int,
                     base_state: Optional[dict] = None) -> tuple[dict, int]:
    """Fold ledger rows starting at byte offset ``start_offset``.

    Returns ``(state, rows_seen)``. When ``base_state`` is supplied
    the rows are applied onto it directly (so a tail update can find
    its prior add in the head); otherwise a fresh state dict is
    built. The base_state path is what the tail-stabilization loop
    in ``compact_ledger`` uses.

    Failure-isolated: malformed lines are skipped.
    """
    state = base_state if base_state is not None else {}
    rows = 0
    if not ledger.exists():
        return state, 0
    try:
        with ledger.open("rb") as f:
            f.seek(start_offset)
            for raw in f:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    row = LedgerRow.from_json(line)
                except Exception:
                    continue
                rows += 1
                _apply_row_to_state(state, row)
    except OSError:
        pass
    return state, rows


def compact_ledger(kb_name: str, force: bool = False) -> CompactionResult:
    """Fold ``.source_ledger.jsonl`` into its minimum-equivalent form.

    Algorithm (race-safe via tail-stabilization loop):

      1. Record live ledger byte size at entry (``start_size``).
      2. Walk the current ledger in order. Maintain a per-doc_id folded
         state: latest add/update wins; delete removes the entry.
      3. Write the surviving state to a temporary
         ``.source_ledger.jsonl.compacted`` file with a fresh hash
         chain (genesis link, new prev_hash linkage).
      4. **Tail-stabilization** — re-stat the live ledger. If it has
         grown since ``start_size`` (a concurrent writer appended
         rows while we were folding), read the new tail rows, apply
         their ops onto the in-memory state, and rewrite the
         compacted file. Loop up to ``_COMPACTION_MAX_TAIL_PASSES``.
         Each pass shrinks the race window; in practice 1-2 passes
         suffice for any realistic write rate.
      5. Snapshot the live ledger into
         ``.source_ledger_history/<UTC-ts>.jsonl`` so the full
         pre-compaction history is recoverable on demand.
      6. Atomic-rename ``.compacted`` over the live ledger.

    Without the tail-stabilization loop, writes appended between
    steps 1 and 6 would be present in the history snapshot but
    missing from the new live ledger — replay would silently lose
    them until the next backfill from history. The loop closes the
    race without needing cross-process write locks.

    Gates:
      * ``force=False`` (default) skips when the ledger has fewer than
        ``_COMPACTION_MIN_ROWS`` rows OR folding would reduce row count
        by less than ``_COMPACTION_MIN_REDUCTION_PCT``.
      * Disabled entirely when the source-ledger master switch is OFF.

    Hash chain semantics: compacted ledger uses a NEW genesis link.
    External verifiers should treat the history file as the authority
    for the pre-compaction chain.

    Returns a ``CompactionResult``. Failure-isolated end-to-end.
    """
    started = time.monotonic()
    result = CompactionResult(ok=False, kb_name=kb_name)

    if not _gate("chromadb_source_ledger_enabled"):
        result.skipped_reason = "source_ledger_disabled"
        return result

    ledger = ledger_path(kb_name)
    if not ledger.exists():
        result.skipped_reason = "ledger_missing"
        return result

    try:
        result.bytes_before = ledger.stat().st_size
        result.rows_before = count_rows(kb_name)
        if not force and result.rows_before < _COMPACTION_MIN_ROWS:
            result.skipped_reason = "below_min_rows"
            result.ok = True
            return result

        # ── Fold pass (initial) ─────────────────────────────────
        # Record the byte offset we read up to so the tail-
        # stabilization loop knows where to pick up if writers
        # appended rows during the fold.
        snapshot_size = ledger.stat().st_size
        state, rows_consumed = _fold_ledger_from(ledger, start_offset=0)

        # ── Tail-stabilization loop ─────────────────────────────
        # If the live ledger grew while we were folding, apply the
        # new tail rows DIRECTLY onto the base state (so tail updates
        # can find their prior adds from the head). Loop until the
        # ledger size stops changing between reads, bounded by
        # ``_COMPACTION_MAX_TAIL_PASSES`` to avoid livelock under a
        # sustained write storm.
        tail_passes = 0
        for _ in range(_COMPACTION_MAX_TAIL_PASSES):
            current_size = ledger.stat().st_size
            if current_size == snapshot_size:
                break  # ledger is stable
            _, tail_consumed = _fold_ledger_from(
                ledger, start_offset=snapshot_size, base_state=state,
            )
            rows_consumed += tail_consumed
            snapshot_size = current_size
            tail_passes += 1

        result.rows_before = rows_consumed  # actual rows folded

        surviving = [
            (col, doc_id, val)
            for (col, doc_id), val in state.items()
            if val is not None and val[0] and val[0].strip()
        ]
        result.rows_after = len(surviving)
        result.rows_dropped = result.rows_before - result.rows_after

        if not force and result.rows_before > 0:
            reduction = result.rows_dropped / result.rows_before
            if reduction < _COMPACTION_MIN_REDUCTION_PCT:
                result.skipped_reason = (
                    f"below_min_reduction ({reduction:.2%} < "
                    f"{_COMPACTION_MIN_REDUCTION_PCT:.2%})"
                )
                result.ok = True
                return result

        # Stable order so re-compacting an already-compacted ledger
        # produces a byte-identical file.
        surviving.sort(key=lambda x: (x[0], x[1]))

        compacted = ledger.with_suffix(ledger.suffix + ".compacted")
        prev = GENESIS_HASH
        with compacted.open("w", encoding="utf-8") as f:
            for col, doc_id, val in surviving:
                text, meta, ts_v = val
                payload = {
                    "ts": ts_v,
                    "collection": col,
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "prev_hash": prev,
                }
                h = _compute_hash(prev, payload)
                row_dict = {
                    "ts": ts_v,
                    "op": OP_ADD,
                    "collection": col,
                    "doc_id": doc_id,
                    "text": text,
                    "metadata": meta,
                    "prev_hash": prev,
                    "hash": h,
                }
                f.write(_canonical_json(row_dict) + "\n")
                prev = h

        # ── Final tail-stabilization check before swap ──────────
        # If the ledger grew between writing the compacted file
        # and now, re-fold + rewrite once more. Bounded to a
        # single extra pass since the window between fold-and-swap
        # is microseconds wide in practice.
        final_size = ledger.stat().st_size
        if final_size > snapshot_size and tail_passes < _COMPACTION_MAX_TAIL_PASSES:
            _, tail_consumed = _fold_ledger_from(
                ledger, start_offset=snapshot_size, base_state=state,
            )
            rows_consumed += tail_consumed
            snapshot_size = final_size
            # Rewrite compacted with the final state.
            surviving = [
                (col, doc_id, val)
                for (col, doc_id), val in state.items()
                if val is not None and val[0] and val[0].strip()
            ]
            surviving.sort(key=lambda x: (x[0], x[1]))
            prev = GENESIS_HASH
            with compacted.open("w", encoding="utf-8") as f:
                for col, doc_id, val in surviving:
                    text, meta, ts_v = val
                    payload = {
                        "ts": ts_v, "collection": col, "doc_id": doc_id,
                        "text": text, "metadata": meta, "prev_hash": prev,
                    }
                    h = _compute_hash(prev, payload)
                    row_dict = {
                        "ts": ts_v, "op": OP_ADD, "collection": col,
                        "doc_id": doc_id, "text": text, "metadata": meta,
                        "prev_hash": prev, "hash": h,
                    }
                    f.write(_canonical_json(row_dict) + "\n")
                    prev = h
            result.rows_before = rows_consumed
            result.rows_after = len(surviving)
            result.rows_dropped = result.rows_before - result.rows_after

        # Snapshot pre-compaction history BEFORE the atomic swap.
        # Hard-link if possible (no extra disk); fall back to copy.
        history_dir = ledger.parent / _HISTORY_DIRNAME
        history_dir.mkdir(parents=True, exist_ok=True)
        history_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ.jsonl")
        history_file = history_dir / history_name
        try:
            os.link(ledger, history_file)
        except (OSError, AttributeError):
            try:
                import shutil
                shutil.copy2(ledger, history_file)
            except Exception as exc:
                compacted.unlink(missing_ok=True)
                result.error = f"history_archive_failed: {exc}"
                return result
        result.history_path = str(history_file)

        # Atomic swap.
        os.replace(compacted, ledger)
        result.bytes_after = ledger.stat().st_size
        result.ok = True
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.duration_s = time.monotonic() - started
    return result


def list_history(kb_name: str) -> list[Path]:
    """Return historical (pre-compaction) ledger snapshots, newest
    first. Concatenating these in reverse-chronological order with the
    current ledger reconstructs the full pre-compaction chain.
    """
    base = ledger_path(kb_name).parent / _HISTORY_DIRNAME
    if not base.exists():
        return []
    return sorted(base.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


# ── Operator-facing state summary ────────────────────────────────────────


def state_summary() -> dict:
    """One-shot health summary for ``/api/cp/source-ledger/state``.

    Per-KB:
      * ``ledger_rows``           — count of rows on disk
      * ``ledger_bytes``          — file size
      * ``ledger_age_s``          — seconds since last write
      * ``chain_ok``              — verify_chain result
      * ``chain_first_bad_row``   — for triage if chain broken
      * ``compaction_history``    — list of historical snapshots (count + bytes)
      * ``last_compaction_at``    — ts from daemon state file
      * ``offhost_state``         — per-destination ``{last_upload_ts, last_object_key}``

    Failure-isolated — any per-KB exception is captured in the row's
    ``error`` field, not propagated. Read-only; never mutates anything.
    """
    out: dict[str, Any] = {"kbs": []}
    now = time.time()

    # Daemon-tracked compaction state (best-effort).
    last_compactions: dict[str, float] = {}
    try:
        ws_root = _workspace_root()
        comp_state = ws_root / "healing" / "source_ledger_compaction_state.json"
        if comp_state.exists():
            last_compactions = json.loads(comp_state.read_text()) or {}
    except Exception:
        last_compactions = {}

    for kb_name in list_kbs():
        row: dict[str, Any] = {"name": kb_name}
        try:
            ledger = ledger_path(kb_name)
            if ledger.exists():
                stat = ledger.stat()
                row["ledger_rows"] = count_rows(kb_name)
                row["ledger_bytes"] = stat.st_size
                row["ledger_age_s"] = int(now - stat.st_mtime)
            else:
                row["ledger_rows"] = 0
                row["ledger_bytes"] = 0
                row["ledger_age_s"] = None

            # Chain verify is the slowest piece — cap by KB row count
            # so a giant ledger doesn't make this endpoint stall.
            if row["ledger_rows"] < 50000:
                verify = verify_chain(kb_name)
                row["chain_ok"] = verify.ok
                row["chain_first_bad_row"] = verify.first_bad_row
                row["chain_first_bad_reason"] = verify.first_bad_reason
            else:
                # Skip verify on huge ledgers — operators can run the
                # quarterly drill if they want the full check.
                row["chain_ok"] = None
                row["chain_first_bad_row"] = -1
                row["chain_first_bad_reason"] = "skipped_large_ledger"

            history = list_history(kb_name)
            row["history_count"] = len(history)
            row["history_bytes"] = 0
            for h in history:
                try:
                    row["history_bytes"] += h.stat().st_size
                except OSError:
                    pass

            row["last_compaction_at"] = last_compactions.get(kb_name)

            # Off-host state per destination.
            offhost: dict[str, dict] = {}
            for dest in ("s3", "gdrive"):
                state_file = ledger.parent / f".source_ledger_{dest}_state.json"
                if state_file.exists():
                    try:
                        offhost[dest] = json.loads(state_file.read_text())
                    except Exception:
                        offhost[dest] = {"error": "state_parse_failed"}
                else:
                    offhost[dest] = None
            row["offhost"] = offhost
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        out["kbs"].append(row)
    return out


# ── Runtime-settings gates ───────────────────────────────────────────────


def _gate(name: str, default: bool = True) -> bool:
    """Failure-isolated runtime_settings read. Default True so missing
    settings file means protection stays ON."""
    try:
        from app import runtime_settings  # type: ignore
        getter = getattr(runtime_settings, f"get_{name}", None)
        if getter is not None:
            return bool(getter())
        return bool(runtime_settings._ensure_initialized().get(name, default))
    except Exception:
        return default
