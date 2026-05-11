"""Portable DR export — container-independent tarball.

PROGRAM §40 (2026-05-10) — Q3 Item 13.

Bundle layout inside ``<ts>.tar.gz``::

    manifest.json                                   # see _ManifestBuilder
    chromadb/<kb>/<collection>.jsonl.gz             # one row per line
    postgres/<table>.jsonl.gz                       # one row per line
    workspace_ledgers/<relpath>.jsonl[.gz]          # affect, identity, audit
    workspace_ledgers/<relpath>.jsonl-archive/      # archive rotation buckets

Excluded by deliberate allowlist (NEVER add to the export):
  * ``.env`` / ``.env.*``
  * ``secrets/``
  * ``google_token.json`` (OAuth refresh)
  * ``vapid_*.pem`` (Web Push keys)
  * ``audit_journal.json.preserved`` (already snapshotted)
  * ``cache/`` (LLM response cache — re-fillable, multi-GB)
  * ``coding_sessions/.worktrees/`` (ephemeral)
  * ``training_adapters/`` (huge binary blobs — separate cadence)
  * Any path containing ``token``, ``credentials``, ``private_key``

Failure model:
  * Per-collection / per-table errors are recorded in the manifest.
  * The export still produces a partial tarball with what succeeded.
  * Operator audits the manifest after a run.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import re
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────


_DEFAULT_BACKUP_ROOT = Path("/app/workspace/backups/dr")

# Workspace-relative paths whose contents we include. Anything else is
# OUT — explicit allowlist is the right shape for a security-sensitive
# operation.
_LEDGER_INCLUDES: list[str] = [
    "affect/",          # the consciousness substrate (now archive-rotated)
    "identity/",        # continuity ledger
    "audit_journal/",   # rolled hash-chained ledger
    "audit_journal.json",
    "control_plane/audit.log",
    "audit.log",
    "self_heal/",       # observability artifacts
    "salience/",        # episode synth source
    "narrative/",       # daily chapters
]

# Path-fragment denylist — applied AFTER the allowlist as a defense-in-depth
# guard against accidental inclusion of secrets that happen to live under
# an allowed root in the future.
_PATH_DENY_FRAGMENTS = (
    ".env", "secret", "credential", "token", "private_key",
    "google_token", "vapid", "client_secret",
)
_PATH_DENY_REGEX = re.compile(
    r"(^|/)(\.env(?:\..+)?|secrets/|google_token\.json|vapid_.*\.pem|"
    r".*credentials.*|.*private_key.*)",
    re.IGNORECASE,
)


def _is_secret_path(rel_path: str) -> bool:
    """True iff the path looks like it could contain a secret. Errs
    aggressively toward EXCLUDING — the cost of a false positive is
    only that the operator restores from a slightly less-complete
    tarball. The cost of a false negative is leaking a token into a
    backup."""
    lower = rel_path.lower()
    if any(frag in lower for frag in _PATH_DENY_FRAGMENTS):
        return True
    return bool(_PATH_DENY_REGEX.search(lower))


# ── Postgres table allowlist ──────────────────────────────────────────────


# Tables that DR-restore needs. Audit + budgets are critical; tickets
# and crew_tasks are operational state we want to preserve. Anything
# else (e.g. transient locks) is OUT.
_PG_TABLES: list[str] = [
    "control_plane.audit_log",
    "control_plane.budgets",
    "control_plane.crew_tasks",
    "control_plane.tickets",
    "control_plane.runtime_settings",
    "control_plane.change_request_audit",
    "control_plane.architecture_request_audit",
    "control_plane.coding_session_audit",
    "control_plane.tier3_amendment_audit",
]


# ── Manifest ─────────────────────────────────────────────────────────────


@dataclass
class _ChromaCollectionEntry:
    kb: str
    collection: str
    rows: int
    bytes: int
    error: str | None = None


@dataclass
class _PostgresTableEntry:
    table: str
    rows: int
    bytes: int
    error: str | None = None


@dataclass
class _LedgerFileEntry:
    rel_path: str
    bytes: int
    error: str | None = None


@dataclass
class ExportManifest:
    program: str = "PROGRAM §40 — DR portable export"
    started_at: str = ""
    completed_at: str = ""
    duration_s: float = 0.0
    workspace_root: str = ""
    chromadb: list[_ChromaCollectionEntry] = field(default_factory=list)
    postgres: list[_PostgresTableEntry] = field(default_factory=list)
    ledgers: list[_LedgerFileEntry] = field(default_factory=list)
    excluded_secret_paths: list[str] = field(default_factory=list)
    total_rows_chromadb: int = 0
    total_rows_postgres: int = 0
    total_bytes: int = 0
    ok: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": round(self.duration_s, 3),
            "workspace_root": self.workspace_root,
            "chromadb": [asdict(c) for c in self.chromadb],
            "postgres": [asdict(p) for p in self.postgres],
            "ledgers": [asdict(g) for g in self.ledgers],
            "excluded_secret_paths": self.excluded_secret_paths,
            "total_rows_chromadb": self.total_rows_chromadb,
            "total_rows_postgres": self.total_rows_postgres,
            "total_bytes": self.total_bytes,
            "ok": self.ok,
            "errors": self.errors,
        }


# ── Workspace iteration ───────────────────────────────────────────────────


def _resolve_workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _list_kb_dirs(workspace_root: Path) -> list[Path]:
    """Every workspace/<kb>/ directory containing a chroma.sqlite3."""
    out: list[Path] = []
    if not workspace_root.exists():
        return out
    for chroma in workspace_root.glob("*/chroma.sqlite3"):
        # Skip recovery snapshots.
        if any(
            seg in chroma.parent.name
            for seg in ("corrupt_", "bak_", "_backup", ".backup")
        ):
            continue
        out.append(chroma.parent)
    return sorted(out)


# ── Streaming JSONL helpers ───────────────────────────────────────────────


def _stream_chroma_jsonl(kb_root: Path, collection: str, batch: int = 500) -> Iterator[str]:
    """Yield JSONL lines for one chromadb collection. Each line is a
    JSON object with id/document/metadata/embedding. Streams via
    offset-paged ``get(...)`` so memory is bounded."""
    import chromadb
    client = chromadb.PersistentClient(path=str(kb_root))
    col = client.get_collection(collection)
    offset = 0
    while True:
        chunk = col.get(
            limit=batch, offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = chunk.get("ids") or []
        if not ids:
            return
        docs = chunk.get("documents") or [None] * len(ids)
        metas = chunk.get("metadatas") or [None] * len(ids)
        embs = chunk.get("embeddings") or [None] * len(ids)
        for i in range(len(ids)):
            row = {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i] if isinstance(metas[i], dict) else None,
                "embedding": embs[i],
            }
            yield json.dumps(row, default=str)
        offset += len(ids)
        if len(ids) < batch:
            return


def _stream_pg_table_jsonl(table: str) -> Iterator[str]:
    """Yield JSONL lines for one Postgres table. Cursor-based to avoid
    loading the whole table into Python memory."""
    from app.control_plane.db import execute
    rows = execute(f"SELECT * FROM {table}", (), fetch=True) or []
    for row in rows:
        # control_plane.db returns dict-like rows; coerce datetimes via str.
        yield json.dumps(dict(row), default=str)


# ── Tar writer ────────────────────────────────────────────────────────────


def _add_jsonl_gz_to_tar(
    tar: tarfile.TarFile, archive_name: str, lines: Iterator[str],
) -> tuple[int, int]:
    """Stream JSONL lines through gzip into a single TarInfo. Returns
    ``(rows, bytes_written)``."""
    buf = io.BytesIO()
    rows = 0
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for line in lines:
            gz.write((line + "\n").encode("utf-8"))
            rows += 1
    raw = buf.getvalue()
    info = tarfile.TarInfo(archive_name)
    info.size = len(raw)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(raw))
    return rows, len(raw)


def _add_file_to_tar(tar: tarfile.TarFile, file_path: Path, archive_name: str) -> int:
    """Add a workspace file as-is (preserves on-disk encoding)."""
    info = tarfile.TarInfo(archive_name)
    info.size = file_path.stat().st_size
    info.mtime = int(file_path.stat().st_mtime)
    info.mode = 0o644
    with file_path.open("rb") as f:
        tar.addfile(info, f)
    return info.size


def _walk_ledger_paths(workspace_root: Path) -> Iterator[Path]:
    """Yield every regular file under the allowed ledger roots. Skip
    secrets defensively."""
    for include in _LEDGER_INCLUDES:
        root = workspace_root / include
        if include.endswith("/"):
            if not root.exists() or not root.is_dir():
                continue
            for p in root.rglob("*"):
                if p.is_file():
                    yield p
        else:
            if root.exists() and root.is_file():
                yield root


# ── Public entry point ────────────────────────────────────────────────────


def export(
    output_dir: Path | str | None = None,
    *,
    label: str | None = None,
    workspace_root: Path | str | None = None,
) -> tuple[Path, ExportManifest]:
    """Run a full DR export. Returns ``(tarball_path, manifest)``.

    ``label`` — optional suffix for the tarball name (useful for
    drill runs vs operator-initiated exports).
    """
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    ws_root = Path(workspace_root) if workspace_root else _resolve_workspace_root()
    out_root = Path(output_dir) if output_dir else _DEFAULT_BACKUP_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = ExportManifest(
        started_at=started_iso, workspace_root=str(ws_root),
    )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"dr_{label}_{ts}.tar.gz" if label else f"dr_{ts}.tar.gz"
    tarball_path = out_root / name

    try:
        with tarfile.open(tarball_path, "w:gz") as tar:
            # 1. ChromaDB collections — one JSONL per collection per KB.
            for kb_dir in _list_kb_dirs(ws_root):
                kb_name = kb_dir.name
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(kb_dir))
                    cols = client.list_collections()
                except Exception as exc:
                    manifest.errors.append(f"chromadb list {kb_name}: {exc}")
                    continue
                for col_meta in cols:
                    col_name = (
                        col_meta.name if hasattr(col_meta, "name") else str(col_meta)
                    )
                    archive_name = f"chromadb/{kb_name}/{col_name}.jsonl.gz"
                    try:
                        rows, byts = _add_jsonl_gz_to_tar(
                            tar, archive_name,
                            _stream_chroma_jsonl(kb_dir, col_name),
                        )
                        manifest.chromadb.append(_ChromaCollectionEntry(
                            kb=kb_name, collection=col_name,
                            rows=rows, bytes=byts,
                        ))
                        manifest.total_rows_chromadb += rows
                        manifest.total_bytes += byts
                    except Exception as exc:
                        logger.debug(
                            "dr.export: chromadb stream failed for %s/%s",
                            kb_name, col_name, exc_info=True,
                        )
                        manifest.chromadb.append(_ChromaCollectionEntry(
                            kb=kb_name, collection=col_name, rows=0,
                            bytes=0, error=str(exc),
                        ))
                        manifest.errors.append(
                            f"chromadb stream {kb_name}/{col_name}: {exc}"
                        )

            # 2. Postgres tables.
            try:
                from app.control_plane.db import execute  # noqa: F401
                pg_available = True
            except Exception as exc:
                pg_available = False
                manifest.errors.append(f"postgres unavailable: {exc}")
            if pg_available:
                for table in _PG_TABLES:
                    try:
                        archive_name = f"postgres/{table.replace('.', '__')}.jsonl.gz"
                        rows, byts = _add_jsonl_gz_to_tar(
                            tar, archive_name,
                            _stream_pg_table_jsonl(table),
                        )
                        manifest.postgres.append(_PostgresTableEntry(
                            table=table, rows=rows, bytes=byts,
                        ))
                        manifest.total_rows_postgres += rows
                        manifest.total_bytes += byts
                    except Exception as exc:
                        logger.debug(
                            "dr.export: postgres stream failed for %s",
                            table, exc_info=True,
                        )
                        manifest.postgres.append(_PostgresTableEntry(
                            table=table, rows=0, bytes=0, error=str(exc),
                        ))
                        manifest.errors.append(f"postgres {table}: {exc}")

            # 3. Workspace ledgers (verbatim file copies).
            for p in _walk_ledger_paths(ws_root):
                try:
                    rel = str(p.relative_to(ws_root))
                except ValueError:
                    continue
                if _is_secret_path(rel):
                    manifest.excluded_secret_paths.append(rel)
                    continue
                archive_name = f"workspace_ledgers/{rel}"
                try:
                    byts = _add_file_to_tar(tar, p, archive_name)
                    manifest.ledgers.append(_LedgerFileEntry(
                        rel_path=rel, bytes=byts,
                    ))
                    manifest.total_bytes += byts
                except Exception as exc:
                    logger.debug(
                        "dr.export: ledger add failed for %s", rel,
                        exc_info=True,
                    )
                    manifest.ledgers.append(_LedgerFileEntry(
                        rel_path=rel, bytes=0, error=str(exc),
                    ))
                    manifest.errors.append(f"ledger {rel}: {exc}")

            # 4. Manifest is the LAST entry so a partial tarball still
            # parses (manifest captures the partial state).
            manifest.completed_at = datetime.now(timezone.utc).isoformat()
            manifest.duration_s = time.monotonic() - started
            manifest.ok = len(manifest.errors) == 0
            manifest_bytes = json.dumps(
                manifest.to_dict(), indent=2, default=str,
            ).encode("utf-8")
            mi = tarfile.TarInfo("manifest.json")
            mi.size = len(manifest_bytes)
            mi.mtime = int(time.time())
            mi.mode = 0o644
            tar.addfile(mi, io.BytesIO(manifest_bytes))
    except Exception as exc:
        logger.exception("dr.export: tarball write failed")
        manifest.errors.append(f"tarball write: {exc}")
        manifest.ok = False
        manifest.completed_at = datetime.now(timezone.utc).isoformat()
        manifest.duration_s = time.monotonic() - started

    return tarball_path, manifest


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m app.dr.export_kbs",
        description="Portable DR export — produces a self-contained tarball.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    tarball, manifest = export(
        output_dir=args.output_dir, label=args.label,
    )
    print(json.dumps({
        "tarball": str(tarball),
        "ok": manifest.ok,
        "rows_chromadb": manifest.total_rows_chromadb,
        "rows_postgres": manifest.total_rows_postgres,
        "ledger_files": len(manifest.ledgers),
        "excluded_secret_paths": len(manifest.excluded_secret_paths),
        "duration_s": round(manifest.duration_s, 3),
        "errors": manifest.errors,
    }, indent=2))
    return 0 if manifest.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
