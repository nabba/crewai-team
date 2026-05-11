"""Portable DR import — read an export tarball into ephemeral targets.

PROGRAM §40 (2026-05-10) — Q3 Item 13.

Two consumer paths:

  * **Boot drill** (``app/dr/boot_drill.py``) — restore into a temporary
    directory, run sanity queries, clean up. The drill never touches
    the live workspace.

  * **Operator restore** — restore one KB into a sandbox directory for
    inspection (e.g. "what was in `philosophy` six months ago?").

We deliberately do NOT provide an "overwrite live workspace" mode in
this module. That's a destructive op the operator should run by
hand with full awareness of what they're destroying — by extracting
the tarball with ``tar`` and moving directories themselves. A naive
``--restore-live`` flag here is exactly the kind of footgun the audit
told us to avoid.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import tarfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImportSummary:
    tarball_path: str
    target_dir: str
    chromadb_kbs_restored: int = 0
    chromadb_collections_restored: int = 0
    chromadb_rows_restored: int = 0
    postgres_tables_dumped: int = 0
    postgres_rows_dumped: int = 0
    ledger_files_restored: int = 0
    duration_s: float = 0.0
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    manifest_seen: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = round(self.duration_s, 3)
        return d


def _read_manifest(tar: tarfile.TarFile) -> dict[str, Any] | None:
    try:
        member = tar.getmember("manifest.json")
    except KeyError:
        return None
    f = tar.extractfile(member)
    if f is None:
        return None
    return json.loads(f.read().decode("utf-8"))


def _restore_chromadb_from_jsonl(
    target_kb_root: Path, collection_name: str, lines: list[str],
) -> int:
    """Recreate one collection under ``target_kb_root`` from JSONL
    lines (one row per line). Returns row count restored."""
    target_kb_root.mkdir(parents=True, exist_ok=True)
    import chromadb
    client = chromadb.PersistentClient(path=str(target_kb_root))
    # If a stale collection exists in the target (e.g. previous drill
    # run), wipe it. Target is ephemeral, so this is safe.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    rows: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        return 0
    # Add in batches of 200 — same shape as the rebuild module.
    BATCH = 200
    n = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        col.add(
            ids=[r["id"] for r in chunk],
            documents=[r.get("document") for r in chunk],
            metadatas=[
                r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
                for r in chunk
            ],
            embeddings=[r["embedding"] for r in chunk],
        )
        n += len(chunk)
    return n


def _read_jsonl_gz_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    """Read a gz-jsonl member as a list of decoded lines."""
    f = tar.extractfile(member)
    if f is None:
        return []
    raw = f.read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as gz:
        return gz.read().decode("utf-8").split("\n")


def import_tarball(
    tarball_path: Path | str,
    target_dir: Path | str,
    *,
    skip_chromadb: bool = False,
    skip_postgres: bool = True,   # postgres restore is a separate op
    skip_ledgers: bool = False,
) -> ImportSummary:
    """Read an export tarball into ``target_dir``.

    ChromaDB collections re-build into ``target_dir/chromadb/<kb>/``.
    Postgres rows are dumped as JSONL into
    ``target_dir/postgres/<table>.jsonl`` (operator can ``\\copy`` them
    back later if they want a real cluster restore).
    Ledger files write under ``target_dir/workspace_ledgers/<relpath>``.
    """
    started = time.monotonic()
    tp = Path(tarball_path)
    td = Path(target_dir)
    summary = ImportSummary(tarball_path=str(tp), target_dir=str(td))

    if not tp.exists():
        summary.ok = False
        summary.errors.append(f"tarball not found: {tp}")
        summary.duration_s = time.monotonic() - started
        return summary

    td.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(tp, "r:gz") as tar:
            summary.manifest_seen = _read_manifest(tar)
            chroma_target_root = td / "chromadb"
            postgres_target = td / "postgres"
            ledger_target = td / "workspace_ledgers"
            seen_kbs: set[str] = set()

            for member in tar.getmembers():
                name = member.name

                # ChromaDB: chromadb/<kb>/<collection>.jsonl.gz
                if (
                    not skip_chromadb
                    and name.startswith("chromadb/")
                    and name.endswith(".jsonl.gz")
                ):
                    parts = name.split("/")
                    if len(parts) != 3:
                        continue
                    _, kb, col_file = parts
                    col_name = col_file[:-len(".jsonl.gz")]
                    seen_kbs.add(kb)
                    try:
                        lines = _read_jsonl_gz_member(tar, member)
                        target_kb_root = chroma_target_root / kb
                        rows = _restore_chromadb_from_jsonl(
                            target_kb_root, col_name, lines,
                        )
                        summary.chromadb_collections_restored += 1
                        summary.chromadb_rows_restored += rows
                    except Exception as exc:
                        logger.debug(
                            "dr.import: chromadb restore failed for %s",
                            name, exc_info=True,
                        )
                        summary.errors.append(f"chromadb {kb}/{col_name}: {exc}")
                    continue

                # Postgres: postgres/<table>.jsonl.gz — we just dump
                # the JSONL out for the operator. No live cluster
                # restore here; that's a separate op.
                if (
                    not skip_postgres
                    and name.startswith("postgres/")
                    and name.endswith(".jsonl.gz")
                ):
                    try:
                        lines = _read_jsonl_gz_member(tar, member)
                        out_name = (
                            name[len("postgres/"):-len(".gz")]
                        )
                        out_path = postgres_target / out_name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        out_path.write_text("\n".join(lines), encoding="utf-8")
                        summary.postgres_tables_dumped += 1
                        # crude row count: lines minus blanks
                        summary.postgres_rows_dumped += sum(
                            1 for l in lines if l.strip()
                        )
                    except Exception as exc:
                        summary.errors.append(f"postgres {name}: {exc}")
                    continue

                # Ledgers: verbatim copy under workspace_ledgers/.
                if (
                    not skip_ledgers
                    and name.startswith("workspace_ledgers/")
                    and member.isfile()
                ):
                    try:
                        rel = name[len("workspace_ledgers/"):]
                        out_path = ledger_target / rel
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        f = tar.extractfile(member)
                        if f is not None:
                            with out_path.open("wb") as outf:
                                outf.write(f.read())
                        summary.ledger_files_restored += 1
                    except Exception as exc:
                        summary.errors.append(f"ledger {name}: {exc}")
                    continue

            summary.chromadb_kbs_restored = len(seen_kbs)
    except Exception as exc:
        logger.exception("dr.import: tarball read failed")
        summary.errors.append(f"tarball read: {exc}")
        summary.ok = False

    summary.duration_s = time.monotonic() - started
    summary.ok = summary.ok and not summary.errors
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m app.dr.import_kbs",
        description="Portable DR import — restore an export tarball into a sandbox.",
    )
    parser.add_argument("--tarball", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--skip-chromadb", action="store_true")
    parser.add_argument("--skip-postgres", action="store_true", default=True)
    parser.add_argument("--no-skip-postgres", action="store_false", dest="skip_postgres")
    parser.add_argument("--skip-ledgers", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    summary = import_tarball(
        tarball_path=args.tarball, target_dir=args.target_dir,
        skip_chromadb=args.skip_chromadb,
        skip_postgres=args.skip_postgres,
        skip_ledgers=args.skip_ledgers,
    )
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
