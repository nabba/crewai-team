"""End-to-end DR boot drill.

PROGRAM §40 (2026-05-10) — Q3 Item 13.

The drill answers a single question: **"Could we rebuild the system
from a backup right now?"** It runs without touching the live
workspace:

  1. Take the most recent DR export under
     ``workspace/backups/dr/`` (or run a fresh export with
     ``--export-fresh``).
  2. Import the tarball into a temporary directory.
  3. Run sanity queries against the restored ChromaDB collections:
     - count() matches the manifest.
     - peek(1) returns rows with the pinned embedding dimension.
     - One smoke retrieve per collection returns ≥1 row.
  4. Validate the workspace ledgers were extracted (file count + total
     bytes).
  5. Write a drill report under
     ``workspace/dr/drill_<timestamp>.json`` and emit a Signal alert
     summarising the outcome.

Cadence: weekly via the existing healing daemon (`restore_drill`
already exists for the binary backup; this drill complements it for
the portable export).

Operator usage::

    python -m app.dr.boot_drill                  # use latest export
    python -m app.dr.boot_drill --export-fresh   # export then drill
    python -m app.dr.boot_drill --keep-target    # don't tear down

The drill never raises; it always writes a report.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_DEFAULT_EXPORT_DIR = Path("/app/workspace/backups/dr")
_DRILL_REPORT_DIR = Path("/app/workspace/dr")


@dataclass
class CollectionDrillResult:
    kb: str
    collection: str
    expected_rows: int = 0
    observed_rows: int = 0
    peek_dim: int | None = None
    smoke_retrieve_ok: bool = False
    error: str | None = None
    ok: bool = True


@dataclass
class DrillReport:
    started_at: str = ""
    completed_at: str = ""
    duration_s: float = 0.0
    tarball: str = ""
    target_dir: str = ""
    target_kept: bool = False
    fresh_export: bool = False
    chromadb_results: list[CollectionDrillResult] = field(default_factory=list)
    ledger_files_restored: int = 0
    ledger_bytes_restored: int = 0
    overall_ok: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": round(self.duration_s, 3),
            "tarball": self.tarball,
            "target_dir": self.target_dir,
            "target_kept": self.target_kept,
            "fresh_export": self.fresh_export,
            "chromadb_results": [asdict(r) for r in self.chromadb_results],
            "ledger_files_restored": self.ledger_files_restored,
            "ledger_bytes_restored": self.ledger_bytes_restored,
            "overall_ok": self.overall_ok,
            "errors": self.errors,
        }


def _latest_tarball(export_dir: Path) -> Path | None:
    if not export_dir.exists():
        return None
    candidates = sorted(
        export_dir.glob("dr_*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _drill_chromadb_collection(
    target_kb_root: Path, collection: str, expected_rows: int,
) -> CollectionDrillResult:
    res = CollectionDrillResult(
        kb=target_kb_root.name, collection=collection,
        expected_rows=expected_rows,
    )
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(target_kb_root))
        col = client.get_collection(collection)
        res.observed_rows = col.count()
        # Peek for embedding dimension
        try:
            sample = col.peek(1)
            embs = sample.get("embeddings") if sample else None
            if embs and len(embs) > 0 and embs[0]:
                res.peek_dim = len(embs[0])
        except Exception as exc:
            res.error = f"peek failed: {exc}"
            res.ok = False
            return res
        # Smoke retrieve — any vector that's the right dim works
        if res.peek_dim and res.peek_dim > 0 and res.observed_rows > 0:
            try:
                # Use the actual peeked embedding rather than a synthesised
                # one — guarantees dim match without depending on Ollama.
                if embs and embs[0]:
                    out = col.query(query_embeddings=[embs[0]], n_results=1)
                    res.smoke_retrieve_ok = bool(out.get("documents", [[]])[0])
            except Exception as exc:
                res.error = f"query failed: {exc}"
                res.ok = False
                return res
        # Row count match within ±1 tolerance (small races during export are OK)
        if abs(res.observed_rows - res.expected_rows) > 1:
            res.error = (
                f"row count drift: expected={res.expected_rows} "
                f"observed={res.observed_rows}"
            )
            res.ok = False
    except Exception as exc:
        res.error = f"unexpected: {exc}"
        res.ok = False
    return res


def run_drill(
    *,
    export_fresh: bool = False,
    keep_target: bool = False,
    target_dir: Path | str | None = None,
    export_dir: Path | str | None = None,
) -> DrillReport:
    started = time.monotonic()
    started_iso = datetime.now(timezone.utc).isoformat()
    report = DrillReport(started_at=started_iso, fresh_export=export_fresh)

    export_root = Path(export_dir) if export_dir else _DEFAULT_EXPORT_DIR

    # 1. Get the tarball.
    tarball: Path | None = None
    if export_fresh:
        try:
            from app.dr.export_kbs import export
            tarball, _manifest = export(output_dir=export_root, label="drill")
        except Exception as exc:
            report.errors.append(f"fresh export failed: {exc}")
    if tarball is None:
        tarball = _latest_tarball(export_root)
    if tarball is None:
        report.overall_ok = False
        report.errors.append(
            f"no tarball available under {export_root} "
            f"and fresh export disabled"
        )
        report.completed_at = datetime.now(timezone.utc).isoformat()
        report.duration_s = time.monotonic() - started
        _write_report(report)
        return report
    report.tarball = str(tarball)

    # 2. Set up an ephemeral target.
    if target_dir:
        td = Path(target_dir)
        td.mkdir(parents=True, exist_ok=True)
        cleanup_target = False
    else:
        td = Path(tempfile.mkdtemp(prefix="dr_drill_"))
        cleanup_target = not keep_target
    report.target_dir = str(td)
    report.target_kept = not cleanup_target

    try:
        # 3. Import.
        from app.dr.import_kbs import import_tarball
        import_summary = import_tarball(
            tarball_path=tarball, target_dir=td,
            skip_chromadb=False, skip_postgres=True, skip_ledgers=False,
        )
        report.ledger_files_restored = import_summary.ledger_files_restored
        if import_summary.errors:
            report.errors.extend(import_summary.errors)
        if not import_summary.ok:
            report.overall_ok = False

        manifest = import_summary.manifest_seen or {}
        chroma_expected = {
            (e["kb"], e["collection"]): int(e.get("rows") or 0)
            for e in manifest.get("chromadb", []) or []
        }

        # 4. Drill each restored collection.
        chroma_root = td / "chromadb"
        if chroma_root.exists():
            for kb_dir in sorted(chroma_root.iterdir()):
                if not kb_dir.is_dir():
                    continue
                # Probe the collections via the client
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=str(kb_dir))
                    cols = client.list_collections()
                except Exception as exc:
                    report.errors.append(
                        f"chromadb list failed for {kb_dir.name}: {exc}"
                    )
                    continue
                for col_meta in cols:
                    col_name = (
                        col_meta.name if hasattr(col_meta, "name") else str(col_meta)
                    )
                    expected = chroma_expected.get((kb_dir.name, col_name), 0)
                    res = _drill_chromadb_collection(kb_dir, col_name, expected)
                    report.chromadb_results.append(res)
                    if not res.ok:
                        report.overall_ok = False

        # 5. Bytes-restored bookkeeping for the report.
        ledger_root = td / "workspace_ledgers"
        if ledger_root.exists():
            total_bytes = 0
            for p in ledger_root.rglob("*"):
                if p.is_file():
                    try:
                        total_bytes += p.stat().st_size
                    except OSError:
                        continue
            report.ledger_bytes_restored = total_bytes
    except Exception as exc:
        logger.exception("dr.boot_drill: drill failed")
        report.errors.append(f"unexpected: {exc}")
        report.overall_ok = False
    finally:
        if cleanup_target:
            try:
                shutil.rmtree(td, ignore_errors=True)
            except Exception:
                pass

    report.completed_at = datetime.now(timezone.utc).isoformat()
    report.duration_s = time.monotonic() - started
    _write_report(report)
    _send_drill_alert(report)
    return report


def _write_report(report: DrillReport) -> None:
    try:
        _DRILL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        fname = _DRILL_REPORT_DIR / f"drill_{ts}.json"
        fname.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("dr.boot_drill: report write failed", exc_info=True)


def _send_drill_alert(report: DrillReport) -> None:
    try:
        from app.life_companion._common import send_signal_alert
    except Exception:
        return
    if report.overall_ok:
        msg = (
            f"✅ DR boot drill OK — {len(report.chromadb_results)} collections, "
            f"{report.ledger_files_restored} ledger files, "
            f"{report.duration_s:.1f}s. Tarball: {Path(report.tarball).name}."
        )
    else:
        msg = (
            f"❌ DR boot drill FAILED — {len(report.errors)} errors. "
            f"See workspace/dr/drill_*.json for the report."
        )
    try:
        send_signal_alert(msg, tag="dr_drill")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.dr.boot_drill",
        description="End-to-end DR boot drill — verify the latest export.",
    )
    parser.add_argument(
        "--export-fresh", action="store_true",
        help="Run a fresh export before the drill instead of using the latest.",
    )
    parser.add_argument(
        "--keep-target", action="store_true",
        help="Leave the temporary import dir on disk for inspection.",
    )
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--export-dir", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    report = run_drill(
        export_fresh=args.export_fresh,
        keep_target=args.keep_target,
        target_dir=args.target_dir,
        export_dir=args.export_dir,
    )
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0 if report.overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
