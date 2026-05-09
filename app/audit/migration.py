"""One-shot migration of legacy single-file audit logs into rolled-segment form.

Two source formats are supported:

* **JSON list-of-dicts** — e.g. ``workspace/audit_journal.json``:
  one JSON document at the top level, expected to be ``list[dict]``.
* **JSONL** — e.g. ``workspace/logs/errors.jsonl``: one JSON object
  per line, optionally preceded by other rotated suffixes
  (``errors.jsonl.1``, ``errors.jsonl.2``, ...) which are all
  consumed in chronological order.

After migration:

* The rolled log directory at ``<base_dir>/<log_name>/`` contains
  ``seg-0000-<ts>.jsonl`` with the synthetic chain, plus
  ``current.jsonl`` (containing only a ``segment_root`` line linking
  to seg-0000).
* The legacy source file is renamed to ``<source>.preserved`` and
  is left in place indefinitely as a forensic artefact.
* ``INDEX.json`` carries ``synthesized_from_legacy = true`` on
  seg-0000 so the verifier records a ``legacy_boundary`` rather
  than reporting a chain break.

Migration is idempotent: re-running on an already-migrated source
short-circuits when ``<source>.preserved`` exists and the rolled
directory is non-empty.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from app.audit.rolled_log import (
    GENESIS,
    LEGACY_PREFIX,
    INDEX_VERSION,
    _LogPaths,
    _canonical,
    _hash_line,
    _last_line_hash_of,
    _segment_filename,
    _utc_now_compact,
    _utc_now_iso,
)
from app.safe_io import safe_write_json

logger = logging.getLogger(__name__)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_ts(entry: dict) -> str:
    """Best-effort timestamp extraction from a legacy entry; falls back to migration-now."""
    for k in ("ts", "timestamp", "time", "created_at", "at"):
        v = entry.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, (int, float)):
            try:
                return datetime.fromtimestamp(v, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
            except (ValueError, OverflowError):
                pass
    return _utc_now_iso()


def _read_json_list(source: Path) -> list[dict]:
    raw = source.read_text()
    obj = json.loads(raw)
    if not isinstance(obj, list):
        raise ValueError(
            f"{source}: expected JSON list-of-dicts at top level, got {type(obj).__name__}"
        )
    return [e if isinstance(e, dict) else {"value": e} for e in obj]


def _read_jsonl(source: Path) -> list[dict]:
    out: list[dict] = []
    with open(source, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "migration: skipping malformed JSONL in %s", source,
                )
                continue
            if isinstance(obj, dict):
                out.append(obj)
            else:
                out.append({"value": obj})
    return out


def _build_legacy_seg(
    paths: _LogPaths,
    entries: Iterable[dict],
    legacy_sentinel: str,
) -> tuple[Path, dict]:
    """Write seg-0000 with synthesized hash chain. Returns (path, segment_info)."""
    paths.root.mkdir(parents=True, exist_ok=True)
    opened_at = _utc_now_compact()
    seg_path = paths.root / _segment_filename(0, opened_at)

    n = 0
    first_seq = first_ts = last_seq = last_ts = None
    prev_for_chain = legacy_sentinel
    seq = 1

    with open(seg_path, "w", encoding="utf-8") as f:
        for entry in entries:
            ts = _coerce_ts(entry)
            envelope = {
                "seq": seq,
                "ts": ts,
                "prev_hash": prev_for_chain,
                "payload": entry,
            }
            line = _canonical(envelope) + "\n"
            f.write(line)
            n += 1
            if first_seq is None:
                first_seq = seq
                first_ts = ts
            last_seq = seq
            last_ts = ts
            prev_for_chain = _hash_line(line.rstrip("\n"))
            seq += 1
        f.flush()
        os.fsync(f.fileno())

    last_hash = _last_line_hash_of(seg_path) or GENESIS
    return seg_path, {
        "segment_id": 0,
        "path": seg_path.name,
        "first_seq": first_seq,
        "last_seq": last_seq,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "last_hash": last_hash,
        "n_entries": n,
        "synthesized_from_legacy": True,
    }


def _open_post_migration_current(paths: _LogPaths, seg_info: dict) -> None:
    root = {
        "type": "segment_root",
        "segment_id": 1,
        "prev_segment_id": 0,
        "prev_root_hash": seg_info["last_hash"],
        "opened_at": _utc_now_iso(),
    }
    line = _canonical(root) + "\n"
    with open(paths.current, "w", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _migrate_common(
    source: Path,
    base_dir: Path,
    log_name: str,
    entries: list[dict],
) -> dict:
    paths = _LogPaths(base_dir, log_name)
    preserved = source.with_suffix(source.suffix + ".preserved")

    if preserved.exists() and paths.index.exists():
        logger.info(
            "migration: %s already migrated (preserved file + index exist); skipping",
            source,
        )
        return {"status": "already_migrated", "n_entries": 0, "preserved": str(preserved)}

    src_sha = _file_sha256(source)
    sentinel = f"{LEGACY_PREFIX}{src_sha}"

    seg_path, seg_info = _build_legacy_seg(paths, entries, sentinel)
    idx = {
        "version": INDEX_VERSION,
        "log_name": log_name,
        "segments": [seg_info],
        "current_seg_id": 1,
        "next_seq": (seg_info["last_seq"] or 0) + 1,
    }
    safe_write_json(paths.index, idx)
    _open_post_migration_current(paths, seg_info)

    os.replace(source, preserved)

    return {
        "status": "migrated",
        "n_entries": seg_info["n_entries"],
        "source_sha256": src_sha,
        "seg_0_path": str(seg_path),
        "preserved": str(preserved),
    }


def migrate_json_list(
    source: Path | str,
    base_dir: Path | str,
    log_name: str,
) -> dict:
    """Migrate a legacy JSON list-of-dicts file into rolled-segment form.

    Idempotent: re-running on an already-migrated source short-circuits.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    entries = _read_json_list(source)
    return _migrate_common(source, Path(base_dir), log_name, entries)


def migrate_jsonl(
    source: Path | str,
    base_dir: Path | str,
    log_name: str,
    rotated_companions: Iterable[Path | str] = (),
) -> dict:
    """Migrate a legacy JSONL file (with optional rotated companions).

    ``rotated_companions`` are read in the order given (oldest first)
    and prepended to the active source's entries. The active file is
    the one renamed to ``.preserved`` at the end; companions are
    left untouched and may be archived separately.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    entries: list[dict] = []
    for c in rotated_companions:
        cp = Path(c)
        if cp.exists():
            entries.extend(_read_jsonl(cp))
    entries.extend(_read_jsonl(source))
    return _migrate_common(source, Path(base_dir), log_name, entries)
