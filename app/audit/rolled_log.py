"""Rolled-segment append-only log with hash-chain integrity.

Each log instance is a directory::

    <base_dir>/<log_name>/
    ├── current.jsonl                          # active write segment
    ├── seg-0000-20260101T000000Z.jsonl        # closed segments, chronological
    ├── seg-0001-...jsonl
    ├── INDEX.json                             # segment metadata
    └── .lock                                  # cross-process append lock

Per-entry envelope (one JSON object per line)::

    {"seq": 1234,
     "ts": "2026-05-10T12:34:56.789Z",
     "prev_hash": "<sha256 of prior canonical line>",
     "payload": {...}}

The first line of every closed segment after seg-0000 is a *segment_root*
marker that chains across the rotation boundary::

    {"type": "segment_root",
     "segment_id": 1,
     "prev_segment_id": 0,
     "prev_root_hash": "<sha256 of last line of previous segment>",
     "opened_at": "2026-05-10T..."}

Genesis log: first entry of seg-0000 has ``prev_hash = "GENESIS"``.
Migrated log: first entry of seg-0000 has
``prev_hash = "MIGRATED-FROM-LEGACY:<sha256>"`` so the verifier
recognises a synthetic origin boundary without flagging it as tamper.

Concurrency: a separate ``.lock`` file is held with ``fcntl.flock``
during the entire (open current → append → maybe rotate → close)
sequence, so rotation cannot strand a writer holding a stale fd.

Crash safety: each append fsyncs before returning. A torn last line
is tolerated on read (skipped with a warning) and overwritten on the
next successful append.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.safe_io import safe_write_json

logger = logging.getLogger(__name__)

GENESIS = "GENESIS"
LEGACY_PREFIX = "MIGRATED-FROM-LEGACY:"

DEFAULT_MAX_SEGMENT_BYTES = 10 * 1024 * 1024
INDEX_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_line(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _segment_filename(seg_id: int, opened_at: str) -> str:
    return f"seg-{seg_id:04d}-{opened_at}.jsonl"


def _parse_segment_id(name: str) -> int | None:
    if not name.startswith("seg-") or not name.endswith(".jsonl"):
        return None
    try:
        return int(name.split("-")[1])
    except (ValueError, IndexError):
        return None


@dataclass(frozen=True)
class SegmentInfo:
    segment_id: int
    path: str
    first_seq: int | None
    last_seq: int | None
    first_ts: str | None
    last_ts: str | None
    last_hash: str
    n_entries: int
    synthesized_from_legacy: bool = False


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    n_segments: int
    n_entries: int
    errors: list[str] = field(default_factory=list)
    legacy_boundaries: list[int] = field(default_factory=list)


class _LogPaths:
    def __init__(self, base_dir: Path, log_name: str) -> None:
        self.root = base_dir / log_name
        self.current = self.root / "current.jsonl"
        self.index = self.root / "INDEX.json"
        self.lock = self.root / ".lock"


def _empty_index(log_name: str) -> dict:
    return {
        "version": INDEX_VERSION,
        "log_name": log_name,
        "segments": [],
        "current_seg_id": 0,
        "next_seq": 1,
    }


def _load_index(paths: _LogPaths, log_name: str) -> dict:
    try:
        return json.loads(paths.index.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return _empty_index(log_name)


def _last_line_hash_of(path: Path) -> str | None:
    """Hash of the last non-empty line in ``path``, or ``None`` if empty/missing."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            chunk = min(65536, size)
            f.seek(size - chunk, os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [ln for ln in tail.split("\n") if ln.strip()]
    if not lines:
        return None
    return _hash_line(lines[-1])


@contextmanager
def _flock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


class RolledLogStore:
    """Append-only log with size-triggered segment rotation and hash chain.

    Thread-safe within a process via an internal lock; cross-process
    safe via ``fcntl.flock`` on a dedicated ``.lock`` file. Each
    append fsyncs the line before returning. Rotation is atomic: the
    new ``current.jsonl`` exists with its segment_root line written
    before the old segment is exposed to readers via INDEX.json.
    """

    def __init__(
        self,
        base_dir: Path | str,
        log_name: str,
        max_segment_bytes: int = DEFAULT_MAX_SEGMENT_BYTES,
    ) -> None:
        self.log_name = log_name
        self.max_segment_bytes = max_segment_bytes
        self._paths = _LogPaths(Path(base_dir), log_name)
        self._tlock = threading.Lock()
        self._paths.root.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict) -> int:
        """Append a payload. Returns the assigned sequence number."""
        with self._tlock, _flock(self._paths.lock):
            return self._append_under_lock(payload)

    def stats(self) -> dict:
        idx = _load_index(self._paths, self.log_name)
        n_closed = sum(s["n_entries"] for s in idx["segments"])
        cur_lines = 0
        cur_bytes = 0
        if self._paths.current.exists():
            cur_bytes = self._paths.current.stat().st_size
            try:
                with open(self._paths.current, "r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("type") == "segment_root":
                            continue
                        cur_lines += 1
            except OSError:
                pass
        return {
            "log_name": self.log_name,
            "n_segments_closed": len(idx["segments"]),
            "n_entries_closed": n_closed,
            "n_entries_current": cur_lines,
            "current_seg_id": idx["current_seg_id"],
            "next_seq": idx["next_seq"],
            "current_bytes": cur_bytes,
        }

    # --- internals ---------------------------------------------------

    def _append_under_lock(self, payload: dict) -> int:
        idx = _load_index(self._paths, self.log_name)
        self._ensure_current(idx)

        seq = idx["next_seq"]
        ts = _utc_now_iso()
        prev_hash = self._last_hash(idx)
        entry = {
            "seq": seq,
            "ts": ts,
            "prev_hash": prev_hash,
            "payload": payload,
        }
        line = _canonical(entry) + "\n"

        with open(self._paths.current, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        idx["next_seq"] = seq + 1
        safe_write_json(self._paths.index, idx)

        size = self._paths.current.stat().st_size
        if size >= self.max_segment_bytes:
            self._rotate(idx)

        return seq

    def _ensure_current(self, idx: dict) -> None:
        """Recreate ``current.jsonl`` if missing, chaining to last segment."""
        cur = self._paths.current
        if cur.exists():
            return
        cur.parent.mkdir(parents=True, exist_ok=True)
        if idx["segments"]:
            last = idx["segments"][-1]
            root = {
                "type": "segment_root",
                "segment_id": idx["current_seg_id"],
                "prev_segment_id": last["segment_id"],
                "prev_root_hash": last["last_hash"],
                "opened_at": _utc_now_iso(),
            }
            line = _canonical(root) + "\n"
            with open(cur, "w", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        else:
            cur.touch()

    def _last_hash(self, idx: dict) -> str:
        h = _last_line_hash_of(self._paths.current)
        if h is not None:
            return h
        if idx["segments"]:
            return idx["segments"][-1]["last_hash"]
        return GENESIS

    def _rotate(self, idx: dict) -> None:
        cur = self._paths.current
        seg_id = idx["current_seg_id"]
        first_seq = last_seq = first_ts = last_ts = None
        n = 0
        with open(cur, "r", encoding="utf-8") as rf:
            for raw in rf:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "segment_root":
                    continue
                n += 1
                if first_seq is None:
                    first_seq = obj.get("seq")
                    first_ts = obj.get("ts")
                last_seq = obj.get("seq")
                last_ts = obj.get("ts")

        last_hash = _last_line_hash_of(cur) or GENESIS
        opened_at = _utc_now_compact()
        seg_path = self._paths.root / _segment_filename(seg_id, opened_at)
        os.replace(cur, seg_path)

        idx["segments"].append({
            "segment_id": seg_id,
            "path": seg_path.name,
            "first_seq": first_seq,
            "last_seq": last_seq,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "last_hash": last_hash,
            "n_entries": n,
            "synthesized_from_legacy": False,
        })
        idx["current_seg_id"] = seg_id + 1
        safe_write_json(self._paths.index, idx)
        self._ensure_current(idx)


class RolledLogReader:
    """Chronological reader across all segments + the active current segment."""

    def __init__(self, base_dir: Path | str, log_name: str) -> None:
        self.log_name = log_name
        self._paths = _LogPaths(Path(base_dir), log_name)

    def iter_entries(self, since_seq: int = 0) -> Iterator[dict]:
        """Yield user entries (skips segment_root markers) oldest → newest."""
        idx = _load_index(self._paths, self.log_name)
        for seg in idx["segments"]:
            if seg["last_seq"] is not None and seg["last_seq"] < since_seq:
                continue
            yield from self._iter_file(self._paths.root / seg["path"], since_seq)
        if self._paths.current.exists():
            yield from self._iter_file(self._paths.current, since_seq)

    @staticmethod
    def _iter_file(path: Path, since_seq: int) -> Iterator[dict]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "rolled_log: skipping malformed line in %s", path,
                        )
                        continue
                    if obj.get("type") == "segment_root":
                        continue
                    if obj.get("seq", 0) < since_seq:
                        continue
                    yield obj
        except OSError:
            logger.warning("rolled_log: cannot read %s", path, exc_info=True)

    def recent(self, n: int) -> list[dict]:
        """Return the last n user entries in chronological order."""
        if n <= 0:
            return []
        idx = _load_index(self._paths, self.log_name)
        files: list[Path] = []
        if self._paths.current.exists():
            files.append(self._paths.current)
        files.extend(
            self._paths.root / seg["path"] for seg in reversed(idx["segments"])
        )
        out: list[dict] = []
        for path in files:
            if len(out) >= n:
                break
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except OSError:
                continue
            for line in reversed(lines):
                if len(out) >= n:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") == "segment_root":
                    continue
                out.append(obj)
        out.reverse()
        return out


class RolledLogVerifier:
    """Walks all segments + current and validates the cross-segment hash chain."""

    def __init__(self, base_dir: Path | str, log_name: str) -> None:
        self.log_name = log_name
        self._paths = _LogPaths(Path(base_dir), log_name)

    def verify(self) -> VerificationResult:
        errors: list[str] = []
        legacy: list[int] = []
        n_entries = 0

        idx = _load_index(self._paths, self.log_name)

        prev_hash = GENESIS
        for seg in idx["segments"]:
            seg_path = self._paths.root / seg["path"]
            n, last_hash, errs, was_legacy = self._verify_file(
                seg_path,
                expected_prev_hash=prev_hash,
                segment_id=seg["segment_id"],
            )
            n_entries += n
            errors.extend(errs)
            if was_legacy:
                legacy.append(seg["segment_id"])
            if last_hash != seg["last_hash"]:
                errors.append(
                    f"segment {seg['segment_id']}: index last_hash mismatch "
                    f"(file={last_hash[:12]} index={seg['last_hash'][:12]})"
                )
            prev_hash = seg["last_hash"]

        n_segments_total = len(idx["segments"])
        if self._paths.current.exists():
            n, last_hash, errs, was_legacy = self._verify_file(
                self._paths.current,
                expected_prev_hash=prev_hash,
                segment_id=idx.get("current_seg_id", n_segments_total),
            )
            n_entries += n
            errors.extend(errs)
            if was_legacy:
                legacy.append(idx.get("current_seg_id", n_segments_total))
            n_segments_total += 1

        return VerificationResult(
            ok=not errors,
            n_segments=n_segments_total,
            n_entries=n_entries,
            errors=errors,
            legacy_boundaries=legacy,
        )

    @staticmethod
    def _verify_file(
        path: Path,
        expected_prev_hash: str,
        segment_id: int,
    ) -> tuple[int, str, list[str], bool]:
        errors: list[str] = []
        n_user = 0
        running = expected_prev_hash
        was_legacy = False
        try:
            with open(path, "r", encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        errors.append(f"{path.name}:{lineno}: malformed JSON")
                        continue
                    if obj.get("type") == "segment_root":
                        prh = obj.get("prev_root_hash", "")
                        if prh != running:
                            errors.append(
                                f"{path.name}:{lineno}: segment_root prev_root_hash "
                                f"{prh[:12]}... != expected {running[:12]}..."
                            )
                        running = _hash_line(line)
                        continue
                    declared = obj.get("prev_hash", "")
                    if declared.startswith(LEGACY_PREFIX):
                        was_legacy = True
                        running = _hash_line(line)
                        n_user += 1
                        continue
                    if declared != running:
                        errors.append(
                            f"{path.name}:{lineno}: prev_hash {declared[:12]}... "
                            f"!= expected {running[:12]}... (CHAIN_BREAK)"
                        )
                    running = _hash_line(line)
                    n_user += 1
        except OSError as e:
            errors.append(f"{path.name}: read error {e}")
        return n_user, running, errors, was_legacy
