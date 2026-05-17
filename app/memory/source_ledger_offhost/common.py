"""
common.py — Shared incremental-upload primitives for off-host source ledgers.

The two backends (S3, Google Drive) share these:

  * Read the latest byte offset that was uploaded for this (kb, dest)
  * Slice the file from that offset to current EOF
  * Gzip the slice
  * Caller does the actual transport (S3 PutObject / Drive files.create)
  * On success, persist the new offset

Daily passes always create a new object keyed by the local date (UTC).
This guarantees:

  1. Append-only object lineage — never overwritten, never deleted
  2. Recovery is trivial — list, sort, concatenate
  3. Operator can sample any date's snapshot without rebuilding the
     whole thing
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _host_tag() -> str:
    """A stable per-host identifier used as the object-key prefix.

    Defaults to the system hostname so multiple hosts uploading to the
    same bucket don't collide. Override with ``LEDGER_HOST_TAG`` env
    var if the operator wants a deterministic name (recommended for
    production: keeps the prefix stable across hostname changes).
    """
    env_val = os.getenv("LEDGER_HOST_TAG", "").strip()
    if env_val:
        return env_val
    try:
        return socket.gethostname() or "unknown-host"
    except Exception:
        return "unknown-host"


@dataclass
class OffhostUploadResult:
    ok: bool
    kb_name: str
    destination: str
    bytes_uploaded: int = 0
    object_key: str = ""
    rows_uploaded: int = 0
    error: str = ""
    skipped: bool = False
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "kb_name": self.kb_name,
            "destination": self.destination,
            "bytes_uploaded": self.bytes_uploaded,
            "object_key": self.object_key,
            "rows_uploaded": self.rows_uploaded,
            "error": self.error,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 3),
        }


def _state_path(kb_name: str, destination: str) -> Path:
    """Per-(kb, destination) state file. Tracks last-uploaded byte
    offset so the next upload picks up exactly where the previous
    one left off."""
    return _workspace_root() / kb_name / f".source_ledger_{destination}_state.json"


def _read_state(kb_name: str, destination: str) -> dict:
    path = _state_path(kb_name, destination)
    if not path.exists():
        return {"last_offset": 0, "last_upload_ts": 0.0, "last_object_key": ""}
    try:
        return json.loads(path.read_text())
    except Exception:
        logger.debug(
            "source_ledger_offhost: state read failed for %s/%s",
            kb_name, destination, exc_info=True,
        )
        return {"last_offset": 0, "last_upload_ts": 0.0, "last_object_key": ""}


def record_upload(kb_name: str, destination: str, new_offset: int,
                  object_key: str) -> None:
    """Persist the new offset + object_key after a successful upload.

    Atomic (temp + rename). Best-effort — caller doesn't fail if this
    fails, but next upload will redo the day's window which costs
    bandwidth not correctness.
    """
    path = _state_path(kb_name, destination)
    state = {
        "last_offset": int(new_offset),
        "last_upload_ts": time.time(),
        "last_object_key": object_key,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(path)
    except Exception:
        logger.debug(
            "source_ledger_offhost: state write failed for %s/%s",
            kb_name, destination, exc_info=True,
        )


def canonical_object_key(kb_name: str, ts: Optional[datetime] = None) -> str:
    """Construct the canonical destination key for today's incremental
    upload. The shape is::

        <host_tag>/<kb_name>/source_ledger/YYYY-MM-DD.jsonl.gz

    Date is UTC-anchored. If you upload twice on the same UTC day,
    the second one overwrites (which is intended — full-of-day always
    contains the latest snapshot of that day's appends).
    """
    ts = ts or datetime.now(timezone.utc)
    return f"{_host_tag()}/{kb_name}/source_ledger/{ts.strftime('%Y-%m-%d')}.jsonl.gz"


def incremental_payload(kb_name: str, destination: str) -> tuple[bytes, int, int, int]:
    """Build the gzip payload for the next upload window.

    Returns ``(gzip_bytes, n_rows, start_offset, end_offset)``:
      * gzip_bytes   — ready to PutObject / drive.files.create
      * n_rows       — newline count in the slice
      * start_offset — for telemetry
      * end_offset   — pass to ``record_upload`` after successful send

    If the ledger hasn't grown since the last upload, returns an empty
    payload and the same offsets — caller skips the network call.

    Strategy: today's object includes EVERYTHING for today (not just
    the delta since last upload). This is intentional: it means
    recovery only needs the latest copy of each date, even if earlier
    copies are missing. The cost is ~1 day's data re-uploaded per
    pass; cheap, well-bounded.
    """
    from app.memory.source_ledger import ledger_path  # late import
    ledger = ledger_path(kb_name)
    if not ledger.exists():
        return b"", 0, 0, 0

    # Determine today's date boundary by mtime — fast enough and
    # avoids parsing JSON.
    end_offset = ledger.stat().st_size

    # Walk backward to find today's first row. We re-upload all of
    # today's data each pass (idempotent at the destination).
    cutoff_dt = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    start_offset = 0

    with ledger.open("rb") as f:
        pos = 0
        for line in f:
            try:
                d = json.loads(line)
                row_ts = float(d.get("ts") or 0)
                if row_ts >= cutoff_dt:
                    start_offset = pos
                    break
            except Exception:
                pass
            pos += len(line)
        else:
            # No row crosses today's start; upload nothing
            return b"", 0, 0, end_offset

        # Read from start_offset to EOF.
        f.seek(start_offset)
        body = f.read()

    n_rows = body.count(b"\n")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        gz.write(body)
    return buf.getvalue(), n_rows, start_offset, end_offset
