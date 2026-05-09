"""Hash-chained amendment audit — separate from the global audit log.

Why a separate file: the global ``control_plane.audit_log`` is
INSERT-only Postgres (good — agents can't erase their tracks) but the
Postgres rows can be lost in a DR scenario. The Tier-3 amendment trail
is sensitive enough that we ship it BOTH places: the Postgres audit
gives operators the dashboard surface, and this JSONL gives them a
recoverable, hash-chained, file-based record that survives even when
Postgres is gone.

Each line is a JSON object with ``prev_hash`` linking to the previous
line's SHA-256. The chain head is recorded on the proposal so a
verifier can walk forward from there. Tampering with any past line
breaks the chain.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_AUDIT_PATH = (
    Path(__file__).resolve().parents[2] / "workspace" / "governance"
    / "tier3_amendments" / "audit.jsonl"
)

_GENESIS_HASH = "0" * 64

_lock = threading.Lock()


def _last_hash() -> str:
    """Return the SHA-256 of the last line, or genesis if empty."""
    if not _AUDIT_PATH.exists():
        return _GENESIS_HASH
    try:
        with _AUDIT_PATH.open("rb") as f:
            f.seek(0, 2)  # end
            size = f.tell()
            if size == 0:
                return _GENESIS_HASH
            # Walk back to find the last full line.
            backstep = min(size, 4096)
            f.seek(size - backstep)
            tail = f.read(backstep)
        last_line = tail.decode("utf-8", errors="replace").rstrip("\n")
        if "\n" in last_line:
            last_line = last_line.rsplit("\n", 1)[-1]
        if not last_line.strip():
            return _GENESIS_HASH
        rec = json.loads(last_line)
        return str(rec.get("hash") or _GENESIS_HASH)
    except Exception:
        logger.debug(
            "tier3_amendment.audit: last_hash read failed",
            exc_info=True,
        )
        return _GENESIS_HASH


def _line_hash(prev_hash: str, body: dict[str, Any]) -> str:
    raw = (prev_hash + json.dumps(body, sort_keys=True)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def append(*, proposal_id: str, action: str, **detail: Any) -> str:
    """Append a hash-chained entry. Returns the new chain head hash.

    Best-effort: I/O failure logs and returns the previous head so the
    proposal's ``audit_chain_head`` field doesn't drift.
    """
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "wall_clock_unix": time.time(),
        "proposal_id": proposal_id,
        "action": action,
        "detail": detail,
    }

    with _lock:
        prev = _last_hash()
        body["prev_hash"] = prev
        body["hash"] = _line_hash(prev, body)
        try:
            with _AUDIT_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(body, sort_keys=True))
                f.write("\n")
                f.flush()
        except Exception:
            logger.warning(
                "tier3_amendment.audit: append failed for %s/%s",
                proposal_id, action, exc_info=True,
            )
            return prev

    # Mirror to the global Postgres audit so operators see it on the
    # dashboard. Best-effort — JSONL is the durable record.
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(
            actor="tier3_amendment",
            action=action,
            resource_type="amendment",
            resource_id=proposal_id,
            detail={**detail, "chain_hash": body["hash"]},
        )
    except Exception:
        logger.debug(
            "tier3_amendment.audit: postgres mirror failed",
            exc_info=True,
        )

    return body["hash"]


def verify_chain() -> tuple[bool, list[dict[str, Any]]]:
    """Walk the JSONL chain and confirm every link.

    Returns ``(ok, broken_entries)``. The broken_entries list is the
    rows whose ``hash`` doesn't match the recomputed value — empty when
    the chain is intact.
    """
    if not _AUDIT_PATH.exists():
        return True, []

    broken: list[dict[str, Any]] = []
    prev = _GENESIS_HASH
    try:
        with _AUDIT_PATH.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    broken.append({"line_no": line_no, "reason": "invalid_json"})
                    continue
                claimed_prev = rec.get("prev_hash")
                claimed_hash = rec.get("hash")
                if claimed_prev != prev:
                    broken.append({
                        "line_no": line_no,
                        "reason": "prev_hash_mismatch",
                        "claimed_prev": claimed_prev,
                        "expected_prev": prev,
                    })
                # Recompute hash without ``hash`` field.
                body = {k: v for k, v in rec.items() if k != "hash"}
                expected = _line_hash(claimed_prev or _GENESIS_HASH, body)
                if expected != claimed_hash:
                    broken.append({
                        "line_no": line_no,
                        "reason": "hash_mismatch",
                        "claimed": claimed_hash,
                        "expected": expected,
                    })
                prev = claimed_hash or prev
    except Exception:
        logger.debug(
            "tier3_amendment.audit: verify_chain read failed",
            exc_info=True,
        )

    return (not broken), broken
