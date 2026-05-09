"""Hash-chained ratchet audit — separate from the global Postgres audit.

Same shape as ``app.governance_amendment.audit`` but for ratchet events.
Lives at ``workspace/governance/ratchet_audit.jsonl``. Each line is

    {"ts", "prev_hash", "hash", "action", "detail"}

where ``hash = sha256(prev_hash + json.dumps(body, sort_keys=True))``.
A break in the chain is detectable by ``verify_chain()``. Belt-and-
suspenders: every event is also mirrored to
``control_plane.audit_log`` (Postgres) via the existing
``AuditTrail.log()``.
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
    / "ratchet_audit.jsonl"
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
        logger.debug("ratchet.audit: last_hash read failed", exc_info=True)
        return _GENESIS_HASH


def _line_hash(prev_hash: str, body: dict[str, Any]) -> str:
    raw = (prev_hash + json.dumps(body, sort_keys=True)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def append(*, action: str, **detail: Any) -> str:
    """Append one event. Returns the new chain head hash."""
    _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "wall_clock_unix": time.time(),
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
                "ratchet.audit: append failed for %s", action, exc_info=True,
            )
            return prev

    # Mirror to global Postgres audit. Best-effort.
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(
            actor="governance_ratchet",
            action=action,
            detail={**detail, "chain_hash": body["hash"]},
        )
    except Exception:
        logger.debug("ratchet.audit: postgres mirror failed", exc_info=True)

    return body["hash"]


def verify_chain() -> tuple[bool, list[dict[str, Any]]]:
    """Walk the chain forward; return ``(ok, broken_lines)``.

    ``broken_lines`` is empty when the chain is intact.
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
        logger.debug("ratchet.audit: verify_chain read failed", exc_info=True)

    return (not broken), broken
