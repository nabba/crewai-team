"""Read-only verifier for the coding-session hash-chained audit log.

Mirrors ``app.governance_amendment.audit.verify_chain`` but for the
older audit format in ``app.coding_session.store``. The chain shape:

    {
      "ts": "...",
      "prev_hash": "<16 hex chars>",
      "entry_hash": "<16 hex chars>",
      "payload": {...}
    }

where ``entry_hash = sha256(prev_hash + json.dumps(payload, sort_keys=True))[:16]``.
``prev_hash`` of the first entry is the empty string (matching the
genesis convention from ``store._last_audit_hash``).

This module is purely read-only and does NOT modify the chain. If the
chain is broken, ``verify_chain`` reports the broken lines; an
operator decides how to recover (file integrity is a human call).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mirror the path constant from store.py — keep them in sync if it
# ever moves. We don't import store directly because verifying the
# chain shouldn't trigger the lazy index load.
_AUDIT_LOG = Path("/app/workspace/coding_sessions/audit.jsonl")


def _expected_entry_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    """Recompute ``entry_hash`` exactly as ``store._append_audit`` does."""
    body = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(
        (prev_hash + body).encode("utf-8"),
    ).hexdigest()[:16]


def verify_chain(
    *, audit_path: Path | str | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Walk the JSONL chain forward from genesis.

    Returns ``(ok, broken)``. ``broken`` is a list of dicts describing
    each defect (missing file, malformed JSON, prev-hash mismatch,
    entry-hash mismatch). ``ok`` is True iff the list is empty.

    Empty / missing chain returns ``(True, [])`` — vacuously valid.
    """
    path = Path(audit_path) if audit_path else _AUDIT_LOG
    if not path.exists():
        return True, []

    broken: list[dict[str, Any]] = []
    prev_hash = ""  # genesis — matches store._last_audit_hash empty case
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as exc:
                    broken.append({
                        "line_no": line_no,
                        "reason": "invalid_json",
                        "detail": str(exc),
                    })
                    continue

                claimed_prev = rec.get("prev_hash", "")
                claimed_hash = rec.get("entry_hash", "")
                payload = rec.get("payload")

                if not isinstance(payload, dict):
                    broken.append({
                        "line_no": line_no,
                        "reason": "missing_payload",
                    })
                    # Don't try to recompute — just skip ahead.
                    prev_hash = claimed_hash or prev_hash
                    continue

                if claimed_prev != prev_hash:
                    broken.append({
                        "line_no": line_no,
                        "reason": "prev_hash_mismatch",
                        "claimed_prev": claimed_prev,
                        "expected_prev": prev_hash,
                    })

                expected = _expected_entry_hash(claimed_prev or "", payload)
                if expected != claimed_hash:
                    broken.append({
                        "line_no": line_no,
                        "reason": "entry_hash_mismatch",
                        "claimed": claimed_hash,
                        "expected": expected,
                    })

                # Always advance to the claimed hash so we report each
                # break independently rather than cascading every line
                # after the first defect.
                prev_hash = claimed_hash or prev_hash
    except Exception as exc:
        logger.debug("audit_verify: read failed for %s", path, exc_info=True)
        broken.append({
            "line_no": 0,
            "reason": "read_error",
            "detail": str(exc),
        })

    return (not broken), broken


def chain_summary(audit_path: Path | str | None = None) -> dict[str, Any]:
    """Compact summary for dashboards / Signal alerts.

    Returns:
        ``{"path", "exists", "lines", "ok", "broken_count",
           "first_break_line", "last_entry_hash"}``
    """
    path = Path(audit_path) if audit_path else _AUDIT_LOG
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "lines": 0,
        "ok": True,
        "broken_count": 0,
        "first_break_line": None,
        "last_entry_hash": "",
    }
    if not path.exists():
        return summary

    # Count lines + capture the last entry hash for at-a-glance display.
    try:
        with path.open("r", encoding="utf-8") as f:
            last_line = ""
            for line in f:
                summary["lines"] += 1
                if line.strip():
                    last_line = line.strip()
        if last_line:
            try:
                summary["last_entry_hash"] = (
                    json.loads(last_line).get("entry_hash", "")
                )
            except Exception:
                pass
    except Exception:
        logger.debug("audit_verify: line-count read failed", exc_info=True)

    ok, broken = verify_chain(audit_path=path)
    summary["ok"] = ok
    summary["broken_count"] = len(broken)
    if broken:
        summary["first_break_line"] = broken[0].get("line_no")
        summary["first_break_reason"] = broken[0].get("reason")

    return summary
