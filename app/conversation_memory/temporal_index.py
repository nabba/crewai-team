"""temporal_index — periodic audit.log → searchable index (Q17.8)."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_LOCK = threading.Lock()
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s()-]{7,}\d")
_MAX_TOKENS_PER_ROW = 64
_MAX_PREVIEW_CHARS = 240


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _audit_path() -> Path:
    for c in (_workspace_root() / "audit.log", _workspace_root() / "audit_log.jsonl"):
        if c.exists():
            return c
    return _workspace_root() / "audit.log"


def _index_path() -> Path:
    return _workspace_root() / "conversation_memory" / "index.jsonl"


def _cursor_path() -> Path:
    return _workspace_root() / "conversation_memory" / "scan_cursor.json"


def _read_cursor() -> dict[str, Any]:
    p = _cursor_path()
    if not p.exists():
        return {"offset": 0, "last_scan": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"offset": 0, "last_scan": None}


def _write_cursor(state: dict[str, Any]) -> None:
    p = _cursor_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("temporal_index: cursor write failed", exc_info=True)


def _redact(text: str) -> str:
    text = _EMAIL_RE.sub("<email>", text or "")
    text = _PHONE_RE.sub("<phone>", text)
    return text


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")][:_MAX_TOKENS_PER_ROW]


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_conversation_memory_enabled
        return get_conversation_memory_enabled()
    except Exception:
        return True


def _extract_message_body(row: dict[str, Any]) -> str:
    for k in ("message", "body", "text", "content", "prompt", "response"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _row_kind(row: dict[str, Any]) -> str:
    for k in ("kind", "event", "type", "action"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return "unknown"


def _row_ts(row: dict[str, Any]) -> str:
    for k in ("ts", "timestamp", "time", "at"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return datetime.now(timezone.utc).isoformat()


def _append_index_row(row: dict[str, Any]) -> None:
    p = _index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except OSError:
        logger.debug("temporal_index: append failed", exc_info=True)


def scan_audit_log(*, max_lines: int = 5000) -> dict[str, Any]:
    summary = {"n_scanned": 0, "n_indexed": 0, "errors": 0}
    if not _enabled():
        summary["skipped"] = True
        return summary
    audit = _audit_path()
    if not audit.exists():
        summary["no_audit_log"] = True
        return summary
    with _LOCK:
        cursor = _read_cursor()
        offset = int(cursor.get("offset") or 0)
        try:
            with open(audit, "r", encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                lines_read = 0
                for line in f:
                    if lines_read >= max_lines:
                        break
                    lines_read += 1
                    line = line.strip()
                    if not line:
                        continue
                    summary["n_scanned"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    body = _extract_message_body(row)
                    if not body:
                        continue
                    redacted = _redact(body)
                    toks = _tokens(redacted)
                    if not toks:
                        continue
                    _append_index_row({
                        "ts": _row_ts(row),
                        "kind": _row_kind(row),
                        "tokens": toks,
                        "preview": redacted[:_MAX_PREVIEW_CHARS],
                        "ref": row.get("id") or row.get("request_id") or row.get("uuid"),
                    })
                    summary["n_indexed"] += 1
                cursor["offset"] = f.tell()
        except OSError:
            summary["errors"] += 1
            return summary
        cursor["last_scan"] = datetime.now(timezone.utc).isoformat()
        _write_cursor(cursor)
    return summary


def rebuild_index() -> dict[str, Any]:
    for p in (_cursor_path(), _index_path()):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass
    return scan_audit_log(max_lines=10_000_000)
