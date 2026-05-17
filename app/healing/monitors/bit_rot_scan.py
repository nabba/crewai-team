"""bit_rot_scan — silent data-corruption detector (Q17.3).

Daily probe, weekly internal cadence. Walks an 11-path identity-
critical set. Records SHA-256 + 1MB-prefix hash + line count per
file. Detects: shrunk / inplace_mutated / prefix_mutated.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NAME = "bit_rot_scan"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "bit_rot_scan_enabled"
_INTERNAL_CADENCE_S = 7 * 24 * 3600
_PREFIX_HASH_BYTES = 1 << 20

_BASELINE_FILE = "bit_rot_baseline.json"
_STATE_FILE = "bit_rot_state.json"


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _critical_paths() -> list[Path]:
    root = _workspace_root()
    candidates = [
        root / "identity" / "continuity_ledger.jsonl",
        root / "audit.log",
        root / "audit_journal.json",
        root / "resilience" / "drill_audit.jsonl",
        root / "coding_sessions" / "audit.jsonl",
        root / "change_requests" / "audit.jsonl",
        root / "epistemic" / "claims.jsonl",
        root / "epistemic" / "overrides.jsonl",
        root / "affect" / "trace.jsonl",
        root / "subia" / "integrity_manifest.json",
        root / "self_model" / "agreement_ledger.jsonl",
    ]
    # PROGRAM §56 — include every KB's source ledger. These are
    # identity-critical: they're the canonical source from which the
    # ChromaDB KBs can be reconstructed. Bit-rot here = silent data
    # loss that survives every other protection layer.
    try:
        for p in root.glob("*/.source_ledger.jsonl"):
            # Same filter chromadb_integrity uses — skip quarantined
            # snapshots so we don't false-alarm on intentionally
            # frozen historical files.
            parent = p.parent
            if any(seg in parent.name for seg in ("corrupt_", "bak_", "_backup", ".backup")):
                continue
            candidates.append(p)
    except Exception:
        pass
    return [p for p in candidates if p.exists()]


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _read_prefix(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    size = stat.st_size
    prefix = _read_prefix(path, min(size, _PREFIX_HASH_BYTES))
    line_count = prefix.count(b"\n") if size <= _PREFIX_HASH_BYTES else None
    if size <= _PREFIX_HASH_BYTES:
        with open(path, "rb") as f:
            data = f.read()
        return {
            "size": size,
            "mtime": stat.st_mtime,
            "sha256_full": _sha256_of_bytes(data),
            "sha256_prefix": _sha256_of_bytes(prefix),
            "line_count": line_count,
        }
    return {
        "size": size,
        "mtime": stat.st_mtime,
        "sha256_full": None,
        "sha256_prefix": _sha256_of_bytes(prefix),
        "line_count": None,
    }


def _classify_change(prev: dict[str, Any], cur: dict[str, Any], path: Path | None = None) -> str:
    if prev is None:
        return "new"
    if cur["size"] < prev["size"]:
        return "shrunk"
    if (prev.get("line_count") is not None
            and cur.get("line_count") is not None
            and cur["line_count"] < prev["line_count"]):
        return "shrunk"
    if cur["size"] == prev["size"]:
        if prev.get("sha256_full") and cur.get("sha256_full"):
            if cur["sha256_full"] != prev["sha256_full"]:
                return "inplace_mutated"
        elif cur["sha256_prefix"] != prev["sha256_prefix"]:
            return "inplace_mutated"
        return "ok"
    if path is not None:
        try:
            with open(path, "rb") as f:
                old_size_bytes = f.read(prev["size"])
            old_prefix_window = min(prev["size"], _PREFIX_HASH_BYTES)
            recomputed_prefix = _sha256_of_bytes(old_size_bytes[:old_prefix_window])
            ref_prefix = prev.get("sha256_full") if prev["size"] <= _PREFIX_HASH_BYTES else prev.get("sha256_prefix")
            if ref_prefix is not None and recomputed_prefix != ref_prefix:
                return "prefix_mutated"
            return "append_ok"
        except OSError:
            return "append_ok"
    return "append_ok"


def _read_baseline() -> dict[str, dict[str, Any]]:
    p = _workspace_root() / "healing" / _BASELINE_FILE
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_baseline(baseline: dict[str, dict[str, Any]]) -> None:
    p = _workspace_root() / "healing" / _BASELINE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("bit_rot_scan: baseline write failed", exc_info=True)


def _read_state() -> dict[str, Any]:
    p = _workspace_root() / "healing" / _STATE_FILE
    if not p.exists():
        return {"last_run": 0, "outstanding_alerts": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": 0, "outstanding_alerts": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _workspace_root() / "healing" / _STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("bit_rot_scan: state write failed", exc_info=True)


def _emit_alert(path_key: str, change: str, detail: dict[str, Any]) -> None:
    try:
        from app.notify import notify
        notify(
            title="🧬 Bit rot suspected",
            body=f"Silent integrity violation on {path_key}\nType: {change}",
            url="/cp/monitor",
            topic=f"bit_rot:{path_key}",
            critical=True,
            arbitrate=False,
        )
    except Exception:
        logger.debug("bit_rot_scan: notify failed", exc_info=True)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="q17_landmark",
            actor="bit_rot_scan",
            summary=f"bit-rot suspected on {path_key} ({change})",
            detail={"subsystem": "bit_rot", "path": path_key, "change": change, **detail},
        )
    except Exception:
        logger.debug("bit_rot_scan: ledger emit failed", exc_info=True)


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_bit_rot_scan_enabled
        return get_bit_rot_scan_enabled()
    except Exception:
        return True


def _cadence_due(state: dict[str, Any]) -> bool:
    last = float(state.get("last_run") or 0)
    return (datetime.now(timezone.utc).timestamp() - last) >= _INTERNAL_CADENCE_S


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "checked": False, "n_files": 0, "n_new": 0, "n_append_ok": 0,
        "n_ok": 0, "alerts": [], "errors": 0,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary
    state = _read_state()
    if not _cadence_due(state):
        summary["skipped_cadence"] = True
        return summary
    try:
        baseline = _read_baseline()
        files = _critical_paths()
        summary["n_files"] = len(files)
        root = _workspace_root()
        for f in files:
            try:
                key = str(f.relative_to(root))
            except ValueError:
                key = str(f)
            try:
                cur = _file_fingerprint(f)
            except OSError:
                summary["errors"] += 1
                continue
            prev = baseline.get(key)
            change = _classify_change(prev, cur, path=f)
            if change == "new":
                summary["n_new"] += 1
                baseline[key] = cur
            elif change == "append_ok":
                summary["n_append_ok"] += 1
                baseline[key] = cur
            elif change == "ok":
                summary["n_ok"] += 1
                baseline[key] = cur
            else:
                summary["alerts"].append({"path": key, "change": change, "prev_size": prev.get("size"), "cur_size": cur.get("size")})
                _emit_alert(key, change, {"prev_size": prev.get("size"), "cur_size": cur.get("size")})
        _write_baseline(baseline)
        state["last_run"] = datetime.now(timezone.utc).timestamp()
        _write_state(state)
        summary["checked"] = True
    except Exception:
        logger.debug("bit_rot_scan: probe failed", exc_info=True)
        summary["errors"] += 1
    return summary
