"""manifest — snapshot of identity-critical files for replication.

Q17.1. Walks a curated path set, computes SHA-256 + 1MB-prefix hash
per file. Output JSON is the canonical "what we need to keep alive
across substrate failure" list. Excludes cache/build/tmp/dotfiles.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


_CRITICAL_RELATIVE_PATHS: tuple[str, ...] = (
    "identity/continuity_ledger.jsonl",
    "subia/integrity_manifest.json",
    "subia/observations",
    "audit.log",
    "audit_journal.json",
    "resilience/drill_audit.jsonl",
    "coding_sessions/audit.jsonl",
    "change_requests",
    "governance",
    "vacation_mode",
    "epistemic",
    "episteme",
    "aesthetics",
    "tensions",
    "lessons_learned",
    "wiki",
    "companion",
    "self_model",
    "affect/trace.jsonl",
    "personality",
    "runtime_settings.json",
    "skills_registry.json",
    "web_push_subscriptions.json",
    "notes",
    "output",
    "skills",
    "warm_spare",
)

_EXCLUDE_DIRS = {"__pycache__", ".cache", "tmp", "node_modules", ".venv", "venv"}
_EXCLUDE_NAME_PREFIXES = {".", "_"}
_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
_PREFIX_HASH_BYTES = 1 << 20


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size: int
    mtime: float
    sha256: str | None
    prefix_sha256: str | None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _is_excluded(p: Path) -> bool:
    for part in p.parts:
        if part in _EXCLUDE_DIRS:
            return True
        if part[:1] in _EXCLUDE_NAME_PREFIXES and part not in (".",):
            if not part.startswith(".env."):
                return True
    return False


def _hash_file(p: Path) -> tuple[str | None, str | None, str]:
    try:
        size = p.stat().st_size
    except OSError:
        return None, None, "stat_failed"
    if size > _MAX_FILE_SIZE_BYTES:
        try:
            with open(p, "rb") as f:
                prefix = f.read(_PREFIX_HASH_BYTES)
            return None, hashlib.sha256(prefix).hexdigest(), f"large_{size}"
        except OSError:
            return None, None, "read_failed"
    try:
        with open(p, "rb") as f:
            data = f.read()
        full = hashlib.sha256(data).hexdigest()
        prefix = hashlib.sha256(data[:_PREFIX_HASH_BYTES]).hexdigest()
        return full, prefix, ""
    except OSError:
        return None, None, "read_failed"


def _walk_paths() -> Iterable[Path]:
    root = _workspace_root()
    for rel in _CRITICAL_RELATIVE_PATHS:
        abs_p = root / rel
        if not abs_p.exists():
            continue
        if abs_p.is_file():
            if _is_excluded(Path(rel)):
                continue
            yield abs_p
            continue
        for dirpath, dirnames, filenames in os.walk(abs_p):
            dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS and not d.startswith((".",))]
            for fn in filenames:
                if any(fn.startswith(pref) for pref in _EXCLUDE_NAME_PREFIXES):
                    continue
                yield Path(dirpath) / fn


def build_manifest() -> dict[str, Any]:
    root = _workspace_root()
    entries: list[ManifestEntry] = []
    for f in _walk_paths():
        try:
            rel = str(f.relative_to(root))
        except ValueError:
            rel = str(f)
        try:
            st = f.stat()
        except OSError:
            continue
        full, prefix, note = _hash_file(f)
        entries.append(ManifestEntry(
            path=rel,
            size=st.st_size,
            mtime=st.st_mtime,
            sha256=full,
            prefix_sha256=prefix,
            note=note,
        ))
    entries.sort(key=lambda e: e.path)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "n_files": len(entries),
        "total_bytes": sum(e.size for e in entries),
        "workspace_root": str(root),
        "entries": [e.to_dict() for e in entries],
    }


def _manifest_path() -> Path:
    return _workspace_root() / "warm_spare" / "manifest.json"


def save_manifest(manifest: dict[str, Any]) -> Path:
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def load_manifest() -> dict[str, Any] | None:
    p = _manifest_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("warm_spare.manifest: load failed", exc_info=True)
        return None
