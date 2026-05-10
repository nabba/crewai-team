"""
subia.integrity — canonical SubIA integrity manifest.

Phase 3 hardening. The existing SafetyGuardian.enforce_tier_boundaries
machinery baselines checksums on first boot and then detects drift
from that baseline. That catches tampering after deploy but not a
compromised deploy (e.g. an attacker modifies a file and restarts
before the baseline is saved).

This module adds a canonical, in-repo manifest — `app/subia/.integrity_manifest.json`
— whose entries are generated from committed source and shipped in
git. At startup the manifest is compared to the live file hashes.
Drift from the committed baseline is a hard fault:

    MANIFEST match   → proceed
    MISSING file     → fail loud
    HASH mismatch    → fail loud

Drift is not automatically silenced by the "first-boot baselining"
of the older mechanism. That older mechanism still runs — it catches
runtime tampering. This one catches deploy-time tampering.

Usage:
    from app.subia.integrity import (
        compute_manifest,      # dev/CI: regenerate the manifest
        load_manifest,         # read from disk
        verify_integrity,      # compare live files to manifest
        write_manifest,        # dev/CI: save regenerated manifest
    )

The module is intentionally infrastructure-level with no imports from
the rest of the SubIA tree — if something goes wrong deeper in the
consciousness stack, integrity verification must still be runnable.

See PROGRAM.md Phase 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ── Manifest location (in-repo, ships with code) ──────────────────

# Repo root discovered relative to this file: app/subia/integrity.py
# parents[0] = app/subia
# parents[1] = app
# parents[2] = <repo-root>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "app" / "subia" / ".integrity_manifest.json"

# Files covered by the canonical manifest: every file under app/subia/
# that is part of the consciousness infrastructure. We exclude
# __pycache__ and the manifest itself.
_DEFAULT_INCLUDE_ROOT = _REPO_ROOT / "app" / "subia"


@dataclass
class IntegrityResult:
    """Structured outcome of a verify_integrity call."""
    ok: bool = True
    manifest_version: int = 1
    n_files: int = 0
    missing: list = field(default_factory=list)       # declared, not on disk
    extra: list = field(default_factory=list)         # on disk, not declared
    mismatched: list = field(default_factory=list)    # declared, hash differs

    @property
    def has_drift(self) -> bool:
        return bool(self.missing or self.mismatched)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "manifest_version": self.manifest_version,
            "n_files": self.n_files,
            "missing": list(self.missing),
            "extra": list(self.extra),
            "mismatched": list(self.mismatched),
            "has_drift": self.has_drift,
        }


class IntegrityFault(RuntimeError):
    """Raised by verify_integrity(strict=True) when drift is detected."""


# ── Manifest API ──────────────────────────────────────────────────

_MANIFEST_VERSION = 1


def _iter_covered_files(root: Path) -> Iterable[Path]:
    """Yield every .py file under root, excluding __pycache__ and
    the manifest itself. Order is deterministic (sorted).
    """
    if not root.exists():
        return
    files: list[Path] = []
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        files.append(p)
    # Also include the manifest target's parent JSON sibling if present.
    for p in sorted(files):
        yield p


def _hash_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(
    root: Path | str | None = None,
    repo_root: Path | str | None = None,
) -> dict:
    """Generate a manifest for every .py file under `root`.

    Args:
        root:       Directory to cover. Defaults to <repo>/app/subia.
        repo_root:  Root against which relative paths are computed.
                    Defaults to the detected repo root.

    Returns a dict of shape:
        {
          "version": 1,
          "files": {
            "app/subia/<path>.py": {"sha256": "<hex>", "size": <int>},
            ...
          }
        }
    """
    repo = Path(repo_root) if repo_root else _REPO_ROOT
    tree = Path(root) if root else _DEFAULT_INCLUDE_ROOT

    entries: dict = {}
    for path in _iter_covered_files(tree):
        try:
            rel = str(path.relative_to(repo))
        except ValueError:
            # Path outside repo root — skip defensively.
            continue
        digest = _hash_file(path)
        entries[rel] = {
            "sha256": digest,
            "size": path.stat().st_size,
        }

    return {
        "version": _MANIFEST_VERSION,
        "files": entries,
    }


def load_manifest(manifest_path: Path | str | None = None) -> dict | None:
    """Read the canonical manifest from disk. Returns None if absent.

    Never raises: a malformed JSON file is treated as a missing manifest.
    """
    path = Path(manifest_path) if manifest_path else _MANIFEST_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("subia integrity: failed to load manifest (%s)", exc)
        return None


def write_manifest(
    manifest: dict,
    manifest_path: Path | str | None = None,
) -> None:
    """Write the manifest atomically. Used by regeneration tooling
    (dev or CI), never at runtime from application code.

    Emits an ``integrity_regen`` event into the identity continuity
    ledger so the annual reflection sees when the integrity boundary
    was last refreshed. Failure-isolated: a missing identity package
    never blocks the manifest write.
    """
    path = Path(manifest_path) if manifest_path else _MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(
            kind="integrity_regen",
            actor="dev_or_ci",
            summary=f"regenerated SubIA integrity manifest ({len(manifest.get('files', {}))} files)",
            detail={"n_files": len(manifest.get("files", {}))},
        )
    except Exception:
        logger.debug("identity ledger emission failed", exc_info=True)


def verify_integrity(
    manifest: dict | None = None,
    repo_root: Path | str | None = None,
    strict: bool = False,
) -> IntegrityResult:
    """Compare live `app/subia/**` hashes to a manifest.

    Args:
        manifest:   Parsed manifest dict. If None, loaded from disk.
                    If still None (no manifest present on disk),
                    returns an ok=False result with everything listed
                    as missing (rather than silently passing — that
                    was the safety bug of the older baseline scheme).
        repo_root:  Root against which manifest paths are resolved.
        strict:     When True, raises IntegrityFault on any drift.
                    When False (default), returns the structured
                    IntegrityResult for the caller to act on.

    Returns an IntegrityResult. Never silently succeeds in the
    absence of a manifest.
    """
    repo = Path(repo_root) if repo_root else _REPO_ROOT
    if manifest is None:
        manifest = load_manifest()

    if manifest is None:
        result = IntegrityResult(
            ok=False,
            manifest_version=0,
            missing=["<MANIFEST>"],
        )
        if strict:
            raise IntegrityFault(
                "SubIA integrity manifest not found — refusing to run"
            )
        return result

    declared = manifest.get("files", {})
    result = IntegrityResult(
        manifest_version=int(manifest.get("version", 0)),
        n_files=len(declared),
    )

    # Check declared files against live disk.
    for rel, entry in declared.items():
        full = repo / rel
        if not full.exists():
            result.missing.append(rel)
            continue
        live = _hash_file(full)
        expected = entry.get("sha256", "")
        if live != expected:
            result.mismatched.append({
                "file": rel,
                "expected": expected,
                "actual": live,
            })

    # Check for files present on disk but not declared. Only scan
    # files that sit under `repo` — if the caller overrode the repo
    # root (e.g. tests point at a temp dir), the default include
    # root may not be relative to it and we skip the check rather
    # than raising.
    include_root = repo / "app" / "subia"
    if include_root.exists():
        live_files = set()
        for p in _iter_covered_files(include_root):
            try:
                rel = str(p.relative_to(repo))
            except ValueError:
                continue
            if rel.endswith(".integrity_manifest.json"):
                continue
            live_files.add(rel)
        for rel in live_files - set(declared):
            result.extra.append(rel)

    result.ok = not (result.missing or result.mismatched)

    if strict and not result.ok:
        raise IntegrityFault(
            f"SubIA integrity drift: {len(result.missing)} missing, "
            f"{len(result.mismatched)} mismatched"
        )
    return result
