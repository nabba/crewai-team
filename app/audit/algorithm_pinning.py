"""Cryptographic algorithm pinning + rotation drill (§2.1).

The system pins SHA-256 across:
  - SubIA integrity manifest (``app/subia/.integrity_manifest.json``)
  - Rolled audit-log hash chains (:mod:`app.audit.rolled_log`)
  - Tier-3 amendment audit chain (``app/governance_amendment/``)
  - Coding-session audit chain
  - Change-request audit chain

SHA-256 is fine for the next ~10 years against any expected adversary
class. But the *operational* discipline of rotating algorithms — when
SHA-3 supersedes SHA-256, or quantum-resistant hashes ship, or a flaw
in SHA-256 is discovered — doesn't yet exist in the codebase. Without
discipline, year 7's operator finds out the system has no migration
path the day they need one.

This module is the discipline:

  1. **Manifest of which algorithm is pinned where.**
     ``workspace/audit/algorithm_pinning.json`` records, per-artifact-
     class, the algorithm currently in use plus when it was pinned.
     Operator-curated.

  2. **Rotation drill (annual cadence).**
     Given a target algorithm (e.g. SHA-3-256 or BLAKE3), walk a
     rolled-audit chain twice — once under the legacy algorithm, once
     under the target — and prove that BOTH compute consistent state.
     If the drill passes, the operator can confidently switch. If it
     fails (e.g. a target hash function isn't available in the runtime),
     the operator finds out years before the actual rotation deadline.

  3. **Weakness probe.**
     Read the manifest; flag any algorithm whose pinned date is older
     than ``ALGORITHM_REVIEW_INTERVAL_DAYS`` (default 730 = 2 years)
     so the operator periodically re-confirms the choice is still
     considered strong.

This is light infrastructure — a manifest + two pure functions + a
monitor. Production rotation involves coordinated changes across
multiple TIER_IMMUTABLE files; this module's job is *operator
notification + drill confidence*, not auto-rotation.

Master switch: ``ALGORITHM_PINNING_ENABLED`` (default ``true``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


_DEFAULT_MANIFEST_PATH = Path("/app/workspace/audit/algorithm_pinning.json")
_DEFAULT_REVIEW_INTERVAL_DAYS = 730  # 2 years

# Canonical artifact-class names. Adding new entries when a new
# hash-using subsystem ships keeps the manifest comprehensive.
KNOWN_ARTIFACT_CLASSES: frozenset[str] = frozenset({
    "subia_integrity_manifest",
    "rolled_audit_log",
    "tier3_amendment_audit",
    "coding_session_audit",
    "change_request_audit",
    "architecture_request_audit",
})


def _enabled() -> bool:
    return os.getenv("ALGORITHM_PINNING_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


@dataclass(frozen=True)
class AlgorithmPin:
    """One artifact-class → algorithm record."""

    artifact_class: str
    algorithm: str  # canonical hashlib name: "sha256", "sha3_256", "blake2b", ...
    pinned_at: str  # ISO-8601 UTC
    rationale: str = ""


@dataclass(frozen=True)
class RotationDrillResult:
    """Outcome of one rotation drill."""

    artifact_class: str
    legacy_algorithm: str
    target_algorithm: str
    legacy_chain_root: str
    target_chain_root: str
    n_entries: int
    ok: bool
    error: str = ""


# ── Manifest I/O ──────────────────────────────────────────────────────


def _read_manifest(path: Path) -> dict:
    if not path.exists():
        return {"pins": [], "version": 1}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("algorithm_pinning: manifest unreadable", exc_info=True)
        return {"pins": [], "version": 1}
    if not isinstance(data, dict):
        return {"pins": [], "version": 1}
    return data


def _write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def list_pins(*, path: Path | str | None = None) -> list[AlgorithmPin]:
    p = Path(path) if path else _DEFAULT_MANIFEST_PATH
    data = _read_manifest(p)
    out: list[AlgorithmPin] = []
    for raw in data.get("pins", []):
        try:
            out.append(AlgorithmPin(
                artifact_class=raw["artifact_class"],
                algorithm=raw["algorithm"],
                pinned_at=raw["pinned_at"],
                rationale=raw.get("rationale", ""),
            ))
        except (KeyError, TypeError):
            continue
    return out


def pin_algorithm(
    artifact_class: str,
    algorithm: str,
    *,
    rationale: str = "",
    path: Path | str | None = None,
    now: datetime | None = None,
) -> AlgorithmPin:
    """Record (or replace) the pin for one artifact class.

    Validates that ``algorithm`` is a real :mod:`hashlib` algorithm
    name. Latest write per artifact_class wins on read.
    """
    if not _enabled():
        raise RuntimeError("ALGORITHM_PINNING_ENABLED=false")
    try:
        hashlib.new(algorithm)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unknown algorithm {algorithm!r}: {exc}")
    if not artifact_class or not artifact_class.strip():
        raise ValueError("artifact_class must be non-empty")

    p = Path(path) if path else _DEFAULT_MANIFEST_PATH
    data = _read_manifest(p)
    pins = data.setdefault("pins", [])
    pinned_at = (now or datetime.now(timezone.utc)).isoformat()
    pin = AlgorithmPin(
        artifact_class=artifact_class.strip(),
        algorithm=algorithm,
        pinned_at=pinned_at,
        rationale=rationale.strip(),
    )
    # Replace any prior pin for the same class.
    pins = [
        p for p in pins
        if p.get("artifact_class") != pin.artifact_class
    ]
    pins.append(asdict(pin))
    data["pins"] = pins
    _write_manifest(p, data)
    return pin


# ── Weakness probe ────────────────────────────────────────────────────


def _days_since(iso: str, now: datetime | None = None) -> float | None:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    cur = now or datetime.now(timezone.utc)
    return (cur - dt).total_seconds() / 86400.0


def stale_pins(
    *,
    interval_days: int | None = None,
    path: Path | str | None = None,
    now: datetime | None = None,
) -> list[AlgorithmPin]:
    """Return pins whose ``pinned_at`` is older than the review interval.

    Also returns pins with unparseable timestamps (treated as ancient)
    so the operator can fix the manifest entry.
    """
    threshold = interval_days or _DEFAULT_REVIEW_INTERVAL_DAYS
    out: list[AlgorithmPin] = []
    for pin in list_pins(path=path):
        days = _days_since(pin.pinned_at, now)
        if days is None or days > threshold:
            out.append(pin)
    return out


def missing_artifact_classes(*, path: Path | str | None = None) -> list[str]:
    """Return artifact classes from KNOWN_ARTIFACT_CLASSES that have NO
    pin yet — operator hasn't recorded the algorithm in use."""
    pinned = {p.artifact_class for p in list_pins(path=path)}
    return sorted(KNOWN_ARTIFACT_CLASSES - pinned)


# ── Rotation drill ────────────────────────────────────────────────────


HashFn = Callable[[bytes], str]


def _hash_factory(algorithm: str) -> HashFn:
    # Probe the algorithm now so an unsupported name fails at factory
    # creation rather than mid-walk — the operator sees "not available
    # in runtime" instead of "chain walk raised".
    try:
        hashlib.new(algorithm)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"algorithm not available in runtime: {algorithm!r} ({exc})")

    def h(data: bytes) -> str:
        m = hashlib.new(algorithm)
        m.update(data)
        return m.hexdigest()
    return h


def _walk_chain(
    entries: Iterable[bytes],
    hash_fn: HashFn,
    *,
    genesis: str = "GENESIS",
) -> tuple[str, int]:
    """Walk a sequence of entries, computing a hash chain. Returns
    (final_root_hash, n_entries)."""
    prev = genesis
    n = 0
    for entry in entries:
        # Each step's hash depends on the previous root + the entry,
        # mirroring the rolled_log semantics (line-level hash chain).
        prev = hash_fn(prev.encode("utf-8") + b"|" + entry)
        n += 1
    return prev, n


def run_rotation_drill(
    artifact_class: str,
    *,
    legacy_algorithm: str = "sha256",
    target_algorithm: str = "sha3_256",
    sample_entries: list[bytes] | None = None,
) -> RotationDrillResult:
    """Walk a sample chain under both algorithms; return diagnostic.

    The drill *doesn't* mutate any manifest — it proves the runtime
    can compute hashes under the target algorithm, that the chain
    semantics are deterministic (same entries → reproducible state),
    and that the target algorithm's output differs from the legacy's
    (sanity check that we're really computing something new).

    ``sample_entries`` is injectable for tests. Production callers
    pass a slice of an actual rolled-log file.
    """
    samples = sample_entries or [
        b"genesis-sample",
        b"second-entry",
        b"third-entry-with-payload",
    ]
    try:
        legacy_fn = _hash_factory(legacy_algorithm)
        target_fn = _hash_factory(target_algorithm)
    except ValueError as exc:
        return RotationDrillResult(
            artifact_class=artifact_class,
            legacy_algorithm=legacy_algorithm,
            target_algorithm=target_algorithm,
            legacy_chain_root="",
            target_chain_root="",
            n_entries=0,
            ok=False,
            error=f"algorithm not available in runtime: {exc}",
        )

    try:
        legacy_root, n1 = _walk_chain(samples, legacy_fn)
        target_root, n2 = _walk_chain(samples, target_fn)
    except Exception as exc:  # noqa: BLE001
        return RotationDrillResult(
            artifact_class=artifact_class,
            legacy_algorithm=legacy_algorithm,
            target_algorithm=target_algorithm,
            legacy_chain_root="",
            target_chain_root="",
            n_entries=0,
            ok=False,
            error=f"chain walk raised: {exc}",
        )

    if n1 != n2:
        return RotationDrillResult(
            artifact_class=artifact_class,
            legacy_algorithm=legacy_algorithm,
            target_algorithm=target_algorithm,
            legacy_chain_root=legacy_root,
            target_chain_root=target_root,
            n_entries=n1,
            ok=False,
            error=f"entry-count mismatch ({n1} vs {n2})",
        )

    if legacy_root == target_root:
        return RotationDrillResult(
            artifact_class=artifact_class,
            legacy_algorithm=legacy_algorithm,
            target_algorithm=target_algorithm,
            legacy_chain_root=legacy_root,
            target_chain_root=target_root,
            n_entries=n1,
            ok=False,
            error=(
                f"target output identical to legacy — "
                f"{target_algorithm} likely aliased to {legacy_algorithm} "
                f"in this runtime"
            ),
        )

    # Determinism check: re-walking should yield the same roots.
    legacy_root_2, _ = _walk_chain(samples, legacy_fn)
    target_root_2, _ = _walk_chain(samples, target_fn)
    if legacy_root != legacy_root_2 or target_root != target_root_2:
        return RotationDrillResult(
            artifact_class=artifact_class,
            legacy_algorithm=legacy_algorithm,
            target_algorithm=target_algorithm,
            legacy_chain_root=legacy_root,
            target_chain_root=target_root,
            n_entries=n1,
            ok=False,
            error="non-deterministic chain walk — algorithm not pure",
        )

    return RotationDrillResult(
        artifact_class=artifact_class,
        legacy_algorithm=legacy_algorithm,
        target_algorithm=target_algorithm,
        legacy_chain_root=legacy_root,
        target_chain_root=target_root,
        n_entries=n1,
        ok=True,
    )
