"""
memory.wiki_index_reconciler — Drift-scan + shadow-rebuild for wiki/index.md.

Adjacent track of the consciousness-roadmap (§4 of
`docs/CONSCIOUSNESS_ROADMAP.md`). The Anthropic-dreams parallel collapsed
to almost nothing once the existing 7 consolidation passes were
accounted for; this is the one operational gap that survived the audit.

Today, `wiki/index.md` is rebuilt event-driven by `WikiWriteTool` via
`_rebuild_master_index()` in `app.tools.wiki_tools` (called on
create / update / delete). It is **not** rebuilt by any idle pass. Out-of-
band changes — a manual file move, a Companion idea-promotion that fails
partway, a Compass component rename — leave the master index drifted from
on-disk truth without any signal.

This module closes the gap. It is read-only against `wiki/`, and on
detected drift it writes a CANDIDATE file to ``workspace/dreams/`` plus
an audit log entry. **It never overwrites the live index.** Adoption
flows through the existing change-request gate (Signal 👍 / `/cp/changes`)
— same shadow-output discipline as Anthropic's dreams, applied to the
narrowest gap.

Architecture:

  1. Compute canonical content via
     `app.tools.wiki_tools._compute_master_index_content()`. Pure function;
     no side effects on the wiki.
  2. Compare the canonical hash against the live `wiki/index.md` hash.
     `updated_at` in frontmatter is excluded from the comparison so a
     stale timestamp alone doesn't trigger drift.
  3. If hashes match: log "no drift" and exit cheaply.
  4. If hashes differ:
       a. Write the candidate to `workspace/dreams/wiki_index.candidate.md`.
       b. Append a hash-chained audit entry to
          `workspace/dreams/wiki_index_audit.jsonl`.
       c. Open a change-request via `app.change_requests.create_request()`
          with `requestor="wiki_index_reconciler"`. Adoption is
          operator-gated; this module never auto-applies.
  5. Concurrent-runs guard: a tiny lock file under `DREAMS_ROOT` prevents
     two simultaneous reconciler invocations from racing on candidate /
     audit writes. The lock is best-effort; expired lock files are taken
     over after `_LOCK_STALENESS_SECONDS`.

`superseded_by` invariant (lifted from `app.self_improvement.consolidator`):
audit entries describing a previous candidate are never deleted; a new
candidate that replaces an older one carries `supersedes` pointing at the
older audit entry's id. Adoption is therefore always reversible from the
audit chain.

Wiring: registered as LIGHT idle in `app.idle_scheduler` under
`wiki-index-reconciler`. Drift is rare; the LIGHT tier is appropriate
even for weekly cadence because the canonical compute over <100 pages is
sub-second.

Subsystem boundary: this module does NOT modify
`app/tools/wiki_tools.py:_rebuild_master_index` (event-driven path stays
intact) or any wiki page. It is a read-only drift detector with shadow-
output adoption discipline. TIER_IMMUTABLE-bypass is structurally
impossible because the only writer remains `_rebuild_master_index()`
(via `WikiWriteTool`) plus the change-request `apply_change()` path
gated on operator approval.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.paths import DREAMS_ROOT, WIKI_INDEX_AUDIT, WIKI_INDEX_CANDIDATE
from app.tools.wiki_tools import WIKI_ROOT, _compute_master_index_content

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

_LOCK_PATH = DREAMS_ROOT / ".wiki_index_reconciler.lock"
_LOCK_STALENESS_SECONDS = 600  # 10 min — well over a normal run

# Pin-down a fixed timestamp when computing the canonical content for hash
# comparison. The frontmatter `updated_at` field naturally drifts every
# time `_compute_master_index_content()` is called; pinning it removes that
# noise. The CANDIDATE file written to disk uses the live timestamp so
# operators see when the candidate was produced.
_HASH_PIN_TIME = datetime(2000, 1, 1, tzinfo=timezone.utc)
_UPDATED_AT_RE = re.compile(r"^updated_at:\s*'?[^'\n]*'?\s*$", re.MULTILINE)


# ── Result types ─────────────────────────────────────────────────────────


@dataclass
class DriftResult:
    """Outcome of one drift-scan pass."""
    drift_detected: bool
    live_hash: str
    canonical_hash: str
    live_size_bytes: int
    canonical_size_bytes: int
    audit_id: Optional[str] = None      # populated when drift_detected
    candidate_path: Optional[str] = None  # populated when drift_detected
    change_request_id: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "live_hash": self.live_hash[:16] if self.live_hash else None,
            "canonical_hash": self.canonical_hash[:16] if self.canonical_hash else None,
            "live_size_bytes": self.live_size_bytes,
            "canonical_size_bytes": self.canonical_size_bytes,
            "audit_id": self.audit_id,
            "candidate_path": self.candidate_path,
            "change_request_id": self.change_request_id,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
        }


# ── Pure helpers ─────────────────────────────────────────────────────────


def _normalize_for_hashing(content: str) -> str:
    """Strip the frontmatter `updated_at` line so timestamp drift alone
    doesn't trigger a false-positive drift signal.

    Everything else in the frontmatter (title, total_pages, sections)
    counts toward the hash — those changes ARE meaningful drift.
    """
    return _UPDATED_AT_RE.sub("updated_at: '<pinned>'", content)


def _content_hash(content: str) -> str:
    """SHA-256 of the normalized content (hex)."""
    return hashlib.sha256(_normalize_for_hashing(content).encode("utf-8")).hexdigest()


def compute_canonical_master_content(
    *, now: Optional[datetime] = None
) -> str:
    """Wrapper around `wiki_tools._compute_master_index_content`.

    Exposed at module level so callers (idle scheduler, tests) don't have
    to reach into a private helper of the wiki tools module.
    """
    return _compute_master_index_content(now=now or _HASH_PIN_TIME)


def read_live_master_index() -> tuple[str, str, int]:
    """Read the live `wiki/index.md`. Returns (content, hash, size_bytes).

    Returns ("", "", 0) if the file does not exist (which is itself a
    drift signal — the canonical content will be non-empty whenever any
    page exists).
    """
    live_path = Path(WIKI_ROOT) / "index.md"
    if not live_path.is_file():
        return "", "", 0
    content = live_path.read_text(encoding="utf-8")
    return content, _content_hash(content), len(content.encode("utf-8"))


# ── Lock helpers ─────────────────────────────────────────────────────────


def _acquire_lock() -> bool:
    """Best-effort lock to prevent concurrent reconciler runs.

    Returns True on acquisition, False if a fresh lock is already held.
    Stale locks (older than `_LOCK_STALENESS_SECONDS`) are taken over.
    """
    DREAMS_ROOT.mkdir(parents=True, exist_ok=True)
    if _LOCK_PATH.exists():
        try:
            age = time.time() - _LOCK_PATH.stat().st_mtime
            if age < _LOCK_STALENESS_SECONDS:
                return False
            logger.warning(
                "wiki_index_reconciler: stale lock (age=%.0fs); taking over",
                age,
            )
            _LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            return False
    try:
        _LOCK_PATH.write_text(str(os.getpid()))
        return True
    except OSError as exc:
        logger.debug("wiki_index_reconciler: lock write failed: %s", exc)
        return False


def _release_lock() -> None:
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("wiki_index_reconciler: lock release failed: %s", exc)


# ── Audit + candidate writers ────────────────────────────────────────────


def _last_audit_id() -> Optional[str]:
    """Return the id of the most recent audit entry, for `supersedes`
    chain lifting. None on first run.
    """
    if not WIKI_INDEX_AUDIT.is_file():
        return None
    try:
        with WIKI_INDEX_AUDIT.open("r", encoding="utf-8") as f:
            last_line = ""
            for line in f:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        prev = json.loads(last_line)
        return prev.get("id")
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("wiki_index_reconciler: audit-tail read failed: %s", exc)
        return None


def _append_audit(
    *,
    drift: DriftResult,
    candidate_content: str,
    change_request_id: Optional[str],
) -> str:
    """Append a hash-chained audit entry. Returns the entry's id."""
    DREAMS_ROOT.mkdir(parents=True, exist_ok=True)
    audit_id = uuid.uuid4().hex[:16]
    entry = {
        "id": audit_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "supersedes": _last_audit_id(),  # `superseded_by` invariant: chain entries, never delete
        "drift_detected": drift.drift_detected,
        "live_hash": drift.live_hash,
        "canonical_hash": drift.canonical_hash,
        "live_size_bytes": drift.live_size_bytes,
        "canonical_size_bytes": drift.canonical_size_bytes,
        "candidate_size_bytes": len(candidate_content.encode("utf-8")),
        "candidate_path": str(WIKI_INDEX_CANDIDATE),
        "change_request_id": change_request_id,
    }
    with WIKI_INDEX_AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return audit_id


def _write_candidate(content: str) -> None:
    """Write the candidate file. Overwrites the previous candidate;
    history is preserved in the audit chain.
    """
    DREAMS_ROOT.mkdir(parents=True, exist_ok=True)
    WIKI_INDEX_CANDIDATE.write_text(content, encoding="utf-8")


def _open_change_request(*, candidate_content: str, live_content: str, audit_id: str) -> Optional[str]:
    """Open a change-request to update `wiki/index.md`. Returns the
    request id on success, None on failure (lack of imports, validator
    rejection, etc.). Failure is non-fatal — the candidate file + audit
    entry are still written so the operator can act manually.
    """
    try:
        from app.change_requests import create_request
    except Exception as exc:
        logger.debug("wiki_index_reconciler: change_requests import failed: %s", exc)
        return None
    try:
        cr = create_request(
            requestor="wiki_index_reconciler",
            path="wiki/index.md",
            new_content=candidate_content,
            old_content=live_content,
            reason=(
                "wiki/index.md drift detected by the reconciler "
                f"(audit_id={audit_id}). The on-disk page set diverges "
                "from what the master index reflects. Approving this "
                "applies the canonical rebuild via the change-request "
                "gate; rejecting leaves the live index untouched."
            ),
        )
        return cr.id
    except Exception as exc:
        # Validator may reject (e.g. wiki/index.md ends up TIER_IMMUTABLE
        # in some future hardening) — that's the gate doing its job, not
        # an error we should escalate.
        logger.info("wiki_index_reconciler: create_request returned %s", exc)
        return None


# ── Public entry ──────────────────────────────────────────────────────────


def run_reconciler() -> DriftResult:
    """One drift-scan pass. Idempotent on no-drift; on drift, writes
    the candidate file + audit entry + (best-effort) change-request.

    Never modifies `wiki/index.md` directly. Returns a `DriftResult`
    describing what happened — useful for tests, the operator dashboard,
    and idle-scheduler logging.
    """
    if not _acquire_lock():
        return DriftResult(
            drift_detected=False,
            live_hash="",
            canonical_hash="",
            live_size_bytes=0,
            canonical_size_bytes=0,
            skipped=True,
            skip_reason="another reconciler run holds the lock",
        )

    try:
        live_content, live_hash, live_size = read_live_master_index()
        try:
            canonical_content = compute_canonical_master_content()
        except Exception as exc:
            logger.warning("wiki_index_reconciler: canonical compute failed: %s", exc)
            return DriftResult(
                drift_detected=False,
                live_hash=live_hash,
                canonical_hash="",
                live_size_bytes=live_size,
                canonical_size_bytes=0,
                error=f"canonical_compute_failed: {type(exc).__name__}",
            )
        canonical_hash = _content_hash(canonical_content)
        canonical_size = len(canonical_content.encode("utf-8"))

        result = DriftResult(
            drift_detected=(live_hash != canonical_hash),
            live_hash=live_hash,
            canonical_hash=canonical_hash,
            live_size_bytes=live_size,
            canonical_size_bytes=canonical_size,
        )

        if not result.drift_detected:
            logger.debug("wiki_index_reconciler: no drift (hash=%s)", live_hash[:12])
            return result

        # Drift detected — write candidate (with a fresh timestamp so the
        # operator sees when it was produced) + audit + open change-request.
        candidate_for_disk = _compute_master_index_content()  # uses now()
        _write_candidate(candidate_for_disk)
        cr_id = _open_change_request(
            candidate_content=candidate_for_disk,
            live_content=live_content,
            audit_id="<pending>",  # placeholder; audit id is allocated next
        )
        audit_id = _append_audit(
            drift=result,
            candidate_content=candidate_for_disk,
            change_request_id=cr_id,
        )
        result.audit_id = audit_id
        result.candidate_path = str(WIKI_INDEX_CANDIDATE)
        result.change_request_id = cr_id

        logger.info(
            "wiki_index_reconciler: drift detected — "
            "live_hash=%s canonical_hash=%s audit_id=%s cr=%s",
            live_hash[:12], canonical_hash[:12], audit_id, cr_id or "<none>",
        )
        return result

    finally:
        _release_lock()


# ── CLI ──────────────────────────────────────────────────────────────────


def _main() -> int:
    """`python -m app.memory.wiki_index_reconciler` — manual run."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_reconciler()
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if not result.error else 1


if __name__ == "__main__":
    raise SystemExit(_main())
