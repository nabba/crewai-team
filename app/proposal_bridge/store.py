"""Staging-area persistence for the proposal bridge.

Files live at ``workspace/proposal_bridge/<source>/<signature>.{md,json}``.
The ``.md`` is the body that lands in the repo on CR approval; the
``.json`` is a sibling metadata record (``ProposalState`` serialised).

Design notes:
  * One pair per (source, signature). The signature is caller-supplied
    and stable across producer runs — same cluster / discovery /
    paper produces the same signature.
  * Atomic writes: tempfile + ``replace`` so a crashed half-write
    can't leave a corrupt JSON.
  * ``stage()`` is idempotent on body content. Re-staging the same
    body is a no-op (preserves cooldown clock). Re-staging a
    DIFFERENT body for the same signature bumps the cooldown clock —
    the producer thinks the proposal evolved enough to re-evaluate.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


# ── Layout ────────────────────────────────────────────────────────────────


_DEFAULT_BASE = Path("/app/workspace/proposal_bridge")


def _base_dir() -> Path:
    """Override with PROPOSAL_BRIDGE_DIR for tests."""
    override = os.getenv("PROPOSAL_BRIDGE_DIR")
    if override:
        return Path(override)
    return _DEFAULT_BASE


# Caller-supplied source labels are accepted strict — only known
# labels are allowed so a typo doesn't silently create a parallel
# staging tree.
_KNOWN_SOURCES: frozenset[str] = frozenset({
    "capability_gap",
    "library_radar",
    "paper_pipeline",
    "dependency_radar",  # PROGRAM §48 — Q13.2 (year-2+ resilience #2.3)
})

_SAFE_SIG_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


# ── Types ────────────────────────────────────────────────────────────────


class ProposalStatus(str, enum.Enum):
    STAGED = "staged"           # body written, in cooldown
    CR_FILED = "cr_filed"       # CR submitted; awaiting operator
    APPLIED = "applied"         # CR approved + applied
    REJECTED = "rejected"       # CR rejected or invalidated
    EXPIRED = "expired"         # never resolved within window


@dataclass
class ProposalState:
    source: str
    signature: str
    title: str
    target_path: str
    body_hash: str            # sha256 hex of body_markdown at stage time
    staged_at: str            # ISO-8601 UTC
    status: ProposalStatus = ProposalStatus.STAGED
    cooldown_days: int = 7
    cr_id: Optional[str] = None
    cr_filed_at: Optional[str] = None
    resolved_at: Optional[str] = None  # applied OR rejected
    expired_at: Optional[str] = None
    notes: dict[str, Any] = field(default_factory=dict)
    # Q2 §39: optional coding-session spec for non-Tier-3 proposals.
    # When present, the promoter renders it as a YAML block in the
    # CR body so an agent (or operator copy-pasting into a chat with
    # the coder) has a concrete scaffold for actually implementing
    # the proposal. Tier-3-targeted proposals never get a spec —
    # those paths route through governance_amendment, not coding
    # sessions. Schema: {intent, files[], acceptance[],
    # expected_duration_min}.
    coding_session_spec: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposalState":
        status_raw = data.get("status", ProposalStatus.STAGED.value)
        return cls(
            source=data["source"],
            signature=data["signature"],
            title=data.get("title") or "",
            target_path=data["target_path"],
            body_hash=data.get("body_hash") or "",
            staged_at=data["staged_at"],
            status=ProposalStatus(status_raw),
            cooldown_days=int(data.get("cooldown_days", 7)),
            cr_id=data.get("cr_id"),
            cr_filed_at=data.get("cr_filed_at"),
            resolved_at=data.get("resolved_at"),
            expired_at=data.get("expired_at"),
            notes=dict(data.get("notes") or {}),
            coding_session_spec=(
                dict(data["coding_session_spec"])
                if isinstance(data.get("coding_session_spec"), dict)
                else None
            ),
        )


# ── Helpers ──────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _validate_source(source: str) -> None:
    if source not in _KNOWN_SOURCES:
        raise ValueError(
            f"unknown proposal source {source!r}; "
            f"add to _KNOWN_SOURCES in app.proposal_bridge.store",
        )


def _validate_signature(signature: str) -> None:
    if not signature or not _SAFE_SIG_RE.match(signature):
        raise ValueError(
            f"signature must be [A-Za-z0-9_.-]+, got {signature!r}",
        )


def _validate_target_path(target_path: str, body_markdown: str) -> None:
    """Run the change-request validator at stage time so impossible
    targets fail immediately, not 7 days later at promotion.

    Reuses the canonical ``app.change_requests.validator.validate``
    so the bridge cannot disagree with the CR gate about what's
    permissible. When the validator module isn't importable (e.g.
    early-boot test environments), we silently allow staging — the
    promoter's CR-gate call will surface the error then.
    """
    try:
        from app.change_requests.validator import validate
    except Exception:
        return
    result = validate(path=target_path, new_content=body_markdown)
    if not result.ok:
        layer = "TIER_IMMUTABLE" if result.is_tier_immutable else "validator"
        raise ValueError(
            f"target_path {target_path!r} rejected by {layer}: {result.reason}"
        )


def _paths_for(source: str, signature: str) -> tuple[Path, Path]:
    """Return (body_md_path, meta_json_path)."""
    src_dir = _base_dir() / source
    return src_dir / f"{signature}.md", src_dir / f"{signature}.json"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _read_meta(path: Path) -> Optional[ProposalState]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProposalState.from_dict(data)
    except Exception:
        logger.debug("proposal_bridge: meta read failed for %s", path, exc_info=True)
        return None


def _write_meta(path: Path, state: ProposalState) -> None:
    _atomic_write_text(
        path, json.dumps(state.to_dict(), indent=2, sort_keys=True),
    )


# ── Public API ────────────────────────────────────────────────────────────


def stage(
    *,
    source: str,
    signature: str,
    title: str,
    body_markdown: str,
    target_path: str,
    cooldown_days: int = 7,
    coding_session_spec: Optional[dict[str, Any]] = None,
) -> tuple[ProposalState, bool]:
    """Stage a proposal for eventual CR promotion.

    Idempotent on body content: re-staging the SAME body for the same
    (source, signature) is a no-op and preserves the cooldown clock.
    Different body content bumps the cooldown clock (producer revised
    the proposal — give the new content its own review window).

    Returns ``(state, was_new)`` where:
      * ``state`` is the resulting ``ProposalState`` after this call.
      * ``was_new`` is ``True`` when this call created the proposal
        OR replaced the body for an existing non-terminal proposal
        (cooldown clock bumped). ``False`` when the call was a
        no-op (body unchanged) or the proposal was already in a
        terminal state (APPLIED / REJECTED) and we did not re-stage.

    Raises ``ValueError`` on unknown source or malformed signature.
    """
    _validate_source(source)
    _validate_signature(signature)
    if cooldown_days < 0 or cooldown_days > 60:
        raise ValueError(f"cooldown_days out of bounds: {cooldown_days}")
    _validate_target_path(target_path, body_markdown)

    body_path, meta_path = _paths_for(source, signature)
    new_hash = _hash_body(body_markdown)

    existing = _read_meta(meta_path)
    if existing is not None:
        # Already staged. Check terminal status first.
        if existing.status in (
            ProposalStatus.APPLIED,
            ProposalStatus.REJECTED,
        ):
            # Operator has decided; do not re-stage.
            return existing, False
        if existing.body_hash == new_hash and body_path.exists():
            # Identical body — no-op. Cooldown clock unchanged.
            return existing, False
        # Body changed — bump cooldown clock and update.
        existing.body_hash = new_hash
        existing.staged_at = _now_iso()
        existing.title = title
        existing.target_path = target_path
        existing.status = ProposalStatus.STAGED
        existing.cooldown_days = cooldown_days
        existing.coding_session_spec = (
            dict(coding_session_spec) if coding_session_spec else None
        )
        # Clear any stale CR pointer if the proposal regressed back
        # to STAGED — the previous CR (if any) is no longer the
        # canonical artefact.
        existing.cr_id = None
        existing.cr_filed_at = None
        _atomic_write_text(body_path, body_markdown)
        _write_meta(meta_path, existing)
        return existing, True

    state = ProposalState(
        source=source,
        signature=signature,
        title=title,
        target_path=target_path,
        body_hash=new_hash,
        staged_at=_now_iso(),
        status=ProposalStatus.STAGED,
        cooldown_days=cooldown_days,
        coding_session_spec=(
            dict(coding_session_spec) if coding_session_spec else None
        ),
    )
    _atomic_write_text(body_path, body_markdown)
    _write_meta(meta_path, state)
    return state, True


def list_proposals(
    *,
    source: Optional[str] = None,
    status: Optional[ProposalStatus] = None,
) -> list[ProposalState]:
    """List staged proposals matching the (optional) filters."""
    out: list[ProposalState] = []
    base = _base_dir()
    if not base.exists():
        return out
    for src_dir in sorted(base.iterdir()):
        if not src_dir.is_dir():
            continue
        if source and src_dir.name != source:
            continue
        for meta_path in sorted(src_dir.glob("*.json")):
            state = _read_meta(meta_path)
            if state is None:
                continue
            if status is not None and state.status != status:
                continue
            out.append(state)
    return out


def get_proposal(source: str, signature: str) -> Optional[ProposalState]:
    """Look up a specific proposal."""
    _, meta_path = _paths_for(source, signature)
    return _read_meta(meta_path)


def update_proposal(state: ProposalState) -> None:
    """Persist updates to an existing proposal. Used by the promoter
    when transitioning STAGED → CR_FILED → APPLIED/REJECTED/EXPIRED."""
    _validate_source(state.source)
    _validate_signature(state.signature)
    _, meta_path = _paths_for(state.source, state.signature)
    _write_meta(meta_path, state)


def cleanup_resolved(state: ProposalState) -> None:
    """Remove the workspace artefacts for a resolved proposal.

    Called by the promoter after the audit-retention window. The
    audit trail of the underlying CR persists in the change-request
    store (and its hash-chained audit.jsonl); the bridge's local
    files are housekeeping only.
    """
    _validate_source(state.source)
    _validate_signature(state.signature)
    body_path, meta_path = _paths_for(state.source, state.signature)
    for p in (body_path, meta_path):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            logger.debug("proposal_bridge: cleanup failed for %s", p, exc_info=True)


def iter_proposals() -> Iterator[ProposalState]:
    """Generator over every proposal on disk. Used by the promoter."""
    base = _base_dir()
    if not base.exists():
        return
    for src_dir in sorted(base.iterdir()):
        if not src_dir.is_dir():
            continue
        for meta_path in sorted(src_dir.glob("*.json")):
            state = _read_meta(meta_path)
            if state is not None:
                yield state


def read_body(state: ProposalState) -> str:
    """Read the staged markdown body for a proposal. Empty string on
    miss (operator may have manually deleted it — treat as a soft
    EXPIRED signal).
    """
    body_path, _ = _paths_for(state.source, state.signature)
    if not body_path.exists():
        return ""
    try:
        return body_path.read_text(encoding="utf-8")
    except OSError:
        return ""
