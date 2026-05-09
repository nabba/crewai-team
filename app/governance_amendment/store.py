"""Proposal store — JSON files under ``workspace/governance/tier3_amendments/``.

One file per proposal. Atomic writes (tmp + rename) so a crashed write
never produces a half-baked JSON the loader has to defend against.

State files live OUTSIDE the source tree on purpose: they're operator
artefacts, not code, and shouldn't be amenable to amendment via this
protocol.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterator

from app.governance_amendment._state import AmendmentProposal, State

logger = logging.getLogger(__name__)


_STATE_DIR = (
    Path(__file__).resolve().parents[2] / "workspace" / "governance"
    / "tier3_amendments"
)
_lock = threading.Lock()


def _ensure_dir() -> Path:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return _STATE_DIR


def proposal_path(proposal_id: str) -> Path:
    """Path to one proposal's JSON file. Caller must validate ``proposal_id``
    is hex-only — we never accept arbitrary strings on disk paths."""
    if not proposal_id or not all(c in "0123456789abcdef" for c in proposal_id):
        raise ValueError(f"invalid proposal_id: {proposal_id!r}")
    return _ensure_dir() / f"{proposal_id}.json"


def save(proposal: AmendmentProposal) -> None:
    """Atomic write."""
    path = proposal_path(proposal.id)
    payload = proposal.to_dict()
    with _lock:
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            tmp.replace(path)
        except Exception:
            logger.warning(
                "tier3_amendment.store: save failed for %s", proposal.id,
                exc_info=True,
            )


def load(proposal_id: str) -> AmendmentProposal | None:
    """Read one proposal. Returns None on missing or malformed file."""
    try:
        path = proposal_path(proposal_id)
    except ValueError:
        return None
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "tier3_amendment.store: malformed JSON at %s", path, exc_info=True,
        )
        return None
    try:
        return AmendmentProposal.from_dict(payload)
    except Exception:
        logger.warning(
            "tier3_amendment.store: from_dict failed for %s", proposal_id,
            exc_info=True,
        )
        return None


def iter_all() -> Iterator[AmendmentProposal]:
    """Walk all stored proposals — order is filesystem-dependent."""
    if not _STATE_DIR.exists():
        return
    for path in sorted(_STATE_DIR.glob("*.json")):
        if path.stem.endswith(".tmp"):
            continue
        proposal = load(path.stem)
        if proposal is not None:
            yield proposal


def list_by_state(state: State | None = None) -> list[AmendmentProposal]:
    """Materialised list (helpful for tests / dashboards)."""
    out = []
    for proposal in iter_all():
        if state is None or proposal.state == state:
            out.append(proposal)
    return out
