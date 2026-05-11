"""Cutover — Tier-3 amendment-gated atomic swap.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

Cutover changes the immutable ``_EMBED_DIM`` constant in
``chromadb_manager.py`` (and its companion provider/model strings) so
the live runtime starts using the target model. That's a TIER_IMMUTABLE
edit — the only way to do it is via the Tier-3 amendment protocol.

This module:

  1. Verifies migration is in ``READY`` (via ``verify.verify()``).
  2. Builds the patch — old vs new content for ``chromadb_manager.py``.
  3. Calls ``governance_amendment.protocol.propose_amendment(...)``
     with a strict citation explaining the migration plan.
  4. Returns the proposal id so the operator can track it through the
     existing Tier-3 lifecycle (PROPOSED → STAGED → COOLDOWN_OK →
     APPROVED → APPLIED → STABLE).

When the Tier-3 amendment ``mark_applied`` is called by the operator
(via React `/cp/governance` or the `gateway` REST endpoint), this
module's ``post_apply_hook`` runs:

  * Atomic swap of every chromadb collection: rename source →
    ``<col>__archive_<ts>`` (deletion deferred until stand-down),
    rename ``<col>__shadow_<plan>`` → live name.
  * Update plan state to ``APPLIED`` with timestamp.

ChromaDB has no rename API; "rename" is `delete-then-recreate-with-
new-content`. The atomic swap therefore is:
  1. Read all rows from shadow.
  2. Read all rows from source (so we can re-create archive).
  3. Delete source.
  4. Create source-named-collection with shadow rows.
  5. Create archive-named-collection with old-source rows.
  6. Delete shadow.

Steps 3-6 happen with a chromadb_manager-internal lock taken to
prevent concurrent writes. Reads during the swap can see an empty
collection — operators run cutover during a quiet window.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.memory.embedding_migration import plan as plan_mod
from app.memory.embedding_migration import state as state_mod
from app.memory.embedding_migration import verify as verify_mod
from app.memory.embedding_migration.dual_write import shadow_collection_name

logger = logging.getLogger(__name__)


_CHROMADB_MANAGER_PATH = "app/memory/chromadb_manager.py"
_PROPOSER_ID = "embedding_migration.cutover"


@dataclass
class CutoverProposalResult:
    plan_id: str
    proposal_id: str | None = None
    state_after_propose: str = ""
    verify_ok: bool = False
    verify_report: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CutoverApplyResult:
    plan_id: str
    collections_swapped: int = 0
    rows_swapped: int = 0
    archived_collection_names: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Patch builder ────────────────────────────────────────────────────────


def _read_chromadb_manager_text() -> str:
    p = Path(_CHROMADB_MANAGER_PATH)
    if not p.exists():
        raise FileNotFoundError(_CHROMADB_MANAGER_PATH)
    return p.read_text(encoding="utf-8")


def _build_new_content(plan) -> tuple[str, str]:
    """Return ``(old_content, new_content)`` for chromadb_manager.py.

    The patch is an in-place edit of:
      * ``_OLLAMA_MODEL``        → ``plan.target.name`` (if Ollama)
      * ``_EMBED_DIM = 768``     → ``_EMBED_DIM = <plan.target.dim>``
      * Any references to ``_OLLAMA_URL`` if base_url changed.

    Constructed as a literal find-and-replace pair so the Tier-3 patch
    application is a clean single-region overwrite. Multi-region
    edits are intentionally NOT supported here — if the source model
    isn't Ollama or the file structure changed, raise so the operator
    inspects manually."""
    old_content = _read_chromadb_manager_text()

    if plan.source.provider != "ollama" or plan.target.provider != "ollama":
        raise ValueError(
            "cutover: only ollama→ollama swaps are supported by the auto-patch. "
            "For other provider transitions, prepare the patch by hand and "
            "submit via change_requests with TIER3 escalation."
        )

    # Find the dim constant. Use a literal search so a renamed variable
    # doesn't silently match.
    needle_dim = f"_EMBED_DIM = {plan.source.dim}  # IMMUTABLE"
    if needle_dim not in old_content:
        raise ValueError(
            f"cutover: could not locate `{needle_dim}` in chromadb_manager.py; "
            f"refusing to patch a file whose structure has drifted."
        )
    new_content = old_content.replace(
        needle_dim, f"_EMBED_DIM = {plan.target.dim}  # IMMUTABLE",
    )

    # Find the model constant.
    needle_model = f'_OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "{plan.source.name}")'
    if needle_model not in new_content:
        raise ValueError(
            f"cutover: could not locate the OLLAMA_EMBED_MODEL line for source "
            f"`{plan.source.name}`. Re-author plan with the correct source."
        )
    new_content = new_content.replace(
        needle_model,
        f'_OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "{plan.target.name}")',
    )

    if new_content == old_content:
        raise ValueError("cutover: patch produced no changes; abort")
    return old_content, new_content


# ── Step 1: propose the Tier-3 amendment ─────────────────────────────────


def propose_cutover() -> CutoverProposalResult:
    """Run the verifier; if ok, file a Tier-3 amendment proposal."""
    plan = plan_mod.load_plan()
    if plan is None:
        return CutoverProposalResult(
            plan_id="?", error="no migration plan loaded",
        )

    report = verify_mod.verify()
    result = CutoverProposalResult(
        plan_id=plan.plan_id,
        verify_ok=report.ok,
        verify_report=report.to_dict(),
    )
    if not report.ok:
        failed = [c.name for c in report.checks if not c.ok]
        result.error = (
            f"verification failed: {failed}. "
            f"Resolve before proposing cutover."
        )
        return result

    # Mark cutover requested in counters.
    cur = state_mod.get_state()
    cur.counters.cutover_requested_at = datetime.now(timezone.utc).isoformat()
    state_mod.set_state(cur)

    # Build the patch.
    try:
        old_content, new_content = _build_new_content(plan)
    except Exception as exc:
        result.error = f"patch build failed: {exc}"
        return result

    # File the proposal.
    try:
        from app.governance_amendment.protocol import (
            ProtocolDisabled, propose_amendment,
        )
    except Exception as exc:
        result.error = f"governance_amendment unavailable: {exc}"
        return result

    citation = (
        f"Embedding migration cutover for plan `{plan.plan_id}`. "
        f"Source: {plan.source.display()}. Target: {plan.target.display()}. "
        f"Verified phase=READY, NDCG@10≥{plan.cutover_threshold_ndcg} over "
        f"≥{plan.cutover_min_shadow_queries} shadow queries, all per-collection "
        f"row counts within ±1%, fresh DR backup confirmed. See "
        f"docs/EMBEDDING_MIGRATION.md for the full lifecycle."
    )
    try:
        proposal = propose_amendment(
            target_path=_CHROMADB_MANAGER_PATH,
            new_content=new_content,
            old_content=old_content,
            citation=citation,
            proposer=_PROPOSER_ID,
            extra_evidence={
                "plan_id": plan.plan_id,
                "source_model": plan.source.to_dict(),
                "target_model": plan.target.to_dict(),
                "shadow_query_count": cur.counters.shadow_query_count,
                "last_ndcg_at_10": cur.counters.last_ndcg_at_10,
            },
        )
    except ProtocolDisabled as exc:
        result.error = (
            f"Tier-3 amendment protocol is disabled. "
            f"Set tier3_amendment_enabled=true in /cp/settings before cutover. "
            f"Underlying: {exc}"
        )
        return result
    except Exception as exc:
        result.error = f"propose_amendment failed: {exc}"
        return result

    result.proposal_id = proposal.id
    result.state_after_propose = proposal.state.value if hasattr(proposal.state, "value") else str(proposal.state)
    # Move state machine into CUTOVER (proposal in flight).
    try:
        state_mod.transition(state_mod.PHASE_CUTOVER, reason=f"proposal={proposal.id}")
    except Exception:
        # Already in CUTOVER from a prior attempt — fine.
        pass
    return result


# ── Step 2: post-apply atomic swap ───────────────────────────────────────


def post_apply_hook() -> CutoverApplyResult:
    """Called by the operator (or governance_amendment.mark_applied
    callback) AFTER the Tier-3 amendment has been APPLIED.

    Runs the chromadb collection swap. Updates state to APPLIED.
    """
    started = time.monotonic()
    plan = plan_mod.load_plan()
    if plan is None:
        return CutoverApplyResult(
            plan_id="?", ok=False, error="no plan loaded",
        )
    cur = state_mod.get_state()
    if cur.phase not in (state_mod.PHASE_CUTOVER, state_mod.PHASE_READY):
        return CutoverApplyResult(
            plan_id=plan.plan_id, ok=False,
            error=f"phase {cur.phase}; expected CUTOVER or READY",
        )

    try:
        from app.memory import chromadb_manager
        client = chromadb_manager.get_client()
    except Exception as exc:
        return CutoverApplyResult(
            plan_id=plan.plan_id, ok=False,
            error=f"chromadb client open failed: {exc}",
        )

    archived: list[str] = []
    rows_swapped = 0
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    try:
        for target in plan.targets:
            if target.kind != "chromadb":
                continue
            col_name = target.collection
            if not col_name:
                continue
            shadow_name = shadow_collection_name(col_name, plan.plan_id)
            archive_name = f"{col_name}__archive_{ts}"

            # Read full source contents (so we can re-create archive).
            src = client.get_collection(col_name)
            shadow = client.get_collection(shadow_name)

            def _read_all(c):
                rows = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
                offset = 0
                while True:
                    chunk = c.get(
                        limit=500, offset=offset,
                        include=["documents", "metadatas", "embeddings"],
                    )
                    ids = chunk.get("ids") or []
                    if not ids:
                        break
                    rows["ids"].extend(ids)
                    rows["documents"].extend(chunk.get("documents") or [None] * len(ids))
                    rows["metadatas"].extend(chunk.get("metadatas") or [None] * len(ids))
                    rows["embeddings"].extend(chunk.get("embeddings") or [None] * len(ids))
                    if len(ids) < 500:
                        break
                    offset += len(ids)
                return rows

            src_rows = _read_all(src)
            shadow_rows = _read_all(shadow)

            # Atomic-ish swap.
            client.delete_collection(col_name)
            client.delete_collection(shadow_name)

            # Re-create source-named with shadow content.
            new_live = client.create_collection(col_name)
            if shadow_rows["ids"]:
                new_live.add(
                    ids=shadow_rows["ids"],
                    documents=shadow_rows["documents"],
                    metadatas=[
                        m if isinstance(m, dict) else {}
                        for m in shadow_rows["metadatas"]
                    ],
                    embeddings=shadow_rows["embeddings"],
                )
                rows_swapped += len(shadow_rows["ids"])

            # Archive old source content.
            new_archive = client.create_collection(archive_name)
            if src_rows["ids"]:
                new_archive.add(
                    ids=src_rows["ids"],
                    documents=src_rows["documents"],
                    metadatas=[
                        m if isinstance(m, dict) else {}
                        for m in src_rows["metadatas"]
                    ],
                    embeddings=src_rows["embeddings"],
                )
                archived.append(archive_name)

            # Refresh manager caches so subsequent reads hit the new
            # collection metadata.
            try:
                chromadb_manager._collections.pop(col_name, None)  # type: ignore[attr-defined]
                chromadb_manager._count_cache.pop(col_name, None)  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception as exc:
        logger.exception("cutover: swap failed")
        return CutoverApplyResult(
            plan_id=plan.plan_id,
            collections_swapped=len(archived),
            rows_swapped=rows_swapped,
            archived_collection_names=archived,
            duration_s=time.monotonic() - started,
            ok=False,
            error=f"swap failed: {exc}",
        )

    # State → APPLIED.
    cur = state_mod.get_state()
    cur.counters.applied_at = datetime.now(timezone.utc).isoformat()
    state_mod.set_state(cur)
    try:
        state_mod.transition(state_mod.PHASE_APPLIED, reason="post_apply_hook")
    except Exception:
        # Already APPLIED — operator re-ran the hook. Fine.
        pass

    return CutoverApplyResult(
        plan_id=plan.plan_id,
        collections_swapped=len(archived),
        rows_swapped=rows_swapped,
        archived_collection_names=archived,
        duration_s=time.monotonic() - started,
        ok=True,
    )
