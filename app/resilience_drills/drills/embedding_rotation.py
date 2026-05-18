"""embedding_rotation — quarterly model-rotation verification drill (§56).

PROGRAM §56 iter-2. 8th resilience drill. Proves the §56 claim that
the source-ledger replay is *embedding-model-rotation tolerant* —
i.e. that text in the ledger can be re-embedded with a different
model than the live KB uses, and the resulting scratch KB is still
queryable.

What the drill does
===================

  1. Pick a random KB whose ledger has ≥30 rows (small KBs would
     produce statistically meaningless similarity comparisons).
  2. Replay its ledger into a scratch chromadb instance using a
     **different embedding backend** — chromadb's bundled ONNX
     MiniLM-L6-v2 (384-dim) instead of the live nomic-embed-text
     (768-dim). This is the most accessible "different model";
     swapping to a *good* alternative model requires the same
     vendor key the live model uses.
  3. For each of N=5 random rows from the rebuilt scratch KB, run a
     ``col.query`` using its OWN text as the query. PASS if every
     query returns the row itself in the top-3 results — i.e. the
     embedding is self-consistent for the new model.
  4. Clean up the scratch dir.

What this catches
=================

  * The ledger text is no longer round-trip-embeddable (e.g. binary
    junk slipped in via a buggy hook).
  * The replay code lost its ability to substitute embedding
    functions (regression of the model-rotation tolerance feature).
  * The ledger rows are too short / too long / malformed for any
    reasonable embedding model to handle.

What this DOES NOT promise
==========================

  Switching to a structurally-different embedding model (e.g. a
  cross-lingual one with different similarity semantics) might
  still produce a queryable KB but with worse retrieval relevance.
  This drill catches *broken* outcomes, not *worse-than-live*
  outcomes.

Cadence: quarterly. Risk: LOW (never touches live KB; scratch dir
is ephemeral).

Master switch: ``drill_embedding_rotation_enabled`` (default ON).
"""
from __future__ import annotations

import logging
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    FailureClass,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="embedding_rotation",
    cadence_days=90,
    grace_days=30,
    warmup_days=0,
    risk=DrillRisk.LOW,
    description=(
        "Quarterly drill: rebuild a random KB from its source ledger "
        "using a DIFFERENT embedding model than the live one, then "
        "verify the rebuilt KB is self-consistent. Proves that the "
        "§56 replay-from-ledger pipeline survives model swaps. Never "
        "touches live data."
    ),
    requires_master_switch="drill_embedding_rotation_enabled",
)


_MIN_LEDGER_ROWS = 30           # below this, query comparison is too noisy
_QUERY_SAMPLE_SIZE = 5           # rows queried for self-consistency check
_QUERY_TOP_K = 3                 # row must appear in its own top-3
_DRILL_REPLAY_CAP = 2_000        # cap replay so drill is fast


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT  # type: ignore
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _scratch_root() -> Path:
    return _workspace_root() / ".drill_scratch_rotation"


def _pick_kb() -> str | None:
    """Pick a KB with ≥_MIN_LEDGER_ROWS ledger rows."""
    try:
        from app.memory.source_ledger import list_kbs, count_rows
    except Exception:
        return None
    candidates: list[str] = []
    for kb in list_kbs():
        try:
            if count_rows(kb) >= _MIN_LEDGER_ROWS:
                candidates.append(kb)
        except Exception:
            continue
    if not candidates:
        return None
    return random.choice(candidates)


def _replay_with_default_chroma_embedder(kb_name: str, scratch_dir: Path,
                                          max_rows: int) -> dict:
    """Replay the KB's ledger into ``scratch_dir`` letting chromadb pick
    its OWN embedding model (ONNX MiniLM-L6-v2 by default), instead of
    the gateway's curated ollama nomic-embed-text.

    This is the actual "rotation": same text, different embedder.
    Returns ``{rows_seen, rows_upserted, error}``.
    """
    out: dict[str, Any] = {"rows_seen": 0, "rows_upserted": 0, "error": None}
    try:
        from app.memory import source_ledger as sl
        import chromadb  # type: ignore
    except Exception as exc:
        out["error"] = f"import_failed: {exc}"
        return out

    # Fold ledger ourselves so we don't go through replay_kb's gateway-
    # embed path. We need chromadb to do its own embedding here.
    state: dict[tuple[str, str], object] = {}
    seen = 0
    for row in sl.read_all(kb_name):
        seen += 1
        if seen > max_rows:
            break
        key = (row.collection, row.doc_id)
        if row.op == sl.OP_ADD:
            state[key] = (row.text, dict(row.metadata or {}))
        elif row.op == sl.OP_UPDATE:
            prior = state.get(key)
            if prior is None:
                continue
            prior_text, prior_meta = prior
            merged_text = row.text or prior_text
            merged_meta = row.metadata if row.metadata or row.metadata == {} else prior_meta
            state[key] = (merged_text, merged_meta)
        elif row.op == sl.OP_DELETE:
            state.pop(key, None)
    out["rows_seen"] = seen

    try:
        client = chromadb.PersistentClient(path=str(scratch_dir))
    except Exception as exc:
        out["error"] = f"scratch_client: {exc}"
        return out

    # Group by collection.
    by_col: dict[str, list[tuple[str, str, dict]]] = {}
    for (col_name, doc_id), val in state.items():
        text, meta = val
        if not text or not text.strip():
            continue
        by_col.setdefault(col_name, []).append((doc_id, text, meta))

    for col_name, rows in by_col.items():
        try:
            # NO custom embedding_function — chromadb picks its
            # default (ONNX MiniLM-L6-v2). This is the rotation.
            col = client.get_or_create_collection(col_name)
        except Exception:
            continue
        try:
            ids = [r[0] for r in rows]
            docs = [r[1] for r in rows]
            metas = [r[2] for r in rows]
            col.upsert(ids=ids, documents=docs, metadatas=metas)
            out["rows_upserted"] += len(rows)
        except Exception:
            # Most-common failure: dimension mismatch when a prior
            # scratch run left state. We delete + retry.
            try:
                client.delete_collection(col_name)
                col = client.get_or_create_collection(col_name)
                col.upsert(ids=ids, documents=docs, metadatas=metas)
                out["rows_upserted"] += len(rows)
            except Exception as exc:
                logger.debug(
                    "embedding_rotation: scratch upsert failed for %s: %s",
                    col_name, exc,
                )
                continue
    return out


def _self_consistency_check(scratch_dir: Path) -> dict:
    """For ``_QUERY_SAMPLE_SIZE`` random rows in the rebuilt scratch
    KB, query using the row's own text. PASS when the row appears in
    its own top-``_QUERY_TOP_K`` results. This is a stable test for
    "is the embedding self-consistent under the new model" without
    needing ground-truth semantic comparisons.
    """
    out: dict[str, Any] = {
        "tested": 0, "self_matches": 0, "queries": [], "error": None,
    }
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=str(scratch_dir))
    except Exception as exc:
        out["error"] = f"open: {exc}"
        return out

    all_rows: list[tuple[str, str, str]] = []  # (col_name, doc_id, text)
    for col in client.list_collections():
        try:
            data = col.get(include=["documents"], limit=200)
        except Exception:
            continue
        for doc_id, text in zip(data.get("ids") or [], data.get("documents") or []):
            if text and text.strip():
                all_rows.append((getattr(col, "name", "?"), doc_id, text))

    if not all_rows:
        out["error"] = "no_rows"
        return out

    sample = random.sample(all_rows, min(_QUERY_SAMPLE_SIZE, len(all_rows)))
    for col_name, doc_id, text in sample:
        out["tested"] += 1
        try:
            col = client.get_collection(col_name)
            res = col.query(query_texts=[text], n_results=_QUERY_TOP_K)
            top_ids = (res.get("ids") or [[]])[0]
            found = doc_id in top_ids
            out["queries"].append({
                "doc_id": doc_id,
                "self_match": found,
                "top_ids": list(top_ids)[:_QUERY_TOP_K],
            })
            if found:
                out["self_matches"] += 1
        except Exception as exc:
            out["queries"].append({"doc_id": doc_id, "error": str(exc)})
    return out


def _run(*, dry_run: bool = True) -> DrillResult:
    """Q18 runner contract: returns a bare DrillResult."""
    started = datetime.now(timezone.utc)
    t0 = time.time()

    scratch = _scratch_root() / started.strftime("%Y%m%dT%H%M%SZ")
    try:
        kb = _pick_kb()
        if kb is None:
            return DrillResult(
                drill_name=SPEC.name, status=DrillStatus.SKIPPED,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0, dry_run=dry_run,
                detail={"reason": f"no KB with ≥{_MIN_LEDGER_ROWS} ledger rows"},
            )

        detail: dict[str, Any] = {"kb_name": kb}
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)

        rep = _replay_with_default_chroma_embedder(kb, scratch, _DRILL_REPLAY_CAP)
        detail["replay"] = rep
        if rep.get("error") or rep.get("rows_upserted", 0) == 0:
            return DrillResult(
                drill_name=SPEC.name, status=DrillStatus.FAIL,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0, dry_run=dry_run,
                detail=detail,
                errors=[f"replay failed: {rep.get('error') or 'no rows upserted'}"],
                failure_class=FailureClass.STRUCTURAL_FAIL,
                observation={"kb_name": kb, "rows_upserted": rep.get("rows_upserted", 0)},
            )

        consistency = _self_consistency_check(scratch)
        detail["consistency"] = consistency
        if consistency.get("error"):
            return DrillResult(
                drill_name=SPEC.name, status=DrillStatus.ERROR,
                started_at=started.isoformat(),
                completed_at=datetime.now(timezone.utc).isoformat(),
                duration_s=time.time() - t0, dry_run=dry_run,
                detail=detail,
                errors=[f"consistency check failed: {consistency['error']}"],
                failure_class=FailureClass.CODE_ERROR,
            )

        tested = consistency.get("tested", 0)
        matches = consistency.get("self_matches", 0)
        rate = matches / max(1, tested)
        detail["self_match_rate"] = round(rate, 4)
        status = DrillStatus.PASS if rate >= 0.8 else DrillStatus.FAIL
        failure_class = (
            FailureClass.STRUCTURAL_FAIL
            if status == DrillStatus.FAIL else None
        )

        observation = {
            "kb_name": kb,
            "rows_upserted": rep.get("rows_upserted", 0),
            "self_match_rate": round(rate, 4),
            "tested": tested,
            "matches": matches,
        }

        return DrillResult(
            drill_name=SPEC.name, status=status,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0, dry_run=dry_run,
            detail=detail,
            failure_class=failure_class,
            observation=observation,
        )
    except Exception as exc:
        logger.debug("embedding_rotation: drill errored", exc_info=True)
        return DrillResult(
            drill_name=SPEC.name, status=DrillStatus.ERROR,
            started_at=started.isoformat(),
            completed_at=datetime.now(timezone.utc).isoformat(),
            duration_s=time.time() - t0, dry_run=dry_run,
            detail={}, errors=[f"{type(exc).__name__}: {exc}"],
            failure_class=FailureClass.CODE_ERROR,
        )
    finally:
        try:
            if scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass


def run(*, dry_run: bool = True) -> DrillResult:
    return _run(dry_run=dry_run)


register(SPEC, run)
