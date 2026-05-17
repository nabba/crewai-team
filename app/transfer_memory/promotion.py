"""Shadow → active promotion (Phase 17c, gated).

Idle-scheduler MEDIUM job. Walks records at status="shadow" and applies
a deterministic eligibility check; eligible records are either promoted
to status="active" (when ``transfer_memory_auto_promote_enabled`` is
True) or recorded as candidates for operator review (default).

Eligibility (all must hold):
    age           ≥ ``_MIN_AGE_SECONDS``        — 7 days in shadow
    surface_count ≥ ``_MIN_SURFACE_COUNT``      — would-have-been
                                                  retrieved ≥3 times
                                                  per shadow_retrievals
    blacklist     not present                   — attribution OK
    neg_transfer  zero log entries              — no failures attributed

Promotion mutations (when enabled):
    1. Index entry (skill_records collection): status flipped to "active"
       via ``self_improvement.integrator.update_record``.
    2. Underlying KB metadata (episteme/experiential/aesthetics/tensions)
       updated via the store's ``_collection.update`` so retrieval's
       ChromaDB ``where={"status":"active"}`` filter finds the record.

Audit:
    ``workspace/transfer_memory/promotion_log.jsonl`` — one row per
    review event (promoted, deferred, ineligible) with reasons.
    ``workspace/transfer_memory/promotion_candidates.jsonl`` — current
    snapshot of eligible records when auto-promote is OFF (operator
    inspects to drive manual promotion via dashboard).

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_PROMOTION_LOG = "promotion_log.jsonl"
_PROMOTION_CANDIDATES = "promotion_candidates.jsonl"
_LAST_PROMOTION_FILENAME = ".last_promotion_at"

# Eligibility thresholds — conservative; tunable via constants here.
_MIN_AGE_SECONDS = 7 * 24 * 60 * 60       # 7 days in shadow
_MIN_SURFACE_COUNT = 3                     # shadow_retrievals hits

# Cadence guard — promotion is heavier than attribution; run at most
# once per 6 hours regardless of idle-scheduler invocation.
_MIN_INTERVAL_SECONDS = 6 * 60 * 60

# Hard cap on records reviewed per run to bound the job's wall time.
_MAX_REVIEWED_PER_RUN = 200


# ── Public entry point ───────────────────────────────────────────────

def run_promotion() -> dict:
    """Idle-scheduler entry. Returns a summary dict.

    When ``transfer_memory_auto_promote_enabled`` is False (default in
    Phase 17c), runs in audit-only mode: counts eligible candidates,
    refreshes ``promotion_candidates.jsonl`` for dashboard / operator
    review, and exits. Manual promotion is via dashboard action.

    When the flag is True, eligible candidates are flipped to active
    in both the index and the underlying KB metadata.
    """
    summary = {
        "ran": False, "reviewed": 0, "promoted": 0, "deferred": 0,
        "candidates": 0, "errors": 0, "skipped_cadence": False,
        "auto_promote": _auto_promote_enabled(),
    }

    now = time.time()
    last = _read_last_run()
    if last and (now - last) < _MIN_INTERVAL_SECONDS:
        summary["skipped_cadence"] = True
        return summary

    shadow_records = _list_shadow_records()
    if not shadow_records:
        _write_last_run(now)
        return summary

    summary["ran"] = True
    eligible: list[Any] = []
    surface_counts = _surface_counts_from_shadow_log()
    blacklist = _read_blacklist()

    for rec in shadow_records[:_MAX_REVIEWED_PER_RUN]:
        if _should_yield_safe():
            logger.info("transfer_memory.promotion: yielding to user task")
            break
        summary["reviewed"] += 1
        verdict = _evaluate(rec, surface_counts=surface_counts, blacklist=blacklist)
        if verdict.eligible:
            eligible.append(rec)
            summary["candidates"] += 1
        else:
            _append_promotion_log({
                "ts": now, "skill_record_id": rec.id, "action": "deferred",
                "reasons": verdict.reasons,
            })
            summary["deferred"] += 1

    _refresh_candidates_file(eligible)

    if summary["auto_promote"]:
        for rec in eligible:
            try:
                if _promote(rec):
                    summary["promoted"] += 1
                    _append_promotion_log({
                        "ts": time.time(),
                        "skill_record_id": rec.id,
                        "action": "promoted",
                        "kb": rec.kb,
                        "transfer_scope": (rec.provenance or {}).get(
                            "transfer_scope", ""
                        ),
                    })
                else:
                    summary["errors"] += 1
            except Exception:
                summary["errors"] += 1
                logger.debug(
                    "transfer_memory.promotion: promote failed",
                    exc_info=True,
                )

    _write_last_run(time.time())
    logger.info(
        "transfer_memory.promotion: ran (reviewed=%d eligible=%d "
        "promoted=%d deferred=%d auto=%s)",
        summary["reviewed"], summary["candidates"],
        summary["promoted"], summary["deferred"], summary["auto_promote"],
    )
    return summary


def manual_promote(record_id: str) -> bool:
    """Operator-driven single-record promotion. Bypasses cadence guard
    but still requires the record to pass eligibility checks.

    Used by the dashboard's "promote" button. Returns True on success.
    """
    record = _load_skill_record(record_id)
    if record is None:
        return False
    surface_counts = _surface_counts_from_shadow_log()
    blacklist = _read_blacklist()
    verdict = _evaluate(record, surface_counts=surface_counts, blacklist=blacklist)
    if not verdict.eligible:
        _append_promotion_log({
            "ts": time.time(), "skill_record_id": record_id,
            "action": "manual_rejected", "reasons": verdict.reasons,
        })
        return False
    if _promote(record):
        _append_promotion_log({
            "ts": time.time(), "skill_record_id": record_id,
            "action": "manual_promoted", "kb": record.kb,
        })
        return True
    return False


# ── Eligibility ──────────────────────────────────────────────────────

class _Verdict:
    __slots__ = ("eligible", "reasons")

    def __init__(self, eligible: bool, reasons: list[str]):
        self.eligible = eligible
        self.reasons = reasons


def _evaluate(
    record: Any, *, surface_counts: dict[str, int], blacklist: set[str],
) -> _Verdict:
    reasons: list[str] = []
    now = time.time()

    # Age
    created = record.created_at or ""
    age_seconds = _seconds_since_iso(created)
    if age_seconds is None or age_seconds < _MIN_AGE_SECONDS:
        reasons.append("too_young")

    # Surface count
    surf = surface_counts.get(record.id, 0)
    if surf < _MIN_SURFACE_COUNT:
        reasons.append(f"low_surface_count={surf}")

    # Blacklist
    if record.id in blacklist:
        reasons.append("blacklisted")

    # Negative transfer entries
    if _has_neg_transfer(record.id):
        reasons.append("negative_transfer_logged")

    # Status sanity (only shadow → active)
    if record.status != "shadow":
        reasons.append(f"unexpected_status={record.status}")

    return _Verdict(eligible=not reasons, reasons=reasons)


# ── Mutations ────────────────────────────────────────────────────────

def _promote(record: Any) -> bool:
    """Flip status from shadow → active in both the index and the KB.

    Best-effort — partial promotion (index only) is logged but treated
    as failure so the operator can re-trigger manually.
    """
    record.status = "active"
    try:
        from app.self_improvement.integrator import update_record
        if not update_record(record):
            return False
    except Exception:
        return False

    return _set_kb_status(record.kb, record.id, "active")


def _set_kb_status(kb: str, record_id: str, new_status: str) -> bool:
    """Update the underlying KB collection's metadata.

    Required because the retrieval ``where={"status": ...}`` filter
    operates on the KB-level metadata, not the index. Skipping this
    step would leave promoted records still tagged ``status="shadow"``
    in their KB and therefore invisible to production retrieval.
    """
    try:
        if kb == "episteme":
            from app.episteme.vectorstore import get_store
        elif kb == "experiential":
            from app.experiential.vectorstore import get_store
        elif kb == "aesthetics":
            from app.aesthetics.vectorstore import get_store
        elif kb == "tensions":
            from app.tensions.vectorstore import get_store
        else:
            return False
    except Exception:
        return False

    try:
        store = get_store()
        col = getattr(store, "_collection", None)
        if col is None:
            return False
        existing = col.get(ids=[record_id])
        if not existing.get("ids"):
            return False
        metas = existing.get("metadatas") or [{}]
        meta = dict(metas[0] or {})
        meta["status"] = new_status
        col.update(ids=[record_id], metadatas=[meta])
        # PROGRAM §56 iter-2 — ledger update so replay preserves the
        # status change (not just the original add).
        try:
            from app.memory.source_ledger import hook_collection_update
            # ``kb`` here is the canonical KB name passed in by the
            # caller; collection name is just the record's home.
            collection_name = getattr(col, "name", "transfer_memory")
            hook_collection_update(
                kb, collection_name, [record_id], metadatas=[meta],
            )
        except Exception:
            logger.debug(
                "transfer_memory.promotion: ledger update hook failed",
                exc_info=True,
            )
        return True
    except Exception:
        logger.debug(
            "transfer_memory.promotion: _set_kb_status(%s, %s) failed",
            kb, record_id, exc_info=True,
        )
        return False


# ── Persistence helpers ──────────────────────────────────────────────

def _resolve_dir() -> Path:
    from app.transfer_memory.queue import _resolve_dir as base_dir, _ensure_dir
    _ensure_dir()
    return base_dir()


def _append_promotion_log(row: dict) -> None:
    p = _resolve_dir() / _PROMOTION_LOG
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass


def _refresh_candidates_file(records: list) -> None:
    """Rewrite the candidates snapshot file. One row per eligible record."""
    p = _resolve_dir() / _PROMOTION_CANDIDATES
    rows: list[dict] = []
    for rec in records:
        prov = rec.provenance or {}
        rows.append({
            "skill_record_id": rec.id,
            "topic": rec.topic[:160],
            "kb": rec.kb,
            "source_kind": prov.get("source_kind", ""),
            "source_domain": prov.get("source_domain", ""),
            "transfer_scope": prov.get("transfer_scope", ""),
            "abstraction_score": float(prov.get("abstraction_score", 0.0) or 0.0),
            "leakage_risk": float(prov.get("leakage_risk", 0.0) or 0.0),
            "created_at": rec.created_at,
        })
    try:
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    except Exception:
        pass


def _read_last_run() -> float:
    p = _resolve_dir() / _LAST_PROMOTION_FILENAME
    if not p.exists():
        return 0.0
    try:
        return float(p.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0.0


def _write_last_run(epoch: float) -> None:
    try:
        (_resolve_dir() / _LAST_PROMOTION_FILENAME).write_text(
            f"{epoch}\n", encoding="utf-8",
        )
    except Exception:
        pass


def _surface_counts_from_shadow_log() -> dict[str, int]:
    """Count how many times each record id appears in shadow_retrievals.

    Cheap heuristic for "would have been useful". A record that's never
    surfaced in shadow mode hasn't earned promotion regardless of age.
    """
    p = _resolve_dir() / "shadow_retrievals.jsonl"
    if not p.exists():
        return {}
    counts: Counter = Counter()
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                for surf in row.get("surfaced", []):
                    rid = surf.get("skill_record_id")
                    if rid:
                        counts[rid] += 1
    except Exception:
        return {}
    return dict(counts)


def _read_blacklist() -> set[str]:
    p = _resolve_dir() / "demotion_blacklist.jsonl"
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    except Exception:
        return set()


def _has_neg_transfer(record_id: str) -> bool:
    p = _resolve_dir() / "negative_transfer.jsonl"
    if not p.exists():
        return False
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("skill_record_id") == record_id:
                    return True
    except Exception:
        return False
    return False


def _list_shadow_records() -> list:
    """Pull all SkillRecords with status='shadow' from the index."""
    try:
        from app.self_improvement.integrator import list_records
        return list_records(status="shadow", limit=2000)
    except Exception:
        return []


def _load_skill_record(record_id: str):
    try:
        from app.self_improvement.integrator import load_record
        return load_record(record_id)
    except Exception:
        return None


# ── Settings / utilities ─────────────────────────────────────────────

def _auto_promote_enabled() -> bool:
    try:
        from app.config import get_settings
        return bool(getattr(get_settings(), "transfer_memory_auto_promote_enabled", False))
    except Exception:
        return False


def _seconds_since_iso(iso_string: str) -> float | None:
    if not iso_string:
        return None
    try:
        from datetime import datetime
        # tolerate "Z" suffix and tz-aware iso strings
        s = iso_string.rstrip("Z")
        # Best-effort: try fromisoformat, fall back to None
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            return None
        ts = dt.timestamp() if dt.tzinfo else dt.timestamp()
        return max(0.0, time.time() - ts)
    except Exception:
        return None


def _should_yield_safe() -> bool:
    try:
        from app.idle_scheduler import should_yield
        return bool(should_yield())
    except Exception:
        return False
