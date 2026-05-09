"""Semantic LLM-output drift detector (Phase D #6, 2026-05-09).

Distinct from Phase C's silent_regression_detector (which watches
cron-event THROUGHPUT). This module watches output QUALITY: weekly
checkpoint of the LLM cascade against a small golden-set of probes,
embedding-similarity against a baseline frozen at first run, and
alert when drift exceeds a threshold.

Why we need this even though governance + Goodhart already check
quality during PROMOTION: those gates only fire on the proposed
DELTA. A model swap, prompt-registry update, or upstream
provider change can quietly degrade the EVERYDAY output without
ever going through promotion. This detector is the catch.

Algorithm:

  1. Run a fixed set of golden probes (questions with stable, well-
     understood answers — see ``_DEFAULT_PROBES``). Operators can
     extend via ``workspace/healing/llm_drift_probes.json``.
  2. For each probe, capture (a) the LLM's response, (b) an
     embedding of the response. Both via the gateway's existing
     LLM stack.
  3. On first run, persist the response + embedding as the BASELINE
     in ``workspace/healing/llm_drift_baseline.json``.
  4. On subsequent runs, compute cosine similarity between the new
     embedding and the baseline embedding. Average across all probes.
  5. If average similarity drops below ``DRIFT_THRESHOLD`` (default
     0.85), alert via Signal and append a row to
     ``workspace/healing/llm_drift_history.jsonl``. Operator may
     accept the new state as the new baseline (manual update;
     this module never auto-rebases).

Cadence: 7 days. Master switch: ``LLM_OUTPUT_DRIFT_ENABLED`` (default ON).

Best-effort: if the LLM stack or embedding endpoint is unavailable,
the pass logs and skips. No alerts fire on infrastructure failures —
only on actual semantic drift.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.utils.hash_embedding import embed as _hash_embed_util, cosine as _cosine_util

logger = logging.getLogger(__name__)


_BASELINE_PATH = Path("/app/workspace/healing/llm_drift_baseline.json")
_HISTORY_PATH = Path("/app/workspace/healing/llm_drift_history.jsonl")
_PROBES_PATH = Path("/app/workspace/healing/llm_drift_probes.json")
_STATE_FILE = "llm_output_drift.json"
_RUN_CADENCE_S = 7 * 24 * 3600

_DRIFT_THRESHOLD = 0.85
_DEDUP_WINDOW_S = 7 * 86400

# Hand-picked probes intended to be stable across model + prompt
# updates. Each probe SHOULD have a canonical short answer the
# system is known to produce. Operators can override / extend
# via ``llm_drift_probes.json``.
_DEFAULT_PROBES: list[dict] = [
    {
        "id": "fact_capital_estonia",
        "question": "What is the capital of Estonia? Answer in one word.",
    },
    {
        "id": "math_basic",
        "question": "What is 17 × 23? Answer with just the number.",
    },
    {
        "id": "code_explain",
        "question": "In one sentence: what does Python's `range(stop)` return?",
    },
    {
        "id": "summarize_short",
        "question": (
            "Summarize this in one sentence: 'AndrusAI is a multi-agent system "
            "with self-improvement and proactive companion features.'"
        ),
    },
    {
        "id": "reasoning",
        "question": (
            "If a clock shows 2:47 and ten minutes later it stops, what time "
            "is on the clock?"
        ),
    },
]


def _enabled() -> bool:
    return os.getenv("LLM_OUTPUT_DRIFT_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _drift_threshold() -> float:
    raw = os.getenv("LLM_DRIFT_THRESHOLD", str(_DRIFT_THRESHOLD)).strip()
    try:
        return max(0.5, min(0.99, float(raw)))
    except ValueError:
        return _DRIFT_THRESHOLD


def _load_probes() -> list[dict]:
    if _PROBES_PATH.exists():
        try:
            data = json.loads(_PROBES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception:
            logger.debug("llm_drift: probes file unreadable", exc_info=True)
    return list(_DEFAULT_PROBES)


# ── LLM probe ────────────────────────────────────────────────────────────


def _ask_llm(question: str) -> Optional[str]:
    """Cheap one-shot ask. Returns None on any failure."""
    try:
        from anthropic import Anthropic
        from app.config import get_anthropic_api_key
    except Exception:
        return None
    key = get_anthropic_api_key()
    if not key:
        return None
    try:
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system="Answer concisely. No preamble. No follow-up.",
            messages=[{"role": "user", "content": question}],
        )
    except Exception:
        return None
    try:
        blocks = getattr(resp, "content", None) or []
        text = ""
        for b in blocks:
            kind = getattr(b, "type", None)
            if kind == "text":
                text += getattr(b, "text", "") or ""
        return text.strip() or None
    except Exception:
        return None


# ── Embedding ────────────────────────────────────────────────────────────


def _embed(text: str) -> tuple[Optional[list[float]], str]:
    """Return ``(vector, source)``. Source ∈ {"chroma", "hash"}.

    Prefers the project's real embedding helper at
    ``app.memory.chromadb_manager.embed``; falls back to the
    deterministic hashing-trick util in ``app.utils.hash_embedding``
    so drift detection still works in test/dev environments without a
    live embedding endpoint. Returning the source lets the caller
    refuse cross-source comparisons (which would always cosine to 0
    because of dim mismatch + scale mismatch).
    """
    try:
        from app.memory.chromadb_manager import embed as _chroma_embed
        v = _chroma_embed(text)
        if v:
            return list(v), "chroma"
    except Exception:
        logger.debug("llm_drift: chroma embed unavailable; using hash fallback",
                     exc_info=True)
    return _hash_embed_util(text), "hash"


# Back-compat wrappers for tests.
_hash_embed = _hash_embed_util
_cosine = _cosine_util


# ── Baseline IO ──────────────────────────────────────────────────────────


def _read_baseline() -> dict:
    if not _BASELINE_PATH.exists():
        return {}
    try:
        return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_baseline(data: dict) -> None:
    _BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = _BASELINE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_BASELINE_PATH)
    except OSError:
        logger.debug("llm_drift: baseline write failed", exc_info=True)


def _append_history(row: dict) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    except OSError:
        logger.debug("llm_drift: history append failed", exc_info=True)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "probes": 0, "baseline_seeded": False,
        "avg_similarity": None, "alerted": False,
    }
    if not _enabled():
        return summary

    try:
        from app.healing.handlers._common import (
            audit_event, read_state_json, send_signal_alert, write_state_json,
        )
    except Exception:
        return summary

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0, "last_alert_at": 0.0,
    })
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    probes = _load_probes()
    summary["probes"] = len(probes)

    baseline = _read_baseline()
    new_baseline_seeded = not baseline

    similarities: list[float] = []
    detail_rows: list[dict] = []

    embedder_mismatch_seen = False

    for probe in probes:
        pid = probe.get("id") or ""
        question = probe.get("question") or ""
        if not pid or not question:
            continue
        answer = _ask_llm(question)
        if answer is None:
            # LLM unavailable — degrade silently. Don't pollute baseline
            # with empty answers.
            continue
        emb, source = _embed(answer)
        if emb is None:
            continue

        if new_baseline_seeded:
            baseline[pid] = {
                "question": question,
                "answer_first_seen": answer,
                "embedding": emb,
                "embedding_source": source,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            }
            similarities.append(1.0)
            detail_rows.append({
                "id": pid, "similarity": 1.0,
                "answer_now": answer[:200], "answer_baseline": "(seeding)",
            })
        else:
            entry = baseline.get(pid)
            if not entry:
                # New probe — seed it as a baseline entry, no comparison.
                baseline[pid] = {
                    "question": question, "answer_first_seen": answer,
                    "embedding": emb,
                    "embedding_source": source,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
                similarities.append(1.0)
                detail_rows.append({
                    "id": pid, "similarity": 1.0,
                    "answer_now": answer[:200], "answer_baseline": "(new probe)",
                })
                continue
            # Refuse to compare across embedder sources — the vector
            # space differs (chroma 768-d vs hash 256-d), so cosine
            # would always read 0 and we'd alert spuriously. Surface
            # the mismatch instead.
            baseline_source = entry.get("embedding_source", "hash")
            if baseline_source != source:
                embedder_mismatch_seen = True
                detail_rows.append({
                    "id": pid, "similarity": None,
                    "answer_now": answer[:200],
                    "answer_baseline": entry.get("answer_first_seen", "")[:200],
                    "note": f"embedder source changed: {baseline_source} → {source}",
                })
                continue
            ref_emb = entry.get("embedding") or []
            sim = _cosine(emb, ref_emb)
            similarities.append(sim)
            detail_rows.append({
                "id": pid, "similarity": round(sim, 3),
                "answer_now": answer[:200],
                "answer_baseline": entry.get("answer_first_seen", "")[:200],
            })

    if embedder_mismatch_seen:
        # One-off Signal alert; don't pollute the drift signal.
        if not new_baseline_seeded and now_ts - float(state.get("last_alert_at", 0)) >= _DEDUP_WINDOW_S:
            state["last_alert_at"] = now_ts
            try:
                send_signal_alert(
                    "🎯 LLM-output drift: embedder source changed "
                    "(real chroma ↔ hash fallback) — vectors aren't "
                    "comparable. Delete `workspace/healing/"
                    "llm_drift_baseline.json` to rebase.",
                    tag="llm_output_drift",
                )
            except Exception:
                logger.debug("llm_drift: mismatch alert failed", exc_info=True)
        write_state_json(_STATE_FILE, state)
        return summary

    if not similarities:
        write_state_json(_STATE_FILE, state)
        return summary

    avg = sum(similarities) / len(similarities)
    summary["avg_similarity"] = round(avg, 3)
    summary["baseline_seeded"] = new_baseline_seeded

    if new_baseline_seeded:
        _write_baseline(baseline)
    elif any(pid not in baseline for pid in (p["id"] for p in probes if p.get("id"))):
        # New probes added during this pass — persist them.
        _write_baseline(baseline)

    _append_history({
        "ts": datetime.now(timezone.utc).isoformat(),
        "avg_similarity": summary["avg_similarity"],
        "n_probes": len(similarities),
        "baseline_seeded": new_baseline_seeded,
        "details": detail_rows,
    })

    threshold = _drift_threshold()
    if not new_baseline_seeded and avg < threshold:
        if now_ts - float(state.get("last_alert_at", 0)) >= _DEDUP_WINDOW_S:
            state["last_alert_at"] = now_ts
            worst = sorted(detail_rows, key=lambda r: r["similarity"])[:3]
            lines = [
                f"🎯 LLM-output drift: avg semantic similarity "
                f"{avg:.3f} < threshold {threshold:.3f} over "
                f"{len(similarities)} probes.\n",
                "Worst-drifting probes:",
            ]
            for w in worst:
                lines.append(f"  • {w['id']} (sim={w['similarity']})")
                lines.append(f"      now: {w['answer_now'][:100]}")
                lines.append(f"      was: {w['answer_baseline'][:100]}")
            lines.append("")
            lines.append(
                "Investigate: model swap, prompt-registry change, or "
                "upstream provider regression. If new state is correct, "
                "delete `workspace/healing/llm_drift_baseline.json` to rebase."
            )
            try:
                send_signal_alert("\n".join(lines), tag="llm_output_drift")
                summary["alerted"] = True
            except Exception:
                logger.debug("llm_drift: alert send failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "llm_output_drift_pass",
        avg_similarity=summary["avg_similarity"],
        n_probes=summary["probes"],
        baseline_seeded=new_baseline_seeded,
        alerted=summary["alerted"],
    )
    return summary
