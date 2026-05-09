"""Rejected-hypothesis lessons-learned KB (Phase D #7, 2026-05-09).

Closes the gap from Wave 3 #6 in the original plan: cluster
rejected proposals across the system, distill *why* they were
rejected, and persist a lessons KB that future synthesis can consult
to avoid re-proposing the same shape.

Sources of "rejection":

  1. **Change requests** (REJECTED state) — `app.change_requests.store`.
     Each carries a ``reason`` (what was proposed) and an optional
     ``decision_reason`` (why operator said no).
  2. **Companion FEEDBACK events** with polarity=DOWN — the operator
     thumbs-downed an idea via Signal/React. Rejected idea + comment.
  3. **Goodhart-flagged signals** — `workspace/goodhart_reports.json`
     entries with severity ∈ {medium, high} that resulted in promotion
     blocks. Rejection of the proposed change.

Output:

  * ``workspace/companion/lessons_learned.json`` — clustered, deduped
    list of lessons. Each lesson has::

        {id, signature, count, sources: [...], example_proposals,
         example_reasons, first_seen, last_seen}

  * Public API: ``check_against(proposal_text)`` returns the closest
    lesson(s) with similarity score ≥ ``MATCH_THRESHOLD``. Future
    synthesis modules call this BEFORE proposing to bias away from
    known-rejected patterns.

Algorithm:

  1. Walk the three sources for events newer than ``LOOKBACK_DAYS``.
  2. For each rejection, compute a hashing-trick embedding of
     (proposal_text + decision_reason).
  3. Cluster by embedding similarity ≥ ``CLUSTER_THRESHOLD`` (0.75).
     Use union-find against existing lessons in the KB so re-runs
     keep extending the same cluster.
  4. For each cluster ≥ ``MIN_CLUSTER_SIZE`` (default 2), upsert a
     lesson row in the KB.

Cadence: 24 h. Master switch: ``LESSONS_LEARNED_ENABLED`` (default ON).

The KB is read-only from agent code — only this module mutates it.
The downside-bias step (synthesis-time consultation) is a separate
hook each consumer wires; this module's job is producing the KB.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from app.utils.hash_embedding import embed as _hash_embed, cosine as _hash_cosine

logger = logging.getLogger(__name__)


_KB_PATH = Path("/app/workspace/companion/lessons_learned.json")
_STATE_FILE = "lessons_learned.json"
_RUN_CADENCE_S = 24 * 3600

_LOOKBACK_DAYS = 90
# The hashing-trick embedding produces lower absolute cosine values
# for "similar but not identical" text than a real LLM embedding does
# (sparse buckets, no semantic compression). The thresholds below are
# tuned to that signal — they're meaningful relative scores, not what
# you'd see from sentence-transformers.
_CLUSTER_THRESHOLD = 0.45
_MATCH_THRESHOLD = 0.40
_MIN_CLUSTER_SIZE = 2
_EMBED_DIM = 256


def _enabled() -> bool:
    return os.getenv("LESSONS_LEARNED_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Embedding ─────────────────────────────────────────────────────────────
# Phase E #7: delegated to ``app.utils.hash_embedding`` so the same
# deterministic hashing-trick used by ``llm_output_drift`` is the
# single source. The thresholds in this module (CLUSTER_THRESHOLD /
# MATCH_THRESHOLD) were tuned against THAT distribution; aligning the
# implementations means the thresholds remain meaningful.


def _embed(text: str, dim: int = _EMBED_DIM) -> list[float]:
    return _hash_embed(text, dim=dim)


def _cosine(a, b):
    return _hash_cosine(a, b)


# ── Sources ──────────────────────────────────────────────────────────────


def _from_change_requests(cutoff: datetime) -> list[dict]:
    out: list[dict] = []
    try:
        from app.change_requests.store import list_all
        from app.change_requests.models import Status
    except Exception:
        return []
    try:
        rejected = list_all(status=Status.REJECTED, limit=300)
    except Exception:
        return []
    for cr in rejected:
        try:
            ts = datetime.fromisoformat(str(cr.created_at).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            continue
        proposal_text = f"{cr.path}: {cr.reason}"
        out.append({
            "source": "change_request",
            "ts": ts.isoformat(),
            "proposal_text": proposal_text[:1000],
            "decision_reason": (cr.decision_reason or "")[:500],
        })
    return out


def _from_companion_feedback(cutoff: datetime) -> list[dict]:
    events_path = Path("/app/workspace/companion/events.jsonl")
    if not events_path.exists():
        return []
    out: list[dict] = []
    cutoff_ts = cutoff.timestamp()
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "FEEDBACK":
                    continue
                payload = ev.get("payload") or {}
                if (payload.get("polarity") or "").lower() != "down":
                    continue
                ts = float(ev.get("ts") or 0)
                if ts < cutoff_ts:
                    continue
                proposal_text = (payload.get("comment") or "")[:1000]
                if not proposal_text:
                    proposal_text = f"idea_id={ev.get('idea_id', '')}"
                out.append({
                    "source": "feedback_down",
                    "ts": datetime.fromtimestamp(
                        ts, tz=timezone.utc
                    ).isoformat(),
                    "proposal_text": proposal_text,
                    "decision_reason": "operator thumbs-down via Signal/React",
                })
    except OSError:
        return []
    return out


def _from_goodhart_reports(cutoff: datetime) -> list[dict]:
    p = Path("/app/workspace/goodhart_reports.json")
    if not p.exists():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return []
    except Exception:
        return []
    out: list[dict] = []
    cutoff_ts = cutoff.timestamp()
    for r in rows:
        if not isinstance(r, dict):
            continue
        sev = str(r.get("severity") or "").lower()
        if sev not in ("medium", "high"):
            continue
        try:
            ts = float(r.get("detected_at") or 0)
        except (TypeError, ValueError):
            continue
        if ts < cutoff_ts:
            continue
        out.append({
            "source": "goodhart",
            "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "proposal_text": (r.get("description") or "")[:1000],
            "decision_reason": (
                f"Goodhart {sev}: {r.get('signal_type', '')}"
            ),
        })
    return out


# ── KB IO ────────────────────────────────────────────────────────────────


def _read_kb() -> list[dict]:
    if not _KB_PATH.exists():
        return []
    try:
        data = json.loads(_KB_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_kb(lessons: list[dict]) -> None:
    _KB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = _KB_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(lessons, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_KB_PATH)
    except OSError:
        logger.debug("lessons_learned: KB write failed", exc_info=True)


# ── Clustering ───────────────────────────────────────────────────────────


def _cluster_into_kb(events: list[dict], existing: list[dict]) -> list[dict]:
    """Merge new events into the existing lesson set.

    Each event is compared to every existing lesson centroid; if
    similarity ≥ CLUSTER_THRESHOLD, the event joins that lesson and
    the centroid is incrementally updated. Otherwise a new lesson is
    seeded.
    """
    lessons = [dict(l) for l in existing]
    # Hydrate centroids if missing (for KBs from older versions).
    for lesson in lessons:
        if not lesson.get("centroid"):
            lesson["centroid"] = _embed(lesson.get("signature_text", ""))

    for ev in events:
        text = ev.get("proposal_text", "") + " " + ev.get("decision_reason", "")
        emb = _embed(text)

        best_idx = -1
        best_sim = 0.0
        for idx, lesson in enumerate(lessons):
            sim = _cosine(emb, lesson["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= _CLUSTER_THRESHOLD and best_idx >= 0:
            lesson = lessons[best_idx]
            lesson["count"] = int(lesson.get("count", 0)) + 1
            sources = lesson.setdefault("sources", [])
            if ev["source"] not in sources:
                sources.append(ev["source"])
            examples = lesson.setdefault("example_proposals", [])
            if len(examples) < 5:
                examples.append(ev["proposal_text"][:200])
            reasons = lesson.setdefault("example_reasons", [])
            if ev["decision_reason"] and len(reasons) < 5:
                reasons.append(ev["decision_reason"][:200])
            ts = ev.get("ts", "")
            lesson["last_seen"] = ts or lesson.get("last_seen", "")
            # Incremental centroid update.
            n = max(1, lesson["count"])
            lesson["centroid"] = [
                ((n - 1) * c + e) / n
                for c, e in zip(lesson["centroid"], emb)
            ]
        else:
            lesson_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            lessons.append({
                "id": lesson_id,
                "signature_text": text[:300],
                "centroid": emb,
                "count": 1,
                "sources": [ev["source"]],
                "example_proposals": [ev["proposal_text"][:200]],
                "example_reasons": [ev["decision_reason"][:200]] if ev["decision_reason"] else [],
                "first_seen": ev.get("ts", ""),
                "last_seen": ev.get("ts", ""),
            })
    return lessons


# ── Public consumer API ──────────────────────────────────────────────────


def check_against(proposal_text: str, top_k: int = 3) -> list[dict]:
    """Return the closest lesson(s) for a new proposal.

    Each result: ``{id, similarity, sample_reason, count}``. Empty
    list if no lessons cross ``MATCH_THRESHOLD`` or KB is empty.

    Synthesis modules call this BEFORE proposing — if a high-similarity
    lesson exists, they should either reframe the proposal or skip.
    """
    lessons = _read_kb()
    if not lessons or not proposal_text:
        return []
    emb = _embed(proposal_text)
    scored: list[tuple[float, dict]] = []
    for lesson in lessons:
        sim = _cosine(emb, lesson.get("centroid", []))
        if sim < _MATCH_THRESHOLD:
            continue
        sample_reason = ""
        for r in lesson.get("example_reasons", []) or []:
            if r:
                sample_reason = r
                break
        scored.append((sim, {
            "id": lesson.get("id", ""),
            "similarity": round(sim, 3),
            "sample_reason": sample_reason,
            "count": int(lesson.get("count", 0)),
        }))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:top_k]]


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "events_seen": 0, "lessons_total": 0,
        "lessons_added_or_updated": 0, "alerted": False,
    }
    if not _enabled():
        return summary

    try:
        from app.healing.handlers._common import (
            audit_event, read_state_json, send_signal_alert, write_state_json,
        )
    except Exception:
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now_ts = time.time()
    if now_ts - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now_ts
    summary["ran"] = True

    cutoff = datetime.now(timezone.utc) - timedelta(days=_LOOKBACK_DAYS)
    events: list[dict] = []
    events.extend(_from_change_requests(cutoff))
    events.extend(_from_companion_feedback(cutoff))
    events.extend(_from_goodhart_reports(cutoff))
    summary["events_seen"] = len(events)

    if not events:
        write_state_json(_STATE_FILE, state)
        return summary

    existing = _read_kb()
    pre_count = sum(int(l.get("count", 0)) for l in existing)
    updated = _cluster_into_kb(events, existing)
    post_count = sum(int(l.get("count", 0)) for l in updated)

    # Filter out singleton clusters before persisting (noise).
    final = [l for l in updated if int(l.get("count", 0)) >= _MIN_CLUSTER_SIZE]
    _write_kb(final)
    summary["lessons_total"] = len(final)
    summary["lessons_added_or_updated"] = max(0, post_count - pre_count)

    # Quiet by default — only Signal-alert on a NEW lesson appearing
    # (not on every re-cluster). Use "lessons_total > previous_total + 0"
    # as the heuristic: if KB grew this pass, surface a one-line digest.
    if len(final) > len(existing) and final:
        new_lessons = final[len(existing):]
        lines = [
            f"📚 Lessons-learned KB: {len(new_lessons)} new lesson(s) "
            f"clustered from {summary['events_seen']} rejection events:\n"
        ]
        for l in new_lessons[:3]:
            sample = l.get("signature_text", "")[:120]
            lines.append(f"  • {sample}…")
            reasons = l.get("example_reasons", [])
            if reasons:
                lines.append(f"    why: {reasons[0][:100]}")
        try:
            send_signal_alert("\n".join(lines), tag="lessons_learned")
            summary["alerted"] = True
        except Exception:
            logger.debug("lessons_learned: alert failed", exc_info=True)

    write_state_json(_STATE_FILE, state)
    audit_event(
        "lessons_learned_pass",
        events_seen=summary["events_seen"],
        lessons_total=summary["lessons_total"],
        added=summary["lessons_added_or_updated"],
    )
    return summary
