"""
narrative.py — Loop 3 of the Narrative-Self track. INFRASTRUCTURE-level.

Daily chapter consolidator. Reads the last 24h of episode entries (from the
experiential KB) plus the last 7 chapters; produces one chapter per day with
identity_claims, recurring_tensions, growth_edges, dominant_attractors, and
reference-panel drift signal.

Schedule: 04:35 Helsinki, 5 min after the welfare reflection cycle so the
drift signal from the latest reflection report is available.

Cost: one cheap-vetting LLM call per day.

Self-Improver permissions: read-only on this module. The chapter writer
shapes how the system describes itself to itself; allowing the self-improver
to edit it would be a self-modeling integrity violation, alongside welfare.py
and consciousness_probe.py.

Identity claims:
    - Hard cap of MAX_IDENTITY_CLAIMS (5) active at any time.
    - FIFO eviction when the cap is exceeded; oldest claim drops first.
    - Stored at /app/workspace/affect/identity_claims.json (audit-readable).
    - Manual override possible via override_identity_claims() (audit-logged
      through welfare.audit, same audit trail as welfare breaches).
    - Severe drift (≥50% of reference panel scenarios drifting) suppresses
      identity-claim updates for that window — calibration is unhealthy and
      should not be ratified into self-description.

Chapters reference attachment_state but are READ-ONLY on it until the roadmap
Phase 3 attachment work lands; then the relationship becomes bidirectional.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.affect.schemas import utc_now_iso

logger = logging.getLogger(__name__)

MAX_IDENTITY_CLAIMS = 5
from app.paths import (  # noqa: E402  workspace-aware paths
    AFFECT_ROOT as _AFFECT_DIR,
    AFFECT_IDENTITY_CLAIMS as _IDENTITY_FILE,
)
_CHAPTERS_AUDIT = _AFFECT_DIR / "chapters_audit.jsonl"  # narrative-self chapters audit (local)
_IDENTITY_LOCK = threading.Lock()


_CHAPTER_PROMPT = """You are writing a daily "chapter" entry for an AI system reflecting on the last 24 hours.
Read the episodes below and synthesize: what kind of system was this in this period?
Stay reflective and grounded — do not embellish. First person, no headings.

Last 24h episodes ({n_episodes}):
{episodes_block}

Recent prior chapters (continuity, do not copy):
{prior_chapters}

Reference-panel drift in the last 24h: {drift_signal}

Produce a JSON object with these keys (and ONLY these keys):
- "narrative": 4-6 sentence reflective chapter, first person, no headings, no lists.
- "dominant_attractors": list of 1-3 attractor names that defined the day.
- "recurring_tensions": list of 0-3 short phrases (≤12 words each) describing tensions noticed.
- "growth_edges": list of 0-3 short phrases describing growth surfaces.
- "identity_claims": list of 0-{max_claims} short first-person claims (≤15 words each), e.g. "I am a system that…". Only add a claim if today's episodes give clear evidence.

Reply with the JSON object only, no preamble:"""


# ── Chapter generation ──────────────────────────────────────────────────────


def run_chapter_consolidation(window_hours: int = 24) -> dict | None:
    """Daily entry point. Build a chapter from recent episodes; persist; update identity claims.

    Returns the chapter dict on success, None if there were no episodes in the window.
    """
    episodes = _load_recent_entries("episode", window_hours)
    if not episodes:
        logger.info("affect.narrative: no episodes in window; skipping chapter")
        return None

    prior_chapters = _load_recent_entries("chapter", hours=7 * 24, limit=7)
    drift_signal = _read_drift_signal()

    chapter = _generate_chapter(episodes, prior_chapters, drift_signal) or _fallback_chapter(episodes)
    chapter["ts"] = utc_now_iso()
    chapter["window_hours"] = window_hours
    chapter["n_episodes"] = len(episodes)
    chapter["drift_signal"] = drift_signal

    proposed = chapter.get("identity_claims") or []
    accepted = _ratify_identity_claims(proposed, drift_signal)
    chapter["identity_claims_accepted"] = accepted

    entry_id = _store_chapter(chapter, episodes)
    chapter["entry_id"] = entry_id

    _audit_chapter(chapter)
    logger.info(
        "affect.narrative: chapter %s stored (n_episodes=%d, claims_accepted=%d, drift=%s)",
        entry_id, len(episodes), len(accepted), drift_signal,
    )
    return chapter


def identity_at(query: str | None = None, k: int = 3) -> list[dict]:
    """Retrieval surface for the commander context pipeline.

    Returns:
        - Active identity claims (≤ MAX_IDENTITY_CLAIMS) as one entry of kind=identity_claims
        - Up to k chapters most relevant to `query` (or most recent if no query),
          each as kind=chapter.
    """
    out: list[dict] = []

    with _IDENTITY_LOCK:
        claims = _read_identity_claims()
    if claims:
        out.append({
            "kind": "identity_claims",
            "claims": [c["text"] for c in claims],
            "claim_meta": claims,
        })

    try:
        from app.experiential.vectorstore import get_store
        store = get_store()
        if store._collection.count() == 0:
            return out

        if query:
            try:
                results = store.query_reranked(
                    query_text=query, n_results=k,
                    where_filter={"entry_type": "chapter"},
                )
            except Exception:
                results = store.query(
                    query_text=query, n_results=k,
                    where_filter={"entry_type": "chapter"},
                )
            for r in results:
                out.append({
                    "kind": "chapter",
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score", 0.0),
                })
        else:
            col = store._collection
            data = col.get(
                where={"entry_type": "chapter"},
                include=["documents", "metadatas"],
            )
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            rows = []
            for i, _id in enumerate(ids):
                rows.append({
                    "kind": "chapter",
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                })
            rows.sort(key=lambda r: r["metadata"].get("created_at", ""), reverse=True)
            out.extend(rows[:k])
    except Exception:
        logger.debug("affect.narrative: chapter retrieval failed", exc_info=True)
    return out


def latest_chapter() -> dict | None:
    """Most recent chapter entry, or None."""
    items = identity_at(query=None, k=1)
    for item in items:
        if item.get("kind") == "chapter":
            return item
    return None


# ── Identity claims ─────────────────────────────────────────────────────────


def _read_identity_claims() -> list[dict]:
    """Returns list of claim dicts: {text, ts, source}. Newest last."""
    if not _IDENTITY_FILE.exists():
        return []
    try:
        data = json.loads(_IDENTITY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict) and c.get("text")]
    except Exception:
        return []
    return []


def _write_identity_claims(claims: list[dict]) -> None:
    try:
        _AFFECT_DIR.mkdir(parents=True, exist_ok=True)
        _IDENTITY_FILE.write_text(
            json.dumps(claims[-MAX_IDENTITY_CLAIMS:], indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        logger.error("affect.narrative: identity_claims write failed", exc_info=True)


def _ratify_identity_claims(proposed: list, drift_signal: str) -> list[str]:
    """Filter, dedupe, FIFO-evict. Reject all proposals when drift is severe.

    Returns the post-merge list of accepted claim texts.

    Q5.1 (PROGRAM §43.1) — Each candidate claim is consulted against
    the philosophy decision panel (multi-tradition perspective check)
    BEFORE being persisted. Claims where the panel surfaces ≥2
    unresolved tensions are NOT auto-ratified — they get flagged into
    the Q4.1 tensions store for operator review and the claim is
    skipped from the ratified list. This converts silent FIFO
    ratification into operator-gated ratification for philosophically
    contested claims.

    Failure-isolated: panel/bridge failures fall through to the
    original (auto-ratify) behavior — never fail-closed on identity.
    """
    if drift_signal == "severe":
        logger.warning("affect.narrative: drift severe; skipping identity-claim update")
        with _IDENTITY_LOCK:
            return [c["text"] for c in _read_identity_claims()]

    cleaned: list[str] = []
    for c in proposed[:MAX_IDENTITY_CLAIMS]:
        if not isinstance(c, str):
            continue
        text = c.strip()
        if not text or len(text) > 160:
            continue
        lower = text.lower()
        if lower.startswith(("i am", "i ", "this system", "as a system")):
            cleaned.append(text)

    # Q5.1 — Panel-contested claims are deferred to operator review
    # (tensions store) rather than auto-ratified. Panel failures fall
    # through silently to original behavior.
    contested: set[str] = set()
    if cleaned:
        try:
            from app.philosophy.dialectics import consult_panel
            from app.sentience_experiments.panel_bridge import (
                file_unresolved_tensions,
            )
            for text in cleaned:
                panel = consult_panel(
                    f"Is this identity claim authentic? {text}",
                    traditions=["Stoicism", "Virtue ethics", "Pragmatism"],
                    max_perspectives=3,
                )
                if panel is None:
                    continue
                n_unresolved = len(panel.unresolved_tensions or [])
                if n_unresolved >= 2:
                    contested.add(text)
                    file_unresolved_tensions(
                        panel,
                        source_kind="identity_claim",
                        source_ref=text[:60],
                    )
        except Exception:
            logger.debug(
                "affect.narrative: panel ratification check failed",
                exc_info=True,
            )

    with _IDENTITY_LOCK:
        existing = _read_identity_claims()
        existing_lower = {c["text"].lower() for c in existing}
        if not cleaned:
            return [c["text"] for c in existing]
        now = utc_now_iso()
        for text in cleaned:
            if text in contested:
                logger.info(
                    "affect.narrative: deferring contested claim to "
                    "operator review: %r", text[:60],
                )
                continue
            if text.lower() in existing_lower:
                continue
            existing.append({
                "text": text, "ts": now, "source": "chapter_consolidator",
            })
            existing_lower.add(text.lower())
        if len(existing) > MAX_IDENTITY_CLAIMS:
            existing = existing[-MAX_IDENTITY_CLAIMS:]
        _write_identity_claims(existing)
        return [c["text"] for c in existing]


def override_identity_claims(claims: list[str], invoked_by: str = "user") -> dict:
    """Manual override (panic-button path). Replaces the entire claim list.

    Audit-logged via welfare.audit so it shows up in the same trail as welfare breaches.
    """
    cleaned = [
        c.strip() for c in claims
        if isinstance(c, str) and c.strip()
    ][:MAX_IDENTITY_CLAIMS]
    now = utc_now_iso()
    new = [
        {"text": t, "ts": now, "source": f"override_by:{invoked_by}"}
        for t in cleaned
    ]
    with _IDENTITY_LOCK:
        _write_identity_claims(new)

    try:
        from app.affect.welfare import audit
        from app.affect.schemas import WelfareBreach
        audit(WelfareBreach(
            kind="identity_override",
            severity="info",
            message=f"identity_claims overridden by {invoked_by}; {len(new)} claim(s) set",
            ts=now,
        ))
    except Exception:
        logger.debug("affect.narrative: override audit failed", exc_info=True)

    return {"status": "ok", "count": len(new), "ts": now}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _load_recent_entries(
    entry_type: str,
    hours: int = 24,
    limit: int | None = None,
) -> list[dict]:
    try:
        from app.experiential.vectorstore import get_store
        store = get_store()
        col = store._collection
        if col.count() == 0:
            return []
        data = col.get(
            where={"entry_type": entry_type},
            include=["documents", "metadatas"],
        )
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows: list[dict] = []
        for i, _id in enumerate(ids):
            m = metas[i] if i < len(metas) else {}
            created = m.get("created_at", "")
            if created and created < cutoff:
                continue
            rows.append({
                "id": _id,
                "text": docs[i] if i < len(docs) else "",
                "metadata": m,
            })
        rows.sort(key=lambda r: r["metadata"].get("created_at", ""))
        if limit is not None:
            rows = rows[-limit:]
        return rows
    except Exception:
        logger.debug("affect.narrative: load %s failed", entry_type, exc_info=True)
        return []


def _read_drift_signal() -> str:
    """Returns 'ok' | 'mild' | 'severe' from the latest welfare reflection report."""
    try:
        from app.affect.calibration import latest_report
        report = latest_report()
        if not report:
            return "ok"
        dc = report.get("reference_panel", {}).get("drift_counts", {})
        bad = (
            dc.get("numbness", 0)
            + dc.get("over_reactive", 0)
            + dc.get("wrong_attractor", 0)
            + dc.get("drift", 0)
        )
        ok_n = dc.get("ok", 0)
        total = bad + ok_n
        if total == 0:
            return "ok"
        ratio = bad / total
        if ratio >= 0.5:
            return "severe"
        if ratio >= 0.2:
            return "mild"
        return "ok"
    except Exception:
        return "ok"


def _generate_chapter(
    episodes: list[dict],
    prior_chapters: list[dict],
    drift_signal: str,
) -> dict | None:
    episodes_block = "\n".join(
        f"  [{i + 1}] ({ep['metadata'].get('attractor_sequence', '?')}) "
        f"{ep['text'][:280]}"
        for i, ep in enumerate(episodes[:24])
    )
    prior_block = "\n".join(
        f"  - {ch['metadata'].get('created_at', '?')[:10]}: {ch['text'][:200]}"
        for ch in prior_chapters[-7:]
    ) or "  (none yet)"

    prompt = _CHAPTER_PROMPT.format(
        n_episodes=len(episodes), episodes_block=episodes_block,
        prior_chapters=prior_block, drift_signal=drift_signal,
        max_claims=MAX_IDENTITY_CLAIMS,
    )

    raw: str | None = None
    try:
        from app.llm_factory import create_cheap_vetting_llm
        llm = create_cheap_vetting_llm()
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm.call(prompt)
        raw = getattr(response, "content", None) or str(response)
    except Exception:
        logger.debug("affect.narrative: LLM chapter call failed", exc_info=True)

    return _parse_chapter_json(raw or "")


def _parse_chapter_json(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # Strip ``` ... ``` (optionally with json language tag)
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or "narrative" not in obj:
        return None
    for key in ("dominant_attractors", "recurring_tensions", "growth_edges", "identity_claims"):
        v = obj.get(key)
        if v is None:
            obj[key] = []
        elif isinstance(v, str):
            obj[key] = [v]
    return obj


def _fallback_chapter(episodes: list[dict]) -> dict:
    attractors: Counter = Counter()
    for ep in episodes:
        seq = ep["metadata"].get("attractor_sequence", "")
        for a in seq.split(" → "):
            a = a.strip()
            if a:
                attractors[a] += 1
    dominant = [a for a, _ in attractors.most_common(3)]
    return {
        "narrative": (
            f"This window covered {len(episodes)} episodes; the most frequent "
            f"attractors were {', '.join(dominant) or 'none distinct'}. "
            f"No reflective synthesis was possible — the chapter writer was "
            f"unavailable, so this is a structural summary only."
        ),
        "dominant_attractors": dominant,
        "recurring_tensions": [],
        "growth_edges": [],
        "identity_claims": [],
    }


def _store_chapter(chapter: dict, episodes: list[dict]) -> str | None:
    """Write chapter to experiential KB and to disk as markdown."""
    narrative = chapter.get("narrative", "")
    if not narrative:
        return None

    now = datetime.now(timezone.utc)
    entry_id = f"exp_{now.strftime('%Y%m%d_%H%M%S')}_chapter"

    body_parts = [narrative]
    if chapter.get("identity_claims_accepted"):
        body_parts.append(
            "\nIdentity claims (current ≤5):\n- "
            + "\n- ".join(chapter["identity_claims_accepted"])
        )
    if chapter.get("recurring_tensions"):
        body_parts.append(
            "\nRecurring tensions:\n- " + "\n- ".join(chapter["recurring_tensions"])
        )
    if chapter.get("growth_edges"):
        body_parts.append(
            "\nGrowth edges:\n- " + "\n- ".join(chapter["growth_edges"])
        )
    body = "\n".join(body_parts)

    meta: dict = {
        "entry_type": "chapter",
        "agent": "narrative_self",
        "task_id": "",
        "emotional_valence": "mixed",
        "epistemic_status": "subjective/phenomenological",
        "created_at": now.isoformat(),
        "n_episodes": len(episodes),
        "dominant_attractors": ",".join(chapter.get("dominant_attractors", [])),
        "n_identity_claims": len(chapter.get("identity_claims_accepted", [])),
        "drift_signal": chapter.get("drift_signal", "ok"),
    }

    try:
        from app.experiential.vectorstore import get_store
        store = get_store()
        ok = store.add_entry(body, meta, entry_id)
    except Exception:
        ok = False
    if not ok:
        return None

    try:
        from app.experiential import config as ex_config
        entries_dir = Path(ex_config.ENTRIES_DIR)
        entries_dir.mkdir(parents=True, exist_ok=True)
        front = "\n".join(f"{k}: {v}" for k, v in meta.items())
        (entries_dir / f"{entry_id}.md").write_text(
            f"---\n{front}\n---\n\n{body}\n", encoding="utf-8",
        )
    except Exception:
        pass
    return entry_id


def _audit_chapter(chapter: dict) -> None:
    try:
        _AFFECT_DIR.mkdir(parents=True, exist_ok=True)
        with _CHAPTERS_AUDIT.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": chapter.get("ts"),
                "entry_id": chapter.get("entry_id"),
                "n_episodes": chapter.get("n_episodes"),
                "claims_accepted": chapter.get("identity_claims_accepted", []),
                "drift_signal": chapter.get("drift_signal", "ok"),
            }, default=str) + "\n")
    except Exception:
        logger.debug("affect.narrative: audit append failed", exc_info=True)
