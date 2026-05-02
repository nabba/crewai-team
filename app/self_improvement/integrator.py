"""
app.self_improvement.integrator — route SkillDraft to the right KB.

Phase 3 of the overhaul. Replaces the prior "write markdown to
workspace/skills/" endpoint with a typed pipeline:

    SkillDraft  →  classify(proposed_kb)  →  write to KB  →  record provenance
                                                            →  emit SkillRecord

Four destinations:

    episteme       — cites external sources, describes *what is true*
    experiential   — distilled from observed task outcomes, narrative
    aesthetics     — style, tone, taste judgements
    tensions       — unresolved contradictions (open questions, not closed skills)

Classification is either:
    - caller-supplied (SkillDraft.proposed_kb) — trusted,
    - derived by a short LLM call when missing or unknown.

Writes go through a single `_write_to_kb(kb, record)` adapter that shields
callers from each store's idiosyncratic add_* API.

Also provides `regenerate_disk_mirror(out_dir)` for the presentation-layer
use case: any code still reading `workspace/skills/*.md` gets a fresh dump
from the KBs on demand.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.self_improvement.types import (
    SkillDraft, SkillRecord, NoveltyDecision,
)
from app.self_improvement.novelty import novelty_report
from app.self_improvement.store import update_gap_status
from app.self_improvement.types import GapStatus

logger = logging.getLogger(__name__)


# ── KB routing ───────────────────────────────────────────────────────────────

KB_CHOICES = ("episteme", "experiential", "aesthetics", "tensions")

# Provenance keys that flow Draft → Record.provenance → Chroma metadata.
# Adding a new transferred provenance field is a one-line change here; the
# integrate() and _write_to_kb() loops iterate this constant rather than
# hard-coding key names. The Chroma where-filter machinery (used by the
# retrieval orchestrator and trajectory context_builder) reads these keys
# verbatim, so a key listed here is automatically filterable downstream.
_PROVENANCE_KEYS_TO_INDEX: tuple[str, ...] = (
    # Trajectory-sourced (Phase 6, arXiv:2603.10600)
    "tip_type", "source_trajectory_id", "agent_role",
    # Transfer-memory (Phase 17) — set by app.transfer_memory.compiler
    "source_kind", "source_domain", "transfer_scope", "project_origin",
    "abstraction_score", "leakage_risk", "negative_transfer_tags",
    "evidence_refs",
)

_CLASSIFIER_PROMPT = """Classify this knowledge artifact into exactly one knowledge base.

The four destinations:

- episteme: theoretical, cites external sources, describes WHAT IS TRUE about
  a domain. Research summaries, methodology writeups, factual references.
- experiential: distilled from lived task experience. Narrative "we tried X,
  learned Y" entries. Lessons-learned, failure post-mortems, playbooks built
  from real runs.
- aesthetics: style, tone, taste, judgement. "How to write well", "what good
  code looks like", formatting guides.
- tensions: unresolved contradictions, open questions, ambiguities that
  resist closure. Not a closed skill — a flag that something is *not* settled.

Respond with EXACTLY ONE WORD from this list: episteme, experiential, aesthetics, tensions
No explanation, no punctuation.

Artifact topic: {topic}
Artifact rationale: {rationale}
Artifact content excerpt:
{excerpt}
"""


def classify_kb(draft: SkillDraft) -> str:
    """Determine the KB destination for a draft.

    Returns one of KB_CHOICES. If proposed_kb is already a valid choice,
    uses that (Learner's classification is trusted). Otherwise invokes a
    short LLM call against a fast model.

    Falls back to 'episteme' on any failure — it's the safest default for
    a research-backed skill and the largest existing KB.
    """
    if draft.proposed_kb and draft.proposed_kb.lower() in KB_CHOICES:
        return draft.proposed_kb.lower()

    excerpt = (draft.content_markdown or "")[:800]
    prompt = _CLASSIFIER_PROMPT.format(
        topic=draft.topic[:200],
        rationale=draft.rationale[:300],
        excerpt=excerpt,
    )

    try:
        from app.llm_factory import create_specialist_llm
        llm = create_specialist_llm(max_tokens=16, role="classify")
        raw = str(llm.call(prompt)).strip().lower()
        # Extract the first valid choice that appears in the response
        for choice in KB_CHOICES:
            if choice in raw:
                return choice
    except Exception:
        logger.debug("classify_kb: LLM classifier failed", exc_info=True)

    # Safe default
    return "episteme"


# ── KB writers ───────────────────────────────────────────────────────────────

def _write_to_kb(kb: str, record: SkillRecord) -> bool:
    """Unified write to any of the four KBs.

    Returns True on success. Each KB has its own add_* API; this function
    is the only place that knows the shape of each.
    """
    meta = {
        "topic": record.topic,
        "status": record.status,
        "created_at": record.created_at,
        "skill_record_id": record.id,
        "created_from_gap": record.provenance.get("gap_id", ""),
        "source": "self_improvement",
    }
    if record.superseded_by:
        meta["superseded_by"] = record.superseded_by
    # Forwarded provenance (trajectory + transfer-memory) — mirrored into
    # meta so retrieval can filter via ChromaDB where-clauses without
    # dereferencing SkillRecord.provenance. Truthy filter preserves the
    # original behaviour for empty strings; numeric zeros are treated as
    # "absent" which matches the existing tip_type semantics.
    for _key in _PROVENANCE_KEYS_TO_INDEX:
        _val = record.provenance.get(_key)
        if _val:
            meta[_key] = _val

    try:
        if kb == "episteme":
            from app.episteme.vectorstore import get_store
            # Episteme expects chunks; we treat the whole record as one chunk.
            # Downstream chunking/retrieval is handled by the KB's own logic.
            n = get_store().add_documents(
                chunks=[record.content_markdown],
                metadatas=[meta],
                ids=[record.id],
            )
            return n > 0
        elif kb == "experiential":
            from app.experiential.vectorstore import get_store
            return get_store().add_entry(
                text=record.content_markdown, metadata=meta, entry_id=record.id,
            )
        elif kb == "aesthetics":
            from app.aesthetics.vectorstore import get_store
            return get_store().add_pattern(
                text=record.content_markdown, metadata=meta, pattern_id=record.id,
            )
        elif kb == "tensions":
            from app.tensions.vectorstore import get_store
            return get_store().add_tension(
                text=record.content_markdown, metadata=meta, tension_id=record.id,
            )
        else:
            logger.warning(f"_write_to_kb: unknown kb '{kb}'")
            return False
    except Exception as exc:
        logger.debug(f"_write_to_kb[{kb}] failed: {exc}", exc_info=True)
        return False


# ── Provenance store ─────────────────────────────────────────────────────────

# Lightweight SkillRecord index kept in ChromaDB so we can enumerate all
# active records regardless of which KB they live in. This is the source
# of truth the Evaluator (Phase 4) and Consolidator (Phase 5) read from.

_RECORDS_COLLECTION = "skill_records"


def _get_records_collection():
    try:
        from app.memory.chromadb_manager import get_client
        client = get_client()
        return client.get_or_create_collection(
            _RECORDS_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
    except Exception:
        return None


def _persist_record(record: SkillRecord) -> bool:
    """Persist the SkillRecord index entry (not the content itself —
    that lives in the destination KB). Used for cross-KB enumeration."""
    col = _get_records_collection()
    if col is None:
        return False
    try:
        doc = json.dumps({
            "id": record.id, "topic": record.topic, "kb": record.kb,
            "status": record.status, "superseded_by": record.superseded_by,
            "usage_count": record.usage_count, "last_used_at": record.last_used_at,
            "created_at": record.created_at, "provenance": record.provenance,
            "content_markdown": record.content_markdown,
            "requires_mode": record.requires_mode,
            "requires_tier": record.requires_tier,
            "fallback_for_mode": record.fallback_for_mode,
            "requires_tools": record.requires_tools,
        })
        meta = {
            "kb": record.kb, "status": record.status,
            "topic": record.topic[:200], "created_at": record.created_at,
        }
        from app.memory.chromadb_manager import embed
        col.upsert(ids=[record.id], documents=[doc], metadatas=[meta],
                    embeddings=[embed(doc)])
        return True
    except Exception:
        logger.debug("_persist_record failed", exc_info=True)
        return False


def load_record(record_id: str) -> Optional[SkillRecord]:
    col = _get_records_collection()
    if col is None:
        return None
    try:
        r = col.get(ids=[record_id])
        if not r.get("ids"):
            return None
        d = json.loads(r["documents"][0])
        return SkillRecord(
            id=d["id"], topic=d["topic"], content_markdown="",
            kb=d["kb"], status=d["status"],
            superseded_by=d.get("superseded_by", ""),
            usage_count=int(d.get("usage_count", 0)),
            last_used_at=d.get("last_used_at", ""),
            provenance=d.get("provenance", {}),
            created_at=d.get("created_at", ""),
        )
    except Exception:
        return None


def list_records(
    kb: Optional[str] = None,
    status: str = "active",
    limit: int = 500,
) -> list[SkillRecord]:
    """Enumerate SkillRecords. Used by Evaluator/Consolidator."""
    col = _get_records_collection()
    if col is None:
        return []
    try:
        where: dict = {"status": status}
        if kb:
            where = {"$and": [{"status": status}, {"kb": kb}]}
        r = col.get(where=where, limit=limit)
        out: list[SkillRecord] = []
        for doc in r.get("documents", []):
            try:
                d = json.loads(doc)
                out.append(SkillRecord(
                    id=d["id"], topic=d["topic"], content_markdown=d.get("content_markdown", ""),
                    kb=d["kb"], status=d["status"],
                    superseded_by=d.get("superseded_by", ""),
                    usage_count=int(d.get("usage_count", 0)),
                    last_used_at=d.get("last_used_at", ""),
                    provenance=d.get("provenance", {}),
                    created_at=d.get("created_at", ""),
                    requires_mode=d.get("requires_mode", ""),
                    requires_tier=d.get("requires_tier", ""),
                    fallback_for_mode=d.get("fallback_for_mode", ""),
                    requires_tools=d.get("requires_tools", []),
                ))
            except Exception:
                continue
        return out
    except Exception:
        return []


def _record_from_doc(doc: str) -> Optional[SkillRecord]:
    try:
        d = json.loads(doc)
        return SkillRecord(
            id=d["id"], topic=d["topic"],
            content_markdown=d.get("content_markdown", ""),
            kb=d["kb"], status=d.get("status", "active"),
            superseded_by=d.get("superseded_by", ""),
            usage_count=int(d.get("usage_count", 0)),
            last_used_at=d.get("last_used_at", ""),
            provenance=d.get("provenance", {}),
            created_at=d.get("created_at", ""),
            requires_mode=d.get("requires_mode", ""),
            requires_tier=d.get("requires_tier", ""),
            fallback_for_mode=d.get("fallback_for_mode", ""),
            requires_tools=d.get("requires_tools", []),
        )
    except Exception:
        return None


def search_skills_scored(query: str, n: int = 6) -> list[tuple[SkillRecord, float]]:
    """Semantic search returning (record, cosine_distance) pairs.

    The records collection uses `hnsw:space=cosine` (see
    `_get_records_collection`), so distance is in [0, ~2]: 0 = identical,
    ~1 = orthogonal, >1 = anti-correlated. Callers gate on this distance
    to suppress weak surface-keyword matches — without that gate, a
    short subject-less query like "execute the plan" pulls the top-N
    nearest skills regardless of how irrelevant they actually are
    (the May 2026 weather-vs-forest contamination incident).

    Falls back to `list_records()` (with distance 0.0) when the collection
    is a stub without `.query()` support, matching the prior `search_skills`
    behaviour during unit tests.
    """
    col = _get_records_collection()
    if col is None:
        return []
    try:
        if hasattr(col, "query"):
            from app.memory.chromadb_manager import embed
            res = col.query(
                query_embeddings=[embed(query)],
                n_results=n,
                include=["documents", "distances"],
            )
            docs = res.get("documents", [[]])[0]
            dists = res.get("distances", [[]])[0]
            if not dists:
                dists = [0.0] * len(docs)
        else:
            return [(r, 0.0) for r in list_records(limit=n)]
    except Exception:
        return [(r, 0.0) for r in list_records(limit=n)]

    out: list[tuple[SkillRecord, float]] = []
    for doc, dist in zip(docs, dists):
        rec = _record_from_doc(doc)
        if rec is not None:
            out.append((rec, float(dist)))
    return out


def search_skills(query: str, n: int = 6) -> list[SkillRecord]:
    """Semantic search over active SkillRecords (distances discarded).

    Prefer `search_skills_scored` when callers need to gate on distance.
    """
    return [rec for rec, _ in search_skills_scored(query, n=n)]


def update_record(record: SkillRecord) -> bool:
    """Upsert the record index entry (used for status changes,
    usage_count updates, supersession marks)."""
    return _persist_record(record)


# ── Main entry point ─────────────────────────────────────────────────────────

def integrate(
    draft: SkillDraft,
    novelty_kbs: Optional[list[str]] = None,
    initial_status: str = "active",
) -> Optional[SkillRecord]:
    """Route a SkillDraft to the right KB and persist a SkillRecord.

    Workflow:
        1. Novelty check on content — reject if COVERED.
        2. Classify destination KB (honors draft.proposed_kb if valid).
        3. Allocate a stable ID from (kb, topic, created_at).
        4. Write content to the chosen KB.
        5. Mark supersession on any old records the draft replaces.
        6. Persist SkillRecord to the index.
        7. Mark the originating gap as RESOLVED_NEW.

    Args:
        draft: the candidate skill
        novelty_kbs: override the set of collections the novelty check
                     queries. Used by the legacy recovery flow to exclude
                     `team_shared` (where the source content still lives).
                     None = default (all 5 KBs).
        initial_status: status the persisted SkillRecord starts in.
                     Defaults to "active" (immediately retrievable).
                     Phase 17b transfer-memory drafts pass "shadow" so
                     records are persisted but invisible to the existing
                     retrieval path (which filters status="active");
                     they become retrievable only after Phase 17c
                     promotion. Other valid values: "superseded",
                     "archived" — generally only set by lifecycle code.

    Returns the persisted SkillRecord on success, None on rejection or error.
    """
    if not draft.content_markdown.strip():
        logger.debug("integrate: empty draft, rejecting")
        return None

    # 1. Content-level novelty check (decisive — Layer 2 of the Gate).
    #    The topic-level check happened before the Learner ran; this one
    #    catches the case where the topic was novel but the LLM wrote
    #    essentially-duplicate content from training data.
    try:
        rep = novelty_report(draft.content_markdown, kbs=novelty_kbs)
        if rep.decision == NoveltyDecision.COVERED:
            logger.info(
                f"integrate: rejecting draft '{draft.topic}' as COVERED "
                f"(d={rep.nearest_distance:.3f}, near={rep.nearest_kb})"
            )
            return None
    except Exception:
        logger.debug("integrate: novelty check errored, continuing", exc_info=True)

    # 2. Classify KB destination
    kb = classify_kb(draft)

    # 3. Allocate deterministic ID
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(
        f"{kb}::{draft.topic.lower().strip()}::{now}".encode()
    ).hexdigest()[:16]
    record_id = f"skill_{kb}_{digest}"

    provenance: dict = {
        "gap_id": draft.created_from_gap,
        "draft_id": draft.id,
        "rationale": draft.rationale[:500],
        "novelty_at_creation": float(draft.novelty_at_creation),
        "supersedes": list(draft.supersedes),
    }
    # Forwarded provenance (trajectory + transfer-memory) — only attach
    # when the draft carries a non-empty value, so SkillRecord stays
    # identical to the external-topic path when these fields are unused.
    for _key in _PROVENANCE_KEYS_TO_INDEX:
        _val = getattr(draft, _key, None)
        if _val:
            provenance[_key] = _val

    record = SkillRecord(
        id=record_id,
        topic=draft.topic,
        content_markdown=draft.content_markdown,
        kb=kb,
        status=initial_status,
        provenance=provenance,
        created_at=now,
    )

    # 4. Write to KB
    if not _write_to_kb(kb, record):
        logger.warning(f"integrate: write_to_kb[{kb}] failed for '{draft.topic}'")
        return None

    # 5. Mark supersession on old records
    for old_id in draft.supersedes:
        try:
            old = load_record(old_id)
            if old and old.status == "active":
                old.status = "superseded"
                old.superseded_by = record_id
                _persist_record(old)
        except Exception:
            logger.debug(f"integrate: supersede({old_id}) failed", exc_info=True)

    # 6. Persist to index
    _persist_record(record)

    # 7. Close the originating gap
    if draft.created_from_gap:
        try:
            update_gap_status(
                draft.created_from_gap,
                GapStatus.RESOLVED_NEW,
                notes=f"Resolved by skill {record_id} in {kb}",
            )
        except Exception:
            pass

    logger.info(
        f"integrate: '{draft.topic[:60]}' → {kb} (id={record_id}, "
        f"supersedes={len(draft.supersedes)})"
    )
    return record


# ── Disk mirror (presentation layer) ─────────────────────────────────────────

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_\-]")


def _safe_filename(topic: str, record_id: str) -> str:
    base = _SAFE_FILENAME_RE.sub("_", topic.strip().lower())[:80]
    return f"{base}__{record_id[-8:]}.md"


_MIRROR_MARKER = "<!-- generated-by: self_improvement.integrator -->"
_MIRROR_MIN_RECORDS = 5  # sparse-index guard: refuse to wipe when empty


def regenerate_disk_mirror(
    out_dir: Path = Path("/app/workspace/skills"),
    include_kbs: Optional[list[str]] = None,
    include_superseded: bool = False,
    force: bool = False,
) -> int:
    """Rewrite `workspace/skills/` from the KB index.

    The skills directory is a presentation layer for any legacy code that
    expects to read markdown files from disk. Source of truth is the KBs.

    Safety invariants (learned the hard way):

      1. **Sparse-index guard** — if fewer than _MIRROR_MIN_RECORDS exist
         in the index, this function refuses to run unless `force=True`.
         Prevents wiping existing files when the index is cold-starting.

      2. **Marker-scoped deletion** — only deletes files containing the
         `_MIRROR_MARKER` sentinel. Files written by other processes or
         users are preserved.

    Files written by this function carry _MIRROR_MARKER in their header.

    Returns the number of skills written (0 if guard tripped).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kbs = include_kbs or list(KB_CHOICES)
    # Pre-count what we'd write to apply the sparse-index guard
    total_records = sum(
        len(list_records(kb=kb, status="active", limit=2000)) for kb in kbs
    )
    if total_records < _MIRROR_MIN_RECORDS and not force:
        logger.debug(
            f"regenerate_disk_mirror: index has {total_records} records, "
            f"< threshold {_MIRROR_MIN_RECORDS}; skipping (use force=True to override)"
        )
        return 0

    # Wipe ONLY files bearing our marker — never touches foreign content
    preserve = {"learning_queue.md", "README.md"}
    for f in out_dir.glob("*.md"):
        if f.name in preserve:
            continue
        try:
            head = f.read_text(encoding="utf-8", errors="ignore")[:400]
            if _MIRROR_MARKER in head:
                f.unlink()
        except Exception:
            pass

    written = 0

    for kb in kbs:
        records = list_records(kb=kb, status="active", limit=2000)
        if include_superseded:
            records.extend(list_records(kb=kb, status="superseded", limit=2000))

        for rec in records:
            content = _load_record_content(rec)
            if not content:
                continue
            try:
                path = out_dir / _safe_filename(rec.topic, rec.id)
                header = (
                    f"{_MIRROR_MARKER}\n"
                    f"# {rec.topic}\n\n"
                    f"*kb: {rec.kb} | id: {rec.id} | status: {rec.status} "
                    f"| usage: {rec.usage_count} | created: {rec.created_at}*\n\n"
                )
                if rec.status == "superseded" and rec.superseded_by:
                    header += f"> **Superseded by:** `{rec.superseded_by}`\n\n"
                path.write_text(header + content)
                written += 1
            except Exception:
                logger.debug(f"regenerate_disk_mirror: write {rec.id} failed",
                             exc_info=True)

    logger.info(f"regenerate_disk_mirror: wrote {written} files to {out_dir}")
    return written


def _load_record_content(record: SkillRecord) -> str:
    """Fetch the actual markdown content from the KB the record lives in.

    The index stores metadata only; content lives in the KB. This helper
    hides the per-KB get() idiosyncrasies.
    """
    kb = record.kb
    try:
        if kb == "episteme":
            from app.episteme.vectorstore import get_store
            r = get_store()._collection.get(ids=[record.id])
        elif kb == "experiential":
            from app.experiential.vectorstore import get_store
            r = get_store()._collection.get(ids=[record.id])
        elif kb == "aesthetics":
            from app.aesthetics.vectorstore import get_store
            r = get_store()._collection.get(ids=[record.id])
        elif kb == "tensions":
            from app.tensions.vectorstore import get_store
            r = get_store()._collection.get(ids=[record.id])
        else:
            return ""
        docs = r.get("documents", [])
        return docs[0] if docs else ""
    except Exception:
        return ""
