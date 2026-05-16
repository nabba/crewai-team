"""Paper-to-experiment pipeline (Phase C #3, 2026-05-09).

Pulls recent papers from arXiv that match the operator's current
research interests (sourced from ``app.companion.interest_model``),
summarizes each via a cheap LLM, and proposes 1–3 concrete
experiments per paper that the gateway could actually run.

The intent is to close a loop the operator has wanted from day one:

    "Tell me about new research that's relevant to what I'm working
    on this week, and propose specific things to try."

Architecture, in order of dependency:

  1. Build query terms from ``interest_model.current_profile()``
     (top-3 topics) + a static list of always-on terms (e.g. the
     subsystems the operator declared they care about — affect,
     memory, governance, self-improvement).
  2. Fetch arXiv via the public ATOM API. No key, polite rate-limit.
     Restrict to the last ``LOOKBACK_DAYS`` and to a set of cs.*
     categories that match the operator's domain.
  3. For each candidate paper, dedup against
     ``workspace/episteme/papers_seen.json`` (by arxiv id).
  4. Summarize via a cheap LLM (Anthropic Haiku 4.5 or DeepSeek), passing
     the paper's title + abstract and asking for: one-paragraph
     summary, three implications for AndrusAI (what could we try),
     one experimental protocol.
  5. Append the result to ``workspace/proposed_experiments.jsonl``
     (one row per paper).
  6. Send a single Signal digest with the top-3 by relevance score.

Cadence: 7 days. Master switch: ``PAPER_PIPELINE_ENABLED`` (default
ON). If the LLM call or arXiv fetch fails the pass is a no-op — never
crashes the idle scheduler.

This is deliberately *additive*. It does not modify the episteme
vector store or the wiki index; the operator can ingest a paper into
the KB by hand once they decide it's worth keeping.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.utils import feed_parser

logger = logging.getLogger(__name__)


_PROPOSALS_PATH = Path("/app/workspace/proposed_experiments.jsonl")
_SEEN_PATH = Path("/app/workspace/episteme/papers_seen.json")
_STATE_FILE = "paper_pipeline.json"
_RUN_CADENCE_S = 7 * 24 * 3600

_ARXIV_BASE = "http://export.arxiv.org/api/query"
_ARXIV_TIMEOUT_S = 15.0
_LOOKBACK_DAYS = 14
_MAX_PAPERS_PER_PASS = 10
_TOP_DIGEST = 3

_DEFAULT_CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE")
_ALWAYS_ON_TERMS = (
    "agent", "self-improvement", "memory", "alignment", "reflection",
)


def _enabled() -> bool:
    return os.getenv("PAPER_PIPELINE_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Query construction ───────────────────────────────────────────────────


def _build_terms() -> list[str]:
    """Top interests + always-on terms, deduped."""
    terms: list[str] = []
    try:
        from app.companion.interest_model import current_profile
        prof = current_profile()
        for t in (prof.get("topics") or [])[:3]:
            name = (t.get("name") or "").strip() if isinstance(t, dict) else ""
            if name and len(name) <= 60:
                terms.append(name)
    except Exception:
        logger.debug("paper_pipeline: interest_model unavailable", exc_info=True)
    for t in _ALWAYS_ON_TERMS:
        if t not in terms:
            terms.append(t)
    # Deduplicate by lowercase, preserve order.
    seen = set()
    out: list[str] = []
    for t in terms:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out[:8]


def _build_arxiv_query(terms: list[str], categories: tuple[str, ...]) -> str:
    """Compose an arXiv ATOM query string."""
    cat_clause = " OR ".join(f"cat:{c}" for c in categories)
    term_clause = " OR ".join(f'all:"{t}"' for t in terms)
    return f"({term_clause}) AND ({cat_clause})"


# ── arXiv fetch ──────────────────────────────────────────────────────────


def _fetch_arxiv_atom(query: str, max_results: int) -> str:
    """Return the raw ATOM XML for one query. '' on any failure."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    })
    url = f"{_ARXIV_BASE}?{params}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "AndrusAI-Episteme/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_ARXIV_TIMEOUT_S) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        logger.debug("paper_pipeline: arxiv fetch failed", exc_info=True)
        return ""


def _parse_atom(xml: str, lookback_days: int) -> list[dict]:
    """Extract recent paper records from arXiv ATOM XML.

    Wraps the shared feed parser (``app.utils.feed_parser``) and
    augments each entry with our schema (``abstract`` instead of
    ``summary``; ``published`` parsed and filtered by lookback).
    Returns list of ``{id, title, abstract, published, categories}``.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []
    for entry in feed_parser.parse(xml, max_items=_MAX_PAPERS_PER_PASS * 4):
        try:
            pub_dt = datetime.fromisoformat(
                entry["published"].replace("Z", "+00:00"),
            )
        except (ValueError, TypeError):
            continue
        if pub_dt < cutoff:
            continue
        out.append({
            "id": entry["id"] or entry["link"],
            "title": " ".join(entry["title"].split()),
            "abstract": " ".join(entry["summary"].split()),
            "published": pub_dt.isoformat(),
            "categories": [],  # categories are parser-side; not exposed here
        })
    return out


# ── Dedup ────────────────────────────────────────────────────────────────


def _load_seen() -> dict[str, float]:
    """Return ``{paper_id: first_seen_ts}``. Empty on miss.

    The on-disk format is a list of ``[id, ts]`` pairs (or a bare list
    of ids for back-compat with older runs). We always normalize to
    a dict so the caller has insertion-ordered access.
    """
    if not _SEEN_PATH.exists():
        return {}
    try:
        data = json.loads(_SEEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    if isinstance(data, list):
        now = time.time()
        for entry in data:
            if isinstance(entry, list) and len(entry) == 2:
                pid, ts = entry
                try:
                    out[str(pid)] = float(ts)
                except (TypeError, ValueError):
                    out[str(pid)] = now
            elif isinstance(entry, str):
                out[entry] = now  # legacy: no ts available
    return out


def _save_seen(seen: dict[str, float]) -> None:
    """Persist ``seen``; cap to the 5000 newest entries by ts.

    Storing ts means the cap evicts ACTUAL oldest entries — the prior
    set-based implementation could evict arbitrary IDs (set ordering
    is not insertion order across Python runs).
    """
    _SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Sort by ts ascending → tail = newest 5000.
        ordered = sorted(seen.items(), key=lambda kv: kv[1])[-5000:]
        _SEEN_PATH.write_text(
            json.dumps([[pid, ts] for pid, ts in ordered]),
            encoding="utf-8",
        )
    except OSError:
        logger.debug("paper_pipeline: seen save failed", exc_info=True)


# ── Summarization + experiments ──────────────────────────────────────────


_LLM_SYSTEM_PROMPT = """You are AndrusAI's research analyst.

Given an arXiv paper title + abstract, output a STRICT JSON document
with these keys:

  summary:      One paragraph (≤120 words) plain-English summary.
  implications: Array of 1-3 short strings — concrete things AndrusAI
                might try based on this work. Speak in imperative
                ("try ... ", "wire ... into ..."). Each ≤30 words.
  experiment:   ONE paragraph describing a small experiment we could
                run inside our own gateway (≤80 words). Reference our
                actual subsystems where relevant: companion loop,
                memory belief outbox, affect trace, healing runbooks,
                meta-agent recipes, brainstorm, self-improvement.
  relevance:    Float 0.0-1.0 — how relevant THIS paper is to running
                an autonomous agent system. 1.0 = directly applicable,
                0.0 = unrelated.

  codeable:     Boolean. TRUE iff the experiment can be implemented
                as a single Python script (<30 minutes of work,
                clear inputs + clear pass/fail criterion, no
                novel-architecture dependency). Theoretical papers,
                hardware experiments, large-scale training runs →
                FALSE. Default to FALSE if uncertain.

  scaffold:     ONLY when codeable=true. Object with keys:
                  driver_purpose:  one-line spec of what the
                                    Python driver should do, written
                                    so a coder agent could implement
                                    it without re-reading the paper
                  acceptance:       array of 1-3 verifiable post-run
                                    assertions (e.g. "results.jsonl
                                    has ≥N rows", "mean accuracy
                                    > baseline + 0.05")
                  duration_min:     int — pessimistic time estimate
                                    (5..60)
                Omit ``scaffold`` entirely when codeable=false.

Output ONLY the JSON object, no preface, no markdown fence."""


def _summarize(title: str, abstract: str) -> dict | None:
    """Call the cheap LLM. Returns parsed JSON dict or None on failure."""
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
    except Exception:
        return None

    user_msg = f"Title: {title}\n\nAbstract: {abstract}"
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=_LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
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
        text = text.strip()
        # Tolerate the model wrapping the output in ```json … ```.
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        return json.loads(text)
    except Exception:
        logger.debug("paper_pipeline: LLM parse failed", exc_info=True)
        return None


# ── Persistence ──────────────────────────────────────────────────────────


_PROPOSALS_MAX_LINES = 2000  # Phase F #7: ~3 yrs of weekly digests at 10/wk


def _interest_profile_embedding() -> list[float] | None:
    """Build one centroid embedding from the operator's interest topics.

    Phase F #11 (2026-05-09): replaces the LLM-self-rated relevance
    score (unreliable — same model that wrote the summary scores its
    own enthusiasm) with cosine similarity against this centroid.
    Returns None when interest_model has no profile yet.
    """
    try:
        from app.companion.interest_model import current_profile
        from app.utils.hash_embedding import embed
    except Exception:
        return None
    profile = current_profile()
    topics = profile.get("topics") or []
    if not topics:
        return None
    # Concatenate the top topic names, weighted implicitly by frequency
    # (top-scored topics appear first; the hash-embed sums tokens so
    # repetition naturally weights them).
    text = " ".join(
        (t.get("name") or "") for t in topics[:15]
        if isinstance(t, dict) and t.get("name")
    )
    if not text.strip():
        return None
    return embed(text)


def _embedding_relevance(paper: dict, profile_emb: list[float] | None) -> float:
    """Cosine similarity between paper text and the interest centroid.

    Returns 0.0 when no profile is available — caller should fall
    back to the LLM relevance in that case.
    """
    if profile_emb is None:
        return 0.0
    try:
        from app.utils.hash_embedding import embed, cosine
    except Exception:
        return 0.0
    paper_text = f"{paper.get('title', '')}\n\n{paper.get('abstract', '')}"
    paper_emb = embed(paper_text)
    return max(0.0, cosine(profile_emb, paper_emb))


def _arxiv_signature(arxiv_id: str) -> str:
    """Sanitise an arXiv URL into a proposal-bridge-safe signature.

    The ATOM API returns URLs like ``http://arxiv.org/abs/2611.12345v1``.
    The bridge accepts ``[A-Za-z0-9_.-]+`` so we extract the final
    path component and strip any remaining unsafe characters.
    """
    if not arxiv_id:
        return ""
    tail = arxiv_id.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", tail)[:60]


def _render_paper_proposal(paper: dict, llm_out: dict) -> str:
    """Render the bridge-staged markdown for a paper proposal.

    Operator-readable, repo-friendly. On CR approval the file lands
    at ``docs/proposed_experiments/<sig>.md`` as a permanent record
    of "we considered trying this experiment."
    """
    arxiv_id = paper.get("id") or ""
    title = paper.get("title") or "Untitled paper"
    published = paper.get("published") or "?"
    summary = (llm_out.get("summary") or "").strip()
    experiment = (llm_out.get("experiment") or "").strip()
    implications = llm_out.get("implications") or []
    if not isinstance(implications, list):
        implications = [str(implications)]
    impl_block = "\n".join(f"- {str(i).strip()[:240]}" for i in implications[:3])
    relevance = float(llm_out.get("relevance") or 0.0)
    embedding_relevance = float(llm_out.get("embedding_relevance") or 0.0)
    abstract = (paper.get("abstract") or "").strip()
    return (
        f"# Paper-to-experiment proposal — {title[:120]}\n"
        f"\n"
        f"> Auto-generated by `app.episteme.paper_pipeline`. arXiv\n"
        f"> ATOM fetch matched the operator's interest profile +\n"
        f"> always-on terms.\n"
        f"\n"
        f"- **arXiv:** {arxiv_id}\n"
        f"- **Published:** {published}\n"
        f"- **LLM relevance:** {relevance:.2f}\n"
        f"- **Embedding relevance:** {embedding_relevance:.2f}\n"
        f"\n"
        f"## Summary\n"
        f"\n"
        f"{summary or '(no summary produced)'}\n"
        f"\n"
        f"## Implications for AndrusAI\n"
        f"\n"
        f"{impl_block or '- (none extracted)'}\n"
        f"\n"
        f"## Suggested experiment\n"
        f"\n"
        f"{experiment or '(no experiment proposed)'}\n"
        f"\n"
        f"## Original abstract\n"
        f"\n"
        f"{abstract[:1200]}\n"
        f"\n"
        f"## Operator action\n"
        f"\n"
        f"- **Try it**: approve the change-request to land this proposal\n"
        f"  at `docs/proposed_experiments/<sig>.md` as a permanent\n"
        f"  record, then run the experiment manually.\n"
        f"- **Defer**: leave the staged draft. After 30 days a\n"
        f"  staged-but-unpromoted proposal expires.\n"
        f"- **Reject**: rejecting the CR cleans up the staged record\n"
        f"  after the audit retention window.\n"
    )


def _stage_paper_proposal(paper: dict, llm_out: dict) -> None:
    """Stage a per-paper proposal via the proposal bridge.

    Best-effort: bridge unavailable / signature collision are logged
    at debug and never raise into the pipeline.
    """
    sig = _arxiv_signature(paper.get("id") or "")
    if not sig:
        return
    try:
        from app.proposal_bridge import stage
    except Exception:
        logger.debug("paper_pipeline: proposal_bridge unavailable", exc_info=True)
        return
    try:
        stage(
            source="paper_pipeline",
            signature=sig,
            title=(paper.get("title") or "")[:80],
            body_markdown=_render_paper_proposal(paper, llm_out),
            target_path=f"docs/proposed_experiments/{sig}.md",
            coding_session_spec=_build_coding_session_spec(paper, llm_out, sig),
        )
    except Exception:
        logger.debug("paper_pipeline: stage failed for %s", sig, exc_info=True)


def _build_coding_session_spec(paper: dict, llm_out: dict, sig: str) -> dict | None:
    """Q10.2 (PROGRAM §46.14): scaffold for running a paper's
    experiment proposal — produced ONLY when the LLM marked the
    paper as ``codeable``.

    Theoretical / hardware-bound / large-scale papers yield no
    spec — the proposal still lands as prose for the operator to
    read, but the daily-briefing 'queued codeable ideas' section
    won't surface them.

    When ``codeable=true`` AND the LLM produced a ``scaffold``
    object, we honor it (driver_purpose, acceptance, duration);
    otherwise we fall back to the generic shape so a missing
    ``scaffold`` field on a codeable=true paper still yields a
    workable spec.
    """
    if not bool(llm_out.get("codeable", False)):
        return None
    title = (paper.get("title") or "Untitled")[:80]
    experiment = (llm_out.get("experiment") or "").strip()[:200]
    scaffold = llm_out.get("scaffold") or {}
    if not isinstance(scaffold, dict):
        scaffold = {}
    driver_purpose = (
        (scaffold.get("driver_purpose") or "").strip()[:400]
        or f"experiment driver — implements: {experiment or 'see proposal'}"
    )
    duration = scaffold.get("duration_min")
    try:
        duration_min = max(5, min(int(duration), 240)) if duration else 60
    except (TypeError, ValueError):
        duration_min = 60
    raw_acceptance = scaffold.get("acceptance") or []
    if isinstance(raw_acceptance, list) and raw_acceptance:
        acceptance: list[str] = [
            str(a).strip()[:200]
            for a in raw_acceptance[:5]
            if str(a).strip()
        ]
    else:
        acceptance = [
            f"python workspace/experiments/{sig}/run.py --seed 42",
            f"verify workspace/experiments/{sig}/results.jsonl is non-empty",
            "compare against baseline noted in the paper proposal",
        ]
    return {
        "intent": f"Try paper experiment: {title}",
        "codeable": True,
        "files": [
            {
                "path": f"workspace/experiments/{sig}/run.py",
                "action": "create",
                "purpose": driver_purpose,
                "size_estimate": "~80 LOC",
            },
            {
                "path": f"workspace/experiments/{sig}/results.jsonl",
                "action": "create",
                "purpose": "append-only experiment outcome log",
                "size_estimate": "(generated)",
            },
        ],
        "acceptance": acceptance,
        "expected_duration_min": duration_min,
    }


def _append_proposal(paper: dict, llm_out: dict) -> None:
    """Persist the per-paper row to the JSONL ledger AND stage a
    proposal-bridge draft for eventual CR promotion.

    The JSONL ledger remains the source of truth for the Signal
    digest and the operator's running list of all reviewed papers
    (capped at ``_PROPOSALS_MAX_LINES`` lines). The bridge stage is
    additive — it lets a paper's proposal reach the change-request
    gate after the bridge's cooldown window if the operator doesn't
    act on it sooner.
    """
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arxiv_id": paper["id"],
        # Q10.3 (PROGRAM §46.15) — distinguish source. Existing
        # consumers reading the JSONL still get arxiv_id; new
        # consumers (daily briefing's queued-codeable section, the
        # React paper-list view) can filter by source.
        "source": paper.get("source", "arxiv"),
        "title": paper["title"],
        "published": paper["published"],
        "categories": paper["categories"],
        "summary": llm_out.get("summary", ""),
        "implications": llm_out.get("implications", []),
        "experiment": llm_out.get("experiment", ""),
        "relevance": float(llm_out.get("relevance", 0.0) or 0.0),
        # Q10.2 (PROGRAM §46.14) — operator-visible "is this codeable?"
        # flag + the LLM's optional scaffold object. The daily briefing
        # surfaces top codeable ideas; non-codeable papers still land
        # in the digest but without a "queued idea" promotion.
        "codeable": bool(llm_out.get("codeable", False)),
        "scaffold": (
            llm_out.get("scaffold")
            if bool(llm_out.get("codeable", False))
            and isinstance(llm_out.get("scaffold"), dict)
            else None
        ),
    }
    try:
        from app.utils.jsonl_retention import append_with_cap
        append_with_cap(
            _PROPOSALS_PATH, json.dumps(row, sort_keys=True),
            _PROPOSALS_MAX_LINES,
        )
    except Exception:
        logger.debug("paper_pipeline: proposal append failed", exc_info=True)

    _stage_paper_proposal(paper, llm_out)


# ── Main ──────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ran": False, "fetched": 0, "proposed": 0, "alerted": False,
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

    terms = _build_terms()
    if not terms:
        write_state_json(_STATE_FILE, state)
        return summary

    query = _build_arxiv_query(terms, _DEFAULT_CATEGORIES)
    xml = _fetch_arxiv_atom(query, max_results=_MAX_PAPERS_PER_PASS * 2)
    papers = _parse_atom(xml, lookback_days=_LOOKBACK_DAYS)
    # Tag arXiv rows with a source for downstream visibility.
    for p in papers:
        p.setdefault("source", "arxiv")

    # Q10.3 (PROGRAM §46.15) — multi-source feeds.
    # Pulls from OpenReview (NeurIPS/ICML/ICLR) + Python PEPs + W3C +
    # Hugging Face papers daily. Each source is failure-isolated; a
    # broken fetch from one source never blocks the others. Dedup
    # across sources happens via the existing _load_seen() ledger
    # below, keyed by the source-prefixed id.
    try:
        from app.episteme.feed_sources import fetch_extra_sources
        extras = fetch_extra_sources(
            lookback_days=_LOOKBACK_DAYS,
            max_per_source=max(3, _MAX_PAPERS_PER_PASS // 2),
        )
    except Exception:
        logger.debug(
            "paper_pipeline: extra-source fetch failed", exc_info=True,
        )
        extras = []
    papers.extend(extras)
    summary["fetched"] = len(papers)
    summary["fetched_arxiv"] = len(papers) - len(extras)
    summary["fetched_extra"] = len(extras)

    seen = _load_seen()
    proposals: list[tuple[dict, dict, float]] = []
    profile_emb = _interest_profile_embedding()

    for p in papers:
        if p["id"] in seen:
            continue
        if len(proposals) >= _MAX_PAPERS_PER_PASS:
            break
        llm_out = _summarize(p["title"], p["abstract"])
        if not llm_out:
            continue
        # Phase F #11: embedding similarity is the primary ranking
        # signal. LLM-self-rated relevance still recorded but used as
        # tiebreaker (and persisted for audit). When no profile exists
        # yet, fall back to LLM relevance only.
        emb_rel = _embedding_relevance(p, profile_emb)
        llm_rel = float(llm_out.get("relevance") or 0.0)
        ranking = emb_rel if profile_emb is not None else llm_rel
        llm_out["embedding_relevance"] = round(emb_rel, 3)
        llm_out["ranking_score"] = round(ranking, 3)
        _append_proposal(p, llm_out)
        seen[p["id"]] = time.time()
        proposals.append((p, llm_out, ranking))

    summary["proposed"] = len(proposals)
    _save_seen(seen)
    write_state_json(_STATE_FILE, state)

    if proposals:
        # Top by combined ranking score.
        proposals.sort(key=lambda t: t[2], reverse=True)
        lines = [
            f"📚 Paper-to-experiment: {len(proposals)} new arXiv paper(s) "
            f"reviewed against current interests "
            f"({', '.join(terms[:3])}, …):\n"
        ]
        for p, llm, ranking in proposals[:_TOP_DIGEST]:
            title = p["title"][:80]
            arxiv_id = p["id"].rsplit("/", 1)[-1]
            lines.append(
                f"  • [{arxiv_id}] {title}  "
                f"(rank {ranking:.2f} · emb {llm.get('embedding_relevance', 0.0):.2f} "
                f"· llm {float(llm.get('relevance') or 0.0):.2f})"
            )
            lines.append(f"    {llm.get('summary', '')[:200]}")
            implications = llm.get("implications", []) or []
            if implications:
                lines.append(f"    → try: {implications[0][:120]}")
            lines.append("")
        if len(proposals) > _TOP_DIGEST:
            lines.append(
                f"…and {len(proposals) - _TOP_DIGEST} more. "
                f"Full ledger: `workspace/proposed_experiments.jsonl`."
            )
        try:
            send_signal_alert("\n".join(lines), tag="paper_pipeline")
            summary["alerted"] = True
        except Exception:
            logger.debug("paper_pipeline: alert send failed", exc_info=True)

    audit_event(
        "paper_pipeline_pass",
        terms=terms,
        fetched=summary["fetched"],
        proposed=summary["proposed"],
        alerted=summary["alerted"],
    )
    return summary
