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


# Minimal regex parser — atom XML, one entry block at a time. Avoids
# pulling lxml as a dep just for this. Title / abstract / ID / date /
# categories — that's what we need.
_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ID_RE = re.compile(r"<id>(.*?)</id>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
_PUBLISHED_RE = re.compile(r"<published>(.*?)</published>", re.DOTALL)
_CATEGORY_RE = re.compile(r'<category[^>]*term="([^"]+)"', re.DOTALL)


def _parse_atom(xml: str, lookback_days: int) -> list[dict]:
    """Extract paper records from ATOM XML.

    Returns list of dicts with id (arxiv URL), title, abstract,
    published, categories (list).
    """
    if not xml:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    out: list[dict] = []
    for entry in _ENTRY_RE.findall(xml):
        m_id = _ID_RE.search(entry)
        m_title = _TITLE_RE.search(entry)
        m_abs = _SUMMARY_RE.search(entry)
        m_pub = _PUBLISHED_RE.search(entry)
        if not (m_id and m_title and m_abs and m_pub):
            continue
        try:
            pub_dt = datetime.fromisoformat(m_pub.group(1).strip().replace("Z", "+00:00"))
        except Exception:
            continue
        if pub_dt < cutoff:
            continue
        cats = _CATEGORY_RE.findall(entry)
        out.append({
            "id": m_id.group(1).strip(),
            "title": " ".join(m_title.group(1).split()),
            "abstract": " ".join(m_abs.group(1).split()),
            "published": pub_dt.isoformat(),
            "categories": cats,
        })
    return out


# ── Dedup ────────────────────────────────────────────────────────────────


def _load_seen() -> set[str]:
    if not _SEEN_PATH.exists():
        return set()
    try:
        data = json.loads(_SEEN_PATH.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def _save_seen(seen: set[str]) -> None:
    _SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Cap at 5000 — papers we've never proposed in N years are unlikely to matter.
        entries = list(seen)[-5000:]
        _SEEN_PATH.write_text(json.dumps(entries), encoding="utf-8")
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


def _append_proposal(paper: dict, llm_out: dict) -> None:
    _PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arxiv_id": paper["id"],
        "title": paper["title"],
        "published": paper["published"],
        "categories": paper["categories"],
        "summary": llm_out.get("summary", ""),
        "implications": llm_out.get("implications", []),
        "experiment": llm_out.get("experiment", ""),
        "relevance": float(llm_out.get("relevance", 0.0) or 0.0),
    }
    try:
        with _PROPOSALS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    except OSError:
        logger.debug("paper_pipeline: proposal append failed", exc_info=True)


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
    summary["fetched"] = len(papers)

    seen = _load_seen()
    proposals: list[tuple[dict, dict]] = []

    for p in papers:
        if p["id"] in seen:
            continue
        if len(proposals) >= _MAX_PAPERS_PER_PASS:
            break
        llm_out = _summarize(p["title"], p["abstract"])
        if not llm_out:
            continue
        _append_proposal(p, llm_out)
        seen.add(p["id"])
        proposals.append((p, llm_out))

    summary["proposed"] = len(proposals)
    _save_seen(seen)
    write_state_json(_STATE_FILE, state)

    if proposals:
        # Top by relevance.
        proposals.sort(key=lambda t: float(t[1].get("relevance") or 0.0), reverse=True)
        lines = [
            f"📚 Paper-to-experiment: {len(proposals)} new arXiv paper(s) "
            f"reviewed against current interests "
            f"({', '.join(terms[:3])}, …):\n"
        ]
        for p, llm in proposals[:_TOP_DIGEST]:
            rel = float(llm.get("relevance") or 0.0)
            title = p["title"][:80]
            arxiv_id = p["id"].rsplit("/", 1)[-1]
            lines.append(f"  • [{arxiv_id}] {title}  (relevance {rel:.2f})")
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
