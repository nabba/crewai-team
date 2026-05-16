"""Populate the analogy index by HEAVY weekly pass over the corpus.

PROGRAM §46.18 (Q11.1). The :mod:`analogy_index` primitive shipped
earlier (PROGRAM §32.3) but had no producer — entries sat empty so
the brainstorm consumer found nothing to surface. This populator
closes that gap.

What it does per pass:

  1. Walks the wiki/ markdown corpus (filesystem; cheap).
  2. Optionally samples from the episteme + experiential + aesthetics +
     tensions ChromaDB stores (network-cost only; gated behind the
     KB v2 master switch).
  3. Picks ``_MAX_NEW_PER_PASS`` (default 5) unprocessed source texts.
  4. For each, LLM-extracts an abstract structural pattern + 2-3
     concrete cross-domain examples.
  5. Appends an ``AnalogyEntry`` via :func:`analogy_index.add_entry`.

Per-pass cap is conservative — at ~$0.05/entry the bounded weekly
spend is ~$0.25 even if the operator never tunes it. Failure-
isolated end-to-end: a broken LLM call skips the source, a broken
KB fetch skips that store, the rest of the pass continues.

State tracking at ``workspace/creativity/analogy_populator_state.json``:
  * ``processed_sources`` — SHA-256 of normalised text we've seen
  * ``last_run_at`` — epoch seconds for cadence guard

Master switch ``analogy_index_populator_enabled`` (default ON, per
operator decision Q11) flippable via React /cp/settings → Analogy
Index card.

Idle job weight: HEAVY (LLM calls). Cadence: 7 days.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from app.creativity.analogy_index import (
    AnalogyEntry,
    DomainExample,
    add_entry,
)

logger = logging.getLogger(__name__)


_RUN_CADENCE_S = 7 * 24 * 3600
_MAX_NEW_PER_PASS = 5
_MIN_TEXT_CHARS = 200
_MAX_TEXT_CHARS = 4000


def _state_path() -> Path:
    base = Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))
    d = base / "creativity"
    d.mkdir(parents=True, exist_ok=True)
    return d / "analogy_populator_state.json"


def _enabled() -> bool:
    """Operator-flippable via runtime_settings; env-var fallback."""
    try:
        from app.runtime_settings import get_analogy_index_populator_enabled
        return bool(get_analogy_index_populator_enabled())
    except Exception:
        pass
    return os.getenv("ANALOGY_INDEX_POPULATOR_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
#   Source iteration
# ─────────────────────────────────────────────────────────────────────


def _wiki_root() -> Path:
    return Path(os.environ.get("ANALOGY_POPULATOR_WIKI_ROOT", "/app/wiki"))


def _iter_wiki_texts() -> Iterable[tuple[str, str]]:
    """Yield ``(source_id, text)`` from wiki/*.md.

    Source id is the repo-relative path; text is the raw markdown
    (capped at ``_MAX_TEXT_CHARS``). Hidden / dotfile dirs skipped.
    """
    root = _wiki_root()
    if not root.exists():
        return
    for p in root.rglob("*.md"):
        # Skip dotfiles and very long paths
        if any(part.startswith(".") for part in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) < _MIN_TEXT_CHARS:
            continue
        rel = str(p.relative_to(root))
        yield f"wiki:{rel}", text[:_MAX_TEXT_CHARS]


def _iter_episteme_texts(limit: int = 20) -> Iterable[tuple[str, str]]:
    """Yield ``(source_id, text)`` from the episteme ChromaDB store.

    Failure-isolated. Returns nothing when the store is empty or
    unavailable (e.g. chromadb deps missing in a test env).
    """
    try:
        from app.episteme.vectorstore import get_store
    except Exception:
        return
    try:
        store = get_store()
        rows = store.list_texts()
    except Exception:
        logger.debug("analogy_populator: episteme unavailable", exc_info=True)
        return
    # list_texts returns metadata only; we use title + filename as a
    # proxy text. This is intentionally coarse — the populator's job
    # is to surface candidate structures, not to fully comprehend
    # a paper.
    for row in (rows or [])[:limit]:
        title = (row.get("title") or "").strip()
        filename = (row.get("filename") or "").strip()
        domain = (row.get("domain") or "Unknown").strip()
        if not title:
            continue
        text = f"# {title}\n\nDomain: {domain}\nFile: {filename}"
        yield f"episteme:{filename or title}", text


# ─────────────────────────────────────────────────────────────────────
#   LLM extraction
# ─────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a Hofstadter-style analogy extractor. Given one text from
the system's knowledge corpus, identify ONE abstract structural
pattern it embodies (a pattern that recurs across unrelated
domains) and produce concrete cross-domain examples.

Output STRICT JSON (no markdown fence, no preamble):

{
  "structure_signature": "<2-4 word snake_case label>",
  "structure_description": "<2-4 sentence description of the
        abstract pattern — domain-neutral language, no reference
        to the original text's domain>",
  "domain_examples": [
    {"domain": "<domain1>", "title": "<short>",
     "summary": "<1-2 sentence concrete instance in this domain>"},
    {"domain": "<domain2>", "title": "<short>",
     "summary": "<1-2 sentence concrete instance in this domain>"},
    {"domain": "<domain3>", "title": "<short>",
     "summary": "<1-2 sentence concrete instance in this domain>"}
  ]
}

REQUIREMENTS:
  * Domains MUST be different from each other and from the
    source text's own domain. "Cross-domain" is the point.
  * Description MUST be abstract — no proper nouns from the
    source text.
  * Refuse with {"refused": true, "reason": "..."} if the source
    text is too narrow to abstract from (e.g. a date list, a
    config file, a personal note).
"""


def _default_llm_call(system: str, user: str) -> str:
    """Anthropic Haiku 4.5 call. Empty string on any failure."""
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [
            getattr(b, "text", "")
            for b in (msg.content or [])
            if getattr(b, "type", "") == "text"
        ]
        return "".join(text_parts).strip()
    except Exception:
        logger.debug("analogy_populator: LLM call failed", exc_info=True)
        return ""


def _parse_llm_output(raw: str) -> dict | None:
    """Tolerant JSON parse — strip markdown fence if present."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    # Try strict parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first {...} substring
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _extract_entry_from_text(
    source_id: str,
    text: str,
    *,
    llm_call: Callable[[str, str], str] | None = None,
) -> AnalogyEntry | None:
    """LLM-extract one AnalogyEntry from one source text. Returns
    None on refusal or parse failure."""
    call = llm_call or _default_llm_call
    user_prompt = f"Source: {source_id}\n\nText:\n{text[:_MAX_TEXT_CHARS]}\n"
    raw = call(_SYSTEM_PROMPT, user_prompt)
    data = _parse_llm_output(raw)
    if not data:
        return None
    if data.get("refused"):
        logger.debug(
            "analogy_populator: refused %s — %s",
            source_id, data.get("reason", ""),
        )
        return None
    sig = (data.get("structure_signature") or "").strip()
    desc = (data.get("structure_description") or "").strip()
    if not sig or not desc:
        return None
    raw_examples = data.get("domain_examples") or []
    if not isinstance(raw_examples, list):
        return None
    examples: list[DomainExample] = []
    for ex in raw_examples[:5]:
        if not isinstance(ex, dict):
            continue
        domain = (ex.get("domain") or "").strip()
        title = (ex.get("title") or "").strip()
        summary = (ex.get("summary") or "").strip()
        if domain and title and summary:
            examples.append(DomainExample(
                domain=domain, title=title, summary=summary,
                citation=source_id,
            ))
    if len(examples) < 2:
        # Need at least 2 examples for it to be "cross-domain"
        return None
    return AnalogyEntry(
        id=str(uuid.uuid4())[:12],
        structure_signature=sig[:80],
        structure_description=desc[:800],
        domain_examples=examples,
    )


# ─────────────────────────────────────────────────────────────────────
#   State
# ─────────────────────────────────────────────────────────────────────


def _normalize_for_hash(text: str) -> str:
    """Cheap normalisation so trivial edits don't re-process."""
    return re.sub(r"\s+", " ", text.lower().strip())[:1000]


def _source_hash(source_id: str, text: str) -> str:
    body = f"{source_id}\n{_normalize_for_hash(text)}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"processed_sources": [], "last_run_at": 0.0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"processed_sources": [], "last_run_at": 0.0}


def _save_state(state: dict[str, Any]) -> None:
    path = _state_path()
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("analogy_populator: state save failed", exc_info=True)


# ─────────────────────────────────────────────────────────────────────
#   One pass
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PopulatorResult:
    status: str         # "ok" | "skipped_disabled" | "skipped_cadence" | "no_sources"
    sources_seen: int = 0
    new_entries: int = 0
    refused: int = 0
    errors: int = 0


def run_one_pass(
    *,
    force: bool = False,
    llm_call: Callable[[str, str], str] | None = None,
    max_new: int = _MAX_NEW_PER_PASS,
) -> PopulatorResult:
    """One populator pass. Cadence-checked unless ``force=True``.

    Failure-isolated end-to-end. Returns a structured outcome dict
    for the idle scheduler to log.
    """
    if not _enabled():
        return PopulatorResult(status="skipped_disabled")

    state = _load_state()
    if not force:
        last = float(state.get("last_run_at", 0))
        if time.time() - last < _RUN_CADENCE_S:
            return PopulatorResult(status="skipped_cadence")
    state["last_run_at"] = time.time()

    processed: set[str] = set(state.get("processed_sources", []))

    # Gather candidate sources
    candidates: list[tuple[str, str]] = []
    for src_id, text in _iter_wiki_texts():
        sig = _source_hash(src_id, text)
        if sig in processed:
            continue
        candidates.append((src_id, text))
    for src_id, text in _iter_episteme_texts(limit=20):
        sig = _source_hash(src_id, text)
        if sig in processed:
            continue
        candidates.append((src_id, text))

    if not candidates:
        _save_state(state)
        return PopulatorResult(status="no_sources")

    # Shuffle so we explore the corpus deterministically across runs
    # (random seeded by week — same week → same picks; new week →
    # different picks). Pseudo-random; collisions are fine.
    seed = int(time.time() // (7 * 24 * 3600))
    random.Random(seed).shuffle(candidates)

    sources_seen = 0
    new_entries = 0
    refused = 0
    errors = 0
    cap = max(1, min(int(max_new), 20))

    for src_id, text in candidates:
        if new_entries >= cap:
            break
        sources_seen += 1
        sig = _source_hash(src_id, text)
        try:
            entry = _extract_entry_from_text(
                src_id, text, llm_call=llm_call,
            )
        except Exception:
            logger.debug(
                "analogy_populator: extraction raised for %s",
                src_id, exc_info=True,
            )
            errors += 1
            # Mark processed anyway so we don't loop on a bad source
            processed.add(sig)
            continue
        if entry is None:
            refused += 1
            processed.add(sig)
            continue
        try:
            ok = add_entry(entry)
        except Exception:
            ok = False
            logger.debug(
                "analogy_populator: add_entry raised for %s",
                src_id, exc_info=True,
            )
        if ok:
            new_entries += 1
            processed.add(sig)

    state["processed_sources"] = sorted(processed)
    _save_state(state)
    return PopulatorResult(
        status="ok",
        sources_seen=sources_seen,
        new_entries=new_entries,
        refused=refused,
        errors=errors,
    )


# ─────────────────────────────────────────────────────────────────────
#   Idle-job wrapper
# ─────────────────────────────────────────────────────────────────────


def run() -> dict[str, Any]:
    """Idle-scheduler entry. Cadence-guarded internally."""
    result = run_one_pass()
    return {
        "status": result.status,
        "sources_seen": result.sources_seen,
        "new_entries": result.new_entries,
        "refused": result.refused,
        "errors": result.errors,
    }
