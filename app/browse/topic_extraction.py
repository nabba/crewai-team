"""LLM-driven topic extraction over yesterday's browse titles.

Phase B (PROGRAM §50 — Q15.2). The Phase A store collects canonical
events (domain + path + title), but raw titles are noisy and the
unigram/bigram tokenizer in :mod:`app.companion.interest_model` would
fragment them. This module runs **once per day** to cluster titles
into topical labels via Anthropic Haiku 4.5 — cheap (~$0.0005/day)
and bounded.

Privacy contract
----------------

The titles ARE sent to Anthropic; this is the first point in the
browse pipeline where text leaves the host. Two structural guards
back that decision:

  1. **Blocklisted domains are already absent from events.jsonl** —
     they were dropped at the reader's edge in Phase A. So a
     "kanta.fi" or "paypal.com" title can't reach the prompt no
     matter what we do here.
  2. **Pre-flight redaction** strips email-shaped and phone-shaped
     tokens from titles before the prompt is assembled. The
     ``_redact_pii`` helper is pinned by tests.

Two privacy invariants are also pinned by tests:

  * ``test_blocklisted_titles_never_in_llm_batch`` — a synthetic
    paypal.com title goes in, asserts absent from the prompt payload.
  * ``test_raw_urls_never_in_llm_batch`` — only titles + counts ever
    appear in the prompt body. URLs (domain + path) stay local.

Output
------

Per-day JSON at ``workspace/browse/topics/YYYY-MM-DD.json``::

    {
      "day": "2026-05-16",
      "generated_at": "2026-05-17T03:14:00+00:00",
      "model": "claude-haiku-4-5-20251001",
      "topics": [
        {"label": "claude code", "title_count": 12,
         "sample_titles": ["Claude Code — Anthropic", "Claude Code Docs"]},
        {"label": "finnish nature", "title_count": 4, ...}
      ]
    }

Idempotent: if the per-day file already exists, the daily pass for
that day is a no-op (operator can force a rerun by deleting the file).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from app.browse import store
from app.browse.models import BrowseEvent

logger = logging.getLogger(__name__)


_MIN_INTERVAL_S = 23 * 3600  # one batch per day
_MAX_TITLES_PER_BATCH = 250  # safety cap on prompt size + cost
_TOPIC_DIRNAME = "topics"


def _enabled() -> bool:
    """Two-layer master switch:
       * ``BROWSE_INGESTION_ENABLED`` must be on (Phase A gate)
       * ``BROWSE_LLM_TOPICS_ENABLED`` must be on (Phase B gate)

    The second switch is the explicit "yes I'm OK with titles leaving
    the host" opt-in. Default ON when Phase A is on — but it's its own
    flag so the operator can disable just the LLM step while keeping
    on-host event collection running.
    """
    if not store.enabled():
        return False
    return os.getenv("BROWSE_LLM_TOPICS_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Redaction ────────────────────────────────────────────────────────


_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
# International phone-ish patterns. Loose on purpose — the cost of a
# false positive is a redacted word; the cost of a false negative is a
# phone number reaching Anthropic.
_PHONE_RE = re.compile(
    r"(?:\+\d[\d\s()-]{6,}\d|\b\d{3,}[ -]\d{3,}[ -]\d{3,}\b)"
)


def _redact_pii(text: str) -> str:
    """Strip email + phone-shaped tokens before sending titles to the
    LLM. Returns the cleaned text. Pinned by tests."""
    if not text:
        return text
    text = _EMAIL_RE.sub("<email>", text)
    text = _PHONE_RE.sub("<phone>", text)
    return text


# ── Data model ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BatchInput:
    """One row going into the LLM prompt."""

    title: str
    count: int
    domain: str  # included in the prompt for context, NOT as a URL


@dataclass
class TopicCluster:
    label: str
    title_count: int = 0
    sample_titles: list[str] = field(default_factory=list)


@dataclass
class TopicsResult:
    day: date
    generated_at: str
    model: str
    topics: list[TopicCluster] = field(default_factory=list)
    note: str | None = None  # "no_titles" / "llm_unavailable" / "llm_failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day.isoformat(),
            "generated_at": self.generated_at,
            "model": self.model,
            "topics": [
                {
                    "label": t.label,
                    "title_count": t.title_count,
                    "sample_titles": list(t.sample_titles),
                }
                for t in self.topics
            ],
            "note": self.note,
        }


# ── Batch assembly ────────────────────────────────────────────────────


def _aggregate_titles(events: list[BrowseEvent]) -> list[_BatchInput]:
    """Deduplicate titles + accumulate visit counts. Titles missing or
    empty (``None``) are excluded from the batch — they carry no signal.

    Returns the top ``_MAX_TITLES_PER_BATCH`` rows by visit count.
    """
    counter: Counter[tuple[str, str]] = Counter()
    for e in events:
        if not e.title:
            continue
        clean = _redact_pii(e.title).strip()
        if not clean:
            continue
        counter[(clean, e.domain)] += 1
    rows: list[_BatchInput] = [
        _BatchInput(title=t, count=c, domain=d)
        for (t, d), c in counter.most_common(_MAX_TITLES_PER_BATCH)
    ]
    return rows


_SYSTEM_PROMPT = """\
You are clustering browser visit titles into topical themes.

Input: a deduplicated list of page titles with visit counts and the
host domain (one per line). The titles describe what the user has
read; cluster them into 5-15 topics representing the user's
interests.

Output STRICT JSON of the form:
{
  "topics": [
    {"label": "topic slug", "title_indexes": [0, 5, 7]},
    {"label": "another topic", "title_indexes": [1, 2]}
  ]
}

Rules:
  * Each label is a short noun phrase (2-4 words, lowercase).
  * Each title belongs to exactly ONE topic.
  * Skip titles that look meaningless ("Untitled", "Loading", empty).
    Just leave them out of any topic — don't invent a "junk" bucket.
  * Group only when 2+ titles share a clear theme. Singletons go in a
    catch-all topic labelled "miscellaneous".
  * Do not invent topics not represented by titles. The labels must
    summarise what's actually present.
  * No more than 15 topics total.
"""


def _build_user_prompt(rows: list[_BatchInput]) -> str:
    """Assemble the user-side prompt.

    The format is ``<idx>\\t<count>\\t<domain>\\t<title>`` — tab-separated
    for stability. Domains are included so the LLM can disambiguate
    titles like "Home" (could be GitHub home, could be Wikipedia home),
    but NO paths or query strings ever appear here.
    """
    lines = ["Titles (idx\\tcount\\tdomain\\ttitle):"]
    for i, r in enumerate(rows):
        lines.append(f"{i}\t{r.count}\t{r.domain}\t{r.title}")
    return "\n".join(lines)


# ── LLM call ──────────────────────────────────────────────────────────


_MODEL_DEFAULT = "claude-haiku-4-5-20251001"


def _default_llm_call(system: str, user: str) -> str:
    """Anthropic Haiku 4.5. Empty string on any failure."""
    try:
        import anthropic
    except ImportError:
        return ""
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=_MODEL_DEFAULT,
            max_tokens=1500,
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
        logger.debug("browse.topic_extraction: LLM call failed", exc_info=True)
        return ""


def _parse_llm_output(raw: str) -> dict | None:
    """Tolerant JSON parse — accept markdown-fenced JSON too."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Composition ───────────────────────────────────────────────────────


def _compose_clusters(
    raw_topics: list[dict[str, Any]],
    rows: list[_BatchInput],
) -> list[TopicCluster]:
    """Materialise TopicCluster objects from the LLM's index-based
    output. Drops out-of-range indices defensively."""
    out: list[TopicCluster] = []
    used_indices: set[int] = set()
    for t in raw_topics or []:
        if not isinstance(t, dict):
            continue
        label = str(t.get("label") or "").strip().lower()
        if not label:
            continue
        idxs = t.get("title_indexes") or []
        if not isinstance(idxs, list):
            continue
        valid_idxs: list[int] = []
        for i in idxs:
            try:
                ii = int(i)
            except (TypeError, ValueError):
                continue
            if 0 <= ii < len(rows) and ii not in used_indices:
                valid_idxs.append(ii)
                used_indices.add(ii)
        if not valid_idxs:
            continue
        # Title count = sum of visit counts across members.
        member_count = sum(rows[i].count for i in valid_idxs)
        samples = [rows[i].title for i in valid_idxs[:3]]
        out.append(
            TopicCluster(
                label=label,
                title_count=member_count,
                sample_titles=samples,
            )
        )
    return out


# ── Public API ────────────────────────────────────────────────────────


def topics_path_for(day: date, *, base: Path | None = None) -> Path:
    root = base if base else store.resolve_base()
    return root / _TOPIC_DIRNAME / f"{day.isoformat()}.json"


def topics_for_day(
    day: date, *, base: Path | None = None,
) -> TopicsResult | None:
    """Load yesterday's cluster output. ``None`` if not yet generated."""
    p = topics_path_for(day, base=base)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return TopicsResult(
            day=date.fromisoformat(raw["day"]),
            generated_at=raw["generated_at"],
            model=raw.get("model", ""),
            topics=[
                TopicCluster(
                    label=str(t.get("label", "")),
                    title_count=int(t.get("title_count", 0)),
                    sample_titles=[str(s) for s in t.get("sample_titles", [])],
                )
                for t in raw.get("topics", [])
            ],
            note=raw.get("note"),
        )
    except (KeyError, ValueError, TypeError):
        return None


def extract_topics_for_day(
    day: date,
    *,
    llm_call: Callable[[str, str], str] | None = None,
    base: Path | None = None,
) -> TopicsResult:
    """Cluster all the day's titles via the LLM. Idempotent — if the
    per-day output already exists, returns it without re-calling.

    Failure-isolated. When the LLM is unavailable or returns garbage,
    returns a result with ``note`` set and an empty topics list."""
    existing = topics_for_day(day, base=base)
    if existing is not None:
        return existing
    events = store.list_events_for_day(day, base=base)
    rows = _aggregate_titles(events)
    out_path = topics_path_for(day, base=base)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        result = TopicsResult(
            day=day,
            generated_at=datetime.now(timezone.utc).isoformat(),
            model="",
            topics=[],
            note="no_titles",
        )
        try:
            out_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass
        return result

    call = llm_call or _default_llm_call
    system = _SYSTEM_PROMPT
    user = _build_user_prompt(rows)
    raw = call(system, user)
    parsed = _parse_llm_output(raw)

    note: str | None = None
    clusters: list[TopicCluster] = []
    model_id = _MODEL_DEFAULT
    if parsed is None:
        note = "llm_unavailable" if not raw else "llm_failed"
    else:
        clusters = _compose_clusters(parsed.get("topics") or [], rows)
        if not clusters:
            note = "no_clusters"

    result = TopicsResult(
        day=day,
        generated_at=datetime.now(timezone.utc).isoformat(),
        model=model_id,
        topics=clusters,
        note=note,
    )
    try:
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        logger.warning("browse.topic_extraction: write failed for %s", out_path)
    return result


# ── Idle job ──────────────────────────────────────────────────────────


def _last_run_path(*, base: Path | None = None) -> Path:
    root = base if base else store.resolve_base()
    return root / ".last_topics_at"


def _due(*, base: Path | None = None) -> bool:
    p = _last_run_path(base=base)
    if not p.exists():
        return True
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return True
    return age >= _MIN_INTERVAL_S


def _touch(*, base: Path | None = None) -> None:
    p = _last_run_path(base=base)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8",
        )
    except OSError:
        pass


def run_topic_extraction_tick(
    *,
    llm_call: Callable[[str, str], str] | None = None,
    base: Path | None = None,
    now: datetime | None = None,
) -> None:
    """One daily pass: cluster yesterday's titles.

    Cadence-guarded internally so calling this on the LIGHT idle loop
    every minute is fine — it returns within microseconds when not due.
    """
    if not _enabled():
        return
    if not _due(base=base):
        return
    target_day = ((now or datetime.now(timezone.utc)) - timedelta(days=1)).date()
    try:
        result = extract_topics_for_day(target_day, llm_call=llm_call, base=base)
    except Exception:
        logger.debug("browse.topic_extraction: tick raised", exc_info=True)
        return
    _touch(base=base)
    note = f" note={result.note}" if result.note else ""
    logger.info(
        "browse.topics: day=%s topics=%d%s",
        target_day, len(result.topics), note,
    )


def get_idle_jobs() -> list[tuple[str, Callable[[], None], str]]:
    from app.idle_scheduler import JobWeight
    return [("browse-topics", run_topic_extraction_tick, JobWeight.LIGHT)]
