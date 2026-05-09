"""Aggregated user-interest profile (Phase B #1, 2026-05-09).

Mines five low-cost signals into a unified per-user "what does the
operator care about" profile:

  * ``conversations.db``       — outbound & inbound text (14-day window)
  * Inbox triage results       — workspace/life_companion/email_monitor.json
  * Calendar event titles      — Google Calendar (next 30d + last 14d)
  * Companion FEEDBACK events  — workspace/companion/events.jsonl
  * Affect-trace topic tags    — workspace/affect/episode_affect_tags.jsonl

The output is a small JSON document at
``workspace/companion/interest_profile.json``::

    {
      "generated_at": "2026-05-09T12:00:00+00:00",
      "lookback_days": 14,
      "topics": [
        {"name": "forest carbon",
         "score": 0.83,
         "last_seen": "2026-05-08T18:42Z",
         "sources": {"convs": 7, "emails": 2, "events": 1, ...}},
        ...
      ],
      "questions_open": [...]
    }

Read by:

  * ``daily_briefing`` — surface "topics you care about" section
  * ``personalized_signal`` (Phase B #5) — tone hint
  * ``grand_task`` synth — favor topics over generic ideation

Cadence: 12 h. Unlikely to drift faster than that and the underlying
sources are themselves bounded-rate. Master switch:
``LIFE_COMPANION_ENABLED`` (cadence-guarded; no separate kill switch).

Algorithm: minimal. Tokenize → drop stopwords → unigram + bigram
frequencies → recency-weight (exponential decay over 14 days) →
source-diversity bonus → top-N by score. Heavier NLP belongs in the
SubIA topic-introspector pipeline; this module is a fast aggregator.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "interest_model.json"
_PROFILE_PATH = Path("/app/workspace/companion/interest_profile.json")
_RUN_CADENCE_S = 12 * 3600
_LOOKBACK_DAYS = 14
_TOP_N_TOPICS = 30
_MIN_FREQ = 2          # term must appear ≥2× to count
_RECENCY_HALFLIFE_DAYS = 7

# Tiny stopword list — keeps the module self-contained. Spacy / NLTK
# would be overkill here; this list is good enough for the ~few KB of
# text per day this scans.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "in",
    "on", "at", "to", "for", "with", "from", "by", "as", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "us", "you", "your", "he", "she", "his",
    "her", "i", "me", "my", "mine", "what", "which", "who", "whom",
    "where", "when", "why", "how", "all", "each", "every", "any",
    "some", "no", "not", "so", "than", "too", "very", "can", "just",
    "now", "into", "out", "up", "down", "over", "under", "again",
    "further", "once", "more", "most", "other", "such", "only", "own",
    "same", "also", "use", "used", "using", "make", "made", "get",
    "got", "go", "going", "see", "seen", "know", "knew", "well",
    "way", "back", "still", "even", "much", "many", "like", "want",
    "need", "let", "lets", "say", "said", "yes", "ok", "okay",
    # AI/agent boilerplate
    "agent", "task", "user", "assistant", "system", "claude", "gpt",
    "sonnet", "opus", "haiku", "model", "llm",
})

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


# ── Source collectors ─────────────────────────────────────────────────────


def _conversations_text(lookback_days: int) -> Iterable[tuple[str, float]]:
    """Yield ``(text, age_days)`` from conversations.db over the lookback."""
    try:
        from app import conversation_store
        conn = conversation_store._get_conn()
    except Exception:
        return
    cutoff = datetime.now(timezone.utc).timestamp() - lookback_days * 86400
    try:
        rows = conn.execute(
            "SELECT content, created_at FROM messages "
            "WHERE created_at >= datetime(?, 'unixepoch') "
            "ORDER BY created_at DESC LIMIT 1000",
            (cutoff,),
        ).fetchall()
    except Exception:
        logger.debug("interest_model: conversations.db scan failed", exc_info=True)
        return
    now = datetime.now(timezone.utc).timestamp()
    for content, created_at_iso in rows:
        if not content:
            continue
        try:
            ts = datetime.fromisoformat(str(created_at_iso).replace(" ", "T")).timestamp()
        except Exception:
            ts = now
        age_days = max(0.0, (now - ts) / 86400)
        yield (str(content), age_days)


def _email_subject_text(lookback_days: int) -> Iterable[tuple[str, float]]:
    """Yield triaged-email subject lines from the email_monitor state."""
    state_path = Path("/app/workspace/life_companion/email_monitor.json")
    if not state_path.exists():
        return
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return
    items = (data.get("recent_top_subjects") or [])
    for entry in items:
        subject = entry.get("subject") if isinstance(entry, dict) else str(entry)
        if subject:
            yield (subject, 0.5)  # subjects don't carry timestamps; treat as fresh


def _calendar_titles_text(lookback_days: int) -> Iterable[tuple[str, float]]:
    """Yield calendar event titles around now."""
    try:
        from app.tools.gcal_tools import _list_events
    except Exception:
        return
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=lookback_days)).isoformat().replace("+00:00", "Z")
    time_max = (now + timedelta(days=14)).isoformat().replace("+00:00", "Z")
    try:
        events = _list_events(time_min=time_min, time_max=time_max, max_results=50)
    except Exception:
        return
    for ev in events or []:
        title = (ev.get("summary") or "").strip()
        if not title:
            continue
        # Approximate age — mid-window fresh.
        yield (title, 1.0)


def _feedback_events_text(lookback_days: int) -> Iterable[tuple[str, float]]:
    """Yield comments from POSITIVE companion FEEDBACK events only.

    Originally this yielded all FEEDBACK comments, but the consumer
    treats every snippet as a positive-weight contribution. A 👎
    comment on "forest carbon" was therefore upweighting "forest
    carbon" in the interest profile — wrong directionality. We now
    filter to ``polarity == "up"`` so only thumbs-up comments
    contribute to interest scoring. Negative feedback flows through
    ``feedback_weights`` (workspace selection) instead.
    """
    events_path = Path("/app/workspace/companion/events.jsonl")
    if not events_path.exists():
        return
    cutoff = time.time() - lookback_days * 86400
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
                if (payload.get("polarity") or "").lower() != "up":
                    continue
                ts = float(ev.get("ts", 0))
                if ts < cutoff:
                    continue
                comment = (payload.get("comment") or "").strip()
                if comment:
                    age_days = max(0.0, (time.time() - ts) / 86400)
                    yield (comment, age_days)
    except Exception:
        logger.debug("interest_model: events.jsonl scan failed", exc_info=True)
        return


def _affect_topics_text(lookback_days: int) -> Iterable[tuple[str, float]]:
    """Yield topic tags from the affect trace."""
    tags_path = Path("/app/workspace/affect/episode_affect_tags.jsonl")
    if not tags_path.exists():
        return
    cutoff = time.time() - lookback_days * 86400
    try:
        with tags_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = float(row.get("ts", 0))
                if ts < cutoff:
                    continue
                topic = (row.get("topic") or row.get("subject") or "").strip()
                if topic:
                    age_days = max(0.0, (time.time() - ts) / 86400)
                    yield (topic, age_days)
    except Exception:
        logger.debug("interest_model: affect-tags scan failed", exc_info=True)
        return


# ── Tokenizer + scorer ────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase + filter stopwords + drop short tokens."""
    return [
        tok for tok in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if tok not in _STOPWORDS and len(tok) >= 3
    ]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]


def _recency_weight(age_days: float, halflife: float = _RECENCY_HALFLIFE_DAYS) -> float:
    if halflife <= 0:
        return 1.0
    # Exponential decay: weight halves every halflife days.
    return float(0.5 ** (age_days / halflife))


def _score_terms(
    streams: dict[str, Iterable[tuple[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregated scores across streams.

    Returns {term: {score, last_seen_age, sources: {stream_name: count}}}.
    """
    scores: dict[str, dict[str, Any]] = {}

    for stream_name, items in streams.items():
        for text, age_days in items:
            tokens = _tokenize(text)
            grams = tokens + _bigrams(tokens)
            seen_in_doc = set()  # avoid double-counting within one snippet
            w = _recency_weight(age_days)
            for term in grams:
                if term in seen_in_doc:
                    continue
                seen_in_doc.add(term)
                row = scores.setdefault(term, {
                    "score": 0.0, "last_seen_age": age_days,
                    "sources": defaultdict(int),
                })
                row["score"] += w
                row["last_seen_age"] = min(row["last_seen_age"], age_days)
                row["sources"][stream_name] += 1
    return scores


def _diversity_bonus(sources: dict[str, int]) -> float:
    """Reward terms appearing across multiple streams (×1.0 → ×1.5)."""
    n = len(sources)
    if n <= 1:
        return 1.0
    return 1.0 + 0.1 * min(5, n - 1)  # cap at +50%


def compile_interest_profile(lookback_days: int = _LOOKBACK_DAYS) -> dict[str, Any]:
    """Run all collectors and write the profile JSON. Returns the dict."""
    streams = {
        "convs": _conversations_text(lookback_days),
        "emails": _email_subject_text(lookback_days),
        "events": _calendar_titles_text(lookback_days),
        "feedback": _feedback_events_text(lookback_days),
        "affect": _affect_topics_text(lookback_days),
    }
    raw_scores = _score_terms(streams)

    topics: list[dict[str, Any]] = []
    for term, row in raw_scores.items():
        sources = dict(row["sources"])
        total_count = sum(sources.values())
        if total_count < _MIN_FREQ:
            continue
        score = row["score"] * _diversity_bonus(sources)
        topics.append({
            "name": term,
            "score": round(score, 3),
            "last_seen_age_days": round(row["last_seen_age"], 1),
            "sources": sources,
        })
    topics.sort(key=lambda r: r["score"], reverse=True)
    topics = topics[:_TOP_N_TOPICS]

    profile = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "topics": topics,
    }

    try:
        _PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PROFILE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(_PROFILE_PATH)
    except Exception:
        logger.debug("interest_model: profile write failed", exc_info=True)

    return profile


def current_profile() -> dict[str, Any]:
    """Read the latest profile from disk. Returns empty if not generated yet."""
    if not _PROFILE_PATH.exists():
        return {"generated_at": "", "lookback_days": 0, "topics": []}
    try:
        return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"generated_at": "", "lookback_days": 0, "topics": []}


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally."""
    summary = {"ran": False, "topic_count": 0}
    if not background_enabled():
        return summary

    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return summary
    state["last_run_at"] = now

    profile = compile_interest_profile()
    summary["ran"] = True
    summary["topic_count"] = len(profile.get("topics", []))
    state["last_topic_count"] = summary["topic_count"]
    write_state_json(_STATE_FILE, state)

    audit_event(
        "interest_model_pass",
        topic_count=summary["topic_count"],
        lookback_days=profile.get("lookback_days", 0),
    )
    return summary
