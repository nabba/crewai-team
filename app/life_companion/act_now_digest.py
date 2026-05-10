"""Act-now email digest — LLM-graded "what needs your attention right now".

Sibling of ``email_monitor`` but with a fundamentally different shape:

  * **email_monitor** runs every ~10 min, uses the heuristic
    ``email_importance.score_email`` (no LLM), and surfaces up to
    3 unread emails above a low-confidence threshold.  Its job is
    "real-time noise filter that flags the obvious".

  * **act_now_digest** (this module) runs every 3 hours between
    07:00–22:00 local, looks back 48 h, applies an LLM content
    analysis to the unread inbox, and surfaces up to 7 emails the
    user must ACT on now (with a one-line reason + suggested
    action + Gmail link).  Its job is "thoughtful synthesis with
    actionable next steps".

The two jobs co-exist on purpose — the operator gets immediate
heuristic alerts AND a thrice-daily curated digest.  Neither
de-dups against the other; an email can legitimately appear in
both if it's both real-time-flag-worthy AND an act-now item.

Cadence + business-hours gating
-------------------------------

Fires at the six fixed slots ``07:00 / 10:00 / 13:00 / 16:00 /
19:00 / 22:00`` (local clock), with a ±15 min tolerance so the
idle scheduler doesn't have to land on the minute.  State key is
``YYYY-MM-DD-HH`` (slot hour) so we never re-fire the same slot.
Outside the 07–22 window: silent skip — operators don't want
notifications in the middle of the night.

LLM contract
------------

The model receives a structured prompt with the candidate emails
(sender + subject + body excerpt + date) and is asked to return
JSON: ``{"ranked": [{email_id, why_now, suggested_action,
deadline_hint, rank}, …]}``.  The schema is parsed with
``safe_json_parse`` (markdown-fence tolerant) and validated; any
malformed response falls back to the heuristic scorer's top-K so
the digest still ships, just less informed.

Cost: 6 runs/day × ~30 candidate emails × ~500 token excerpts ≈
~$0.40/day on Sonnet.  Tunable via
``LIFE_COMPANION_ACT_NOW_MAX_CANDIDATES`` (default 30).

Master switches
---------------

  * ``LIFE_COMPANION_ENABLED`` (Life Companion umbrella, default on)
  * ``LIFE_COMPANION_ACT_NOW_DIGEST_ENABLED`` (this job, default on)
  * ``idle_scheduler.is_enabled()`` (global background-tasks)

State file: ``workspace/life_companion/act_now_digest.json``
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.life_companion._common import (
    audit_event,
    background_enabled,
    feature_enabled,
    read_state_json,
    send_signal_alert,
    user_email_address,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "act_now_digest.json"

# Local-clock hours when the digest may fire.  3-hour cadence from
# 7am to 10pm — six slots/day.  Outside this window: silent skip.
_FIRE_HOURS: tuple[int, ...] = (7, 10, 13, 16, 19, 22)
_TOLERANCE_MIN = 15

_DEFAULT_TOP_K = 7
_DEFAULT_LOOKBACK_HOURS = 48
_DEFAULT_MAX_CANDIDATES = 30
_DEFAULT_BODY_CHARS = 500

# Gmail labels that are NEVER act-now — pre-filter drops them
# before the LLM call to save tokens.  Keep CATEGORY_UPDATES
# (flight changes, banking, package tracking can all be act-now).
_BULK_LABELS_DROP = frozenset((
    "CATEGORY_PROMOTIONS",
    "CATEGORY_SOCIAL",
    "CATEGORY_FORUMS",
))


# ── Config ───────────────────────────────────────────────────────────


def _top_k() -> int:
    try:
        return int(os.getenv("LIFE_COMPANION_ACT_NOW_TOP_K", str(_DEFAULT_TOP_K)))
    except ValueError:
        return _DEFAULT_TOP_K


def _lookback_hours() -> int:
    try:
        return int(os.getenv(
            "LIFE_COMPANION_ACT_NOW_LOOKBACK_HOURS",
            str(_DEFAULT_LOOKBACK_HOURS),
        ))
    except ValueError:
        return _DEFAULT_LOOKBACK_HOURS


def _max_candidates() -> int:
    try:
        return int(os.getenv(
            "LIFE_COMPANION_ACT_NOW_MAX_CANDIDATES",
            str(_DEFAULT_MAX_CANDIDATES),
        ))
    except ValueError:
        return _DEFAULT_MAX_CANDIDATES


def _body_chars() -> int:
    try:
        return int(os.getenv(
            "LIFE_COMPANION_ACT_NOW_BODY_CHARS",
            str(_DEFAULT_BODY_CHARS),
        ))
    except ValueError:
        return _DEFAULT_BODY_CHARS


# ── Cadence ──────────────────────────────────────────────────────────


def _now_local() -> datetime:
    """Local-clock datetime; matches daily_briefing's convention.
    The system already runs in the operator's TZ."""
    return datetime.now()


def _slot_key_for(now: datetime) -> str | None:
    """Return ``YYYY-MM-DD-HH`` for the matching fire slot, or None
    when ``now`` is outside any slot's tolerance window."""
    for hour in _FIRE_HOURS:
        target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        delta = abs((now - target).total_seconds()) / 60.0
        if delta <= _TOLERANCE_MIN:
            return f"{now.strftime('%Y-%m-%d')}-{hour:02d}"
    return None


# ── Email gathering ──────────────────────────────────────────────────


@dataclass
class _Candidate:
    """One unread email under consideration."""
    id: str
    from_: str
    subject: str
    date: str
    snippet: str
    body: str
    label_ids: tuple[str, ...]


def _fetch_unread_with_bodies(hours: int, *, max_n: int) -> list[_Candidate]:
    """Fetch unread inbox emails newer than N hours, with full body.

    Empty list on any failure — the digest skips silently rather
    than alerting on a degraded state (the email_monitor's
    real-time path will surface anything truly urgent regardless).
    """
    try:
        from app.tools.gmail_tools import _list_recent, _read_one
    except Exception:
        logger.debug("act_now_digest: gmail tools import failed", exc_info=True)
        return []

    # Gmail's `newer_than:Xd` operator takes whole days.  Round up.
    days = max(1, (hours + 23) // 24)
    query = f"in:inbox is:unread newer_than:{days}d"

    try:
        stubs = _list_recent(limit=max_n, query=query) or []
    except Exception:
        logger.debug("act_now_digest: list failed", exc_info=True)
        return []

    out: list[_Candidate] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    for stub in stubs:
        # Stub-level filters (cheap, no extra API call):
        date_str = (stub.get("date") or "").strip()
        if date_str:
            try:
                d = parsedate_to_datetime(date_str)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                if d < cutoff:
                    continue
            except Exception:
                pass  # bad date — keep, the LLM can decide

        # Full read for body content. One API call per message.
        try:
            full = _read_one(stub.get("id", "")) or {}
        except Exception:
            full = {}
        body = (full.get("body") or "")[: _body_chars()]

        out.append(_Candidate(
            id=stub.get("id", ""),
            from_=stub.get("from", ""),
            subject=stub.get("subject", ""),
            date=date_str,
            snippet=(stub.get("snippet") or "")[:200],
            body=body,
            label_ids=tuple(stub.get("label_ids") or []),
        ))

    return out


def _pre_filter(candidates: list[_Candidate]) -> list[_Candidate]:
    """Drop obvious-bulk emails before the LLM call.

    We keep CATEGORY_UPDATES — flight changes, banking, package
    tracking, calendar invites can all be legitimately act-now.
    Promotions / Social / Forums never are.

    Cheap pre-filter saves LLM tokens on the bulk that
    ``email_importance`` would have scored deeply negative anyway.
    """
    kept = []
    for c in candidates:
        labels = set(c.label_ids)
        if labels & _BULK_LABELS_DROP:
            continue
        kept.append(c)
    return kept


# ── LLM ranking ──────────────────────────────────────────────────────


_LLM_PROMPT = """You are an inbox triage assistant. The user has been away from \
their inbox for up to 48 hours and needs to know which emails require ACTION right \
now — replies, decisions, deadlines, or time-sensitive responses.

Rank the top {top_k} emails the user should act on RIGHT NOW.

Mark act-now if the email has any of:
  - explicit deadline (today, this week, EOD, ASAP)
  - direct personal request or expected reply from a human
  - time-sensitive issue (flight change, payment failure, calendar conflict, security alert)
  - decision required (approval, sign-off, scheduling)
  - reply continuing a real human thread

NOT act-now (omit from ranking):
  - newsletters, marketing, promotions
  - automated notifications with no action required
  - receipts / statements / order confirmations (unless something is wrong)
  - read-only digests, summaries, weekly reports
  - social-network notifications

Return JSON ONLY (no preamble, no markdown fence):

{{
  "ranked": [
    {{
      "email_id": "<exact id from input>",
      "why_now": "<1 sentence, max 80 chars: why this needs attention now>",
      "suggested_action": "<1 sentence, max 100 chars: concrete next step>",
      "deadline_hint": "<deadline if mentioned in body, else empty string>",
      "rank": <1-{top_k}, 1 = most urgent>
    }}
  ]
}}

Rules:
  - Only include emails TRULY worth acting on. If fewer than {top_k} qualify, return fewer.
  - If NONE qualify, return {{"ranked": []}}.
  - email_id must EXACTLY match an input id.
  - Do not invent emails not in the input.

Inbox candidates ({n} unread, last 48h, after bulk-filter):

{emails_block}
"""


def _build_emails_block(candidates: list[_Candidate]) -> str:
    """Format candidates for the LLM prompt — one block per email."""
    parts = []
    for i, c in enumerate(candidates, 1):
        parts.append(
            f"--- email #{i} ---\n"
            f"id: {c.id}\n"
            f"from: {c.from_}\n"
            f"subject: {c.subject}\n"
            f"date: {c.date}\n"
            f"body:\n{c.body or c.snippet or '(empty)'}"
        )
    return "\n\n".join(parts)


def _get_ranking_llm():
    """Build the LLM used for content ranking. Uses the existing
    full-vetting LLM (Sonnet via litellm cascade) — same shape used
    by app.vetting for structured-output Claude.

    Lazy-built so test paths can monkey-patch without booting the
    full LLM stack.
    """
    from app.llm_factory import create_vetting_llm
    return create_vetting_llm()


def _rank_with_llm(
    candidates: list[_Candidate], *, top_k: int,
) -> list[dict] | None:
    """Call the LLM. Returns the parsed `ranked` list, or None on
    any failure — caller falls back to the heuristic ranker."""
    if not candidates:
        return []

    prompt = _LLM_PROMPT.format(
        top_k=top_k,
        n=len(candidates),
        emails_block=_build_emails_block(candidates),
    )

    try:
        llm = _get_ranking_llm()
    except Exception:
        logger.debug("act_now_digest: LLM build failed", exc_info=True)
        return None

    try:
        raw = llm.call([{"role": "user", "content": prompt}])
    except Exception:
        logger.warning("act_now_digest: LLM call failed", exc_info=True)
        return None

    try:
        from app.utils import safe_json_parse
        parsed, err = safe_json_parse(raw or "")
    except Exception:
        logger.debug("act_now_digest: safe_json_parse import failed", exc_info=True)
        return None

    if err or not isinstance(parsed, dict):
        logger.warning("act_now_digest: LLM returned non-JSON: %s", err or "(no error)")
        return None

    ranked = parsed.get("ranked")
    if not isinstance(ranked, list):
        logger.warning("act_now_digest: LLM 'ranked' field missing/non-list")
        return None

    # Validate each entry references a real candidate id.
    valid_ids = {c.id for c in candidates}
    out: list[dict] = []
    for entry in ranked[:top_k]:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("email_id", ""))
        if eid not in valid_ids:
            continue  # hallucinated / mismatched id
        out.append({
            "email_id": eid,
            "why_now": str(entry.get("why_now", ""))[:120],
            "suggested_action": str(entry.get("suggested_action", ""))[:140],
            "deadline_hint": str(entry.get("deadline_hint", ""))[:80],
            "rank": int(entry.get("rank", 0) or 0),
        })

    return out


def _heuristic_fallback(
    candidates: list[_Candidate], *, top_k: int,
) -> list[dict]:
    """LLM unavailable or failed — fall back to the existing
    heuristic scorer's top-K. Keeps the digest deliverable even
    on degraded paths."""
    try:
        from app.tools.email_importance import EmailHeaders, score_email
        user_addr = user_email_address()

        scored: list[tuple[float, _Candidate]] = []
        for c in candidates:
            d: datetime | None = None
            try:
                d = parsedate_to_datetime(c.date) if c.date else None
                if d and d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
            except Exception:
                pass
            h = EmailHeaders(
                from_=c.from_, to="", cc="", subject=c.subject,
                gmail_labels=c.label_ids, unread=True, date=d,
            )
            r = score_email(h, user_address=user_addr)
            scored.append((r.score, c))

        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for rank, (score, c) in enumerate(scored[:top_k], 1):
            out.append({
                "email_id": c.id,
                "why_now": f"heuristic score {score:.1f} (LLM unavailable)",
                "suggested_action": "open the message and decide",
                "deadline_hint": "",
                "rank": rank,
            })
        return out
    except Exception:
        logger.debug("act_now_digest: heuristic fallback failed", exc_info=True)
        return []


# ── Output ───────────────────────────────────────────────────────────


def _gmail_link(message_id: str) -> str:
    """Build the Gmail web URL for a given message id.

    Format used by Gmail's web UI: ``/mail/u/0/#inbox/<id>``.
    Works for any account (the ``u/0`` selects the first signed-
    in user — accurate for single-account operators)."""
    if not message_id:
        return ""
    return f"https://mail.google.com/mail/u/0/#inbox/{message_id}"


def _format_digest(
    ranked: list[dict],
    *,
    candidates_by_id: dict[str, _Candidate],
    total_unread: int,
    pre_filtered_dropped: int,
) -> str:
    """Single Signal message with the act-now items."""
    n = len(ranked)
    total = total_unread
    after_filter = total - pre_filtered_dropped
    head = (
        f"✉️ Top {n} act-now emails — last 48h "
        f"({total} unread → {after_filter} after bulk-filter):\n"
    )
    if n == 0:
        return (
            head
            + "\nNothing requires action right now. "
            "Marketing / notifications / informational only."
        )

    parts = [head]
    for entry in ranked:
        c = candidates_by_id.get(entry["email_id"])
        if c is None:
            continue
        sender = (c.from_ or "(unknown)")[:60]
        subject = (c.subject or "(no subject)")[:80]
        why = entry.get("why_now") or ""
        action = entry.get("suggested_action") or ""
        deadline = entry.get("deadline_hint") or ""
        link = _gmail_link(c.id)
        rank = entry.get("rank") or 0

        block = [
            f"\n{rank}. {sender}",
            f"   {subject}",
        ]
        if why:
            block.append(f"   why: {why}")
        if action:
            block.append(f"   action: {action}")
        if deadline:
            block.append(f"   deadline: {deadline}")
        if link:
            block.append(f"   📨 {link}")
        parts.append("\n".join(block))

    return "\n".join(parts)


# ── Main ────────────────────────────────────────────────────────────


def run() -> None:
    """One pass of the act-now digest. Cadence-checked.

    Safe to call from an idle scheduler that ticks more often than
    the configured slot interval; the slot-key dedup ensures we
    only do real work once per slot.
    """
    if not feature_enabled("act_now_digest"):
        return
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {
        "last_run_key": "", "last_run_at": 0.0,
    })

    now = _now_local()
    slot_key = _slot_key_for(now)
    if slot_key is None:
        return  # outside the 6 fire windows
    if state.get("last_run_key") == slot_key:
        return  # already ran this slot today

    # ── Gather + filter ──
    candidates_all = _fetch_unread_with_bodies(
        hours=_lookback_hours(), max_n=_max_candidates(),
    )
    candidates = _pre_filter(candidates_all)
    candidates_by_id = {c.id: c for c in candidates}

    audit_event(
        "act_now_digest_pass",
        slot=slot_key,
        total_unread=len(candidates_all),
        after_filter=len(candidates),
    )

    # ── Rank ──
    if not candidates:
        # Nothing left after bulk filter — skip silently.  Mark slot
        # so we don't re-attempt for this 3-hour window.
        state["last_run_key"] = slot_key
        state["last_run_at"] = time.time()
        write_state_json(_STATE_FILE, state)
        return

    ranked = _rank_with_llm(candidates, top_k=_top_k())
    used_fallback = False
    if ranked is None:
        ranked = _heuristic_fallback(candidates, top_k=_top_k())
        used_fallback = True

    if not ranked:
        # LLM said "nothing act-now" OR fallback also empty.
        # Don't send a noisy "no act-now" message — operators
        # don't want a Signal ping every 3 hours saying "all
        # clear."  Just mark the slot and move on.
        state["last_run_key"] = slot_key
        state["last_run_at"] = time.time()
        write_state_json(_STATE_FILE, state)
        return

    body = _format_digest(
        ranked,
        candidates_by_id=candidates_by_id,
        total_unread=len(candidates_all),
        pre_filtered_dropped=len(candidates_all) - len(candidates),
    )
    sent = send_signal_alert(body, tag="act_now_digest")

    state["last_run_key"] = slot_key
    state["last_run_at"] = time.time()
    if sent:
        state["last_top"] = ranked
        state["last_used_fallback"] = used_fallback
    write_state_json(_STATE_FILE, state)

    audit_event(
        "act_now_digest_alert",
        slot=slot_key,
        n=len(ranked),
        used_fallback=used_fallback,
    )
