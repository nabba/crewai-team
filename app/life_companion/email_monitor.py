"""Email monitor — proactive triage of unread inbox.

Wraps the existing scoring heuristic in ``app/tools/email_importance.py``
and the Gmail listing helper in ``app/tools/gmail_tools.py``. Every
~10 min (cadence enforced inside ``run()``), fetches up to 25 unread
messages, scores each, and surfaces the top-3 above an urgency
threshold via Signal — but only the FIRST time each message is seen,
so a sustained spike of unread doesn't repeat alerts.

State lives in ``workspace/life_companion/email_monitor.json``:

    {
      "alerted_ids": [<msg_id>, ...],   # capped at 500, FIFO
      "last_run_at": <unix>,
      "last_top": [{...}],              # last surfacing payload, debug-only
    }

Scoring is deterministic and cheap (no LLM); the score is the same one
the existing on-demand "rank my emails" tool uses, so output is
explainable and consistent.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
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

_STATE_FILE = "email_monitor.json"
_DEFAULT_CHECK_INTERVAL_S = 10 * 60  # 10 min cadence
_DEFAULT_FETCH_LIMIT = 25
_DEFAULT_TOP_N = 3
_DEFAULT_URGENCY_THRESHOLD = 1.0
_ALERTED_IDS_CAP = 500


def _check_interval_s() -> int:
    import os
    try:
        return int(os.getenv("LIFE_COMPANION_EMAIL_CHECK_MIN", "10")) * 60
    except ValueError:
        return _DEFAULT_CHECK_INTERVAL_S


def _urgency_threshold() -> float:
    import os
    try:
        return float(os.getenv("LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD", "1.0"))
    except ValueError:
        return _DEFAULT_URGENCY_THRESHOLD


def _fetch_unread() -> list[dict]:
    """Return up to N unread inbox stubs. Empty list on any failure."""
    try:
        from app.tools.gmail_tools import _list_recent
    except Exception:
        logger.debug("email_monitor: gmail tools import failed", exc_info=True)
        return []
    try:
        return _list_recent(
            limit=_DEFAULT_FETCH_LIMIT, query="in:inbox is:unread",
        ) or []
    except Exception:
        logger.debug("email_monitor: _list_recent raised", exc_info=True)
        return []


def _build_headers(stub: dict) -> "EmailHeaders | None":  # type: ignore[name-defined]
    """Convert a Gmail message stub into an ``EmailHeaders`` for scoring."""
    try:
        from app.tools.email_importance import EmailHeaders
    except Exception:
        return None

    date_str = (stub.get("date") or "").strip()
    parsed_dt: datetime | None = None
    if date_str:
        try:
            parsed_dt = parsedate_to_datetime(date_str)
            if parsed_dt and parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        except Exception:
            parsed_dt = None

    label_ids = stub.get("label_ids") or []
    is_unread = "UNREAD" in label_ids

    return EmailHeaders(
        from_=stub.get("from", ""),
        to="",  # gmail list-format doesn't include To/Cc; full read would
        cc="",
        subject=stub.get("subject", ""),
        list_unsubscribe=None,
        list_id=None,
        auto_submitted=None,
        precedence=None,
        in_reply_to=None,
        references=None,
        date=parsed_dt,
        unread=is_unread,
    )


def _score_one(stub: dict, *, user_addr: str, important_senders: tuple[str, ...]) -> dict:
    """Return ``{"id", "from", "subject", "date", "score", "reasons"}``."""
    headers = _build_headers(stub)
    score = 0.0
    reasons: list[str] = []
    if headers is not None:
        try:
            from app.tools.email_importance import score_email
            result = score_email(
                headers,
                user_address=user_addr,
                important_senders=important_senders,
            )
            score = float(result.score)
            reasons = list(result.reasons)
        except Exception:
            logger.debug("email_monitor: score_email raised", exc_info=True)

    return {
        "id": stub.get("id", ""),
        "from": stub.get("from", ""),
        "subject": stub.get("subject", ""),
        "date": stub.get("date", ""),
        "snippet": (stub.get("snippet") or "")[:160],
        "score": score,
        "reasons": reasons,
    }


def _format_alert(top: list[dict]) -> str:
    """Single Signal message with up to N urgent unread items."""
    n = len(top)
    head = f"📬 Email triage — {n} urgent unread:\n"
    bullets = []
    for item in top:
        sender = item["from"][:60] or "(unknown)"
        subj = (item["subject"] or "(no subject)")[:80]
        bullets.append(
            f"  • {sender}\n    {subj}\n    score={item['score']:.1f}"
        )
    return head + "\n".join(bullets)


def _important_senders() -> tuple[str, ...]:
    """Parse ``EMAIL_IMPORTANT_SENDERS`` env var (comma-separated)."""
    import os
    raw = os.getenv("EMAIL_IMPORTANT_SENDERS", "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def run() -> None:
    """One pass — cadence-checked. Safe to call from an idle job that ticks
    more often than the configured interval; the cadence guard ensures we
    only do real work every ``LIFE_COMPANION_EMAIL_CHECK_MIN`` minutes.
    """
    if not feature_enabled("email"):
        return
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {"alerted_ids": [], "last_run_at": 0.0})
    now = time.time()
    last_run = float(state.get("last_run_at", 0))
    if now - last_run < _check_interval_s():
        return  # not time yet

    state["last_run_at"] = now

    try:
        stubs = _fetch_unread()
    except Exception:
        write_state_json(_STATE_FILE, state)
        return

    if not stubs:
        audit_event("email_monitor_pass", checked=0, alerted=0)
        write_state_json(_STATE_FILE, state)
        return

    user_addr = user_email_address()
    senders = _important_senders()
    scored = [_score_one(s, user_addr=user_addr, important_senders=senders) for s in stubs]

    # Sort by score descending; take top N above threshold; exclude already-alerted.
    threshold = _urgency_threshold()
    alerted: list[str] = list(state.get("alerted_ids", []))
    alerted_set = set(alerted)
    candidates = sorted(
        (it for it in scored if it["score"] >= threshold and it["id"] and it["id"] not in alerted_set),
        key=lambda x: x["score"],
        reverse=True,
    )[:_DEFAULT_TOP_N]

    audit_event(
        "email_monitor_pass",
        checked=len(scored),
        candidates=len(candidates),
        threshold=threshold,
    )

    if not candidates:
        write_state_json(_STATE_FILE, state)
        return

    body = _format_alert(candidates)
    sent = send_signal_alert(body, tag="email_monitor")
    if sent:
        # Persist so we don't re-alert the same messages.
        for item in candidates:
            alerted.append(item["id"])
        # FIFO cap so the file doesn't grow unbounded.
        if len(alerted) > _ALERTED_IDS_CAP:
            alerted = alerted[-_ALERTED_IDS_CAP:]
        state["alerted_ids"] = alerted
        state["last_top"] = candidates
        audit_event(
            "email_monitor_alert",
            n=len(candidates),
            top_score=candidates[0]["score"],
        )

    write_state_json(_STATE_FILE, state)
