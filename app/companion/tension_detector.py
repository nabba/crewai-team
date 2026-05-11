"""Tension autonomous detector — scans recent user messages for
open-question markers and files tensions.

PROGRAM §41.4 (2026-05-11) — Q4.1 Item 1.

The Q4 Phase A `tensions.detect_from_text` helper was shipped but
nothing invoked it. Tensions could only be created via /tensions add
or the REST endpoint — the operator had to manually file every one.

This module closes that loop: an idle job periodically scans recent
user messages in `conversation_store`, runs the regex detector
against them, and files tensions when patterns hit. The 30-OPEN cap
in `tensions.create_tension` self-throttles over-detection.

Cadence: 6h (LIGHT). Scans the trailing 24h window. State file
records the last-scanned ts so we don't re-process old messages on
every pass.

Detection scope: ONLY user-role messages (role='user'). Assistant
replies are excluded — the system shouldn't track its OWN
"wonderings" as user tensions (those go through SubIA wonder, which
is deliberately separate).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


_STATE_FILE = "tension_detector.json"
_RUN_CADENCE_S = 6 * 3600       # 6 hours
_SCAN_WINDOW_S = 24 * 3600      # look back 24h on each pass


def _read_recent_user_messages(since_ts: float, limit: int = 200) -> list[dict]:
    """Pull recent user-role messages from conversation_store newer
    than ``since_ts``. Returns ``[{content, ts}, ...]`` newest-first.
    Failure-isolated; returns [] on any error."""
    try:
        from app import conversation_store
        conn = conversation_store._get_conn()
    except Exception:
        logger.debug("tension_detector: conversation_store unavailable", exc_info=True)
        return []
    # The `ts` column is TEXT (ISO-8601 in current writers, occasionally
    # a stringified float in legacy rows). We compare client-side
    # rather than relying on SQLite date arithmetic.
    try:
        rows = conn.execute(
            """SELECT content, ts FROM messages
                WHERE role = 'user'
             ORDER BY id DESC
                LIMIT ?""",
            (int(limit),),
        ).fetchall()
    except Exception:
        logger.debug("tension_detector: query failed", exc_info=True)
        return []
    out: list[dict] = []
    for content, ts_raw in rows:
        if not content:
            continue
        # Normalise ts.
        try:
            if isinstance(ts_raw, (int, float)):
                ts_val = float(ts_raw)
            else:
                ts_val = datetime.fromisoformat(
                    str(ts_raw).replace("Z", "+00:00")
                ).timestamp()
        except Exception:
            continue
        if ts_val < since_ts:
            continue
        out.append({"content": str(content), "ts": ts_val})
    return out


def run() -> dict[str, Any]:
    """One pass — cadence-guarded internally. Read-only on
    conversation_store; writes only to the tensions store. Failure-
    isolated."""
    try:
        from app.healing.handlers._common import (
            read_state_json, write_state_json,
        )
    except Exception:
        logger.debug(
            "tension_detector: healing handlers unavailable", exc_info=True,
        )
        return {"ok": False, "ran": False, "reason": "no state I/O"}

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "last_scanned_ts": 0.0,
        "lifetime_detected": 0,
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return {"ok": True, "ran": False, "reason": "cadence guard"}

    # Cover at least the cadence window so we don't miss anything
    # between passes; bounded by _SCAN_WINDOW_S to avoid re-scanning
    # ancient data after a long outage.
    since_ts = max(
        now - _SCAN_WINDOW_S,
        float(state.get("last_scanned_ts", 0.0)),
    )

    state["last_run_at"] = now

    try:
        from app.companion.tensions import detect_from_text
    except Exception:
        write_state_json(_STATE_FILE, state)
        return {"ok": False, "ran": True, "reason": "tensions module unavailable"}

    messages = _read_recent_user_messages(since_ts=since_ts, limit=200)
    if not messages:
        # Bump last_scanned_ts forward — empty window is fine.
        state["last_scanned_ts"] = now
        write_state_json(_STATE_FILE, state)
        return {"ok": True, "ran": True, "messages_scanned": 0, "detected": 0}

    detected_count = 0
    newest_ts = since_ts
    for msg in messages:
        ts = float(msg.get("ts") or 0.0)
        if ts > newest_ts:
            newest_ts = ts
        text = (msg.get("content") or "").strip()
        if len(text) < 30:
            # Very short messages are unlikely to contain question
            # markers worth tracking ("ok", "yes", "what?").
            continue
        try:
            new_tensions = detect_from_text(
                text, source_kind="conversation",
                source_ref=f"conv:{int(ts)}",
            )
        except Exception:
            logger.debug(
                "tension_detector: detect_from_text raised", exc_info=True,
            )
            continue
        detected_count += len(new_tensions)
        if detected_count >= 5:
            # Hard cap per pass — even if regex hits a lot, we don't
            # create more than 5 tensions in one cycle. The 30-OPEN
            # cap in tensions.create_tension is the absolute backstop;
            # this per-pass cap is operator-friendly throttling.
            break

    state["last_scanned_ts"] = max(newest_ts, now)
    state["lifetime_detected"] = int(state.get("lifetime_detected") or 0) + detected_count
    write_state_json(_STATE_FILE, state)

    return {
        "ok": True,
        "ran": True,
        "messages_scanned": len(messages),
        "detected": detected_count,
    }
