"""Daily integrity check for the coding-session audit chain.

The chain in ``workspace/coding_sessions/audit.jsonl`` was hash-chained
on write (see ``app/coding_session/store.py:_append_audit``) but never
verified on read. This monitor runs the verifier daily and alerts on
breaks. Read-only — never modifies the chain. If the chain is broken,
the operator decides recovery (file integrity is a human call).

State at ``workspace/life_companion/audit_chain_check.json`` —
re-using the life-companion state dir to avoid scattering JSON files.
We track the last-known-clean hash so re-alerts only fire when the
state of the chain actually changes.
"""
from __future__ import annotations

import logging
import time

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    send_signal_alert,
    write_state_json,
)

logger = logging.getLogger(__name__)

_STATE_FILE = "audit_chain_check.json"
_CHECK_CADENCE_S = 23 * 3600   # ~daily — slightly under 24h so it slides
_ALERT_COOLDOWN_S = 24 * 3600  # at most one Signal alert per chain-state per day


def run() -> None:
    """One pass — cadence-checked. Safe to call from a chatty idle scheduler."""
    if not background_enabled():
        return

    state = read_state_json(_STATE_FILE, {
        "last_run_at": 0.0,
        "last_known_clean_hash": "",
        "last_alert_at": 0.0,
        "last_first_break_line": None,
    })
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _CHECK_CADENCE_S:
        return

    state["last_run_at"] = now

    try:
        from app.coding_session.audit_verify import chain_summary
        summary = chain_summary()
    except Exception:
        logger.debug("audit_chain_check: verifier raised", exc_info=True)
        write_state_json(_STATE_FILE, state)
        return

    audit_event(
        "coding_session_audit_chain_check",
        ok=summary.get("ok"),
        lines=summary.get("lines"),
        broken_count=summary.get("broken_count"),
        first_break_line=summary.get("first_break_line"),
    )

    if summary.get("ok"):
        # Track the latest clean hash so we have a recovery checkpoint.
        state["last_known_clean_hash"] = summary.get("last_entry_hash", "")
        state["last_first_break_line"] = None
        write_state_json(_STATE_FILE, state)
        return

    # Broken chain — alert if (a) cooldown expired OR (b) the break
    # location moved (different first_break_line indicates fresh damage).
    cool = (now - float(state.get("last_alert_at", 0))) >= _ALERT_COOLDOWN_S
    new_break = (
        summary.get("first_break_line") != state.get("last_first_break_line")
    )
    if cool or new_break:
        state["last_alert_at"] = now
        state["last_first_break_line"] = summary.get("first_break_line")
        write_state_json(_STATE_FILE, state)
        send_signal_alert(
            "🛑 Self-heal: coding-session audit chain integrity check failed.\n\n"
            f"  • file: `{summary.get('path')}`\n"
            f"  • lines: {summary.get('lines')}\n"
            f"  • broken entries: {summary.get('broken_count')}\n"
            f"  • first break at line: {summary.get('first_break_line')} "
            f"({summary.get('first_break_reason', 'unknown')})\n"
            f"  • last known clean hash: "
            f"`{state.get('last_known_clean_hash') or '(none)'}`\n\n"
            "Inspect with `python -c \"from app.coding_session.audit_verify "
            "import verify_chain; print(verify_chain())\"`. Recovery is an "
            "operator call — the verifier never modifies the chain.",
            tag="audit_chain_check",
        )
        return

    write_state_json(_STATE_FILE, state)
