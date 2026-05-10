"""
forge_queue.py — recovery strategy: file the refusal as a skill gap
for offline forge processing.

Last-resort strategy. Doesn't recover the current request — instead
it ensures the same refusal will be cheaper next time, and returns
a structured diagnostic answer so the user gets something actionable
right now.

Per the 2026-04-28 design (decision 3 = ii):
  * Track refusal frequency per gap-key (task+category)
  * Same gap fires 3+ times in 7 days → auto-queue forge experiment
  * Below threshold → just log

User-driven sync trigger (decision 3 = iii) is implemented in
``app/recovery/loop.py`` — a Signal/dashboard command that bumps a
pending gap to immediate processing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.recovery.librarian import Alternative
from app.recovery.strategies import StrategyResult

logger = logging.getLogger(__name__)


# Where we record refusal frequency. JSON, atomic-rewrite. Cheap +
# survives restarts.
_FREQUENCY_PATH = Path("/app/workspace/recovery/refusal_frequency.json")
# Where the offline forge picks up new skill ideas.
_LEARNING_QUEUE = Path("/app/workspace/skills/learning_queue.md")
# Threshold: same gap N times in WINDOW days → queue.
_AUTO_FORGE_THRESHOLD = 3
_AUTO_FORGE_WINDOW_DAYS = 7


def _gap_key(task: str, category: str) -> str:
    """Stable identifier for "this kind of refusal".

    Strips dates, numbers, and the user's specific entities so that
    "find Sales Lead at Montonio" and "find Sales Lead at Paysera"
    map to the same gap key. Coarse but adequate for frequency
    tracking — false collapses are fine (they just speed up the
    threshold trigger).
    """
    text = (task or "").lower()
    # Strip personal names, company names (rough — keep verbs + objects)
    text = re.sub(r"\b\d+\b", "", text)
    text = re.sub(r"['\"<>@]", "", text)
    text = re.sub(r"\s+", " ", text).strip()[:120]
    return f"{category}::{text}"


def _load_frequency() -> dict:
    if not _FREQUENCY_PATH.exists():
        return {}
    try:
        return json.loads(_FREQUENCY_PATH.read_text())
    except Exception:
        return {}


def _save_frequency(data: dict) -> None:
    try:
        _FREQUENCY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FREQUENCY_PATH.write_text(json.dumps(data, indent=2, default=str))
    except Exception as exc:
        logger.debug("forge_queue: frequency save failed: %s", exc)


def _record_and_count(gap_key: str) -> tuple[int, list[str]]:
    """Append timestamp for ``gap_key``; return (count_in_window, all_iso_ts)."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=_AUTO_FORGE_WINDOW_DAYS)
    data = _load_frequency()
    entry = data.setdefault(gap_key, {"timestamps": []})
    timestamps: list[str] = entry["timestamps"]
    # Filter old + add new
    fresh = [ts for ts in timestamps if ts > cutoff.isoformat()]
    fresh.append(now.isoformat())
    entry["timestamps"] = fresh
    entry["last_seen"] = now.isoformat()
    entry["count_in_window"] = len(fresh)
    data[gap_key] = entry
    _save_frequency(data)
    return len(fresh), fresh


def _queue_for_forge(gap_key: str, task: str, category: str, count: int) -> None:
    """Append to the learning queue. The forge picks up entries on
    its next idle cycle."""
    try:
        _LEARNING_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        entry = (
            f"\n## refusal-recovery: {gap_key[:60]}\n"
            f"* added: {ts}\n"
            f"* category: {category}\n"
            f"* refusal count in 7d: {count}\n"
            f"* example task: {task[:300]}\n"
            f"* suggested skill: a tool that addresses '{category}' "
            f"failures for prompts of this shape. Investigate which "
            f"existing tool/adapter could be wired in.\n"
        )
        with open(_LEARNING_QUEUE, "a") as f:
            f.write(entry)
        logger.info(
            "forge_queue: added '%s' to learning queue (count=%d in %dd)",
            gap_key[:60], count, _AUTO_FORGE_WINDOW_DAYS,
        )
    except Exception as exc:
        logger.warning("forge_queue: write failed: %s", exc)


def _diagnostic_text(task: str, category: str, count: int, queued: bool) -> str:
    """Compose the user-facing diagnostic answer."""
    when_note = ""
    if queued:
        when_note = (
            f"\n\nThis is the {count}{_ord(count)} time I've hit this gap "
            f"in the last {_AUTO_FORGE_WINDOW_DAYS} days, so I've queued a "
            f"skill-forge experiment to close it. Check the Skills tab "
            f"in ~6h."
        )
    elif count > 1:
        when_note = (
            f"\n\nThis is the {count}{_ord(count)} time I've hit this gap "
            f"in the last {_AUTO_FORGE_WINDOW_DAYS} days. {_AUTO_FORGE_THRESHOLD - count} "
            f"more and I'll auto-queue a forge experiment."
        )
    actionable = _category_action(category)
    return (
        f"I tried to answer your request but couldn't fully complete it.\n\n"
        f"**What went wrong:** {category.replace('_', ' ')}.\n\n"
        f"**To make this work next time:**\n{actionable}"
        f"{when_note}"
    )


def _ord(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _category_action(category: str) -> str:
    """Per-category actionable hint for the diagnostic answer."""
    actions = {
        "missing_tool": (
            "* The crew that handled this request doesn't have the right "
            "tool. If you have an API key for the underlying service "
            "(Apollo, Proxycurl, etc.), set it in `.env` and restart.\n"
            "* Or rephrase the request so it routes to a crew that does "
            "have the tool — e.g., `pim` for email/calendar, `coding` "
            "for code execution.\n"
        ),
        "auth": (
            "* A required API key is missing. Set it in `.env` and "
            "restart the gateway (`docker compose up -d gateway`).\n"
        ),
        "execution": (
            "* No code-execution sandbox is wired up. The coding crew "
            "wrote a script but couldn't run it. Tell me what output "
            "you needed and I'll try a different path.\n"
        ),
        "data_unavailable": (
            "* The data sources I have access to don't have what you "
            "asked for. If you can point me at a specific URL or "
            "give me an attachment, I can use that instead.\n"
        ),
        "generic": (
            "* The agent gave up without a clear reason. Try rephrasing, "
            "or ask me to escalate to a stronger model with `force "
            "this` after the request.\n"
        ),
    }
    return actions.get(category, actions["generic"])


def execute(task: str, alt: Alternative, ctx: dict) -> StrategyResult:
    """Record + maybe-queue + return diagnostic.

    Always returns ``success=True`` because we always produce SOMETHING
    better than a bare refusal — either the diagnostic text or
    (eventually) the auto-forged skill's answer.
    """
    category = ctx.get("refusal_category", "generic")
    gap_key = _gap_key(task, category)

    try:
        count, _ = _record_and_count(gap_key)
    except Exception as exc:
        logger.debug("forge_queue: frequency tracking failed: %s", exc)
        count = 1

    queued = False
    if count >= _AUTO_FORGE_THRESHOLD:
        try:
            _queue_for_forge(gap_key, task, category, count)
            queued = True
        except Exception:
            pass

    # Surface the refusal as a LearningGap so the capability_gap_analyzer
    # can cluster recurring refusals into architecture-request drafts.
    # Failure-isolated: an emitter outage must NEVER mask the diagnostic.
    try:
        from app.self_improvement.gap_detector import emit_recovery_refusal
        emit_recovery_refusal(
            task=task,
            category=category,
            gap_key=gap_key,
            attempts=count,
            queued_for_forge=queued,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("forge_queue: emit_recovery_refusal failed: %s", exc)

    return StrategyResult(
        success=True,
        text=_diagnostic_text(task, category, count, queued),
        note=None,                  # diagnostic is the answer; no extra note
        route_changed=False,
    )
