"""PROGRAM §41.4 — Q4.1 Companion-depth wiring tests.

Targets the three gaps the Q4 ship-out left open:
  * Tension autonomous detection from conversation_store (idle job)
  * Queued-notification digest in daily briefing
  * arbitrate= kwarg wired into notify_on_complete + concrete callers
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import pytest


def _load_isolated(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────
#   Q4.1 #1 — Tension autonomous detection
# ─────────────────────────────────────────────────────────────────────────


def test_tension_detector_module_exists():
    """Source-level: the module exists with the expected entry point."""
    src = Path("app/companion/tension_detector.py").read_text()
    assert "def run()" in src
    assert "_RUN_CADENCE_S" in src
    assert "from app.companion.tensions import detect_from_text" in src
    # Detection scope: user-role only (not assistant)
    assert "role = 'user'" in src


def test_tension_detector_registered_in_loop():
    """Source-level: tension_detector job is registered in companion.loop."""
    src = Path("app/companion/loop.py").read_text()
    assert "tension_detector" in src
    assert "tension-detector" in src


def test_tension_detector_per_pass_cap_present():
    """Source-level: the per-pass detection cap (5) is enforced."""
    src = Path("app/companion/tension_detector.py").read_text()
    assert "detected_count >= 5" in src


# ─────────────────────────────────────────────────────────────────────────
#   Q4.1 #2 — Queued-notification digest
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fatigue():
    return _load_isolated("fatigue_q41", "app/notify/fatigue.py")


def test_fatigue_record_event_retains_body_only_for_queue(fatigue, tmp_path):
    p = tmp_path / "fatigue.json"
    fatigue.record_event(
        tag="t", topic="x", decision="send_now",
        title="immediate", body="should not be retained",
        path=p,
    )
    fatigue.record_event(
        tag="t", topic="x", decision="queue_for_digest",
        title="deferred", body="should be retained",
        path=p,
    )
    fatigue.record_event(
        tag="t", topic="x", decision="suppress_low_value",
        title="dropped", body="should not be retained",
        path=p,
    )
    events = fatigue.list_recent(window_hours=1.0, path=p)
    by_decision = {e["decision"]: e for e in events}
    # Only queue_for_digest carries the body.
    assert by_decision["queue_for_digest"]["body"] == "should be retained"
    assert by_decision["queue_for_digest"]["title"] == "deferred"
    assert by_decision["send_now"]["body"] is None
    assert by_decision["suppress_low_value"]["body"] is None


def test_pending_digest_entries_filters_correctly(fatigue, tmp_path):
    p = tmp_path / "fatigue.json"
    fatigue.record_event(
        tag="t", topic="a", decision="queue_for_digest",
        title="t1", body="b1", path=p,
    )
    fatigue.record_event(
        tag="t", topic="b", decision="queue_for_digest",
        title="t2", body="b2", path=p,
    )
    fatigue.record_event(
        tag="t", topic="c", decision="send_now",
        title="sent", body="should not appear in digest",
        path=p,
    )
    pending = fatigue.pending_digest_entries(window_hours=1.0, path=p)
    assert len(pending) == 2
    titles = sorted(p["title"] for p in pending)
    assert titles == ["t1", "t2"]


def test_mark_digest_consumed_excludes_from_pending(fatigue, tmp_path):
    p = tmp_path / "fatigue.json"
    fatigue.record_event(
        tag="t", topic="a", decision="queue_for_digest",
        title="t1", body="b1", path=p,
    )
    fatigue.record_event(
        tag="t", topic="b", decision="queue_for_digest",
        title="t2", body="b2", path=p,
    )
    pending_before = fatigue.pending_digest_entries(path=p)
    assert len(pending_before) == 2
    ts_set = {float(e["ts"]) for e in pending_before}
    marked = fatigue.mark_digest_consumed(ts_set, path=p)
    assert marked == 2
    # Re-marking is idempotent
    re_marked = fatigue.mark_digest_consumed(ts_set, path=p)
    assert re_marked == 0
    pending_after = fatigue.pending_digest_entries(path=p)
    assert len(pending_after) == 0


def test_briefing_composer_returns_tuple_with_consume_ts():
    """Source-level: the composers were refactored to return
    (body, consume_ts_list) so run() only consumes on Signal-send
    success."""
    src = Path("app/life_companion/daily_briefing.py").read_text()
    assert "def _compose_morning() -> tuple[str, list[float]]" in src
    assert "def _compose_evening() -> tuple[str, list[float]]" in src
    assert "def _compose_weekly() -> tuple[str, list[float]]" in src
    # The run() unpacks the tuple.
    assert "body, consume_ts = composer()" in src
    # And only marks on successful send.
    assert "_mark_digest_consumed(consume_ts)" in src


def test_briefing_includes_queued_section_when_pending():
    """Source-level: morning + evening composers include the queued
    notifications section when there's anything pending."""
    src = Path("app/life_companion/daily_briefing.py").read_text()
    assert "📨 Queued notifications (deferred by arbiter)" in src


# ─────────────────────────────────────────────────────────────────────────
#   Q4.1 #3 — arbitrate= wiring into concrete callers
# ─────────────────────────────────────────────────────────────────────────


def test_notify_on_complete_accepts_arbitrate_kwargs():
    """Source-level: the decorator's signature was extended with
    arbitrate / topic / critical_on_failure kwargs."""
    src = Path("app/notify/api.py").read_text()
    assert "arbitrate: bool = False" in src
    assert "topic: Optional[str] = None" in src
    assert "critical_on_failure: bool = False" in src


def test_emit_completion_forwards_arbitration():
    """Source-level: _emit_completion propagates arbitrate/topic/critical
    into the notify() call. Critical promotes to failure path so
    success-pings arbitrate but failures of critical_on_failure jobs
    always reach Signal."""
    src = Path("app/notify/api.py").read_text()
    fn_start = src.find("def _emit_completion(")
    body = src[fn_start:fn_start + 3000]
    assert "arbitrate=arbitrate" in body
    assert "topic=topic" in body
    assert "critical=critical_flag" in body
    # On failure, critical_flag = critical_on_failure
    assert "critical_flag = critical_on_failure" in body


def test_schedule_manager_opts_in_to_arbitrate():
    """Source-level: user scheduled tasks now opt in to arbitration."""
    src = Path("app/tools/schedule_manager_tools.py").read_text()
    # Use a generous window — the decorator + comments span multiple
    # lines and the matching close-paren isn't trivially findable.
    deco_start = src.find("@notify_on_complete")
    assert deco_start > 0
    deco_body = src[deco_start:deco_start + 800]
    assert "arbitrate=True" in deco_body
    assert 'topic=f"schedule:{name}"' in deco_body
    assert "critical_on_failure=True" in deco_body


def test_inbox_scheduler_arbitrates_successes_critical_on_failures():
    """Source-level: inbox notify call arbitrates only when no failures."""
    src = Path("app/inbox/scheduler.py").read_text()
    assert "has_failures = bool(result.failed)" in src
    assert "arbitrate=not has_failures" in src
    assert "critical=has_failures" in src


# ─────────────────────────────────────────────────────────────────────────
#   End-to-end: arbiter records body for queue decisions
# ─────────────────────────────────────────────────────────────────────────


def test_arbiter_record_helper_forwards_title_body():
    """Source-level: _record forwards title/body to record_event so
    the queue_for_digest decisions are body-retained."""
    src = Path("app/notify/arbiter.py").read_text()
    # _record signature accepts title + body
    helper_start = src.find("def _record(")
    helper_body = src[helper_start:helper_start + 1500]
    assert "title: str | None = None" in helper_body
    assert "body: str | None = None" in helper_body
    assert "title=title" in helper_body
    assert "body=body" in helper_body
