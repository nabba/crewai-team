"""Phase F targeted tests — pin behavior for the 11 audit fixes."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest


# ── F1: events module's iter_all_workspaces helper ───────────────────────


def test_events_iter_all_workspaces(tmp_path, monkeypatch):
    from app.companion import events as _events
    monkeypatch.setattr(_events, "_EVENTS_DIR", tmp_path)
    # Two workspaces, each with one FEEDBACK event.
    _events.append(_events.Event(
        workspace_id="ws-A", idea_id="i1",
        type=_events.EventType.FEEDBACK,
        payload={"polarity": "up", "comment": "great"},
    ))
    _events.append(_events.Event(
        workspace_id="ws-B", idea_id="i2",
        type=_events.EventType.FEEDBACK,
        payload={"polarity": "down", "comment": "no"},
    ))
    _events.append(_events.Event(
        workspace_id="ws-A", idea_id="i3",
        type=_events.EventType.SURFACED, payload={},
    ))
    out = _events.iter_all_workspaces(type_filter=_events.EventType.FEEDBACK)
    assert len(out) == 2
    assert {e.workspace_id for e in out} == {"ws-A", "ws-B"}
    # Order is newest-first.
    assert out[0].ts >= out[1].ts


def test_events_iter_filters_old_events(tmp_path, monkeypatch):
    from app.companion import events as _events
    monkeypatch.setattr(_events, "_EVENTS_DIR", tmp_path)
    _events.append(_events.Event(
        workspace_id="ws", idea_id="i1",
        type=_events.EventType.FEEDBACK,
        ts=1000.0, payload={"polarity": "up"},
    ))
    _events.append(_events.Event(
        workspace_id="ws", idea_id="i2",
        type=_events.EventType.FEEDBACK,
        ts=time.time(), payload={"polarity": "up"},
    ))
    out = _events.iter_all_workspaces(
        type_filter=_events.EventType.FEEDBACK,
        since_ts=time.time() - 86400,
    )
    assert len(out) == 1
    assert out[0].idea_id == "i2"


# ── F2: adapter_performance recipe-ledger uses extra_tool_names ──────────


def test_adapter_performance_recipe_check_uses_extra_tool_names(monkeypatch):
    """The fixed code must look at AgentRecipe.extra_tool_names and
    join via recipe_id (NOT RecipeOutcome.tool_names which doesn't
    exist)."""
    from app.training import adapter_performance

    class _R:
        def __init__(self, rid, tools):
            self.id = rid
            self.extra_tool_names = tools

    class _O:
        def __init__(self, recipe_id, success, recorded_at):
            self.recipe_id = recipe_id
            self.success = success
            self.recorded_at = recorded_at

    now = datetime.now(timezone.utc).isoformat()
    monkeypatch.setattr(
        "app.self_improvement.meta_agent.store.list_recipes",
        lambda **kw: [
            _R("r-match", ["adapter:weak_specialist", "web_search"]),
            _R("r-other", ["other_tool"]),
        ],
    )
    def fake_list_outcomes(recipe_id=None, **kw):
        if recipe_id == "r-match":
            return [
                _O("r-match", True, now),
                _O("r-match", False, now),
                _O("r-match", False, now),
            ]
        return []
    monkeypatch.setattr(
        "app.self_improvement.meta_agent.store.list_outcomes",
        fake_list_outcomes,
    )

    rate = adapter_performance._recipe_winrate_for_adapter(
        "weak_specialist", days=7,
    )
    assert rate is not None
    assert abs(rate - 1/3) < 1e-3


def test_adapter_performance_returns_none_when_no_recipe_match(monkeypatch):
    from app.training import adapter_performance
    monkeypatch.setattr(
        "app.self_improvement.meta_agent.store.list_recipes",
        lambda **kw: [],
    )
    monkeypatch.setattr(
        "app.self_improvement.meta_agent.store.list_outcomes",
        lambda *a, **k: [],
    )
    assert adapter_performance._recipe_winrate_for_adapter("x") is None


# ── F3: notify_on_complete forwards metadata ─────────────────────────────


def test_notify_on_complete_forwards_metadata(monkeypatch):
    from app.notify import notify_on_complete
    captured: dict = {}

    def fake_notify(title, body="", *, url="", tag="", signal=True,
                    web_push=True, metadata=None):
        captured["metadata"] = metadata
        return {"signal": False, "web_push_count": 0, "signal_ts": None}

    monkeypatch.setattr("app.notify.api.notify", fake_notify)

    @notify_on_complete(label="X", metadata={"job_id": "j-1"})
    def _job():
        return "ok"

    _job()
    assert captured["metadata"] == {"job_id": "j-1"}


# ── F4: signal_context maps SubIA circadian modes ────────────────────────


def test_signal_context_maps_circadian_modes(monkeypatch):
    from app.personality import signal_context

    monkeypatch.setattr(
        "app.subia.temporal.circadian.current_circadian_mode",
        lambda: "deep_work_hours",
    )
    assert signal_context._time_of_day() == "evening"

    monkeypatch.setattr(
        "app.subia.temporal.circadian.current_circadian_mode",
        lambda: "consolidation_hours",
    )
    assert signal_context._time_of_day() == "night"


def test_signal_context_falls_back_when_circadian_unmapped(monkeypatch):
    """Unknown SubIA mode → falls through to hour-based heuristic."""
    from app.personality import signal_context
    monkeypatch.setattr(
        "app.subia.temporal.circadian.current_circadian_mode",
        lambda: "totally_unknown_mode",
    )
    out = signal_context._time_of_day()
    assert out in ("morning", "afternoon", "evening", "night")


# ── F5: change_request consults lessons KB ───────────────────────────────


def test_change_request_consults_lessons_kb(monkeypatch, tmp_path):
    """``create_request`` must annotate the CR with lesson matches."""
    import app.change_requests.lifecycle as _lc
    import app.change_requests.store as _store

    monkeypatch.setattr(_store, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(_store, "_INDEX", None, raising=False)
    monkeypatch.setattr(
        "app.companion.lessons_learned.check_against",
        lambda txt, **kw: [{
            "id": "abc12", "similarity": 0.66,
            "sample_reason": "frozen pending audit",
            "count": 4,
        }],
    )
    cr = _lc.create_request(
        requestor="coder", path="app/foo.py",
        new_content="x = 1\n", old_content="",
        reason="rewrite auth module use jwt",
    )
    assert "abc12" in cr.reason
    assert "0.66" in cr.reason
    assert "frozen pending audit" in cr.reason


# ── F6: daily_briefing surfaces top interests ────────────────────────────


def test_daily_briefing_weekly_includes_interests(monkeypatch):
    from app.life_companion import daily_briefing
    monkeypatch.setattr(
        "app.companion.interest_model.current_profile",
        lambda: {"topics": [
            {"name": "forest carbon", "score": 1.5},
            {"name": "kaicart", "score": 1.2},
            {"name": "plg", "score": 0.9},
        ]},
    )
    monkeypatch.setattr(daily_briefing, "_gather_calendar_24h", lambda: [])
    monkeypatch.setattr(daily_briefing, "_gather_open_tickets", lambda n: [])
    monkeypatch.setattr(daily_briefing, "_gather_companion_surfaced", lambda: [])
    body = daily_briefing._compose_weekly()
    assert "Topics you've cared about" in body
    assert "forest carbon" in body
    assert "kaicart" in body


def test_daily_briefing_morning_does_not_include_interests(monkeypatch):
    """Morning digest stays clean — interests only in weekly."""
    from app.life_companion import daily_briefing
    monkeypatch.setattr(daily_briefing, "_gather_calendar_24h", lambda: [])
    monkeypatch.setattr(daily_briefing, "_gather_top_emails", lambda n: [])
    monkeypatch.setattr(daily_briefing, "_gather_open_tickets", lambda n: [])
    body = daily_briefing._compose_morning()
    assert "Topics you've cared about" not in body


# ── F7: jsonl_retention util ─────────────────────────────────────────────


def test_jsonl_retention_caps_to_max(tmp_path):
    from app.utils.jsonl_retention import cap_jsonl
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    dropped = cap_jsonl(p, 30)
    assert dropped == 70
    lines = p.read_text().splitlines()
    assert len(lines) == 30
    assert lines[0] == "line 70"
    assert lines[-1] == "line 99"


def test_jsonl_retention_noop_when_under_cap(tmp_path):
    from app.utils.jsonl_retention import cap_jsonl
    p = tmp_path / "log.jsonl"
    p.write_text("a\nb\nc\n")
    dropped = cap_jsonl(p, 100)
    assert dropped == 0
    assert p.read_text() == "a\nb\nc\n"


def test_append_with_cap(tmp_path):
    from app.utils.jsonl_retention import append_with_cap
    p = tmp_path / "log.jsonl"
    for i in range(50):
        append_with_cap(p, f'{{"i": {i}}}', max_lines=20)
    lines = p.read_text().splitlines()
    assert len(lines) == 20
    assert json.loads(lines[0])["i"] == 30
    assert json.loads(lines[-1])["i"] == 49


# ── F8: feedback_router single JOIN ──────────────────────────────────────


def test_resolve_send_ts_reads_from_event_field():
    """After F8, _resolve_send_ts is a thin reader over the JOINed
    ``msg_timestamp`` field — no SQL inside."""
    from app.companion.feedback_router import _resolve_send_ts
    assert _resolve_send_ts({"msg_timestamp": 1715200000123}) == 1715200000123
    assert _resolve_send_ts({"msg_timestamp": None}) is None
    assert _resolve_send_ts({}) is None


def test_fetch_uses_left_join():
    import inspect
    from app.companion import feedback_router
    src = inspect.getsource(feedback_router._fetch_new_events)
    assert "LEFT JOIN" in src
    assert "feedback.response_metadata" in src
    assert "rm.msg_timestamp" in src


# ── F9: missing-heartbeat alerts ─────────────────────────────────────────


def test_missing_heartbeat_alerts_when_known_listener_absent(tmp_path, monkeypatch):
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(listener_heartbeats, "_HEARTBEAT_DIR", tmp_path / "hb")

    sent: list[str] = []
    monkeypatch.setattr(listener_heartbeat, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(listener_heartbeat, "audit_event", lambda *a, **k: None)
    # Touch only ONE listener — others in KNOWN_LISTENERS must alert as missing.
    listener_heartbeats.touch("firebase-mode-poll")

    listener_heartbeat.run()
    # Should alert "no heartbeat" for the 8 missing known listeners.
    assert any("NO\nheartbeat" in s or "produced NO" in s for s in sent)


def test_missing_heartbeat_skipped_when_subsystem_off(tmp_path, monkeypatch):
    """Empty heartbeats dir → no missing-listener alerts.

    Avoids spurious alerts when Firebase is intentionally disabled."""
    from app.healing import listener_heartbeats
    from app.healing.monitors import listener_heartbeat
    from app.healing.handlers import _common as _h_common

    monkeypatch.setattr(_h_common, "_STATE_DIR", tmp_path / "self_heal")
    monkeypatch.setattr(listener_heartbeats, "_HEARTBEAT_DIR", tmp_path / "hb")
    sent: list[str] = []
    monkeypatch.setattr(listener_heartbeat, "send_signal_alert",
                        lambda body, **kw: sent.append(body) or True)
    monkeypatch.setattr(listener_heartbeat, "audit_event", lambda *a, **k: None)
    monkeypatch.setattr(listener_heartbeat, "_LIVENESS_PROBES", [])

    listener_heartbeat.run()
    # No missing-listener alerts, only the workspace-fallback "∞" alert.
    assert not any("produced NO" in s for s in sent)


# ── F10: /commitment slash command ───────────────────────────────────────


def test_commitment_help_lists_subcommands():
    from app.agents.commander.commands import _handle_commitment_command
    out = _handle_commitment_command("/commitment")
    assert out is not None
    assert "fulfilled" in out
    assert "broken" in out
    assert "deferred" in out
    assert "unmute" in out


def test_commitment_unknown_subcommand_returns_help():
    from app.agents.commander.commands import _handle_commitment_command
    out = _handle_commitment_command("/commitment whatever")
    assert "fulfilled" in out


# ── F11: paper_pipeline embedding-similarity ranking ─────────────────────


def test_paper_pipeline_embedding_relevance_against_profile():
    from app.episteme.paper_pipeline import _embedding_relevance
    from app.utils.hash_embedding import embed
    profile_emb = embed("forest carbon estonia")
    relevant = _embedding_relevance(
        {"title": "Carbon flux in Estonian forests",
         "abstract": "We study forest carbon"},
        profile_emb,
    )
    unrelated = _embedding_relevance(
        {"title": "Quaternion rotation in 3D graphics",
         "abstract": "Linear algebra trick for rotations"},
        profile_emb,
    )
    assert relevant > unrelated
    assert relevant >= 0.0
    assert unrelated >= 0.0


def test_paper_pipeline_no_profile_returns_zero():
    from app.episteme.paper_pipeline import _embedding_relevance
    rel = _embedding_relevance(
        {"title": "x", "abstract": "y"}, None,
    )
    assert rel == 0.0
