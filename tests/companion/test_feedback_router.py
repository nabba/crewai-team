"""Tests for ``app.companion.feedback_router`` (Phase B #3, 2026-05-09)."""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from app.companion import feedback_router, notify_meta

    monkeypatch.setattr(feedback_router, "_STATE_PATH", tmp_path / "fr_state.json")
    monkeypatch.setattr(notify_meta, "_META_PATH", tmp_path / "notify_meta.jsonl")

    skill_calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "app.skills.registry.record_run_result",
        lambda name, *, success: skill_calls.append((name, success)),
    )
    recipe_calls: list[dict] = []
    monkeypatch.setattr(
        "app.self_improvement.meta_agent.recorder.record_outcome",
        lambda **kw: recipe_calls.append(kw),
    )
    companion_calls: list[dict] = []
    monkeypatch.setattr(
        "app.companion.feedback.record",
        lambda **kw: companion_calls.append(kw),
    )

    yield tmp_path, skill_calls, recipe_calls, companion_calls


def test_disabled_returns_early(monkeypatch, isolated):
    monkeypatch.setenv("FEEDBACK_ROUTER_ENABLED", "0")
    from app.companion import feedback_router
    summary = feedback_router.run()
    assert summary["ran"] is False


def test_no_pg_no_events(monkeypatch, isolated):
    """When mem0_postgres_url is unset (laptop dev), the router should
    no-op cleanly without raising."""
    from app.companion import feedback_router

    class _FakeSettings:
        mem0_postgres_url = None
    monkeypatch.setattr("app.config.get_settings", lambda: _FakeSettings())

    summary = feedback_router.run()
    assert summary["ran"] is True
    assert summary["events_seen"] == 0


def test_dispatches_skill_sink(isolated, monkeypatch):
    """A 👍 reaction with skill_id metadata bumps the skill counter."""
    tmp_path, skill_calls, recipe_calls, companion_calls = isolated
    from app.companion import feedback_router, notify_meta

    import time as _t
    fresh_ts = int(_t.time() * 1000)
    notify_meta.record(fresh_ts, {"skill_id": "weather_check"})

    fake_event = {
        "id": "evt-1",
        "feedback_type": "explicit_positive",
        "raw_signal": "👍",
        "original_response": "rain tomorrow",
        "crew_used": "researcher",
    }
    monkeypatch.setattr(feedback_router, "_fetch_new_events",
                        lambda since_id, limit=200: [fake_event])
    monkeypatch.setattr(feedback_router, "_resolve_send_ts",
                        lambda ev: fresh_ts)

    summary = feedback_router.run()
    assert summary["events_dispatched"] == 1
    assert skill_calls == [("weather_check", True)]
    assert recipe_calls == []


def test_dispatches_recipe_sink(isolated, monkeypatch):
    tmp_path, skill_calls, recipe_calls, companion_calls = isolated
    from app.companion import feedback_router, notify_meta

    import time as _t
    fresh_ts = int(_t.time() * 1000)
    notify_meta.record(fresh_ts, {
        "recipe_id": "r-uuid", "task_id": "t-1", "crew_name": "researcher",
    })

    fake_event = {
        "id": "evt-2",
        "feedback_type": "explicit_negative",
        "raw_signal": "👎",
        "original_response": "weather report",
        "crew_used": "researcher",
    }
    monkeypatch.setattr(feedback_router, "_fetch_new_events",
                        lambda since_id, limit=200: [fake_event])
    monkeypatch.setattr(feedback_router, "_resolve_send_ts",
                        lambda ev: fresh_ts)

    summary = feedback_router.run()
    assert summary["events_dispatched"] == 1
    assert len(recipe_calls) == 1
    assert recipe_calls[0]["recipe_id"] == "r-uuid"
    assert recipe_calls[0]["user_feedback"] == "👎"
    assert recipe_calls[0]["success"] is False


def test_dispatches_companion_sink(isolated, monkeypatch):
    tmp_path, skill_calls, recipe_calls, companion_calls = isolated
    from app.companion import feedback_router, notify_meta

    import time as _t
    fresh_ts = int(_t.time() * 1000)
    notify_meta.record(fresh_ts, {
        "idea_id": "idea-1", "workspace_id": "ws-1",
    })

    fake_event = {
        "id": "evt-3",
        "feedback_type": "explicit_positive",
        "raw_signal": "👍",
        "original_response": "idea synthesized",
        "crew_used": "writer",
    }
    monkeypatch.setattr(feedback_router, "_fetch_new_events",
                        lambda since_id, limit=200: [fake_event])
    monkeypatch.setattr(feedback_router, "_resolve_send_ts",
                        lambda ev: fresh_ts)

    summary = feedback_router.run()
    assert summary["events_dispatched"] == 1
    assert len(companion_calls) == 1
    assert companion_calls[0]["idea_id"] == "idea-1"


def test_skips_event_with_no_metadata(isolated, monkeypatch):
    tmp_path, skill_calls, recipe_calls, companion_calls = isolated
    from app.companion import feedback_router

    fake_event = {
        "id": "evt-4",
        "feedback_type": "explicit_positive",
        "raw_signal": "👍",
        "original_response": "x",
        "crew_used": "",
    }
    monkeypatch.setattr(feedback_router, "_fetch_new_events",
                        lambda since_id, limit=200: [fake_event])
    monkeypatch.setattr(feedback_router, "_resolve_send_ts",
                        lambda ev: 1715200000000)

    summary = feedback_router.run()
    assert summary["events_dispatched"] == 0
    assert skill_calls == []


def test_cursor_advances(isolated, monkeypatch):
    tmp_path, skill_calls, recipe_calls, companion_calls = isolated
    from app.companion import feedback_router, notify_meta

    import time as _t
    fresh_ts = int(_t.time() * 1000)
    notify_meta.record(fresh_ts, {"skill_id": "s1"})
    fake_event = {
        "id": "evt-cursor",
        "feedback_type": "explicit_positive",
        "raw_signal": "👍",
        "original_response": "x",
        "crew_used": "",
    }
    monkeypatch.setattr(feedback_router, "_fetch_new_events",
                        lambda since_id, limit=200: [fake_event])
    monkeypatch.setattr(feedback_router, "_resolve_send_ts",
                        lambda ev: fresh_ts)

    feedback_router.run()
    state = feedback_router._read_state()
    assert state["last_event_id_seen"] == "evt-cursor"
