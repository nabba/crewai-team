"""HTTP API tests for the brainstorm router.

Drives the FastAPI router via TestClient. Auth dep is pass-through in dev
mode; we still pin a fake gateway secret via the v2 shim for safety.
"""

import os

import pytest

from tests._v2_shim import install_settings_shim

install_settings_shim()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINSTORM_DIR", str(tmp_path / "brainstorm"))
    monkeypatch.setenv("BRAINSTORM_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("BRAINSTORM_DISABLE_WRITER", "1")
    monkeypatch.setenv("BRAINSTORM_WEB_SENDER", "+19990001000")

    # Build a tiny FastAPI app that mounts only our router (avoids loading
    # the full main.py, which has many heavy imports).
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.brainstorm.api import router

    app = FastAPI()
    app.include_router(router)
    yield TestClient(app)


# ── Read endpoints ────────────────────────────────────────────────────────


def test_list_techniques(client):
    resp = client.get("/api/cp/brainstorm/techniques")
    assert resp.status_code == 200
    data = resp.json()
    names = {t["name"] for t in data}
    assert names == {
        "scamper",
        "six_hats",
        "how_might_we",
        "reverse",
        "crazy_8s",
        "rapid_ideation",
        "starbursting",
    }
    for t in data:
        assert t["title"]
        assert t["description"]
        assert t["total_steps"] > 0


def test_list_sessions_empty(client):
    resp = client.get("/api/cp/brainstorm/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_active_session_empty(client):
    resp = client.get("/api/cp/brainstorm/sessions/active")
    assert resp.status_code == 200
    assert resp.json() == {"session": None}


def test_get_session_404(client):
    resp = client.get("/api/cp/brainstorm/sessions/nope")
    assert resp.status_code == 404


# ── Lifecycle: solo mode ──────────────────────────────────────────────────


def test_start_solo_session(client):
    resp = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "improve onboarding"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["technique"] == "scamper"
    assert body["session"]["topic"] == "improve onboarding"
    assert body["session"]["mode"] == "solo"
    assert body["session"]["participants"] == []
    assert body["delivery"]["prompt"]
    assert body["delivery"]["seed"] == []
    assert body["delivery"]["react"] == []


def test_start_unknown_technique_returns_400(client):
    resp = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "not_a_thing", "topic": "x"},
    )
    assert resp.status_code == 400
    assert "Unknown technique" in resp.json()["detail"]


def test_respond_records_user_turn(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    resp = client.post(
        f"/api/cp/brainstorm/sessions/{sid}/respond",
        json={"message": "first answer"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["advanced"] is True
    assert body["session"]["step_index"] == 1
    # Transcript contains the user turn
    assert any(
        t.get("role") == "user" and t.get("content") == "first answer"
        for t in body["session"]["transcript"]
    )


def test_respond_to_wrong_session_returns_409(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    # Pause it — now no active session
    client.post(f"/api/cp/brainstorm/sessions/{sid}/pause")
    resp = client.post(
        f"/api/cp/brainstorm/sessions/{sid}/respond", json={"message": "x"}
    )
    assert resp.status_code == 409


def test_skip_advances_state(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    resp = client.post(f"/api/cp/brainstorm/sessions/{sid}/skip")
    assert resp.status_code == 200
    assert resp.json()["session"]["step_index"] == 1


def test_pause_then_resume(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    paused = client.post(f"/api/cp/brainstorm/sessions/{sid}/pause").json()
    assert paused["status"] == "paused"
    # Active is now empty
    assert client.get("/api/cp/brainstorm/sessions/active").json()["session"] is None
    resumed = client.post(f"/api/cp/brainstorm/sessions/{sid}/resume").json()
    assert resumed["session"]["session_id"] == sid
    assert resumed["session"]["status"] == "active"


def test_cancel_drops_session(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    cancelled = client.post(f"/api/cp/brainstorm/sessions/{sid}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert client.get("/api/cp/brainstorm/sessions/active").json()["session"] is None


def test_finish_generates_report(client):
    """Walk a 6-step reverse session to completion and finish it."""
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "reverse", "topic": "morning routine"},
    ).json()
    sid = start["session"]["session_id"]
    for i in range(6):
        client.post(
            f"/api/cp/brainstorm/sessions/{sid}/respond",
            json={"message": f"answer {i}"},
        )
    finished = client.post(
        f"/api/cp/brainstorm/sessions/{sid}/finish"
    ).json()
    assert finished["status"] == "complete"
    # With Writer disabled via env, the deterministic fallback runs.
    assert finished["final_report"] is not None
    assert "Brainstorming Report" in finished["final_report"]
    assert finished["final_report_path"]


def test_finish_without_report(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    finished = client.post(
        f"/api/cp/brainstorm/sessions/{sid}/finish?generate_report=false"
    ).json()
    assert finished["status"] == "complete"
    assert finished["final_report"] is None


def test_session_listing_after_activity(client):
    s1 = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic A"},
    ).json()
    sid = s1["session"]["session_id"]
    client.post(f"/api/cp/brainstorm/sessions/{sid}/pause")
    s2 = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "six_hats", "topic": "topic B"},
    ).json()
    sessions = client.get("/api/cp/brainstorm/sessions").json()
    assert len(sessions) == 2
    techs = {s["technique"] for s in sessions}
    assert techs == {"scamper", "six_hats"}


def test_get_session_returns_serialized_shape(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    s = client.get(f"/api/cp/brainstorm/sessions/{sid}").json()
    assert s["session_id"] == sid
    assert s["total_steps"] == 7  # SCAMPER has 7 steps
    assert s["technique_title"] == "SCAMPER"
    assert s["mode"] == "solo"


def test_delete_session(client):
    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic"},
    ).json()
    sid = start["session"]["session_id"]
    client.post(f"/api/cp/brainstorm/sessions/{sid}/pause")
    deleted = client.delete(f"/api/cp/brainstorm/sessions/{sid}").json()
    assert deleted["deleted"] == sid
    assert client.get(f"/api/cp/brainstorm/sessions/{sid}").status_code == 404


# ── Team-mode endpoint with mocked gatherers ─────────────────────────────


def test_start_team_mode_with_mocked_gatherers(client, monkeypatch):
    from app.brainstorm.multi_agent import AgentResponse
    import app.brainstorm.facilitator as fac

    def fake_seed(*, roster, **kwargs):
        return [
            AgentResponse(role=r, text=f"seed-{r}", duration_s=0.0)
            for r in roster
        ]

    def fake_react(*, roster, **kwargs):
        return [
            AgentResponse(role=r, text=f"react-{r}", duration_s=0.0)
            for r in roster
        ]

    monkeypatch.setattr(fac, "_default_gather_seed", fake_seed)
    monkeypatch.setattr(fac, "_default_gather_react", fake_react)

    resp = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic", "with_agents": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session"]["mode"] == "team"
    assert len(body["session"]["participants"]) == 3
    assert len(body["delivery"]["seed"]) == 3
    assert body["delivery"]["seed"][0]["role"] == "researcher"
    assert "seed-researcher" in body["delivery"]["seed"][0]["text"]


def test_team_mode_respond_returns_react_and_next_seed(client, monkeypatch):
    from app.brainstorm.multi_agent import AgentResponse
    import app.brainstorm.facilitator as fac

    def fake_gather(*, roster, **kwargs):
        return [AgentResponse(role=r, text="ok", duration_s=0.0) for r in roster]

    monkeypatch.setattr(fac, "_default_gather_seed", fake_gather)
    monkeypatch.setattr(fac, "_default_gather_react", fake_gather)

    start = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic", "with_agents": 2},
    ).json()
    sid = start["session"]["session_id"]
    resp = client.post(
        f"/api/cp/brainstorm/sessions/{sid}/respond",
        json={"message": "answer"},
    ).json()
    assert resp["advanced"] is True
    assert len(resp["delivery"]["react"]) == 2
    assert len(resp["delivery"]["seed"]) == 2  # next step's seed


def test_with_agents_clamped_at_max(client):
    """Pydantic should reject with_agents > 4 (the roster max)."""
    resp = client.post(
        "/api/cp/brainstorm/sessions",
        json={"technique": "scamper", "topic": "topic", "with_agents": 99},
    )
    assert resp.status_code == 422


# ── Sender isolation ──────────────────────────────────────────────────────


def test_sender_query_param_isolates_sessions(client):
    client.post(
        "/api/cp/brainstorm/sessions?sender=alice",
        json={"technique": "scamper", "topic": "alice topic"},
    )
    client.post(
        "/api/cp/brainstorm/sessions?sender=bob",
        json={"technique": "six_hats", "topic": "bob topic"},
    )
    a = client.get("/api/cp/brainstorm/sessions?sender=alice").json()
    b = client.get("/api/cp/brainstorm/sessions?sender=bob").json()
    assert len(a) == 1 and a[0]["technique"] == "scamper"
    assert len(b) == 1 and b[0]["technique"] == "six_hats"


def test_include_other_senders_returns_all(client):
    client.post(
        "/api/cp/brainstorm/sessions?sender=alice",
        json={"technique": "scamper", "topic": "topic"},
    )
    client.post(
        "/api/cp/brainstorm/sessions?sender=bob",
        json={"technique": "six_hats", "topic": "topic"},
    )
    out = client.get(
        "/api/cp/brainstorm/sessions?include_other_senders=true"
    ).json()
    assert len(out) == 2
