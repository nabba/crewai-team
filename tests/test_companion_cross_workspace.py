"""Tests for app.companion.cross_workspace — sanitiser + relevance gates."""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import cross_workspace as _xw
from app.companion import events as _ev
from app.companion import idea_store as _is


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ideas = tmp_path / "ideas"
    events = tmp_path / "events"
    monkeypatch.setattr(_is, "_IDEAS_DIR", ideas)
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    monkeypatch.setattr(_ev, "_EVENTS_DIR", events)
    return tmp_path


def _persist_idea(workspace_id, *, transferability=0.9, text=None,
                   idea_id=None):
    rec = _is.IdeaRecord(
        workspace_id=workspace_id,
        text=text or "Feedback loops with delayed rewards self-stabilise.",
        transferability=transferability,
        panel_score=0.8,
    )
    if idea_id:
        rec.idea_id = idea_id
    _is.persist(rec)
    return rec


def _stub_config(*, enabled=True, transferability_t=0.7, seed=""):
    from app.companion.config import CompanionConfig
    return CompanionConfig(
        enabled=enabled,
        transferability_threshold=transferability_t,
        seed_prompt=seed,
    ).clamp()


# ── propagate_eligible ─────────────────────────────────────────────────────

def test_propagate_no_eligible_ideas(tmp_dirs):
    _persist_idea("ws-a", transferability=0.4)
    rows = [{"id": "ws-a", "config_json": {"companion": {"enabled": True}}}]
    with patch("app.companion.config.load",
               lambda ws: _stub_config()), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows):
        n = _xw.propagate_eligible("ws-a")
    assert n == 0


def test_propagate_disabled_workspace(tmp_dirs):
    _persist_idea("ws-a")
    with patch("app.companion.config.load",
               lambda ws: _stub_config(enabled=False)):
        assert _xw.propagate_eligible("ws-a") == 0


def test_propagate_emits_inbox_when_all_gates_pass(tmp_dirs):
    src = _persist_idea("ws-a", text="Abstract structural feedback loop.")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "biological feedback systems"}}},
        {"id": "ws-b", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "ecological feedback dynamics"}}},
    ]
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="biological feedback systems")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.85):
        n = _xw.propagate_eligible("ws-a")

    assert n == 1
    inbox_events = [
        e for e in _ev.read_all("ws-b")
        if e.type == _ev.EventType.CROSS_WORKSPACE_INBOX
    ]
    assert len(inbox_events) == 1
    payload = inbox_events[0].payload
    assert payload["source_workspace_id"] == "ws-a"
    assert payload["source_idea_id"] == src.idea_id


def test_propagate_skips_when_sanitiser_blocks(tmp_dirs):
    _persist_idea("ws-a")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "x"}}},
        {"id": "ws-b", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "y"}}},
    ]
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="x")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: False), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.95):
        n = _xw.propagate_eligible("ws-a")
    assert n == 0


def test_propagate_skips_when_relevance_below_threshold(tmp_dirs):
    _persist_idea("ws-a")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "forests"}}},
        {"id": "ws-b", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "e-commerce SaaS"}}},
    ]
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="forests")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.3):
        n = _xw.propagate_eligible("ws-a")
    assert n == 0


def test_propagate_skips_target_with_no_seed(tmp_dirs):
    _persist_idea("ws-a")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "x"}}},
        {"id": "ws-b", "config_json": {"companion": {"enabled": True}}},
    ]
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="x")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.99):
        n = _xw.propagate_eligible("ws-a")
    assert n == 0


def test_propagate_does_not_repropose_same_kernel(tmp_dirs):
    src = _persist_idea("ws-a")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "x"}}},
        {"id": "ws-b", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "y"}}},
    ]
    # First run.
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="x")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.9):
        n1 = _xw.propagate_eligible("ws-a")
    # Second run — same idea, same target.
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="x")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.9):
        n2 = _xw.propagate_eligible("ws-a")
    assert n1 == 1
    assert n2 == 0


def test_propagate_caps_at_max_per_run(tmp_dirs):
    for i in range(20):
        _persist_idea("ws-a", text=f"distinct idea {i} body content body")
    rows = [
        {"id": "ws-a", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "x"}}},
        {"id": "ws-b", "config_json": {"companion": {
            "enabled": True, "seed_prompt": "y"}}},
    ]
    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="x")), \
         patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace._passes_sanitiser",
               lambda t: True), \
         patch("app.companion.cross_workspace._invoke_relevance",
               lambda k, s: 0.9):
        n = _xw.propagate_eligible("ws-a")
    assert n <= _xw.MAX_PROPAGATIONS_PER_RUN


# ── inbox / accept / dismiss ───────────────────────────────────────────────

def test_inbox_lists_undecided_only(tmp_dirs):
    _ev.append(_ev.Event(workspace_id="ws-b", idea_id="xw_a",
                          type=_ev.EventType.CROSS_WORKSPACE_INBOX,
                          payload={"kernel_id": "xw_a", "text": "alpha",
                                   "source_workspace_id": "ws-a",
                                   "source_idea_id": "idea_1",
                                   "relevance_score": 0.8}))
    _ev.append(_ev.Event(workspace_id="ws-b", idea_id="xw_b",
                          type=_ev.EventType.CROSS_WORKSPACE_INBOX,
                          payload={"kernel_id": "xw_b", "text": "beta",
                                   "source_workspace_id": "ws-a",
                                   "source_idea_id": "idea_2",
                                   "relevance_score": 0.9}))
    # Decide one of them.
    _ev.append(_ev.Event(workspace_id="ws-b", idea_id="xw_a",
                          type=_ev.EventType.CROSS_WORKSPACE_DISMISSED,
                          payload={"kernel_id": "xw_a"}))

    pending = _xw.inbox("ws-b")
    kernel_ids = [p.kernel_id for p in pending]
    assert "xw_b" in kernel_ids
    assert "xw_a" not in kernel_ids


def test_accept_records_event(tmp_dirs):
    _ev.append(_ev.Event(workspace_id="ws-b", idea_id="xw_a",
                          type=_ev.EventType.CROSS_WORKSPACE_INBOX,
                          payload={"kernel_id": "xw_a", "text": "x",
                                   "source_workspace_id": "ws-a",
                                   "source_idea_id": "idea_1",
                                   "relevance_score": 0.85}))
    assert _xw.accept("ws-b", "xw_a") is True
    accepted = [e for e in _ev.read_all("ws-b")
                if e.type == _ev.EventType.CROSS_WORKSPACE_ACCEPTED]
    assert len(accepted) == 1
    assert accepted[0].payload["source_workspace_id"] == "ws-a"


def test_accept_unknown_kernel_returns_false(tmp_dirs):
    assert _xw.accept("ws-b", "nonexistent") is False


def test_dismiss_records_reason(tmp_dirs):
    _ev.append(_ev.Event(workspace_id="ws-b", idea_id="xw_a",
                          type=_ev.EventType.CROSS_WORKSPACE_INBOX,
                          payload={"kernel_id": "xw_a", "text": "x",
                                   "source_workspace_id": "ws-a",
                                   "source_idea_id": "idea_1",
                                   "relevance_score": 0.85}))
    assert _xw.dismiss("ws-b", "xw_a", reason="not on topic") is True
    dismissed = [e for e in _ev.read_all("ws-b")
                 if e.type == _ev.EventType.CROSS_WORKSPACE_DISMISSED]
    assert len(dismissed) == 1
    assert dismissed[0].payload["reason"] == "not on topic"


# ── relevance + cosine ─────────────────────────────────────────────────────

def test_cosine_zero_vectors():
    assert _xw._cosine([], []) == 0.0
    assert _xw._cosine([0, 0, 0], [1, 2, 3]) == 0.0


def test_cosine_identical_vectors():
    v = [0.5, 0.3, 0.8]
    assert _xw._cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_maps_to_half():
    """Pure orthogonal → similarity normalised to 0.5 in this scheme."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _xw._cosine(a, b) == pytest.approx(0.5)


# ── Idle job + run-all ─────────────────────────────────────────────────────

def test_get_idle_jobs_returns_xworkspace_light():
    jobs = _xw.get_idle_jobs()
    assert len(jobs) == 1
    name, fn, weight = jobs[0]
    assert name == "companion-xworkspace"
    from app.idle_scheduler import JobWeight
    assert weight == JobWeight.LIGHT


def test_run_for_all_handles_listing_failure(tmp_dirs):
    def _broken():
        raise RuntimeError("DB down")

    with patch("app.companion.cross_workspace._list_projects", _broken):
        n = _xw.run_propagation_for_all_workspaces()
    assert n == 0


def test_run_for_all_skips_disabled(tmp_dirs):
    rows = [
        {"id": "ws-a", "config_json": {"companion": {"enabled": False}}},
        {"id": "ws-b", "config_json": {"companion": {"enabled": True}}},
    ]
    visited: list[str] = []

    def _fake_propagate(ws):
        visited.append(ws)
        return 0

    with patch("app.companion.cross_workspace._list_projects",
               lambda: rows), \
         patch("app.companion.cross_workspace.propagate_eligible",
               _fake_propagate):
        _xw.run_propagation_for_all_workspaces()
    assert visited == ["ws-b"]


def test_passes_sanitiser_fails_closed_on_import_error(tmp_dirs):
    """If transfer_memory.sanitizer can't be imported, propagation must
    NOT happen — the safety gate fails closed."""
    import sys
    # Force ImportError on the next sanitiser import inside _passes_sanitiser
    with patch.dict(sys.modules, {"app.transfer_memory.sanitizer": None}):
        # Setting to None makes import raise — sufficient for the closed-fail
        # path. The function returns False rather than letting kernels through.
        assert _xw._passes_sanitiser("anything") is False
