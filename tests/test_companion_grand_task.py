"""Tests for app.companion.grand_task — 12 h cadence synthesis."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.companion import config as _config
from app.companion import events as _ev
from app.companion import grand_task as _gt
from app.companion import idea_store as _is


@pytest.fixture
def tmp_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ideas = tmp_path / "ideas"
    events = tmp_path / "events"
    monkeypatch.setattr(_is, "_IDEAS_DIR", ideas)
    monkeypatch.setattr(_is, "_index_chromadb", lambda r: None)
    monkeypatch.setattr(_ev, "_EVENTS_DIR", events)
    return tmp_path


def _persist_polished_ideas(workspace_id="ws-1", n=3):
    """Persist N CONVERGED ideas (count as polished for synthesis)."""
    out = []
    for i in range(n):
        rec = _is.IdeaRecord(
            workspace_id=workspace_id,
            text=f"Polished idea {i} body about forest dynamics",
            state=_is.IdeaState.CONVERGED,
            panel_score=0.7 + 0.05 * i,
            novelty=0.6 + 0.05 * i,
        )
        _is.persist(rec)
        out.append(rec)
    return out


def _stub_config(seed="forests", enabled=True):
    return _config.CompanionConfig(seed_prompt=seed, enabled=enabled).clamp()


# ── synthesize() ───────────────────────────────────────────────────────────

def test_synthesize_skips_when_workspace_unknown(tmp_dirs):
    with patch("app.companion.config.load", lambda ws: None):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_skips_when_disabled(tmp_dirs):
    with patch("app.companion.config.load",
               lambda ws: _stub_config(enabled=False)):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_skips_when_too_few_ideas(tmp_dirs):
    _persist_polished_ideas(n=2)  # below MIN_IDEAS_FOR_SYNTHESIS=3
    with patch("app.companion.config.load", lambda ws: _stub_config()):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_skips_when_too_recent(tmp_dirs):
    _persist_polished_ideas(n=4)
    # A proposal 1 hour ago.
    _ev.append(_ev.Event(
        workspace_id="ws-1", idea_id="gt_old",
        type=_ev.EventType.GRAND_TASK_PROPOSED,
        ts=time.time() - 3600,
        payload={"proposal_id": "gt_old", "text": "previous"},
    ))
    with patch("app.companion.config.load", lambda ws: _stub_config()):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_runs_after_cadence_window(tmp_dirs):
    _persist_polished_ideas(n=4)
    # Old proposal — outside cadence window.
    _ev.append(_ev.Event(
        workspace_id="ws-1", idea_id="gt_old",
        type=_ev.EventType.GRAND_TASK_PROPOSED,
        ts=time.time() - (_gt.CADENCE_S + 1000),
        payload={"proposal_id": "gt_old", "text": "old"},
    ))
    raw = ("GRAND_TASK: Map mycorrhizal cycles across boreal seasons.\n"
           "RATIONALE: All ideas pivot on coupled below-ground signalling.")
    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: raw):
        proposal = _gt.synthesize("ws-1")
    assert proposal is not None
    assert "mycorrhizal" in proposal.text.lower()
    assert "below-ground" in proposal.rationale.lower()


def test_synthesize_emits_proposed_event(tmp_dirs):
    _persist_polished_ideas(n=4)
    raw = "GRAND_TASK: A solid grand task.\nRATIONALE: makes sense."
    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: raw):
        proposal = _gt.synthesize("ws-1")

    events = [e for e in _ev.read_all("ws-1")
              if e.type == _ev.EventType.GRAND_TASK_PROPOSED]
    assert len(events) == 1
    payload = events[0].payload
    assert payload["proposal_id"] == proposal.proposal_id
    assert payload["text"] == "A solid grand task."
    assert payload["superseded_seed"] == "forests"


def test_synthesize_handles_llm_failure(tmp_dirs):
    _persist_polished_ideas(n=4)

    def _broken(p):
        raise RuntimeError("LLM down")

    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer", _broken):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_handles_unparseable_response(tmp_dirs):
    _persist_polished_ideas(n=4)
    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: "I refuse."):
        assert _gt.synthesize("ws-1") is None


def test_synthesize_passes_seed_into_prompt(tmp_dirs):
    _persist_polished_ideas(n=4)
    captured: list[str] = []

    def _capture(p):
        captured.append(p)
        return "GRAND_TASK: ok\nRATIONALE: yes"

    with patch("app.companion.config.load",
               lambda ws: _stub_config(seed="Estonian boreal forests")), \
         patch("app.companion.grand_task._invoke_synthesizer", _capture):
        _gt.synthesize("ws-1")

    assert "Estonian boreal forests" in captured[0]


# ── accept / reject ────────────────────────────────────────────────────────

def test_accept_rotates_seed_prompt(tmp_dirs):
    _persist_polished_ideas(n=4)
    raw = "GRAND_TASK: New grand goal.\nRATIONALE: aligns with ideas."
    saved_configs: list = []

    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: raw):
        proposal = _gt.synthesize("ws-1")

    def _save(ws, cfg):
        saved_configs.append((ws, cfg.seed_prompt))
        return True

    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.config.save", _save):
        ok = _gt.accept("ws-1", proposal.proposal_id)

    assert ok is True
    assert saved_configs == [("ws-1", "New grand goal.")]
    accepted_events = [e for e in _ev.read_for_idea(
        "ws-1", proposal.proposal_id)
        if e.type == _ev.EventType.GRAND_TASK_ACCEPTED]
    assert len(accepted_events) == 1


def test_accept_unknown_proposal_returns_false(tmp_dirs):
    assert _gt.accept("ws-1", "gt_does_not_exist") is False


def test_accept_save_failure_returns_false(tmp_dirs):
    _persist_polished_ideas(n=4)
    raw = "GRAND_TASK: ok.\nRATIONALE: r"
    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: raw):
        proposal = _gt.synthesize("ws-1")

    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.config.save", lambda ws, cfg: False):
        assert _gt.accept("ws-1", proposal.proposal_id) is False


def test_reject_records_event_with_reason(tmp_dirs):
    _persist_polished_ideas(n=4)
    raw = "GRAND_TASK: new goal.\nRATIONALE: r"
    with patch("app.companion.config.load", lambda ws: _stub_config()), \
         patch("app.companion.grand_task._invoke_synthesizer",
               lambda p: raw):
        proposal = _gt.synthesize("ws-1")

    ok = _gt.reject("ws-1", proposal.proposal_id,
                     reason="too far from seed")
    assert ok is True
    rejected = [e for e in _ev.read_for_idea("ws-1", proposal.proposal_id)
                if e.type == _ev.EventType.GRAND_TASK_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].payload["reason"] == "too far from seed"


def test_reject_unknown_proposal_returns_false(tmp_dirs):
    assert _gt.reject("ws-1", "gt_unknown") is False


# ── list_proposals / find_proposal ─────────────────────────────────────────

def test_list_proposals_returns_newest_first(tmp_dirs):
    for i in range(3):
        _ev.append(_ev.Event(
            workspace_id="ws-1", idea_id=f"gt_{i}",
            type=_ev.EventType.GRAND_TASK_PROPOSED,
            ts=float(i),
            payload={"proposal_id": f"gt_{i}", "text": f"grand {i}"},
        ))
    out = _gt.list_proposals("ws-1")
    assert [p.proposal_id for p in out] == ["gt_2", "gt_1", "gt_0"]


def test_list_proposals_respects_limit(tmp_dirs):
    for i in range(10):
        _ev.append(_ev.Event(
            workspace_id="ws-1", idea_id=f"gt_{i}",
            type=_ev.EventType.GRAND_TASK_PROPOSED,
            ts=float(i),
            payload={"proposal_id": f"gt_{i}", "text": "x"},
        ))
    out = _gt.list_proposals("ws-1", limit=3)
    assert len(out) == 3


def test_find_proposal_locates_by_id(tmp_dirs):
    _ev.append(_ev.Event(
        workspace_id="ws-1", idea_id="gt_target",
        type=_ev.EventType.GRAND_TASK_PROPOSED,
        ts=time.time(),
        payload={"proposal_id": "gt_target", "text": "found me"},
    ))
    found = _gt.find_proposal("ws-1", "gt_target")
    assert found is not None
    assert found.text == "found me"


# ── Idle job entry ─────────────────────────────────────────────────────────

def test_get_idle_jobs_returns_grand_task_medium():
    jobs = _gt.get_idle_jobs()
    assert len(jobs) == 1
    name, fn, weight = jobs[0]
    assert name == "companion-grand-task"
    from app.idle_scheduler import JobWeight
    assert weight == JobWeight.MEDIUM


def test_run_synthesis_for_all_handles_listing_failure(tmp_dirs):
    def _broken():
        raise RuntimeError("DB down")

    with patch("app.companion.grand_task._list_projects", _broken):
        n = _gt.run_synthesis_for_all_workspaces()
    assert n == 0


def test_run_synthesis_for_all_skips_disabled_workspaces(tmp_dirs):
    rows = [
        {"id": "a", "config_json": {"companion": {"enabled": False}}},
        {"id": "b", "config_json": {"companion": {"enabled": True}}},
    ]
    visited: list[str] = []

    def _fake_synthesize(ws):
        visited.append(ws)
        return None

    with patch("app.companion.grand_task._list_projects", lambda: rows), \
         patch("app.companion.grand_task.synthesize", _fake_synthesize):
        _gt.run_synthesis_for_all_workspaces()
    assert visited == ["b"]


# ── Parser ─────────────────────────────────────────────────────────────────

def test_parse_proposal_extracts_both_fields():
    raw = ("GRAND_TASK: Build cathedral-scale forest decision support.\n"
           "RATIONALE: All ideas converge on multi-stakeholder decision flows.")
    text, rationale = _gt._parse_proposal(raw)
    assert "cathedral-scale" in text
    assert "multi-stakeholder" in rationale.lower()


def test_parse_proposal_handles_missing_rationale():
    text, rationale = _gt._parse_proposal("GRAND_TASK: just the task")
    assert "just the task" in text
    assert rationale == ""


def test_parse_proposal_returns_empty_on_garbage():
    text, rationale = _gt._parse_proposal("totally unrelated output")
    assert text == ""
    assert rationale == ""


def test_parse_proposal_clamps_long_text():
    long = "x" * 1000
    raw = f"GRAND_TASK: {long}\nRATIONALE: y"
    text, _ = _gt._parse_proposal(raw)
    assert len(text) <= 240
