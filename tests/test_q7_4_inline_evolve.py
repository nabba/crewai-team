"""PROGRAM §45.4 — Q7.4 inline ShinkaEvolve operator-visibility tests.

The bridge + agent tool + safety refusals were shipped earlier (see
``tests/coding_session/test_evolution_bridge.py`` and
``tests/coding_session/test_evolve_tool.py``). Q7.4 closes the
operator-visibility gap with:

  1. Per-session evolution audit JSONL persisted OUTSIDE the worktree
     so it survives session cleanup.
  2. Master switch ``shinka_inline_evolve_enabled`` (default ON;
     when OFF the bridge returns ``status="disabled"``).
  3. REST endpoint ``GET /api/cp/coding-sessions/<id>/evolution_runs``
     serving the audit + a one-shot summary.
  4. React surface — verified at source level (no Vitest harness).

These tests cover (1) + (2) + (3) wiring. The bulk subsystem at
``app.shinka_engine`` is unaffected (it has its own master switch).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# ─────────────────────────────────────────────────────────────────────
#   Fixtures — point the audit at a tmp workspace
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_workspace(monkeypatch, tmp_path: Path) -> Path:
    """Redirect the evolution-audit module to a temporary workspace."""
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    return tmp_path


@dataclass
class _FakeResult:
    """Mimics ``EvolutionResult`` for the audit test (avoids importing
    the bridge module so audit tests stand alone)."""
    status: str
    baseline_score: float = 0.0
    best_score: float = 0.0
    delta: float = 0.0
    diff: str = ""
    generations_run: int = 0
    variants_evaluated: int = 0
    duration_seconds: float = 0.0
    error: str = ""
    refusal_reason: str = ""


# ─────────────────────────────────────────────────────────────────────
#   evolution_audit module — persistence
# ─────────────────────────────────────────────────────────────────────


def test_audit_append_creates_jsonl_outside_worktree(tmp_workspace: Path) -> None:
    """append_run() lives under workspace/coding_sessions/<id>/ so it
    survives worktree cleanup."""
    from app.coding_session import evolution_audit

    ok = evolution_audit.append_run(
        session_id="cs_test_42",
        agent_id="coder",
        initial_path="app/foo.py",
        evaluate_path="evals/foo.py",
        num_generations=3,
        num_islands=1,
        max_cost_usd=0.5,
        result=_FakeResult(
            status="improved",
            baseline_score=0.5,
            best_score=0.85,
            delta=0.35,
            diff="--- a/x\n+++ b/x\n",
            generations_run=3,
            variants_evaluated=8,
            duration_seconds=12.0,
        ),
    )
    assert ok is True
    expected = (
        tmp_workspace / "coding_sessions" / "cs_test_42"
        / "evolution_audit.jsonl"
    )
    assert expected.exists()
    rows = expected.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["session_id"] == "cs_test_42"
    assert row["status"] == "improved"
    assert row["baseline_score"] == 0.5
    assert row["best_score"] == 0.85
    assert row["delta"] == 0.35
    # The diff itself is NOT persisted — only the hash + length.
    assert "diff" not in row
    assert row["diff_sha256"]  # non-empty hex digest
    assert row["diff_length"] == len("--- a/x\n+++ b/x\n")


def test_audit_append_failure_isolated(tmp_workspace: Path, monkeypatch) -> None:
    """A broken filesystem mustn't fail the bridge — append_run returns
    False but never raises."""
    from app.coding_session import evolution_audit

    def boom(*a, **k):
        raise OSError("disk full")

    # Force the canonical helper to fail; fallback open() also fails.
    monkeypatch.setattr(
        evolution_audit, "_audit_path",
        lambda sid: Path("/proc/1/cannot-write/here.jsonl"),
    )
    ok = evolution_audit.append_run(
        session_id="cs_test_43",
        agent_id="coder",
        initial_path="app/foo.py",
        evaluate_path="evals/foo.py",
        num_generations=3,
        num_islands=1,
        max_cost_usd=0.5,
        result=_FakeResult(status="improved"),
    )
    assert ok is False  # silent failure — never raises


def test_audit_rejects_unsafe_session_id(tmp_workspace: Path) -> None:
    """Session ids with path separators are refused at _audit_path."""
    from app.coding_session import evolution_audit

    # The path-construction helper rejects unsafe ids.
    with pytest.raises(ValueError):
        evolution_audit._audit_path("../escape/attempt")
    with pytest.raises(ValueError):
        evolution_audit._audit_path("cs/with/slashes")
    # append_run swallows the ValueError and returns False.
    ok = evolution_audit.append_run(
        session_id="../escape",
        agent_id="coder",
        initial_path="x",
        evaluate_path="y",
        num_generations=1,
        num_islands=1,
        max_cost_usd=0.5,
        result=_FakeResult(status="improved"),
    )
    assert ok is False


def test_audit_read_runs_returns_newest_first(tmp_workspace: Path) -> None:
    """read_runs returns rows newest-first (tail of the JSONL)."""
    from app.coding_session import evolution_audit

    sid = "cs_test_44"
    for i, status in enumerate(["refused", "no_improvement", "improved"]):
        evolution_audit.append_run(
            session_id=sid,
            agent_id="coder",
            initial_path="app/foo.py",
            evaluate_path="evals/foo.py",
            num_generations=i + 1,
            num_islands=1,
            max_cost_usd=0.1 * (i + 1),
            result=_FakeResult(status=status, best_score=float(i), delta=float(i)),
        )
    rows = evolution_audit.read_runs(sid, limit=10)
    assert [r["status"] for r in rows] == ["improved", "no_improvement", "refused"]


def test_audit_read_runs_missing_file_returns_empty(tmp_workspace: Path) -> None:
    from app.coding_session import evolution_audit
    assert evolution_audit.read_runs("never_existed") == []


def test_audit_summary_aggregates_runs(tmp_workspace: Path) -> None:
    """session_summary rolls up counts, best delta, total cost + duration."""
    from app.coding_session import evolution_audit

    sid = "cs_test_45"
    evolution_audit.append_run(
        session_id=sid, agent_id="coder",
        initial_path="x", evaluate_path="y",
        num_generations=5, num_islands=1, max_cost_usd=1.0,
        result=_FakeResult(
            status="improved", delta=0.4, duration_seconds=10.0,
        ),
    )
    evolution_audit.append_run(
        session_id=sid, agent_id="coder",
        initial_path="x", evaluate_path="y",
        num_generations=3, num_islands=1, max_cost_usd=0.5,
        result=_FakeResult(
            status="no_improvement", delta=0.0, duration_seconds=5.0,
        ),
    )
    s = evolution_audit.session_summary(sid)
    assert s["n_runs"] == 2
    assert s["by_status"] == {"improved": 1, "no_improvement": 1}
    assert abs(s["best_delta"] - 0.4) < 1e-9
    assert abs(s["total_max_cost_usd"] - 1.5) < 1e-9
    assert abs(s["total_duration_seconds"] - 15.0) < 1e-9
    assert s["last_run_at"] is not None


# ─────────────────────────────────────────────────────────────────────
#   bridge integration — every call writes an audit row
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _FakeSession:
    id: str
    worktree_path: str
    status: object  # Status enum — kept loose to avoid the import shape


class _FakeManager:
    def __init__(self, sessions: list[_FakeSession]) -> None:
        self._by_id = {s.id: s for s in sessions}

    def get(self, sid: str):
        return self._by_id.get(sid)


def _populate_worktree(tmp_path: Path) -> tuple[Path, Path]:
    initial = tmp_path / "initial.py"
    initial.write_text("x = 1\n")
    evaluate = tmp_path / "evaluate.py"
    evaluate.write_text("# evaluator\n")
    return initial, evaluate


def test_bridge_writes_audit_on_refusal(tmp_workspace: Path, tmp_path: Path) -> None:
    """A refusal still gets an audit row — operators can see WHAT the
    agent tried and WHY it was refused."""
    from app.coding_session import evolution_audit
    from app.coding_session.evolution_bridge import evolve_in_session

    mgr = _FakeManager([])  # session not found
    result = evolve_in_session(
        session_id="cs_unknown",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    rows = evolution_audit.read_runs("cs_unknown")
    assert len(rows) == 1
    assert rows[0]["status"] == "refused"
    assert "not found" in rows[0]["refusal_reason"]


def test_bridge_writes_audit_on_improvement(tmp_workspace: Path, tmp_path: Path) -> None:
    """A successful evolution writes an audit row with the improvement
    metrics + diff hash (not the diff itself)."""
    from app.coding_session import evolution_audit
    from app.coding_session.evolution_bridge import (
        RunnerOutput, evolve_in_session,
    )
    from app.coding_session.models import Status

    worktree = tmp_path / "wt"
    worktree.mkdir()
    initial, evaluate = _popula_simple(worktree)

    session = _FakeSession(
        id="cs_evol_1",
        worktree_path=str(worktree),
        status=Status.ACTIVE,
    )
    mgr = _FakeManager([session])

    best_path = worktree / "best.py"
    best_path.write_text("x = 42\n# evolved!\n")

    def fake_runner(**_kwargs) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.9,
            baseline_score=0.4,
            best_program_path=best_path,
            generations_run=4,
            variants_evaluated=12,
        )

    result = evolve_in_session(
        session_id="cs_evol_1",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        num_generations=4,
        runner_factory=fake_runner,
        manager=mgr,
    )
    assert result.status == "improved"
    assert result.delta > 0
    rows = evolution_audit.read_runs("cs_evol_1")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "improved"
    assert row["baseline_score"] == 0.4
    assert row["best_score"] == 0.9
    assert row["diff_sha256"]  # non-empty
    assert row["diff_length"] > 0


def _popula_simple(worktree: Path) -> tuple[Path, Path]:
    initial = worktree / "initial.py"
    initial.write_text("x = 1\n")
    evaluate = worktree / "evaluate.py"
    evaluate.write_text("# evaluator\n")
    return initial, evaluate


# ─────────────────────────────────────────────────────────────────────
#   Master switch — OFF path
# ─────────────────────────────────────────────────────────────────────


def test_bridge_returns_disabled_when_master_switch_off(
    tmp_workspace: Path, tmp_path: Path, monkeypatch,
) -> None:
    """When ``shinka_inline_evolve_enabled`` is False the bridge skips
    validation + runner invocation and returns ``status="disabled"``.
    The audit row distinguishes this from ``refused``."""
    from app.coding_session import evolution_audit, evolution_bridge

    # Patch the master-switch lookup directly (avoids the runtime_settings
    # → app.config → pydantic_settings dep chain in the test env).
    monkeypatch.setattr(
        evolution_bridge, "_inline_evolve_enabled", lambda: False,
    )

    def boom_runner(**_kwargs):
        raise AssertionError("runner must not be invoked when disabled")

    result = evolution_bridge.evolve_in_session(
        session_id="cs_off_path",
        initial_path="some/file.py",
        evaluate_path="evals/file.py",
        runner_factory=boom_runner,
    )
    assert result.status == "disabled"
    assert "shinka_inline_evolve_enabled is OFF" in result.refusal_reason

    rows = evolution_audit.read_runs("cs_off_path")
    assert len(rows) == 1
    assert rows[0]["status"] == "disabled"


def test_bridge_fail_open_when_runtime_settings_unavailable(
    tmp_workspace: Path, monkeypatch,
) -> None:
    """``_inline_evolve_enabled`` is fail-open (returns True) so the
    bridge's existing refusal layers stay the safety boundary even when
    the master-switch lookup itself fails."""
    from app.coding_session import evolution_bridge

    # Simulate runtime_settings unavailable.
    def boom_getter():
        raise RuntimeError("runtime_settings module unavailable")

    # The function reads inside; patch its import target by inserting a
    # broken module into sys.modules. Simpler: patch the helper to
    # exercise its except branch.
    real = evolution_bridge._inline_evolve_enabled
    # The implementation catches ANY Exception and returns True.
    # Verify by mock that this is true.
    import sys
    sys.modules["app.runtime_settings"] = MagicMock(
        get_shinka_inline_evolve_enabled=boom_getter,
    )
    try:
        assert evolution_bridge._inline_evolve_enabled() is True
    finally:
        sys.modules.pop("app.runtime_settings", None)


# ─────────────────────────────────────────────────────────────────────
#   Source-level wiring (no Vitest harness)
# ─────────────────────────────────────────────────────────────────────


def test_config_api_handles_master_switch() -> None:
    """The /api/cp/settings POST handler routes the new key through
    set_shinka_inline_evolve_enabled."""
    src = Path("app/api/config_api.py").read_text(encoding="utf-8")
    assert "set_shinka_inline_evolve_enabled" in src
    assert '"shinka_inline_evolve_enabled" in payload' in src


def test_coding_sessions_api_serves_evolution_runs() -> None:
    """REST endpoint GET /api/cp/coding-sessions/<id>/evolution_runs
    exists and wires through the audit module."""
    src = Path("app/control_plane/coding_sessions_api.py").read_text(
        encoding="utf-8",
    )
    assert "/{session_id}/evolution_runs" in src
    assert "from app.coding_session.evolution_audit" in src
    assert "read_runs" in src and "session_summary" in src


def test_react_card_exists_and_card_mounted() -> None:
    """The React master-switch card exists and the Settings page mounts
    it."""
    card = Path("dashboard-react/src/components/InlineEvolveCard.tsx").read_text(
        encoding="utf-8",
    )
    assert "shinka_inline_evolve_enabled" in card
    assert "Q7.4" in card or "PROGRAM §45.4" in card

    settings = Path("dashboard-react/src/components/SettingsPage.tsx").read_text(
        encoding="utf-8",
    )
    assert "import { InlineEvolveCard }" in settings
    assert "<InlineEvolveCard" in settings


def test_react_evolution_runs_panel_wired() -> None:
    """CodingSessionsPage renders the EvolutionRunsSection on the
    detail drawer; api/coding_sessions exposes the query hook."""
    page = Path("dashboard-react/src/components/CodingSessionsPage.tsx").read_text(
        encoding="utf-8",
    )
    assert "EvolutionRunsSection" in page
    assert "useCodingSessionEvolutionRunsQuery" in page

    api = Path("dashboard-react/src/api/coding_sessions.ts").read_text(
        encoding="utf-8",
    )
    assert "useCodingSessionEvolutionRunsQuery" in api
    assert "evolution_runs" in api


def test_runtime_settings_defaults_inline_evolve_on() -> None:
    """Default is ON — the user said `ship`, the feature is opt-OUT."""
    src = Path("app/runtime_settings.py").read_text(encoding="utf-8")
    assert '"shinka_inline_evolve_enabled": True,' in src
    assert "def get_shinka_inline_evolve_enabled" in src
    assert "def set_shinka_inline_evolve_enabled" in src
