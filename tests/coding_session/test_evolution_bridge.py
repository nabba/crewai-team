"""Tests for app.coding_session.evolution_bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.coding_session.evolution_bridge import (
    MAX_COST_USD_INLINE,
    MAX_GENERATIONS_INLINE,
    MAX_ISLANDS_INLINE,
    EvolutionResult,
    RunnerOutput,
    evolve_in_session,
)
from app.coding_session.models import Status


@dataclass
class _FakeSession:
    id: str
    worktree_path: str
    status: Status = Status.ACTIVE


class _FakeManager:
    """Minimal manager stand-in for tests — supports only ``get(id)``."""

    def __init__(self, sessions: list[_FakeSession]) -> None:
        self._by_id = {s.id: s for s in sessions}

    def get(self, sid: str) -> _FakeSession | None:
        return self._by_id.get(sid)


def _populate_worktree(tmp_path: Path, initial_text: str = "x = 1\n") -> tuple[Path, Path]:
    initial = tmp_path / "initial.py"
    initial.write_text(initial_text)
    evaluate = tmp_path / "evaluate.py"
    evaluate.write_text("# evaluator\n")
    return initial, evaluate


# ── refusal paths ────────────────────────────────────────────────────────


def test_refuses_unknown_session(tmp_path: Path) -> None:
    mgr = _FakeManager([])
    result = evolve_in_session(
        session_id="missing",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "not found" in result.refusal_reason


def test_refuses_inactive_session(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path), status=Status.SUBMITTED)
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "not ACTIVE" in result.refusal_reason


def test_refuses_path_traversal(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="../escape.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "traversal" in result.refusal_reason or "outside" in result.refusal_reason


def test_refuses_path_outside_worktree(tmp_path: Path) -> None:
    # Create a sibling dir to the worktree; attempt to point at it via symlink.
    _populate_worktree(tmp_path)
    sibling = tmp_path.parent / "outside_worktree"
    sibling.mkdir(exist_ok=True)
    (sibling / "outside.py").write_text("x = 99\n")

    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    # Symlink the file into the worktree, then evaluate-path points at the symlink.
    link = tmp_path / "linked_outside.py"
    if link.exists():
        link.unlink()
    link.symlink_to(sibling / "outside.py")

    result = evolve_in_session(
        session_id="abc",
        initial_path="linked_outside.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "outside" in result.refusal_reason


def test_refuses_missing_initial_path(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="does_not_exist.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "does not exist" in result.refusal_reason


def test_refuses_tier_immutable_repo_path(tmp_path: Path) -> None:
    # A worktree shaped like the repo with safety_guardian.py present.
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "safety_guardian.py").write_text("x = 1\n")
    (tmp_path / "app" / "evaluate.py").write_text("# eval\n")
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="app/safety_guardian.py",
        evaluate_path="app/evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "TIER_IMMUTABLE" in result.refusal_reason or "protected" in result.refusal_reason


def test_refuses_subia_path(tmp_path: Path) -> None:
    (tmp_path / "app" / "subia").mkdir(parents=True)
    (tmp_path / "app" / "subia" / "kernel.py").write_text("x = 1\n")
    (tmp_path / "evaluate.py").write_text("# eval\n")
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="app/subia/kernel.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"
    assert "consciousness" in result.refusal_reason or "Tier-3" in result.refusal_reason


def test_refuses_goal_emitter_path(tmp_path: Path) -> None:
    (tmp_path / "app" / "affect").mkdir(parents=True)
    (tmp_path / "app" / "affect" / "goal_emitter.py").write_text("x = 1\n")
    (tmp_path / "evaluate.py").write_text("# eval\n")
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])
    result = evolve_in_session(
        session_id="abc",
        initial_path="app/affect/goal_emitter.py",
        evaluate_path="evaluate.py",
        manager=mgr,
    )
    assert result.status == "refused"


# ── happy paths via injected runner ──────────────────────────────────────


def test_no_improvement_returns_empty_diff(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    def fake_runner(**_: Any) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.5, baseline_score=0.5,  # zero delta
            best_program_path=None,
            generations_run=3, variants_evaluated=10,
        )

    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert result.status == "no_improvement"
    assert result.delta == 0.0
    assert result.diff == ""
    assert result.generations_run == 3


def test_improvement_returns_unified_diff(tmp_path: Path) -> None:
    initial, _ = _populate_worktree(tmp_path, initial_text="x = 1\n")
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    best = tmp_path / "best.py"
    best.write_text("x = 2\n")

    def fake_runner(**_: Any) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.9, baseline_score=0.5,
            best_program_path=best,
            generations_run=5, variants_evaluated=20,
        )

    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert result.status == "improved"
    assert result.delta == 0.4
    assert "x = 1" in result.diff
    assert "x = 2" in result.diff
    assert result.diff.startswith("---")  # unified diff header


def test_caps_clamp_oversized_requests(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    captured: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> RunnerOutput:
        captured.update(kwargs)
        return RunnerOutput(
            best_score=0.0, baseline_score=0.0,
            best_program_path=None,
            generations_run=0, variants_evaluated=0,
        )

    evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        num_generations=10_000,
        num_islands=10_000,
        max_cost_usd=10_000.0,
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert captured["num_generations"] == MAX_GENERATIONS_INLINE
    assert captured["num_islands"] == MAX_ISLANDS_INLINE
    assert captured["max_cost_usd"] == MAX_COST_USD_INLINE


def test_runner_error_surfaces_as_error_status(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    def fake_runner(**_: Any) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.0, baseline_score=0.0,
            best_program_path=None,
            generations_run=0, variants_evaluated=0,
            error="runner crashed unexpectedly",
        )

    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert result.status == "error"
    assert "crashed" in result.error


def test_shinka_unavailable_status_distinguished(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    def fake_runner(**_: Any) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.0, baseline_score=0.0,
            best_program_path=None,
            generations_run=0, variants_evaluated=0,
            error="shinka not installed: No module named 'shinka'",
        )

    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert result.status == "shinka_unavailable"


def test_results_dir_is_inside_worktree(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    captured: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> RunnerOutput:
        captured.update(kwargs)
        return RunnerOutput(
            best_score=0.5, baseline_score=0.5,
            best_program_path=None,
            generations_run=0, variants_evaluated=0,
        )

    evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    results_dir: Path = captured["results_dir"]
    assert tmp_path.resolve() in results_dir.resolve().parents
    assert results_dir.exists()


def test_duration_is_recorded(tmp_path: Path) -> None:
    _populate_worktree(tmp_path)
    s = _FakeSession(id="abc", worktree_path=str(tmp_path))
    mgr = _FakeManager([s])

    def fake_runner(**_: Any) -> RunnerOutput:
        return RunnerOutput(
            best_score=0.5, baseline_score=0.5,
            best_program_path=None,
            generations_run=0, variants_evaluated=0,
        )

    result = evolve_in_session(
        session_id="abc",
        initial_path="initial.py",
        evaluate_path="evaluate.py",
        manager=mgr,
        runner_factory=fake_runner,
    )
    assert result.duration_seconds >= 0
