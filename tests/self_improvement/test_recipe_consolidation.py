"""Tests for app.self_improvement.meta_agent.consolidation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.self_improvement.meta_agent import consolidation


@dataclass
class _FakeRecipe:
    id: str
    crew_name: str
    uses: int
    successes: int
    last_used_at: str
    created_at: str


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)


def _recipe(
    rid: str,
    *,
    uses: int = 10,
    successes: int = 5,
    last_used_days: float = 1.0,
    created_days: float = 30.0,
    crew: str = "coder",
) -> _FakeRecipe:
    n = _now()
    return _FakeRecipe(
        id=rid,
        crew_name=crew,
        uses=uses,
        successes=successes,
        last_used_at=_iso(n - timedelta(days=last_used_days)),
        created_at=_iso(n - timedelta(days=created_days)),
    )


# ── compute_health ─────────────────────────────────────────────────────


def test_health_high_for_winning_recent_recipe() -> None:
    h = consolidation.compute_health(
        _recipe("good", uses=20, successes=18, last_used_days=2, created_days=30),
        now=_now(),
    )
    assert h.winrate == 0.9
    assert h.selection_recency > 0.9
    assert h.health > 0.65


def test_health_low_for_failing_stale_recipe() -> None:
    h = consolidation.compute_health(
        _recipe("bad", uses=10, successes=1, last_used_days=120, created_days=200),
        now=_now(),
    )
    assert h.winrate == 0.1
    assert h.health < 0.30


def test_health_unused_recipe_decays_via_age() -> None:
    h = consolidation.compute_health(
        _recipe("untouched", uses=0, successes=0, last_used_days=365, created_days=365),
        now=_now(),
    )
    # All four terms should be low; especially winrate (0) and recency (~0).
    assert h.health < 0.20


# ── _should_propose dedup logic ────────────────────────────────────────


def test_should_propose_low_health_no_prior() -> None:
    h = consolidation.compute_health(
        _recipe("x", uses=10, successes=1, last_used_days=120, created_days=200),
        now=_now(),
    )
    ok, reason = consolidation._should_propose(
        h, prior=None,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert ok
    assert "below threshold" in reason


def test_should_skip_when_above_threshold() -> None:
    h = consolidation.compute_health(
        _recipe("x", uses=20, successes=18, last_used_days=2, created_days=30),
        now=_now(),
    )
    ok, _ = consolidation._should_propose(
        h, prior=None,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert not ok


def test_should_skip_below_min_uses() -> None:
    h = consolidation.compute_health(
        _recipe("y", uses=1, successes=0),
        now=_now(),
    )
    ok, reason = consolidation._should_propose(
        h, prior=None,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert not ok
    assert "min uses" in reason


def test_should_skip_within_dedup_window_no_drop() -> None:
    h = consolidation.compute_health(
        _recipe("z", uses=10, successes=1, last_used_days=120, created_days=200),
        now=_now(),
    )
    prior = {
        "recipe_id": "z",
        "health": h.health,  # same as before
        "proposed_at": _iso(_now() - timedelta(days=5)),
    }
    ok, _ = consolidation._should_propose(
        h, prior=prior,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert not ok


def test_should_propose_after_dedup_window() -> None:
    h = consolidation.compute_health(
        _recipe("z", uses=10, successes=1, last_used_days=120, created_days=200),
        now=_now(),
    )
    prior = {
        "recipe_id": "z",
        "health": h.health,
        "proposed_at": _iso(_now() - timedelta(days=45)),
    }
    ok, reason = consolidation._should_propose(
        h, prior=prior,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert ok
    assert "days old" in reason


def test_should_propose_within_window_when_health_dropped() -> None:
    h = consolidation.compute_health(
        _recipe("z", uses=10, successes=1, last_used_days=120, created_days=200),
        now=_now(),
    )
    prior = {
        "recipe_id": "z",
        "health": h.health + 0.10,  # was 0.10 higher; counts as a meaningful drop
        "proposed_at": _iso(_now() - timedelta(days=5)),
    }
    ok, reason = consolidation._should_propose(
        h, prior=prior,
        threshold=0.30, dedup_days=30, drop_delta=0.05, now=_now(),
    )
    assert ok
    assert "dropped" in reason


# ── run_one_pass orchestration ─────────────────────────────────────────


def test_run_one_pass_writes_proposals_for_low_health(tmp_path: Path) -> None:
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "proposals.jsonl",
        recipes=[
            _recipe("good", uses=20, successes=18, last_used_days=2, created_days=30),
            _recipe("bad", uses=15, successes=2, last_used_days=120, created_days=300),
        ],
        now=_now(),
    )
    assert out["status"] == "ok"
    assert out["proposed"] == 1
    assert {p["recipe_id"] for p in out["proposals"]} == {"bad"}


def test_run_one_pass_dedups_via_jsonl_history(tmp_path: Path) -> None:
    p = tmp_path / "proposals.jsonl"
    bad = _recipe("bad", uses=15, successes=2, last_used_days=120, created_days=300)

    first = consolidation.run_one_pass(
        proposals_path=p, recipes=[bad], now=_now(),
    )
    assert first["proposed"] == 1

    # Re-run "the next day" — within dedup window, no health drop → suppressed.
    second_now = _now() + timedelta(days=1)
    second = consolidation.run_one_pass(
        proposals_path=p, recipes=[bad], now=second_now,
    )
    assert second["proposed"] == 0


def test_run_one_pass_no_recipes(tmp_path: Path) -> None:
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "proposals.jsonl", recipes=[],
    )
    assert out["status"] == "no_recipes"
    assert out["proposed"] == 0


def test_run_one_pass_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RECIPE_CONSOLIDATION_ENABLED", "false")
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl",
        recipes=[_recipe("bad", uses=10, successes=0)],
    )
    assert out["status"] == "disabled"


def test_run_one_pass_load_failed_when_store_raises(monkeypatch, tmp_path: Path) -> None:
    """When list_recipes raises, the pass returns load_failed without writing."""
    def boom(**_):
        raise RuntimeError("Postgres down")

    import app.self_improvement.meta_agent.store as store_mod
    monkeypatch.setattr(store_mod, "list_recipes", boom)
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl", recipes=None,
    )
    assert out["status"] == "load_failed"
    assert "Postgres" in out["error"]


def test_proposals_jsonl_format(tmp_path: Path) -> None:
    p = tmp_path / "proposals.jsonl"
    consolidation.run_one_pass(
        proposals_path=p,
        recipes=[
            _recipe("bad", uses=15, successes=2, last_used_days=120, created_days=300),
        ],
        now=_now(),
    )
    text = p.read_text(encoding="utf-8")
    assert text.strip()
    row = json.loads(text.strip().split("\n")[0])
    assert row["recipe_id"] == "bad"
    assert row["crew_name"] == "coder"
    assert "health" in row
    assert "proposed_at" in row
    assert "reason" in row


# ── daemon discipline ──────────────────────────────────────────────────


def test_disabled_short_circuits_start(monkeypatch, caplog) -> None:
    monkeypatch.setenv("RECIPE_CONSOLIDATION_ENABLED", "false")
    assert consolidation._enabled() is False
    with caplog.at_level(logging.INFO, logger="app.self_improvement.meta_agent.consolidation"):
        consolidation.start()
    assert any("disabled via" in r.message for r in caplog.records)


def test_stop_sets_event() -> None:
    consolidation.stop()
    assert consolidation._stop_event.is_set()
