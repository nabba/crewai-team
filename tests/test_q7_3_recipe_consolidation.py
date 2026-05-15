"""PROGRAM §45.3 — Q7.3 recipe-consolidation selection-rate trigger.

The pre-existing tests in ``tests/self_improvement/test_recipe_consolidation.py``
cover the weekly health-score trigger. This module covers the parallel
selection-rate trigger added in Q7.3, plus the operator-approved
``superseded_by`` writer and the boot anchor.

Both triggers compose: a recipe can fail health-score OR selection-rate
(or both). The two triggers are deduped within a single pass via
``proposed_in_pass`` — a recipe is never proposed twice in one pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.self_improvement.meta_agent import consolidation


# ─────────────────────────────────────────────────────────────────────
#   Fixtures (mirroring the existing test_recipe_consolidation.py
#   shape — simple namespaces / dataclasses, not real AgentRecipe).
# ─────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRecipe:
    id: str
    crew_name: str
    uses: int
    successes: int
    last_used_at: str
    created_at: str
    # Q7.3 fields — all optional, exercised when shape variants apply.
    offered_at_history: list[str] | None = None
    selected_at_history: list[str] | None = None
    times_offered_90d: int | None = None
    times_selected_90d: int | None = None
    times_offered: int | None = None
    superseded_by: str | None = None
    is_active: bool = True


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
    offered_at_history: list[str] | None = None,
    selected_at_history: list[str] | None = None,
    times_offered_90d: int | None = None,
    times_selected_90d: int | None = None,
    times_offered: int | None = None,
) -> _FakeRecipe:
    n = _now()
    return _FakeRecipe(
        id=rid,
        crew_name=crew,
        uses=uses,
        successes=successes,
        last_used_at=_iso(n - timedelta(days=last_used_days)),
        created_at=_iso(n - timedelta(days=created_days)),
        offered_at_history=offered_at_history,
        selected_at_history=selected_at_history,
        times_offered_90d=times_offered_90d,
        times_selected_90d=times_selected_90d,
        times_offered=times_offered,
    )


# ─────────────────────────────────────────────────────────────────────
#   compute_selection_rate — shape variants
# ─────────────────────────────────────────────────────────────────────


def test_selection_rate_from_history_within_window() -> None:
    """offered_at_history + selected_at_history → counted only when ISO ts ≥ cutoff."""
    n = _now()
    # 30 offers in window, 1 selection in window — rate 1/30 ≈ 0.033 (< 5%)
    offered = [_iso(n - timedelta(days=d)) for d in range(1, 31)]
    selected = [_iso(n - timedelta(days=5))]
    rec = _recipe("low", offered_at_history=offered, selected_at_history=selected)
    n_off, n_sel, rate = consolidation.compute_selection_rate(rec, now=n)
    assert n_off == 30
    assert n_sel == 1
    assert rate < 0.05


def test_selection_rate_excludes_pre_window_timestamps() -> None:
    """Offers older than the 90d window are excluded."""
    n = _now()
    # 5 fresh offers, 100 ancient offers (>120d ago)
    fresh = [_iso(n - timedelta(days=d)) for d in (1, 2, 3, 4, 5)]
    ancient = [_iso(n - timedelta(days=120 + d)) for d in range(100)]
    rec = _recipe(
        "old",
        offered_at_history=fresh + ancient,
        selected_at_history=[_iso(n - timedelta(days=2))],
    )
    n_off, n_sel, _ = consolidation.compute_selection_rate(rec, now=n)
    assert n_off == 5  # only the in-window offers
    assert n_sel == 1


def test_selection_rate_from_precomputed_counters() -> None:
    """times_offered_90d / times_selected_90d skip per-timestamp counting."""
    rec = _recipe(
        "precounted",
        times_offered_90d=50,
        times_selected_90d=1,
    )
    n_off, n_sel, rate = consolidation.compute_selection_rate(rec, now=_now())
    assert n_off == 50
    assert n_sel == 1
    assert abs(rate - 0.02) < 0.001


def test_selection_rate_fallback_to_lifetime() -> None:
    """No history / no 90d counters → lifetime uses + times_offered (conservative)."""
    rec = _recipe(
        "lifetime",
        uses=2,
        successes=2,
        times_offered=100,
    )
    n_off, n_sel, rate = consolidation.compute_selection_rate(rec, now=_now())
    assert n_off == 100
    assert n_sel == 2
    assert abs(rate - 0.02) < 0.001


def test_selection_rate_zero_offers_returns_zero_rate() -> None:
    """Divide-by-zero protection: zero offers → rate 0.0 (and trigger guards)."""
    rec = _recipe("never-offered", times_offered=0)
    n_off, n_sel, rate = consolidation.compute_selection_rate(rec, now=_now())
    assert n_off == 0
    assert rate == 0.0


# ─────────────────────────────────────────────────────────────────────
#   _should_propose_via_selection_rate
# ─────────────────────────────────────────────────────────────────────


def test_propose_below_threshold_with_enough_offers() -> None:
    ok, reason = consolidation._should_propose_via_selection_rate(
        n_offered=50, rate=0.02, prior=None, now=_now(),
    )
    assert ok
    assert "below threshold" in reason
    assert "n_offered=50" in reason


def test_min_offers_gate_blocks_small_sample() -> None:
    """n_offered=19 < 20 (the min-N gate) → never proposed regardless of rate."""
    ok, reason = consolidation._should_propose_via_selection_rate(
        n_offered=19, rate=0.0, prior=None, now=_now(),
    )
    assert not ok
    assert "below min offers" in reason


def test_skip_when_rate_at_or_above_threshold() -> None:
    """Rate at 5% exactly → not proposed (threshold is strict less-than)."""
    ok, reason = consolidation._should_propose_via_selection_rate(
        n_offered=100, rate=0.05, prior=None, now=_now(),
    )
    assert not ok
    assert "above threshold" in reason


def test_skip_within_dedup_window_no_drop() -> None:
    """Proposed 5 days ago, still low rate → suppressed (re-propose after 30d)."""
    prior = {
        "recipe_id": "x",
        "health": 0.10,
        "proposed_at": _iso(_now() - timedelta(days=5)),
    }
    ok, _ = consolidation._should_propose_via_selection_rate(
        n_offered=50, rate=0.02, prior=prior, now=_now(),
    )
    assert not ok


def test_repropose_after_dedup_window() -> None:
    """Proposed >30d ago → re-propose."""
    prior = {
        "recipe_id": "x",
        "health": 0.10,
        "proposed_at": _iso(_now() - timedelta(days=45)),
    }
    ok, reason = consolidation._should_propose_via_selection_rate(
        n_offered=50, rate=0.02, prior=prior, now=_now(),
    )
    assert ok
    assert "d old" in reason  # "45d old; re-propose ..."


# ─────────────────────────────────────────────────────────────────────
#   run_one_pass — both triggers compose
# ─────────────────────────────────────────────────────────────────────


def test_run_one_pass_fires_selection_rate_trigger(tmp_path: Path) -> None:
    """Recipe with healthy winrate but low selection rate → selection-rate fires."""
    # Construct: high winrate (5/5 = 100%), recent use (1d), but only
    # 1 selection out of 100 offers — operators aren't picking it.
    n = _now()
    rec = _recipe(
        "unpopular",
        uses=5,
        successes=5,
        last_used_days=1,
        created_days=60,
        offered_at_history=[_iso(n - timedelta(days=d)) for d in range(1, 101)],
        selected_at_history=[_iso(n - timedelta(days=5))],
    )
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl",
        recipes=[rec],
        now=_now(),
    )
    assert out["status"] == "ok"
    assert out["proposed"] == 1
    text = (tmp_path / "p.jsonl").read_text(encoding="utf-8")
    row = json.loads(text.strip())
    assert row["recipe_id"] == "unpopular"
    assert row["reason"].startswith("selection_rate:")


def test_run_one_pass_health_score_takes_precedence_in_same_pass(tmp_path: Path) -> None:
    """When BOTH triggers would fire for one recipe in one pass, only the
    health-score row is written (single propagated proposal per pass).
    The selection-rate row would surface on the next pass once the
    health-score one is deduped — both triggers remain active long-term."""
    n = _now()
    # Bad health (low winrate, stale) AND low selection rate.
    rec = _recipe(
        "double_bad",
        uses=15,
        successes=2,
        last_used_days=120,
        created_days=300,
        offered_at_history=[_iso(n - timedelta(days=d)) for d in range(1, 101)],
        selected_at_history=[_iso(n - timedelta(days=2))],
    )
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl",
        recipes=[rec],
        now=_now(),
    )
    assert out["proposed"] == 1
    rows = (tmp_path / "p.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(rows) == 1
    assert json.loads(rows[0])["reason"].startswith("health_score:")


def test_run_one_pass_selection_rate_skips_recipe_without_signal(tmp_path: Path) -> None:
    """A recipe with no offered-history / no 90d counters / no
    times_offered field doesn't trigger the selection-rate path (n_off=0
    below min-N). Existing-tests behavior unchanged."""
    rec = _recipe("plain", uses=20, successes=15)
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl",
        recipes=[rec],
        now=_now(),
    )
    # High winrate → health-score doesn't fire either. No proposal.
    assert out["proposed"] == 0
    assert out["n_selection_rate_evaluated"] == 0


def test_run_one_pass_reports_selection_rate_eval_count(tmp_path: Path) -> None:
    """Result dict includes ``n_selection_rate_evaluated`` so operators
    can see how many recipes the selection-rate trigger actually
    considered (vs. how many were below the sample-size gate)."""
    n = _now()
    # 3 recipes, only 2 have enough offers in window.
    rec_a = _recipe(  # 50 offers, 1 selection → triggers
        "a",
        uses=10,
        successes=8,
        offered_at_history=[_iso(n - timedelta(days=d)) for d in range(1, 51)],
        selected_at_history=[_iso(n - timedelta(days=2))],
    )
    rec_b = _recipe(  # 25 offers, 5 selections → above threshold (20%)
        "b",
        uses=10,
        successes=8,
        offered_at_history=[_iso(n - timedelta(days=d)) for d in range(1, 26)],
        selected_at_history=[_iso(n - timedelta(days=d)) for d in (2, 4, 6, 8, 10)],
    )
    rec_c = _recipe(  # 5 offers only → below sample-size gate
        "c",
        uses=10,
        successes=8,
        offered_at_history=[_iso(n - timedelta(days=d)) for d in (1, 2, 3, 4, 5)],
    )
    out = consolidation.run_one_pass(
        proposals_path=tmp_path / "p.jsonl",
        recipes=[rec_a, rec_b, rec_c],
        now=_now(),
    )
    # 2 recipes above the min-N gate (a and b); a fires, b suppressed.
    assert out["n_selection_rate_evaluated"] == 2
    assert out["proposed"] == 1


# ─────────────────────────────────────────────────────────────────────
#   mark_superseded_by — operator approval writer
# ─────────────────────────────────────────────────────────────────────


def test_mark_superseded_by_writes_field_via_store_fallback(monkeypatch) -> None:
    """When ``store.mark_recipe_superseded`` is absent (current shape),
    the fallback path mutates the recipe object directly via
    ``get_recipe`` + ``save_recipe``."""
    captured: dict[str, object] = {}

    @dataclass
    class _MutableRecipe:
        id: str
        crew_name: str = "coder"
        superseded_by: str = ""
        is_active: bool = True

    recipe = _MutableRecipe(id="legacy")

    def fake_get(recipe_id: str):
        return recipe if recipe_id == "legacy" else None

    def fake_save(r):
        captured["saved"] = r

    # Stub the store module: no mark_recipe_superseded export.
    import app.self_improvement.meta_agent.store as store_mod
    monkeypatch.setattr(store_mod, "get_recipe", fake_get, raising=False)
    monkeypatch.setattr(store_mod, "save_recipe", fake_save, raising=False)
    monkeypatch.delattr(
        store_mod, "mark_recipe_superseded", raising=False,
    )

    ok = consolidation.mark_superseded_by(
        "legacy",
        superseded_by="retired:operator_approved:2026-05-15",
        operator_actor="andrus",
    )
    assert ok is True
    assert captured["saved"] is recipe
    assert recipe.superseded_by == "retired:operator_approved:2026-05-15"
    assert recipe.is_active is False


def test_mark_superseded_by_returns_false_when_missing() -> None:
    """Bogus recipe-id → mark_superseded_by returns False without raising."""
    import app.self_improvement.meta_agent.store as store_mod
    # Use the actual store (its get_recipe will return None for the bogus id).
    ok = consolidation.mark_superseded_by("does-not-exist-xyz-q7-3")
    assert ok is False


# ─────────────────────────────────────────────────────────────────────
#   Boot anchor — app.healing imports consolidation eagerly
# ─────────────────────────────────────────────────────────────────────


def test_boot_anchor_lists_consolidation_in_healing_init() -> None:
    """app/healing/__init__.py imports consolidation so the weekly
    daemon eager-starts at production boot."""
    src = Path("app/healing/__init__.py").read_text(encoding="utf-8")
    assert "meta_agent import consolidation" in src
    assert "PROGRAM §45.3" in src or "Q7.3" in src
