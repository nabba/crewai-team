"""Recipe-ledger consolidation — soft-retirement proposer.

Parallels :mod:`app.training.adapter_performance` for adapter
retirement. Where ``prune_dead_recipes`` (in this package's
:mod:`store`) hard-deletes never-used recipes after 90 days, this
module computes *two complementary signals* for each recipe and
proposes retirement of low performers — without ever hard-deleting.

The proposal is *advisory*: it lands as a JSONL row in
``workspace/training/recipe_retirement_proposals.jsonl`` and a
Signal alert. The operator approves; :func:`mark_superseded_by`
then writes ``superseded_by`` + ``is_active=False`` on the recipe.
This module never auto-retires; the proposal layer is the deliverable.

Two parallel triggers (Q7.3 / PROGRAM §45.3):

  1. **Health score** — composite quality signal (the original
     trigger, weekly cadence)::

         health =   winrate            × 0.55      # success / uses
                  + selection_recency  × 0.20      # 1 / (1 + days_since_last_used / 30)
                  + age_normalized     × 0.15      # 1 / (1 + days_since_created / 60)
                  + use_count_log      × 0.10      # log10(1 + uses) / log10(100)

     All four terms are 0..1 (clamped). Below
     ``RECIPE_RETIREMENT_THRESHOLD`` (default 0.30) → propose retirement.

  2. **Selection rate** — quarterly cadence; selection rate over
     the last 90 days. Below ``_SELECTION_RATE_THRESHOLD`` (5%)
     and above ``_SELECTION_RATE_MIN_OFFERS`` (20 — sample-size
     gate) → propose retirement. Catches recipes nobody picks
     even if their internal win-rate looks fine.

Both triggers compose: a recipe failing either signal is proposed.
Within a single pass the two triggers are deduped (no recipe gets two
rows simultaneously — health-score wins precedence in that pass and
selection-rate would surface on the next pass).

Cross-pass dedup: per-recipe-id within ``RECIPE_RETIREMENT_DEDUP_DAYS``
(default 30). Re-proposing the same recipe within that window is
suppressed unless the health score has dropped further by ``DROP_DELTA``
(default 0.05).

Daemon thread eager-starts; weekly cadence (selection-rate is checked
on the same cadence — the *trigger* is quarterly in the sense of its
data window, not its evaluation frequency); master switch
``RECIPE_CONSOLIDATION_ENABLED`` (default ``true``). Same discipline as
healing/monitors and the inquiry scheduler.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "recipe-consolidation"
_WARMUP_S = 90
_POLL_INTERVAL_S = 7 * 24 * 3600  # weekly (health-score trigger cadence)

# Q7.3 — selection-rate trigger (PROGRAM §45.3). Spec wording: retire
# recipes with <5% selection rate over the last 90 days. Lives
# alongside the weekly health-score trigger; both can produce
# retirement proposals (complementary signals — health-score catches
# quality decay; selection-rate catches recipes nobody chooses).
_SELECTION_RATE_WINDOW_DAYS = 90
_SELECTION_RATE_THRESHOLD = 0.05
_SELECTION_RATE_MIN_OFFERS = 20  # min-N gate; below this, sample is too small

_DEFAULT_PROPOSALS_PATH = Path(
    "/app/workspace/training/recipe_retirement_proposals.jsonl"
)

RECIPE_RETIREMENT_THRESHOLD = 0.30
RECIPE_RETIREMENT_DEDUP_DAYS = 30
RECIPE_DROP_DELTA = 0.05  # only re-propose if health dropped this much further
RECIPE_MIN_USES_TO_EVALUATE = 3  # skip recipes that haven't accumulated evidence yet

_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _enabled() -> bool:
    return os.getenv("RECIPE_CONSOLIDATION_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


@dataclass(frozen=True)
class RecipeHealth:
    recipe_id: str
    crew_name: str
    uses: int
    successes: int
    winrate: float
    selection_recency: float
    age_normalized: float
    use_count_log: float
    health: float
    last_used_at: str
    created_at: str


@dataclass(frozen=True)
class RetirementProposal:
    recipe_id: str
    crew_name: str
    health: float
    reason: str
    proposed_at: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _days_since(ts: str | None, now: datetime | None = None) -> float:
    parsed = _parse_iso(ts)
    if parsed is None:
        return 365.0  # unknown → treat as ancient
    if now is None:
        now = _now()
    delta = now - parsed
    return max(0.0, delta.total_seconds() / 86400.0)


def compute_health(recipe: object, now: datetime | None = None) -> RecipeHealth:
    """Compute the four-term health score for one recipe.

    Defensive attribute access — ``AgentRecipe`` is the production type
    but tests inject simple namespaces.
    """
    uses = int(getattr(recipe, "uses", 0) or 0)
    successes = int(getattr(recipe, "successes", 0) or 0)
    winrate = (successes / uses) if uses > 0 else 0.0

    days_since_used = _days_since(getattr(recipe, "last_used_at", "") or "", now)
    days_since_created = _days_since(getattr(recipe, "created_at", "") or "", now)

    selection_recency = 1.0 / (1.0 + days_since_used / 30.0)
    age_normalized = 1.0 / (1.0 + days_since_created / 60.0)
    use_count_log = min(1.0, math.log10(1 + uses) / math.log10(100))

    health = (
        winrate * 0.55
        + selection_recency * 0.20
        + age_normalized * 0.15
        + use_count_log * 0.10
    )
    return RecipeHealth(
        recipe_id=str(getattr(recipe, "id", "") or ""),
        crew_name=str(getattr(recipe, "crew_name", "") or ""),
        uses=uses,
        successes=successes,
        winrate=winrate,
        selection_recency=selection_recency,
        age_normalized=age_normalized,
        use_count_log=use_count_log,
        health=health,
        last_used_at=getattr(recipe, "last_used_at", "") or "",
        created_at=getattr(recipe, "created_at", "") or "",
    )


def _read_dedup_state(path: Path) -> dict[str, dict]:
    """Map recipe_id → {last_health, last_proposed_at}."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("recipe_id")
            if not rid:
                continue
            prior = out.get(rid)
            if prior is None or row.get("proposed_at", "") > prior.get("proposed_at", ""):
                out[rid] = row
    except OSError:
        return {}
    return out


def _should_propose(
    health: RecipeHealth,
    prior: dict | None,
    *,
    threshold: float,
    dedup_days: int,
    drop_delta: float,
    now: datetime | None = None,
) -> tuple[bool, str]:
    if health.uses < RECIPE_MIN_USES_TO_EVALUATE:
        return False, "below min uses to evaluate"
    if health.health >= threshold:
        return False, f"health {health.health:.3f} above threshold"
    if prior is None:
        return True, f"health {health.health:.3f} below threshold {threshold:.2f}"
    last_proposed = _parse_iso(prior.get("proposed_at"))
    if last_proposed is None:
        return True, "prior proposal has unparseable timestamp"
    days = (_now() if now is None else now) - last_proposed
    if days.total_seconds() / 86400.0 >= dedup_days:
        return True, f"prior proposal {days.days} days old; re-propose"
    prior_health = float(prior.get("health", 1.0))
    if prior_health - health.health >= drop_delta:
        return True, (
            f"health dropped {prior_health:.3f} → {health.health:.3f} "
            f"(>= {drop_delta:.2f} delta)"
        )
    return False, "within dedup window and no significant drop"


def run_one_pass(
    *,
    proposals_path: Path | str | None = None,
    recipes: list[object] | None = None,
    threshold: float = RECIPE_RETIREMENT_THRESHOLD,
    now: datetime | None = None,
) -> dict:
    """Single consolidation pass.

    Test/operator hooks:
      ``proposals_path``  overrides the JSONL output location.
      ``recipes``         overrides the source (skips
                          :func:`store.list_recipes` for unit tests).
      ``threshold``       per-call override of the retirement bar.
      ``now``             override the clock (testing).
    """
    if not _enabled():
        return {"status": "disabled", "proposed": 0}

    out_path = Path(proposals_path) if proposals_path else _DEFAULT_PROPOSALS_PATH

    if recipes is None:
        try:
            from app.self_improvement.meta_agent.store import list_recipes
            recipes = list_recipes(limit=1000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recipe_consolidation: list_recipes failed: %s", exc)
            return {"status": "load_failed", "proposed": 0, "error": str(exc)}

    if not recipes:
        return {"status": "no_recipes", "proposed": 0}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prior_state = _read_dedup_state(out_path)

    healths = [compute_health(r, now=now) for r in recipes]
    proposed_rows: list[RetirementProposal] = []
    # Track which recipes get proposed in this pass so the parallel
    # selection-rate trigger doesn't double-propose the same recipe.
    proposed_in_pass: set[str] = set()

    # ───── Trigger 1: health-score (weekly, four-term composite) ─────
    for h in healths:
        prior = prior_state.get(h.recipe_id)
        ok, reason = _should_propose(
            h,
            prior,
            threshold=threshold,
            dedup_days=RECIPE_RETIREMENT_DEDUP_DAYS,
            drop_delta=RECIPE_DROP_DELTA,
            now=now,
        )
        if not ok:
            continue
        proposed_rows.append(RetirementProposal(
            recipe_id=h.recipe_id,
            crew_name=h.crew_name,
            health=h.health,
            reason=f"health_score: {reason}",
            proposed_at=(now or _now()).isoformat(),
        ))
        proposed_in_pass.add(h.recipe_id)

    # ───── Trigger 2: selection-rate (quarterly, <5% over 90d) ─────
    # Complementary to health-score: catches recipes nobody picks even
    # if their internal win-rate looks fine. Q7.3 / PROGRAM §45.3.
    n_selection_rate_evaluated = 0
    for recipe, h in zip(recipes, healths):
        if h.recipe_id in proposed_in_pass:
            continue  # already proposed via health-score this pass
        n_offered, n_selected, rate = compute_selection_rate(
            recipe, now=now,
        )
        if n_offered < _SELECTION_RATE_MIN_OFFERS:
            continue  # below sample-size gate
        n_selection_rate_evaluated += 1
        prior = prior_state.get(h.recipe_id)
        ok, reason = _should_propose_via_selection_rate(
            n_offered, rate, prior, now=now,
        )
        if not ok:
            continue
        proposed_rows.append(RetirementProposal(
            recipe_id=h.recipe_id,
            crew_name=h.crew_name,
            health=h.health,
            reason=f"selection_rate: {reason}",
            proposed_at=(now or _now()).isoformat(),
        ))
        proposed_in_pass.add(h.recipe_id)

    # Append-only writes.
    if proposed_rows:
        with open(out_path, "a", encoding="utf-8") as f:
            for p in proposed_rows:
                row = {
                    "recipe_id": p.recipe_id,
                    "crew_name": p.crew_name,
                    "health": round(p.health, 4),
                    "reason": p.reason,
                    "proposed_at": p.proposed_at,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "status": "ok",
        "n_recipes": len(recipes),
        "n_evaluated": sum(
            1 for h in healths if h.uses >= RECIPE_MIN_USES_TO_EVALUATE
        ),
        "n_selection_rate_evaluated": n_selection_rate_evaluated,
        "proposed": len(proposed_rows),
        "proposals": [
            {"recipe_id": p.recipe_id, "health": round(p.health, 4)}
            for p in proposed_rows
        ],
    }


def compute_selection_rate(
    recipe: object,
    *,
    window_days: int = _SELECTION_RATE_WINDOW_DAYS,
    now: datetime | None = None,
) -> tuple[int, int, float]:
    """Q7.3 — compute (n_offered, n_selected, rate) over the rolling
    window for one recipe.

    Defensive against shape variance:
      * recipe may have ``offered_at_history``/``selected_at_history``
        (preferred — list of ISO timestamps)
      * or fall back to ``times_offered_90d`` / ``times_selected_90d``
        (precomputed counters)
      * or fall back to ``uses`` / ``successes`` (least precise — uses
        the lifetime numbers, which over-counts but is conservative
        since we only retire when rate is LOW)

    Returns (n_offered, n_selected, rate). Rate is 0.0 when offered is 0."""
    now = now or _now()
    cutoff = now - timedelta(days=window_days)

    offered_history = getattr(recipe, "offered_at_history", None)
    selected_history = getattr(recipe, "selected_at_history", None)
    if isinstance(offered_history, (list, tuple)):
        n_offered = sum(
            1 for ts in offered_history
            if (_parse_iso(ts) or cutoff - timedelta(days=1)) >= cutoff
        )
        n_selected = 0
        if isinstance(selected_history, (list, tuple)):
            n_selected = sum(
                1 for ts in selected_history
                if (_parse_iso(ts) or cutoff - timedelta(days=1)) >= cutoff
            )
    elif getattr(recipe, "times_offered_90d", None) is not None:
        # ``hasattr`` is too permissive — dataclass fields with default
        # ``None`` would satisfy it but still mean "no data here." Use
        # explicit ``is not None`` to fall through to the lifetime path.
        n_offered = int(getattr(recipe, "times_offered_90d", 0) or 0)
        n_selected = int(getattr(recipe, "times_selected_90d", 0) or 0)
    else:
        # Least-precise fallback. Use lifetime counters.
        # Conservative: over-counts offered → rate is higher → less
        # likely to retire (fail-open against false positives).
        n_offered = int(getattr(recipe, "times_offered", 0) or 0)
        n_selected = int(getattr(recipe, "uses", 0) or 0)

    rate = (n_selected / n_offered) if n_offered > 0 else 0.0
    return n_offered, n_selected, rate


def _should_propose_via_selection_rate(
    n_offered: int,
    rate: float,
    prior: dict | None,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Q7.3 — selection-rate trigger. Returns (should_propose, reason).

    Trigger conditions:
      * n_offered >= _SELECTION_RATE_MIN_OFFERS (sample-size gate)
      * rate < _SELECTION_RATE_THRESHOLD
      * not within dedup window (or rate dropped meaningfully)
    """
    if n_offered < _SELECTION_RATE_MIN_OFFERS:
        return False, (
            f"below min offers ({n_offered} < {_SELECTION_RATE_MIN_OFFERS})"
        )
    if rate >= _SELECTION_RATE_THRESHOLD:
        return False, (
            f"selection rate {rate:.3f} above threshold "
            f"{_SELECTION_RATE_THRESHOLD:.3f}"
        )
    if prior is None:
        return True, (
            f"selection rate {rate:.3f} over {_SELECTION_RATE_WINDOW_DAYS}d "
            f"below threshold (n_offered={n_offered})"
        )
    last_proposed = _parse_iso(prior.get("proposed_at"))
    if last_proposed is None:
        return True, "prior proposal has unparseable timestamp"
    days_since = (_now() if now is None else now) - last_proposed
    if days_since.total_seconds() / 86400.0 >= RECIPE_RETIREMENT_DEDUP_DAYS:
        return True, (
            f"prior proposal {days_since.days}d old; re-propose at "
            f"selection rate {rate:.3f}"
        )
    return False, "within dedup window"


def mark_superseded_by(
    recipe_id: str,
    *,
    superseded_by: str = "retired:no_successor",
    operator_actor: str = "operator",
) -> bool:
    """Q7.3 — when the operator approves a retirement proposal, mark
    the recipe with ``superseded_by`` and ``is_active=False`` so the
    selection logic skips it. Audit trail preserved — recipe stays
    in the store, never deleted.

    The ``superseded_by`` field can be:
      * Another recipe-id (recipe X replaces this one)
      * A string marker like ``"retired:operator_approved:<ts>"`` or
        ``"retired:no_successor"`` (no replacement)

    Failure-isolated. Returns True on success, False on any failure."""
    try:
        from app.self_improvement.meta_agent.store import mark_recipe_superseded
    except ImportError:
        # The store may not have this helper yet (added in Q7.3).
        # Try the next-best: direct mutation through store.update.
        try:
            from app.self_improvement.meta_agent import store
        except Exception:
            logger.debug("mark_superseded_by: meta_agent.store unavailable")
            return False
        try:
            recipe = store.get_recipe(recipe_id)
            if recipe is None:
                return False
            # Defensive: only set the fields if they exist.
            if hasattr(recipe, "superseded_by"):
                recipe.superseded_by = superseded_by
            if hasattr(recipe, "is_active"):
                recipe.is_active = False
            store.save_recipe(recipe)
            return True
        except Exception:
            logger.debug(
                "mark_superseded_by: fallback path failed for %s",
                recipe_id, exc_info=True,
            )
            return False
    try:
        return bool(mark_recipe_superseded(
            recipe_id,
            superseded_by=superseded_by,
            actor=operator_actor,
        ))
    except Exception:
        logger.debug(
            "mark_superseded_by: store call failed for %s",
            recipe_id, exc_info=True,
        )
        return False


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            result = run_one_pass()
            if result.get("proposed", 0) > 0:
                logger.info(
                    "recipe_consolidation: proposed retirement of %d recipe(s)",
                    result["proposed"],
                )
        except Exception:
            logger.debug("recipe_consolidation: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    global _driver_started
    if not _enabled():
        logger.info(
            "recipe_consolidation: disabled via RECIPE_CONSOLIDATION_ENABLED",
        )
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "recipe_consolidation: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "recipe_consolidation: daemon started (warm-up=%ds, poll=%dh)",
            _WARMUP_S, _POLL_INTERVAL_S // 3600,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import — same discipline as healing/monitors.
start()
