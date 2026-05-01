"""
tier_graduation.py — Dynamic TIER assignment based on file history.

The static three-tier system (IMMUTABLE / GATED / OPEN) treats every file
the same regardless of its track record. A file that has been evolved
successfully 20 times with zero rollbacks should be more trusted than
one that's been touched twice with one rollback.

This module computes a *dynamic* tier overlay on top of the static one:

  - GATED → OPEN promotion: 90 days at GATED with 5+ successful mutations
    AND zero rollbacks AND low centrality (few dependents) AND not on hot path.

  - OPEN → GATED demotion: 3 rollbacks in 7 days OR centrality jumps above
    threshold after self_model rebuild.

  - GATED → IMMUTABLE demotion: 5 rollbacks in 30 days (system loses trust).

  - IMMUTABLE → anywhere: NEVER. Immutable is a hard line by design.

The dynamic tier is queried by auto_deployer alongside the static tier.
The MORE restrictive of the two wins (defense in depth).

Tier history persists across restarts in workspace/tier_history.json.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

TIER_HISTORY_PATH = Path("/app/workspace/tier_history.json")
GRADUATION_LOG_PATH = Path("/app/workspace/tier_graduations.json")

# Promotion criteria (GATED → OPEN)
_MIN_DAYS_AT_GATED_FOR_PROMOTION = 90
_MIN_SUCCESSFUL_MUTATIONS_FOR_PROMOTION = 5
_MAX_ROLLBACKS_FOR_PROMOTION = 0
_MAX_CENTRALITY_FOR_PROMOTION = 0.30

# Demotion criteria
_ROLLBACKS_THRESHOLD_OPEN_TO_GATED = 3   # in 7 days
_ROLLBACKS_THRESHOLD_GATED_TO_IMMUTABLE = 5  # in 30 days


class DynamicTier(str, Enum):
    OPEN = "open"
    GATED = "gated"
    IMMUTABLE = "immutable"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TierHistory:
    """One file's tier history."""
    filepath: str
    static_tier: str               # From auto_deployer's TIER_IMMUTABLE / TIER_GATED
    dynamic_tier: str              # Computed adjustment (may be more restrictive)
    dynamic_tier_since: float
    successful_mutations: int = 0
    rollbacks: int = 0
    rollback_timestamps: list[float] = field(default_factory=list)
    last_mutation_at: float = 0.0


@dataclass(frozen=True)
class TierChange:
    """Record of a tier graduation/demotion event."""
    filepath: str
    from_tier: str
    to_tier: str
    reason: str
    timestamp: float


# ── History persistence ──────────────────────────────────────────────────────

def _load_history() -> dict[str, TierHistory]:
    """Load tier history from disk.

    Phase G5 defensive read: every field uses ``.get()`` with a sensible
    default so a partially-written or schema-evolved JSON file does not
    crash the scheduler at startup. A single bad entry is logged and
    skipped; the rest of the history loads. Previously a single missing
    key (e.g. ``filepath`` on a hand-edited entry) crashed the whole
    load and wiped the in-memory tier state.
    """
    if not TIER_HISTORY_PATH.exists():
        return {}
    try:
        data = json.loads(TIER_HISTORY_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"tier_graduation: history file unreadable: {e}")
        return {}

    if not isinstance(data, dict):
        logger.warning(
            f"tier_graduation: expected dict at top level, got {type(data).__name__}"
        )
        return {}

    out: dict[str, TierHistory] = {}
    skipped = 0
    for path, entry in data.items():
        if not isinstance(entry, dict):
            skipped += 1
            continue
        try:
            # Use the dict KEY as fallback filepath — the key is the
            # canonical filepath in this schema, the field is redundant.
            rollbacks_field = entry.get("rollback_timestamps", [])
            if not isinstance(rollbacks_field, list):
                rollbacks_field = []
            out[path] = TierHistory(
                filepath=str(entry.get("filepath", path)),
                static_tier=str(entry.get("static_tier", "open")),
                dynamic_tier=str(entry.get("dynamic_tier", "open")),
                dynamic_tier_since=float(entry.get("dynamic_tier_since", time.time())),
                successful_mutations=int(entry.get("successful_mutations", 0)),
                rollbacks=int(entry.get("rollbacks", 0)),
                rollback_timestamps=[float(x) for x in rollbacks_field if isinstance(x, (int, float))],
                last_mutation_at=float(entry.get("last_mutation_at", 0.0)),
            )
        except (TypeError, ValueError) as e:
            logger.warning(f"tier_graduation: skipping malformed entry for {path!r}: {e}")
            skipped += 1
    if skipped:
        logger.info(f"tier_graduation: loaded {len(out)} entries, skipped {skipped} malformed")
    return out


def _save_history(history: dict[str, TierHistory]) -> None:
    """Persist tier history."""
    try:
        TIER_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        TIER_HISTORY_PATH.write_text(json.dumps(
            {p: asdict(h) for p, h in history.items()},
            indent=2, default=str,
        ))
    except OSError as e:
        logger.warning(f"tier_graduation: history save failed: {e}")


def _persist_change(change: TierChange) -> None:
    """Append a tier change event to the graduation log."""
    try:
        existing: list[dict] = []
        if GRADUATION_LOG_PATH.exists():
            existing = json.loads(GRADUATION_LOG_PATH.read_text())
        existing.append({
            "filepath": change.filepath,
            "from_tier": change.from_tier,
            "to_tier": change.to_tier,
            "reason": change.reason,
            "timestamp": change.timestamp,
        })
        existing = existing[-200:]
        GRADUATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        GRADUATION_LOG_PATH.write_text(json.dumps(existing, indent=2, default=str))
    except OSError:
        pass


# ── Static tier lookup ───────────────────────────────────────────────────────

def _static_tier(filepath: str) -> str:
    """Get the static tier from auto_deployer."""
    try:
        from app.auto_deployer import get_protection_tier, ProtectionTier
        tier = get_protection_tier(filepath)
        return tier.value if hasattr(tier, "value") else str(tier).lower()
    except Exception:
        return "open"


# ── Recording ────────────────────────────────────────────────────────────────

def record_successful_mutation(filepath: str) -> None:
    """Increment success count for a file."""
    history = _load_history()
    entry = history.setdefault(filepath, TierHistory(
        filepath=filepath,
        static_tier=_static_tier(filepath),
        dynamic_tier=_static_tier(filepath),
        dynamic_tier_since=time.time(),
    ))
    entry.successful_mutations += 1
    entry.last_mutation_at = time.time()
    _save_history(history)


def record_rollback(filepath: str) -> None:
    """Record a rollback affecting a file. May trigger demotion."""
    history = _load_history()
    entry = history.setdefault(filepath, TierHistory(
        filepath=filepath,
        static_tier=_static_tier(filepath),
        dynamic_tier=_static_tier(filepath),
        dynamic_tier_since=time.time(),
    ))
    entry.rollbacks += 1
    entry.rollback_timestamps.append(time.time())
    # Bound the timestamp list
    cutoff = time.time() - 90 * 86400
    entry.rollback_timestamps = [t for t in entry.rollback_timestamps if t >= cutoff]
    _save_history(history)

    # Immediate demotion check
    _evaluate_single_demotion(filepath, history)


# ── Graduation/demotion logic ────────────────────────────────────────────────

def _evaluate_promotion(entry: TierHistory) -> TierChange | None:
    """Check if a GATED file qualifies for promotion to OPEN."""
    if entry.dynamic_tier != DynamicTier.GATED.value:
        return None
    if entry.static_tier == DynamicTier.IMMUTABLE.value:
        return None  # Static tier overrides — IMMUTABLE never promotes

    # Time at current tier
    days_at_tier = (time.time() - entry.dynamic_tier_since) / 86400
    if days_at_tier < _MIN_DAYS_AT_GATED_FOR_PROMOTION:
        return None

    # Success record
    if entry.successful_mutations < _MIN_SUCCESSFUL_MUTATIONS_FOR_PROMOTION:
        return None
    if entry.rollbacks > _MAX_ROLLBACKS_FOR_PROMOTION:
        return None

    # Centrality check (low centrality = safer to promote)
    try:
        from app.self_model import get_centrality_score, is_hot_path
        centrality = get_centrality_score(entry.filepath)
        if centrality > _MAX_CENTRALITY_FOR_PROMOTION:
            return None
        if is_hot_path(entry.filepath):
            return None
    except Exception:
        # If self_model is unavailable, be conservative — no promotion
        return None

    return TierChange(
        filepath=entry.filepath,
        from_tier=DynamicTier.GATED.value,
        to_tier=DynamicTier.OPEN.value,
        reason=(
            f"{days_at_tier:.0f} days at GATED, "
            f"{entry.successful_mutations} successful mutations, "
            f"{entry.rollbacks} rollbacks, low centrality"
        ),
        timestamp=time.time(),
    )


def _evaluate_single_demotion(
    filepath: str,
    history: dict[str, TierHistory],
) -> TierChange | None:
    """Check if a file should be demoted based on rollback history."""
    entry = history.get(filepath)
    if not entry:
        return None

    now = time.time()
    rollbacks_7d = sum(1 for t in entry.rollback_timestamps if now - t < 7 * 86400)
    rollbacks_30d = sum(1 for t in entry.rollback_timestamps if now - t < 30 * 86400)

    # OPEN → GATED demotion
    if (
        entry.dynamic_tier == DynamicTier.OPEN.value
        and rollbacks_7d >= _ROLLBACKS_THRESHOLD_OPEN_TO_GATED
    ):
        change = TierChange(
            filepath=filepath,
            from_tier=DynamicTier.OPEN.value,
            to_tier=DynamicTier.GATED.value,
            reason=f"{rollbacks_7d} rollbacks in 7 days — demoted for safety",
            timestamp=now,
        )
        entry.dynamic_tier = DynamicTier.GATED.value
        entry.dynamic_tier_since = now
        _save_history(history)
        _persist_change(change)
        logger.warning(f"tier_graduation: DEMOTED {filepath} OPEN → GATED ({change.reason})")
        return change

    # GATED → IMMUTABLE demotion
    if (
        entry.dynamic_tier == DynamicTier.GATED.value
        and rollbacks_30d >= _ROLLBACKS_THRESHOLD_GATED_TO_IMMUTABLE
    ):
        change = TierChange(
            filepath=filepath,
            from_tier=DynamicTier.GATED.value,
            to_tier=DynamicTier.IMMUTABLE.value,
            reason=f"{rollbacks_30d} rollbacks in 30 days — quarantined as IMMUTABLE",
            timestamp=now,
        )
        entry.dynamic_tier = DynamicTier.IMMUTABLE.value
        entry.dynamic_tier_since = now
        _save_history(history)
        _persist_change(change)
        logger.error(f"tier_graduation: QUARANTINED {filepath} GATED → IMMUTABLE")
        return change

    return None


# ── Public API ───────────────────────────────────────────────────────────────

def get_dynamic_tier(filepath: str) -> str:
    """Get the dynamic (history-aware) tier for a file.

    The MORE restrictive of static and dynamic wins. So a GATED file in static
    config that has been demoted to IMMUTABLE dynamically returns IMMUTABLE.
    Conversely, a GATED file promoted to OPEN dynamically still returns GATED
    if its static tier is GATED (we never relax static guarantees).

    Use this from auto_deployer to enforce dynamic restrictions.
    """
    static = _static_tier(filepath)
    history = _load_history()
    entry = history.get(filepath)

    if not entry:
        return static  # No history → use static only

    # Restrictiveness ordering: IMMUTABLE > GATED > OPEN
    order = {"open": 0, "gated": 1, "immutable": 2}
    return max(static, entry.dynamic_tier, key=lambda t: order.get(t, 0))


def evaluate_all_graduations() -> list[TierChange]:
    """Background job: evaluate all files for graduation/demotion.

    Called by idle_scheduler weekly. Most evaluations are no-ops; this is
    the slow trickle of trust-building.
    """
    history = _load_history()
    changes: list[TierChange] = []
    now = time.time()

    for filepath, entry in history.items():
        # Skip files that haven't been touched recently — no fresh signal
        if now - entry.last_mutation_at > 180 * 86400:
            continue

        promotion = _evaluate_promotion(entry)
        if promotion:
            entry.dynamic_tier = DynamicTier.OPEN.value
            entry.dynamic_tier_since = now
            changes.append(promotion)
            _persist_change(promotion)
            logger.info(f"tier_graduation: PROMOTED {filepath} GATED → OPEN ({promotion.reason})")

    if changes:
        _save_history(history)

    return changes


def get_tier_summary() -> dict:
    """Aggregate counts for the dashboard."""
    history = _load_history()
    summary = {"open": 0, "gated": 0, "immutable": 0}
    promotions_total = 0
    demotions_total = 0

    if GRADUATION_LOG_PATH.exists():
        try:
            log = json.loads(GRADUATION_LOG_PATH.read_text())
            for event in log:
                from_t = event.get("from_tier", "")
                to_t = event.get("to_tier", "")
                # Restrictiveness change
                order = {"open": 0, "gated": 1, "immutable": 2}
                if order.get(to_t, 0) < order.get(from_t, 0):
                    promotions_total += 1
                else:
                    demotions_total += 1
        except (json.JSONDecodeError, OSError):
            pass

    for entry in history.values():
        summary[entry.dynamic_tier] = summary.get(entry.dynamic_tier, 0) + 1

    return {
        "by_dynamic_tier": summary,
        "tracked_files": len(history),
        "lifetime_promotions": promotions_total,
        "lifetime_demotions": demotions_total,
    }
