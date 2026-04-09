"""
sentience_config.py — Externalized configuration for sentience thresholds.

Read by dual_channel, certainty_vector, meta_cognitive.
Written by cogito when applying proposals (bounded, logged, reversible).

SAFETY: Every parameter has min/max bounds. Changes limited to ±20% per cycle.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("/app/workspace/sentience_config.json")

# Safety bounds: (min, max) for each configurable parameter
PARAM_BOUNDS = {
    "certainty_high_threshold": (0.5, 0.95),
    "certainty_low_threshold": (0.2, 0.6),
    "valence_positive_threshold": (0.05, 0.5),
    "valence_negative_threshold": (-0.5, -0.05),
    "slow_path_trigger_threshold": (0.2, 0.6),
    "slow_path_variance_threshold": (0.01, 0.1),
    "reassessment_cooldown_steps": (1, 10),
}

DEFAULTS = {
    "certainty_high_threshold": 0.7,
    "certainty_low_threshold": 0.4,
    "valence_positive_threshold": 0.2,
    "valence_negative_threshold": -0.2,
    "slow_path_trigger_threshold": 0.4,
    "slow_path_variance_threshold": 0.03,
    "reassessment_cooldown_steps": 3,
}

MAX_CHANGE_PER_CYCLE = 0.20  # ±20% per cogito cycle


def load_config() -> dict:
    """Load sentience config from disk, falling back to defaults."""
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text())
            # Merge with defaults (new params get default values)
            merged = dict(DEFAULTS)
            merged.update(data)
            return merged
        except Exception:
            pass
    return dict(DEFAULTS)


def save_config(config: dict) -> None:
    """Atomic write of sentience config to disk."""
    # Validate all values within bounds
    validated = {}
    for key, value in config.items():
        if key in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[key]
            if isinstance(value, (int, float)):
                validated[key] = max(lo, min(hi, value))
            else:
                validated[key] = DEFAULTS.get(key, value)
        else:
            validated[key] = value

    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=CONFIG_PATH.parent, suffix=".tmp", delete=False
        )
        tmp.write(json.dumps(validated, indent=2))
        tmp.close()
        os.replace(tmp.name, str(CONFIG_PATH))
    except Exception as e:
        logger.warning(f"sentience_config: failed to save: {e}")


def propose_change(param: str, new_value: float) -> tuple[bool, str]:
    """Validate a proposed parameter change against safety bounds.

    Returns (approved, reason).
    """
    if param not in PARAM_BOUNDS:
        return False, f"Unknown parameter: {param}"

    current = load_config()
    current_value = current.get(param, DEFAULTS[param])

    lo, hi = PARAM_BOUNDS[param]
    if not (lo <= new_value <= hi):
        return False, f"{param}={new_value} outside bounds [{lo}, {hi}]"

    # Check ±20% change limit
    if current_value != 0:
        change_pct = abs(new_value - current_value) / abs(current_value)
        if change_pct > MAX_CHANGE_PER_CYCLE:
            return False, f"{param} change {change_pct:.0%} exceeds {MAX_CHANGE_PER_CYCLE:.0%} limit"

    return True, "approved"


def apply_change(param: str, new_value: float) -> bool:
    """Apply a validated parameter change. Returns True if applied."""
    approved, reason = propose_change(param, new_value)
    if not approved:
        logger.info(f"sentience_config: rejected {param}={new_value}: {reason}")
        return False

    config = load_config()
    old_value = config.get(param, DEFAULTS[param])
    config[param] = new_value
    save_config(config)

    logger.info(f"sentience_config: {param} changed {old_value} → {new_value}")

    # Log to journal
    try:
        from app.self_awareness.journal import get_journal, JournalEntry, JournalEntryType
        get_journal().write(JournalEntry(
            entry_type=JournalEntryType.CONFIGURATION_CHANGE,
            summary=f"Sentience config: {param} = {old_value} → {new_value}",
            agents_involved=["cogito"],
            details={"param": param, "old": old_value, "new": new_value},
        ))
    except Exception:
        pass

    return True
