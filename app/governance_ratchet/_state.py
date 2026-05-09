"""State types for the governance-ratchet protocol.

Two thresholds are ratchet-controlled in V1: ``safety_minimum`` and
``quality_minimum`` from ``app/governance.py``. Both are FLOORS — the
ratchet raises them over time as the system earns trust, and (with
the operator's typed confirmation) can also be relaxed back down.

Other governance constants are NOT ratchet-controlled in V1:

  * ``MAX_REGRESSION``      — a CEILING; ratcheting it down (stricter)
    is the right direction but adds API complexity. Future V2.
  * ``MAX_PROMOTIONS_PER_DAY`` — same shape. Future V2.

The hardcoded floor in ``governance.py`` is INVIOLABLE: even a
corrupted ratchet state file can't drop the effective value below it
(``effective_value = max(FLOOR, ratcheted_value)``). Type-level
guarantee.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ThresholdName(str, enum.Enum):
    SAFETY_MINIMUM = "safety_minimum"
    QUALITY_MINIMUM = "quality_minimum"


# ── Direction labels for audit clarity ──────────────────────────────────


class Direction(str, enum.Enum):
    UP = "up"          # ratchet — raises the floor
    DOWN = "down"      # relax — lowers the floor (operator + typed confirm)
    BASELINE = "baseline"  # initial value at first boot


@dataclass
class RatchetEntry:
    """One historical entry for a threshold."""
    ts: str
    direction: Direction
    old_value: float
    new_value: float
    source: str             # "operator_react" | "boot_baseline" | "operator_cli"
    reason: str             # operator-supplied rationale (esp. for relax)
    audit_chain: str = ""   # SHA-256 chain link from ratchet_audit.jsonl

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "direction": self.direction.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "source": self.source,
            "reason": self.reason,
            "audit_chain": self.audit_chain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RatchetEntry":
        return cls(
            ts=d.get("ts", ""),
            direction=Direction(d.get("direction", "baseline")),
            old_value=float(d.get("old_value", 0.0)),
            new_value=float(d.get("new_value", 0.0)),
            source=d.get("source", "unknown"),
            reason=d.get("reason", ""),
            audit_chain=d.get("audit_chain", ""),
        )


@dataclass
class ThresholdState:
    """Current state for one ratchet-controlled threshold."""
    name: str               # ThresholdName.value
    current: float
    history: list[RatchetEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current": self.current,
            "history": [e.to_dict() for e in self.history],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ThresholdState":
        return cls(
            name=d.get("name", ""),
            current=float(d.get("current", 0.0)),
            history=[
                RatchetEntry.from_dict(e) for e in (d.get("history") or [])
            ],
        )


class RatchetViolation(ValueError):
    """Base for protocol violations — direction wrong, below floor, etc."""


class MonotonicViolation(RatchetViolation):
    """``set_ratchet`` rejected because the new value <= current."""


class FloorViolation(RatchetViolation):
    """``relax`` rejected because the new value < FLOOR. The hardcoded
    floor in ``governance.py`` is the post-bootstrap safety contract;
    relax can never drop below it.
    """


class CeilingViolation(RatchetViolation):
    """Sanity ceiling — values must be in [0.0, 1.0]."""


class UnknownThresholdViolation(RatchetViolation):
    """Threshold name not in ``ThresholdName``."""
