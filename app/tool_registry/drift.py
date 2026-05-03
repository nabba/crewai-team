"""Description-hash drift detection.

A tool's description is what an LLM uses to decide *whether* to call
it. Silently changing the description bypasses code review of the
LLM-facing contract. The drift detector compares each tool's current
in-memory hash against its last persisted snapshot and surfaces:

  - **NEW**: tool registered this boot, no prior snapshot
  - **CHANGED**: hash differs from prior snapshot
  - **REMOVED**: prior snapshot has a tool not in the current registry
  - **TIER_CHANGED**: same tool, different tier (e.g. SHADOW → CANARY)

The detector is **observational, not enforcing** — it logs and emits
a control-plane event. CI gating, if desired, lives in a separate
test that asserts no CHANGED entries against the committed snapshot.

Output is structured for ``/api/cp/tools/drift`` and the error monitor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from app.tool_registry.persistence import load_snapshot
from app.tool_registry.types import ToolSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DriftEntry:
    name: str
    kind: str  # "new" | "changed" | "removed" | "tier_changed"
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "detail": self.detail}


def detect_drift(current: Iterable[ToolSpec]) -> list[DriftEntry]:
    """Compare ``current`` (in-memory registry) against the last
    Postgres snapshot. Returns drift entries.

    On Postgres failure, returns []: no snapshot means we have nothing
    to compare against, so there's no drift to report.
    """
    snapshot = load_snapshot()
    if snapshot is None:
        return []

    snap_by_name = {row["name"]: row for row in snapshot}
    current_by_name = {s.name: s for s in current}

    drift: list[DriftEntry] = []

    for name, spec in current_by_name.items():
        prior = snap_by_name.get(name)
        if prior is None:
            drift.append(DriftEntry(
                name=name, kind="new",
                detail=f"first registration ({spec.tier.value})",
            ))
            continue
        if prior["description_hash"] != spec.description_hash:
            drift.append(DriftEntry(
                name=name, kind="changed",
                detail=(
                    f"description hash {prior['description_hash']} → "
                    f"{spec.description_hash}"
                ),
            ))
        if prior["tier"] != spec.tier.value:
            drift.append(DriftEntry(
                name=name, kind="tier_changed",
                detail=f"{prior['tier']} → {spec.tier.value}",
            ))

    for name, prior in snap_by_name.items():
        if name not in current_by_name:
            drift.append(DriftEntry(
                name=name, kind="removed",
                detail=f"was {prior['tier']} from {prior['source_module']}",
            ))

    return drift


def log_drift(drift: list[DriftEntry]) -> None:
    """Log drift entries at sensible severities + emit error-monitor
    signal for CHANGED + REMOVED (those bypass code review)."""
    if not drift:
        logger.info("tool_registry: no drift detected")
        return

    new = [d for d in drift if d.kind == "new"]
    changed = [d for d in drift if d.kind == "changed"]
    tier_changed = [d for d in drift if d.kind == "tier_changed"]
    removed = [d for d in drift if d.kind == "removed"]

    if new:
        logger.info("tool_registry: %d new tools registered: %s",
                    len(new), [d.name for d in new])
    if tier_changed:
        logger.info("tool_registry: %d tools changed tier: %s",
                    len(tier_changed),
                    [(d.name, d.detail) for d in tier_changed])
    if changed:
        # Description changes bypass LLM-contract review — surface louder.
        logger.warning(
            "tool_registry: %d tools changed description without prior "
            "review: %s. Either commit the change deliberately or revert.",
            len(changed), [d.name for d in changed],
        )
    if removed:
        logger.warning(
            "tool_registry: %d tools removed since last snapshot: %s. "
            "If this is intentional (Forge teardown, refactor), the next "
            "snapshot will reflect it.",
            len(removed), [d.name for d in removed],
        )
