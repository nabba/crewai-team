"""
workspace_publish.py — Thin helper for publishing subsystem summaries to
the SubIA Global Workspace.

Consciousness-roadmap §3.G5 (wider GW publisher coverage). The Global
Workspace already exists (Butlin GWT-2/3/4 STRONG); this helper standardizes
the post-hook pattern so each idle reconciler / affect subsystem can publish
a small `(source, content, salience, signal_type)` tuple on completion
without each call site reimplementing the boilerplate.

Why this lives outside `app/subia/`:
  * `app/subia/` is covered by the integrity manifest (Tier-3, see
    `app/subia/integrity.py`); adding a helper file there would require
    manifest regeneration on every change.
  * The helper does not modify the workspace itself — it only CALLS
    `GlobalWorkspace.compete_for_broadcast()`. The IMMUTABLE invariant on
    `app/subia/scene/global_workspace.py` is preserved.

Salience discipline (carried over from GWT-2 buffer.py docstring):
  * Salience floor: `_NOISE_FLOOR = 0.05`. Below this we skip publish
    entirely — saves the workspace from low-signal noise that wouldn't pass
    the 0.3 ignition threshold anyway.
  * Salience ceiling: 1.0 (clamp). Callers should compute salience from
    their own signal magnitude (drift count, deviation, anomaly z-score).
  * Signal types: reuse the 5 canonical types declared in
    `WorkspaceCandidate` (certainty_shift / somatic_flip / trend_reversal /
    free_energy_spike / disposition). New types would require modifying the
    workspace module (IMMUTABLE).

Failure mode: every publish is wrapped in a try/except that logs at DEBUG
and returns. A failed publish never crashes the calling subsystem. This is
the same defensive pattern used by `belief_outbox.py` for Postgres ↔ Neo4j.
"""
from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

_NOISE_FLOOR = 0.05  # publish skipped when salience falls below this

# Type alias for the 5 canonical signal types (mirrors the comment on
# `WorkspaceCandidate.signal_type` in `app/subia/scene/global_workspace.py`).
SignalType = Literal[
    "certainty_shift",
    "somatic_flip",
    "trend_reversal",
    "free_energy_spike",
    "disposition",
]


# ── Public API ───────────────────────────────────────────────────────────


def publish_to_workspace(
    *,
    source: str,
    content: str,
    salience: float,
    signal_type: SignalType,
    truncate_chars: int = 280,
) -> bool:
    """Submit a candidate to the Global Workspace's competitive broadcast.

    Args:
      source: subsystem name, e.g. "decentered-pass", "narrative-chapter",
        "wiki-index-reconciler". Goes into `WorkspaceCandidate.source_agent`
        for downstream attribution.
      content: one-line summary (truncated to `truncate_chars`). Should be
        operator-readable.
      salience: signal magnitude in [0, 1]. Below `_NOISE_FLOOR`, the
        publish is skipped entirely. Below `0.3`, the workspace's ignition
        threshold gates it out — published-but-not-ignited; still useful for
        observability.
      signal_type: one of the 5 canonical types declared on
        `WorkspaceCandidate`.
      truncate_chars: max content length sent to the workspace. Default 280
        matches the workspace's own internal `to_dict()` truncation at 500
        with margin for headers.

    Returns:
      True if the publish was attempted (may or may not have ignited),
      False if it was skipped (below noise floor) or failed (exception
      logged at DEBUG).
    """
    try:
        salience = max(0.0, min(1.0, float(salience)))
    except (TypeError, ValueError):
        logger.debug("workspace_publish: bad salience from %s: %r", source, salience)
        return False

    if salience < _NOISE_FLOOR:
        return False

    if not content:
        return False

    snippet = content if len(content) <= truncate_chars else content[:truncate_chars] + "…"

    try:
        # Late import: avoids a hard dependency at module-load time. The
        # workspace pulls in Postgres bindings at first instance creation;
        # subsystems that publish during boot get a clean error path if the
        # workspace isn't reachable yet.
        from app.subia.scene.global_workspace import (
            GlobalWorkspace,
            WorkspaceCandidate,
        )
        gw = GlobalWorkspace.get_instance()
        gw.compete_for_broadcast([
            WorkspaceCandidate(
                content=snippet,
                salience=salience,
                signal_type=signal_type,
                source_agent=source,
            )
        ])
        return True
    except Exception as exc:
        # Defensive: a workspace publish must never propagate failure into
        # the calling subsystem. The reconciler / consolidator / affect
        # tick continues regardless. DEBUG level keeps the noise off the
        # operator dashboard.
        logger.debug(
            "workspace_publish: %s publish failed: %s",
            source, type(exc).__name__,
        )
        return False


def publish_idle_outcome(
    *,
    source: str,
    signal_type: SignalType,
    counts: object,
    salience_key: str,
    content_template: str,
    salience_floor: float = 0.2,
    salience_per_unit: float = 0.05,
    salience_ceiling: float = 0.7,
) -> bool:
    """Convenience wrapper for idle-scheduler post-hooks.

    Most idle reconcilers return a `dict[str, int]` with counts (e.g.
    `{"synced": 3, "failed": 0}`). This helper scales salience from the
    `salience_key` count and skips when the count is zero (no-op runs
    are not worth publishing).

    Salience formula:
        if count <= 0:               skip
        else:                        clamp(floor + count * per_unit, 0, ceiling)
    """
    if not isinstance(counts, dict):
        return False
    try:
        magnitude = int(counts.get(salience_key, 0) or 0)
    except (TypeError, ValueError):
        return False
    if magnitude <= 0:
        return False

    salience = min(salience_floor + magnitude * salience_per_unit, salience_ceiling)

    try:
        content = content_template.format(**counts)
    except (KeyError, IndexError, ValueError):
        # Don't crash on a malformed template — report what we have.
        content = f"{source}: {counts}"

    return publish_to_workspace(
        source=source,
        content=content,
        salience=salience,
        signal_type=signal_type,
    )
