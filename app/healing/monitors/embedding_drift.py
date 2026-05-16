"""embedding_drift — silent embedding-model swap detector (PROGRAM §49 Q14.4).

Closes year-2+ resilience gap §10.4. Distinct from
:mod:`app.healing.llm_output_drift` (which embeds LLM OUTPUTS and
watches their cosine similarity to baseline outputs — drift in
quality of generated answers) and from the §40 embedding-migration
framework (which handles INTENTIONAL embedding-model swaps via
:mod:`app.memory.embedding_migration`).

This monitor watches the EMBEDDING MODEL ITSELF for silent vendor
rotation: the case where Ollama / OpenAI / etc. silently swaps the
weights behind ``nomic-embed-text`` (or whatever's pinned) without
changing the model name. The symptom is invisible in
``llm_output_drift`` because that captures only generative drift.
The symptom is invisible in the migration framework because no
"migration" was ever declared. ChromaDB silently produces different
neighbours; retrieval quality degrades for months.

Algorithm — one weekly pass:

  1. Maintain N=20 canonical anchor queries at
     ``workspace/healing/embedding_anchors.json``. Operator-curated;
     seeded with neutral short strings on first run.
  2. On first run for each anchor, compute its embedding via the
     production embedding function and persist as baseline.
  3. On subsequent runs, re-embed each anchor, compute cosine
     similarity to baseline. ANY anchor whose self-similarity drops
     below ``_DRIFT_THRESHOLD`` (default 0.95) → vendor likely
     swapped weights silently → alert.
  4. Operator may then either: re-baseline (accept the new model)
     or escalate the vendor / pin a specific revision.

Why 0.95? With nomic-embed-text v1.5 the same text embeds to ≈1.0
self-similarity (deterministic). A real model swap drops to ~0.6–0.8
because the representation manifold changes. 0.95 is generous
margin for floating-point noise and minor patch updates.

Master switch: ``embedding_drift_monitor_enabled`` (default ON).
Cadence: weekly. Alert dedup: 7 days per anchor.

Failure-isolated: embedding endpoint unavailable → log skip + no
alerts (we don't want false-positive alerts on transient network
issues).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


NAME = "embedding_drift"
CADENCE_SECONDS = 7 * 24 * 3600
MASTER_SWITCH_KEY = "embedding_drift_monitor_enabled"

_DRIFT_THRESHOLD = 0.95          # any anchor below this → alert
_MIN_BASELINE_PASSES = 1         # baseline accepted after this many passes
_DEDUP_WINDOW_S = 7 * 86400
_STATE_FILE_NAME = "embedding_drift_state.json"
_BASELINE_FILE_NAME = "embedding_anchors.json"

# Seed anchor set — short, neutral, varied to span the manifold.
# Operators may extend by hand-editing workspace/healing/embedding_anchors.json.
_SEED_ANCHORS: tuple[str, ...] = (
    "sky", "ocean", "mountain", "forest",
    "Python 3.13", "the cat sat on the mat",
    "machine learning", "Helsinki winter",
    "operating system", "neural network",
    "quarterly report", "user feedback",
    "exception handling", "JSON parsing",
    "running tests", "logging level",
    "calendar event", "weather forecast",
    "Estonian forest", "Baltic sea",
)


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_embedding_drift_monitor_enabled
        return get_embedding_drift_monitor_enabled()
    except Exception:
        return os.getenv("EMBEDDING_DRIFT_MONITOR_ENABLED", "true").lower() in (
            "true", "1", "yes", "on",
        )


def _workspace() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path("/app/workspace")


def _state_path() -> Path:
    return _workspace() / "healing" / _STATE_FILE_NAME


def _baseline_path() -> Path:
    return _workspace() / "healing" / _BASELINE_FILE_NAME


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run_at": 0.0, "last_alert_at": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run_at": 0.0, "last_alert_at": {}}


def _write_state(state: dict[str, Any]) -> None:
    p = _state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(state, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("embedding_drift: state write failed", exc_info=True)


def _read_baseline() -> dict[str, Any]:
    """Baseline JSON has shape::

        {
          "model_hint": "nomic-embed-text",
          "anchors": [
            {"text": "sky", "vec": [...], "pinned_at": "2026-05-16T..."},
            ...
          ]
        }
    """
    p = _baseline_path()
    if not p.exists():
        return {"model_hint": "", "anchors": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"model_hint": "", "anchors": []}


def _write_baseline(baseline: dict[str, Any]) -> None:
    p = _baseline_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8",
        )
    except Exception:
        logger.debug("embedding_drift: baseline write failed", exc_info=True)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _default_embed(text: str) -> Optional[list[float]]:
    """Production embedding function. Returns None on any failure
    (so the monitor skips that anchor rather than alerting falsely)."""
    try:
        from app.memory.chromadb_manager import embed as _embed
        v = _embed(text)
        if not v:
            return None
        return list(v)
    except Exception:
        logger.debug("embedding_drift: production embed failed", exc_info=True)
        return None


def run(
    *,
    embed_fn: Optional[Callable[[str], Optional[list[float]]]] = None,
    anchor_texts: Optional[tuple[str, ...]] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """One weekly pass. Returns summary dict.

    Test/operator hooks: ``embed_fn`` injects a deterministic embedder;
    ``anchor_texts`` overrides the anchor set; ``now`` for cadence
    determinism."""
    summary: dict[str, Any] = {
        "ran": False,
        "n_anchors": 0,
        "n_baselined": 0,
        "n_diverged": 0,
        "diverged_anchors": [],
        "alerts_fired": 0,
        "baseline_initialized": False,
    }
    if not _enabled():
        summary["skipped"] = True
        return summary

    cur = float(now) if now is not None else time.time()
    state = _read_state()
    # Cadence gate: only apply when there's a recorded prior run. A
    # fresh state file should not trigger the gate (otherwise a small
    # synthetic ``now=`` for tests gets locked out and, more importantly,
    # in production a state-file deletion would silently disable the
    # monitor for a week).
    last_run = float(state.get("last_run_at", 0))
    if last_run > 0 and cur - last_run < CADENCE_SECONDS:
        return summary
    state["last_run_at"] = cur
    summary["ran"] = True

    fn = embed_fn or _default_embed
    anchors = anchor_texts if anchor_texts is not None else _SEED_ANCHORS

    baseline = _read_baseline()
    baseline_by_text: dict[str, list[float]] = {
        a["text"]: a.get("vec") or [] for a in baseline.get("anchors", [])
    }
    summary["n_anchors"] = len(anchors)

    # First-run path: persist baseline. No alerts.
    if not baseline_by_text:
        new_anchors: list[dict[str, Any]] = []
        for text in anchors:
            vec = fn(text)
            if vec is None:
                continue
            new_anchors.append({
                "text": text,
                "vec": vec,
                "pinned_at": datetime.now(timezone.utc).isoformat(),
            })
        if new_anchors:
            _write_baseline({
                "model_hint": "production",
                "anchors": new_anchors,
            })
            summary["baseline_initialized"] = True
            summary["n_baselined"] = len(new_anchors)
        _write_state(state)
        return summary

    # Subsequent run: compare each anchor against baseline.
    diverged: list[dict[str, Any]] = []
    for text in anchors:
        base_vec = baseline_by_text.get(text)
        if base_vec is None:
            # New anchor added to seed list after baseline; skip
            # (baseline-refresh CLI handles this; the monitor doesn't
            # silently expand the baseline because that would mask drift).
            continue
        new_vec = fn(text)
        if new_vec is None:
            continue
        sim = _cosine(base_vec, new_vec)
        if sim < _DRIFT_THRESHOLD:
            diverged.append({
                "text": text,
                "similarity": round(sim, 4),
                "threshold": _DRIFT_THRESHOLD,
            })
    summary["n_diverged"] = len(diverged)
    summary["diverged_anchors"] = diverged

    if diverged:
        # Topic-keyed dedup so transient anchor flakes don't spam.
        last_alerts = state.setdefault("last_alert_at", {})
        if not isinstance(last_alerts, dict):
            last_alerts = {}
            state["last_alert_at"] = last_alerts
        last = float(last_alerts.get("any", 0))
        if cur - last >= _DEDUP_WINDOW_S:
            try:
                from app.notify import notify
                top_3 = diverged[:3]
                body = (
                    f"🔬 Embedding-model drift detected: "
                    f"{len(diverged)} of {len(anchors)} anchor queries "
                    f"dropped below {_DRIFT_THRESHOLD:.2f} self-similarity "
                    f"to baseline.\n\n"
                    "Examples:\n"
                    + "\n".join(
                        f"  • {a['text']!r}: cos={a['similarity']:.3f}"
                        for a in top_3
                    )
                    + "\n\nLikely cause: vendor silently swapped the "
                    "embedding model behind the same name. Either:\n"
                    "  1. Accept new state — re-baseline via the operator "
                    "    CLI (`python -m app.healing.monitors.embedding_drift "
                    "    --rebaseline`).\n"
                    "  2. Pin a specific model revision (Ollama / OpenAI / "
                    "    etc.) and re-baseline.\n"
                    "  3. Migrate via app.memory.embedding_migration framework."
                )
                notify(
                    title="🔬 Embedding-model drift",
                    body=body,
                    url="/cp/health",
                    topic="embedding_model_swap",
                    critical=False,
                    arbitrate=True,
                )
                summary["alerts_fired"] = 1
                last_alerts["any"] = cur
            except Exception:
                logger.debug("embedding_drift: notify failed", exc_info=True)

        # Continuity-ledger landmark — operator should see this in
        # next year's annual reflection drift summary.
        try:
            from app.identity.continuity_ledger import record_event
            record_event(
                kind="embedding_model_swap",
                actor="embedding_drift_monitor",
                summary=(
                    f"Embedding-model drift on {len(diverged)}/{len(anchors)} "
                    f"anchors (threshold {_DRIFT_THRESHOLD})."
                ),
                detail={
                    "n_diverged": len(diverged),
                    "n_total": len(anchors),
                    "top_3": diverged[:3],
                },
            )
        except Exception:
            logger.debug("embedding_drift: ledger emit failed", exc_info=True)

    _write_state(state)
    return summary
