"""Cluster LearningGap evidence into architecture-request drafts.

Where :mod:`app.self_improvement.gap_detector` *emits* per-event
``LearningGap`` records (retrieval miss, reflexion failure, low
confidence, user correction, tension, mapelites void, usage decay,
trajectory attribution, observer mis-prediction), this module is the
*consumer* that turns recurring clusters into proposals routed
through the proposal-bridge (``app.proposal_bridge``).

Pipeline (one daily pass):

  1. ``list_open_gaps`` from the LearningGap store — bounded.
  2. Greedy hash-embedding clustering against a similarity threshold.
     Same shape as :mod:`app.companion.lessons_learned`.
  3. Filter clusters to size ≥ ``MIN_CLUSTER_SIZE``.
  4. Compute a stable signature per cluster (sha256 of representative
     evidence). The proposal-bridge dedups by (source, signature) so
     re-running the same cluster is idempotent.
  5. For each surviving cluster, ``proposal_bridge.stage(...)`` the
     architecture-request markdown. After a 7-day cooldown of stable
     content, the bridge files a CR landing the markdown at
     ``docs/proposed_capabilities/<sig>.md`` for permanent record.

This closes the producer side of Piece 2 (architecture-requests):
recurring capability gaps surface as concrete proposals routed
through the human-gated change-request flow, not just buried log
lines. Operator review is preserved at every step — the draft is
*advisory*, not auto-promoted. The ``change_requests.validator``
re-runs at the bridge promotion time, so TIER_IMMUTABLE refusals
still hold.

Daemon-thread pattern mirrors :mod:`app.healing.monitors`: eager
start at import time, env-gated, idempotent restart, watchdog-friendly
liveness check via ``threading.enumerate()``. The pass NEVER raises
into the runtime.

Master switch: ``CAPABILITY_GAP_ANALYZER_ENABLED`` (default ``true``).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass

from app.self_improvement.store import list_open_gaps
from app.self_improvement.types import LearningGap
from app.utils.hash_embedding import cosine, embed

logger = logging.getLogger(__name__)


_DAEMON_THREAD_NAME = "capability-gap-analyzer"
_WARMUP_S = 60
_POLL_INTERVAL_S = 24 * 3600  # daily

_MIN_CLUSTER_SIZE = 3
# Hash-trick embeddings are token-overlap heavy. Empirically:
# topically-related short descriptions land in 0.3–0.6; unrelated
# topics drop below 0.10. 0.30 is the right threshold for capturing
# topic-level clusters without false-merging across domains.
_CLUSTER_SIMILARITY_THRESHOLD = 0.30
_MAX_GAPS_PER_RUN = 500
_SAMPLES_PER_DRAFT = 5

_driver_lock = threading.Lock()
_driver_started = False
_stop_event = threading.Event()


def _enabled() -> bool:
    return os.getenv("CAPABILITY_GAP_ANALYZER_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


def _is_running() -> bool:
    return any(
        t.name == _DAEMON_THREAD_NAME and t.is_alive()
        for t in threading.enumerate()
    )


@dataclass(frozen=True)
class CapabilityCluster:
    signature: str
    label: str
    size: int
    sources: dict[str, int]
    samples: list[str]
    first_seen: str
    last_seen: str


def _signature_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _slug_from_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:30]
    return slug or "capability"


def _cluster_gaps(gaps: list[LearningGap]) -> list[CapabilityCluster]:
    """Greedy single-pass embedding clustering. Returns clusters
    above the size floor, sorted by size descending."""
    centroids: list[list[float]] = []
    members: list[list[LearningGap]] = []

    for gap in gaps:
        text = (gap.description or "").strip()
        if not text:
            continue
        emb = embed(text)
        best_idx, best_sim = -1, -1.0
        for i, c in enumerate(centroids):
            sim = cosine(emb, c)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx >= 0 and best_sim >= _CLUSTER_SIMILARITY_THRESHOLD:
            members[best_idx].append(gap)
            n = len(members[best_idx])
            old = centroids[best_idx]
            # Running mean — preserves cluster centroid as it grows.
            centroids[best_idx] = [(o * (n - 1) + e) / n for o, e in zip(old, emb)]
        else:
            centroids.append(list(emb))
            members.append([gap])

    out: list[CapabilityCluster] = []
    for cgs in members:
        if len(cgs) < _MIN_CLUSTER_SIZE:
            continue
        # Representative text: longest sample (proxy for richest evidence).
        descriptions = [g.description or "" for g in cgs]
        rep = max(descriptions, key=len)
        sig = _signature_for(rep)
        sources = Counter(g.source.value for g in cgs)
        # detected_at is an ISO string; lexicographic sort is chronological.
        firsts = sorted(g.detected_at for g in cgs if g.detected_at)
        first_seen = firsts[0] if firsts else ""
        last_seen = firsts[-1] if firsts else ""
        # Top samples by length, deduped.
        seen: set[str] = set()
        samples: list[str] = []
        for d in sorted(descriptions, key=len, reverse=True):
            if d in seen:
                continue
            seen.add(d)
            samples.append(d)
            if len(samples) >= _SAMPLES_PER_DRAFT:
                break
        out.append(CapabilityCluster(
            signature=sig,
            label=samples[0][:80] if samples else rep[:80],
            size=len(cgs),
            sources=dict(sources),
            samples=samples,
            first_seen=first_seen,
            last_seen=last_seen,
        ))
    out.sort(key=lambda c: c.size, reverse=True)
    return out


def _render_draft(cluster: CapabilityCluster) -> str:
    package_path = f"app/{_slug_from_label(cluster.label)}/"
    sources_str = ", ".join(f"{c} {k}" for k, c in cluster.sources.items())
    samples_md = "\n".join(f"- {s[:240]}" for s in cluster.samples)
    return (
        f"# Capability gap draft — {cluster.label}\n"
        f"\n"
        f"> Auto-generated by `app.self_improvement.capability_gap_analyzer`.  \n"
        f"> Cluster signature: `{cluster.signature}`  \n"
        f"> Cluster size: {cluster.size} evidence item(s)  \n"
        f"> Sources: {sources_str}  \n"
        f"> First seen: {cluster.first_seen}  \n"
        f"> Last seen: {cluster.last_seen}\n"
        f"\n"
        f"## Sample evidence\n"
        f"\n"
        f"{samples_md}\n"
        f"\n"
        f"## Suggested architecture-request draft\n"
        f"\n"
        f"Edit and POST to `/api/cp/architecture-requests`:\n"
        f"\n"
        f"```json\n"
        f"{{\n"
        f'  "intent": "Address capability gap: {cluster.label}",\n'
        f'  "motivation": "{cluster.size} learning-gap signals over the recent window indicate the system lacks coverage for this class of task. Sources: {sources_str}.",\n'
        f'  "package_path": "{package_path}",\n'
        f'  "file_layout": [\n'
        f'    {{ "path": "{package_path}__init__.py", "purpose": "public surface" }},\n'
        f'    {{ "path": "{package_path}core.py",     "purpose": "main implementation" }}\n'
        f"  ],\n"
        f'  "integration_points": [],\n'
        f'  "env_switches": {{}},\n'
        f'  "test_plan": "Cover the {cluster.size} sample-evidence cases above with deterministic tests."\n'
        f"}}\n"
        f"```\n"
        f"\n"
        f"## Operator action\n"
        f"\n"
        f"1. Read the cluster + sample evidence above.\n"
        f"2. Edit the draft JSON to add real `integration_points`, `env_switches`,\n"
        f"   and an honest `test_plan`.\n"
        f"3. POST to `/api/cp/architecture-requests` to file as a proposal.\n"
        f"4. Approve via Signal 👍 or `/cp/architecture-requests`; the standard\n"
        f"   scaffold + per-file change-request flow handles the rest.\n"
        f"\n"
        f"If this gap is already covered by an existing subsystem, delete this\n"
        f"file. Dedup-by-signature prevents re-emitting the same cluster, but\n"
        f"the file system is the durable record.\n"
    )


def run_one_pass(
    *,
    gaps: list[LearningGap] | None = None,
) -> dict:
    """Single analyzer pass. Returns a structured result dict.

    Test hook: ``gaps`` overrides the LearningGap source so unit tests
    don't need a populated store. Proposals are routed through
    ``app.proposal_bridge`` to the canonical
    ``docs/proposed_capabilities/<sig>.md`` target; tests should
    monkeypatch ``PROPOSAL_BRIDGE_DIR`` and inspect via
    ``proposal_bridge.list_proposals(source='capability_gap')``.
    """
    if not _enabled():
        return {"status": "disabled", "drafts_written": 0}

    if gaps is None:
        try:
            gaps = list_open_gaps(limit=_MAX_GAPS_PER_RUN)
        except Exception as exc:  # noqa: BLE001
            logger.warning("capability_gap_analyzer: list_open_gaps failed: %s", exc)
            return {
                "status": "load_failed",
                "drafts_written": 0,
                "error": str(exc),
            }

    if not gaps:
        return {"status": "no_evidence", "drafts_written": 0}

    clusters = _cluster_gaps(gaps)
    if not clusters:
        return {
            "status": "no_clusters",
            "drafts_written": 0,
            "n_evidence": len(gaps),
        }

    try:
        from app.proposal_bridge import stage
    except Exception:
        logger.warning("capability_gap_analyzer: proposal_bridge unavailable",
                       exc_info=True)
        return {
            "status": "bridge_unavailable",
            "drafts_written": 0,
            "n_evidence": len(gaps),
            "n_clusters": len(clusters),
        }

    written = 0
    skipped = 0
    for cluster in clusters:
        target_path = f"docs/proposed_capabilities/{cluster.signature}.md"
        try:
            state, was_new = stage(
                source="capability_gap",
                signature=cluster.signature,
                title=cluster.label[:80] or "capability gap",
                body_markdown=_render_draft(cluster),
                target_path=target_path,
            )
        except Exception:
            logger.warning(
                "capability_gap_analyzer: stage failed for %s",
                cluster.signature, exc_info=True,
            )
            continue
        # ``was_new`` from the bridge tells us whether this call
        # created or replaced the proposal (counts as written) versus
        # a no-op idempotent re-stage (counts as skipped/dedup).
        if was_new:
            written += 1
            logger.info(
                "capability_gap_analyzer: staged %s (size=%d, status=%s)",
                cluster.signature, cluster.size, state.status.value,
            )
        else:
            skipped += 1

    return {
        "status": "ok",
        "n_evidence": len(gaps),
        "n_clusters": len(clusters),
        "drafts_written": written,
        "drafts_skipped_dedup": skipped,
    }


def _driver() -> None:
    if _stop_event.wait(_WARMUP_S):
        return
    while not _stop_event.is_set():
        try:
            result = run_one_pass()
            if result.get("drafts_written", 0) > 0:
                logger.info(
                    "capability_gap_analyzer: %d new draft(s) written",
                    result["drafts_written"],
                )
        except Exception:
            logger.debug("capability_gap_analyzer: pass raised", exc_info=True)
        if _stop_event.wait(_POLL_INTERVAL_S):
            return


def start() -> None:
    """Idempotent daemon launch. Mirrors healing/monitors discipline."""
    global _driver_started
    if not _enabled():
        logger.info(
            "capability_gap_analyzer: disabled via CAPABILITY_GAP_ANALYZER_ENABLED",
        )
        return
    with _driver_lock:
        if _is_running():
            return
        if _driver_started:
            logger.warning(
                "capability_gap_analyzer: previous thread is dead, re-spawning",
            )
        _stop_event.clear()
        thread = threading.Thread(
            target=_driver, name=_DAEMON_THREAD_NAME, daemon=True,
        )
        thread.start()
        _driver_started = True
        logger.info(
            "capability_gap_analyzer: daemon started "
            "(warm-up=%ds, poll=%dh, min-cluster=%d)",
            _WARMUP_S, _POLL_INTERVAL_S // 3600, _MIN_CLUSTER_SIZE,
        )


def stop() -> None:
    _stop_event.set()


# Eager start at import time. The warm-up + thread isolation make this
# safe even when the surrounding process is still booting.
start()
