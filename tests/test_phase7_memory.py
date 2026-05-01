"""
Phase 7: dual-tier memory regression tests.

Four subsystems:

  1. consolidator — significance formula + selective write (always
     full, threshold curated) + Neo4j relations.

  2. dual_tier memory access — recall/recall_deep/recall_around/
     find_overlooked/promote_to_curated with duck-typed clients.

  3. spontaneous memory surfacing — curated-only associative lookup
     that produces SceneItem candidates.

  4. retrospective promotion — below-threshold records get promoted
     when wiki state or accuracy_tracker signals fire.

Plus loop integration: _step_consolidate writes to attached clients.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)

import pytest

from app.subia.kernel import (
    Commitment,
    HomeostaticState,
    Prediction,
    SceneItem,
    SubjectivityKernel,
)
from app.subia.memory.consolidator import (
    ConsolidationResult,
    build_enriched_episode,
    build_lightweight_record,
    compute_episode_significance,
    consolidate,
    extract_relations,
)
from app.subia.memory.dual_tier import DualTierMemoryAccess
from app.subia.memory.retrospective import (
    RetrospectiveReport,
    retrospective_review,
)
from app.subia.memory.spontaneous import check_spontaneous_memories


# ── In-memory fake clients ────────────────────────────────────────

class FakeMemoryClient:
    """Duck-typed stand-in for a Mem0 client."""

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self._next_id = 0

    def add(self, record: dict) -> str:
        self._next_id += 1
        rec_id = f"rec-{self._next_id}"
        # Stash a copy so caller mutations don't leak
        self.records[rec_id] = dict(record)
        self.records[rec_id]["id"] = rec_id
        return rec_id

    def get(self, rec_id: str) -> dict | None:
        return self.records.get(rec_id)

    def update(self, rec_id: str, record: dict) -> None:
        if rec_id in self.records:
            self.records[rec_id] = {**record, "id": rec_id}

    def search(self, query: str, limit: int = 10) -> list[dict]:
        # Dumb substring + score; returns top-N hits.
        hits = []
        q_lower = query.lower()
        # Temporal-style queries ("recent experiences", "experiences
        # around ...") are treated as "return all, newest first" so
        # retrospective_review/find_overlooked see every record.
        temporal_markers = ("recent", "experience", "around")
        is_temporal = any(m in q_lower for m in temporal_markers)
        for r in self.records.values():
            haystack = " ".join(
                str(r.get(k, "")) for k in
                ("result_summary", "summary", "operation", "content")
            ).lower()
            overlap = sum(
                1 for tok in q_lower.split()
                if len(tok) > 2 and tok in haystack
            )
            if overlap > 0:
                out = dict(r)
                out["similarity_score"] = min(1.0, 0.5 + 0.2 * overlap)
                hits.append(out)
            elif is_temporal:
                out = dict(r)
                out["similarity_score"] = 0.0
                hits.append(out)
        hits.sort(key=lambda r: r["similarity_score"], reverse=True)
        return hits[:limit]


class FakeNeo4j:
    def __init__(self) -> None:
        self.relations: list[dict] = []

    def add_relation(self, rel: dict) -> None:
        self.relations.append(dict(rel))


# ── Fixtures ─────────────────────────────────────────────────────

def _rich_kernel() -> SubjectivityKernel:
    k = SubjectivityKernel(loop_count=10, last_loop_at="2026-04-13T10:00:00+00:00")
    k.scene.append(SceneItem(
        id="s1", source="wiki", content_ref="archibal/landscape.md",
        summary="Truepic Series C analysis", salience=0.8,
        entered_at="", dominant_affect="urgency",
    ))
    k.scene.append(SceneItem(
        id="s2", source="wiki", content_ref="kaicart/api.md",
        summary="KaiCart API constraints", salience=0.6,
        entered_at="",
    ))
    k.homeostasis = HomeostaticState(
        variables={"coherence": 0.6, "progress": 0.4},
        set_points={"coherence": 0.5, "progress": 0.55},
        deviations={"coherence": 0.1, "progress": -0.4, "safety": 0.25},
        restoration_queue=["progress", "safety"],
    )
    k.predictions.append(Prediction(
        id="p1", operation="researcher:ingest",
        predicted_outcome={}, predicted_self_change={},
        predicted_homeostatic_effect={},
        confidence=0.7, created_at="",
        resolved=True, prediction_error=0.6,
    ))
    k.self_state.active_commitments.append(Commitment(
        id="c1", description="investor memo", venture="archibal",
        created_at="",
        related_wiki_pages=["archibal/memo.md"],
    ))
    return k


# ── Consolidator: significance formula ───────────────────────────

class TestSignificanceFormula:
    def test_empty_kernel_gives_zero(self):
        k = SubjectivityKernel()
        s = compute_episode_significance(k, {})
        assert s == 0.0

    def test_high_salience_high_error_high_dev_scores_high(self):
        k = _rich_kernel()
        s = compute_episode_significance(k, {"summary": "x"})
        # Rich kernel has: avg_salience≈0.7, |pe|=0.6,
        # mean|dev|≈0.25, 1 active commitment → min(1,0.2)
        # Hand calc: 0.3*0.7 + 0.3*0.6 + 0.2*0.25 + 0.2*0.2 = 0.48
        assert 0.40 <= s <= 0.60

    def test_score_clamped_to_unit_interval(self):
        """Even with crazy-large deviations the score stays in [0,1]."""
        k = SubjectivityKernel()
        k.homeostasis = HomeostaticState(deviations={"x": 10.0})
        # Will still be <= 1.0 due to clamp
        s = compute_episode_significance(k, {})
        assert 0.0 <= s <= 1.0

    def test_custom_weights(self):
        k = _rich_kernel()
        baseline = compute_episode_significance(k, {})
        zero_weights = compute_episode_significance(
            k, {}, weights={"salience": 0, "prediction_error": 0,
                            "homeostatic": 0, "commitment": 0},
        )
        assert zero_weights == 0.0
        assert baseline > 0.0


# ── Consolidator: record shapes ──────────────────────────────────

class TestRecordShapes:
    def test_lightweight_record(self):
        k = _rich_kernel()
        r = build_lightweight_record(
            k, {"summary": "done"}, "researcher", "ingest", 0.75,
        )
        assert r["type"] == "full_record"
        assert r["agent"] == "researcher"
        assert r["operation"] == "ingest"
        assert r["significance"] == 0.75
        assert r["promoted_to_curated"] is True   # above threshold
        assert isinstance(r["scene_topics"], list)
        assert r["prediction_error"] == 0.6

    def test_enriched_episode(self):
        k = _rich_kernel()
        r = build_enriched_episode(
            k, {"summary": "done", "wiki_pages_affected": ["a.md"]},
            "researcher", "ingest", 0.8,
        )
        assert r["type"] == "curated_episode"
        assert len(r["scene_snapshot"]) == 2
        assert r["scene_snapshot"][0]["affect"] == "urgency"
        assert r["prediction"]["error"] == 0.6
        assert r["self_state_snapshot"]["active_commitments"] == 1
        assert r["wiki_pages_affected"] == ["a.md"]
        # Full homeostatic state for every configured variable
        assert "coherence" in r["homeostatic_state"]

    def test_relations_ownership_and_causation(self):
        k = _rich_kernel()
        rels = extract_relations(k, {
            "wiki_pages_created": ["archibal/new.md"],
            "summary": "big surprise",
        })
        assert any(r["type"] == "OWNED_BY" for r in rels)
        assert any(r["type"] == "CAUSED_STATE_CHANGE" for r in rels)


# ── Consolidator: dual-tier write logic ─────────────────────────

class TestConsolidateWritePath:
    def test_always_writes_full_when_attached(self):
        k = _rich_kernel()
        full = FakeMemoryClient()
        out = consolidate(
            k, {"summary": "x"}, "researcher", "ingest",
            mem0_full=full,
        )
        assert out.wrote_full is True
        assert len(full.records) == 1

    def test_curated_only_above_threshold(self):
        k = _rich_kernel()  # sig≈0.48 — below default 0.5
        full = FakeMemoryClient()
        curated = FakeMemoryClient()
        out = consolidate(
            k, {"summary": "x"}, "researcher", "ingest",
            mem0_full=full, mem0_curated=curated,
        )
        assert out.wrote_full is True
        assert out.wrote_curated is False
        assert len(curated.records) == 0

    def test_curated_writes_when_above_threshold(self):
        k = _rich_kernel()
        full = FakeMemoryClient()
        curated = FakeMemoryClient()
        # Force above-threshold with a very low explicit threshold.
        out = consolidate(
            k, {"summary": "x"}, "researcher", "ingest",
            mem0_full=full, mem0_curated=curated,
            episode_threshold=0.1,
        )
        assert out.wrote_curated is True
        assert len(curated.records) == 1

    def test_neo4j_only_with_curated(self):
        """Neo4j relations are curated-tier only — no relations when
        significance is too low to promote.
        """
        k = _rich_kernel()
        full = FakeMemoryClient()
        curated = FakeMemoryClient()
        neo = FakeNeo4j()
        # Force below-threshold
        out = consolidate(
            k, {"summary": "x", "wiki_pages_created": ["a.md"]},
            "researcher", "ingest",
            mem0_full=full, mem0_curated=curated, neo4j_client=neo,
            episode_threshold=0.99,
        )
        assert out.wrote_curated is False
        assert out.relations_written == 0
        assert len(neo.relations) == 0

    def test_neo4j_with_curated_writes_relations(self):
        k = _rich_kernel()
        full = FakeMemoryClient()
        curated = FakeMemoryClient()
        neo = FakeNeo4j()
        out = consolidate(
            k, {"summary": "x", "wiki_pages_created": ["a.md"]},
            "researcher", "ingest",
            mem0_full=full, mem0_curated=curated, neo4j_client=neo,
            episode_threshold=0.1,
        )
        assert out.wrote_curated is True
        assert out.relations_written > 0
        assert len(neo.relations) == out.relations_written

    def test_no_client_graceful_noop(self):
        out = consolidate(SubjectivityKernel(), {}, "x", "y")
        assert isinstance(out, ConsolidationResult)
        assert out.wrote_full is False
        assert out.wrote_curated is False

    def test_failing_client_does_not_crash(self):
        k = _rich_kernel()

        class ExplodingClient:
            def add(self, r):
                raise RuntimeError("kaboom")

        out = consolidate(
            k, {"summary": "x"}, "r", "i",
            mem0_full=ExplodingClient(),
        )
        # No exception propagated; wrote_full stays False because add failed
        assert out.wrote_full is False


# ── DualTierMemoryAccess ─────────────────────────────────────────

class TestDualTierAccess:
    def _setup(self) -> DualTierMemoryAccess:
        curated = FakeMemoryClient()
        full = FakeMemoryClient()
        curated.add({"result_summary": "curated API retry logic",
                     "loop_count": 5})
        # Same loop_count as curated → should be deduped on recall_deep.
        full.add({"result_summary": "full API retry minor log",
                  "loop_count": 5, "significance": 0.4})
        # Different loop_count → should survive dedup and appear.
        full.add({"result_summary": "full API tangent note",
                  "loop_count": 9, "significance": 0.35})
        full.add({"result_summary": "full deployment note",
                  "loop_count": 7, "significance": 0.35})
        return DualTierMemoryAccess(mem0_curated=curated, mem0_full=full)

    def test_recall_returns_only_curated(self):
        m = self._setup()
        hits = m.recall("API retry")
        assert all(h["_memory_tier"] == "curated" for h in hits)
        assert any("curated" in str(h.get("result_summary", ""))
                   for h in hits)

    def test_recall_deep_returns_both_deduped(self):
        m = self._setup()
        hits = m.recall_deep("API")
        tiers = {h["_memory_tier"] for h in hits}
        assert tiers == {"curated", "full"}
        # Loop 5 collision: curated wins, full-tier loop 5 excluded
        loop_counts_by_tier = {
            (h["_memory_tier"], h.get("loop_count")) for h in hits
        }
        # curated has loop_count=5 — full-tier loop_count=5 must NOT appear
        assert ("full", 5) not in loop_counts_by_tier

    def test_find_overlooked_skips_promoted(self):
        curated = FakeMemoryClient()
        full = FakeMemoryClient()
        full.add({"result_summary": "A", "significance": 0.4,
                  "promoted_to_curated": False})
        full.add({"result_summary": "B", "significance": 0.4,
                  "promoted_to_curated": True})   # already promoted
        m = DualTierMemoryAccess(mem0_curated=curated, mem0_full=full)
        overlooked = m.find_overlooked()
        ids = [r.get("id") for r in overlooked]
        # Only the unpromoted one returned
        assert len(ids) == 1

    def test_promote_to_curated_writes_and_marks(self):
        m = self._setup()
        # Pick a known full-tier record to promote
        full_records = list(m.full.records.values())
        assert full_records
        target_id = full_records[0]["id"]
        ok = m.promote_to_curated(target_id, reason="test-case")
        assert ok
        # Curated now has the promoted record
        promoted = [
            r for r in m.curated.records.values()
            if r.get("type") == "promoted_episode"
        ]
        assert len(promoted) == 1
        assert promoted[0]["promoted_reason"] == "test-case"
        # Full-tier record flagged
        assert m.full.records[target_id]["promoted_to_curated"] is True

    def test_recall_with_no_client_empty(self):
        m = DualTierMemoryAccess()
        assert m.recall("x") == []
        assert m.recall_deep("x") == []

    def test_broken_client_graceful(self):
        class Broken:
            def search(self, q, limit=10):
                raise RuntimeError("down")
        m = DualTierMemoryAccess(
            mem0_curated=Broken(), mem0_full=Broken(),
        )
        assert m.recall("anything") == []


# ── Spontaneous surfacing ────────────────────────────────────────

class TestSpontaneousSurfacing:
    def test_high_relevance_surfaces(self):
        curated = FakeMemoryClient()
        curated.add({"result_summary": "investor relations memo",
                     "loop_count": 1})
        items = check_spontaneous_memories(
            ["investor memo"], curated, threshold=0.5,
        )
        assert len(items) > 0
        assert items[0].source == "memory"
        assert items[0].summary.startswith("[Memory]")
        assert items[0].content_ref.startswith("mem0_curated:")

    def test_low_relevance_does_not_surface(self):
        curated = FakeMemoryClient()
        curated.add({"result_summary": "completely unrelated",
                     "loop_count": 1})
        items = check_spontaneous_memories(
            ["investor memo"], curated, threshold=0.7,
        )
        # "completely unrelated" has no token overlap → no hit
        assert items == []

    def test_damping_keeps_live_edge(self):
        curated = FakeMemoryClient()
        curated.add({"result_summary": "exactly matching topic",
                     "similarity_score": 0.9, "loop_count": 1})
        items = check_spontaneous_memories(
            ["exactly matching topic"], curated, threshold=0.5,
        )
        # At least one memory surfaces
        assert items
        # Salience < 1.0 thanks to 0.7 damping
        assert all(i.salience < 1.0 for i in items)

    def test_no_client_returns_empty(self):
        assert check_spontaneous_memories(["x"], None) == []

    def test_no_topics_returns_empty(self):
        curated = FakeMemoryClient()
        curated.add({"result_summary": "X"})
        assert check_spontaneous_memories([], curated) == []


# ── Retrospective review ─────────────────────────────────────────

class TestRetrospectiveReview:
    def _make(self) -> DualTierMemoryAccess:
        curated = FakeMemoryClient()
        full = FakeMemoryClient()
        full.add({
            "result_summary": "API update noted 3 weeks ago",
            "significance": 0.35,
            "promoted_to_curated": False,
            "agent": "researcher",
            "operation": "ingest",
        })
        full.add({
            "result_summary": "unrelated note",
            "significance": 0.35,
            "promoted_to_curated": False,
            "agent": "researcher",
            "operation": "lint",
        })
        return DualTierMemoryAccess(mem0_curated=curated, mem0_full=full)

    def test_wiki_presence_triggers_promotion(self):
        m = self._make()

        def wiki_search(topic):
            return "API" in topic

        report = retrospective_review(
            memory_access=m, wiki_search=wiki_search,
        )
        assert isinstance(report, RetrospectiveReport)
        assert report.promoted >= 1
        # The API-related record was promoted; unrelated one wasn't
        ids = report.promoted_ids
        assert len(ids) == 1

    def test_sustained_error_triggers_promotion(self):
        m = self._make()

        class FakeTracker:
            def has_sustained_error(self, domain):
                return domain == "researcher:ingest"

        report = retrospective_review(
            memory_access=m, accuracy_tracker=FakeTracker(),
        )
        # Researcher:ingest has sustained error → that record promoted
        # Researcher:lint does not → its record stays
        assert report.promoted == 1
        assert "sustained" in list(report.reasons.values())[0].lower()

    def test_no_signal_no_promotion(self):
        m = self._make()
        report = retrospective_review(memory_access=m)
        assert report.promoted == 0
        assert report.candidates_reviewed >= 1

    def test_cap_max_promotions(self):
        """Even with everything qualifying, promotions are capped."""
        curated = FakeMemoryClient()
        full = FakeMemoryClient()
        for i in range(20):
            full.add({
                "result_summary": f"record {i}",
                "significance": 0.4,
                "agent": "r", "operation": "i",
            })
        m = DualTierMemoryAccess(mem0_curated=curated, mem0_full=full)

        def always_hit(topic):
            return True

        report = retrospective_review(
            memory_access=m, wiki_search=always_hit,
            max_promotions=5,
        )
        assert report.promoted == 5

    def test_non_dual_tier_graceful(self):
        class Broken:
            pass
        report = retrospective_review(memory_access=Broken())
        assert report.promoted == 0


# ── Loop integration ────────────────────────────────────────────

class TestLoopIntegration:
    def _mk_loop(self, full=None, curated=None):
        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate

        def predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        return SubIALoop(
            kernel=_rich_kernel(),
            scene_gate=CompetitiveGate(capacity=5),
            predict_fn=predict,
            mem0_full=full,
            mem0_curated=curated,
        )

    def test_post_task_writes_to_full_tier(self):
        full = FakeMemoryClient()
        loop = self._mk_loop(full=full)
        loop.post_task(
            agent_role="researcher", task_description="x",
            operation_type="task_execute",
            task_result={"summary": "done"},
        )
        # At least one full-tier write occurred
        assert len(full.records) >= 1

    def test_post_task_honors_threshold(self):
        """Use a low-significance kernel; curated should not be written."""
        from app.subia.loop import SubIALoop
        from app.subia.scene.buffer import CompetitiveGate

        def predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        full = FakeMemoryClient()
        curated = FakeMemoryClient()
        loop = SubIALoop(
            kernel=SubjectivityKernel(),   # minimal, low-significance
            scene_gate=CompetitiveGate(capacity=5),
            predict_fn=predict,
            mem0_full=full,
            mem0_curated=curated,
        )
        loop.post_task(
            agent_role="x", task_description="x",
            operation_type="task_execute",
            task_result={"summary": "s"},
        )
        assert len(full.records) == 1
        assert len(curated.records) == 0

    def test_no_clients_still_works(self):
        loop = self._mk_loop(full=None, curated=None)
        result = loop.post_task(
            agent_role="r", task_description="x",
            operation_type="task_execute",
            task_result={"summary": "done"},
        )
        # _step_consolidate still ran, step was marked OK
        step = result.step("10_consolidate")
        assert step is not None
        assert step.ok
