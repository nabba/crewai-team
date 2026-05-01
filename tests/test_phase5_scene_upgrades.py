"""
Phase 5: scene upgrade regression tests (Amendment A + B.5).

Verifies:
  - build_attentional_tiers: top-N focal, next-M peripheral, rest dropped
  - min_salience floor drops low-salience items
  - Peripheral alerts fire for deadlines + conflicts
  - protect_commitment_items force-injects orphaned commitments with
    forced_reason='commitment_orphan' + alert
  - run_strategic_scan groups items by section, filters by commitment
    sections, excludes items already in focal/peripheral
  - build_compact_context produces terse output under token target
  - CIL loop step 3 builds tiers and preserves them on self._tiers
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import MagicMock

# Stub heavy deps so tests run offline.
for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "app.control_plane", "app.control_plane.db",
             "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest

from app.subia.scene.buffer import CompetitiveGate, WorkspaceItem
from app.subia.scene.compact_context import (
    build_compact_context,
    estimate_tokens,
)
from app.subia.scene.strategic_scan import (
    StrategicScanEntry,
    StrategicScanReport,
    format_scan_block,
    run_strategic_scan,
)
from app.subia.scene.tiers import (
    AttentionalTiers,
    PeripheralEntry,
    build_attentional_tiers,
    protect_commitment_items,
)


def _mk_item(
    item_id: str,
    salience: float = 0.5,
    content: str = "content",
    section: str | None = None,
    deadline: str | None = None,
    conflicts: list | None = None,
    content_ref: str | None = None,
) -> WorkspaceItem:
    metadata = {}
    if section:
        metadata["section"] = section
    if deadline:
        metadata["deadline"] = deadline
    item = WorkspaceItem(
        item_id=item_id,
        content=content,
        salience_score=salience,
        source_channel=section or "",
        metadata=metadata,
    )
    # WorkspaceItem doesn't carry conflicts_with natively — attach
    # it as an attribute so tier code (duck-typed) can see it.
    item.conflicts_with = list(conflicts or [])
    if content_ref is not None:
        item.content_ref = content_ref
    return item


@dataclass
class FakeCommitment:
    id: str
    description: str
    venture: str = "archibal"
    status: str = "active"
    deadline: str | None = None
    related_wiki_pages: list = field(default_factory=list)


# ── build_attentional_tiers ──────────────────────────────────────

class TestBuildTiers:
    def test_top_N_go_to_focal(self):
        items = [_mk_item(f"i{i}", salience=1.0 - i * 0.1) for i in range(10)]
        tiers = build_attentional_tiers(items, focal_capacity=3)
        assert len(tiers.focal) == 3
        assert tiers.focal[0].item_id == "i0"
        assert tiers.focal[1].item_id == "i1"
        assert tiers.focal[2].item_id == "i2"

    def test_next_M_go_to_peripheral(self):
        items = [_mk_item(f"i{i}", salience=1.0 - i * 0.1) for i in range(10)]
        tiers = build_attentional_tiers(
            items, focal_capacity=3, peripheral_capacity=4,
        )
        assert len(tiers.peripheral) == 4
        assert tiers.peripheral[0].item_id == "i3"
        assert tiers.peripheral[3].item_id == "i6"

    def test_items_below_min_salience_dropped(self):
        items = [
            _mk_item("high", salience=0.9),
            _mk_item("mid", salience=0.5),
            _mk_item("low", salience=0.05),
        ]
        tiers = build_attentional_tiers(items, min_salience=0.2)
        ids_seen = {i.item_id for i in tiers.focal}
        ids_seen.update(p.item_id for p in tiers.peripheral)
        assert "low" not in ids_seen

    def test_beyond_peripheral_capacity_dropped(self):
        items = [_mk_item(f"i{i}", salience=0.9 - i * 0.01)
                 for i in range(30)]
        tiers = build_attentional_tiers(
            items, focal_capacity=5, peripheral_capacity=5,
        )
        total = len(tiers.focal) + len(tiers.peripheral)
        assert total == 10

    def test_peripheral_entry_has_metadata_only(self):
        items = [_mk_item(f"i{i}", salience=0.5, section="plg")
                 for i in range(5)]
        tiers = build_attentional_tiers(
            items, focal_capacity=1, peripheral_capacity=4,
        )
        assert isinstance(tiers.peripheral[0], PeripheralEntry)
        assert tiers.peripheral[0].section == "plg"
        # No embedding, no ownership, no affect — metadata only.
        assert not hasattr(tiers.peripheral[0], "content_embedding")
        assert not hasattr(tiers.peripheral[0], "ownership")


# ── Peripheral alerts ────────────────────────────────────────────

class TestPeripheralAlerts:
    def test_deadline_in_peripheral_fires_alert(self):
        items = [
            _mk_item("focal1", salience=0.9),
            _mk_item("focal2", salience=0.85),
            _mk_item("deadline_item", salience=0.4,
                     deadline="2026-05-15", section="plg"),
        ]
        tiers = build_attentional_tiers(items, focal_capacity=2)
        assert any("deadline" in a.lower() for a in tiers.peripheral_alerts)
        assert any(
            p.deadline == "2026-05-15" for p in tiers.peripheral
        )

    def test_conflict_in_peripheral_fires_alert(self):
        items = [
            _mk_item("focal1", salience=0.9),
            _mk_item("peripheral1", salience=0.4, conflicts=["focal1"]),
        ]
        tiers = build_attentional_tiers(items, focal_capacity=1)
        assert any(
            "conflict" in a.lower() for a in tiers.peripheral_alerts
        )

    def test_focal_items_dont_fire_peripheral_alerts(self):
        items = [
            _mk_item("f1", salience=0.9, deadline="x"),
            _mk_item("f2", salience=0.85),
        ]
        tiers = build_attentional_tiers(items, focal_capacity=5)
        assert tiers.peripheral_alerts == []


# ── Commitment-orphan protection ─────────────────────────────────

class TestCommitmentProtection:
    def test_represented_commitment_is_not_orphaned(self):
        """A commitment whose wiki page is in the scored list is
        represented — no orphan injection.
        """
        items = [
            _mk_item("i1", salience=0.8,
                     content_ref="archibal/investor-memo.md"),
            _mk_item("i2", salience=0.5),
        ]
        commitments = [
            FakeCommitment(
                id="c1", description="investor memo",
                related_wiki_pages=["archibal/investor-memo.md"],
            ),
        ]
        tiers = build_attentional_tiers(items, focal_capacity=5)
        tiers = protect_commitment_items(
            tiers, items, commitments,
        )
        # No forced entries
        assert not any(
            p.forced_reason == "commitment_orphan"
            for p in tiers.peripheral
        )

    def test_orphaned_commitment_is_force_injected(self):
        """Commitment with no wiki-page match in the scored set gets
        a peripheral placeholder + alert.
        """
        items = [_mk_item("i1", salience=0.8)]
        commitments = [
            FakeCommitment(
                id="c-orphan",
                description="critical regulatory filing",
                venture="plg",
                deadline="2026-05-15",
                related_wiki_pages=["plg/regulatory.md"],
            ),
        ]
        tiers = build_attentional_tiers(items, focal_capacity=5)
        tiers = protect_commitment_items(
            tiers, items, commitments,
        )
        orphans = [p for p in tiers.peripheral
                   if p.forced_reason == "commitment_orphan"]
        assert len(orphans) == 1
        assert "critical regulatory filing" in orphans[0].summary
        assert orphans[0].deadline == "2026-05-15"
        assert orphans[0].section == "plg"
        assert any("ORPHANED COMMITMENT" in a
                   for a in tiers.peripheral_alerts)

    def test_fulfilled_commitment_not_forced(self):
        items = [_mk_item("i1")]
        commitments = [
            FakeCommitment(
                id="c-done", description="done thing",
                status="fulfilled",
                related_wiki_pages=["done/page.md"],
            ),
        ]
        tiers = build_attentional_tiers(items)
        tiers = protect_commitment_items(tiers, items, commitments)
        assert not any(p.forced_reason == "commitment_orphan"
                       for p in tiers.peripheral)

    def test_commitment_with_no_pages_still_protected(self):
        """If a commitment has no related_wiki_pages, we can't match
        anything — treat as orphaned so the deadline surfaces.
        """
        items = [_mk_item("i1")]
        commitments = [
            FakeCommitment(
                id="c-bare",
                description="vague commitment",
                deadline="2026-06-01",
            ),
        ]
        tiers = build_attentional_tiers(items)
        tiers = protect_commitment_items(tiers, items, commitments)
        assert any(p.forced_reason == "commitment_orphan"
                   for p in tiers.peripheral)


# ── Strategic scan ───────────────────────────────────────────────

class TestStrategicScan:
    def test_scan_excludes_focal_and_peripheral_ids(self):
        universe = [
            _mk_item("in_focal", section="archibal"),
            _mk_item("in_peripheral", section="archibal"),
            _mk_item("outside", section="archibal"),
        ]
        report = run_strategic_scan(
            universe,
            focal_ids=["in_focal"],
            peripheral_ids=["in_peripheral"],
        )
        assert report.excluded_in_focal == 1
        assert report.excluded_in_peripheral == 1
        all_ids = {e.item_id for entries in report.by_section.values()
                   for e in entries}
        assert "outside" in all_ids
        assert "in_focal" not in all_ids

    def test_scan_groups_by_section(self):
        universe = [
            _mk_item("a1", section="archibal", salience=0.6),
            _mk_item("a2", section="archibal", salience=0.5),
            _mk_item("k1", section="kaicart", salience=0.4),
        ]
        report = run_strategic_scan(universe)
        assert set(report.by_section.keys()) == {"archibal", "kaicart"}

    def test_scan_filters_to_active_commitment_sections(self):
        universe = [
            _mk_item("a1", section="archibal"),
            _mk_item("k1", section="kaicart"),
            _mk_item("o1", section="other"),
        ]
        commitments = [
            FakeCommitment(id="c1", description="x", venture="archibal"),
            FakeCommitment(id="c2", description="y", venture="kaicart"),
        ]
        report = run_strategic_scan(
            universe, active_commitments=commitments,
        )
        assert "archibal" in report.by_section
        assert "kaicart" in report.by_section
        assert "other" not in report.by_section

    def test_scan_ventures_override(self):
        universe = [
            _mk_item("a1", section="archibal"),
            _mk_item("k1", section="kaicart"),
        ]
        report = run_strategic_scan(universe, ventures=["kaicart"])
        assert "archibal" not in report.by_section
        assert "kaicart" in report.by_section

    def test_scan_token_estimate_under_budget(self):
        """Target per Amendment A.3: ~200 tokens per invocation."""
        universe = [
            _mk_item(f"i{i}", section=f"v{i % 3}", salience=0.5)
            for i in range(30)
        ]
        report = run_strategic_scan(universe)
        assert report.token_estimate < 250

    def test_format_scan_block_readable(self):
        universe = [
            _mk_item("a1", content="Archibal Q2 plan",
                     section="archibal", salience=0.6),
        ]
        report = run_strategic_scan(universe)
        block = format_scan_block(report)
        assert "[strategic_scan]" in block
        assert "archibal" in block
        assert "Archibal Q2 plan" in block

    def test_format_scan_block_empty(self):
        """Empty scan returns a short explicit marker."""
        report = run_strategic_scan([])
        assert "no items" in format_scan_block(report)


# ── Compact context format (Amendment B.5) ──────────────────────

class TestCompactContext:
    def test_empty_inputs_yield_wrapped_empty_block(self):
        out = build_compact_context()
        assert out.startswith("[SubIA]")
        assert out.endswith("[/SubIA]")

    def test_focal_items_rendered(self):
        tiers = AttentionalTiers(
            focal=[_mk_item("i1", salience=0.82, content="Truepic C")],
        )
        out = build_compact_context(tiers=tiers)
        assert "F1:" in out
        assert "Truepic C" in out
        assert "0.82" in out

    def test_peripheral_inlined(self):
        tiers = AttentionalTiers(
            peripheral=[
                PeripheralEntry(summary="PLG plan",
                                 section="plg", salience=0.4),
                PeripheralEntry(summary="TikTok sellers",
                                 section="kaicart", salience=0.3),
            ],
        )
        out = build_compact_context(tiers=tiers)
        assert "PLG plan(plg)" in out
        assert "TikTok sellers(kaicart)" in out

    def test_alerts_rendered(self):
        tiers = AttentionalTiers(
            peripheral_alerts=["deadline: 2026-05-15"],
        )
        out = build_compact_context(tiers=tiers)
        assert "⚠" in out
        assert "2026-05-15" in out

    def test_homeostasis_shows_only_above_threshold(self):
        from app.subia.kernel import HomeostaticState
        h = HomeostaticState(
            deviations={"coherence": 0.05, "progress": -0.3, "safety": 0.35},
        )
        out = build_compact_context(homeostasis=h)
        # 0.05 is below threshold; should not appear
        assert "cohe" not in out.lower()
        # 0.3 and 0.35 should appear (both have abs > 0.2)
        assert "prog" in out.lower()
        assert "safe" in out.lower()

    def test_cached_prediction_not_shown(self):
        """Cached predictions are suppressed to keep the block compact
        (no information added)."""
        from app.subia.kernel import Prediction
        p = Prediction(
            id="p", operation="o", predicted_outcome={},
            predicted_self_change={}, predicted_homeostatic_effect={},
            confidence=0.7, created_at="",
            cached=True,
        )
        out = build_compact_context(prediction=p)
        assert "Pred:" not in out

    def test_live_prediction_is_shown(self):
        from app.subia.kernel import Prediction
        p = Prediction(
            id="p", operation="o", predicted_outcome={"summary": "x"},
            predicted_self_change={}, predicted_homeostatic_effect={},
            confidence=0.7, created_at="",
        )
        out = build_compact_context(prediction=p)
        assert "Pred: conf=0.70" in out

    def test_block_cascade_not_shown_when_maintain(self):
        out = build_compact_context(cascade_recommendation="maintain")
        assert "Cascade:" not in out

    def test_block_cascade_shown_when_escalating(self):
        out = build_compact_context(cascade_recommendation="escalate")
        assert "Cascade: escalate" in out

    def test_dispatch_only_when_not_allow(self):
        from app.subia.belief.dispatch_gate import DispatchDecision
        d = DispatchDecision(verdict="ALLOW", reason="ok")
        out = build_compact_context(dispatch=d)
        assert "Dispatch:" not in out

        d2 = DispatchDecision(verdict="BLOCK", reason="suspended belief")
        out2 = build_compact_context(dispatch=d2)
        assert "Dispatch: BLOCK" in out2
        assert "suspended belief" in out2

    def test_token_target_met(self):
        """Realistic block should be around 100-150 tokens per
        Amendment B.5 target of ~120.
        """
        from app.subia.kernel import HomeostaticState, Prediction
        tiers = AttentionalTiers(
            focal=[
                _mk_item("f1", salience=0.82, content="Truepic Series C"),
                _mk_item("f2", salience=0.71, content="KaiCart API"),
                _mk_item("f3", salience=0.65, content="Cross-venture patterns"),
                _mk_item("f4", salience=0.53, content="Accuracy review"),
                _mk_item("f5", salience=0.48, content="Fundraising pipeline"),
            ],
            peripheral=[
                PeripheralEntry(summary="PLG Q2 planning",
                                 section="plg", salience=0.4),
                PeripheralEntry(summary="Protect Group",
                                 section="plg", salience=0.35),
                PeripheralEntry(summary="TikTok seller tiers",
                                 section="kaicart", salience=0.3),
            ],
            peripheral_alerts=["PLG regulatory filing — 2026-05-15"],
        )
        h = HomeostaticState(
            deviations={"contradiction_pressure": 0.27, "progress": -0.18},
        )
        p = Prediction(
            id="p", operation="o",
            predicted_outcome={"summary": "incremental update expected"},
            predicted_self_change={}, predicted_homeostatic_effect={},
            confidence=0.7, created_at="",
        )
        out = build_compact_context(
            tiers=tiers, homeostasis=h, prediction=p,
        )
        tokens = estimate_tokens(out)
        # Generous upper bound; amendment targets ~120.
        assert tokens < 200, f"compact block too large: {tokens} tokens\n{out}"


# ── Loop integration: _step_attend builds tiers ────────────────

class TestLoopIntegration:
    def test_step_attend_populates_tiers(self):
        from app.subia.kernel import Prediction, SubjectivityKernel
        from app.subia.loop import SubIALoop

        def predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=3),
            predict_fn=predict,
        )
        input_items = [
            _mk_item(f"i{i}", salience=0.9 - i * 0.05)
            for i in range(8)
        ]
        result = loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
            input_items=input_items,
        )
        assert loop._tiers is not None
        assert len(loop._tiers.focal) <= 5
        # Details include focal + peripheral counts
        attend = result.step("3_attend")
        assert attend is not None
        assert "focal" in attend.details
        assert "peripheral" in attend.details

    def test_step_attend_commitment_orphan_injected(self):
        from app.subia.kernel import Prediction, SubjectivityKernel
        from app.subia.kernel import Commitment
        from app.subia.loop import SubIALoop

        def predict(_ctx):
            return Prediction(
                id="p", operation="o", predicted_outcome={},
                predicted_self_change={}, predicted_homeostatic_effect={},
                confidence=0.7, created_at="",
            )

        kernel = SubjectivityKernel()
        kernel.self_state.active_commitments.append(Commitment(
            id="c-orphan", description="regulatory filing",
            venture="plg", created_at="",
            related_wiki_pages=["plg/regulatory.md"],
        ))
        loop = SubIALoop(
            kernel=kernel,
            scene_gate=CompetitiveGate(capacity=3),
            predict_fn=predict,
        )
        # Items that don't touch plg/regulatory.md
        input_items = [
            _mk_item(f"i{i}", salience=0.9 - i * 0.1,
                     content_ref=f"archibal/other-{i}.md")
            for i in range(3)
        ]
        loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
            input_items=input_items,
        )
        assert loop._tiers is not None
        orphans = [
            p for p in loop._tiers.peripheral
            if p.forced_reason == "commitment_orphan"
        ]
        assert len(orphans) == 1
        assert "regulatory filing" in orphans[0].summary

    def test_compact_block_appears_in_context(self):
        from app.subia.kernel import Prediction, SubjectivityKernel
        from app.subia.loop import SubIALoop

        def predict(_ctx):
            return Prediction(
                id="p", operation="o",
                predicted_outcome={"summary": "ok"},
                predicted_self_change={},
                predicted_homeostatic_effect={},
                confidence=0.75, created_at="",
            )

        loop = SubIALoop(
            kernel=SubjectivityKernel(),
            scene_gate=CompetitiveGate(capacity=3),
            predict_fn=predict,
        )
        result = loop.pre_task(
            agent_role="researcher",
            task_description="x",
            operation_type="task_execute",
            input_items=[_mk_item("i1", salience=0.8,
                                   content="Truepic analysis")],
        )
        compact = result.context_for_agent.get("compact", "")
        assert "[SubIA]" in compact
        assert "Truepic analysis" in compact
        assert "[/SubIA]" in compact
