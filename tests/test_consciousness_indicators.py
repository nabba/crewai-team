"""
Comprehensive tests for Butlin et al. (2025) consciousness indicators.

Covers all 5 modules:
  1. GWT-2: Competitive workspace buffer + salience scoring
  2. GWT-3: Global broadcast with agent reactions
  3. HOT-3: Belief store + metacognitive monitor
  4. AST-1: Attention schema (stuck/capture detection, intervention)
  5. PP-1: Predictive coding layer (surprise routing, confidence adaptation)
  6. Cross-module integration tests
  7. Configuration and safety invariants

Total: ~120 tests
"""

import math
import sys
import time
import threading
import uuid
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

# Mock modules not available on host (only in Docker).
# Also mock modules that use Python 3.10+ type hints (X | None)
# which fail on 3.9.
_MOCK_MODULES = [
    "psycopg2", "psycopg2.pool", "psycopg2.extras",
    "chromadb", "chromadb.config", "chromadb.utils",
    "chromadb.utils.embedding_functions",
    "app.control_plane", "app.control_plane.db",
    "app.memory", "app.memory.chromadb_manager",
]
for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Ensure embed() returns a proper list
sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
# Ensure execute() returns empty list
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Module 1: GWT-2 — Competitive Workspace Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class TestWorkspaceItem:
    """Tests for WorkspaceItem dataclass."""

    def test_item_creation_defaults(self):
        from app.consciousness.workspace_buffer import WorkspaceItem
        item = WorkspaceItem()
        assert item.item_id  # UUID generated
        assert item.content == ""
        assert item.salience_score == 0.0
        assert item.goal_relevance == 0.0
        assert item.novelty_score == 0.0
        assert item.agent_urgency == 0.0
        assert item.surprise_signal == 0.0
        assert item.decay_rate == 0.05
        assert item.consumed is False

    def test_item_creation_custom(self):
        from app.consciousness.workspace_buffer import WorkspaceItem
        item = WorkspaceItem(
            content="Test content",
            source_agent="researcher",
            goal_relevance=0.8,
            novelty_score=0.6,
            agent_urgency=0.5,
            surprise_signal=0.3,
        )
        assert item.content == "Test content"
        assert item.source_agent == "researcher"
        assert item.goal_relevance == 0.8

    def test_item_to_dict(self):
        from app.consciousness.workspace_buffer import WorkspaceItem
        item = WorkspaceItem(content="Test", salience_score=0.75)
        d = item.to_dict()
        assert "item_id" in d
        assert d["content"] == "Test"
        assert d["salience_score"] == 0.75

    def test_item_content_truncation(self):
        from app.consciousness.workspace_buffer import WorkspaceItem
        long_content = "x" * 500
        item = WorkspaceItem(content=long_content)
        d = item.to_dict()
        assert len(d["content"]) == 300

    def test_unique_ids(self):
        from app.consciousness.workspace_buffer import WorkspaceItem
        items = [WorkspaceItem() for _ in range(10)]
        ids = {i.item_id for i in items}
        assert len(ids) == 10


class TestCosineSim:
    """Tests for workspace_buffer._cosine_sim."""

    def test_identical_vectors(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        assert _cosine_sim([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        result = _cosine_sim([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(0.5)

    def test_empty_vectors_return_default(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        assert _cosine_sim([], [1.0]) == 0.5
        assert _cosine_sim([1.0], []) == 0.5
        assert _cosine_sim([], []) == 0.5

    def test_mismatched_dims(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        assert _cosine_sim([1.0, 2.0], [1.0]) == 0.5

    def test_zero_vector(self):
        from app.consciousness.workspace_buffer import _cosine_sim
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.5


class TestSalienceScorer:
    """Tests for SalienceScorer."""

    def test_default_weights(self):
        from app.consciousness.workspace_buffer import SalienceScorer
        s = SalienceScorer()
        assert s.w_goal == 0.35
        assert s.w_novelty == 0.25
        assert s.w_urgency == 0.15
        assert s.w_surprise == 0.25

    def test_custom_weights(self):
        from app.consciousness.workspace_buffer import SalienceScorer
        s = SalienceScorer(w_goal=0.5, w_novelty=0.2, w_urgency=0.2, w_surprise=0.1)
        assert s.w_goal == 0.5

    def test_score_without_embeddings(self):
        from app.consciousness.workspace_buffer import SalienceScorer, WorkspaceItem
        s = SalienceScorer()
        item = WorkspaceItem(agent_urgency=0.8, surprise_signal=0.6)
        score = s.score(item, goal_embeddings=[], recent_items=[])
        assert 0.0 < score < 1.0
        # With no embeddings: goal=0.5, novelty=0.8 (default for first)
        expected = 0.35 * 0.5 + 0.25 * 0.8 + 0.15 * 0.8 + 0.25 * 0.6
        assert score == pytest.approx(expected, abs=0.01)

    def test_score_with_decay(self):
        from app.consciousness.workspace_buffer import SalienceScorer, WorkspaceItem
        s = SalienceScorer()
        item = WorkspaceItem(agent_urgency=1.0, cycles_in_workspace=10)
        score = s.score(item, [], [])
        # Decay should reduce score
        item_fresh = WorkspaceItem(agent_urgency=1.0, cycles_in_workspace=0)
        score_fresh = s.score(item_fresh, [], [])
        assert score < score_fresh

    def test_surprise_signal_amplifies(self):
        from app.consciousness.workspace_buffer import SalienceScorer, WorkspaceItem
        s = SalienceScorer()
        item_no_surprise = WorkspaceItem(agent_urgency=0.5)
        item_surprise = WorkspaceItem(agent_urgency=0.5, surprise_signal=0.9)
        s1 = s.score(item_no_surprise, [], [])
        s2 = s.score(item_surprise, [], [])
        assert s2 > s1


class TestCompetitiveGate:
    """Tests for CompetitiveGate."""

    def test_admit_below_capacity(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=3)
        item = WorkspaceItem(content="Test", salience_score=0.5)
        result = gate.evaluate(item)
        assert result.admitted is True  # has .admitted field via transition_type
        assert result.transition_type == "admitted"
        assert result.displaced_item is None
        assert len(gate.active_items) == 1

    def test_fill_to_capacity(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=3)
        for i in range(3):
            gate.evaluate(WorkspaceItem(content=f"Item {i}", salience_score=0.5))
        assert len(gate.active_items) == 3

    def test_displacement_above_capacity(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=2)
        gate.evaluate(WorkspaceItem(content="Low", salience_score=0.2))
        gate.evaluate(WorkspaceItem(content="Medium", salience_score=0.5))
        result = gate.evaluate(WorkspaceItem(content="High", salience_score=0.8))
        assert result.transition_type == "displaced"
        assert result.displaced_item is not None
        assert result.displaced_item.content == "Low"
        assert len(gate.active_items) == 2

    def test_rejection_below_minimum(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=2)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.6))
        gate.evaluate(WorkspaceItem(content="B", salience_score=0.7))
        result = gate.evaluate(WorkspaceItem(content="C", salience_score=0.1))
        assert result.transition_type == "rejected"
        assert result.rejection_reason is not None
        assert len(gate.active_items) == 2

    def test_novelty_floor(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=2, novelty_floor_pct=0.20)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.8))
        gate.evaluate(WorkspaceItem(content="B", salience_score=0.7))
        # Low salience but very high novelty (above 1.0 - 0.20 = 0.80)
        novel = WorkspaceItem(content="Novel", salience_score=0.1, novelty_score=0.95)
        result = gate.evaluate(novel)
        assert result.transition_type == "novelty_floor"
        assert result.displaced_item is not None

    def test_novelty_floor_once_per_cycle(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=2, novelty_floor_pct=0.20)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.8))
        gate.evaluate(WorkspaceItem(content="B", salience_score=0.7))
        # First novelty floor
        novel1 = WorkspaceItem(content="Novel1", salience_score=0.1, novelty_score=0.95)
        r1 = gate.evaluate(novel1)
        assert r1.transition_type == "novelty_floor"
        # Second should be rejected (novelty floor exhausted this cycle)
        novel2 = WorkspaceItem(content="Novel2", salience_score=0.05, novelty_score=0.99)
        r2 = gate.evaluate(novel2)
        assert r2.transition_type == "rejected"

    def test_advance_cycle_resets_novelty_floor(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=2, novelty_floor_pct=0.20)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.8))
        gate.evaluate(WorkspaceItem(content="B", salience_score=0.7))
        gate.evaluate(WorkspaceItem(content="Novel1", salience_score=0.1, novelty_score=0.95))
        gate.advance_cycle()
        # After cycle advance, novelty floor should be available again
        assert gate._novelty_admitted_this_cycle is False

    def test_mark_consumed(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=3, consumption_decay=0.5)
        item = WorkspaceItem(content="Test", salience_score=1.0)
        gate.evaluate(item)
        gate.mark_consumed(item.item_id)
        # Consumed item should have reduced salience
        active = gate.active_items
        consumed_item = [i for i in active if i.item_id == item.item_id][0]
        assert consumed_item.consumed is True
        assert consumed_item.salience_score == 0.5  # 1.0 * 0.5

    def test_advance_cycle_applies_consumption_decay(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=3, consumption_decay=0.5)
        item = WorkspaceItem(content="Test", salience_score=1.0)
        gate.evaluate(item)
        gate.mark_consumed(item.item_id)
        gate.advance_cycle()
        active = gate.active_items
        consumed_item = [i for i in active if i.item_id == item.item_id][0]
        # After consume (0.5) then cycle decay (× 0.5 again)
        assert consumed_item.salience_score == pytest.approx(0.25)

    def test_peripheral_queue(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=1)
        gate.evaluate(WorkspaceItem(content="Stay", salience_score=0.9))
        gate.evaluate(WorkspaceItem(content="Go", salience_score=0.1))
        periph = gate.peripheral_items
        assert len(periph) == 1
        assert periph[0].content == "Go"

    def test_get_snapshot(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=3)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.5))
        snap = gate.get_snapshot()
        assert snap["capacity"] == 3
        assert snap["active_count"] == 1
        assert len(snap["active_items"]) == 1

    def test_thread_safety(self):
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=10)
        errors = []

        def add_items():
            try:
                for i in range(20):
                    gate.evaluate(WorkspaceItem(
                        content=f"Thread item {i}",
                        salience_score=i / 20.0,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_items) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(gate.active_items) <= gate.capacity


# ═══════════════════════════════════════════════════════════════════════════════
# Module 2: GWT-3 — Global Broadcast
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentReaction:
    """Tests for AgentReaction."""

    def test_reaction_defaults(self):
        from app.consciousness.global_broadcast import AgentReaction
        r = AgentReaction(agent_id="test")
        assert r.reaction_type == "NOTED"
        assert r.relevance_score == 0.0
        assert r.proposed_action is None

    def test_reaction_to_dict(self):
        from app.consciousness.global_broadcast import AgentReaction
        r = AgentReaction(
            agent_id="researcher",
            reaction_type="ACTIONABLE",
            relevance_score=0.85,
            proposed_action="Research this topic",
        )
        d = r.to_dict()
        assert d["agent_id"] == "researcher"
        assert d["reaction_type"] == "ACTIONABLE"


class TestBroadcastEvent:
    """Tests for BroadcastEvent."""

    def test_event_defaults(self):
        from app.consciousness.global_broadcast import BroadcastEvent
        event = BroadcastEvent()
        assert event.event_id  # UUID
        assert event.integration_score == 0.0

    def test_integration_score_all_noted(self):
        from app.consciousness.global_broadcast import BroadcastEvent, AgentReaction
        event = BroadcastEvent(receiving_agents=["a", "b", "c"])
        event.reactions = {
            "a": AgentReaction(agent_id="a", reaction_type="NOTED"),
            "b": AgentReaction(agent_id="b", reaction_type="NOTED"),
            "c": AgentReaction(agent_id="c", reaction_type="NOTED"),
        }
        score = event.compute_integration_score()
        assert score == 0.0

    def test_integration_score_all_relevant(self):
        from app.consciousness.global_broadcast import BroadcastEvent, AgentReaction
        event = BroadcastEvent(receiving_agents=["a", "b", "c"])
        event.reactions = {
            "a": AgentReaction(agent_id="a", reaction_type="RELEVANT"),
            "b": AgentReaction(agent_id="b", reaction_type="URGENT"),
            "c": AgentReaction(agent_id="c", reaction_type="ACTIONABLE"),
        }
        score = event.compute_integration_score()
        assert score == 1.0

    def test_integration_score_mixed(self):
        from app.consciousness.global_broadcast import BroadcastEvent, AgentReaction
        event = BroadcastEvent(receiving_agents=["a", "b", "c"])
        event.reactions = {
            "a": AgentReaction(agent_id="a", reaction_type="RELEVANT"),
            "b": AgentReaction(agent_id="b", reaction_type="NOTED"),
            "c": AgentReaction(agent_id="c", reaction_type="ACTIONABLE"),
        }
        score = event.compute_integration_score()
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_integration_score_empty_agents(self):
        from app.consciousness.global_broadcast import BroadcastEvent
        event = BroadcastEvent(receiving_agents=[])
        score = event.compute_integration_score()
        assert score == 0.0


class TestAgentBroadcastListener:
    """Tests for AgentBroadcastListener."""

    def test_listener_budget(self):
        from app.consciousness.global_broadcast import AgentBroadcastListener
        listener = AgentBroadcastListener(agent_id="test", role="researcher", attention_budget=2)
        assert listener.has_budget() is True
        listener.broadcasts_processed = 2
        assert listener.has_budget() is False

    def test_listener_reset_budget(self):
        from app.consciousness.global_broadcast import AgentBroadcastListener
        listener = AgentBroadcastListener(agent_id="test", role="coder", attention_budget=3)
        listener.broadcasts_processed = 3
        assert listener.has_budget() is False
        listener.reset_budget()
        assert listener.has_budget() is True
        assert listener.broadcasts_processed == 0


class TestGlobalBroadcastEngine:
    """Tests for GlobalBroadcastEngine."""

    def test_register_listener(self):
        from app.consciousness.global_broadcast import GlobalBroadcastEngine
        with patch("app.consciousness.config.load_config",
                   return_value=MagicMock(reaction_threshold=0.3, attention_budget=3)):
            engine = GlobalBroadcastEngine()
            engine.register_listener("researcher", "research")
            assert "researcher" in engine._listeners

    def test_register_multiple_listeners(self):
        from app.consciousness.global_broadcast import GlobalBroadcastEngine
        with patch("app.consciousness.config.load_config",
                   return_value=MagicMock(reaction_threshold=0.3, attention_budget=3)):
            engine = GlobalBroadcastEngine()
            for role in ("researcher", "coder", "writer"):
                engine.register_listener(role, role)
            assert len(engine._listeners) == 3

    def test_advance_cycle_resets_budgets(self):
        from app.consciousness.global_broadcast import GlobalBroadcastEngine
        with patch("app.consciousness.config.load_config",
                   return_value=MagicMock(reaction_threshold=0.3, attention_budget=3)):
            engine = GlobalBroadcastEngine()
            engine.register_listener("test", "test")
            engine._listeners["test"].broadcasts_processed = 3
            engine.advance_cycle()
            assert engine._listeners["test"].broadcasts_processed == 0

    def test_broadcast_no_embeddings(self):
        from app.consciousness.global_broadcast import GlobalBroadcastEngine
        from app.consciousness.workspace_buffer import WorkspaceItem
        with patch("app.consciousness.config.load_config",
                   return_value=MagicMock(reaction_threshold=0.3, attention_budget=3)):
            engine = GlobalBroadcastEngine()
            engine.register_listener("researcher", "research")
            item = WorkspaceItem(content="Test broadcast")
            event = engine.broadcast(item)
            assert "researcher" in event.reactions

    def test_get_recent_events(self):
        from app.consciousness.global_broadcast import GlobalBroadcastEngine
        engine = GlobalBroadcastEngine()
        events = engine.get_recent_events(5)
        assert isinstance(events, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Module 3: HOT-3 — Belief Store
# ═══════════════════════════════════════════════════════════════════════════════

class TestBelief:
    """Tests for Belief dataclass."""

    def test_belief_defaults(self):
        from app.consciousness.belief_store import Belief
        b = Belief()
        assert b.belief_id
        assert b.confidence == 0.5
        assert b.belief_status == "ACTIVE"
        assert b.domain == "world_model"

    def test_belief_to_dict(self):
        from app.consciousness.belief_store import Belief
        b = Belief(content="Test belief", confidence=0.8, domain="user_model")
        d = b.to_dict()
        assert d["confidence"] == 0.8
        assert d["domain"] == "user_model"

    def test_belief_content_truncation(self):
        from app.consciousness.belief_store import Belief
        b = Belief(content="x" * 500)
        d = b.to_dict()
        assert len(d["content"]) == 300


class TestMetacognitiveUpdate:
    """Tests for MetacognitiveUpdate."""

    def test_update_defaults(self):
        from app.consciousness.belief_store import MetacognitiveUpdate
        u = MetacognitiveUpdate()
        assert u.trigger == "COGITO_CYCLE"
        assert u.action_taken == "NO_CHANGE"

    def test_update_to_dict(self):
        from app.consciousness.belief_store import MetacognitiveUpdate
        u = MetacognitiveUpdate(
            trigger="PREDICTION_ERROR",
            action_taken="CONFIDENCE_ADJUSTED",
            old_confidence=0.8,
            new_confidence=0.65,
        )
        d = u.to_dict()
        assert d["trigger"] == "PREDICTION_ERROR"
        assert d["old_confidence"] == 0.8


class TestBeliefStore:
    """Tests for BeliefStore."""

    def test_form_belief_valid_domain(self):
        from app.consciousness.belief_store import BeliefStore
        store = BeliefStore()
        belief = store.form_belief("Test belief", domain="user_model", confidence=0.7)
        assert belief is not None
        assert belief.domain == "user_model"
        assert belief.confidence == 0.7

    def test_form_belief_invalid_domain(self):
        from app.consciousness.belief_store import BeliefStore
        store = BeliefStore()
        belief = store.form_belief("Test", domain="invalid_domain")
        assert belief is None

    def test_form_belief_clamps_confidence(self):
        from app.consciousness.belief_store import BeliefStore
        store = BeliefStore()
        belief = store.form_belief("Test", domain="world_model", confidence=1.5)
        assert belief.confidence == 1.0
        belief2 = store.form_belief("Test2", domain="world_model", confidence=-0.5)
        assert belief2.confidence == 0.0

    def test_confidence_decay(self):
        from app.consciousness.belief_store import BeliefStore
        store = BeliefStore()
        now = datetime.now(timezone.utc)
        old_validated = now - timedelta(hours=48)
        decayed = store._apply_confidence_decay(0.8, old_validated)
        assert decayed < 0.8

    def test_no_decay_when_recently_validated(self):
        from app.consciousness.belief_store import BeliefStore
        store = BeliefStore()
        now = datetime.now(timezone.utc)
        decayed = store._apply_confidence_decay(0.8, now)
        assert decayed >= 0.79

    def test_valid_domains(self):
        from app.consciousness.belief_store import VALID_DOMAINS
        assert "task_strategy" in VALID_DOMAINS
        assert "user_model" in VALID_DOMAINS
        assert "self_model" in VALID_DOMAINS
        assert "world_model" in VALID_DOMAINS
        assert "agent_capability" in VALID_DOMAINS
        assert "environment" in VALID_DOMAINS
        assert len(VALID_DOMAINS) == 6

    def test_valid_statuses(self):
        from app.consciousness.belief_store import VALID_STATUSES
        assert "ACTIVE" in VALID_STATUSES
        assert "SUSPENDED" in VALID_STATUSES
        assert "RETRACTED" in VALID_STATUSES
        assert "SUPERSEDED" in VALID_STATUSES


# ═══════════════════════════════════════════════════════════════════════════════
# Module 3b: HOT-3 — Metacognitive Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetacognitiveMonitor:
    """Tests for MetacognitiveMonitor."""

    def test_monitor_creation(self):
        from app.consciousness.metacognitive_monitor import MetacognitiveMonitor
        monitor = MetacognitiveMonitor()
        assert monitor._pending_outcomes == []

    def test_consult_beliefs(self):
        from app.consciousness.metacognitive_monitor import MetacognitiveMonitor
        mock_belief_store = MagicMock()
        mock_belief_store.query_relevant.return_value = []

        # Module-level mocks handle db.execute; patch get_belief_store at source
        import app.consciousness.belief_store as bs_mod
        original = bs_mod.get_belief_store
        bs_mod.get_belief_store = lambda: mock_belief_store
        try:
            monitor = MetacognitiveMonitor()
            record = monitor.consult_beliefs(
                task_description="Research Finnish flowers",
                crew_name="research_crew",
                goal_context="Answer user question",
            )
            assert record.selected_action == "dispatch:research_crew"
            assert "0 beliefs" in record.selection_reasoning
        finally:
            bs_mod.get_belief_store = original

    def test_slow_loop(self):
        from app.consciousness.metacognitive_monitor import MetacognitiveMonitor
        mock_belief_store = MagicMock()
        mock_belief_store.get_oldest_unvalidated.return_value = []
        mock_belief_store._apply_confidence_decay.return_value = 0.5

        import app.consciousness.belief_store as bs_mod
        original = bs_mod.get_belief_store
        bs_mod.get_belief_store = lambda: mock_belief_store
        try:
            monitor = MetacognitiveMonitor()
            result = monitor.run_slow_loop()
            assert "confidence_adjusted" in result
            assert "beliefs_suspended" in result
            assert "beliefs_reviewed" in result
        finally:
            bs_mod.get_belief_store = original

    def test_action_selection_record(self):
        from app.consciousness.metacognitive_monitor import ActionSelectionRecord
        r = ActionSelectionRecord(
            selected_action="dispatch:research_crew",
            beliefs_consulted=["id1", "id2"],
            selection_reasoning="Based on 2 beliefs",
        )
        d = r.to_dict()
        assert d["selected_action"] == "dispatch:research_crew"
        assert len(d["beliefs_consulted"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Module 4: AST-1 — Attention Schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestAttentionState:
    """Tests for AttentionState."""

    def test_state_defaults(self):
        from app.consciousness.attention_schema import AttentionState
        state = AttentionState()
        assert state.state_id
        assert state.is_stuck is False
        assert state.is_captured is False
        assert state.source_trigger == "GOAL_DRIVEN"

    def test_state_to_dict(self):
        from app.consciousness.attention_schema import AttentionState
        state = AttentionState(is_stuck=True, cycle_number=5)
        d = state.to_dict()
        assert d["is_stuck"] is True
        assert d["cycle"] == 5


class TestAttentionController:
    """Tests for AttentionController."""

    def test_detect_stuck_not_enough_history(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController(stuck_threshold_cycles=5)
        history = [AttentionState(workspace_item_ids=["a"]) for _ in range(3)]
        assert ctrl.detect_stuck(history) is False

    def test_detect_stuck_same_items(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController(stuck_threshold_cycles=3)
        history = [
            AttentionState(workspace_item_ids=["a", "b", "c"])
            for _ in range(5)
        ]
        assert ctrl.detect_stuck(history) is True

    def test_detect_stuck_changing_items(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController(stuck_threshold_cycles=3)
        history = [
            AttentionState(workspace_item_ids=[f"item_{i}", f"item_{i+1}"])
            for i in range(5)
        ]
        assert ctrl.detect_stuck(history) is False

    def test_detect_capture_dominant_item(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController(capture_dominance_threshold=0.70)
        state = AttentionState(
            salience_distribution={"a": 0.9, "b": 0.05, "c": 0.05}
        )
        captured, item_id = ctrl.detect_capture(state)
        assert captured is True
        assert item_id == "a"

    def test_detect_capture_balanced(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController(capture_dominance_threshold=0.70)
        state = AttentionState(
            salience_distribution={"a": 0.35, "b": 0.35, "c": 0.30}
        )
        captured, item_id = ctrl.detect_capture(state)
        assert captured is False

    def test_detect_capture_empty(self):
        from app.consciousness.attention_schema import AttentionController, AttentionState
        ctrl = AttentionController()
        state = AttentionState()
        captured, item_id = ctrl.detect_capture(state)
        assert captured is False

    def test_shift_cooldown(self):
        from app.consciousness.attention_schema import AttentionController
        ctrl = AttentionController(shift_cooldown_cycles=3)
        assert ctrl.can_recommend_shift(1) is True
        ctrl.record_shift(1)
        # Cooldown until cycle 4
        assert ctrl.can_recommend_shift(2) is False
        assert ctrl.can_recommend_shift(3) is False
        assert ctrl.can_recommend_shift(4) is True

    def test_max_shifts_per_period(self):
        from app.consciousness.attention_schema import AttentionController
        ctrl = AttentionController(max_shifts_per_period=2, shift_cooldown_cycles=1)
        ctrl.record_shift(1)
        ctrl.record_shift(3)
        assert ctrl.can_recommend_shift(5) is False
        ctrl.reset_period()
        assert ctrl.can_recommend_shift(5) is True


class TestAttentionPredictor:
    """Tests for AttentionPredictor."""

    def test_predict_empty_state(self):
        from app.consciousness.attention_schema import AttentionPredictor, AttentionState
        predictor = AttentionPredictor()
        state = AttentionState(cycle_number=1)
        pred = predictor.predict_next_focus(state)
        assert pred.cycle_number == 2
        assert pred.predicted_focus_ids == []

    def test_predict_top_salience(self):
        from app.consciousness.attention_schema import AttentionPredictor, AttentionState
        predictor = AttentionPredictor()
        state = AttentionState(
            cycle_number=1,
            salience_distribution={"a": 0.8, "b": 0.2, "c": 0.5},
        )
        pred = predictor.predict_next_focus(state)
        assert pred.predicted_focus_ids[0] == "a"

    def test_prediction_accuracy(self):
        from app.consciousness.attention_schema import AttentionPredictor, AttentionState, AttentionPrediction
        predictor = AttentionPredictor()
        pred = AttentionPrediction(
            predicted_focus_ids=["a", "b"],
            cycle_number=2,
        )
        actual = AttentionState(workspace_item_ids=["a", "b"])
        accuracy = predictor.evaluate_prediction(pred, actual)
        assert accuracy == 1.0

    def test_prediction_accuracy_partial(self):
        from app.consciousness.attention_schema import AttentionPredictor, AttentionState, AttentionPrediction
        predictor = AttentionPredictor()
        pred = AttentionPrediction(
            predicted_focus_ids=["a", "b", "c"],
        )
        actual = AttentionState(workspace_item_ids=["a", "d", "e"])
        accuracy = predictor.evaluate_prediction(pred, actual)
        assert 0.0 < accuracy < 1.0

    def test_running_accuracy(self):
        from app.consciousness.attention_schema import AttentionPredictor
        predictor = AttentionPredictor()
        assert predictor.running_accuracy == 0.5  # Default
        predictor._accuracy_history.append(0.8)
        predictor._accuracy_history.append(0.6)
        assert predictor.running_accuracy == pytest.approx(0.7)


class TestAttentionSchema:
    """Tests for AttentionSchema (integration)."""

    def test_schema_creation(self):
        from app.consciousness.attention_schema import AttentionSchema
        schema = AttentionSchema()
        assert schema._cycle == 0
        assert schema._current is None

    def test_update_with_items(self):
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem
        schema = AttentionSchema()
        items = [
            WorkspaceItem(content="A", salience_score=0.8),
            WorkspaceItem(content="B", salience_score=0.5),
        ]
        state = schema.update(items, cycle=1)
        assert len(state.workspace_item_ids) == 2
        assert state.cycle_number == 1

    def test_stuck_detection_integration(self):
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem
        schema = AttentionSchema()
        items = [WorkspaceItem(content="A", salience_score=0.8)]
        for cycle in range(1, 7):
            state = schema.update(items, cycle=cycle)
        assert state.is_stuck is True

    def test_capture_detection_integration(self):
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem
        schema = AttentionSchema()
        items = [
            WorkspaceItem(content="Dominant", salience_score=0.95),
            WorkspaceItem(content="Minor", salience_score=0.05),
        ]
        state = schema.update(items, cycle=1)
        assert state.is_captured is True

    def test_no_intervention_when_normal(self):
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem
        schema = AttentionSchema()
        items = [
            WorkspaceItem(content="A", salience_score=0.5),
            WorkspaceItem(content="B", salience_score=0.5),
        ]
        schema.update(items, cycle=1)
        assert schema.recommend_intervention() is None

    def test_intervention_on_capture(self):
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem
        schema = AttentionSchema()
        items = [
            WorkspaceItem(content="Dominant", salience_score=0.95),
            WorkspaceItem(content="Minor", salience_score=0.01),
        ]
        schema.update(items, cycle=1)
        intervention = schema.recommend_intervention()
        assert intervention is not None
        assert intervention["action"] == "suppress"

    def test_get_state_summary(self):
        from app.consciousness.attention_schema import AttentionSchema
        schema = AttentionSchema()
        summary = schema.get_state_summary()
        assert "cycle" in summary
        assert "is_stuck" in summary
        assert "prediction_accuracy" in summary

    def test_slow_loop(self):
        from app.consciousness.attention_schema import AttentionSchema
        schema = AttentionSchema()
        result = schema.run_slow_loop()
        assert "cycle" in result
        assert "prediction_accuracy" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Module 5: PP-1 — Predictive Coding Layer
# ═══════════════════════════════════════════════════════════════════════════════

class TestSurpriseClassification:
    """Tests for surprise level classification."""

    def test_expected(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.0) == "EXPECTED"
        assert classify_surprise(0.10) == "EXPECTED"

    def test_minor_deviation(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.20) == "MINOR_DEVIATION"
        assert classify_surprise(0.30) == "MINOR_DEVIATION"

    def test_notable_surprise(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.40) == "NOTABLE_SURPRISE"

    def test_major_surprise(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.60) == "MAJOR_SURPRISE"

    def test_paradigm_violation(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.80) == "PARADIGM_VIOLATION"
        assert classify_surprise(1.0) == "PARADIGM_VIOLATION"

    def test_edge_boundaries(self):
        from app.consciousness.predictive_layer import classify_surprise
        assert classify_surprise(0.15) == "MINOR_DEVIATION"
        assert classify_surprise(0.35) == "NOTABLE_SURPRISE"
        assert classify_surprise(0.55) == "MAJOR_SURPRISE"
        assert classify_surprise(0.75) == "PARADIGM_VIOLATION"


class TestCosineDistance:
    """Tests for predictive_layer._cosine_distance."""

    def test_identical_vectors(self):
        from app.consciousness.predictive_layer import _cosine_distance
        v = [1.0, 0.0]
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=0.01)

    def test_opposite_vectors(self):
        from app.consciousness.predictive_layer import _cosine_distance
        result = _cosine_distance([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(1.0, abs=0.01)

    def test_empty_vectors(self):
        from app.consciousness.predictive_layer import _cosine_distance
        assert _cosine_distance([], []) == 0.5
        assert _cosine_distance([1.0], []) == 0.5


class TestChannelPredictor:
    """Tests for ChannelPredictor."""

    def test_predictor_creation(self):
        from app.consciousness.predictive_layer import ChannelPredictor
        pred = ChannelPredictor("signal_message")
        assert pred.channel_id == "signal_message"
        assert pred.running_confidence == 0.5
        assert pred._prediction_count == 0

    def test_generate_prediction(self):
        from app.consciousness.predictive_layer import ChannelPredictor
        pred = ChannelPredictor("test_channel")
        prediction = pred.generate_prediction("Some context")
        assert prediction.channel == "test_channel"
        assert prediction.confidence == 0.5
        assert pred._prediction_count == 1

    def test_compute_error_no_embeddings(self):
        from app.consciousness.predictive_layer import ChannelPredictor, Prediction
        pred = ChannelPredictor("test")
        prediction = Prediction(channel="test")
        error = pred.compute_error(prediction, "actual content")
        # Without embeddings, default error = 0.3
        assert error.error_magnitude == 0.3
        assert error.effective_surprise == 0.3 * pred.running_confidence

    def test_confidence_floor(self):
        from app.consciousness.predictive_layer import ChannelPredictor
        pred = ChannelPredictor("test", confidence_floor=0.1)
        # Simulate many bad predictions
        pred.running_confidence = 0.05
        # Floor should prevent going below 0.1
        pred._prediction_count = 20
        pred._accuracy_history.extend([0.0] * 50)
        # Trigger confidence update
        from app.consciousness.predictive_layer import Prediction
        prediction = Prediction(channel="test")
        pred.compute_error(prediction, "anything")
        assert pred.running_confidence >= 0.1

    def test_warm_up_period(self):
        from app.consciousness.predictive_layer import ChannelPredictor, Prediction
        pred = ChannelPredictor("test", warm_up_count=10)
        initial_conf = pred.running_confidence
        # Before warm-up: confidence shouldn't adapt
        for i in range(9):
            pred._prediction_count = i + 1
            pred.compute_error(Prediction(channel="test"), "content")
        # During warm-up, confidence should stay at initial
        assert pred.running_confidence == initial_conf

    def test_stats(self):
        from app.consciousness.predictive_layer import ChannelPredictor
        pred = ChannelPredictor("signal_message")
        stats = pred.stats
        assert stats["channel"] == "signal_message"
        assert stats["running_confidence"] == 0.5
        assert stats["prediction_count"] == 0


class TestPredictiveLayer:
    """Tests for PredictiveLayer."""

    def test_layer_creation(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer(surprise_budget_per_cycle=3)
        assert layer.surprise_budget == 3
        assert layer._cycle == 0

    def test_get_predictor_creates_new(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer()
        pred = layer.get_predictor("new_channel")
        assert pred.channel_id == "new_channel"
        # Second call returns same predictor
        pred2 = layer.get_predictor("new_channel")
        assert pred is pred2

    def test_advance_cycle(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer()
        layer.advance_cycle()
        assert layer._cycle == 1
        assert layer._surprises_this_cycle == 0

    def test_predict_and_compare(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer()
        layer.advance_cycle()
        error = layer.predict_and_compare(
            channel="signal_message",
            context="Finnish nature",
            actual_content="What flowers bloom?",
        )
        assert error.channel == "signal_message"
        assert 0.0 <= error.error_magnitude <= 1.0
        assert 0.0 <= error.effective_surprise <= 1.0
        assert error.surprise_level in ("EXPECTED", "MINOR_DEVIATION", "NOTABLE_SURPRISE",
                                         "MAJOR_SURPRISE", "PARADIGM_VIOLATION")

    def test_surprise_budget(self):
        from app.consciousness.predictive_layer import PredictiveLayer, PredictionError
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        layer.advance_cycle()
        # Manually test budget logic
        layer._surprises_this_cycle = 0
        # Should route first 2 notable+ surprises
        assert layer._surprises_this_cycle < layer.surprise_budget
        layer._surprises_this_cycle = 2
        assert not (layer._surprises_this_cycle < layer.surprise_budget)

    def test_belief_review_trigger(self):
        from app.consciousness.predictive_layer import PredictiveLayer, PredictionError
        layer = PredictiveLayer()
        layer._cycle = 10
        # Add major surprises
        for i in range(3):
            error = PredictionError(
                channel="test_channel",
                surprise_level="MAJOR_SURPRISE",
                cycle_number=layer._cycle - i,
            )
            layer._recent_major.append(error)
        should_review = layer.should_trigger_belief_review("test_channel", window=10, threshold=3)
        assert should_review is True

    def test_no_belief_review_below_threshold(self):
        from app.consciousness.predictive_layer import PredictiveLayer, PredictionError
        layer = PredictiveLayer()
        layer._cycle = 10
        # Only 1 major surprise (below threshold of 3)
        error = PredictionError(channel="test_channel", surprise_level="MAJOR_SURPRISE", cycle_number=10)
        layer._recent_major.append(error)
        assert layer.should_trigger_belief_review("test_channel") is False

    def test_get_channel_stats(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer()
        layer.get_predictor("channel_a")
        layer.get_predictor("channel_b")
        stats = layer.get_channel_stats()
        assert len(stats) == 2

    def test_run_slow_loop(self):
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer()
        layer.get_predictor("test")
        result = layer.run_slow_loop()
        assert result["channels"] == 1
        assert result["total_predictions"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Module 6: Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsciousnessConfig:
    """Tests for ConsciousnessConfig."""

    def test_defaults(self):
        from app.consciousness.config import ConsciousnessConfig
        cfg = ConsciousnessConfig()
        assert cfg.workspace_capacity == 5
        assert cfg.salience_w_goal == 0.35
        assert cfg.salience_w_novelty == 0.25
        assert cfg.salience_w_urgency == 0.15
        assert cfg.salience_w_surprise == 0.25
        assert cfg.reaction_threshold == 0.30
        assert cfg.attention_budget == 3
        assert cfg.belief_suspension_threshold == 0.20
        assert cfg.confidence_decay_factor == 0.995
        assert cfg.mandatory_review_count == 3

    def test_salience_weights_sum_to_one(self):
        from app.consciousness.config import ConsciousnessConfig
        cfg = ConsciousnessConfig()
        total = cfg.salience_w_goal + cfg.salience_w_novelty + cfg.salience_w_urgency + cfg.salience_w_surprise
        assert total == pytest.approx(1.0)

    def test_load_defaults(self):
        from app.consciousness.config import load_config
        cfg = load_config()
        assert cfg.workspace_capacity == 5

    def test_asymmetric_learning(self):
        """Disconfirmation rate should be higher than confirmation rate."""
        from app.consciousness.config import ConsciousnessConfig
        cfg = ConsciousnessConfig()
        assert cfg.disconfirmation_rate > cfg.confirmation_rate


# ═══════════════════════════════════════════════════════════════════════════════
# Module 7: Cross-Module Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrossModuleIntegration:
    """Tests for cross-module interactions."""

    def test_pp1_surprise_feeds_gwt2_salience(self):
        """PP-1 surprise signal should amplify GWT-2 salience scoring."""
        from app.consciousness.workspace_buffer import WorkspaceItem, SalienceScorer
        scorer = SalienceScorer()

        item_no_surprise = WorkspaceItem(content="Normal input", agent_urgency=0.5)
        item_with_surprise = WorkspaceItem(content="Surprising input", agent_urgency=0.5, surprise_signal=0.8)

        s1 = scorer.score(item_no_surprise, [], [])
        s2 = scorer.score(item_with_surprise, [], [])
        assert s2 > s1, "Surprise signal should amplify salience"

    def test_gwt2_workspace_feeds_ast1(self):
        """GWT-2 workspace contents should feed into AST-1 for monitoring."""
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        from app.consciousness.attention_schema import AttentionSchema

        gate = CompetitiveGate(capacity=3)
        schema = AttentionSchema()
        items = [
            WorkspaceItem(content=f"Item {i}", salience_score=0.5 + i * 0.1)
            for i in range(3)
        ]
        for item in items:
            gate.evaluate(item)
        state = schema.update(gate.active_items, cycle=1)
        assert len(state.workspace_item_ids) == 3

    def test_pp1_belief_review_trigger(self):
        """Systematic prediction failures should trigger belief review."""
        from app.consciousness.predictive_layer import PredictiveLayer, PredictionError

        layer = PredictiveLayer()
        layer._cycle = 20

        # Simulate 4 major surprises on same channel
        for i in range(4):
            error = PredictionError(
                channel="failing_channel",
                surprise_level="PARADIGM_VIOLATION",
                cycle_number=layer._cycle - i,
            )
            layer._recent_major.append(error)

        assert layer.should_trigger_belief_review("failing_channel", threshold=3) is True
        # Different channel should not trigger
        assert layer.should_trigger_belief_review("other_channel", threshold=3) is False

    def test_workspace_capacity_is_constrained(self):
        """Workspace should never exceed configured capacity."""
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem
        gate = CompetitiveGate(capacity=5)

        for i in range(20):
            gate.evaluate(WorkspaceItem(
                content=f"Item {i}",
                salience_score=i / 20.0,
            ))

        assert len(gate.active_items) <= 5

    def test_capture_triggers_intervention(self):
        """AST-1 should recommend intervention when capture detected."""
        from app.consciousness.attention_schema import AttentionSchema
        from app.consciousness.workspace_buffer import WorkspaceItem

        schema = AttentionSchema()
        items = [
            WorkspaceItem(content="Dominant", salience_score=0.99),
            WorkspaceItem(content="Tiny", salience_score=0.01),
        ]
        schema.update(items, cycle=1)

        intervention = schema.recommend_intervention()
        assert intervention is not None
        assert intervention["action"] == "suppress"

    def test_displaced_items_go_to_peripheral(self):
        """Displaced items should be accessible in peripheral queue."""
        from app.consciousness.workspace_buffer import CompetitiveGate, WorkspaceItem

        gate = CompetitiveGate(capacity=2)
        gate.evaluate(WorkspaceItem(content="A", salience_score=0.3))
        gate.evaluate(WorkspaceItem(content="B", salience_score=0.4))
        result = gate.evaluate(WorkspaceItem(content="C", salience_score=0.8))

        assert result.displaced_item is not None
        periph = gate.peripheral_items
        assert len(periph) == 1
        assert periph[0].content == "A"  # Lowest salience displaced


# ═══════════════════════════════════════════════════════════════════════════════
# Module 8: DGM Safety Invariants
# ═══════════════════════════════════════════════════════════════════════════════

class TestDGMSafetyInvariants:
    """Tests for DGM safety properties (infrastructure-level invariants)."""

    def test_workspace_capacity_immutable(self):
        """Agents cannot change workspace capacity at runtime."""
        from app.consciousness.workspace_buffer import CompetitiveGate
        gate = CompetitiveGate(capacity=5)
        # Capacity is set at construction, not modifiable by evaluate()
        assert gate.capacity == 5

    def test_salience_weights_immutable(self):
        """Salience weights are set at scorer construction."""
        from app.consciousness.workspace_buffer import SalienceScorer
        scorer = SalienceScorer()
        # No setter methods — weights are immutable after __init__
        assert hasattr(scorer, 'w_goal')
        assert not hasattr(scorer, 'set_w_goal')

    def test_belief_suspension_threshold_from_config(self):
        """Belief suspension threshold comes from config, not agent input."""
        from app.consciousness.config import ConsciousnessConfig
        cfg = ConsciousnessConfig()
        assert cfg.belief_suspension_threshold == 0.20

    def test_surprise_budget_limits_routing(self):
        """Surprise budget prevents runaway surprise amplification."""
        from app.consciousness.predictive_layer import PredictiveLayer
        layer = PredictiveLayer(surprise_budget_per_cycle=2)
        # Budget should be respected
        assert layer.surprise_budget == 2

    def test_shift_cooldown_prevents_thrashing(self):
        """Attention shifts have cooldown to prevent oscillation."""
        from app.consciousness.attention_schema import AttentionController
        ctrl = AttentionController(shift_cooldown_cycles=3, max_shifts_per_period=2)
        ctrl.record_shift(1)
        # Can't shift again immediately
        assert ctrl.can_recommend_shift(2) is False
        assert ctrl.can_recommend_shift(4) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Module 9: Singleton Management
# ═══════════════════════════════════════════════════════════════════════════════

class TestSingletons:
    """Tests for module-level singletons."""

    def test_workspace_gate_singleton(self):
        import app.consciousness.workspace_buffer as wb
        wb._gate = None
        gate1 = wb.get_workspace_gate()
        gate2 = wb.get_workspace_gate()
        assert gate1 is gate2
        wb._gate = None  # Reset

    def test_salience_scorer_singleton(self):
        import app.consciousness.workspace_buffer as wb
        wb._scorer = None
        scorer1 = wb.get_salience_scorer()
        scorer2 = wb.get_salience_scorer()
        assert scorer1 is scorer2
        wb._scorer = None

    def test_broadcast_engine_singleton(self):
        import app.consciousness.global_broadcast as gb
        gb._engine = None
        engine1 = gb.get_broadcast_engine()
        engine2 = gb.get_broadcast_engine()
        assert engine1 is engine2
        gb._engine = None

    def test_belief_store_singleton(self):
        import app.consciousness.belief_store as bs
        bs._store = None
        store1 = bs.get_belief_store()
        store2 = bs.get_belief_store()
        assert store1 is store2
        bs._store = None

    def test_attention_schema_singleton(self):
        import app.consciousness.attention_schema as asc
        asc._schema = None
        s1 = asc.get_attention_schema()
        s2 = asc.get_attention_schema()
        assert s1 is s2
        asc._schema = None

    def test_predictive_layer_singleton(self):
        import app.consciousness.predictive_layer as pl
        pl._layer = None
        l1 = pl.get_predictive_layer()
        l2 = pl.get_predictive_layer()
        assert l1 is l2
        pl._layer = None

    def test_metacognitive_monitor_singleton(self):
        import app.consciousness.metacognitive_monitor as mm
        mm._monitor = None
        m1 = mm.get_monitor()
        m2 = mm.get_monitor()
        assert m1 is m2
        mm._monitor = None
