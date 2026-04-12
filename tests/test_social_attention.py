"""
Tests for Social Attention Modeling — Theory of Mind for Agent Attention.

Extends AST-1 (self-model of attention) to model OTHER agents' attention.
Implements Butlin et al. VIII-3 (unified self-model) by distinguishing
self-attention from other-attention.

Total: ~30 tests
"""

import sys
from collections import deque
from unittest.mock import MagicMock

for _mod in ["psycopg2", "psycopg2.pool", "psycopg2.extras",
             "chromadb", "chromadb.config", "chromadb.utils",
             "chromadb.utils.embedding_functions",
             "app.control_plane", "app.control_plane.db",
             "app.memory", "app.memory.chromadb_manager"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

sys.modules["app.memory.chromadb_manager"].embed = MagicMock(return_value=[0.1] * 768)
sys.modules["app.control_plane.db"].execute = MagicMock(return_value=[])

import pytest


class TestAgentAttentionModel:
    """Tests for individual agent attention models."""

    def test_model_creation(self):
        from app.consciousness.attention_schema import AgentAttentionModel
        model = AgentAttentionModel(agent_id="researcher", role="research")
        assert model.agent_id == "researcher"
        assert model.prediction_accuracy == 0.5
        assert model.activity_level == 0.5

    def test_model_to_dict(self):
        from app.consciousness.attention_schema import AgentAttentionModel
        model = AgentAttentionModel(agent_id="coder", role="coding")
        model.topic_affinities = {"python": 0.9, "javascript": 0.3}
        d = model.to_dict()
        assert d["agent_id"] == "coder"
        assert len(d["top_affinities"]) == 2


class TestSocialAttentionModel:
    """Tests for the social attention model (Theory of Mind)."""

    def test_creation(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        assert len(sam._models) == 0

    def test_get_or_create_model(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        model = sam.get_or_create_model("researcher", "research")
        assert model.agent_id == "researcher"
        # Second call returns same
        model2 = sam.get_or_create_model("researcher")
        assert model is model2

    def test_update_from_broadcast_reaction(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        sam.update_from_broadcast_reaction(
            "researcher", "research", "Finnish flowers", "RELEVANT", 0.8
        )
        model = sam._models["researcher"]
        assert "Finnish flowers" in model.topic_affinities
        assert model.topic_affinities["Finnish flowers"] > 0.5

    def test_topic_affinity_accumulates(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        # Multiple reactions to same topic → affinity should increase
        for _ in range(5):
            sam.update_from_broadcast_reaction(
                "researcher", "research", "nature topics", "RELEVANT", 0.9
            )
        model = sam._models["researcher"]
        assert model.topic_affinities["nature topics"] > 0.7

    def test_activity_level_tracks_reactions(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        # All RELEVANT → high activity
        for _ in range(5):
            sam.update_from_broadcast_reaction(
                "active_agent", "research", "topic", "RELEVANT", 0.8
            )
        assert sam._models["active_agent"].activity_level > 0.8

        # All NOTED → low activity
        sam2 = SocialAttentionModel()
        for _ in range(5):
            sam2.update_from_broadcast_reaction(
                "passive_agent", "research", "topic", "NOTED", 0.1
            )
        assert sam2._models["passive_agent"].activity_level < 0.2

    def test_predict_agent_attention_no_history(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        relevance, reason = sam.predict_agent_attention("unknown", "any task")
        assert relevance == 0.5
        assert "No attention history" in reason

    def test_predict_agent_attention_with_history(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        # Build history: researcher attends to "python coding"
        for _ in range(5):
            sam.update_from_broadcast_reaction(
                "coder", "coding", "python coding task", "ACTIONABLE", 0.95
            )
        # Predict relevance for a python task
        relevance, reason = sam.predict_agent_attention("coder", "Fix python bug")
        assert relevance > 0.5
        assert "topic" in reason.lower()

    def test_predict_best_agent_for_task(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        # Researcher → nature topics
        for _ in range(5):
            sam.update_from_broadcast_reaction(
                "researcher", "research", "Finnish nature flowers", "RELEVANT", 0.9
            )
        # Coder → python bugs
        for _ in range(5):
            sam.update_from_broadcast_reaction(
                "coder", "coding", "python bug fix", "RELEVANT", 0.9
            )
        rankings = sam.predict_best_agent_for_task("What flowers bloom in Finland?")
        assert len(rankings) == 2
        # Researcher should rank higher for nature tasks
        assert rankings[0][0] == "researcher"

    def test_evaluate_prediction_accuracy(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        sam.get_or_create_model("researcher", "research")
        sam._models["researcher"].predicted_relevance = 0.8

        # Correct prediction (predicted relevant, was relevant)
        sam.evaluate_prediction_accuracy("researcher", "RELEVANT")
        assert sam._models["researcher"].prediction_accuracy > 0.5

    def test_prediction_accuracy_degrades_on_wrong(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        model = sam.get_or_create_model("researcher", "research")
        model.prediction_accuracy = 0.8
        model.predicted_relevance = 0.9  # Predicted relevant

        # Wrong: actual was NOTED (not relevant)
        sam.evaluate_prediction_accuracy("researcher", "NOTED")
        assert model.prediction_accuracy < 0.8

    def test_self_other_distinction_empty(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        result = sam.get_self_other_distinction(None)
        assert result["other_models_count"] == 0
        assert result["self_focus"] == ""

    def test_self_other_distinction_with_data(self):
        from app.consciousness.attention_schema import SocialAttentionModel, AttentionState
        sam = SocialAttentionModel()
        sam.update_from_broadcast_reaction(
            "researcher", "research", "nature", "RELEVANT", 0.8
        )

        self_state = AttentionState(
            attending_because="Highest salience: user query about flowers",
            salience_distribution={"a": 0.6, "b": 0.3, "c": 0.1},
        )
        result = sam.get_self_other_distinction(self_state)
        assert result["self_focus"] != ""
        assert result["other_models_count"] == 1
        assert len(result["other_attention_summary"]) == 1
        assert result["self_salience_entropy"] > 0

    def test_self_other_divergence(self):
        """Divergence should be non-zero when self and others attend differently."""
        from app.consciousness.attention_schema import SocialAttentionModel, AttentionState
        sam = SocialAttentionModel()
        model = sam.get_or_create_model("researcher", "research")
        model.predicted_relevance = 0.1  # Others not interested

        self_state = AttentionState(
            salience_distribution={"a": 0.95, "b": 0.05},
        )
        result = sam.get_self_other_distinction(self_state)
        assert result["self_other_divergence"] > 0

    def test_topic_affinity_cap(self):
        """Topic affinities should be capped at 50 to prevent unbounded growth."""
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        for i in range(60):
            sam.update_from_broadcast_reaction(
                "researcher", "research", f"topic_{i}", "RELEVANT", 0.5
            )
        assert len(sam._models["researcher"].topic_affinities) <= 50

    def test_get_summary(self):
        from app.consciousness.attention_schema import SocialAttentionModel
        sam = SocialAttentionModel()
        sam.update_from_broadcast_reaction("r", "research", "t", "RELEVANT", 0.5)
        summary = sam.get_summary()
        assert summary["agents_modeled"] == 1
        assert "prediction_accuracy" in summary
        assert "r" in summary["agent_models"]

    def test_singleton(self):
        import app.consciousness.attention_schema as mod
        mod._social_model = None
        s1 = mod.get_social_attention_model()
        s2 = mod.get_social_attention_model()
        assert s1 is s2
        mod._social_model = None


class TestSocialAttentionWiring:
    """Verify social attention is wired to GWT-3 broadcast."""

    def test_broadcast_updates_social_model(self):
        """GWT-3 broadcast should call social attention model."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "app" / "consciousness" / "global_broadcast.py").read_text()
        assert "get_social_attention_model" in source
        assert "update_from_broadcast_reaction" in source
        assert "evaluate_prediction_accuracy" in source

    def test_social_model_in_attention_schema(self):
        """Social attention should be in attention_schema.py module."""
        from app.consciousness.attention_schema import (
            SocialAttentionModel,
            AgentAttentionModel,
            get_social_attention_model,
        )
        assert SocialAttentionModel is not None
        assert callable(get_social_attention_model)
