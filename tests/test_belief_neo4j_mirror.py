"""Tests for the :Belief Neo4j projection (app.subia.belief.neo4j_mirror)
and the BeliefStore wiring that calls into it.

Strategy: monkeypatch `_get_driver` to return a MagicMock driver so we can
assert Cypher + params without a running Neo4j. For BeliefStore tests, we
also mock `app.control_plane.db.execute` and the chromadb embed function
so the SQL path is exercised in-memory.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
from contextlib import contextmanager

import pytest

# Mock only what's safe to mock without breaking other tests' imports:
#   - psycopg2 internals (driver is installed but we don't want real DB I/O)
# Do NOT mock chromadb (installed and used by other tests via dotted imports).
# Do NOT mock app.control_plane / app.control_plane.db (test_control_plane.py
# needs the real package; the real `execute()` returns None gracefully when
# no Postgres pool is configured, which is what our test path expects).
_MOCK_MODULES = ["psycopg2", "psycopg2.pool", "psycopg2.extras"]
for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()


@contextmanager
def _mock_driver():
    """Patch neo4j_mirror._get_driver to return a session-recording mock.

    Yields the (driver_mock, session_mock) so tests can assert on
    .run() calls. Also resets the module's sticky state on entry/exit.
    """
    from app.subia.belief import neo4j_mirror
    neo4j_mirror._reset_for_tests()
    session_mock = MagicMock()
    session_mock.__enter__ = MagicMock(return_value=session_mock)
    session_mock.__exit__ = MagicMock(return_value=False)
    driver_mock = MagicMock()
    driver_mock.session.return_value = session_mock
    with patch.object(neo4j_mirror, "_get_driver", return_value=driver_mock):
        yield driver_mock, session_mock
    neo4j_mirror._reset_for_tests()


# ═════════════════════════════════════════════════════════════════════════
# mirror_belief
# ═════════════════════════════════════════════════════════════════════════

class TestMirrorBelief:
    def test_mirror_belief_full_fields(self):
        from app.subia.belief import neo4j_mirror
        from datetime import datetime, timezone

        with _mock_driver() as (_, session):
            ts = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)
            ok = neo4j_mirror.mirror_belief(
                "b-1",
                domain="user_model",
                confidence=0.7,
                belief_status="ACTIVE",
                formed_at=ts,
            )
        assert ok is True
        session.run.assert_called_once()
        cypher, params = session.run.call_args[0]
        assert "MERGE (b:Belief {belief_id: $belief_id})" in cypher
        assert "ON CREATE SET b.formed_at = $formed_at" in cypher
        assert "b.domain = $domain" in cypher
        assert "b.confidence = $confidence" in cypher
        assert "b.belief_status = $belief_status" in cypher
        assert params["belief_id"] == "b-1"
        assert params["domain"] == "user_model"
        assert params["confidence"] == 0.7
        assert params["belief_status"] == "ACTIVE"
        assert params["formed_at"] == ts.isoformat()

    def test_mirror_belief_partial_status_only(self):
        """Update path: only belief_status passed — domain/confidence not clobbered."""
        from app.subia.belief import neo4j_mirror

        with _mock_driver() as (_, session):
            ok = neo4j_mirror.mirror_belief("b-2", belief_status="RETRACTED")
        assert ok is True
        cypher, params = session.run.call_args[0]
        assert "b.belief_status = $belief_status" in cypher
        assert "b.domain = $domain" not in cypher
        assert "b.confidence = $confidence" not in cypher
        assert "ON CREATE SET b.formed_at" not in cypher
        assert params == {"belief_id": "b-2", "belief_status": "RETRACTED"}

    def test_mirror_belief_partial_confidence_only(self):
        from app.subia.belief import neo4j_mirror

        with _mock_driver() as (_, session):
            ok = neo4j_mirror.mirror_belief("b-3", confidence=0.42)
        assert ok is True
        cypher, params = session.run.call_args[0]
        assert "b.confidence = $confidence" in cypher
        assert "b.domain = $domain" not in cypher
        assert params["confidence"] == 0.42

    def test_mirror_belief_empty_id_returns_false(self):
        from app.subia.belief import neo4j_mirror
        assert neo4j_mirror.mirror_belief("") is False

    def test_mirror_belief_unavailable_neo4j_returns_false(self):
        """When _get_driver returns None, mirror is a graceful no-op."""
        from app.subia.belief import neo4j_mirror
        neo4j_mirror._reset_for_tests()
        with patch.object(neo4j_mirror, "_get_driver", return_value=None):
            ok = neo4j_mirror.mirror_belief("b-x", domain="user_model")
        assert ok is False
        neo4j_mirror._reset_for_tests()

    def test_mirror_belief_neo4j_raises_caught(self):
        """A Cypher exception is caught and returns False."""
        from app.subia.belief import neo4j_mirror

        with _mock_driver() as (_, session):
            session.run.side_effect = RuntimeError("connection lost")
            ok = neo4j_mirror.mirror_belief("b-4", confidence=0.5)
        assert ok is False


# ═════════════════════════════════════════════════════════════════════════
# mirror_supersession
# ═════════════════════════════════════════════════════════════════════════

class TestMirrorSupersession:
    def test_writes_edge(self):
        from app.subia.belief import neo4j_mirror

        with _mock_driver() as (_, session):
            ok = neo4j_mirror.mirror_supersession("old-1", "new-1")
        assert ok is True
        cypher, params = session.run.call_args[0]
        assert "MERGE (old:Belief {belief_id: $old_id})" in cypher
        assert "MERGE (new:Belief {belief_id: $new_id})" in cypher
        assert "MERGE (old)-[r:SUPERSEDED_BY]->(new)" in cypher
        assert params == {"old_id": "old-1", "new_id": "new-1"}

    def test_missing_id_returns_false(self):
        from app.subia.belief import neo4j_mirror
        assert neo4j_mirror.mirror_supersession("", "new-1") is False
        assert neo4j_mirror.mirror_supersession("old-1", "") is False
        assert neo4j_mirror.mirror_supersession("", "") is False

    def test_unavailable_neo4j_returns_false(self):
        from app.subia.belief import neo4j_mirror
        neo4j_mirror._reset_for_tests()
        with patch.object(neo4j_mirror, "_get_driver", return_value=None):
            ok = neo4j_mirror.mirror_supersession("a", "b")
        assert ok is False
        neo4j_mirror._reset_for_tests()


# ═════════════════════════════════════════════════════════════════════════
# get_supersession_chain
# ═════════════════════════════════════════════════════════════════════════

class TestGetSupersessionChain:
    def test_returns_chain_in_order(self):
        from app.subia.belief import neo4j_mirror

        chain_payload = [
            {"belief_id": "b-1", "domain": "user_model", "confidence": 0.4, "belief_status": "RETRACTED"},
            {"belief_id": "b-2", "domain": "user_model", "confidence": 0.6, "belief_status": "RETRACTED"},
            {"belief_id": "b-3", "domain": "user_model", "confidence": 0.8, "belief_status": "ACTIVE"},
        ]
        with _mock_driver() as (_, session):
            row = MagicMock()
            row.__getitem__ = MagicMock(return_value=chain_payload)
            row.get = MagicMock(return_value=chain_payload)
            session.run.return_value.single.return_value = row
            result = neo4j_mirror.get_supersession_chain("b-1")
        assert result == chain_payload
        cypher = session.run.call_args[0][0]
        assert "MATCH path = (start:Belief {belief_id: $belief_id})" in cypher
        assert "[:SUPERSEDED_BY*0..20]" in cypher

    def test_no_match_returns_empty(self):
        from app.subia.belief import neo4j_mirror
        with _mock_driver() as (_, session):
            session.run.return_value.single.return_value = None
            result = neo4j_mirror.get_supersession_chain("missing")
        assert result == []

    def test_empty_belief_id(self):
        from app.subia.belief import neo4j_mirror
        assert neo4j_mirror.get_supersession_chain("") == []

    def test_unavailable_neo4j(self):
        from app.subia.belief import neo4j_mirror
        neo4j_mirror._reset_for_tests()
        with patch.object(neo4j_mirror, "_get_driver", return_value=None):
            assert neo4j_mirror.get_supersession_chain("b-1") == []
        neo4j_mirror._reset_for_tests()

    def test_max_depth_threaded_into_query(self):
        from app.subia.belief import neo4j_mirror
        with _mock_driver() as (_, session):
            session.run.return_value.single.return_value = None
            neo4j_mirror.get_supersession_chain("b-1", max_depth=5)
        cypher = session.run.call_args[0][0]
        assert "[:SUPERSEDED_BY*0..5]" in cypher


# ═════════════════════════════════════════════════════════════════════════
# BeliefStore wiring
# ═════════════════════════════════════════════════════════════════════════

class TestBeliefStoreMirrorWiring:
    def test_persist_belief_calls_mirror(self):
        from app.subia.belief.store import BeliefStore, Belief

        store = BeliefStore()
        belief = Belief(content="x", domain="world_model", confidence=0.6)
        with patch("app.subia.belief.neo4j_mirror.mirror_belief", return_value=True) as m:
            store._persist_belief(belief)
        m.assert_called_once()
        kwargs = m.call_args.kwargs
        assert kwargs["domain"] == "world_model"
        assert kwargs["confidence"] == 0.6
        assert kwargs["belief_status"] == "ACTIVE"

    def test_persist_belief_neo4j_failure_does_not_raise(self):
        """SQL primary path must survive a Neo4j blow-up."""
        from app.subia.belief.store import BeliefStore, Belief

        store = BeliefStore()
        belief = Belief(content="x", domain="world_model", confidence=0.6)
        with patch("app.subia.belief.neo4j_mirror.mirror_belief",
                   side_effect=RuntimeError("neo4j down")):
            store._persist_belief(belief)  # must not raise

    def test_retract_belief_writes_status_and_supersession(self):
        from app.subia.belief.store import BeliefStore

        store = BeliefStore()
        with patch("app.subia.belief.neo4j_mirror.mirror_belief", return_value=True) as mb, \
             patch("app.subia.belief.neo4j_mirror.mirror_supersession", return_value=True) as ms:
            update = store.retract_belief("old", "stale", replacement_id="new")

        assert update is not None
        mb.assert_called_once_with("old", belief_status="RETRACTED")
        ms.assert_called_once_with("old", "new")

    def test_retract_belief_no_replacement_skips_supersession(self):
        from app.subia.belief.store import BeliefStore

        store = BeliefStore()
        with patch("app.subia.belief.neo4j_mirror.mirror_belief", return_value=True) as mb, \
             patch("app.subia.belief.neo4j_mirror.mirror_supersession", return_value=True) as ms:
            store.retract_belief("old", "stale")
        mb.assert_called_once()
        ms.assert_not_called()

    def test_get_supersession_chain_delegates_to_mirror(self):
        from app.subia.belief.store import BeliefStore
        store = BeliefStore()
        with patch("app.subia.belief.neo4j_mirror.get_supersession_chain",
                   return_value=[{"belief_id": "x"}]) as g:
            result = store.get_supersession_chain("x", max_depth=10)
        g.assert_called_once_with("x", max_depth=10)
        assert result == [{"belief_id": "x"}]

    def test_get_supersession_chain_returns_empty_on_error(self):
        from app.subia.belief.store import BeliefStore
        store = BeliefStore()
        with patch("app.subia.belief.neo4j_mirror.get_supersession_chain",
                   side_effect=RuntimeError("boom")):
            assert store.get_supersession_chain("x") == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
