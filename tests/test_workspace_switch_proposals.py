"""Tests for app.workspace_switch_proposals + ProjectManager sticky-user pick.

Background — 2026-05-02 the user reported their explicit
``switch workspace to eesti mets`` was being silently overridden on
every subsequent message: keywords like "estonia" / "event" / "ticket"
in their forest-related messages triggered PLG's keyword list and the
auto-detector switched the active workspace, causing tickets to file
under PLG.

Two layered fixes covered here:

  1. ``ProjectManager.switch(..., source=...)`` — refuses to overwrite
     an explicit "user" pick from "auto" callers.

  2. ``workspace_switch_proposals`` — when auto-detect differs from a
     sticky user pick, send the user a Signal asking for 👍/👎 instead
     of switching silently.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Sticky-user pick guard ───────────────────────────────────────────


class TestProjectManagerStickyUserPick:

    def test_user_pick_blocks_subsequent_auto_switch(self, monkeypatch):
        """A switch with source='user' must not be overwritten by
        a later switch with source='auto'."""
        from app.control_plane.projects import ProjectManager
        pm = ProjectManager()
        # Reset class-level state to known
        pm._active_project_id = None
        pm._active_project_source = None

        eesti_mets = {"id": "eesti-id", "name": "Eesti mets", "mission": ""}
        plg = {"id": "plg-id", "name": "PLG", "mission": ""}

        with patch.object(pm, "get_by_name") as mock_get:
            with patch("app.project_isolation.get_manager") as _:
                # 1. User explicitly switches
                mock_get.return_value = eesti_mets
                pm.switch("eesti mets", source="user")
                assert pm._active_project_id == "eesti-id"
                assert pm._active_project_source == "user"

                # 2. Auto-detector tries to override
                mock_get.return_value = plg
                pm.switch("plg", source="auto")
                # Must STILL be on eesti-id
                assert pm._active_project_id == "eesti-id"
                assert pm._active_project_source == "user"

    def test_auto_pick_can_be_overwritten_by_user(self, monkeypatch):
        """auto → user is fine — the user is correcting the detector."""
        from app.control_plane.projects import ProjectManager
        pm = ProjectManager()
        pm._active_project_id = None
        pm._active_project_source = None

        eesti_mets = {"id": "eesti-id", "name": "Eesti mets", "mission": ""}
        plg = {"id": "plg-id", "name": "PLG", "mission": ""}

        with patch.object(pm, "get_by_name") as mock_get:
            with patch("app.project_isolation.get_manager") as _:
                # auto sets first
                mock_get.return_value = plg
                pm.switch("plg", source="auto")
                assert pm._active_project_id == "plg-id"

                # user explicitly overrides
                mock_get.return_value = eesti_mets
                pm.switch("eesti mets", source="user")
                assert pm._active_project_id == "eesti-id"
                assert pm._active_project_source == "user"

    def test_auto_can_overwrite_auto(self, monkeypatch):
        """No explicit user pick → auto → auto is fine (no stickiness)."""
        from app.control_plane.projects import ProjectManager
        pm = ProjectManager()
        pm._active_project_id = None
        pm._active_project_source = None

        plg = {"id": "plg-id", "name": "PLG", "mission": ""}
        kaicart = {"id": "kc-id", "name": "KaiCart", "mission": ""}

        with patch.object(pm, "get_by_name") as mock_get:
            with patch("app.project_isolation.get_manager") as _:
                mock_get.return_value = plg
                pm.switch("plg", source="auto")
                mock_get.return_value = kaicart
                pm.switch("kaicart", source="auto")
                assert pm._active_project_id == "kc-id"

    def test_user_can_overwrite_user(self, monkeypatch):
        """user → user is fine — rapid successive explicit switches
        should not be sticky against each other."""
        from app.control_plane.projects import ProjectManager
        pm = ProjectManager()
        pm._active_project_id = None
        pm._active_project_source = None

        eesti = {"id": "eesti-id", "name": "Eesti mets", "mission": ""}
        plg = {"id": "plg-id", "name": "PLG", "mission": ""}

        with patch.object(pm, "get_by_name") as mock_get:
            with patch("app.project_isolation.get_manager") as _:
                mock_get.return_value = eesti
                pm.switch("eesti mets", source="user")
                mock_get.return_value = plg
                pm.switch("plg", source="user")
                assert pm._active_project_id == "plg-id"


# ── Proposal queue + ask flow ────────────────────────────────────────


@pytest.fixture
def isolated_queue(tmp_path, monkeypatch):
    """Redirect the queue file to a temp path so tests don't pollute
    the workspace JSON."""
    from app import workspace_switch_proposals as wsp
    qpath = tmp_path / "wsp.json"
    monkeypatch.setattr(wsp, "_QUEUE_PATH", qpath)
    return qpath


class TestProposeQueue:

    def test_propose_persists_pending_entry(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        notifier_calls = []
        def _notifier(sender, msg):
            notifier_calls.append((sender, msg))
            return 1234567890  # signal_ts
        pid = wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=_notifier,
        )
        assert pid is not None
        assert len(notifier_calls) == 1
        # Check the message body conveys both names
        _, msg = notifier_calls[0]
        assert "PLG" in msg
        assert "Eesti mets" in msg
        # And the queue was persisted
        entries = json.loads(isolated_queue.read_text())
        assert len(entries) == 1
        assert entries[0]["proposal_id"] == pid
        assert entries[0]["decision"] == "pending"
        assert entries[0]["signal_timestamp"] == 1234567890

    def test_propose_returns_none_on_send_failure(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        def _broken_notifier(sender, msg):
            raise RuntimeError("Signal down")
        pid = wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=_broken_notifier,
        )
        assert pid is None
        # Nothing persisted on failure
        assert not isolated_queue.exists() or json.loads(isolated_queue.read_text()) == []

    def test_find_by_signal_ts(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        pid = wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 999,
        )
        assert wsp.find_by_signal_ts(999) == pid
        # Wrong ts → None
        assert wsp.find_by_signal_ts(1) is None
        # Zero / falsy → None
        assert wsp.find_by_signal_ts(0) is None

    def test_has_recent_decision_pending(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 1,
        )
        assert wsp.has_recent_decision("PLG", "+372")
        # Different sender → False
        assert not wsp.has_recent_decision("PLG", "+999")
        # Different detection → False
        assert not wsp.has_recent_decision("KaiCart", "+372")

    def test_has_recent_decision_expires(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        # Inject an old pending entry
        wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 1,
            now=time.time() - 7200,  # 2h ago — past the 30 min PENDING_TTL
        )
        assert not wsp.has_recent_decision("PLG", "+372")


class TestProposalDecisions:

    def test_accept_calls_switch_with_source_user(self, isolated_queue, monkeypatch):
        from app import workspace_switch_proposals as wsp
        pid = wsp.propose(
            detected_name="Eesti mets", current_name="default",
            sender="+372", notifier=lambda s, m: 1,
        )
        # Mock the project switch
        switch_calls = []
        class _FakeProjects:
            def switch(self, name, *, source="user"):
                switch_calls.append({"name": name, "source": source})
                return {"id": "eesti-id", "name": "Eesti mets"}
        monkeypatch.setattr(
            "app.control_plane.projects.get_projects",
            lambda: _FakeProjects(),
        )
        result = wsp.accept(pid)
        assert "Eesti mets" in result
        # Critical: source MUST be "user" so the switch is sticky
        assert len(switch_calls) == 1
        assert switch_calls[0]["source"] == "user"
        assert switch_calls[0]["name"] == "Eesti mets"
        # State updated
        entries = json.loads(isolated_queue.read_text())
        assert entries[0]["decision"] == "accepted"

    def test_decline_marks_declined_and_suppresses_reasking(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        pid = wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 1,
        )
        wsp.decline(pid)
        # Second proposal for same (sender, name) should be suppressed
        # within DECLINE_TTL_S
        assert wsp.has_recent_decision("PLG", "+372")
        # State persisted
        entries = json.loads(isolated_queue.read_text())
        assert entries[0]["decision"] == "declined"

    def test_accept_idempotent_on_already_decided(self, isolated_queue, monkeypatch):
        from app import workspace_switch_proposals as wsp
        pid = wsp.propose(
            detected_name="Eesti mets", current_name="default",
            sender="+372", notifier=lambda s, m: 1,
        )
        wsp.decline(pid)
        # Trying to accept an already-declined proposal → safe no-op
        result = wsp.accept(pid)
        assert "already declined" in result.lower()


class TestExpireStale:

    def test_expires_old_pending(self, isolated_queue):
        from app import workspace_switch_proposals as wsp
        wsp.propose(
            detected_name="PLG", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 1,
            now=time.time() - 7200,  # 2h old
        )
        wsp.propose(
            detected_name="KaiCart", current_name="Eesti mets",
            sender="+372", notifier=lambda s, m: 2,
            now=time.time(),  # fresh
        )
        n = wsp.expire_stale()
        assert n == 1
        entries = json.loads(isolated_queue.read_text())
        # Old one is now expired, fresh one still pending
        decisions = sorted(e["decision"] for e in entries)
        assert decisions == ["expired", "pending"]
