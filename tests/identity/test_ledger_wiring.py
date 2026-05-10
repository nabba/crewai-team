"""Tests for continuity-ledger emission wired into existing subsystems.

Each test patches ``app.identity.continuity_ledger.record_event`` and
exercises the upstream call site to confirm the event fires with the
right kind, actor, and detail. These are *wiring* tests — the ledger
module's own behaviour is covered in ``test_continuity_ledger.py``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.identity import continuity_ledger as cl


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "ledger.jsonl"


def _make_cr(id_: str, requestor: str, path: str):
    """Build a ChangeRequest with the minimum mandatory fields for tests."""
    from app.change_requests.models import ChangeRequest, Status
    return ChangeRequest(
        id=id_,
        created_at="2026-05-10T00:00:00+00:00",
        requestor=requestor,
        path=path,
        new_content="new",
        old_content="old",
        reason="test",
        diff="--- old\n+++ new\n",
        status=Status.APPROVED,
    )


def test_subia_integrity_regen_emits(
    tmp_path: Path,
    ledger_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SubIA write_manifest fires an integrity_regen event."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.subia.integrity import write_manifest

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    target = tmp_path / "manifest.json"
    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        write_manifest(
            {"files": {"a.py": "abc", "b.py": "def"}, "version": 1},
            manifest_path=target,
        )

    assert target.exists()
    assert len(captured) == 1
    assert captured[0]["kind"] == "integrity_regen"
    assert captured[0]["actor"] == "dev_or_ci"
    assert "2 files" in captured[0]["summary"]
    assert captured[0]["detail"] == {"n_files": 2}


def test_governance_ratchet_up_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """set_ratchet fires a governance_ratchet event with direction=up."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.governance_ratchet import protocol, store

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    fake_state = type("FakeState", (), {"current": 0.96, "history": []})()
    monkeypatch.setattr(store, "get", lambda name: fake_state)
    monkeypatch.setattr(store, "set_one", lambda *a, **k: None)
    monkeypatch.setattr(
        protocol.audit, "append",
        lambda **kw: "fake_chain_root",
    )

    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        protocol.set_ratchet(
            name="safety_minimum",
            new_value=0.98,
            source="operator_react",
            reason="raise after track record",
        )

    assert len(captured) == 1
    assert captured[0]["kind"] == "governance_ratchet"
    assert captured[0]["actor"] == "operator_react"
    assert captured[0]["detail"]["direction"] == "up"
    assert captured[0]["detail"]["old_value"] == 0.96
    assert captured[0]["detail"]["new_value"] == 0.98


def test_governance_ratchet_down_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """relax_ratchet fires a governance_ratchet event with direction=down."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.governance_ratchet import protocol, store

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    fake_state = type("FakeState", (), {"current": 0.98, "history": []})()
    monkeypatch.setattr(store, "get", lambda name: fake_state)
    monkeypatch.setattr(store, "set_one", lambda *a, **k: None)
    monkeypatch.setattr(
        protocol.audit, "append",
        lambda **kw: "fake_chain_root",
    )

    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        protocol.relax_ratchet(
            name="safety_minimum",
            new_value=0.96,
            source="operator_react",
            reason="emergency rollback",
        )

    assert len(captured) == 1
    assert captured[0]["kind"] == "governance_ratchet"
    assert captured[0]["detail"]["direction"] == "down"
    assert captured[0]["detail"]["floor"] >= 0.0


def test_change_request_soul_path_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """change_requests.mark_applied fires soul_edit when path matches."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.change_requests import lifecycle, store
    from app.change_requests.models import ChangeRequest, Status

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    cr = _make_cr("cr_test", "coder", "app/souls/coder.md")
    monkeypatch.setattr(store, "get", lambda rid: cr)
    monkeypatch.setattr(store, "save", lambda *a, **k: None)

    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        lifecycle.mark_applied(
            "cr_test",
            git_branch="cr/test",
            git_commit_sha="abc1234",
            pr_url=None,
        )

    assert len(captured) == 1
    assert captured[0]["kind"] == "soul_edit"
    assert captured[0]["actor"] == "coder"
    assert captured[0]["detail"]["path"] == "app/souls/coder.md"


def test_change_request_constitution_path_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The constitution.md path triggers a soul_edit event too."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.change_requests import lifecycle, store
    from app.change_requests.models import ChangeRequest, Status

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    cr = _make_cr("cr_const", "self_improver", "wiki/governance/constitution.md")
    monkeypatch.setattr(store, "get", lambda rid: cr)
    monkeypatch.setattr(store, "save", lambda *a, **k: None)

    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        lifecycle.mark_applied(
            "cr_const",
            git_branch="cr/const",
            git_commit_sha="def5678",
            pr_url=None,
        )

    assert len(captured) == 1
    assert captured[0]["kind"] == "soul_edit"


def test_change_request_unrelated_path_does_not_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-soul path doesn't emit a soul_edit event."""
    monkeypatch.setenv("IDENTITY_LEDGER_ENABLED", "true")

    from app.change_requests import lifecycle, store
    from app.change_requests.models import ChangeRequest, Status

    captured: list[dict] = []

    def fake_record(**kwargs):
        captured.append(kwargs)
        return True

    cr = _make_cr("cr_other", "coder", "app/agents/pim_agent.py")
    monkeypatch.setattr(store, "get", lambda rid: cr)
    monkeypatch.setattr(store, "save", lambda *a, **k: None)

    with patch(
        "app.identity.continuity_ledger.record_event",
        side_effect=fake_record,
    ):
        lifecycle.mark_applied(
            "cr_other",
            git_branch="cr/other",
            git_commit_sha="000",
            pr_url=None,
        )

    assert captured == []


def test_is_soul_path_helper() -> None:
    """The path predicate matches expected identity-shaping paths."""
    from app.change_requests.lifecycle import _is_soul_path

    assert _is_soul_path("app/souls/coder.md") is True
    assert _is_soul_path("app/souls/researcher.md") is True
    assert _is_soul_path("wiki/governance/constitution.md") is True

    assert _is_soul_path("app/agents/coder.py") is False
    assert _is_soul_path("wiki/index.md") is False
    assert _is_soul_path("") is False
