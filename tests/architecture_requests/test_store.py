"""Tests for app.architecture_requests.store."""

from __future__ import annotations

from app.architecture_requests import store
from app.architecture_requests.models import ArchStatus
from .conftest import make_request


def test_save_then_get() -> None:
    req = make_request()
    store.save(req, audit_event="created")
    loaded = store.get(req.id)
    assert loaded is not None
    assert loaded.id == req.id
    assert loaded.intent == req.intent


def test_list_all_filters_by_status() -> None:
    a = make_request(intent="A")
    a.status = ArchStatus.PROPOSED
    b = make_request(intent="B")
    b.status = ArchStatus.APPROVED
    store.save(a, audit_event="created")
    store.save(b, audit_event="created")

    proposed = store.list_all(status=ArchStatus.PROPOSED)
    approved = store.list_all(status=ArchStatus.APPROVED)
    everything = store.list_all()

    assert {r.id for r in proposed} == {a.id}
    assert {r.id for r in approved} == {b.id}
    assert {r.id for r in everything} == {a.id, b.id}


def test_list_all_orders_newest_first() -> None:
    older = make_request()
    older.created_at = "2026-01-01T00:00:00+00:00"
    newer = make_request()
    newer.created_at = "2026-05-01T00:00:00+00:00"
    store.save(older, audit_event="created")
    store.save(newer, audit_event="created")

    out = store.list_all()
    assert [r.created_at for r in out] == [
        newer.created_at,
        older.created_at,
    ]


def test_find_by_signal_ts() -> None:
    req = make_request()
    req.signal_message_ts = 1747900000
    store.save(req, audit_event="created")

    assert store.find_by_signal_ts(1747900000) == req.id
    assert store.find_by_signal_ts(0) is None
    assert store.find_by_signal_ts(1747999999) is None


def test_audit_chain_through_rolled_log() -> None:
    req = make_request()
    store.save(req, audit_event="created")
    req.status = ArchStatus.APPROVED
    store.save(req, audit_event="approved")
    req.status = ArchStatus.SCAFFOLDED
    store.save(req, audit_event="scaffolded")

    events = [p["event"] for p in store.iter_audit_entries()]
    assert events == ["created", "approved", "scaffolded"]


def test_audit_payload_has_status_and_id() -> None:
    req = make_request()
    store.save(req, audit_event="created")

    payloads = list(store.iter_audit_entries())
    assert len(payloads) == 1
    assert payloads[0]["request_id"] == req.id
    assert payloads[0]["status"] == "proposed"
    assert payloads[0]["package_path"] == req.package_path


def test_save_without_audit_event_writes_no_audit_entry() -> None:
    req = make_request()
    store.save(req)
    assert list(store.iter_audit_entries()) == []
