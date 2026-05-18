"""Tests for the Q18 CR spam cleanup tool (PROGRAM §57)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    from app.change_requests import store
    monkeypatch.setattr(store, "_STORE_DIR", tmp_path / "change_requests")
    monkeypatch.setattr(store, "_AUDIT_LOG",
                         tmp_path / "change_requests" / "audit.jsonl")
    store.reset_for_tests()
    yield
    store.reset_for_tests()


@pytest.fixture
def fake_validator(monkeypatch):
    from app.change_requests import lifecycle
    from app.change_requests.validator import ValidationResult
    def _ok(**kwargs):
        return ValidationResult(ok=True, reason="", is_tier_immutable=False)
    monkeypatch.setattr(lifecycle, "validate", _ok)
    monkeypatch.setattr(lifecycle, "validate_auto_apply", _ok)
    yield


def test_consolidate_collapses_duplicates(fake_validator):
    """Replicate the 2026-05-16 incident pattern: 100 identical CRs
    from local_only_drill. After consolidate(), 1 canonical with
    recurrence_count=99 remains; the other 99 are archived."""
    from app.change_requests import store, lifecycle
    from app.change_requests.spam_cleanup import consolidate
    # Build 100 legacy-shape duplicates BYPASSING the Q3 dedup (because
    # the dedup is exactly what we already pinned in test_cr_lifecycle_dedup;
    # here we want to simulate "legacy data with no content_hash").
    for i in range(100):
        from app.change_requests.models import ChangeRequest, Status
        cr = ChangeRequest(
            id=f"cr_{i:03d}",
            created_at=f"2026-05-1{i // 50}T{i:02d}:00:00+00:00",
            requestor="local_only_drill",
            path="docs/RESILIENCE_DRILLS.md",
            new_content="",
            old_content="",
            reason=f"vendor failure batch {i}",
            diff="",
            status=Status.PENDING,
        )
        store.save(cr)
    assert len(store.list_all()) == 100

    summary = consolidate("local_only_drill")
    assert summary["n_groups"] == 1
    assert summary["n_consolidated"] == 1
    assert summary["n_archived"] == 99
    # 1 canonical remains in the index
    pendings = store.list_all(status=Status.PENDING)
    assert len(pendings) == 1
    canonical = pendings[0]
    assert canonical.recurrence_count == 99
    assert canonical.content_hash is not None


def test_consolidate_does_nothing_when_already_unique(fake_validator):
    """No duplicates → no-op."""
    from app.change_requests import store
    from app.change_requests.spam_cleanup import consolidate
    from app.change_requests.models import ChangeRequest, Status
    cr = ChangeRequest(
        id="solo",
        created_at="2026-05-18T00:00:00+00:00",
        requestor="local_only_drill",
        path="p.md",
        new_content="x", old_content="", reason="r", diff="",
        status=Status.PENDING,
    )
    store.save(cr)
    summary = consolidate("local_only_drill")
    assert summary["n_groups"] == 0
    assert summary["n_archived"] == 0


def test_consolidate_skips_terminal_crs(fake_validator):
    """Only PENDING CRs are subject to consolidation."""
    from app.change_requests import store
    from app.change_requests.spam_cleanup import consolidate
    from app.change_requests.models import ChangeRequest, Status
    for i, status in enumerate([Status.PENDING, Status.REJECTED, Status.APPLIED]):
        cr = ChangeRequest(
            id=f"x{i}", created_at=f"2026-05-18T0{i}:00:00+00:00",
            requestor="dr", path="p.md",
            new_content="x", old_content="", reason="r", diff="",
            status=status,
        )
        store.save(cr)
    summary = consolidate("dr")
    # Only 1 PENDING, so no group has >1 members → no consolidation
    assert summary["n_archived"] == 0
    # Terminal CRs are untouched
    assert store.get("x1").status == Status.REJECTED
    assert store.get("x2").status == Status.APPLIED


def test_consolidate_preserves_oldest_as_canonical(fake_validator):
    """The oldest-created CR becomes canonical; newer ones get archived."""
    from app.change_requests import store
    from app.change_requests.spam_cleanup import consolidate
    from app.change_requests.models import ChangeRequest, Status
    for i in range(3):
        cr = ChangeRequest(
            id=f"cr_{i}",
            created_at=f"2026-05-1{i + 5}T00:00:00+00:00",  # i=0 oldest
            requestor="r", path="p.md",
            new_content="x", old_content="", reason="r", diff="",
            status=Status.PENDING,
        )
        store.save(cr)
    consolidate("r")
    pendings = store.list_all(status=Status.PENDING)
    assert len(pendings) == 1
    # The oldest (cr_0) survived
    assert pendings[0].id == "cr_0"


def test_consolidate_idempotent(fake_validator):
    """Running consolidate twice yields the same single canonical (no
    double-archive)."""
    from app.change_requests import store
    from app.change_requests.spam_cleanup import consolidate
    from app.change_requests.models import ChangeRequest, Status
    for i in range(10):
        cr = ChangeRequest(
            id=f"x_{i}",
            created_at=f"2026-05-18T{i:02d}:00:00+00:00",
            requestor="r", path="p.md",
            new_content="x", old_content="", reason="r", diff="",
            status=Status.PENDING,
        )
        store.save(cr)
    consolidate("r")
    initial = store.list_all(status=Status.PENDING)
    assert len(initial) == 1
    initial_rec = initial[0].recurrence_count
    # Second run is a no-op
    summary = consolidate("r")
    assert summary["n_archived"] == 0
    assert summary["n_consolidated"] == 0
    after = store.list_all(status=Status.PENDING)
    assert len(after) == 1
    assert after[0].recurrence_count == initial_rec


def test_archive_dir_under_change_requests(fake_validator, tmp_path):
    """Archived files land in workspace/change_requests/archive/<ts>_drill_spam/."""
    from app.change_requests import store
    from app.change_requests.spam_cleanup import consolidate
    from app.change_requests.models import ChangeRequest, Status
    for i in range(3):
        cr = ChangeRequest(
            id=f"a_{i}", created_at=f"2026-05-18T{i:02d}:00:00+00:00",
            requestor="r", path="p.md", new_content="x",
            old_content="", reason="r", diff="", status=Status.PENDING,
        )
        store.save(cr)
    summary = consolidate("r")
    archive_dir = (tmp_path / "change_requests" / "archive")
    assert archive_dir.exists()
    archived_files = list(archive_dir.rglob("a_*.json"))
    assert len(archived_files) == 2  # 3 originals → 1 canonical + 2 archived
