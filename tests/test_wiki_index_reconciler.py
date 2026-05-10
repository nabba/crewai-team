"""Tests for app.memory.wiki_index_reconciler — drift detector + shadow rebuild."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect WIKI_ROOT and DREAMS_ROOT to a tmp dir and return helpers."""
    wiki_root = tmp_path / "wiki"
    dreams_root = tmp_path / "dreams"
    wiki_root.mkdir()
    dreams_root.mkdir()

    # Seed the section directories so the canonical content matches the
    # production layout. `_compute_master_index_content` walks these.
    for section in ("meta", "self", "philosophy", "plg", "archibal", "kaicart"):
        (wiki_root / section).mkdir()

    # Patch all paths that capture WIKI_ROOT / DREAMS_ROOT at import time.
    from app import paths as _paths
    from app.tools import wiki_tools as _wt
    from app.memory import wiki_index_reconciler as _r

    monkeypatch.setattr(_wt, "WIKI_ROOT", str(wiki_root))
    monkeypatch.setattr(_r, "WIKI_ROOT", str(wiki_root))
    monkeypatch.setattr(_paths, "DREAMS_ROOT", dreams_root)
    monkeypatch.setattr(_paths, "WIKI_INDEX_CANDIDATE",
                        dreams_root / "wiki_index.candidate.md")
    monkeypatch.setattr(_paths, "WIKI_INDEX_AUDIT",
                        dreams_root / "wiki_index_audit.jsonl")
    monkeypatch.setattr(_r, "DREAMS_ROOT", dreams_root)
    monkeypatch.setattr(_r, "WIKI_INDEX_CANDIDATE",
                        dreams_root / "wiki_index.candidate.md")
    monkeypatch.setattr(_r, "WIKI_INDEX_AUDIT",
                        dreams_root / "wiki_index_audit.jsonl")
    monkeypatch.setattr(_r, "_LOCK_PATH",
                        dreams_root / ".wiki_index_reconciler.lock")

    # Block change-request creation by default — tests opt in explicitly.
    monkeypatch.setattr(
        "app.memory.wiki_index_reconciler._open_change_request",
        lambda **kw: None,
    )

    return {
        "wiki": wiki_root,
        "dreams": dreams_root,
        "candidate": dreams_root / "wiki_index.candidate.md",
        "audit": dreams_root / "wiki_index_audit.jsonl",
        "lock": dreams_root / ".wiki_index_reconciler.lock",
    }


def _make_page(section_dir: Path, slug: str, *, title: str, status: str = "active"):
    """Write a minimal frontmatter page that the canonical compute reads."""
    fm = (
        "---\n"
        f"title: {title}\n"
        f"section: {section_dir.name}\n"
        f"status: {status}\n"
        "created_at: '2026-01-01T00:00:00Z'\n"
        "updated_at: '2026-01-01T00:00:00Z'\n"
        "author: test\n"
        "confidence: medium\n"
        "tags: []\n"
        "related: []\n"
        "source: test\n"
        "---\n"
        f"# {title}\n"
    )
    (section_dir / f"{slug}.md").write_text(fm)


def _write_live_index(wiki_root: Path, content: str) -> None:
    (wiki_root / "index.md").write_text(content)


# ── Pure helpers ─────────────────────────────────────────────────────────


def test_normalize_for_hashing_strips_updated_at(isolated_dirs):
    from app.memory.wiki_index_reconciler import _normalize_for_hashing

    a = "---\ntitle: X\nupdated_at: '2026-05-08T12:00:00Z'\nfoo: bar\n---\nbody"
    b = "---\ntitle: X\nupdated_at: '2099-12-31T23:59:59Z'\nfoo: bar\n---\nbody"

    assert _normalize_for_hashing(a) == _normalize_for_hashing(b)


def test_content_hash_changes_on_real_drift(isolated_dirs):
    from app.memory.wiki_index_reconciler import _content_hash

    a = "---\ntotal_pages: 5\n---\n# Wiki"
    b = "---\ntotal_pages: 6\n---\n# Wiki"

    assert _content_hash(a) != _content_hash(b)


def test_compute_canonical_includes_pages(isolated_dirs):
    from app.memory.wiki_index_reconciler import compute_canonical_master_content

    _make_page(isolated_dirs["wiki"] / "meta", "page-one", title="Page One")
    _make_page(isolated_dirs["wiki"] / "meta", "page-two", title="Page Two")

    content = compute_canonical_master_content()

    assert "Page One" in content
    assert "Page Two" in content
    assert "[[meta/page-one]]" in content
    assert "[[meta/page-two]]" in content
    assert "total_pages: 2" in content


def test_compute_canonical_marks_deprecated(isolated_dirs):
    from app.memory.wiki_index_reconciler import compute_canonical_master_content

    _make_page(isolated_dirs["wiki"] / "meta", "old", title="Old", status="deprecated")
    content = compute_canonical_master_content()

    assert "*(deprecated)*" in content


# ── No-drift path ────────────────────────────────────────────────────────


def test_no_drift_when_live_matches_canonical(isolated_dirs):
    from app.memory.wiki_index_reconciler import (
        compute_canonical_master_content,
        run_reconciler,
    )

    # Live index matches what the canonical compute would produce.
    canonical = compute_canonical_master_content()
    _write_live_index(isolated_dirs["wiki"], canonical)

    result = run_reconciler()

    assert not result.drift_detected
    assert result.live_hash == result.canonical_hash
    assert result.audit_id is None
    assert not isolated_dirs["candidate"].exists()
    assert not isolated_dirs["audit"].exists()


def test_no_drift_ignores_updated_at_timestamp(isolated_dirs):
    """A stale timestamp alone must not trigger drift — only structural
    changes count."""
    from app.memory.wiki_index_reconciler import (
        compute_canonical_master_content,
        run_reconciler,
    )

    # Live = canonical with an old timestamp swapped in.
    canonical = compute_canonical_master_content()
    live = canonical.replace(
        "updated_at: '2000-01-01T00:00:00Z'",
        "updated_at: '1999-01-01T00:00:00Z'",
    )
    _write_live_index(isolated_dirs["wiki"], live)

    result = run_reconciler()

    assert not result.drift_detected, (
        "timestamp-only difference must not trigger drift"
    )


def test_no_drift_ignores_body_last_updated_date(isolated_dirs):
    """Regression: ``_compute_master_index_content`` writes today's date
    into the BODY at line ``Total pages: N | Last updated: YYYY-MM-DD``.
    Without normalising that line, the canonical compute (which uses
    the pin date 2000-01-01) hashed differently from the live file
    (which uses today's date), creating false-positive drift on every
    run. This test pins the bug shut.
    """
    from app.memory.wiki_index_reconciler import (
        compute_canonical_master_content,
        run_reconciler,
    )

    _make_page(isolated_dirs["wiki"] / "meta", "p1", title="P1")

    # Compute canonical (pin-dated body), then synthesise a live file
    # with today's date in the body — the production scenario.
    canonical = compute_canonical_master_content()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    live = re.sub(
        r"Last updated: \S+",
        f"Last updated: {today}",
        canonical,
    )
    # Also bump the frontmatter timestamp like a real recently-rebuilt file.
    live = live.replace(
        "updated_at: '2000-01-01T00:00:00Z'",
        f"updated_at: '{today}T12:00:00Z'",
    )
    _write_live_index(isolated_dirs["wiki"], live)

    result = run_reconciler()

    assert not result.drift_detected, (
        "body 'Last updated: YYYY-MM-DD' difference alone must not "
        "trigger drift — this is the bug pattern that produced 335 "
        "stuck CRs in May 2026"
    )


# ── Drift path ───────────────────────────────────────────────────────────


def test_drift_detected_when_page_added_out_of_band(isolated_dirs):
    """Live index reflects 1 page; on disk there are 3. Reconciler should
    detect drift, write candidate, append audit entry."""
    from app.memory.wiki_index_reconciler import (
        compute_canonical_master_content,
        run_reconciler,
    )

    # Snapshot the canonical with 1 page, that's the "live" state.
    _make_page(isolated_dirs["wiki"] / "meta", "first", title="First")
    one_page_canonical = compute_canonical_master_content()
    _write_live_index(isolated_dirs["wiki"], one_page_canonical)

    # Now add 2 more pages out-of-band.
    _make_page(isolated_dirs["wiki"] / "meta", "second", title="Second")
    _make_page(isolated_dirs["wiki"] / "meta", "third", title="Third")

    result = run_reconciler()

    assert result.drift_detected
    assert result.live_hash != result.canonical_hash
    assert result.audit_id is not None
    assert isolated_dirs["candidate"].exists()
    assert isolated_dirs["audit"].exists()

    # Candidate has the new pages.
    cand = isolated_dirs["candidate"].read_text()
    assert "[[meta/second]]" in cand
    assert "[[meta/third]]" in cand
    assert "total_pages: 3" in cand


def test_drift_detected_when_live_index_missing(isolated_dirs):
    from app.memory.wiki_index_reconciler import run_reconciler

    _make_page(isolated_dirs["wiki"] / "meta", "p1", title="P1")
    # No live index file written.

    result = run_reconciler()

    assert result.drift_detected
    assert result.live_size_bytes == 0
    assert result.canonical_size_bytes > 0


# ── Audit chain (`superseded_by` invariant) ──────────────────────────────


def test_audit_chains_via_supersedes(isolated_dirs):
    """Each new audit entry references the previous one. Old entries are
    never deleted (superseded_by invariant lifted from skill consolidator)."""
    from app.memory.wiki_index_reconciler import run_reconciler

    # Drift run #1
    _make_page(isolated_dirs["wiki"] / "meta", "a", title="A")
    r1 = run_reconciler()

    # Drift run #2 — add another page to keep drift present.
    _make_page(isolated_dirs["wiki"] / "meta", "b", title="B")
    r2 = run_reconciler()

    audit_lines = isolated_dirs["audit"].read_text().splitlines()
    assert len(audit_lines) == 2

    e1 = json.loads(audit_lines[0])
    e2 = json.loads(audit_lines[1])

    assert e1["supersedes"] is None      # first entry has no predecessor
    assert e2["supersedes"] == e1["id"]  # second references first
    assert e2["id"] == r2.audit_id


# ── Lock semantics ───────────────────────────────────────────────────────


def test_concurrent_run_skips_when_lock_held(isolated_dirs):
    from app.memory.wiki_index_reconciler import run_reconciler

    # Touch the lock as if another runner held it.
    isolated_dirs["lock"].write_text("99999")

    _make_page(isolated_dirs["wiki"] / "meta", "p", title="P")

    result = run_reconciler()

    assert result.skipped
    assert "lock" in (result.skip_reason or "")
    assert not isolated_dirs["candidate"].exists()


def test_stale_lock_is_taken_over(isolated_dirs):
    """A lock older than the staleness window is reclaimed."""
    from app.memory.wiki_index_reconciler import (
        _LOCK_STALENESS_SECONDS,
        run_reconciler,
    )

    # Write a stale lock — mtime well beyond staleness window.
    isolated_dirs["lock"].write_text("99999")
    stale_age = _LOCK_STALENESS_SECONDS + 60
    past = time.time() - stale_age
    import os
    os.utime(isolated_dirs["lock"], (past, past))

    _make_page(isolated_dirs["wiki"] / "meta", "p", title="P")

    result = run_reconciler()

    assert not result.skipped
    assert result.drift_detected


# ── Module-level smoke ──────────────────────────────────────────────────


def test_run_reconciler_releases_lock_on_completion(isolated_dirs):
    from app.memory.wiki_index_reconciler import run_reconciler

    _make_page(isolated_dirs["wiki"] / "meta", "p", title="P")
    run_reconciler()

    assert not isolated_dirs["lock"].exists()


def test_drift_result_to_dict_is_json_safe(isolated_dirs):
    from app.memory.wiki_index_reconciler import run_reconciler

    _make_page(isolated_dirs["wiki"] / "meta", "p", title="P")
    result = run_reconciler()

    payload = json.dumps(result.to_dict())
    assert "drift_detected" in payload


# ── CR dedup (defensive layer) ──────────────────────────────────────────


class _FakeCR:
    """Minimal stand-in for ``ChangeRequest`` honoring the duck-typed
    fields the dedup helper reads."""
    def __init__(self, *, id, status, new_content, decided_at=None,
                 path="wiki/index.md"):
        self.id = id
        self.status = status
        self.new_content = new_content
        self.decided_at = decided_at
        self.path = path


def test_dedup_blocks_when_pending_cr_exists(monkeypatch):
    """A non-terminal CR for wiki/index.md suppresses new filings —
    duplicate CRs would spam the operator without adding signal.

    Uses ``monkeypatch`` directly (no ``isolated_dirs``) because this
    helper doesn't touch the filesystem; mocking ``list_all`` is the
    only setup needed.
    """
    from app.change_requests import Status
    from app.memory.wiki_index_reconciler import _existing_cr_blocks_filing

    pending = _FakeCR(
        id="cr_pend", status=Status.PENDING, new_content="any content",
    )
    monkeypatch.setattr(
        "app.change_requests.list_all", lambda **kw: [pending],
    )

    blocked = _existing_cr_blocks_filing(candidate_content="<new>")
    assert blocked is not None
    assert "non-terminal" in blocked


def test_dedup_blocks_when_recent_rejection_with_same_content(monkeypatch):
    """If the operator already rejected this exact content within the
    recent window, don't refile — honour the decision."""
    from datetime import datetime, timedelta, timezone
    from app.change_requests import Status
    from app.memory.wiki_index_reconciler import _existing_cr_blocks_filing

    rejected_recent = _FakeCR(
        id="cr_no",
        status=Status.REJECTED,
        new_content="<exact body>",
        decided_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
    )
    monkeypatch.setattr(
        "app.change_requests.list_all", lambda **kw: [rejected_recent],
    )

    blocked = _existing_cr_blocks_filing(candidate_content="<exact body>")
    assert blocked is not None
    assert "rejected" in blocked.lower()


def test_dedup_allows_filing_after_rejection_window(monkeypatch):
    """Old rejection (>7 days ago) does NOT block new filings — the
    operator might have changed their mind."""
    from datetime import datetime, timedelta, timezone
    from app.change_requests import Status
    from app.memory.wiki_index_reconciler import _existing_cr_blocks_filing

    old_rejection = _FakeCR(
        id="cr_old",
        status=Status.REJECTED,
        new_content="<exact body>",
        decided_at=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
    )
    monkeypatch.setattr(
        "app.change_requests.list_all", lambda **kw: [old_rejection],
    )

    assert _existing_cr_blocks_filing(candidate_content="<exact body>") is None


def test_dedup_allows_filing_after_rejection_with_different_content(monkeypatch):
    """A recent rejection of DIFFERENT content does not block — the
    proposal evolved, give it a fresh review."""
    from datetime import datetime, timedelta, timezone
    from app.change_requests import Status
    from app.memory.wiki_index_reconciler import _existing_cr_blocks_filing

    rejected_recent = _FakeCR(
        id="cr_old_body",
        status=Status.REJECTED,
        new_content="<old body>",
        decided_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
    )
    monkeypatch.setattr(
        "app.change_requests.list_all", lambda **kw: [rejected_recent],
    )

    assert _existing_cr_blocks_filing(candidate_content="<new body>") is None


def test_dedup_handles_cr_system_unavailable(monkeypatch):
    """If list_all raises, fail open — the reconciler retries next pass."""
    from app.memory.wiki_index_reconciler import _existing_cr_blocks_filing

    def boom(**_):
        raise RuntimeError("CR store unavailable")

    monkeypatch.setattr("app.change_requests.list_all", boom)

    assert _existing_cr_blocks_filing(candidate_content="<x>") is None


# ── Validator regression: wiki/index.md now passes ──────────────────────


def test_validator_now_accepts_wiki_index_path():
    """Pinpoint regression: 335 stuck CRs were filed by the reconciler
    against ``wiki/index.md`` and ALL bounced because ``wiki/`` was not
    in ``_ALLOWED_ROOT_PREFIXES``. The validator now permits it.
    """
    from app.change_requests.validator import validate

    result = validate(
        path="wiki/index.md",
        new_content="---\ntitle: Wiki\n---\n# Wiki\n",
    )
    assert result.ok is True
    assert result.reason is None
