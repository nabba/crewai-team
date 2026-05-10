"""Tests for app.inbox.router."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.inbox import router
from app.inbox.classifier import FileClassification


@pytest.fixture
def inbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("INBOX_INGESTION_ENABLED", "true")
    p = tmp_path / "inbox"
    p.mkdir()
    return p


def _stub_handler_ok(path: Path, classification: FileClassification, base: Path) -> str:
    return f"processed {classification.kind}"


def _stub_handler_fail(path: Path, classification: FileClassification, base: Path) -> str:
    raise RuntimeError("intentional failure")


def test_disabled_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INBOX_INGESTION_ENABLED", "false")
    p = tmp_path / "inbox"
    p.mkdir()
    r = router.scan_and_route(inbox_dir=p)
    assert r.status == "skipped_disabled"


def test_no_inbox_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INBOX_INGESTION_ENABLED", "true")
    r = router.scan_and_route(inbox_dir=tmp_path / "missing")
    assert r.status == "skipped_no_inbox"


def test_processes_text_file(inbox: Path, tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    notes_dir = tmp_path / "notes"
    monkeypatch.setenv("INBOX_NOTES_DIR", str(notes_dir))
    f = inbox / "note.md"
    f.write_text("hello", encoding="utf-8")
    # Force the file's mtime past the quiet window.
    past = time.time() - 60
    import os
    os.utime(f, (past, past))

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert len(r.processed) == 1
    assert r.processed[0]["kind"] == "text"
    assert not f.exists()  # archived
    archived = list((inbox / ".archive").rglob("note.md"))
    assert len(archived) == 1
    # Text handler now writes directly into the canonical notes root,
    # not a per-day subdir, so the React /cp/files view picks it up.
    assert (notes_dir / "note.md").exists()


def test_unknown_file_skipped(inbox: Path) -> None:
    f = inbox / "weird.xyz"
    f.write_bytes(b"\x00" * 32)
    past = time.time() - 60
    import os
    os.utime(f, (past, past))

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert "weird.xyz" in r.skipped_unknown
    assert f.exists()  # left in place
    manifests = list((inbox / ".processed").glob("*.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text())
    assert data["status"] == "unknown"


def test_handler_failure_records_manifest(
    inbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    f = inbox / "fail.md"
    f.write_text("hello", encoding="utf-8")
    past = time.time() - 60
    import os
    os.utime(f, (past, past))

    handlers = {"text": _stub_handler_fail}
    r = router.scan_and_route(inbox_dir=inbox, handlers=handlers)
    assert r.status == "ok"
    assert len(r.failed) == 1
    assert "intentional failure" in r.failed[0]["reason"]
    assert f.exists()  # not archived
    manifests = list((inbox / ".processed").glob("*.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text())
    assert data["status"] == "failed"


def test_dedup_skips_second_identical_file(inbox: Path) -> None:
    f1 = inbox / "first.md"
    f1.write_text("identical content", encoding="utf-8")
    past = time.time() - 60
    import os
    os.utime(f1, (past, past))

    handlers = {"text": _stub_handler_ok}
    r1 = router.scan_and_route(inbox_dir=inbox, handlers=handlers)
    assert len(r1.processed) == 1

    # Drop a second file with identical content.
    f2 = inbox / "second.md"
    f2.write_text("identical content", encoding="utf-8")
    os.utime(f2, (past, past))

    r2 = router.scan_and_route(inbox_dir=inbox, handlers=handlers)
    # Second file deduped against the first's hash.
    assert "second.md" in r2.skipped_dedup
    # And moved to archive on the dedup branch.
    archived = list((inbox / ".archive").rglob("second.md"))
    assert len(archived) == 1


def test_recently_modified_file_deferred(inbox: Path) -> None:
    """A file modified within the quiet-window is deferred to next tick."""
    f = inbox / "fresh.md"
    f.write_text("fresh", encoding="utf-8")
    # Don't touch mtime — just-created files are inherently recent.

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert "fresh.md" in r.deferred
    assert f.exists()


def test_dotfiles_skipped(inbox: Path) -> None:
    """Hidden files are ignored."""
    (inbox / ".secret").write_text("hidden", encoding="utf-8")
    past = time.time() - 60
    import os
    os.utime(inbox / ".secret", (past, past))

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert r.processed == []


def test_handlers_for_image_audio_pdf_currently_unsupported(inbox: Path) -> None:
    """Until real handlers are wired, recognised-but-unhandled kinds
    fail with a clear reason. Uses valid JPG magic bytes so the file
    gets to the handler dispatch (not rejected by the classifier)."""
    f = inbox / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    past = time.time() - 60
    import os
    os.utime(f, (past, past))

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert any("photo.jpg" in d["name"] for d in r.failed)


def test_apple_health_handler_routes_to_health(
    tmp_path: Path, inbox: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The apple_health_export handler uses the §5.1 importer."""
    monkeypatch.setenv("HEALTH_INGESTION_ENABLED", "true")
    monkeypatch.setenv("HEALTH_BASE_DIR", str(tmp_path / "health"))
    # Build a tiny apple_health_export.zip.
    import zipfile
    src = inbox / "apple_health_export.zip"
    xml_dir = tmp_path / "xmlsrc" / "apple_health_export"
    xml_dir.mkdir(parents=True)
    (xml_dir / "export.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<HealthData>'
        '<Record type="HKQuantityTypeIdentifierStepCount" '
        '        sourceName="iPhone" sourceVersion="u1" '
        '        startDate="2026-05-10 08:00:00 +0300" '
        '        endDate="2026-05-10 08:15:00 +0300" '
        '        value="500"/>'
        '</HealthData>',
        encoding="utf-8",
    )
    with zipfile.ZipFile(src, "w") as zf:
        zf.write(xml_dir / "export.xml", "apple_health_export/export.xml")

    past = time.time() - 60
    import os
    os.utime(src, (past, past))

    r = router.scan_and_route(inbox_dir=inbox)
    assert r.status == "ok"
    assert len(r.processed) == 1
    assert r.processed[0]["kind"] == "apple_health_export"
    assert "imported" in r.processed[0]["outcome"]
