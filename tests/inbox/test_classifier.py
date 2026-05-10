"""Tests for app.inbox.classifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.inbox.classifier import classify_file


def _make_apple_zip(tmp_path: Path, name: str = "apple_health_export.zip") -> Path:
    """Build a tiny zip containing apple_health_export/export.xml."""
    import zipfile
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("apple_health_export/export.xml", "<HealthData/>")
    return p


def test_apple_health_export_zip(tmp_path: Path) -> None:
    p = _make_apple_zip(tmp_path)
    c = classify_file(p)
    assert c.kind == "apple_health_export"
    assert c.confidence == "high"
    assert "export.xml" in c.reason


def test_apple_health_export_zip_renamed(tmp_path: Path) -> None:
    """Renaming the export still works because we peek the zip index."""
    p = _make_apple_zip(tmp_path, name="my-health-2026.zip")
    c = classify_file(p)
    assert c.kind == "apple_health_export"


def test_generic_zip_unknown(tmp_path: Path) -> None:
    """A zip without the apple_health_export member → unknown."""
    import zipfile
    p = tmp_path / "stuff.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "hello")
    c = classify_file(p)
    assert c.kind == "unknown"


def test_apple_health_zip_filename_fallback(tmp_path: Path) -> None:
    """If the zip is unreadable but the filename matches, fall back."""
    p = tmp_path / "apple_health_export.zip"
    p.write_bytes(b"not actually a zip")
    c = classify_file(p)
    # The filename heuristic catches this so the importer can give a
    # specific failed_zip / failed_missing_xml reason instead of
    # "unrecognised".
    assert c.kind == "apple_health_export"


def test_audio_extensions_with_valid_magic(tmp_path: Path) -> None:
    """Each audio extension classifies as 'audio' when magic bytes match."""
    cases = [
        ("voice.mp3", b"ID3\x03" + b"\x00" * 32),
        ("voice.wav", b"RIFF" + b"\x00" * 28),
        ("voice.ogg", b"OggS" + b"\x00" * 32),
        ("voice.flac", b"fLaC" + b"\x00" * 32),
        # M4A magic is at offset 4 — first 4 bytes are the box size.
        ("voice.m4a", b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 32),
    ]
    for name, payload in cases:
        p = tmp_path / name
        p.write_bytes(payload)
        c = classify_file(p)
        assert c.kind == "audio", f"{name} → {c.kind} ({c.reason})"


def test_audio_with_wrong_magic_rejected(tmp_path: Path) -> None:
    """A .mp3 without ID3/MPEG sync bytes is flagged as unknown."""
    p = tmp_path / "fake.mp3"
    p.write_bytes(b"not actually mp3 data")
    c = classify_file(p)
    assert c.kind == "unknown"
    assert "magic bytes" in c.reason


def test_image_extensions_with_valid_magic(tmp_path: Path) -> None:
    """JPG / PNG / WEBP all need their signature."""
    cases = [
        ("photo.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 32),
        ("photo.jpeg", b"\xff\xd8\xff\xe1" + b"\x00" * 32),
        ("photo.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
        ("photo.webp", b"RIFF" + b"\x00" * 32),
    ]
    for name, payload in cases:
        p = tmp_path / name
        p.write_bytes(payload)
        c = classify_file(p)
        assert c.kind == "image", f"{name} → {c.kind} ({c.reason})"


def test_jpg_with_wrong_magic_rejected(tmp_path: Path) -> None:
    p = tmp_path / "fake.jpg"
    p.write_bytes(b"GIF89a" + b"\x00" * 32)
    c = classify_file(p)
    assert c.kind == "unknown"


def test_png_magic_bytes_required(tmp_path: Path) -> None:
    p = tmp_path / "fake.png"
    p.write_bytes(b"not actually png")
    c = classify_file(p)
    assert c.kind == "unknown"
    assert "magic bytes" in c.reason


def test_known_kinds_enforcement() -> None:
    """The module-load assertion blocks typos in _EXTENSION_MAP."""
    from app.inbox.classifier import _EXTENSION_MAP, KNOWN_KINDS
    assert set(_EXTENSION_MAP.values()) <= KNOWN_KINDS


def test_pdf_magic_bytes_required(tmp_path: Path) -> None:
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"not a pdf")
    c = classify_file(p)
    assert c.kind == "unknown"


def test_pdf_with_magic_bytes(tmp_path: Path) -> None:
    p = tmp_path / "real.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"\x00" * 100)
    c = classify_file(p)
    assert c.kind == "pdf"


def test_text_extensions(tmp_path: Path) -> None:
    for ext in ("txt", "md", "markdown"):
        p = tmp_path / f"note.{ext}"
        p.write_text("hello", encoding="utf-8")
        assert classify_file(p).kind == "text"


def test_csv_extension(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    assert classify_file(p).kind == "csv"


def test_unknown_extension(tmp_path: Path) -> None:
    p = tmp_path / "weird.xyz"
    p.write_bytes(b"\x00" * 32)
    c = classify_file(p)
    assert c.kind == "unknown"
    assert "xyz" in c.reason


def test_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "ghost.png"
    c = classify_file(p)
    assert c.kind == "unknown"
    assert "not found" in c.reason
