"""Tests for app.inbox.classifier."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.inbox.classifier import classify_file


def test_apple_health_export_zip(tmp_path: Path) -> None:
    p = tmp_path / "apple_health_export.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    c = classify_file(p)
    assert c.kind == "apple_health_export"
    assert c.confidence == "high"


def test_apple_health_export_zip_with_suffix(tmp_path: Path) -> None:
    p = tmp_path / "apple_health_export-2026-05-10.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    c = classify_file(p)
    assert c.kind == "apple_health_export"


def test_generic_zip_unknown(tmp_path: Path) -> None:
    p = tmp_path / "stuff.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    c = classify_file(p)
    assert c.kind == "unknown"


def test_audio_extensions(tmp_path: Path) -> None:
    for ext in ("m4a", "mp3", "wav", "ogg", "flac"):
        p = tmp_path / f"voice.{ext}"
        p.write_bytes(b"\x00" * 32)
        assert classify_file(p).kind == "audio"


def test_image_extensions(tmp_path: Path) -> None:
    p = tmp_path / "photo.jpg"
    p.write_bytes(b"\x00" * 32)
    assert classify_file(p).kind == "image"


def test_png_magic_bytes_required(tmp_path: Path) -> None:
    p = tmp_path / "fake.png"
    p.write_bytes(b"not actually png")
    c = classify_file(p)
    assert c.kind == "unknown"
    assert "PNG magic" in c.reason


def test_png_with_magic_bytes(tmp_path: Path) -> None:
    p = tmp_path / "real.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    c = classify_file(p)
    assert c.kind == "image"


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
