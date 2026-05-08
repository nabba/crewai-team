"""
test_voice — smoke tests for the voice dispatchers.

These tests do not require whisper.cpp, Piper, Groq, or Google credentials.
They monkeypatch the backend modules so the dispatch logic, voice-mode
gating, fallback chain, and inbound-state cache can be exercised hermetically.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

import app.runtime_settings as rs
from app import voice
from app.voice import inbound_state, stt, tts


@pytest.fixture(autouse=True)
def _reset_runtime_settings(tmp_path, monkeypatch):
    """Redirect the runtime-settings JSON to a tmpdir so each test starts clean."""
    monkeypatch.setattr(rs, "_STATE_PATH", tmp_path / "runtime_settings.json")
    monkeypatch.setattr(rs, "_cache", None, raising=False)
    yield
    monkeypatch.setattr(rs, "_cache", None, raising=False)


@pytest.fixture(autouse=True)
def _reset_inbound_state():
    """Voice inbound cache is module-global — wipe between tests."""
    with inbound_state._lock:
        inbound_state._state.clear()
    yield
    with inbound_state._lock:
        inbound_state._state.clear()


# ── Voice mode gating ──────────────────────────────────────────────────────

def test_transcribe_off_returns_empty():
    rs.set_voice_mode("off")
    assert voice.transcribe(b"\x00\x01") == ""


def test_synthesize_off_returns_none():
    rs.set_voice_mode("off")
    assert voice.synthesize("hello") is None


def test_transcribe_empty_bytes_returns_empty():
    rs.set_voice_mode("local")
    assert voice.transcribe(b"") == ""


def test_synthesize_empty_text_returns_none():
    rs.set_voice_mode("local")
    assert voice.synthesize("") is None


# ── Fallback chain ────────────────────────────────────────────────────────

def test_local_falls_back_to_cloud_on_empty(monkeypatch):
    """When voice_mode=local and the local backend fails, dispatcher tries cloud."""
    rs.set_voice_mode("local")
    calls: list[str] = []

    def local_stub(_audio, *, audio_format, language=None):
        calls.append("local")
        return ""

    def cloud_stub(_audio, *, audio_format, language=None):
        calls.append("cloud")
        return "cloud transcript"

    monkeypatch.setattr("app.voice.local.transcribe", local_stub)
    monkeypatch.setattr("app.voice.cloud.transcribe", cloud_stub)

    assert voice.transcribe(b"\x00", audio_format="m4a") == "cloud transcript"
    assert calls == ["local", "cloud"]


def test_cloud_falls_back_to_local_on_empty(monkeypatch):
    """When voice_mode=cloud and the cloud backend fails, dispatcher tries local."""
    rs.set_voice_mode("cloud")
    calls: list[str] = []

    def cloud_stub(_audio, *, audio_format, language=None):
        calls.append("cloud")
        return ""

    def local_stub(_audio, *, audio_format, language=None):
        calls.append("local")
        return "local transcript"

    monkeypatch.setattr("app.voice.cloud.transcribe", cloud_stub)
    monkeypatch.setattr("app.voice.local.transcribe", local_stub)

    assert voice.transcribe(b"\x00", audio_format="m4a") == "local transcript"
    assert calls == ["cloud", "local"]


def test_synthesize_records_format_per_backend(monkeypatch):
    """TTS_OUTPUT_FORMAT['latest'] reflects which backend produced the bytes."""
    rs.set_voice_mode("cloud")

    def cloud_stub(_text, *, language="en"):
        return b"OPUSDATA"

    def local_stub(_text, *, language="en"):
        return b"WAVDATA"

    monkeypatch.setattr("app.voice.cloud.synthesize", cloud_stub)
    monkeypatch.setattr("app.voice.local.synthesize", local_stub)

    assert voice.synthesize("hi") == b"OPUSDATA"
    assert tts.TTS_OUTPUT_FORMAT["latest"] == "opus"

    rs.set_voice_mode("local")
    assert voice.synthesize("hi") == b"WAVDATA"
    assert tts.TTS_OUTPUT_FORMAT["latest"] == "wav"


# ── Cap on long input ──────────────────────────────────────────────────────

def test_synthesize_caps_long_text(monkeypatch):
    rs.set_voice_mode("cloud")
    captured: dict[str, Any] = {}

    def cloud_stub(text, *, language="en"):
        captured["len"] = len(text)
        return b"audio"

    monkeypatch.setattr("app.voice.cloud.synthesize", cloud_stub)
    voice.synthesize("a" * 10_000)
    assert captured["len"] == 3000


# ── Inbound-state cache ────────────────────────────────────────────────────

def test_voice_inbound_state_marks_and_expires(monkeypatch):
    sender = "+3725550000"
    assert not inbound_state.is_voice_active(sender)
    inbound_state.mark_voice_inbound(sender)
    assert inbound_state.is_voice_active(sender)

    # Force expiry by rewinding the recorded timestamp past the TTL.
    with inbound_state._lock:
        inbound_state._state[sender] = time.time() - (
            inbound_state._VOICE_TTL_SECONDS + 1
        )
    assert not inbound_state.is_voice_active(sender)


def test_voice_inbound_state_clear():
    sender = "+3725550001"
    inbound_state.mark_voice_inbound(sender)
    assert inbound_state.is_voice_active(sender)
    inbound_state.clear(sender)
    assert not inbound_state.is_voice_active(sender)


def test_voice_inbound_state_cap():
    """The OrderedDict eviction caps the map size."""
    cap = inbound_state._MAX_SENDERS
    for i in range(cap + 5):
        inbound_state.mark_voice_inbound(f"+number_{i}")
    assert len(inbound_state._state) == cap
    # Earliest senders should have been evicted.
    assert "+number_0" not in inbound_state._state
    assert f"+number_{cap + 4}" in inbound_state._state


# ── AUDIO_MIME_PREFIXES exposes the audio/ prefix correctly ────────────────

def test_audio_mime_prefixes():
    assert "audio/mp4".startswith(stt.AUDIO_MIME_PREFIXES)
    assert "audio/aac".startswith(stt.AUDIO_MIME_PREFIXES)
    assert "audio/ogg".startswith(stt.AUDIO_MIME_PREFIXES)
    assert not "image/png".startswith(stt.AUDIO_MIME_PREFIXES)
    assert not "application/pdf".startswith(stt.AUDIO_MIME_PREFIXES)
