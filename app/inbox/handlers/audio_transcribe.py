"""Audio → transcript inbox handler.

PROGRAM §46.7 (Q9.4). Drop an MP3 / M4A / WAV into ``workspace/inbox/``;
the file gets transcribed by whatever voice backend is configured in
``app.runtime_settings`` (``voice_mode``: ``"local"`` uses whisper.cpp
via the host bridge; ``"cloud"`` uses Groq Whisper) and lands as a
markdown note at ``workspace/notes/<stem>.transcript.md``.

Re-uses the existing :mod:`app.voice` STT dispatcher so we don't
build a parallel transcription pipeline.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB — guard against the
                                     # user dumping a 4h podcast file


def run(path: Path) -> str:
    """Transcribe ``path`` and write the result to workspace/notes/.

    Returns a one-line outcome the router records in the manifest.
    Raises on transcription failure so the file stays in the inbox
    for operator inspection.
    """
    size = path.stat().st_size if path.exists() else 0
    if size > _MAX_FILE_BYTES:
        raise RuntimeError(
            f"audio file {size / 1024 / 1024:.1f} MB exceeds "
            f"{_MAX_FILE_BYTES / 1024 / 1024:.0f} MB cap"
        )

    try:
        from app.voice import transcribe_audio
    except Exception as exc:
        raise RuntimeError(f"voice subsystem unavailable: {exc}") from exc

    try:
        with open(path, "rb") as f:
            audio_bytes = f.read()
        text = transcribe_audio(audio_bytes, mime_type=_mime_for(path))
    except Exception as exc:
        raise RuntimeError(f"transcription failed: {exc}") from exc

    if not text or not text.strip():
        raise RuntimeError("transcription produced empty output")

    notes_dir = _notes_dir()
    dest = notes_dir / f"{path.stem}.transcript.md"
    if dest.exists():
        stem = dest.stem
        i = 1
        while True:
            cand = notes_dir / f"{stem}.{i}.md"
            if not cand.exists():
                dest = cand
                break
            i += 1

    body = _format_note(path, text)
    dest.write_text(body, encoding="utf-8")
    return f"transcribed → {dest.name} ({len(text)} chars)"


def _notes_dir() -> Path:
    from app.paths import WORKSPACE_ROOT
    d = Path(os.getenv("INBOX_NOTES_DIR", str(WORKSPACE_ROOT / "notes")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mime_for(path: Path) -> str:
    suf = path.suffix.lower().lstrip(".")
    return {
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
    }.get(suf, "audio/mpeg")


def _format_note(src: Path, transcript: str) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    return (
        f"# Transcript: {src.name}\n\n"
        f"_Source file dropped at {ts}._\n\n"
        f"---\n\n"
        f"{transcript.strip()}\n"
    )
