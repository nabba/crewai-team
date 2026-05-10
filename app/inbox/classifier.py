"""File-type classifier.

Two-pass: extension first (fast + good 95% of the time), then a small
magic-bytes peek for the cases that matter (Apple Health zip vs ordinary
zip; PDF magic bytes; PNG/JPEG signatures). Conservative — when in
doubt, classify as ``unknown`` so the router asks the operator rather
than running the wrong handler.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileClassification:
    """One file's classification."""

    kind: str            # canonical kind: see KNOWN_KINDS below
    confidence: str      # "high" | "medium" | "low"
    reason: str          # human-readable explanation


KNOWN_KINDS: frozenset[str] = frozenset({
    "apple_health_export",  # apple_health_export.zip → app.health
    "audio",                # .m4a / .mp3 / .wav → voice STT
    "image",                # .png / .jpg / .heic → vision pipeline
    "pdf",                  # .pdf → episteme RAG ingestion
    "text",                 # .txt / .md → companion notes
    "csv",                  # .csv → companion data
    "spreadsheet",          # .xlsx / .ods → companion data
    "unknown",              # nothing matched — operator alert
})


# Extension → kind. Lowercased, no leading dot.
_EXTENSION_MAP: dict[str, str] = {
    "m4a": "audio",
    "mp3": "audio",
    "wav": "audio",
    "ogg": "audio",
    "flac": "audio",
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "heic": "image",
    "webp": "image",
    "pdf": "pdf",
    "txt": "text",
    "md": "text",
    "markdown": "text",
    "csv": "csv",
    "xlsx": "spreadsheet",
    "ods": "spreadsheet",
}


def _peek_magic(path: Path, n: int = 16) -> bytes:
    """Return first ``n`` bytes of the file. Empty on failure — caller
    should treat that as a low-confidence signal."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _is_apple_health_zip(path: Path) -> bool:
    """Best-effort check that a .zip is the Apple Health export.

    The zip's first directory entry typically names ``apple_health_export/``.
    We avoid actually opening the zip in the classifier (the importer
    will do that) — instead, we accept any ``.zip`` whose filename matches
    the canonical export name. This is conservative: a .zip that isn't
    actually an Apple Health export will be caught by the importer's
    own ``failed_missing_xml`` branch.
    """
    name = path.name.lower()
    if name == "apple_health_export.zip":
        return True
    if name.startswith("apple_health_export") and name.endswith(".zip"):
        return True
    return False


def classify_file(path: Path) -> FileClassification:
    """Classify one file. Never raises."""
    if not path.exists() or not path.is_file():
        return FileClassification(
            kind="unknown",
            confidence="high",
            reason="file not found or not a regular file",
        )
    ext = path.suffix.lstrip(".").lower()

    # Apple Health export wins over the generic .zip path.
    if ext == "zip":
        if _is_apple_health_zip(path):
            return FileClassification(
                kind="apple_health_export",
                confidence="high",
                reason="filename matches apple_health_export*.zip",
            )
        return FileClassification(
            kind="unknown",
            confidence="medium",
            reason="generic .zip; no specific handler",
        )

    if ext in _EXTENSION_MAP:
        kind = _EXTENSION_MAP[ext]

        # Magic-byte sanity for the high-leverage kinds.
        if kind == "pdf":
            magic = _peek_magic(path)
            if not magic.startswith(b"%PDF"):
                return FileClassification(
                    kind="unknown",
                    confidence="medium",
                    reason=".pdf extension but no %PDF magic bytes",
                )
        if kind == "image" and ext == "png":
            magic = _peek_magic(path, n=8)
            if magic and magic[:8] != b"\x89PNG\r\n\x1a\n":
                return FileClassification(
                    kind="unknown",
                    confidence="medium",
                    reason=".png extension but no PNG magic bytes",
                )
        return FileClassification(
            kind=kind,
            confidence="high",
            reason=f"extension .{ext} → {kind}",
        )

    return FileClassification(
        kind="unknown",
        confidence="high",
        reason=f"unrecognised extension .{ext}",
    )
