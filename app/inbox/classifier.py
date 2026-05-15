"""File-type classifier.

Two-pass: extension first (fast + good 95% of the time), then a small
magic-bytes peek for every binary format we recognise. Conservative —
when in doubt, classify as ``unknown`` so the router asks the operator
rather than running the wrong handler.

Apple Health zip detection uses a zip-index peek (looks for the
``apple_health_export/export.xml`` member) before falling back to the
filename heuristic — robust against renames.
"""
from __future__ import annotations

import logging
import zipfile
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
    "youtube_link",         # .url / single-line .txt → transcript+notes (Q9.4)
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
    # Q9.4 — Windows Internet Shortcut files; classified as
    # youtube_link only after content-peek confirms a youtube URL.
    "url": "text",
    # .webloc (macOS) — same treatment.
    "webloc": "text",
}


_YOUTUBE_URL_RE = __import__("re").compile(
    r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/",
    __import__("re").IGNORECASE,
)


def _content_is_youtube_link(path: Path) -> bool:
    """Sniff a short text-ish file for a YouTube URL.

    Triggers for: .url (Windows shortcut), .webloc (macOS shortcut),
    or .txt where the body is a single line containing a YouTube URL.
    Caps the read at 4 KB so we don't slurp a huge file by accident.
    """
    try:
        with open(path, "rb") as f:
            blob = f.read(4096)
    except OSError:
        return False
    try:
        text = blob.decode("utf-8", errors="replace")
    except Exception:
        return False
    return bool(_YOUTUBE_URL_RE.search(text))


def _peek_magic(path: Path, n: int = 16) -> bytes:
    """Return first ``n`` bytes of the file. Empty on failure — caller
    should treat that as a low-confidence signal."""
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _is_apple_health_zip(path: Path) -> bool:
    """Robust check: peek the zip's index for an
    ``apple_health_export/export.xml`` member. Falls back to the
    filename heuristic if the zip is unreadable (the importer's own
    ``failed_zip`` branch catches that anyway)."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            for n in names:
                if n.endswith("apple_health_export/export.xml") or n.endswith(
                    "/export.xml"
                ):
                    return True
            return False
    except (zipfile.BadZipFile, OSError):
        # Fall back to the filename heuristic so a partially-downloaded
        # zip still matches the user's intent.
        name = path.name.lower()
        return (
            name == "apple_health_export.zip"
            or (name.startswith("apple_health_export") and name.endswith(".zip"))
        )


# Per-extension magic-byte signatures. ``None`` means we don't probe
# (the extension is taken at face value — used for plain text). For
# kinds where the extension overlaps non-trivial binary formats
# (audio, image, pdf), we probe to reject obvious mismatches.
_MAGIC_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "png":  (b"\x89PNG\r\n\x1a\n",),
    "jpg":  (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
    "heic": (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1"),  # checked at offset 4
    "webp": (b"RIFF",),  # plus "WEBP" at offset 8 — best-effort
    "pdf":  (b"%PDF",),
    "mp3":  (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    "m4a":  (b"ftypM4A", b"ftypisom", b"ftypmp42"),  # at offset 4
    "wav":  (b"RIFF",),
    "ogg":  (b"OggS",),
    "flac": (b"fLaC",),
}

# Extensions where the magic check happens at a non-zero offset.
_MAGIC_OFFSETS: dict[str, int] = {
    "heic": 4,
    "m4a":  4,
}


def _magic_matches(path: Path, ext: str) -> bool:
    sigs = _MAGIC_SIGNATURES.get(ext)
    if sigs is None:
        return True
    offset = _MAGIC_OFFSETS.get(ext, 0)
    head = _peek_magic(path, n=max(16, offset + 16))
    if not head:
        return False
    target = head[offset:]
    return any(target.startswith(sig) for sig in sigs)


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
                reason="zip contains apple_health_export/export.xml",
            )
        return FileClassification(
            kind="unknown",
            confidence="medium",
            reason="generic .zip; no specific handler",
        )

    if ext in _EXTENSION_MAP:
        kind = _EXTENSION_MAP[ext]
        # Q9.4 — Windows/macOS shortcut files (.url / .webloc) AND
        # plain .txt files: peek the content. If it's a YouTube URL,
        # upgrade the kind to ``youtube_link`` so the dedicated
        # handler picks it up.
        if ext in ("url", "webloc", "txt"):
            if _content_is_youtube_link(path):
                return FileClassification(
                    kind="youtube_link",
                    confidence="high",
                    reason=f".{ext} content matches YouTube URL pattern",
                )
            # .url / .webloc without a YouTube URL is treated as text.
        if not _magic_matches(path, ext):
            return FileClassification(
                kind="unknown",
                confidence="medium",
                reason=f".{ext} extension but magic bytes don't match",
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


# Module-load assertion: every kind in _EXTENSION_MAP must be in
# KNOWN_KINDS. A typo would otherwise produce a kind no handler knows.
assert set(_EXTENSION_MAP.values()) <= KNOWN_KINDS, (
    f"unknown kinds in _EXTENSION_MAP: {set(_EXTENSION_MAP.values()) - KNOWN_KINDS}"
)
