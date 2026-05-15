"""YouTube-link inbox handler.

PROGRAM §46.7 (Q9.4). Drop a ``.url`` / ``.webloc`` shortcut OR a
single-line ``.txt`` containing a YouTube URL, and the inbox routes
the URL into the existing ``watch <url>`` skill — which already
distills the transcript into a skill entry + team memory + Signal
digest.

We do NOT re-implement the transcript pipeline here. The classifier
peeks the file content for the URL; this handler extracts the URL
and dispatches through the existing surface.
"""
from __future__ import annotations

import configparser
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


_YOUTUBE_URL_RE = re.compile(
    r"https?://(?:www\.|m\.)?(?:youtube\.com/[^\s<>\"]+|youtu\.be/[^\s<>\"]+)",
    re.IGNORECASE,
)


def run(path: Path) -> str:
    url = _extract_url(path)
    if not url:
        raise RuntimeError("youtube link handler: no URL found in file")

    # Delegate to the existing learn_from_youtube path that the
    # ``watch <url>`` Signal command uses. The crew handles transcript
    # extraction + skill registration + memory write on its own.
    try:
        from app.crews.self_improvement_crew import SelfImprovementCrew
        crew = SelfImprovementCrew()
        outcome = crew.learn_from_youtube(url)
    except Exception as exc:
        # Soft fallback: drop a note with the URL so the operator
        # knows the file was recognised even when the crew layer is
        # unavailable (test env, gateway-deps).
        logger.debug(
            "youtube_link: learn_from_youtube failed: %s",
            exc, exc_info=True,
        )
        return _fallback_note(path, url)

    summary = (outcome or "").strip().splitlines()[0] if outcome else ""
    return f"youtube → distilled: {summary[:160]}"


def _extract_url(path: Path) -> str:
    """Pull the first YouTube URL out of the file body.

    Recognises:
      - Plain text containing a URL
      - Windows .url shortcut: INI-format ``[InternetShortcut]`` /
        ``URL=…``
      - macOS .webloc: plist-XML ``<key>URL</key><string>…</string>``
    """
    suf = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if suf == ".url":
        try:
            cp = configparser.ConfigParser()
            cp.read_string(text)
            for section in cp.sections():
                if "URL" in cp[section]:
                    url = cp[section]["URL"]
                    if _YOUTUBE_URL_RE.search(url):
                        return url.strip()
        except Exception:
            pass

    if suf == ".webloc":
        # Crude XML peek
        m = re.search(
            r"<key>URL</key>\s*<string>([^<]+)</string>", text, re.I,
        )
        if m and _YOUTUBE_URL_RE.search(m.group(1)):
            return m.group(1).strip()

    m = _YOUTUBE_URL_RE.search(text)
    return m.group(0).strip() if m else ""


def _fallback_note(path: Path, url: str) -> str:
    """Write a note recording the URL when Commander dispatch isn't
    available (test env, degraded boot, etc.)."""
    import os
    from app.paths import WORKSPACE_ROOT
    notes_dir = Path(os.getenv("INBOX_NOTES_DIR", str(WORKSPACE_ROOT / "notes")))
    notes_dir.mkdir(parents=True, exist_ok=True)
    dest = notes_dir / f"{path.stem}.youtube.md"
    if dest.exists():
        stem = dest.stem
        i = 1
        while True:
            cand = notes_dir / f"{stem}.{i}.md"
            if not cand.exists():
                dest = cand
                break
            i += 1
    ts = datetime.now(timezone.utc).isoformat()
    dest.write_text(
        f"# YouTube link (queued for watch)\n\n"
        f"_Captured at {ts}._\n\n{url}\n",
        encoding="utf-8",
    )
    return f"youtube → {dest.name} (Commander queue unavailable)"
