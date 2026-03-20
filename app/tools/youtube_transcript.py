from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
import logging
import re

logger = logging.getLogger(__name__)

# youtube-transcript-api v1.x uses instance methods: api.fetch(), api.list()
_api = YouTubeTranscriptApi()


def _extract_video_id(url_or_id: str) -> str | None:
    """Extract an 11-char YouTube video ID from various URL formats.

    Returns only IDs matching [a-zA-Z0-9_-]{11} to prevent injection.
    """
    url_or_id = url_or_id[:300].strip()
    # Patterns: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID, youtube.com/v/ID
    # Must handle trailing ?si=..., &feature=..., etc.
    match = re.search(r"(?:v=|youtu\.be/|embed/|/v/)([a-zA-Z0-9_-]{11})", url_or_id)
    if match:
        return match.group(1)
    # Maybe it's already a bare ID
    clean = url_or_id.split("?")[0].split("&")[0].strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", clean):
        return clean
    return None


def _entries_to_text(entries) -> str:
    """Convert transcript entries (list of snippet objects or dicts) to plain text."""
    parts = []
    for entry in entries:
        if isinstance(entry, dict):
            parts.append(entry.get("text", ""))
        elif hasattr(entry, "text"):
            parts.append(entry.text)
        else:
            parts.append(str(entry))
    return " ".join(parts)


@tool("get_youtube_transcript")
def get_youtube_transcript(url_or_id: str) -> str:
    """
    Extract the transcript of a YouTube video.
    Accepts full YouTube URL (including youtu.be links with ?si= params) or bare video ID.
    Tries manual captions first, then auto-generated captions in multiple languages.
    Returns plain text transcript, max 12000 chars.
    """
    video_id = _extract_video_id(url_or_id)
    if not video_id:
        return f"Invalid YouTube video ID or URL: {url_or_id[:80]}"

    # Strategy 1: Direct fetch (uses default language selection)
    try:
        entries = _api.fetch(video_id)
        text = _entries_to_text(entries)
        if text.strip():
            logger.info(f"YouTube transcript extracted for {video_id} ({len(text)} chars)")
            return text[:12000]
    except Exception as exc:
        logger.debug(f"Direct fetch failed for {video_id}: {exc}")

    # Strategy 2: List available transcripts and pick the best one
    try:
        transcript_list = _api.list(video_id)

        # Try to find any transcript from the listing
        # The list() result has .manual and .generated attributes in some versions,
        # or is iterable. Try fetching each available one.
        best = None
        for t in transcript_list:
            if hasattr(t, "is_generated") and not t.is_generated:
                best = t  # prefer manual
                break
        if best is None:
            for t in transcript_list:
                best = t
                break

        if best is not None:
            if hasattr(best, "fetch"):
                entries = best.fetch()
            else:
                # Some versions: the transcript object IS the entries
                entries = best
            text = _entries_to_text(entries)
            if text.strip():
                logger.info(f"YouTube transcript (listed) for {video_id} ({len(text)} chars)")
                return text[:12000]
    except Exception as exc:
        logger.debug(f"List-based fetch failed for {video_id}: {exc}")

    # Strategy 3: Fetch with explicit language codes
    for langs in [["en"], ["en-US", "en-GB"]]:
        try:
            entries = _api.fetch(video_id, languages=langs)
            text = _entries_to_text(entries)
            if text.strip():
                logger.info(f"YouTube transcript ({langs}) for {video_id} ({len(text)} chars)")
                return text[:12000]
        except Exception:
            continue

    # Strategy 4: Fallback — use yt-dlp to extract auto-subtitles
    # yt-dlp can extract auto-generated subtitles that youtube-transcript-api misses
    try:
        import subprocess
        result = subprocess.run(
            [
                "yt-dlp", "--skip-download",
                "--write-auto-sub", "--sub-lang", "en",
                "--sub-format", "vtt",
                "-o", "/tmp/yt-%(id)s",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=60, shell=False,
        )
        import glob
        vtt_files = glob.glob(f"/tmp/yt-{video_id}*.vtt")
        if vtt_files:
            with open(vtt_files[0]) as fh:
                raw_vtt = fh.read(1_000_000)  # cap at 1MB
            # Strip VTT headers and timestamps, keep text
            lines = []
            for line in raw_vtt.splitlines():
                line = line.strip()
                if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
                    continue
                if re.match(r"^\d{2}:\d{2}", line) or line.startswith("NOTE"):
                    continue
                if re.match(r"^<\d{2}:\d{2}", line):
                    continue
                # Strip VTT tags like <c> </c> <00:00:01.234>
                clean = re.sub(r"<[^>]+>", "", line)
                if clean.strip():
                    lines.append(clean.strip())
            # Deduplicate consecutive identical lines (VTT repeats them)
            deduped = []
            for ln in lines:
                if not deduped or ln != deduped[-1]:
                    deduped.append(ln)
            text = " ".join(deduped)
            # Cleanup temp files
            for f in vtt_files:
                try:
                    import os
                    os.unlink(f)
                except OSError:
                    pass
            if text.strip():
                logger.info(f"YouTube transcript (yt-dlp) for {video_id} ({len(text)} chars)")
                return text[:12000]
    except FileNotFoundError:
        logger.debug("yt-dlp not installed, skipping fallback")
    except Exception as exc:
        logger.debug(f"yt-dlp fallback failed for {video_id}: {exc}")

    # Strategy 5: Fetch video page metadata as minimal context
    try:
        import requests
        resp = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers={"Accept-Language": "en"},
            timeout=15,
        )
        # Extract title and description from meta tags
        title_match = re.search(r'<meta name="title" content="([^"]*)"', resp.text)
        desc_match = re.search(r'<meta name="description" content="([^"]*)"', resp.text)
        title = title_match.group(1) if title_match else ""
        desc = desc_match.group(1) if desc_match else ""
        if title or desc:
            meta_text = f"Video title: {title}\nDescription: {desc}"
            logger.info(f"YouTube metadata fallback for {video_id}")
            return (
                f"[Note: Full transcript unavailable. Using video metadata instead.]\n\n"
                f"{meta_text[:4000]}"
            )
    except Exception as exc:
        logger.debug(f"Metadata fallback failed for {video_id}: {exc}")

    logger.warning(f"All transcript strategies failed for {video_id}")
    return (
        f"Could not retrieve transcript for video {video_id}. "
        f"The video may not have any captions (manual or auto-generated). "
        f"Try a different video or provide a direct transcript."
    )
