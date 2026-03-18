from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
import logging
import re

logger = logging.getLogger(__name__)


def _extract_video_id(url_or_id: str) -> str | None:
    """Extract an 11-char YouTube video ID from various URL formats."""
    url_or_id = url_or_id[:300].strip()
    # Patterns: youtube.com/watch?v=ID, youtu.be/ID, youtube.com/embed/ID, youtube.com/v/ID
    # Must handle trailing ?si=..., &feature=..., etc.
    match = re.search(r"(?:v=|youtu\.be/|embed/|/v/)([\w-]{11})", url_or_id)
    if match:
        return match.group(1)
    # Maybe it's already a bare ID
    clean = url_or_id.split("?")[0].split("&")[0].strip()
    if re.fullmatch(r"[\w-]{11}", clean):
        return clean
    return None


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

    # Try multiple strategies to get a transcript
    strategies = [
        # 1. Default (manual captions in any language)
        lambda: YouTubeTranscriptApi.get_transcript(video_id),
        # 2. English manual or auto-generated
        lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=["en"]),
        # 3. Auto-generated English specifically
        lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=["en-US", "en-GB"]),
    ]

    # 4. Try listing all available transcripts and pick the best one
    def _try_any_available():
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manual, then generated
        for t in transcript_list:
            if not t.is_generated:
                return t.fetch()
        for t in transcript_list:
            return t.fetch()
        raise Exception("No transcripts available")

    strategies.append(_try_any_available)

    last_error = None
    for strategy in strategies:
        try:
            entries = strategy()
            text = " ".join(
                entry.get("text", "") if isinstance(entry, dict) else str(entry)
                for entry in entries
            )
            if text.strip():
                logger.info(f"YouTube transcript extracted for {video_id} ({len(text)} chars)")
                return text[:12000]
        except Exception as exc:
            last_error = exc
            continue

    logger.warning(f"All transcript strategies failed for {video_id}: {last_error}")
    return (
        f"Could not retrieve transcript for video {video_id}. "
        f"The video may not have any captions (manual or auto-generated). "
        f"Error: {str(last_error)[:150]}"
    )
