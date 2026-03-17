from crewai.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi
import re


@tool("get_youtube_transcript")
def get_youtube_transcript(url_or_id: str) -> str:
    """
    Extract the transcript of a YouTube video.
    Accepts full YouTube URL or bare video ID.
    Returns plain text transcript, max 12000 chars.
    """
    # Bound input length before running any regex to prevent ReDoS / resource abuse
    url_or_id = url_or_id[:200]

    # Extract video ID from URL if needed
    match = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url_or_id)
    video_id = match.group(1) if match else url_or_id.strip()

    # Validate video ID format (exactly 11 alphanumeric/dash/underscore chars)
    if not re.fullmatch(r"[\w-]{11}", video_id):
        return "Invalid YouTube video ID."

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(entry["text"] for entry in transcript_list)
        return text[:12000]
    except Exception:
        return "Could not retrieve transcript for this video."
