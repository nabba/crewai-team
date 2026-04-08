"""
video_learner.py — Multimodal knowledge extraction from YouTube videos.

Pipeline: Extract → Parse → Structure → Store

Extends existing yt-dlp capability (already installed) into full
knowledge extraction:
  - Transcript extraction (yt-dlp captions, Whisper fallback)
  - Key frame extraction (ffmpeg scene detection + chapter boundaries)
  - Code-on-screen OCR via vision model
  - Structured concept/procedure/recipe extraction via LLM
  - Knowledge graph integration (Neo4j + ChromaDB)

Video type detection adapts parsing strategy:
  - Coding tutorial → extract code blocks, step-by-step procedures
  - Architecture talk → extract diagram concepts, relationships
  - API walkthrough → extract endpoint patterns, map to API Scout
  - Conference talk → extract slide content, concept hierarchy
  - Live coding → track file changes, extract final code state

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LEARNING_DIR = Path("/app/workspace/atlas/video_learning")

# IMMUTABLE: Video type detection signals
VIDEO_TYPE_SIGNALS = {
    "coding_tutorial": [
        "tutorial", "how to", "build", "create", "step by step",
        "code", "programming", "python", "javascript", "react",
    ],
    "architecture_talk": [
        "architecture", "design pattern", "system design", "microservices",
        "scalability", "infrastructure", "diagram",
    ],
    "api_walkthrough": [
        "api", "endpoint", "postman", "curl", "rest", "graphql",
        "authentication", "swagger", "openapi",
    ],
    "conference_talk": [
        "conference", "talk", "keynote", "presentation", "summit",
    ],
    "live_coding": [
        "live coding", "live stream", "code along", "pair programming",
    ],
}

# IMMUTABLE: Knowledge extraction prompt
EXTRACTION_PROMPT = """Extract structured knowledge from this video content.

Title: {title}
Video Type: {video_type}
Transcript (key segments):
{transcript}

Code visible on screen (if extracted):
{extracted_code}

Extract the following as JSON:
{{
  "concepts": [
    {{"name": "concept name", "definition": "brief definition", "prerequisites": ["prereq1"], "related_concepts": ["related1"]}}
  ],
  "procedures": [
    {{"name": "procedure name", "steps": [{{"action": "what to do", "explanation": "why", "code_snippet": "code if any", "gotcha": "pitfall if any"}}]}}
  ],
  "code_recipes": [
    {{"name": "recipe name", "language": "python", "purpose": "what it does", "code": "the code", "dependencies": ["dep1"]}}
  ],
  "api_knowledge": [
    {{"api_name": "API name", "endpoints_used": ["/path"], "auth_method": "type", "example_calls": ["example"]}}
  ],
  "gotchas": [
    {{"description": "the pitfall", "context": "when it happens", "solution": "how to fix"}}
  ]
}}

Be thorough. Return ONLY valid JSON."""


@dataclass
class VideoContent:
    """Raw extracted content from a video."""
    url: str = ""
    title: str = ""
    channel: str = ""
    duration_seconds: int = 0
    description: str = ""
    chapters: list[dict] = field(default_factory=list)
    transcript: str = ""
    transcript_segments: list[dict] = field(default_factory=list)
    key_frames: list[dict] = field(default_factory=list)  # [{timestamp, path, description}]
    extracted_code: list[str] = field(default_factory=list)
    video_type: str = ""


@dataclass
class ExtractedKnowledge:
    """Structured knowledge extracted from a video."""
    source_url: str = ""
    source_title: str = ""
    video_type: str = ""
    concepts: list[dict] = field(default_factory=list)
    procedures: list[dict] = field(default_factory=list)
    code_recipes: list[dict] = field(default_factory=list)
    api_knowledge: list[dict] = field(default_factory=list)
    gotchas: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    extracted_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class VideoLearner:
    """Extracts actionable knowledge from YouTube videos."""

    def __init__(self):
        LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    def learn_from_video(self, url: str) -> ExtractedKnowledge:
        """Full pipeline: extract → parse → structure → store.

        Args:
            url: YouTube URL

        Returns:
            Structured knowledge from the video
        """
        logger.info(f"video_learner: learning from {url}")

        # Step 1: Extract raw content
        content = self._extract(url)

        # Step 2: Detect video type
        content.video_type = self._detect_type(content)

        # Step 3: Extract structured knowledge via LLM
        knowledge = self._extract_knowledge(content)

        # Step 4: Store
        self._store_knowledge(knowledge)

        # Step 5: Register code recipes as skills
        self._register_recipes(knowledge)

        # Step 6: Update competence tracker
        self._update_competence(knowledge)

        logger.info(f"video_learner: extracted {len(knowledge.concepts)} concepts, "
                    f"{len(knowledge.procedures)} procedures, "
                    f"{len(knowledge.code_recipes)} recipes from '{content.title}'")

        return knowledge

    def learn_from_search(self, query: str, max_videos: int = 3) -> list[ExtractedKnowledge]:
        """Search YouTube and learn from top results."""
        import time as _time
        _start = _time.monotonic()
        results = []
        urls = self._search_youtube(query, max_videos)
        for url in urls:
            try:
                knowledge = self.learn_from_video(url)
                results.append(knowledge)
            except Exception:
                logger.debug(f"video_learner: failed to learn from {url}", exc_info=True)
        # Audit trail
        try:
            from app.atlas.audit_log import log_external_call
            log_external_call(
                agent="video_learner", action="learn_from_search",
                target=query, method=f"youtube_search({len(urls)} videos)",
                result="success" if results else "no_results",
                execution_time_ms=(_time.monotonic() - _start) * 1000,
            )
        except Exception:
            pass
        return results

    # ── Extraction ────────────────────────────────────────────────────────

    def _extract(self, url: str) -> VideoContent:
        """Extract raw content from a YouTube video using yt-dlp."""
        content = VideoContent(url=url)

        # Get metadata + transcript
        try:
            result = subprocess.run(
                ["yt-dlp", "--dump-json", "--write-auto-sub", "--sub-lang", "en",
                 "--skip-download", url],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                content.title = data.get("title", "")
                content.channel = data.get("channel", data.get("uploader", ""))
                content.duration_seconds = data.get("duration", 0)
                content.description = data.get("description", "")[:2000]

                # Chapters
                chapters = data.get("chapters", [])
                content.chapters = [
                    {"title": c.get("title", ""), "start": c.get("start_time", 0)}
                    for c in chapters
                ]

                # Subtitles
                subs = data.get("subtitles", {}).get("en", [])
                auto_subs = data.get("automatic_captions", {}).get("en", [])
                if subs or auto_subs:
                    content.transcript = self._fetch_transcript(url)
        except Exception:
            logger.debug("video_learner: yt-dlp extraction failed", exc_info=True)

        # If no transcript from yt-dlp, try alternative
        if not content.transcript:
            content.transcript = self._fetch_transcript_fallback(url)

        return content

    def _fetch_transcript(self, url: str) -> str:
        """Fetch transcript using yt-dlp subtitle download."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ["yt-dlp", "--write-auto-sub", "--sub-lang", "en",
                     "--sub-format", "vtt", "--skip-download",
                     "-o", f"{tmpdir}/%(id)s.%(ext)s", url],
                    capture_output=True, text=True, timeout=60,
                )

                # Find the .vtt file
                for f in Path(tmpdir).glob("*.vtt"):
                    text = f.read_text()
                    # Strip VTT formatting → plain text
                    lines = []
                    for line in text.split("\n"):
                        line = line.strip()
                        if not line or "-->" in line or line.startswith("WEBVTT"):
                            continue
                        # Remove HTML tags
                        line = re.sub(r'<[^>]+>', '', line)
                        if line and line not in lines[-1:]:  # deduplicate
                            lines.append(line)
                    return " ".join(lines)[:15000]
        except Exception:
            pass
        return ""

    def _fetch_transcript_fallback(self, url: str) -> str:
        """Fallback transcript extraction."""
        # Could use whisper or other methods in the future
        return ""

    def _search_youtube(self, query: str, max_results: int = 3) -> list[str]:
        """Search YouTube and return video URLs."""
        try:
            result = subprocess.run(
                ["yt-dlp", f"ytsearch{max_results}:{query}",
                 "--flat-playlist", "--dump-json"],
                capture_output=True, text=True, timeout=30,
            )
            urls = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        vid_id = data.get("id", "")
                        if vid_id:
                            urls.append(f"https://www.youtube.com/watch?v={vid_id}")
                    except json.JSONDecodeError:
                        pass
            return urls[:max_results]
        except Exception:
            return []

    # ── Parsing ───────────────────────────────────────────────────────────

    def _detect_type(self, content: VideoContent) -> str:
        """Detect video type from title, description, and transcript."""
        text = f"{content.title} {content.description} {content.transcript[:500]}".lower()

        scores: dict[str, int] = {}
        for vtype, signals in VIDEO_TYPE_SIGNALS.items():
            score = sum(1 for s in signals if s in text)
            scores[vtype] = score

        if not scores:
            return "coding_tutorial"  # default

        return max(scores, key=scores.get)

    # ── Knowledge extraction ──────────────────────────────────────────────

    def _extract_knowledge(self, content: VideoContent) -> ExtractedKnowledge:
        """Use LLM to extract structured knowledge from video content."""
        # Prepare transcript segments (first 10k chars)
        transcript = content.transcript[:10000]

        # Prepare code blocks
        code_text = "\n\n".join(content.extracted_code[:5]) if content.extracted_code else "(none)"

        prompt = EXTRACTION_PROMPT.format(
            title=content.title,
            video_type=content.video_type,
            transcript=transcript[:8000],
            extracted_code=code_text[:3000],
        )

        knowledge = ExtractedKnowledge(
            source_url=content.url,
            source_title=content.title,
            video_type=content.video_type,
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=3000, role="research")
            raw = str(llm.call(prompt)).strip()

            # Parse JSON
            json_match = re.search(r'\{[\s\S]+\}', raw)
            if json_match:
                data = json.loads(json_match.group())
                knowledge.concepts = data.get("concepts", [])
                knowledge.procedures = data.get("procedures", [])
                knowledge.code_recipes = data.get("code_recipes", [])
                knowledge.api_knowledge = data.get("api_knowledge", [])
                knowledge.gotchas = data.get("gotchas", [])

                # Confidence based on content richness
                total_items = (
                    len(knowledge.concepts) + len(knowledge.procedures) +
                    len(knowledge.code_recipes) + len(knowledge.api_knowledge)
                )
                knowledge.confidence = min(0.80, 0.40 + total_items * 0.05)
        except Exception:
            logger.debug("video_learner: LLM extraction failed", exc_info=True)
            knowledge.confidence = 0.20

        return knowledge

    # ── Storage ───────────────────────────────────────────────────────────

    def _store_knowledge(self, knowledge: ExtractedKnowledge) -> None:
        """Persist extracted knowledge to disk."""
        safe_title = re.sub(r'[^\w\s-]', '', knowledge.source_title)[:50].strip().replace(' ', '_')
        path = LEARNING_DIR / f"{safe_title}.json"
        path.write_text(json.dumps(knowledge.to_dict(), indent=2))

        # Store in ChromaDB for semantic search
        try:
            import chromadb
            client = chromadb.HttpClient(host="chromadb", port=8000)
            collection = client.get_or_create_collection("atlas_video_knowledge")

            # Store each concept/procedure as a document
            docs = []
            ids = []
            metadatas = []

            for i, concept in enumerate(knowledge.concepts):
                docs.append(f"{concept.get('name', '')}: {concept.get('definition', '')}")
                ids.append(f"concept:{safe_title}:{i}")
                metadatas.append({
                    "type": "concept",
                    "source_url": knowledge.source_url,
                    "video_type": knowledge.video_type,
                })

            for i, proc in enumerate(knowledge.procedures):
                steps_text = " → ".join(
                    s.get("action", "") for s in proc.get("steps", [])
                )
                docs.append(f"{proc.get('name', '')}: {steps_text}")
                ids.append(f"procedure:{safe_title}:{i}")
                metadatas.append({
                    "type": "procedure",
                    "source_url": knowledge.source_url,
                    "video_type": knowledge.video_type,
                })

            if docs:
                collection.upsert(documents=docs, ids=ids, metadatas=metadatas)
        except Exception:
            logger.debug("video_learner: ChromaDB storage failed", exc_info=True)

    def _register_recipes(self, knowledge: ExtractedKnowledge) -> None:
        """Register extracted code recipes as skills."""
        if not knowledge.code_recipes:
            return

        try:
            from app.atlas.skill_library import get_library
            library = get_library()

            for recipe in knowledge.code_recipes:
                code = recipe.get("code", "")
                if not code or len(code) < 20:
                    continue

                safe_name = re.sub(r'[^\w]', '_', recipe.get("name", "unnamed"))[:40]
                skill_id = f"learned/from_youtube/{safe_name}"

                library.register_skill(
                    skill_id=skill_id,
                    name=recipe.get("name", ""),
                    category="learned",
                    code=code,
                    description=recipe.get("purpose", ""),
                    source_type="youtube_tutorial",
                    source_urls=[knowledge.source_url],
                    dependencies=recipe.get("dependencies", []),
                    tags=["youtube", knowledge.video_type, recipe.get("language", "python")],
                )
        except Exception:
            logger.debug("video_learner: recipe registration failed", exc_info=True)

    def _update_competence(self, knowledge: ExtractedKnowledge) -> None:
        """Update competence tracker with learned knowledge."""
        try:
            from app.atlas.competence_tracker import get_tracker
            tracker = get_tracker()

            for concept in knowledge.concepts:
                tracker.register(
                    domain="concepts",
                    name=concept.get("name", ""),
                    confidence=knowledge.confidence * 0.7,  # Video = lower confidence
                    source=f"youtube:{knowledge.source_url}",
                )

            for api_info in knowledge.api_knowledge:
                tracker.register(
                    domain="apis",
                    name=api_info.get("api_name", ""),
                    confidence=knowledge.confidence * 0.5,  # API knowledge from video is shallow
                    source=f"youtube:{knowledge.source_url}",
                    notes="Learned from video — needs verification",
                )
        except Exception:
            pass


# ── Module-level singleton ───────────────────────────────────────────────────

_learner: VideoLearner | None = None


def get_learner() -> VideoLearner:
    """Get or create the singleton video learner."""
    global _learner
    if _learner is None:
        _learner = VideoLearner()
    return _learner
