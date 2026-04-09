"""
fiction_inspiration.py — Science fiction inspiration RAG with absolute epistemic boundary.

Provides agents with access to science fiction concepts as CREATIVE FUEL ONLY.
The fiction collection is epistemically separate from all factual knowledge —
every piece of content is treated as a writer's hallucination that may
contain interesting patterns, metaphors, and creative sparks, but NEVER
verified facts about reality.

Think of it like dreaming: dreams can inspire breakthroughs, reveal
hidden connections, and spark creativity — but you wouldn't cite a dream
as evidence in a research paper.

Safety architecture (5 layers):
    1. Collection separation — fiction and facts in separate ChromaDB collections
    2. Immutable metadata — source_type:"fiction", epistemic_status:"imaginary"
    3. Tool-level framing — ALL results wrapped in "NOT FACT" envelope
    4. Agent system prompts — explicit epistemic boundary instructions
    5. Selective access — only creative agents get fiction tools (NOT Researcher, NOT Self-Improver)

Agent access:
    ✅ Commander — creative problem framing, strategic imagination
    ✅ Coder — creative architecture design, novel approaches
    ✅ Writer — primary consumer, creative content generation
    ❌ Researcher — must stay purely factual
    ❌ Self-Improver — must stay grounded in empirical evaluation
    ❌ Critic — must evaluate against reality, not fiction

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

FICTION_LIBRARY_DIR = Path("/app/workspace/fiction_library")
FICTION_COLLECTION_NAME = "fiction_inspiration"

# Chunking: larger than technical docs to preserve narrative flow
CHUNK_SIZE_CHARS = 6000        # ~1500 tokens
CHUNK_OVERLAP_CHARS = 800      # ~200 tokens

# Structural patterns in .md fiction files
CHAPTER_PATTERN = re.compile(r"^#{1,2}\s+", re.MULTILINE)
SCENE_BREAK_PATTERN = re.compile(r"\n\s*(?:---|\*\*\*|___)\s*\n")
PARAGRAPH_BREAK = "\n\n"


# ── IMMUTABLE epistemic framing ───────────────────────────────────────────────

# This is the critical safety layer. Every piece of fiction the agent sees
# goes through this framing. The envelope makes it impossible for the agent
# to confuse fiction with fact — the same way a dream journal is clearly
# labeled as "dreams, not reality."

INSPIRATION_HEADER = """
╔═══════════════════════════════════════════════════════════════════╗
║  ⚠️  FICTIONAL INSPIRATION — THIS IS NOT FACTUAL KNOWLEDGE       ║
║  Treat this as a writer's hallucination: creative raw material.  ║
╠═══════════════════════════════════════════════════════════════════╣
║  Source: "{book_title}" by {author}                               ║
║  Themes: {themes}                                                 ║
║  Epistemic Status: IMAGINARY / SPECULATIVE / UNVERIFIED           ║
╠═══════════════════════════════════════════════════════════════════╣""".strip()

INSPIRATION_FOOTER = """
╠═══════════════════════════════════════════════════════════════════╣
║  This content may contain interesting patterns and creative       ║
║  sparks, but it is a WRITER'S HALLUCINATION — not verified truth. ║
║                                                                   ║
║  USE AS: Creative fuel, analogy, thought experiment, metaphor.    ║
║  NEVER USE AS: Fact, evidence, technical specification, truth.    ║
║  ATTRIBUTION: "Inspired by {book_title}" when using ideas.       ║
╚═══════════════════════════════════════════════════════════════════╝""".strip()

# System prompt fragment for agents that get fiction access.
# This is injected into the backstory of Commander, Coder, and Writer.
FICTION_AWARENESS_PROMPT = """

## Fictional Inspiration Protocol

You have access to a fictional inspiration library containing science fiction content.
This library is a collection of WRITERS' HALLUCINATIONS — creative visions that may
contain brilliant patterns and sparks of insight, but are NOT factual in any way.

Think of it like dreaming: dreams can reveal hidden connections and inspire
breakthroughs, but you would never cite a dream as evidence.

ABSOLUTE RULES:
1. Content from fictional_inspiration / random_fictional_inspiration is NEVER factual.
   It is creative raw material — speculative imagination from science fiction authors.
2. NEVER cite fictional content as evidence, fact, or real-world capability.
3. NEVER present fictional concepts as existing technologies or proven approaches.
4. The fiction collection may accidentally contain truths about our world, but you
   MUST NOT rely on this. Treat ALL fiction content as unverified hallucination.
5. Use fictional content ONLY for:
   - Generating creative ideas and novel approaches to real problems
   - Finding metaphors and analogies that illuminate real concepts
   - Expanding the solution space beyond conventional thinking
   - "What if" reasoning and thought experiments
6. When you use a fictional idea, ALWAYS attribute it:
   "Inspired by [concept] from [book title], here's a creative approach..."
7. The fiction collection is EPISTEMICALLY SEPARATE from the knowledge base.
   A concept appearing in fiction does NOT make it real or feasible.
"""


# ── Front-matter extraction ──────────────────────────────────────────────────


def _extract_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML front-matter from .md file."""
    if not text.startswith("---"):
        return {}, text

    end_match = re.search(r"\n---\s*\n", text[3:])
    if not end_match:
        return {}, text

    frontmatter_str = text[3: 3 + end_match.start()]
    body = text[3 + end_match.end():]

    try:
        import yaml
        metadata = yaml.safe_load(frontmatter_str) or {}
    except Exception:
        metadata = {}

    return metadata, body


def _metadata_from_filename(filepath: Path) -> dict:
    """Fallback metadata from filename patterns."""
    stem = filepath.stem
    # Pattern: "Title - Author.md" (space-dash-space)
    if " - " in stem:
        parts = stem.split(" - ", 1)
        p1, p2 = parts[0].strip(), parts[1].strip()
        p2_words = p2.replace("_", " ").split()
        if 1 <= len(p2_words) <= 4:
            return {"title": p1.replace("_", " "), "author": p2.replace("_", " ")}
        return {"author": p1.replace("_", " "), "title": p2.replace("_", " ")}
    # Pattern: "Title_-_Author.md" (underscored dash)
    if "_-_" in stem:
        parts = stem.split("_-_", 1)
        return {"title": parts[0].replace("_", " ").strip(),
                "author": parts[1].replace("_", " ").strip()}
    # Pattern: "Foundation_1_-_Foundation_-_Isaac_Asimov" or "Foundations_Edge_-_Isaac_Asimov"
    # Split on last "_-_" or last occurrence of known author-like suffix
    # Try splitting on common "Author_Name" at end after last dash
    parts = stem.rsplit("-", 1)
    if len(parts) == 2:
        title_part = parts[0].rstrip("_ ").replace("_", " ").strip()
        author_part = parts[1].lstrip("_ ").replace("_", " ").strip()
        if author_part and len(author_part.split()) >= 2:
            # Strip leading numbers/underscores from title
            title_part = re.sub(r'^\d+\s*[-_]\s*', '', title_part).strip()
            return {"title": title_part, "author": author_part}
    # Generic: replace underscores with spaces
    clean = stem.replace("_", " ").replace("-", " ").strip()
    # Strip leading numbers like "1  Foundation  Isaac Asimov"
    clean = re.sub(r'^\d+\s+', '', clean)
    return {"title": clean}


def _enrich_metadata(filepath: Path, frontmatter: dict, body: str) -> dict:
    """3-stage metadata extraction: regex → LLM → web search.

    Returns enriched metadata dict merged with existing frontmatter.
    Only runs expensive stages if earlier stages didn't find clean data.
    """
    result = dict(frontmatter)
    has_author = bool(result.get("author"))
    has_title = bool(result.get("title"))
    has_genre = bool(result.get("genre"))
    has_themes = bool(result.get("themes"))

    # Strip epub/HTML artifacts from content before analysis
    clean_body = body
    clean_body = re.sub(r'```\{=html\}.*?```', '', clean_body, flags=re.DOTALL)  # pandoc HTML blocks
    clean_body = re.sub(r':::\s*\{[^}]*\}', '', clean_body)  # pandoc div markers
    clean_body = re.sub(r'!\[.*?\]\([^)]*\)', '', clean_body)  # markdown images
    clean_body = re.sub(r'\[?\]\{[^}]*\}', '', clean_body)  # epub span markers
    clean_body = re.sub(r'\{[#.][^}]*\}', '', clean_body)  # epub class/id markers
    clean_body = re.sub(r'<[^>]+>', '', clean_body)  # HTML tags
    clean_body = re.sub(r'\n{3,}', '\n\n', clean_body)  # collapse blank lines

    # ── Stage 1: Regex extraction from content (free, instant) ──────────
    if not has_author or not has_title:
        head = clean_body[:2000]

        # "by Author Name" pattern
        by_match = re.search(r'\bby\s+([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+){1,3})', head)
        if by_match and not has_author:
            result["author"] = by_match.group(1).strip()
            has_author = True

        # "Copyright © YEAR Author" or "Copyright YEAR Author"
        copy_match = re.search(
            r'[Cc]opyright\s*(?:©|\(c\))?\s*\d{4}\s+([A-Z][a-z]+(?: [A-Z]\.?)?(?: [A-Z][a-z]+){1,3})',
            head,
        )
        if copy_match and not has_author:
            result["author"] = copy_match.group(1).strip()
            has_author = True

        # Title: first non-empty, non-frontmatter line that looks like a title
        if not has_title:
            for line in head.split("\n"):
                line = line.strip().strip("#").strip()
                if (line and len(line) > 3 and len(line) < 120
                        and not line.startswith("by ")
                        and not line.startswith("![")
                        and not line.startswith("[")
                        and not line.startswith("http")
                        and not line.startswith("{")
                        and not re.match(r'^[\W\d]+$', line)):  # Skip lines that are just symbols/numbers
                    result["title"] = line
                    has_title = True
                    break

    # ── Stage 2: LLM extraction (cheap, ~$0.001) ───────────────────────
    if not has_author or not has_title or not has_genre:
        try:
            from app.llm_factory import create_cheap_vetting_llm
            from app.utils import safe_json_parse

            llm = create_cheap_vetting_llm()
            excerpt = clean_body[:3000]
            prompt = (
                "Extract metadata from this book. IMPORTANT RULES:\n"
                "- The FILENAME often contains the real author and title "
                "(separated by dashes or underscores). Use it as primary signal.\n"
                "- NEVER use HTML tags, markdown images (![...]), or "
                "epub artifacts ({=html}, :::{.cover}) as the title.\n"
                "- Author must be a real person's name, not a country or organization.\n"
                "- Genre should be a standard literary genre (e.g., Science Fiction, Fantasy, Literary Fiction).\n\n"
                "Return ONLY a JSON object (no markdown):\n"
                '{"author": "Full Author Name", "title": "Actual Book Title", '
                '"genre": "Primary Genre", "themes": ["theme1", "theme2", "theme3"]}\n\n'
                f"Filename: {filepath.name}\n\n"
                f"First lines of content:\n{excerpt}"
            )
            raw = str(llm.call(prompt)).strip()
            parsed, _ = safe_json_parse(raw)
            if parsed:
                if parsed.get("author"):
                    llm_author = parsed["author"]
                    # Validate: reject obviously wrong authors (countries, organizations, etc.)
                    _bad_authors = {"nazi germany", "unknown", "anonymous", "various", "n/a"}
                    if llm_author.lower().strip() not in _bad_authors:
                        if not has_author:
                            result["author"] = llm_author
                            has_author = True
                        elif result.get("author", "").lower() in _bad_authors:
                            result["author"] = llm_author
                # LLM title overrides bad/suspicious titles
                if parsed.get("title"):
                    existing_title = result.get("title", "")
                    _artifact_patterns = ("![", "[", "See what", "```", ":::", "{", "<", "http")
                    is_bad_title = (
                        not has_title
                        or any(existing_title.startswith(p) for p in _artifact_patterns)
                        or "\\" in existing_title
                        or "html" in existing_title.lower()
                        or "cover" in existing_title.lower()
                        or len(existing_title) < 3
                        or len(existing_title) > 100
                    )
                    if is_bad_title:
                        result["title"] = parsed["title"]
                        has_title = True
                if not has_genre and parsed.get("genre"):
                    result["genre"] = parsed["genre"]
                    has_genre = True
                if not has_themes and parsed.get("themes"):
                    result["themes"] = parsed["themes"]
                    has_themes = True
                logger.info(f"fiction_enrich: LLM extracted → author={result.get('author')}, "
                            f"title={result.get('title')}, genre={result.get('genre')}")
        except Exception as e:
            logger.debug(f"fiction_enrich: LLM extraction failed: {e}")

    # ── Stage 3: Web enrichment for genre/themes ────────────────────────
    if has_author and has_title and (not has_genre or not has_themes):
        try:
            from app.tools.web_search import search_brave
            from app.llm_factory import create_cheap_vetting_llm
            from app.utils import safe_json_parse

            query = f'"{result["author"]}" "{result["title"]}" book genre themes'
            search_results = search_brave(query, count=3)
            if search_results:
                descriptions = " ".join(
                    r.get("description", "")[:200] for r in search_results
                )
                llm = create_cheap_vetting_llm()
                web_prompt = (
                    f"Based on these search results about the book "
                    f'"{result["title"]}" by {result["author"]}:\n\n'
                    f"{descriptions[:1500]}\n\n"
                    "Extract ONLY JSON: "
                    '{"genre": "Primary Genre", "themes": ["theme1", "theme2", "theme3"]}'
                )
                raw = str(llm.call(web_prompt)).strip()
                parsed, _ = safe_json_parse(raw)
                if parsed:
                    if not has_genre and parsed.get("genre"):
                        result["genre"] = parsed["genre"]
                    if not has_themes and parsed.get("themes"):
                        result["themes"] = parsed["themes"]
                    logger.info(f"fiction_enrich: web enriched → genre={result.get('genre')}, "
                                f"themes={result.get('themes')}")
        except Exception as e:
            logger.debug(f"fiction_enrich: web enrichment failed: {e}")

    # ── Write enriched frontmatter back to file ─────────────────────────
    # Only write if we have clean values — skip artifact titles
    _artifact_check = ("![", "[", "```", "`", ":::", "{", "<", "http", "See what")
    title_val = result.get("title", "")
    author_val = result.get("author", "")
    title_clean = (title_val
                   and not any(title_val.startswith(p) for p in _artifact_check)
                   and "\\" not in title_val
                   and "html" not in title_val.lower()
                   and "cover" not in title_val.lower()
                   and len(title_val) >= 3
                   and len(title_val) <= 100)
    author_clean = (author_val
                    and author_val.lower().strip() not in {"nazi germany", "unknown", "anonymous", "n/a", "various"})

    if result and (title_clean or author_clean) and result != frontmatter:
        # Filter out bad values before writing
        write_fm = {k: v for k, v in result.items() if v}
        if not title_clean:
            write_fm.pop("title", None)
        if not author_clean:
            write_fm.pop("author", None)
        try:
            import yaml
            original_text = filepath.read_text(encoding="utf-8")
            _, original_body = _extract_frontmatter(original_text)
            fm_str = yaml.dump(write_fm, default_flow_style=False, allow_unicode=True)
            enriched_text = f"---\n{fm_str}---\n\n{original_body}"
            filepath.write_text(enriched_text, encoding="utf-8")
            logger.info(f"fiction_enrich: wrote enriched frontmatter to {filepath.name}")
        except Exception as e:
            logger.debug(f"fiction_enrich: failed to write frontmatter: {e}")

    return result


# ── Narrative-aware chunking ─────────────────────────────────────────────────


def _chunk_fiction(text: str) -> list[dict]:
    """Split fiction into chunks respecting chapter/scene/paragraph boundaries.

    Returns list of {text, chapter, chunk_index}.
    """
    # Split by chapters
    chapter_splits = CHAPTER_PATTERN.split(text)
    chapter_titles = CHAPTER_PATTERN.findall(text)

    chapters = []
    if chapter_splits[0].strip():
        chapters.append(("Prologue", chapter_splits[0].strip()))

    for i, title_prefix in enumerate(chapter_titles):
        block = chapter_splits[i + 1] if i + 1 < len(chapter_splits) else ""
        first_line_end = block.find("\n")
        if first_line_end > 0:
            chapter_title = (title_prefix + block[:first_line_end]).strip()
            block = block[first_line_end:].strip()
        else:
            chapter_title = (title_prefix + block).strip()
            block = ""
        chapters.append((chapter_title, block))

    chunks = []
    idx = 0

    for chapter_title, chapter_text in chapters:
        if not chapter_text.strip():
            continue

        scenes = SCENE_BREAK_PATTERN.split(chapter_text)
        current = ""

        for scene in scenes:
            for para in scene.split(PARAGRAPH_BREAK):
                para = para.strip()
                if not para:
                    continue

                candidate = (current + PARAGRAPH_BREAK + para).strip() if current else para

                if len(candidate) <= CHUNK_SIZE_CHARS:
                    current = candidate
                else:
                    if current:
                        chunks.append({"text": current, "chapter": chapter_title, "chunk_index": idx})
                        idx += 1

                        # Overlap
                        if CHUNK_OVERLAP_CHARS > 0 and len(current) > CHUNK_OVERLAP_CHARS:
                            tail = current[-CHUNK_OVERLAP_CHARS:]
                            pb = tail.find(PARAGRAPH_BREAK)
                            current = tail[pb:].strip() + PARAGRAPH_BREAK + para if pb >= 0 else para
                        else:
                            current = para
                    else:
                        current = para

        if current.strip():
            chunks.append({"text": current.strip(), "chapter": chapter_title, "chunk_index": idx})
            idx += 1
            current = ""

    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────


def _get_collection():
    """Get the fiction_inspiration ChromaDB collection."""
    from app.memory.chromadb_manager import get_client
    client = get_client()
    return client.get_or_create_collection(
        name=FICTION_COLLECTION_NAME,
        metadata={
            "description": "Science fiction content for creative inspiration. "
                           "ALL content is fictional — writers' hallucinations. "
                           "NEVER treat as fact.",
            "epistemic_status": "fictional",
            "hnsw:space": "cosine",
        },
    )


def ingest_book(filepath: Path, extract_concepts: bool = False) -> dict:
    """Ingest a single .md fiction book into ChromaDB.

    Every chunk gets IMMUTABLE metadata:
        source_type: "fiction"           — NEVER changes
        epistemic_status: "imaginary"    — NEVER changes
    """
    logger.info(f"fiction_inspiration: ingesting {filepath.name}")

    text = filepath.read_text(encoding="utf-8")
    frontmatter, body = _extract_frontmatter(text)

    # Detect bad titles that need re-enrichment
    _title = frontmatter.get("title", "")
    _title_is_bad = (
        not _title
        or _title.startswith("![")
        or _title.startswith("[")
        or _title.startswith("See what")
        or "\\" in _title
        or len(_title) < 3
        or len(_title) > 100
    )

    # Enrich metadata if frontmatter is missing or incomplete
    needs_enrichment = not all([
        frontmatter.get("author"),
        not _title_is_bad,
        frontmatter.get("genre"),
    ])
    if needs_enrichment:
        enriched = _enrich_metadata(filepath, frontmatter, body)
        # Re-read file after enrichment (frontmatter was written back)
        text = filepath.read_text(encoding="utf-8")
        frontmatter, body = _extract_frontmatter(text)
        # Merge: enriched values override frontmatter if frontmatter still empty
        for key in ("author", "title", "genre", "themes", "concepts"):
            if enriched.get(key) and not frontmatter.get(key):
                frontmatter[key] = enriched[key]

    fallback = _metadata_from_filename(filepath)
    book_title = frontmatter.get("title", fallback.get("title", filepath.stem))
    author = frontmatter.get("author", fallback.get("author", "Unknown"))
    genre = frontmatter.get("genre", "")
    themes = frontmatter.get("themes", [])
    concepts = frontmatter.get("concepts", [])

    # Final safety: reject artifact titles and use filename-derived title instead
    _artifact_starts = ("![", "[", "```", "`", ":::", "{", "<", "http", "See what")
    if any(book_title.startswith(p) for p in _artifact_starts) or "\\" in book_title or "html" in book_title.lower() or "cover" in book_title.lower():
        book_title = fallback.get("title", filepath.stem)
    # Final safety: reject bad authors
    if author.lower() in {"nazi germany", "unknown", "anonymous", "n/a", "various", ""}:
        author = fallback.get("author", "Unknown")

    chunks = _chunk_fiction(body)
    if not chunks:
        return {"ingested": 0, "title": book_title}

    collection = _get_collection()

    ids, documents, metadatas = [], [], []

    for chunk in chunks:
        chunk_id = hashlib.sha256(
            f"{book_title}::chunk::{chunk['chunk_index']}".encode()
        ).hexdigest()[:16]

        # Optional: LLM concept extraction
        chunk_concepts = {}
        if extract_concepts:
            chunk_concepts = _extract_concepts_llm(chunk["text"], themes)

        # ── IMMUTABLE EPISTEMIC METADATA ──
        metadata = {
            # These fields are the epistemic boundary — NEVER modify
            "source_type": "fiction",
            "epistemic_status": "imaginary",

            # Book-level
            "book_title": book_title,
            "author": author,
            "genre": genre,
            "themes": json.dumps(themes),
            "book_concepts": json.dumps(concepts),

            # Chunk-level
            "chapter": chunk.get("chapter", ""),
            "chunk_index": chunk["chunk_index"],

            # LLM-extracted (if available)
            "speculative_concepts": json.dumps(
                chunk_concepts.get("speculative_concepts", [])
            ),
            "creative_patterns": json.dumps(
                chunk_concepts.get("creative_patterns", [])
            ),

            # Provenance
            "source_file": str(filepath.name),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(metadata)

    # Batch upsert (idempotent)
    batch = 100
    for i in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[i:i+batch],
            documents=documents[i:i+batch],
            metadatas=metadatas[i:i+batch],
        )

    logger.info(f"fiction_inspiration: ingested {len(ids)} chunks from '{book_title}'")
    return {"ingested": len(ids), "title": book_title, "author": author}


def _extract_concepts_llm(text: str, themes: list) -> dict:
    """Optional: use budget LLM to extract speculative concepts."""
    try:
        from app.llm_factory import create_specialist_llm
        llm = create_specialist_llm(max_tokens=500, role="self_improve")

        theme_hint = f"\nKnown themes: {', '.join(themes)}" if themes else ""
        prompt = (
            f"Extract from this science fiction excerpt:{theme_hint}\n"
            f"1. speculative_concepts: novel fictional ideas (list of short phrases)\n"
            f"2. creative_patterns: problem-solving approaches (list of short phrases)\n"
            f"Return ONLY valid JSON.\n\nTEXT:\n{text[:3000]}"
        )
        raw = str(llm.call(prompt)).strip()
        json_match = re.search(r'\{[\s\S]*?\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception:
        pass
    return {}


def ingest_library(library_dir: Path = FICTION_LIBRARY_DIR,
                   extract_concepts: bool = False) -> dict:
    """Ingest all .md files from the fiction library directory."""
    if not library_dir.exists():
        library_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"fiction_inspiration: created library dir {library_dir}")
        return {"books_ingested": 0}

    md_files = sorted(library_dir.glob("**/*.md"))
    results = []
    for filepath in md_files:
        try:
            result = ingest_book(filepath, extract_concepts=extract_concepts)
            results.append(result)
        except Exception as e:
            logger.warning(f"fiction_inspiration: failed to ingest {filepath}: {e}")

    total = sum(r.get("ingested", 0) for r in results)
    logger.info(f"fiction_inspiration: ingested {total} chunks from {len(results)} books")
    return {"books_ingested": len(results), "total_chunks": total, "books": results}


# ── Retrieval with epistemic framing ─────────────────────────────────────────


def _format_result(document: str, metadata: dict) -> str:
    """Wrap a retrieval result in the epistemic framing envelope.

    This is the critical safety function. Every piece of fiction the agent
    sees goes through here — wrapped as a writer's hallucination.
    """
    book_title = metadata.get("book_title", "Unknown")
    author = metadata.get("author", "Unknown")
    themes_raw = metadata.get("themes", "[]")
    try:
        themes = ", ".join(json.loads(themes_raw))
    except (json.JSONDecodeError, TypeError):
        themes = str(themes_raw)

    chapter = metadata.get("chapter", "")
    chapter_line = f"\n║  Chapter: {chapter}" if chapter else ""

    concepts_raw = metadata.get("speculative_concepts", "[]")
    try:
        concepts = json.loads(concepts_raw)
    except (json.JSONDecodeError, TypeError):
        concepts = []
    concepts_line = (f"\n║  Speculative Concepts: {', '.join(concepts)}"
                     if concepts else "")

    header = INSPIRATION_HEADER.format(
        book_title=book_title, author=author, themes=themes,
    )
    footer = INSPIRATION_FOOTER.format(book_title=book_title)

    return f"""{header}{chapter_line}{concepts_line}
║
{document}
║
{footer}"""


def search_fiction(query: str, n_results: int = 3,
                   theme_filter: str = "") -> str:
    """Search the fiction library for creative inspiration.

    Returns framed results — every result explicitly marked as fiction.
    """
    try:
        collection = _get_collection()
    except Exception as e:
        return f"Fiction library not available: {e}"

    where_filter: dict = {"source_type": "fiction"}
    if theme_filter:
        where_filter = {
            "$and": [
                {"source_type": "fiction"},
                {"themes": {"$contains": theme_filter}},
            ]
        }

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        return f"Fiction search failed: {e}"

    if not results["documents"] or not results["documents"][0]:
        return ("No fictional inspiration found for this query. "
                "Try broader terms or add books to workspace/fiction_library/")

    formatted = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        relevance = f"Relevance: {1 - dist:.2f}" if dist else ""
        formatted.append(f"{_format_result(doc, meta)}\n{relevance}")

    separator = "\n\n" + "─" * 60 + "\n\n"
    return separator.join(formatted)


def random_inspiration(theme_filter: str = "") -> str:
    """Get a random piece of fiction for serendipitous creativity."""
    try:
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return "Fiction library is empty. Add .md books to workspace/fiction_library/"

        offset = random.randint(0, max(0, total - 1))
        where_filter: dict = {"source_type": "fiction"}
        if theme_filter:
            where_filter = {
                "$and": [
                    {"source_type": "fiction"},
                    {"themes": {"$contains": theme_filter}},
                ]
            }

        results = collection.get(
            limit=1, offset=offset, where=where_filter,
            include=["documents", "metadatas"],
        )
        if results["documents"]:
            return _format_result(results["documents"][0], results["metadatas"][0])
        return "No fiction found matching the filter."
    except Exception as e:
        return f"Fiction library error: {e}"


def list_fiction_catalog() -> str:
    """List all books in the fiction library."""
    try:
        collection = _get_collection()
        total = collection.count()
        if total == 0:
            return "Fiction library is empty. Add .md books to workspace/fiction_library/"

        all_meta = collection.get(limit=total, include=["metadatas"])
        books: dict[str, dict] = {}
        for meta in all_meta["metadatas"]:
            title = meta.get("book_title", "Unknown")
            if title not in books:
                themes_raw = meta.get("themes", "[]")
                try:
                    themes = json.loads(themes_raw)
                except (json.JSONDecodeError, TypeError):
                    themes = []
                books[title] = {
                    "author": meta.get("author", "Unknown"),
                    "themes": themes,
                    "chunks": 0,
                }
            books[title]["chunks"] += 1

        lines = [
            "📚 FICTION INSPIRATION LIBRARY",
            "=" * 50,
            f"Total chunks: {total} | Books: {len(books)}",
            "",
            "⚠️  ALL content is FICTIONAL — writers' hallucinations.",
            "    Use for creative inspiration ONLY. Never as fact.",
            "",
        ]
        for title, info in sorted(books.items()):
            lines.append(f"  📖 \"{title}\" by {info['author']}")
            lines.append(f"     Themes: {', '.join(info['themes'])}")
            lines.append(f"     Chunks: {info['chunks']}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Fiction catalog error: {e}"


# ── CrewAI Tools ──────────────────────────────────────────────────────────────

# These are imported lazily to avoid circular deps at module load time.


def get_fiction_tools():
    """Get all fiction inspiration tools for CrewAI agent configuration.

    Usage:
        from app.fiction_inspiration import get_fiction_tools
        fiction_tools = get_fiction_tools()
        # Add to Commander, Coder, Writer — NOT Researcher, Self-Improver
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field

        class _ConceptInput(BaseModel):
            query: str = Field(description="Concept or theme to find fictional inspiration about")
            n_results: int = Field(default=3, ge=1, le=10)
            theme_filter: str = Field(default="", description="Optional theme filter")

        class _RandomInput(BaseModel):
            theme_filter: str = Field(default="", description="Optional theme filter")

        class _CatalogInput(BaseModel):
            pass

        class FictionalInspirationTool(BaseTool):
            name: str = "fictional_inspiration"
            description: str = (
                "Search the science fiction library for creative inspiration. "
                "Returns fictional passages — WRITERS' HALLUCINATIONS — that can "
                "spark creative ideas. ⚠️ ALL content is FICTIONAL and unverified. "
                "Use for: brainstorming, analogies, expanding solution space. "
                "NEVER use as: fact, evidence, or technical specification."
            )
            args_schema: Type[BaseModel] = _ConceptInput

            def _run(self, query: str, n_results: int = 3, theme_filter: str = "") -> str:
                return search_fiction(query, n_results, theme_filter)

        class RandomInspirationTool(BaseTool):
            name: str = "random_fictional_inspiration"
            description: str = (
                "Get a random piece of science fiction for serendipitous creativity. "
                "⚠️ Content is FICTIONAL — a writer's hallucination, not fact."
            )
            args_schema: Type[BaseModel] = _RandomInput

            def _run(self, theme_filter: str = "") -> str:
                return random_inspiration(theme_filter)

        class FictionCatalogTool(BaseTool):
            name: str = "fiction_library_catalog"
            description: str = "List all science fiction books in the inspiration library."
            args_schema: Type[BaseModel] = _CatalogInput

            def _run(self) -> str:
                return list_fiction_catalog()

        return [FictionalInspirationTool(), RandomInspirationTool(), FictionCatalogTool()]

    except ImportError:
        logger.debug("fiction_inspiration: CrewAI not available for tool creation")
        return []


# ── Which agents get fiction access ──────────────────────────────────────────

# IMMUTABLE: only creative agents get fiction. Factual and self-improvement
# agents must never have access to prevent epistemic contamination.
FICTION_ENABLED_AGENTS = frozenset({
    "commander",     # creative problem framing
    "coder",         # creative architecture design
    "writer",        # primary creative consumer
    "media_analyst", # creative content analysis
})

FICTION_DISABLED_AGENTS = frozenset({
    "researcher",    # must stay purely factual
    "self_improver", # must stay empirically grounded
    "critic",        # must evaluate against reality
})


def agent_has_fiction_access(agent_name: str) -> bool:
    """Check if an agent should have fiction tools."""
    return agent_name.lower() in FICTION_ENABLED_AGENTS


# ── Module-level initialization ──────────────────────────────────────────────


def initialize() -> dict:
    """Initialize fiction library — ingest any books in the library dir."""
    FICTION_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        collection = _get_collection()
        count = collection.count()
        if count > 0:
            return {"status": "ready", "chunks": count}
    except Exception:
        pass

    # Auto-ingest if books present
    result = ingest_library()
    return {"status": "initialized", **result}
