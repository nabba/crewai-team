"""
Firecrawl integration tools for crewai-team.
Wraps firecrawl-py SDK for self-hosted instance.

Five tools:
  1. firecrawl_scrape  — Single page → clean markdown
  2. firecrawl_extract — Page → structured JSON via LLM extraction
  3. firecrawl_crawl   — Multi-page site crawl → markdown corpus
  4. firecrawl_search  — Web search → full page content (requires SearXNG/Serper)
  5. firecrawl_map     — URL discovery (sitemap + link crawling)

CONSTITUTIONAL CONSTRAINT: Rate limiting is enforced at the infrastructure level
(Firecrawl Redis), NOT in this code — consistent with the DGM safety invariant.
Robots.txt is respected by default by Firecrawl.

IMMUTABLE — tool-level module.
"""

import json
import logging
import os
from typing import Optional, Type

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Singleton client ─────────────────────────────────────────────────────────

_firecrawl_client = None
_FIRECRAWL_AVAILABLE = True


def get_firecrawl_client():
    """Returns a singleton Firecrawl client pointed at the self-hosted instance."""
    global _firecrawl_client, _FIRECRAWL_AVAILABLE
    if not _FIRECRAWL_AVAILABLE:
        return None
    if _firecrawl_client is not None:
        return _firecrawl_client
    try:
        from firecrawl import FirecrawlApp
        api_url = os.getenv("FIRECRAWL_API_URL", "http://firecrawl-api:3002")
        api_key = os.getenv("FIRECRAWL_API_KEY", "self-hosted")
        _firecrawl_client = FirecrawlApp(api_key=api_key, api_url=api_url)
        logger.info(f"Firecrawl client initialized: {api_url}")
        return _firecrawl_client
    except ImportError:
        logger.warning("firecrawl-py not installed — Firecrawl tools unavailable")
        _FIRECRAWL_AVAILABLE = False
        return None
    except Exception as e:
        logger.warning(f"Firecrawl client init failed: {e}")
        _FIRECRAWL_AVAILABLE = False
        return None


def is_available() -> bool:
    """Check if Firecrawl is available."""
    return get_firecrawl_client() is not None


# ── Tool 1: Smart Scrape (single page → markdown) ───────────────────────────

class ScrapeInput(BaseModel):
    url: str = Field(description="The URL to scrape")
    only_main_content: bool = Field(default=True, description="Exclude headers/footers/nav")

def firecrawl_scrape(url: str, only_main_content: bool = True) -> str:
    """Scrape a single web page and return clean, LLM-ready markdown.

    Use for reading articles, docs, product pages. Does NOT follow links.
    """
    client = get_firecrawl_client()
    if not client:
        return "Firecrawl not available. Use web_search tool instead."
    try:
        result = client.scrape(url, formats=["markdown"], only_main_content=only_main_content)
        content = getattr(result, "markdown", "") or getattr(result, "html", "") or ""
        meta = getattr(result, "metadata", None)
        title = getattr(meta, "title", "Unknown") if meta else "Unknown"
        source = getattr(meta, "source_url", url) if meta else url

        if len(content) > 8000:
            content = content[:8000] + f"\n\n[TRUNCATED — full page is {len(content)} chars]"

        return f"# {title}\nSource: {source}\n\n{content}"
    except Exception as e:
        logger.error(f"Firecrawl scrape failed for {url}: {e}")
        return f"Error scraping {url}: {str(e)[:200]}"


# ── Tool 2: Structured Extract (page → typed JSON via LLM) ──────────────────

class ExtractInput(BaseModel):
    url: str = Field(description="The URL to extract structured data from")
    prompt: str = Field(description="What to extract (natural language)")
    schema_json: Optional[str] = Field(default=None, description="Optional JSON schema for output")

def firecrawl_extract(url: str, prompt: str, schema_json: str = None) -> str:
    """Extract structured data from a web page using LLM-powered extraction.

    Uses local Ollama model. Best for: pricing tables, contact info, specs.
    """
    client = get_firecrawl_client()
    if not client:
        return "Firecrawl not available."
    try:
        kwargs = {"urls": [url], "prompt": prompt}
        if schema_json:
            try:
                kwargs["schema"] = json.loads(schema_json)
            except json.JSONDecodeError:
                return "Error: schema_json is not valid JSON"

        result = client.extract(**kwargs)
        # Result is a Pydantic model — convert to dict
        data = result.model_dump() if hasattr(result, "model_dump") else {"raw": str(result)}
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Firecrawl extract failed for {url}: {e}")
        return f"Error extracting from {url}: {str(e)[:200]}"


# ── Tool 3: Crawl (multi-page → markdown corpus) ────────────────────────────

class CrawlInput(BaseModel):
    url: str = Field(description="Starting URL to crawl from")
    max_pages: int = Field(default=20, description="Max pages to crawl (keep low)")
    include_patterns: Optional[list[str]] = Field(default=None, description="URL glob patterns to include")
    exclude_patterns: Optional[list[str]] = Field(default=None, description="URL glob patterns to exclude")

def firecrawl_crawl(url: str, max_pages: int = 20,
                    include_patterns: list[str] = None,
                    exclude_patterns: list[str] = None) -> str:
    """Crawl a website, following links to discover and scrape multiple pages.

    WARNING: Resource-intensive. Set max_pages conservatively (default 20, hard cap 50).
    Best for: documentation sites, knowledge bases, competitor analysis.
    """
    max_pages = min(max_pages, 50)  # Hard cap
    client = get_firecrawl_client()
    if not client:
        return "Firecrawl not available."
    try:
        crawl_params = {
            "limit": max_pages,
            "scrapeOptions": {
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        }
        if include_patterns:
            crawl_params["includePaths"] = include_patterns
        if exclude_patterns:
            crawl_params["excludePaths"] = exclude_patterns

        result = client.crawl(url, limit=max_pages,
                              include_paths=include_patterns,
                              exclude_paths=exclude_patterns,
                              scrape_options={"formats": ["markdown"], "onlyMainContent": True})

        # CrawlJob has .data (list of Document objects)
        pages = getattr(result, "data", []) or []
        output_parts = [f"Crawled {len(pages)} pages from {url}\n"]

        for i, page in enumerate(pages):
            meta = getattr(page, "metadata", None)
            title = getattr(meta, "title", f"Page {i+1}") if meta else f"Page {i+1}"
            page_url = getattr(meta, "source_url", "") if meta else ""
            content = (getattr(page, "markdown", "") or "")[:3000]
            output_parts.append(f"\n---\n## [{i+1}] {title}\nURL: {page_url}\n\n{content}")

        return "\n".join(output_parts)
    except Exception as e:
        logger.error(f"Firecrawl crawl failed for {url}: {e}")
        return f"Error crawling {url}: {str(e)[:200]}"


# ── Tool 4: Web Search (search + scrape in one) ─────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=5, description="Number of results (1-10)")

def firecrawl_search(query: str, limit: int = 5) -> str:
    """Search the web and return full page content (not just snippets).

    NOTE: Requires SearXNG or SERPER_API_KEY configured on the Firecrawl instance.
    """
    limit = min(max(limit, 1), 10)
    client = get_firecrawl_client()
    if not client:
        return "Firecrawl not available. Use web_search tool instead."
    try:
        results = client.search(query, limit=limit)
        data = getattr(results, "data", []) or []
        output_parts = [f"Search results for: '{query}'\n"]

        for i, item in enumerate(data):
            meta = getattr(item, "metadata", None)
            title = getattr(meta, "title", f"Result {i+1}") if meta else getattr(item, "title", f"Result {i+1}")
            item_url = getattr(meta, "source_url", "") if meta else getattr(item, "url", "")
            content = (getattr(item, "markdown", "") or "")[:2000]
            output_parts.append(f"\n---\n### [{i+1}] {title}\nURL: {item_url}\n\n{content}")

        return "\n".join(output_parts)
    except Exception as e:
        logger.error(f"Firecrawl search failed for '{query}': {e}")
        return f"Error searching '{query}': {str(e)[:200]}"


# ── Tool 5: Site Map (URL discovery) ─────────────────────────────────────────

class MapInput(BaseModel):
    url: str = Field(description="Website URL to map")

def firecrawl_map(url: str) -> str:
    """Discover all accessible URLs on a website. Fast — uses sitemaps + link crawling.

    Use before firecrawl_crawl to understand site structure.
    """
    client = get_firecrawl_client()
    if not client:
        return "Firecrawl not available."
    try:
        result = client.map(url)
        urls = getattr(result, "links", []) or []
        return f"Found {len(urls)} URLs on {url}:\n\n" + "\n".join(str(u) for u in urls[:100])
    except Exception as e:
        logger.error(f"Firecrawl map failed for {url}: {e}")
        return f"Error mapping {url}: {str(e)[:200]}"


# ── RAG Pipeline: Firecrawl → ChromaDB ingestion ────────────────────────────

def ingest_url_to_chromadb(
    url: str,
    collection_name: str = "web_knowledge",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    tags: dict = None,
) -> dict:
    """Scrape a URL via Firecrawl and ingest into ChromaDB.

    Returns: {chunks_ingested, source_url, content_hash}
    """
    import hashlib
    from datetime import datetime, timezone

    client = get_firecrawl_client()
    if not client:
        return {"chunks_ingested": 0, "error": "Firecrawl not available"}

    try:
        result = client.scrape(url, formats=["markdown"], only_main_content=True)
    except Exception as e:
        return {"chunks_ingested": 0, "error": str(e)[:200]}

    markdown = getattr(result, "markdown", "") or ""
    meta = getattr(result, "metadata", None)

    if not markdown:
        return {"chunks_ingested": 0, "error": "No content extracted"}

    content_hash = hashlib.sha256(markdown.encode()).hexdigest()[:16]

    # Chunk
    chunks = []
    start = 0
    while start < len(markdown):
        end = start + chunk_size
        chunk = markdown[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap

    # Build metadata
    base_meta = {
        "source_url": url,
        "page_title": getattr(meta, "title", "") if meta else "",
        "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": content_hash,
        "epistemological_tag": "web_source",
        "verification_status": "unverified",
    }
    if tags:
        base_meta.update(tags)

    # Ingest
    try:
        import chromadb
        chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
        collection = chroma_client.get_or_create_collection(collection_name)

        ids = [f"{content_hash}_{i}" for i in range(len(chunks))]
        metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as e:
        return {"chunks_ingested": 0, "error": f"ChromaDB ingest failed: {str(e)[:200]}"}

    return {
        "chunks_ingested": len(chunks),
        "source_url": url,
        "content_hash": content_hash,
        "page_title": getattr(meta, "title", "") if meta else "",
    }


def ingest_crawl_to_chromadb(
    url: str,
    max_pages: int = 20,
    collection_name: str = "web_knowledge",
    tags: dict = None,
) -> dict:
    """Crawl a site and ingest all pages into ChromaDB."""
    client = get_firecrawl_client()
    if not client:
        return {"pages_ingested": 0, "error": "Firecrawl not available"}

    try:
        result = client.crawl(url, limit=min(max_pages, 50),
                              scrape_options={"formats": ["markdown"], "onlyMainContent": True})
    except Exception as e:
        return {"pages_ingested": 0, "error": str(e)[:200]}

    import hashlib
    from datetime import datetime, timezone

    total_chunks = 0
    pages_ingested = 0
    pages = getattr(result, "data", []) or []

    for page in pages:
        page_meta = getattr(page, "metadata", None)
        page_url = getattr(page_meta, "source_url", url) if page_meta else url
        markdown = getattr(page, "markdown", "") or ""
        if not markdown:
            continue

        content_hash = hashlib.sha256(markdown.encode()).hexdigest()[:16]
        chunks = []
        start = 0
        while start < len(markdown):
            end = start + 1000
            c = markdown[start:end].strip()
            if c:
                chunks.append(c)
            start = end - 200

        try:
            import chromadb
            chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
            collection = chroma_client.get_or_create_collection(collection_name)
            ids = [f"{content_hash}_{i}" for i in range(len(chunks))]
            meta = {
                "source_url": page_url,
                "page_title": getattr(page_meta, "title", "") if page_meta else "",
                "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
                "epistemological_tag": "web_source",
                "verification_status": "unverified",
            }
            if tags:
                meta.update(tags)
            metadatas = [{**meta, "chunk_index": i} for i in range(len(chunks))]
            collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
            total_chunks += len(chunks)
            pages_ingested += 1
        except Exception:
            logger.debug(f"ChromaDB ingest failed for {page_url}", exc_info=True)

    return {
        "pages_ingested": pages_ingested,
        "total_chunks": total_chunks,
        "source_url": url,
    }


# ── CrewAI Tool wrappers (for tool registration with agents) ─────────────────

def create_firecrawl_tools() -> list:
    """Create CrewAI-compatible tool wrappers. Gracefully returns empty if unavailable."""
    if not is_available():
        logger.info("Firecrawl not available — no tools registered")
        return []

    try:
        from crewai.tools import tool

        @tool("firecrawl_scrape")
        def scrape_tool(url: str, only_main_content: bool = True) -> str:
            """Scrape a single web page and return clean markdown. Use for articles, docs, product pages."""
            return firecrawl_scrape(url, only_main_content)

        @tool("firecrawl_extract")
        def extract_tool(url: str, prompt: str, schema_json: str = "") -> str:
            """Extract structured data from a web page using LLM. Best for pricing, specs, contacts."""
            return firecrawl_extract(url, prompt, schema_json or None)

        @tool("firecrawl_crawl")
        def crawl_tool(url: str, max_pages: int = 20) -> str:
            """Crawl a website and return markdown from multiple pages. Resource-intensive — use sparingly."""
            return firecrawl_crawl(url, max_pages)

        @tool("firecrawl_search")
        def search_tool(query: str, limit: int = 5) -> str:
            """Search the web and return full page content (not just snippets)."""
            return firecrawl_search(query, limit)

        @tool("firecrawl_map")
        def map_tool(url: str) -> str:
            """Discover all URLs on a website. Fast. Use before crawl to understand site structure."""
            return firecrawl_map(url)

        return [scrape_tool, extract_tool, crawl_tool, search_tool, map_tool]
    except Exception as e:
        logger.warning(f"Failed to create Firecrawl CrewAI tools: {e}")
        return []
