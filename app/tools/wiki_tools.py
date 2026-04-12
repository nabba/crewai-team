"""
LLM Wiki Tools for CrewAI agents.

Four tools for managing the AndrusAI Knowledge Wiki:
  - WikiReadTool   — read wiki pages (full or frontmatter-only)
  - WikiWriteTool  — create / update / deprecate pages with locking
  - WikiSearchTool — grep-based keyword search across wiki
  - WikiLintTool   — 8 health checks for wiki integrity

WIKI_ROOT defaults to /app/wiki (Docker) with fallback to repo-relative wiki/.
"""

import json
import math
import os
import re
import glob
import time
import fcntl
import hashlib
import threading
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Optional

import yaml
from crewai.tools import BaseTool
from pydantic import Field


# ---------------------------------------------------------------------------
# Wiki root resolution
# ---------------------------------------------------------------------------

_DOCKER_WIKI = "/app/wiki"
_REPO_WIKI = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "wiki")
)
WIKI_ROOT = _DOCKER_WIKI if os.path.isdir(_DOCKER_WIKI) else _REPO_WIKI

VALID_SECTIONS = {"meta", "self", "philosophy", "plg", "archibal", "kaicart"}
REQUIRED_FRONTMATTER = {
    "title", "section", "created_at", "updated_at", "author",
    "status", "confidence", "tags", "related", "source",
}
VALID_STATUSES = {"draft", "active", "deprecated"}
VALID_CONFIDENCE = {"low", "medium", "high", "verified"}
VALID_RELATIONSHIP_TYPES = {
    "supports", "contradicts", "supersedes", "prerequisite",
    "tested_by", "refines", "extends",
}

LOCKS_DIR = os.path.join(WIKI_ROOT, ".locks")
SLIDES_DIR = os.path.join(WIKI_ROOT, ".slides")

# Multi-agent write coordination: max 3 concurrent writers
_WRITE_SEMAPHORE = threading.Semaphore(3)

# ChromaDB collection for semantic search (Phase 2)
_WIKI_COLLECTION = "andrusai_wiki_pages"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_path(path: str) -> str:
    """Resolve path and ensure it stays inside WIKI_ROOT."""
    resolved = os.path.normpath(os.path.join(WIKI_ROOT, path))
    if not resolved.startswith(os.path.normpath(WIKI_ROOT)):
        raise ValueError(f"Path traversal blocked: {path}")
    return resolved


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown body."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        fm = {}
    body = parts[2].lstrip("\n")
    return fm, body


def _render_frontmatter(fm: dict) -> str:
    """Render dict as YAML frontmatter block."""
    return "---\n" + yaml.dump(fm, default_flow_style=False, allow_unicode=True).rstrip() + "\n---\n"


def _acquire_lock(slug: str, timeout: int = 10) -> Optional[object]:
    """Acquire a file lock for a wiki page slug. Returns file handle or None."""
    os.makedirs(LOCKS_DIR, exist_ok=True)
    _cleanup_stale_locks()  # Remove any stale lock files from crashed processes
    lock_path = os.path.join(LOCKS_DIR, f"{slug}.lock")
    fh = open(lock_path, "w")
    start = time.time()
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except (IOError, OSError):
            if time.time() - start > timeout:
                fh.close()
                return None
            time.sleep(0.2)


def _release_lock(fh):
    """Release a file lock."""
    if fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
            fh.close()
        except Exception:
            pass


def _cleanup_stale_locks(max_age_s: int = 300):
    """Remove lock files older than max_age_s (default 5 min). fcntl releases
    the OS lock on process death, but the file persists — clean it up."""
    try:
        for fname in os.listdir(LOCKS_DIR):
            if not fname.endswith(".lock"):
                continue
            fpath = os.path.join(LOCKS_DIR, fname)
            age = time.time() - os.path.getmtime(fpath)
            if age > max_age_s:
                try:
                    os.remove(fpath)
                except OSError:
                    pass
    except Exception:
        pass


def _embed_wiki_page(section: str, slug: str, title: str, content: str,
                      tags: list = None, confidence: str = "medium"):
    """Embed a wiki page into ChromaDB for semantic search (Phase 2)."""
    try:
        from app.memory.chromadb_manager import store
        # Embedding text: title + first 500 chars of content + tags
        embed_text = f"{title}\n{content[:500]}\n{' '.join(tags or [])}"
        doc_id = f"{section}/{slug}"
        store(_WIKI_COLLECTION, embed_text, metadata={
            "section": section,
            "slug": slug,
            "title": title,
            "confidence": confidence,
            "doc_id": doc_id,
        })
    except Exception:
        pass  # ChromaDB unavailable — grep fallback still works


def _bm25_score(query: str, document: str) -> float:
    """Lightweight BM25-inspired relevance score (no external deps).

    Combines term frequency and inverse document frequency approximation
    for hybrid search ranking alongside ChromaDB semantic scores.
    """
    query_terms = set(query.lower().split())
    doc_terms = document.lower().split()
    if not query_terms or not doc_terms:
        return 0.0
    doc_len = len(doc_terms)
    avg_len = 300  # Approximate average wiki page length in words
    k1 = 1.5
    b = 0.75
    score = 0.0
    term_counts = Counter(doc_terms)
    for term in query_terms:
        tf = term_counts.get(term, 0)
        if tf == 0:
            continue
        # IDF approximation (assume ~100 docs, term appears in ~10)
        idf = math.log((100 - 10 + 0.5) / (10 + 0.5) + 1)
        # BM25 TF component
        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
        score += idf * tf_component
    return score


def _append_log(agent: str, action: str, path: str, summary: str):
    """Append an entry to wiki/log.md."""
    log_path = os.path.join(WIKI_ROOT, "log.md")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"| {timestamp} | {agent} | {action} | {path} | {summary} |\n"
    with open(log_path, "a") as f:
        f.write(entry)


def _rebuild_section_index(section: str):
    """Rebuild a section's index.md based on actual pages present."""
    section_dir = os.path.join(WIKI_ROOT, section)
    if not os.path.isdir(section_dir):
        return

    pages = []
    for fname in sorted(os.listdir(section_dir)):
        if fname == "index.md" or not fname.endswith(".md"):
            continue
        fpath = os.path.join(section_dir, fname)
        with open(fpath, "r") as f:
            fm, _ = _parse_frontmatter(f.read())
        slug = fname[:-3]
        title = fm.get("title", slug)
        status = fm.get("status", "unknown")
        pages.append((slug, title, status))

    page_count = len(pages)

    # Build index content
    fm_data = {
        "title": f"{section.capitalize()} — Section Index",
        "section": section,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_count": page_count,
    }
    lines = [
        _render_frontmatter(fm_data),
        f"# {section.capitalize()} Knowledge Wiki\n",
        "## Pages\n",
    ]
    if pages:
        for slug, title, status in pages:
            marker = " *(deprecated)*" if status == "deprecated" else ""
            lines.append(f"- [[{section}/{slug}]] — {title}{marker}\n")
    else:
        lines.append("(No pages yet.)\n")

    lines.append("\n## Key Relationships\n(Updated as pages are added.)\n")

    index_path = os.path.join(section_dir, "index.md")
    with open(index_path, "w") as f:
        f.write("".join(lines))


def _rebuild_master_index():
    """Rebuild wiki/index.md from all section indexes."""
    section_counts = {}
    for section in sorted(VALID_SECTIONS):
        section_dir = os.path.join(WIKI_ROOT, section)
        if not os.path.isdir(section_dir):
            section_counts[section] = 0
            continue
        count = sum(
            1 for fname in os.listdir(section_dir)
            if fname.endswith(".md") and fname != "index.md"
        )
        section_counts[section] = count

    total = sum(section_counts.values())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    fm_data = {
        "title": "AndrusAI Wiki — Master Index",
        "updated_at": now,
        "total_pages": total,
        "sections": section_counts,
    }

    section_labels = {
        "meta": "Meta (Cross-Venture)",
        "self": "Self (Agent Self-Knowledge)",
        "philosophy": "Philosophy (Compiled Philosophical Frameworks)",
        "plg": "PLG",
        "archibal": "Archibal",
        "kaicart": "KaiCart",
    }

    lines = [
        _render_frontmatter(fm_data),
        f"# AndrusAI Knowledge Wiki — Master Index\n\n",
        f"Total pages: {total} | Last updated: {today}\n",
    ]

    for section in ["meta", "self", "philosophy", "plg", "archibal", "kaicart"]:
        label = section_labels.get(section, section.capitalize())
        lines.append(f"\n## {label}\n")
        section_dir = os.path.join(WIKI_ROOT, section)
        if not os.path.isdir(section_dir):
            lines.append("(No pages yet.)\n")
            continue
        found = False
        for fname in sorted(os.listdir(section_dir)):
            if fname == "index.md" or not fname.endswith(".md"):
                continue
            fpath = os.path.join(section_dir, fname)
            with open(fpath, "r") as f:
                fm, _ = _parse_frontmatter(f.read())
            slug = fname[:-3]
            title = fm.get("title", slug)
            status = fm.get("status", "")
            marker = " *(deprecated)*" if status == "deprecated" else ""
            lines.append(f"- [[{section}/{slug}]] — {title}{marker}\n")
            found = True
        if not found:
            lines.append("(No pages yet.)\n")

    index_path = os.path.join(WIKI_ROOT, "index.md")
    with open(index_path, "w") as f:
        f.write("".join(lines))


# ---------------------------------------------------------------------------
# WikiReadTool
# ---------------------------------------------------------------------------

class WikiReadTool(BaseTool):
    name: str = "wiki_read"
    description: str = (
        "Read a wiki page by its path (e.g. 'meta/some-page' or 'philosophy/index'). "
        "Set frontmatter_only=true to get just the YAML metadata without the body. "
        "Args: path (str) — relative path inside wiki/ without .md extension; "
        "frontmatter_only (bool, default false)."
    )

    def _run(self, path: str, frontmatter_only: bool = False) -> str:
        if not path or not isinstance(path, str):
            return "Error: path is required."

        # Normalize: strip .md if provided
        path = path.strip().rstrip("/")
        if path.endswith(".md"):
            path = path[:-3]

        file_path = _safe_path(path + ".md")

        if not os.path.isfile(file_path):
            return f"Error: page not found — {path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

        if frontmatter_only:
            fm, _ = _parse_frontmatter(content)
            if not fm:
                return f"No frontmatter found in {path}"
            return yaml.dump(fm, default_flow_style=False, allow_unicode=True)

        return content


# ---------------------------------------------------------------------------
# WikiWriteTool
# ---------------------------------------------------------------------------

class WikiWriteTool(BaseTool):
    name: str = "wiki_write"
    description: str = (
        "Create, update, or deprecate a wiki page. "
        "Args: action (str) — 'create' | 'update' | 'deprecate'; "
        "section (str) — one of meta/self/philosophy/plg/archibal/kaicart; "
        "slug (str) — kebab-case page name; "
        "title (str) — human-readable title; "
        "content (str) — full markdown body (for create/update); "
        "author (str) — agent name performing the write; "
        "confidence (str) — low/medium/high/verified; "
        "tags (str) — comma-separated tags; "
        "related (str) — comma-separated related page slugs; "
        "source (str) — raw path, URL, or 'synthesis'; "
        "deprecated_by (str, optional) — slug of replacement page (for deprecate); "
        "relationships (str, optional) — typed links, format: 'supports:page-slug,contradicts:other-slug'."
    )

    def _run(
        self,
        action: str,
        section: str,
        slug: str,
        author: str,
        title: str = "",
        content: str = "",
        confidence: str = "medium",
        tags: str = "",
        related: str = "",
        source: str = "synthesis",
        deprecated_by: str = "",
        relationships: str = "",
    ) -> str:
        # Validate inputs
        action = (action or "").strip().lower()
        if action not in ("create", "update", "deprecate"):
            return "Error: action must be 'create', 'update', or 'deprecate'."

        section = (section or "").strip().lower()
        if section not in VALID_SECTIONS:
            return f"Error: section must be one of {sorted(VALID_SECTIONS)}."

        slug = (slug or "").strip().lower()
        if not slug or not re.match(r"^[a-z0-9][a-z0-9-]*$", slug):
            return "Error: slug must be non-empty kebab-case (lowercase, hyphens, no leading hyphen)."

        if confidence not in VALID_CONFIDENCE:
            return f"Error: confidence must be one of {sorted(VALID_CONFIDENCE)}."

        author = (author or "").strip()
        if not author:
            return "Error: author is required."

        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        related_list = [r.strip() for r in related.split(",") if r.strip()] if related else []

        # Parse typed relationships: "supports:page-slug,contradicts:other-slug"
        rel_list = []
        if relationships:
            for rel in relationships.split(","):
                rel = rel.strip()
                if ":" in rel:
                    rtype, rtarget = rel.split(":", 1)
                    rtype = rtype.strip().lower()
                    rtarget = rtarget.strip()
                    if rtype in VALID_RELATIONSHIP_TYPES and rtarget:
                        rel_list.append({"type": rtype, "target": rtarget})

        file_path = _safe_path(os.path.join(section, slug + ".md"))
        page_ref = f"{section}/{slug}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # ── DGM Epistemic Boundary Enforcement (write-time, not just lint) ──
        # factual/verified pages MUST have a concrete source (not just "synthesis")
        if confidence in ("high", "verified") and source in ("synthesis", ""):
            return (
                f"Error: DGM epistemic violation — confidence='{confidence}' requires "
                f"a concrete source (not '{source}'). Provide a raw/ path or specific reference."
            )
        # Creative content cannot be filed in venture sections
        if source == "creative" and section in ("plg", "archibal", "kaicart"):
            return (
                "Error: DGM epistemic boundary — creative-tagged content cannot be filed "
                "in venture sections. Use section='philosophy' or 'meta' instead."
            )

        # Multi-agent coordination: acquire global write semaphore first
        if not _WRITE_SEMAPHORE.acquire(timeout=15):
            return "Error: wiki write queue full (3 concurrent writers). Try again shortly."

        # Acquire page-level lock
        lock_key = f"{section}_{slug}"
        lock_fh = _acquire_lock(lock_key)
        if lock_fh is None:
            _WRITE_SEMAPHORE.release()
            return f"Error: could not acquire lock for {page_ref} — another agent may be writing."

        try:
            if action == "create":
                if os.path.isfile(file_path):
                    return f"Error: page already exists — {page_ref}. Use action='update' instead."
                if not title:
                    return "Error: title is required for create."
                if not content:
                    return "Error: content is required for create."

                fm = {
                    "title": title,
                    "section": section,
                    "created_at": now,
                    "updated_at": now,
                    "date": now[:10],  # Dataview-compatible ISO date
                    "author": author,
                    "status": "active",
                    "confidence": confidence,
                    "tags": tag_list,
                    "aliases": [slug.replace("-", " ")],  # Dataview: alternative names
                    "related": related_list,
                    "relationships": rel_list,  # Typed: [{type, target}]
                    "source": source,
                    "version": 1,
                }
                full_content = _render_frontmatter(fm) + "\n" + content.strip() + "\n"

                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_content)

                _append_log(author, "CREATE", page_ref, f"Created: {title}")
                _rebuild_section_index(section)
                _rebuild_master_index()
                # ChromaDB: embed page for semantic search (Phase 2)
                _embed_wiki_page(section, slug, title, content, tag_list, confidence)
                return f"Created {page_ref} (v1)."

            elif action == "update":
                if not os.path.isfile(file_path):
                    return f"Error: page not found — {page_ref}. Use action='create' first."
                if not content:
                    return "Error: content is required for update."

                with open(file_path, "r", encoding="utf-8") as f:
                    old_fm, _ = _parse_frontmatter(f.read())

                version = old_fm.get("version", 1) + 1
                fm = dict(old_fm)
                fm["updated_at"] = now
                fm["version"] = version
                fm["author"] = author
                if title:
                    fm["title"] = title
                if confidence:
                    fm["confidence"] = confidence
                if tag_list:
                    fm["tags"] = tag_list
                if related_list:
                    fm["related"] = related_list
                if source:
                    fm["source"] = source
                if rel_list:
                    fm["relationships"] = rel_list

                full_content = _render_frontmatter(fm) + "\n" + content.strip() + "\n"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_content)

                _append_log(author, "UPDATE", page_ref, f"Updated to v{version}")
                _rebuild_section_index(section)
                _rebuild_master_index()
                # ChromaDB: re-embed updated page for semantic search
                _embed_wiki_page(section, slug, fm.get("title", slug), content,
                                 fm.get("tags", []), fm.get("confidence", "medium"))
                return f"Updated {page_ref} (v{version})."

            elif action == "deprecate":
                if not os.path.isfile(file_path):
                    return f"Error: page not found — {page_ref}."

                with open(file_path, "r", encoding="utf-8") as f:
                    old_content = f.read()
                old_fm, old_body = _parse_frontmatter(old_content)

                old_fm["status"] = "deprecated"
                old_fm["updated_at"] = now
                old_fm["version"] = old_fm.get("version", 1) + 1
                if deprecated_by:
                    old_fm["deprecated_by"] = deprecated_by

                full_content = _render_frontmatter(old_fm) + "\n" + old_body
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_content)

                summary = f"Deprecated"
                if deprecated_by:
                    summary += f" (replaced by {deprecated_by})"
                _append_log(author, "DEPRECATE", page_ref, summary)
                _rebuild_section_index(section)
                _rebuild_master_index()
                return f"Deprecated {page_ref}."

        finally:
            _release_lock(lock_fh)
            _WRITE_SEMAPHORE.release()


# ---------------------------------------------------------------------------
# WikiSearchTool
# ---------------------------------------------------------------------------

class WikiSearchTool(BaseTool):
    name: str = "wiki_search"
    description: str = (
        "Search wiki pages by keyword. Returns matching snippets with frontmatter metadata. "
        "Args: query (str) — search keywords; "
        "section (str, optional) — limit search to one section; "
        "max_results (int, default 10) — max pages to return."
    )

    def _run(self, query: str, section: str = "", max_results: int = 10) -> str:
        if not query or not isinstance(query, str):
            return "Error: query is required."

        query = query.strip()
        if not query:
            return "Error: query must contain at least one keyword."

        section = (section or "").strip().lower()
        if section and section not in VALID_SECTIONS:
            return f"Error: section must be one of {sorted(VALID_SECTIONS)} or empty for all."

        # Phase 2: Try ChromaDB semantic search first (hybrid with BM25)
        semantic_results = self._semantic_search(query, section, max_results)
        if semantic_results:
            return semantic_results

        # Fallback: grep-based keyword search
        return self._grep_search(query, section, max_results)

    def _semantic_search(self, query: str, section: str, max_results: int) -> str | None:
        """ChromaDB semantic + BM25 hybrid search. Returns None if unavailable."""
        try:
            from app.memory.chromadb_manager import retrieve_with_metadata
            where = {"section": section} if section else None
            raw = retrieve_with_metadata(_WIKI_COLLECTION, query, n=max_results * 2)
            if not raw:
                return None

            # Filter by section if needed
            if section:
                raw = [r for r in raw if r.get("metadata", {}).get("section") == section]

            # Hybrid scoring: 0.6 × semantic + 0.4 × BM25
            scored = []
            for r in raw:
                semantic_score = max(0, 1.0 - r.get("distance", 1.0))
                doc_text = r.get("document", "")
                bm25 = _bm25_score(query, doc_text)
                hybrid = 0.6 * semantic_score + 0.4 * min(1.0, bm25 / 5.0)
                meta = r.get("metadata", {})
                scored.append((hybrid, meta, doc_text[:200]))

            scored.sort(key=lambda x: x[0], reverse=True)
            scored = scored[:max_results]

            if not scored:
                return None

            results = [f"Found {len(scored)} page(s) matching '{query}' (semantic+BM25 hybrid):\n"]
            for score, meta, snippet in scored:
                page_ref = meta.get("doc_id", "?")
                results.append(
                    f"### {page_ref} (relevance: {score:.2f})\n"
                    f"- **Title**: {meta.get('title', '?')}\n"
                    f"- **Confidence**: {meta.get('confidence', '?')}\n"
                    f"- **Snippet**: {snippet}\n"
                )
            return "\n".join(results)
        except Exception:
            return None  # Fall back to grep

    def _grep_search(self, query: str, section: str, max_results: int) -> str:
        """Grep-based keyword search (Phase 1 fallback)."""
        keywords = query.lower().split()
        search_dirs = [section] if section else sorted(VALID_SECTIONS)
        results = []

        for sec in search_dirs:
            sec_dir = os.path.join(WIKI_ROOT, sec)
            if not os.path.isdir(sec_dir):
                continue
            for fname in sorted(os.listdir(sec_dir)):
                if fname == "index.md" or not fname.endswith(".md"):
                    continue
                fpath = os.path.join(sec_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue

                if not all(kw in content.lower() for kw in keywords):
                    continue

                fm, body = _parse_frontmatter(content)
                slug = fname[:-3]
                snippet = ""
                for line in body.split("\n"):
                    if any(kw in line.lower() for kw in keywords):
                        snippet = line.strip()[:200]
                        break

                results.append(
                    f"### {sec}/{slug}\n"
                    f"- **Title**: {fm.get('title', slug)}\n"
                    f"- **Confidence**: {fm.get('confidence', '?')}\n"
                    f"- **Snippet**: {snippet}\n"
                )
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            return f"No wiki pages matched query: {query}"
        return f"Found {len(results)} page(s) matching '{query}' (keyword):\n\n" + "\n".join(results)


# ---------------------------------------------------------------------------
# WikiLintTool
# ---------------------------------------------------------------------------

class WikiLintTool(BaseTool):
    name: str = "wiki_lint"
    description: str = (
        "Run health checks on the wiki. Returns a structured report of issues. "
        "8 checks: frontmatter completeness, orphan pages, dead wikilinks, "
        "contradiction hints, staleness, index consistency, bidirectional links, "
        "epistemic boundaries. "
        "Args: section (str, optional) — limit to one section or check all."
    )

    def _run(self, section: str = "") -> str:
        section = (section or "").strip().lower()
        if section and section not in VALID_SECTIONS:
            return f"Error: section must be one of {sorted(VALID_SECTIONS)} or empty for all."

        sections_to_check = [section] if section else sorted(VALID_SECTIONS)
        issues = {
            "frontmatter": [],
            "orphans": [],
            "dead_links": [],
            "contradictions": [],
            "stale": [],
            "index_consistency": [],
            "bidirectional": [],
            "epistemic": [],
        }

        # Collect all pages and their metadata
        all_pages = {}  # "section/slug" -> {fm, body, outgoing_links}
        for sec in sorted(VALID_SECTIONS):
            sec_dir = os.path.join(WIKI_ROOT, sec)
            if not os.path.isdir(sec_dir):
                continue
            for fname in sorted(os.listdir(sec_dir)):
                if fname == "index.md" or not fname.endswith(".md"):
                    continue
                fpath = os.path.join(sec_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue
                fm, body = _parse_frontmatter(content)
                slug = fname[:-3]
                page_ref = f"{sec}/{slug}"

                # Extract wikilinks [[section/slug]]
                links = re.findall(r"\[\[([a-z0-9/_-]+)\]\]", body)

                all_pages[page_ref] = {
                    "fm": fm,
                    "body": body,
                    "links": links,
                    "section": sec,
                }

        # Filter to requested sections
        pages_to_check = {
            k: v for k, v in all_pages.items()
            if v["section"] in sections_to_check
        }

        now = datetime.now(timezone.utc)

        for page_ref, data in pages_to_check.items():
            fm = data["fm"]

            # 1. Frontmatter completeness
            missing = REQUIRED_FRONTMATTER - set(fm.keys())
            if missing:
                issues["frontmatter"].append(
                    f"{page_ref}: missing fields — {', '.join(sorted(missing))}"
                )

            # 2. Dead links
            for link in data["links"]:
                if link not in all_pages:
                    issues["dead_links"].append(
                        f"{page_ref} -> [[{link}]] (target not found)"
                    )

            # 5. Staleness (90+ days)
            updated = fm.get("updated_at", "")
            if updated:
                try:
                    updated_dt = datetime.fromisoformat(
                        updated.replace("Z", "+00:00")
                    )
                    if (now - updated_dt) > timedelta(days=90):
                        days = (now - updated_dt).days
                        issues["stale"].append(
                            f"{page_ref}: last updated {days} days ago"
                        )
                except (ValueError, TypeError):
                    pass

            # 7. Bidirectional links
            for link in data["links"]:
                if link in all_pages:
                    back_links = all_pages[link]["links"]
                    if page_ref not in back_links:
                        issues["bidirectional"].append(
                            f"{page_ref} -> [[{link}]] but {link} does not link back"
                        )

            # 8. Epistemic boundaries
            confidence = fm.get("confidence", "")
            source = fm.get("source", "")
            if confidence in ("high", "verified") and source == "synthesis":
                issues["epistemic"].append(
                    f"{page_ref}: confidence={confidence} but source='synthesis' "
                    f"(needs concrete reference)"
                )

        # 3. Orphan detection — pages not referenced from any index or other page
        all_linked = set()
        for data in all_pages.values():
            all_linked.update(data["links"])
        # Also check section index references
        for sec in sorted(VALID_SECTIONS):
            idx_path = os.path.join(WIKI_ROOT, sec, "index.md")
            if os.path.isfile(idx_path):
                with open(idx_path, "r") as f:
                    idx_content = f.read()
                idx_links = re.findall(r"\[\[([a-z0-9/_-]+)\]\]", idx_content)
                all_linked.update(idx_links)

        for page_ref in pages_to_check:
            if page_ref not in all_linked:
                issues["orphans"].append(f"{page_ref}: not linked from anywhere")

        # 4. Contradiction detection (simple: same section, overlapping tags, different claims)
        # Group by section + tag overlap
        tag_groups = {}
        for page_ref, data in pages_to_check.items():
            fm = data["fm"]
            section = data["section"]
            for tag in fm.get("tags", []):
                key = f"{section}:{tag}"
                tag_groups.setdefault(key, []).append(page_ref)

        # Also check explicit typed relationships for contradictions
        for page_ref, data in pages_to_check.items():
            fm = data["fm"]
            for rel in fm.get("relationships", []):
                if isinstance(rel, dict) and rel.get("type") == "contradicts":
                    target = rel.get("target", "")
                    issues["contradictions"].append(
                        f"EXPLICIT: {page_ref} contradicts {target}"
                    )

        for key, refs in tag_groups.items():
            if len(refs) > 1:
                # Resolution heuristic: score by recency × confidence
                confidence_map = {"verified": 4, "high": 3, "medium": 2, "low": 1}
                scored = []
                for ref in refs:
                    fm = pages_to_check[ref]["fm"]
                    conf_score = confidence_map.get(fm.get("confidence", "medium"), 2)
                    # Recency: newer = higher score
                    try:
                        updated = fm.get("updated_at", "2020-01-01")
                        dt = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                        days_old = (datetime.now(timezone.utc) - dt).days
                        recency = max(0.1, 1.0 - days_old / 365)
                    except (ValueError, TypeError):
                        recency = 0.5
                    scored.append((ref, conf_score * recency, conf_score, recency))
                scored.sort(key=lambda x: x[1], reverse=True)
                best = scored[0]
                recommendation = f" RECOMMENDATION: {best[0]} is most authoritative (score={best[1]:.1f})"
                issues["contradictions"].append(
                    f"Overlap '{key}': {', '.join(refs)}.{recommendation}"
                )

        # 6. Index consistency
        for sec in sections_to_check:
            sec_dir = os.path.join(WIKI_ROOT, sec)
            if not os.path.isdir(sec_dir):
                continue
            actual_count = sum(
                1 for f in os.listdir(sec_dir)
                if f.endswith(".md") and f != "index.md"
            )
            idx_path = os.path.join(sec_dir, "index.md")
            if os.path.isfile(idx_path):
                with open(idx_path, "r") as f:
                    idx_fm, _ = _parse_frontmatter(f.read())
                declared = idx_fm.get("page_count", 0)
                if declared != actual_count:
                    issues["index_consistency"].append(
                        f"{sec}/index.md declares page_count={declared} "
                        f"but found {actual_count} pages"
                    )

        # Format report
        total_issues = sum(len(v) for v in issues.values())
        if total_issues == 0:
            scope = section if section else "all sections"
            return f"Wiki lint passed — no issues found ({scope})."

        report_lines = [f"Wiki Lint Report — {total_issues} issue(s) found:\n"]

        check_labels = {
            "frontmatter": "Frontmatter Completeness",
            "orphans": "Orphan Pages",
            "dead_links": "Dead Wikilinks",
            "contradictions": "Potential Contradictions",
            "stale": "Stale Pages (90+ days)",
            "index_consistency": "Index Consistency",
            "bidirectional": "Bidirectional Links",
            "epistemic": "Epistemic Boundaries",
        }

        for key, label in check_labels.items():
            items = issues[key]
            if items:
                report_lines.append(f"\n### {label} ({len(items)})")
                for item in items:
                    report_lines.append(f"  - {item}")

        return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# WikiSlidesTool — Generate Marp-compatible presentations from wiki content
# ---------------------------------------------------------------------------

class WikiSlidesTool(BaseTool):
    name: str = "wiki_slides"
    description: str = (
        "Generate a Marp-compatible slide deck from a wiki page. "
        "Args: page_path (str) — wiki page path (e.g., 'archibal/competitive-landscape'); "
        "title (str, optional) — override slide deck title; "
        "max_slides (int, default 10) — maximum slides to generate."
    )

    def _run(self, page_path: str, title: str = "", max_slides: int = 10) -> str:
        """Convert a wiki page into Marp slide markdown."""
        page_path = page_path.strip()
        if not page_path.endswith(".md"):
            page_path += ".md"

        try:
            file_path = _safe_path(page_path)
        except ValueError as e:
            return f"Error: {e}"

        if not os.path.isfile(file_path):
            return f"Error: page not found — {page_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading {page_path}: {e}"

        fm, body = _parse_frontmatter(content)
        deck_title = title or fm.get("title", page_path)

        # Marp header
        slides = [
            "---",
            "marp: true",
            "theme: default",
            f"title: {deck_title}",
            f"author: AndrusAI",
            "paginate: true",
            "---",
            "",
            f"# {deck_title}",
            "",
            f"*{fm.get('confidence', 'medium')} confidence | {fm.get('source', 'wiki')}*",
            f"*Generated from wiki/{page_path}*",
            "",
        ]

        # Split body into slides by ## headers
        sections = re.split(r"\n## ", body)
        slide_count = 1  # Title slide counts as 1

        for sec in sections:
            if not sec.strip():
                continue
            if slide_count >= max_slides:
                break

            # Get section title and content
            lines = sec.strip().split("\n")
            sec_title = lines[0].strip().lstrip("#").strip()
            sec_body = "\n".join(lines[1:]).strip()

            if not sec_title or sec_title.lower() in ("change history",):
                continue  # Skip metadata sections

            slides.append("---")
            slides.append("")
            slides.append(f"## {sec_title}")
            slides.append("")

            # Truncate long sections to fit slides
            body_lines = sec_body.split("\n")
            for line in body_lines[:15]:  # Max 15 lines per slide
                slides.append(line)
            if len(body_lines) > 15:
                slides.append("")
                slides.append("*(continued...)*")
            slides.append("")
            slide_count += 1

        # Save to .slides directory
        os.makedirs(SLIDES_DIR, exist_ok=True)
        slug = os.path.basename(page_path).replace(".md", "")
        slides_path = os.path.join(SLIDES_DIR, f"{slug}.md")
        with open(slides_path, "w", encoding="utf-8") as f:
            f.write("\n".join(slides))

        return (
            f"Generated {slide_count} slides from {page_path}.\n"
            f"Saved to wiki/.slides/{slug}.md\n"
            f"Render with: marp wiki/.slides/{slug}.md --pdf"
        )
