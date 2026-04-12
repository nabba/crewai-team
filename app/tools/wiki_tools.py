"""
LLM Wiki Tools for CrewAI agents.

Four tools for managing the AndrusAI Knowledge Wiki:
  - WikiReadTool   — read wiki pages (full or frontmatter-only)
  - WikiWriteTool  — create / update / deprecate pages with locking
  - WikiSearchTool — grep-based keyword search across wiki
  - WikiLintTool   — 8 health checks for wiki integrity

WIKI_ROOT defaults to /app/wiki (Docker) with fallback to repo-relative wiki/.
"""

import os
import re
import glob
import time
import fcntl
import hashlib
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

LOCKS_DIR = os.path.join(WIKI_ROOT, ".locks")


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
        "deprecated_by (str, optional) — slug of replacement page (for deprecate)."
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

        file_path = _safe_path(os.path.join(section, slug + ".md"))
        page_ref = f"{section}/{slug}"
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Acquire lock
        lock_key = f"{section}_{slug}"
        lock_fh = _acquire_lock(lock_key)
        if lock_fh is None:
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
                    "author": author,
                    "status": "active",
                    "confidence": confidence,
                    "tags": tag_list,
                    "related": related_list,
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

                full_content = _render_frontmatter(fm) + "\n" + content.strip() + "\n"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_content)

                _append_log(author, "UPDATE", page_ref, f"Updated to v{version}")
                _rebuild_section_index(section)
                _rebuild_master_index()
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
        keywords = query.lower().split()
        if not keywords:
            return "Error: query must contain at least one keyword."

        section = (section or "").strip().lower()
        if section and section not in VALID_SECTIONS:
            return f"Error: section must be one of {sorted(VALID_SECTIONS)} or empty for all."

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

                content_lower = content.lower()
                if not all(kw in content_lower for kw in keywords):
                    continue

                fm, body = _parse_frontmatter(content)
                slug = fname[:-3]

                # Extract matching snippet (first occurrence context)
                snippet = ""
                for line in body.split("\n"):
                    if any(kw in line.lower() for kw in keywords):
                        snippet = line.strip()[:200]
                        break

                result_entry = (
                    f"### {sec}/{slug}\n"
                    f"- **Title**: {fm.get('title', slug)}\n"
                    f"- **Status**: {fm.get('status', '?')} | "
                    f"**Confidence**: {fm.get('confidence', '?')}\n"
                    f"- **Tags**: {', '.join(fm.get('tags', []))}\n"
                    f"- **Updated**: {fm.get('updated_at', '?')}\n"
                )
                if snippet:
                    result_entry += f"- **Snippet**: {snippet}\n"

                results.append(result_entry)

                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        if not results:
            return f"No wiki pages matched query: {query}"

        header = f"Found {len(results)} page(s) matching '{query}':\n\n"
        return header + "\n".join(results)


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

        for key, refs in tag_groups.items():
            if len(refs) > 1:
                issues["contradictions"].append(
                    f"Potential overlap on '{key}': {', '.join(refs)} — "
                    f"review for contradictions"
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
