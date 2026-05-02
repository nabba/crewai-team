"""Workspace wiki + cross-layer memory registration.

Phase 9: a polished idea is registered in four memory layers at once,
so it can be found by content, by topic, by lineage, or by free-form
search depending on which surface the user is on:

  1. Workspace wiki  — ``workspace/companion/wiki/<workspace_id>/<id>-<slug>.md``
                       Per-workspace, owned by the Companion. Lineage
                       cross-links via Markdown relative paths so
                       static-site renderers thread them automatically.
                       An ``_index.md`` per workspace lists every page.
  2. Mem0            — one fact per polished idea, scoped to the
                       workspace via ``user_id="workspace:<id>"`` so
                       cross-session retrieval finds the insight by
                       content, not just by idea_id.
  3. System wiki     — ``workspace/wiki/meta/companion/<id>.md`` so the
                       existing wiki-synthesis / search machinery can
                       surface Companion output alongside human-authored
                       knowledge.
  4. ChromaDB        — already happens at idea creation (Phase 3,
                       ``companion_ideas`` collection); this module
                       does NOT touch it.

A ``WIKI_REGISTERED`` event is appended to the workspace event log so
``idea_store.current_state`` reflects the cross-layer commit. Failures
of any one layer are absorbed — the others still proceed and the
``errors`` field on the result records what went wrong.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.companion import document_pipeline as _dp
from app.companion import events as _events
from app.companion import idea_store as _idea_store

logger = logging.getLogger(__name__)

_WORKSPACE_WIKI_DIR = Path(os.environ.get(
    "COMPANION_WIKI_DIR", "workspace/companion/wiki"))
_SYSTEM_WIKI_DIR = Path(os.environ.get(
    "COMPANION_SYSTEM_WIKI_DIR", "wiki"))
_INDEX_FILENAME = "_index.md"


@dataclass
class WikiResult:
    """Outcome of one publish_to_wiki() call."""
    workspace_id: str
    idea_id: str
    wiki_page: str | None = None         # workspace wiki path
    system_wiki_page: str | None = None  # system wiki path
    mem0_id: str | None = None           # Mem0 record id when available
    errors: list[str] = field(default_factory=list)


def publish_to_wiki(workspace_id: str, idea_id: str) -> WikiResult:
    """Register a polished idea in workspace wiki + Mem0 + system wiki."""
    idea = _idea_store.find_by_id(workspace_id, idea_id)
    if idea is None:
        return WikiResult(workspace_id=workspace_id, idea_id=idea_id,
                           errors=["idea not found"])

    result = WikiResult(workspace_id=workspace_id, idea_id=idea_id)

    try:
        result.wiki_page = str(_write_workspace_wiki_page(workspace_id, idea))
    except Exception as exc:
        result.errors.append(f"workspace_wiki: {type(exc).__name__}")
        logger.debug("companion.wiki: workspace page failed: %s", exc)

    try:
        _refresh_workspace_index(workspace_id)
    except Exception as exc:
        # Index refresh is non-fatal — the page itself still landed.
        result.errors.append(f"workspace_index: {type(exc).__name__}")
        logger.debug("companion.wiki: index refresh failed: %s", exc)

    try:
        result.mem0_id = _register_in_mem0(workspace_id, idea)
    except Exception as exc:
        result.errors.append(f"mem0: {type(exc).__name__}")
        logger.debug("companion.wiki: mem0 register failed: %s", exc)

    try:
        result.system_wiki_page = str(
            _write_system_wiki_page(workspace_id, idea))
    except Exception as exc:
        result.errors.append(f"system_wiki: {type(exc).__name__}")
        logger.debug("companion.wiki: system wiki write failed: %s", exc)

    _emit_wiki_registered_event(workspace_id, idea_id, result)
    return result


# ── Workspace wiki page ────────────────────────────────────────────────────

def _write_workspace_wiki_page(workspace_id: str,
                                idea: _idea_store.IdeaRecord) -> Path:
    """Atomic write of the per-workspace wiki page. Returns its path."""
    body = _compose_workspace_wiki_markdown(workspace_id, idea)
    path = _wiki_page_path(workspace_id, idea)
    _atomic_write(path, body)
    return path


def _compose_workspace_wiki_markdown(workspace_id: str,
                                      idea: _idea_store.IdeaRecord) -> str:
    title = _dp.extract_title(idea.text)
    date = datetime.fromtimestamp(
        idea.created_at, tz=timezone.utc).strftime("%Y-%m-%d")
    parents_csv = ", ".join(idea.lineage_parents) if idea.lineage_parents else ""

    fm = [
        "---",
        f'title: "{_escape_quotes(title)}"',
        f"idea_id: {idea.idea_id}",
        f"workspace_id: {workspace_id}",
        f"cycle_id: {idea.cycle_id}",
        f"date: {date}",
        f"novelty: {idea.novelty:.3f}",
        f"quality: {idea.quality:.3f}",
        f"transferability: {idea.transferability:.3f}",
        f"panel_score: {idea.panel_score:.3f}",
        f'role: "{_escape_quotes(idea.role)}"',
        f"lineage_parents: [{parents_csv}]",
        f"layer: companion-workspace-wiki",
        "---",
        "",
    ]
    body = [
        f"# {title}",
        "",
        (idea.text or "*(no body)*").strip(),
        "",
    ]
    if idea.lineage_parents:
        body.extend(["## Lineage", ""])
        for pid in idea.lineage_parents:
            body.append(f"- [[{pid}]]({pid}.md)")
        body.append("")
    body.extend([
        "---",
        "",
        f"*Companion · workspace `{workspace_id}` · {date}*",
        "",
    ])
    return "\n".join(fm + body)


def _wiki_page_path(workspace_id: str,
                    idea: _idea_store.IdeaRecord) -> Path:
    safe_ws = _safe_path_token(workspace_id)
    slug = _slugify(_dp.extract_title(idea.text))
    safe_id = _safe_path_token(idea.idea_id)
    return _WORKSPACE_WIKI_DIR / safe_ws / f"{safe_id}-{slug}.md"


def _refresh_workspace_index(workspace_id: str) -> Path:
    """Regenerate the workspace's _index.md listing every page."""
    safe_ws = _safe_path_token(workspace_id)
    ws_dir = _WORKSPACE_WIKI_DIR / safe_ws
    ws_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(p for p in ws_dir.iterdir()
                   if p.is_file() and p.suffix == ".md"
                   and p.name != _INDEX_FILENAME)
    lines = [
        f"# Workspace wiki — `{workspace_id}`",
        "",
        f"*{len(pages)} pages · regenerated by Companion at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## Pages",
        "",
    ]
    for p in pages:
        # Strip "<id>-" prefix in display, keep relative link.
        display = p.stem.split("-", 1)[-1] or p.stem
        lines.append(f"- [{display}]({p.name})")
    lines.append("")
    index_path = ws_dir / _INDEX_FILENAME
    _atomic_write(index_path, "\n".join(lines))
    return index_path


# ── Mem0 cross-layer registration ──────────────────────────────────────────

def _register_in_mem0(workspace_id: str,
                       idea: _idea_store.IdeaRecord) -> str | None:
    """Best-effort Mem0 add. Returns the record id or None when unavailable."""
    fact = _compose_mem0_fact(workspace_id, idea)
    metadata = {
        "source": "companion",
        "workspace_id": workspace_id,
        "idea_id": idea.idea_id,
        "cycle_id": idea.cycle_id,
        "novelty": float(idea.novelty),
        "quality": float(idea.quality),
        "transferability": float(idea.transferability),
        "panel_score": float(idea.panel_score),
    }
    return _invoke_mem0_add(workspace_id, idea, fact, metadata)


def _compose_mem0_fact(workspace_id: str,
                        idea: _idea_store.IdeaRecord) -> str:
    title = _dp.extract_title(idea.text)
    body = (idea.text or "").strip()
    if len(body) > 1500:
        body = body[:1497] + "..."
    return (
        f"Workspace `{workspace_id}` polished a Companion idea "
        f'titled "{title}" (panel {idea.panel_score:.2f}, '
        f"novelty {idea.novelty:.2f}). The contemplation:\n\n{body}"
    )


def _invoke_mem0_add(workspace_id: str, idea: _idea_store.IdeaRecord,
                     fact: str, metadata: dict) -> str | None:
    """Indirection over the Mem0 add call.

    Tries to import a Mem0 manager and call ``add``. Mem0 client signatures
    vary across versions; we accept either a dict result with an ``id`` /
    nested ``results[0].id``, or an object with an ``id`` attribute. Any
    failure logs and returns None — the rest of the wiki publication still
    proceeds, and the workspace wiki / system wiki land regardless.
    """
    try:
        from app.memory.mem0_manager import get_mem0_manager
    except Exception:
        return None
    try:
        mem0 = get_mem0_manager()
    except Exception as exc:
        logger.debug("companion.wiki: get_mem0_manager failed: %s", exc)
        return None
    if mem0 is None:
        return None
    try:
        result = mem0.add(
            fact,
            user_id=f"workspace:{workspace_id}",
            metadata=metadata,
        )
    except TypeError:
        # Some Mem0 versions take ``messages=[...]`` instead of a string.
        try:
            result = mem0.add(
                messages=[{"role": "assistant", "content": fact}],
                user_id=f"workspace:{workspace_id}",
                metadata=metadata,
            )
        except Exception as exc:
            logger.debug("companion.wiki: mem0 add (messages) failed: %s", exc)
            return None
    except Exception as exc:
        logger.debug("companion.wiki: mem0 add failed: %s", exc)
        return None
    return _extract_mem0_id(result)


def _extract_mem0_id(result) -> str | None:
    if result is None:
        return None
    if isinstance(result, dict):
        if result.get("id"):
            return str(result["id"])
        results = result.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict) and first.get("id"):
                return str(first["id"])
    rid = getattr(result, "id", None)
    return str(rid) if rid else None


# ── System wiki registration ───────────────────────────────────────────────

def _write_system_wiki_page(workspace_id: str,
                             idea: _idea_store.IdeaRecord) -> Path:
    """Mirror the polished idea into the system wiki under meta/companion/."""
    body = _compose_system_wiki_markdown(workspace_id, idea)
    safe_id = _safe_path_token(idea.idea_id)
    path = _SYSTEM_WIKI_DIR / "meta" / "companion" / f"{safe_id}.md"
    _atomic_write(path, body)
    return path


def _compose_system_wiki_markdown(workspace_id: str,
                                   idea: _idea_store.IdeaRecord) -> str:
    title = _dp.extract_title(idea.text)
    date = datetime.fromtimestamp(
        idea.created_at, tz=timezone.utc).strftime("%Y-%m-%d")
    fm = [
        "---",
        f'title: "{_escape_quotes(title)}"',
        f"idea_id: {idea.idea_id}",
        f"workspace_id: {workspace_id}",
        f"date: {date}",
        f"panel_score: {idea.panel_score:.3f}",
        f"novelty: {idea.novelty:.3f}",
        "section: meta/companion",
        "epistemic_status: companion-polished",
        "source: companion",
        "---",
        "",
    ]
    body = [
        f"# {title}",
        "",
        f"*From Workspace Companion · workspace `{workspace_id}` · {date}*",
        "",
        (idea.text or "*(no body)*").strip(),
        "",
    ]
    return "\n".join(fm + body)


# ── Event emission ─────────────────────────────────────────────────────────

def _emit_wiki_registered_event(workspace_id: str, idea_id: str,
                                  result: WikiResult) -> None:
    try:
        _events.append(_events.Event(
            workspace_id=workspace_id,
            idea_id=idea_id,
            type=_events.EventType.WIKI_REGISTERED,
            ts=time.time(),
            payload={
                "wiki_page": result.wiki_page,
                "system_wiki_page": result.system_wiki_page,
                "mem0_id": result.mem0_id,
                "errors": list(result.errors),
            },
        ))
    except Exception as exc:
        logger.debug("companion.wiki: WIKI_REGISTERED event append failed: %s",
                     exc)


# ── Helpers ────────────────────────────────────────────────────────────────

def _safe_path_token(s: str) -> str:
    out = "".join(c for c in (s or "") if c.isalnum() or c in "-_")
    return out or "default"


def _slugify(text: str, max_len: int = 60) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return (s or "idea")[:max_len]


def _escape_quotes(s: str | None) -> str:
    return (s or "").replace('"', '\\"')


def _atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False,
        prefix=".tmp.", suffix=path.suffix,
    ) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    os.replace(tmp_path, path)
