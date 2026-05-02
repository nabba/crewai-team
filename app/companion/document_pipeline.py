"""Document maturation — polished ideas become downloadable documents.

A SURFACED idea that the user thumbs-ups (or one auto-promoted at high
panel_score, Phase 8.5+) is promoted via this pipeline into a
``workspace/companion/documents/<workspace_id>/<idea_id>.{md,docx,pdf}``
artifact tree. Markdown is the canonical format; docx and pdf are
derivations rendered via pandoc subprocess (best-effort — if pandoc is
unavailable the format is silently skipped, the md still lands).

On promotion, a ``DOCUMENTED`` event is appended to the workspace event
log so ``idea_store.current_state`` reflects the transition. Phase 9
attaches the same idea to the workspace wiki and registers it in the
other memory layers (Mem0, system wiki).

Source of truth remains the IdeaRecord — re-running ``promote`` simply
overwrites the artifact files (deterministic content from idea text +
lineage + scores).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.companion import events as _events
from app.companion import idea_store as _idea_store

logger = logging.getLogger(__name__)

_DOCUMENTS_DIR = Path(os.environ.get(
    "COMPANION_DOCUMENTS_DIR", "workspace/companion/documents"))

CANONICAL_FORMAT = "md"
ALLOWED_FORMATS = ("md", "docx", "pdf")
PANDOC_TIMEOUT_S = 30


@dataclass
class PromotionResult:
    """Outcome of one promote() call."""
    idea_id: str
    workspace_id: str
    formats: dict[str, str] = field(default_factory=dict)  # fmt → path
    error: str | None = None
    # Phase 9: cross-layer wiki/Mem0 publication outcome (when wiki=True)
    wiki_page: str | None = None
    system_wiki_page: str | None = None
    mem0_id: str | None = None
    wiki_errors: list[str] = field(default_factory=list)


def promote(workspace_id: str, idea_id: str, *,
            formats: tuple[str, ...] | list[str] = ("md",),
            publish_wiki: bool = True) -> PromotionResult:
    """Generate document files + emit DOCUMENTED event.

    ``formats`` may include "md", "docx", "pdf". md is always generated
    even when not requested (canonical). Failures of docx/pdf rendering
    are logged and skipped — the user still gets the markdown.

    When ``publish_wiki`` is True (default), the polished idea is also
    registered in the workspace wiki + Mem0 + system wiki via
    ``app.companion.wiki.publish_to_wiki``. Failures of any wiki layer
    are logged and listed in ``wiki_errors`` — the document files
    still land regardless.
    """
    idea = _idea_store.find_by_id(workspace_id, idea_id)
    if idea is None:
        return PromotionResult(idea_id=idea_id, workspace_id=workspace_id,
                                error="idea not found")

    body = build_markdown(workspace_id, idea)
    md_path = _write_canonical(workspace_id, idea_id, body)
    paths: dict[str, str] = {CANONICAL_FORMAT: str(md_path)}

    for fmt in formats:
        if fmt == CANONICAL_FORMAT:
            continue
        if fmt not in ALLOWED_FORMATS:
            logger.debug("companion.document_pipeline: skipping unknown "
                         "format %r", fmt)
            continue
        rendered = _render_format(md_path, fmt)
        if rendered:
            paths[fmt] = str(rendered)

    _emit_documented_event(workspace_id, idea_id, paths)

    result = PromotionResult(idea_id=idea_id, workspace_id=workspace_id,
                              formats=paths)

    if publish_wiki:
        try:
            from app.companion.wiki import publish_to_wiki
            wr = publish_to_wiki(workspace_id, idea_id)
            result.wiki_page = wr.wiki_page
            result.system_wiki_page = wr.system_wiki_page
            result.mem0_id = wr.mem0_id
            result.wiki_errors = list(wr.errors)
        except Exception as exc:
            logger.debug(
                "companion.document_pipeline: wiki publish raised: %s", exc)
            result.wiki_errors.append(f"publish_raised: {type(exc).__name__}")

    return result


def list_formats(workspace_id: str, idea_id: str) -> dict[str, str]:
    """Return {format: path} for every artifact already on disk."""
    out: dict[str, str] = {}
    for fmt in ALLOWED_FORMATS:
        p = _path_for(workspace_id, idea_id, fmt)
        if p.exists():
            out[fmt] = str(p)
    return out


def path_for(workspace_id: str, idea_id: str, fmt: str) -> Path | None:
    """Return the on-disk path for one format, or None if format is invalid."""
    if fmt not in ALLOWED_FORMATS:
        return None
    return _path_for(workspace_id, idea_id, fmt)


# ── Markdown composition ───────────────────────────────────────────────────

def build_markdown(workspace_id: str, idea: _idea_store.IdeaRecord) -> str:
    """Compose the canonical markdown body for one idea."""
    title = extract_title(idea.text)
    date = datetime.fromtimestamp(
        idea.created_at, tz=timezone.utc).strftime("%Y-%m-%d")

    parents_csv = ", ".join(idea.lineage_parents) if idea.lineage_parents else ""

    fm_lines = [
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
        "---",
        "",
    ]

    body_lines = [
        f"# {title}",
        "",
        (idea.text or "*(no body)*").strip(),
        "",
    ]

    if idea.lineage_parents:
        body_lines.extend([
            "## Lineage",
            "",
        ])
        for pid in idea.lineage_parents:
            body_lines.append(f"- [{pid}](./{pid}.md)")
        body_lines.append("")

    body_lines.extend([
        "---",
        "",
        f"*Generated by Workspace Companion · {date}*",
        "",
    ])
    return "\n".join(fm_lines + body_lines)


def extract_title(text: str | None) -> str:
    """Best-effort title from the first short line of the idea body."""
    if not text:
        return "(untitled)"
    first_line = text.strip().split("\n", 1)[0].lstrip("# ").strip()
    if 5 <= len(first_line) <= 100:
        return first_line
    head = (text.strip()[:80]).rstrip(".,;:!?").strip()
    return head or "(untitled)"


# ── Persistence ────────────────────────────────────────────────────────────

def _path_for(workspace_id: str, idea_id: str, fmt: str) -> Path:
    safe_ws = "".join(c for c in workspace_id if c.isalnum() or c in "-_") \
        or "default"
    safe_id = "".join(c for c in idea_id if c.isalnum() or c in "-_") or "doc"
    return _DOCUMENTS_DIR / safe_ws / f"{safe_id}.{fmt}"


def _write_canonical(workspace_id: str, idea_id: str, body: str) -> Path:
    """Atomic md write via temp + rename. Returns final path."""
    p = _path_for(workspace_id, idea_id, CANONICAL_FORMAT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=p.parent, delete=False,
        prefix=".tmp.", suffix=".md",
    ) as tmp:
        tmp.write(body)
        tmp_path = tmp.name
    os.replace(tmp_path, p)
    return p


def _render_format(md_path: Path, fmt: str) -> Path | None:
    """Best-effort pandoc render. Returns output path or None on failure."""
    out = md_path.with_suffix(f".{fmt}")
    pandoc = _pandoc_executable()
    if pandoc is None:
        logger.debug("companion.document_pipeline: pandoc unavailable for %s",
                     fmt)
        return None
    try:
        _invoke_pandoc(pandoc, str(md_path), str(out))
    except Exception as exc:
        logger.debug("companion.document_pipeline: pandoc %s failed: %s",
                     fmt, exc)
        return None
    return out if out.exists() else None


def _invoke_pandoc(pandoc: str, input_path: str, output_path: str) -> None:
    """Indirection over pandoc subprocess, for testability."""
    subprocess.run(
        [pandoc, input_path, "-o", output_path],
        check=True, capture_output=True, timeout=PANDOC_TIMEOUT_S,
    )


def _pandoc_executable() -> str | None:
    """Locate pandoc on PATH. Returns None if unavailable."""
    return shutil.which("pandoc")


def _emit_documented_event(workspace_id: str, idea_id: str,
                            formats: dict[str, str]) -> None:
    try:
        _events.append(_events.Event(
            workspace_id=workspace_id,
            idea_id=idea_id,
            type=_events.EventType.DOCUMENTED,
            ts=time.time(),
            payload={"formats": list(formats.keys())},
        ))
    except Exception as exc:
        logger.debug(
            "companion.document_pipeline: DOCUMENTED event append failed: %s",
            exc,
        )


def _escape_quotes(s: str | None) -> str:
    return (s or "").replace('"', '\\"')
