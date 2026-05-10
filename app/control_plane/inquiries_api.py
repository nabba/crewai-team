"""Control plane — read-only access to weekly inquiry essays.

Operators (via React or curl) can:

  GET /api/cp/inquiries                     list essays, newest first
  GET /api/cp/inquiries/questions           the operator-curated question list
  GET /api/cp/inquiries/{slug}              read one essay by file stem

The inquiry pass writes essays to ``wiki/self/inquiries/<date>-<slug>.md``;
this router exposes them for the React UI without giving anyone a way
to edit them. The question list is similarly read-only here — additions
go through the change-request gate (``/cp/changes``).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from app.control_plane.auth_dep import require_gateway_auth
from app.subia.inquiry.questions import load_questions

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/inquiries",
    tags=["control-plane", "inquiries"],
    dependencies=[Depends(require_gateway_auth)],
)


_INQUIRIES_DIR = Path("/app/wiki/self/inquiries")
_QUESTIONS_FILE = Path("/app/wiki/self/inquiry_questions.md")
_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)\.md$")


def _list_essay_files() -> list[Path]:
    if not _INQUIRIES_DIR.exists():
        return []
    return sorted(_INQUIRIES_DIR.glob("*.md"), reverse=True)


def _parse_filename(name: str) -> tuple[str, str] | None:
    m = _FILENAME_RE.match(name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _summarise(path: Path) -> dict:
    parsed = _parse_filename(path.name)
    if parsed is None:
        return {
            "filename": path.name,
            "date": None,
            "slug": None,
            "question_text": "",
            "preview": "",
            "size_bytes": 0,
            "modified_at": "",
        }
    date, slug = parsed
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        text = ""
    question_text = _extract_question(text)
    preview = _extract_preview(text)
    try:
        stat = path.stat()
        size = stat.st_size
        from datetime import datetime, timezone
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        size = 0
        mtime = ""
    return {
        "filename": path.name,
        "date": date,
        "slug": slug,
        "question_text": question_text,
        "preview": preview,
        "size_bytes": size,
        "modified_at": mtime,
    }


def _extract_question(text: str) -> str:
    """Pull the question line from frontmatter or the H1 heading."""
    for line in text.splitlines()[:20]:
        if line.startswith("question:"):
            return line.split(":", 1)[1].strip()
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_preview(text: str, n: int = 280) -> str:
    """First non-frontmatter, non-heading paragraph."""
    in_fm = False
    out: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---":
            in_fm = not in_fm
            continue
        if in_fm:
            continue
        if line.startswith("#"):
            continue
        out.append(line)
        if sum(len(x) for x in out) > n:
            break
    return " ".join(out).strip()[:n]


@router.get("")
def list_inquiries(limit: int = Query(default=100, ge=1, le=500)) -> dict:
    files = _list_essay_files()[:limit]
    return {
        "count": len(files),
        "inquiries": [_summarise(f) for f in files],
    }


@router.get("/questions")
def list_questions() -> dict:
    """Return the operator-curated question list with each question's
    answer state (most recent answer date, or null)."""
    questions = load_questions(_QUESTIONS_FILE)
    files = _list_essay_files()
    answer_dates: dict[str, str] = {}
    for f in files:
        parsed = _parse_filename(f.name)
        if parsed is None:
            continue
        date, slug = parsed
        prev = answer_dates.get(slug)
        if prev is None or date > prev:
            answer_dates[slug] = date
    return {
        "count": len(questions),
        "questions": [
            {
                "slug": q.slug,
                "text": q.text,
                "framing": q.framing,
                "most_recent_answer_date": answer_dates.get(q.slug),
            }
            for q in questions
        ],
    }


@router.get("/{filename}")
def get_inquiry(filename: str) -> dict:
    # Path-traversal guard: refuse anything that contains `/` or `..`
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid filename")
    path = _INQUIRIES_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="inquiry not found")
    # Defence-in-depth: resolve and confirm under inquiries dir.
    resolved = path.resolve()
    canonical = _INQUIRIES_DIR.resolve()
    if canonical not in resolved.parents and resolved != canonical:
        raise HTTPException(status_code=400, detail="path escapes inquiries dir")
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"read failed: {exc}")
    summary = _summarise(resolved)
    return {**summary, "body": text}
