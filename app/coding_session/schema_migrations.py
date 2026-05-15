"""Schema-change detector + migration-file generator.

PROGRAM §45.2 — Q7.2. When a coding-session's diff touches files
that own the database schema, the submit step generates a numbered
SQL migration stub alongside the per-file CR fanout. The operator
reviews + approves the migration through the normal CR gate; the
migration runner (``app/memory/startup_migrations.py``) applies it
on next boot.

Detection is **PATH-ONLY** for v1 per operator decision: a curated
allow-list of files known to own schema. No content-regex pass
(false positives on .py code that mentions ``CREATE TABLE`` in
docstrings are not worth the noise). If real cases slip through,
add to ``_SCHEMA_OWNING_PATHS`` deliberately.

Generated migrations follow the existing convention in
``migrations/`` (numbered ``<NNNN>_<name>.sql``); no alembic
infrastructure is required.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# Curated allow-list. Each entry is a path-PREFIX match.
# Adding to this list is a deliberate decision — false positives
# manifest as extra (rejectable) migration CRs.
_SCHEMA_OWNING_PATHS: tuple[str, ...] = (
    "migrations/",                      # any change in migrations/ itself
    "app/control_plane/db.py",          # canonical schema registration
    "app/memory/postgres_schemas/",     # postgres schema files (per project convention)
    "app/control_plane/schemas/",       # control-plane table definitions
    "app/budgets/db.py",                # budget tables
    "app/affect/db.py",                 # affect-substrate tables
)


def _default_migrations_dir() -> Path:
    """Path to the canonical migrations directory."""
    try:
        from app.paths import REPO_ROOT
        return Path(REPO_ROOT) / "migrations"
    except Exception:
        # When REPO_ROOT isn't available (e.g. when running from inside
        # a worktree), fall back to two-levels up from this file. Works
        # in production where the package is installed at /app/.
        return Path(__file__).resolve().parent.parent.parent / "migrations"


@dataclass(frozen=True)
class SchemaChangeHint:
    """A single detected schema-shape change in the session diff."""

    detected_paths: tuple[str, ...]      # which schema-owning paths were touched
    inferred_name: str                    # snake-case migration name
    next_migration_number: int            # the NNNN prefix for the new file
    suggested_filename: str               # full filename (relative to migrations/)


def detect_schema_changes(
    changed_paths: Iterable[str],
    *,
    migrations_dir: Path | None = None,
) -> SchemaChangeHint | None:
    """Inspect the changed-paths list. Return a SchemaChangeHint when
    ANY changed path matches a schema-owning prefix, else None.

    Path-only detection per Q7.2 operator decision. False positives
    manifest as an extra migration CR the operator can reject.
    """
    matched: list[str] = []
    for path in changed_paths:
        if not path:
            continue
        # Normalize: strip leading './', '/' etc.
        norm = path.replace("\\", "/").lstrip("./").lstrip("/")
        for prefix in _SCHEMA_OWNING_PATHS:
            if norm == prefix or norm.startswith(prefix):
                matched.append(norm)
                break
    if not matched:
        return None

    migrations_dir = migrations_dir or _default_migrations_dir()
    next_n = _next_migration_number(migrations_dir)
    name = _infer_migration_name(matched)
    suggested = f"{next_n:04d}_{name}.sql"
    return SchemaChangeHint(
        detected_paths=tuple(matched),
        inferred_name=name,
        next_migration_number=next_n,
        suggested_filename=suggested,
    )


def _next_migration_number(migrations_dir: Path) -> int:
    """Return ``max(existing_NNNN) + 1``, or 1 when none exist."""
    if not migrations_dir.exists():
        return 1
    max_n = 0
    pattern = re.compile(r"^(\d+)[_-]")
    try:
        for entry in migrations_dir.iterdir():
            if not entry.is_file():
                continue
            m = pattern.match(entry.name)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            if n > max_n:
                max_n = n
    except OSError:
        return 1
    return max_n + 1


_SNAKE_RE = re.compile(r"[^a-z0-9]+")


def _infer_migration_name(matched_paths: list[str]) -> str:
    """Best-effort: turn the touched paths into a snake-case slug.

    Examples:
      ['app/control_plane/db.py'] → 'control_plane_db'
      ['app/affect/db.py', 'app/budgets/db.py'] → 'affect_budgets_db'
      ['migrations/0042_foo.sql'] → 'amend_migration_0042_foo'
    """
    if not matched_paths:
        return "schema_change"
    parts: list[str] = []
    for p in matched_paths[:3]:  # cap; long names get unwieldy
        if p.startswith("migrations/"):
            # Migration-file edit: prepend 'amend_'.
            stem = Path(p).stem
            parts.append(f"amend_migration_{stem}")
            continue
        # Strip 'app/' prefix and '.py' suffix.
        rel = p
        if rel.startswith("app/"):
            rel = rel[4:]
        if rel.endswith(".py"):
            rel = rel[:-3]
        # Replace separators with underscores.
        slug = _SNAKE_RE.sub("_", rel.lower()).strip("_")
        parts.append(slug)
    name = "_".join(parts)
    # Cap length.
    return name[:80] if len(name) > 80 else name


def render_migration_stub(
    hint: SchemaChangeHint,
    *,
    session_id: str,
    purpose: str,
    detected_at: str | None = None,
) -> str:
    """Render the SQL stub the operator will edit before approving."""
    detected_at = detected_at or datetime.now(timezone.utc).isoformat()
    paths_block = "\n".join(f"--   {p}" for p in hint.detected_paths)
    return f"""-- Migration {hint.next_migration_number:04d}: {hint.inferred_name}
--
-- Generated by PROGRAM §45.2 Q7.2 — schema-aware coding-session submit.
-- Detected at: {detected_at}
-- Originating coding session: {session_id}
-- Purpose: {purpose}
--
-- The session touched these schema-owning paths:
{paths_block}
--
-- TODO: review the diff and write the appropriate ALTER/CREATE/DROP
-- statements below. This file is a STUB — operator must complete
-- the SQL before approving the CR.
--
-- Convention: idempotent statements (IF NOT EXISTS / IF EXISTS) where
-- possible; transactions for related changes; comment each statement
-- with WHY (not just WHAT).

BEGIN;

-- TODO: SQL here.

COMMIT;
"""
