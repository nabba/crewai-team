"""Migration plan — typed declaration of source → target.

PROGRAM §40 (2026-05-10) — Q3 Item 12.

The plan is operator-authored once at the start of a migration and
never modified mid-flight. The state machine in ``state.py`` advances
phases against this fixed plan. If the operator wants to change the
target after a phase has fired, they cancel the migration and start
a new plan.

A plan is persisted as ``workspace/embedding_migration/plan.json``
and is human-readable. Operators are expected to read it before
firing any phase.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _default_plan_file() -> Path:
    """Lazy-resolve so a non-default ``WORKSPACE_ROOT`` is honored even
    when `app.paths` is imported later than this module."""
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT) / "embedding_migration" / "plan.json"
    except Exception:
        return Path("/app/workspace/embedding_migration/plan.json")


# Kept for back-compat in callers that import the constant directly.
# Resolved lazily via _default_plan_file() inside save_plan / load_plan.
_PLAN_FILE = _default_plan_file()


# ── Plan-validation policy ────────────────────────────────────────────────
#
# Q3.1 (2026-05-11) — the framework's chromadb path is correct only when
# the source/target operate against the ``memory`` KB (because
# ``chromadb_manager.get_client()`` is hardcoded to that persist dir).
# Plans that target other KBs need KB-rooted clients, which the dual-
# write helper now provides — but until the path is exercised end-to-end
# we surface the restriction as a hard refusal at save time rather than
# silently shadowing into the wrong directory.
#
# Pgvector migration is declared but NOT wired (the agent-experiences /
# beliefs / workspace_items columns need their own dual-write path that
# isn't built yet). Plans that include pgvector targets are refused.
#
# Operators broadening the allowlist must add a path here AND verify the
# corresponding dual-write code is implemented.

SUPPORTED_CHROMADB_KBS: set[str] = {"memory"}
SUPPORTED_TARGET_KINDS: set[str] = {"chromadb"}


class UnsupportedMigrationTarget(ValueError):
    """Raised when a plan declares a target the framework can't yet
    safely execute. The message identifies the offending target so the
    operator can edit the plan."""


@dataclass
class EmbeddingModel:
    """Identifier for one embedding model — provider, name, dimension."""
    provider: str          # "ollama", "openai", "voyage", …
    name: str              # "nomic-embed-text", "mxbai-embed-large", …
    dim: int               # 768, 1024, 1536, …
    base_url: str | None = None  # for self-hosted endpoints

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def display(self) -> str:
        return f"{self.provider}/{self.name} ({self.dim}-dim)"


@dataclass
class MigrationTarget:
    """One ChromaDB KB or pgvector column under migration."""
    kind: str              # "chromadb" or "pgvector"
    kb: str | None = None  # workspace/<kb>/ for chromadb
    collection: str | None = None  # chromadb collection name
    table: str | None = None       # pgvector table name
    column: str | None = None      # pgvector embedding column
    expected_rows: int | None = None  # filled at backfill start

    def display(self) -> str:
        if self.kind == "chromadb":
            return f"chromadb:{self.kb}/{self.collection}"
        return f"pgvector:{self.table}.{self.column}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationPlan:
    """Source → Target migration plan."""
    plan_id: str                   # operator-chosen, e.g. "ollama-nomic-to-mxbai-2026-Q3"
    source: EmbeddingModel
    target: EmbeddingModel
    targets: list[MigrationTarget]
    cutover_threshold_ndcg: float = 0.95   # cutover gate
    cutover_min_shadow_queries: int = 1_000  # minimum sample size
    standdown_retention_days: int = 30      # keep source collections this long after cutover
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""

    def display_summary(self) -> str:
        lines = [
            f"Plan {self.plan_id}",
            f"  source: {self.source.display()}",
            f"  target: {self.target.display()}",
            f"  cutover @ NDCG≥{self.cutover_threshold_ndcg} "
            f"after ≥{self.cutover_min_shadow_queries} shadow queries",
            f"  stand-down: {self.standdown_retention_days} days",
            f"  targets:",
            *[f"    - {t.display()}" for t in self.targets],
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "targets": [t.to_dict() for t in self.targets],
            "cutover_threshold_ndcg": self.cutover_threshold_ndcg,
            "cutover_min_shadow_queries": self.cutover_min_shadow_queries,
            "standdown_retention_days": self.standdown_retention_days,
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MigrationPlan":
        return cls(
            plan_id=str(d["plan_id"]),
            source=EmbeddingModel(**d["source"]),
            target=EmbeddingModel(**d["target"]),
            targets=[MigrationTarget(**t) for t in d.get("targets", [])],
            cutover_threshold_ndcg=float(d.get("cutover_threshold_ndcg", 0.95)),
            cutover_min_shadow_queries=int(
                d.get("cutover_min_shadow_queries", 1_000),
            ),
            standdown_retention_days=int(
                d.get("standdown_retention_days", 30),
            ),
            created_at=str(d.get("created_at") or ""),
            notes=str(d.get("notes") or ""),
        )


# ── Persistence ──────────────────────────────────────────────────────────


def validate_plan(plan: MigrationPlan) -> None:
    """Refuse plans whose targets exceed what the framework can safely
    execute today. Mutates nothing; raises ``UnsupportedMigrationTarget``
    on the first offending target."""
    for t in plan.targets:
        if t.kind not in SUPPORTED_TARGET_KINDS:
            raise UnsupportedMigrationTarget(
                f"target kind={t.kind!r} not yet wired. "
                f"Supported kinds: {sorted(SUPPORTED_TARGET_KINDS)}. "
                f"pgvector / Neo4j targets need their own dual-write path; "
                f"see docs/EMBEDDING_MIGRATION.md §Future work."
            )
        if t.kind == "chromadb":
            kb = (t.kb or "").strip()
            if not kb:
                raise UnsupportedMigrationTarget(
                    "chromadb target missing kb. Set kb to one of "
                    f"{sorted(SUPPORTED_CHROMADB_KBS)}."
                )
            if kb not in SUPPORTED_CHROMADB_KBS:
                raise UnsupportedMigrationTarget(
                    f"chromadb kb={kb!r} not yet on the migration "
                    f"allowlist. Currently supported: "
                    f"{sorted(SUPPORTED_CHROMADB_KBS)}. Broadening "
                    f"requires a Q3.x follow-up that verifies the KB-"
                    f"rooted client routing end-to-end."
                )
            if not (t.collection or "").strip():
                raise UnsupportedMigrationTarget(
                    f"chromadb target on kb={kb!r} missing collection name."
                )


def save_plan(plan: MigrationPlan, path: Path | None = None) -> Path:
    """Persist a plan. Validates first — refuses to save an
    unsupported plan rather than letting it silently shadow data into
    the wrong place at runtime."""
    validate_plan(plan)
    target = path or _default_plan_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(plan.to_dict(), indent=2), encoding="utf-8",
    )
    return target


def load_plan(path: Path | None = None) -> MigrationPlan | None:
    """Read the persisted plan. We validate on read too — if the on-
    disk plan is broken (operator hand-edited, framework allowlist
    tightened post-write), surface the error rather than silently
    serving a plan we can't execute."""
    target = path or _default_plan_file()
    if not target.exists():
        return None
    try:
        plan = MigrationPlan.from_dict(json.loads(target.read_text("utf-8")))
    except Exception:
        logger.exception("embedding_migration.plan: load failed")
        return None
    try:
        validate_plan(plan)
    except UnsupportedMigrationTarget as exc:
        logger.error(
            "embedding_migration.plan: persisted plan no longer valid: %s", exc,
        )
        return None
    return plan
