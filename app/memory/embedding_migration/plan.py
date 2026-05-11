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


_PLAN_FILE = Path("/app/workspace/embedding_migration/plan.json")


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


def save_plan(plan: MigrationPlan, path: Path | None = None) -> Path:
    target = path or _PLAN_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(plan.to_dict(), indent=2), encoding="utf-8",
    )
    return target


def load_plan(path: Path | None = None) -> MigrationPlan | None:
    target = path or _PLAN_FILE
    if not target.exists():
        return None
    try:
        return MigrationPlan.from_dict(json.loads(target.read_text("utf-8")))
    except Exception:
        logger.exception("embedding_migration.plan: load failed")
        return None
