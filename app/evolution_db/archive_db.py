"""
archive_db.py — PostgreSQL-backed variant archive with UCB parent selection.

Stores every variant, run, and lineage relationship in the `evolution` schema.
Provides UCB1 bandit algorithm for intelligent parent selection during evolution.

Uses SQLAlchemy + psycopg2 for database access. Connection reuses the existing
mem0 PostgreSQL instance (separate `evolution` schema).
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    """Lazy-initialize the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        from app.config import get_settings
        s = get_settings()
        url = s.mem0_postgres_url
        if not url:
            raise RuntimeError("PostgreSQL URL not configured (MEM0_POSTGRES_PASSWORD missing)")
        _engine = create_engine(
            url,
            pool_size=3,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=600,
            poolclass=QueuePool,
        )
    return _engine


# ── Variant CRUD ─────────────────────────────────────────────────────────────

def add_variant(
    agent_name: str,
    target_type: str,
    generation: int,
    parent_id: Optional[str],
    source_code: str,
    file_path: str,
    modification_diff: str = "",
    modification_reasoning: str = "",
    scores: Optional[dict] = None,
    composite_score: float = 0.0,
    passed_threshold: bool = False,
    proposer_model: str = "",
    judge_model: str = "",
    eval_task_set: str = "",
    compute_cost_tokens: int = 0,
    execution_time_seconds: float = 0.0,
    metadata: Optional[dict] = None,
) -> str:
    """Insert a new variant into the archive. Returns UUID string."""
    engine = _get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO evolution.variants (
                agent_name, target_type, generation, parent_id,
                source_code, file_path, modification_diff, modification_reasoning,
                scores, composite_score, passed_threshold,
                proposer_model, judge_model, eval_task_set,
                compute_cost_tokens, execution_time_seconds, metadata
            ) VALUES (
                :agent_name, :target_type, :generation, :parent_id,
                :source_code, :file_path, :diff, :reasoning,
                :scores, :composite_score, :passed,
                :proposer, :judge, :eval_set,
                :tokens, :time, :meta
            ) RETURNING id
        """), {
            "agent_name": agent_name,
            "target_type": target_type,
            "generation": generation,
            "parent_id": parent_id,
            "source_code": source_code,
            "file_path": file_path,
            "diff": modification_diff,
            "reasoning": modification_reasoning,
            "scores": json.dumps(scores or {}),
            "composite_score": composite_score,
            "passed": passed_threshold,
            "proposer": proposer_model,
            "judge": judge_model,
            "eval_set": eval_task_set,
            "tokens": compute_cost_tokens,
            "time": execution_time_seconds,
            "meta": json.dumps(metadata or {}),
        })
        variant_id = str(result.scalar())

        # Update parent's child_count
        if parent_id:
            conn.execute(text("""
                UPDATE evolution.variants
                SET child_count = child_count + 1
                WHERE id = :pid
            """), {"pid": parent_id})

    logger.info(f"archive_db: stored variant {variant_id[:8]} (gen={generation}, score={composite_score:.4f})")
    return variant_id


def get_variant(variant_id: str) -> Optional[dict]:
    """Fetch a single variant by ID."""
    engine = _get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM evolution.variants WHERE id = :vid
        """), {"vid": variant_id}).mappings().first()
        if row:
            return dict(row)
    return None


def get_best_variants(agent_name: str, n: int = 10) -> list[dict]:
    """Get top N variants by composite score for an agent."""
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT * FROM evolution.variants
            WHERE agent_name = :name AND passed_threshold = TRUE
            ORDER BY composite_score DESC
            LIMIT :n
        """), {"name": agent_name, "n": n}).mappings().all()
        return [dict(r) for r in rows]


def get_lineage(variant_id: str) -> list[dict]:
    """Walk the parent chain from a variant back to root."""
    engine = _get_engine()
    lineage = []
    current_id = variant_id
    with engine.connect() as conn:
        for _ in range(100):  # Safety limit
            row = conn.execute(text("""
                SELECT id, parent_id, generation, composite_score,
                       modification_reasoning, agent_name
                FROM evolution.variants WHERE id = :vid
            """), {"vid": current_id}).mappings().first()
            if not row:
                break
            lineage.append(dict(row))
            if row["parent_id"] is None:
                break
            current_id = str(row["parent_id"])
    return lineage


def get_variant_count(agent_name: str = "") -> int:
    """Count total variants, optionally filtered by agent."""
    engine = _get_engine()
    with engine.connect() as conn:
        if agent_name:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM evolution.variants WHERE agent_name = :name
            """), {"name": agent_name})
        else:
            result = conn.execute(text("SELECT COUNT(*) FROM evolution.variants"))
        return result.scalar() or 0


# ── UCB1 Parent Selection ────────────────────────────────────────────────────

def sample_parent_ucb(
    agent_name: str,
    exploration_weight: float = 1.414,
) -> Optional[dict]:
    """Select a parent variant using UCB1 bandit algorithm.

    UCB1 score = composite_score + C * sqrt(ln(total_selections) / times_selected)

    This balances exploitation (high-scoring variants) with exploration
    (less-visited branches of the lineage tree).
    """
    engine = _get_engine()
    with engine.connect() as conn:
        # Get all passed variants for this agent
        rows = conn.execute(text("""
            SELECT id, composite_score, times_selected, child_count
            FROM evolution.variants
            WHERE agent_name = :name AND passed_threshold = TRUE
            ORDER BY composite_score DESC
        """), {"name": agent_name}).mappings().all()

        if not rows:
            return None

        variants = [dict(r) for r in rows]
        total_selections = sum(v["times_selected"] for v in variants) + 1

        # Compute UCB score for each variant
        best_ucb = -float("inf")
        best_variant = None

        for v in variants:
            selections = max(v["times_selected"], 1)
            exploit = v["composite_score"] or 0.0
            explore = exploration_weight * math.sqrt(
                math.log(total_selections) / selections
            )
            ucb_score = exploit + explore

            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_variant = v

        if best_variant:
            # Increment times_selected
            conn.execute(text("""
                UPDATE evolution.variants
                SET times_selected = times_selected + 1
                WHERE id = :vid
            """), {"vid": best_variant["id"]})
            conn.commit()

            # Fetch full variant
            return get_variant(str(best_variant["id"]))

    return None


# ── Run Management ───────────────────────────────────────────────────────────

def create_run(
    agent_name: str,
    target_type: str,
    max_generations: int,
    config: Optional[dict] = None,
) -> str:
    """Create a new evolution run record. Returns run UUID."""
    engine = _get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO evolution.runs (agent_name, target_type, max_generations, config)
            VALUES (:name, :target, :max_gen, :config)
            RETURNING id
        """), {
            "name": agent_name,
            "target": target_type,
            "max_gen": max_generations,
            "config": json.dumps(config or {}),
        })
        run_id = str(result.scalar())
    logger.info(f"archive_db: created run {run_id[:8]} ({agent_name}/{target_type}, {max_generations} gens)")
    return run_id


def update_run(
    run_id: str,
    generations_completed: Optional[int] = None,
    best_variant_id: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    """Update a run's progress."""
    engine = _get_engine()
    updates = []
    params = {"rid": run_id}

    if generations_completed is not None:
        updates.append("generations_completed = :gens")
        params["gens"] = generations_completed
    if best_variant_id is not None:
        updates.append("best_variant_id = :best")
        params["best"] = best_variant_id
    if status is not None:
        updates.append("status = :status")
        params["status"] = status
        if status in ("completed", "killed", "failed"):
            updates.append("completed_at = now()")

    if updates:
        with engine.begin() as conn:
            conn.execute(text(
                f"UPDATE evolution.runs SET {', '.join(updates)} WHERE id = :rid"
            ), params)


def add_lineage(parent_id: str, child_id: str, run_id: str) -> None:
    """Record a parent→child lineage edge."""
    engine = _get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO evolution.lineage (parent_id, child_id, run_id)
            VALUES (:pid, :cid, :rid)
            ON CONFLICT DO NOTHING
        """), {"pid": parent_id, "cid": child_id, "rid": run_id})


def get_run_status(run_id: str) -> Optional[dict]:
    """Get current status of an evolution run."""
    engine = _get_engine()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT * FROM evolution.runs WHERE id = :rid
        """), {"rid": run_id}).mappings().first()
        if row:
            return dict(row)
    return None


# ── Promotions ───────────────────────────────────────────────────────────────

def promote_variant(variant_id: str, notes: str = "") -> str:
    """Record a human-approved promotion. Returns promotion UUID."""
    engine = _get_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO evolution.promotions (variant_id, notes)
            VALUES (:vid, :notes)
            RETURNING id
        """), {"vid": variant_id, "notes": notes})
        promo_id = str(result.scalar())
    logger.info(f"archive_db: promoted variant {variant_id[:8]} (promo={promo_id[:8]})")
    return promo_id


# ── Statistics ───────────────────────────────────────────────────────────────

def get_evolution_stats(agent_name: str = "") -> dict:
    """Get summary statistics for the dashboard."""
    engine = _get_engine()
    with engine.connect() as conn:
        where = "WHERE agent_name = :name" if agent_name else ""
        params = {"name": agent_name} if agent_name else {}

        total = conn.execute(text(
            f"SELECT COUNT(*) FROM evolution.variants {where}"
        ), params).scalar() or 0

        passed = conn.execute(text(
            f"SELECT COUNT(*) FROM evolution.variants {where} {'AND' if where else 'WHERE'} passed_threshold = TRUE"
        ), params).scalar() or 0

        best_score = conn.execute(text(
            f"SELECT MAX(composite_score) FROM evolution.variants {where}"
        ), params).scalar()

        active_runs = conn.execute(text(
            "SELECT COUNT(*) FROM evolution.runs WHERE status = 'running'"
        )).scalar() or 0

        # Recent 5 variants
        recent = conn.execute(text("""
            SELECT id, agent_name, generation, composite_score,
                   passed_threshold, modification_reasoning, created_at
            FROM evolution.variants
            ORDER BY created_at DESC
            LIMIT 5
        """)).mappings().all()

        return {
            "total_variants": total,
            "passed_variants": passed,
            "best_score": round(best_score, 4) if best_score else 0.0,
            "active_runs": active_runs,
            "recent": [dict(r) for r in recent],
        }
