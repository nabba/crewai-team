"""
app.self_improvement.meta_agent.store — persistence for recipes + outcomes.

Two backends, mirroring the existing self_improvement/store.py pattern:

    Postgres  →  authoritative recipe registry + outcome ledger
                 (durable, replicated, queryable for aggregates)
    ChromaDB  →  similarity index over recipe.task_signature
                 (fast nearest-neighbour for selector dispatch)

Schema is created lazily on first use via ensure_schema(). Idempotent;
re-running is a no-op. Schema lives inline in this module so the
gateway image doesn't need migration files for it.

Connection pooling reuses app.control_plane.db (existing infrastructure
— no parallel pool). All errors are logged and never raise; meta-agent
must degrade gracefully to factory defaults if either backend is down.

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.self_improvement.meta_agent.types import AgentRecipe, RecipeOutcome

logger = logging.getLogger(__name__)


# ── Collection / table names ─────────────────────────────────────────────────

RECIPES_COLLECTION = "meta_agent_recipes"   # ChromaDB
RECIPES_TABLE = "meta_agent_recipes"        # Postgres (same name; different store)
OUTCOMES_TABLE = "meta_agent_outcomes"


# ── Schema (idempotent CREATE IF NOT EXISTS) ─────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta_agent_recipes (
    id                  TEXT PRIMARY KEY,
    crew_name           TEXT NOT NULL,
    force_tier          TEXT,
    extra_tool_names    JSONB NOT NULL DEFAULT '[]'::jsonb,
    task_hint           TEXT NOT NULL DEFAULT '',
    max_execution_time  INTEGER,
    task_signature      TEXT NOT NULL DEFAULT '',
    proposed_by         TEXT NOT NULL DEFAULT 'meta_agent',
    notes               TEXT NOT NULL DEFAULT '',
    uses                INTEGER NOT NULL DEFAULT 0,
    successes           INTEGER NOT NULL DEFAULT 0,
    last_used_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS meta_agent_recipes_crew_idx
    ON meta_agent_recipes (crew_name);

CREATE INDEX IF NOT EXISTS meta_agent_recipes_uses_idx
    ON meta_agent_recipes (crew_name, uses DESC);

CREATE TABLE IF NOT EXISTS meta_agent_outcomes (
    id                  TEXT PRIMARY KEY,
    recipe_id           TEXT NOT NULL,
    crew_name           TEXT NOT NULL,
    task_id             TEXT NOT NULL,
    success             BOOLEAN NOT NULL,
    confidence          TEXT NOT NULL DEFAULT '',
    duration_s          DOUBLE PRECISION NOT NULL DEFAULT 0,
    cost_estimate       DOUBLE PRECISION NOT NULL DEFAULT 0,
    error_signature     TEXT NOT NULL DEFAULT '',
    user_feedback       TEXT NOT NULL DEFAULT '',
    task_signature      TEXT NOT NULL DEFAULT '',
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS meta_agent_outcomes_recipe_idx
    ON meta_agent_outcomes (recipe_id);

CREATE INDEX IF NOT EXISTS meta_agent_outcomes_crew_recorded_idx
    ON meta_agent_outcomes (crew_name, recorded_at DESC);
"""


_schema_initialised = False


def ensure_schema() -> bool:
    """Create the meta-agent tables on first use. Idempotent.

    Returns True iff schema is now (or was already) ready. Never raises;
    a False return means callers should fall back to factory defaults.
    """
    global _schema_initialised
    if _schema_initialised:
        return True
    try:
        from app.control_plane.db import execute
        # control_plane.execute uses statement-by-statement semantics; split
        # the schema script and apply each block. CREATE INDEX IF NOT EXISTS
        # makes the whole thing idempotent.
        for stmt in [s.strip() for s in _SCHEMA_SQL.split(";")]:
            if stmt:
                execute(stmt + ";")
        _schema_initialised = True
        logger.info("meta_agent.store: schema ready")
        return True
    except Exception as exc:
        logger.warning(f"meta_agent.store: schema init failed: {exc}")
        return False


# ── Recipe registry (Postgres + ChromaDB) ────────────────────────────────────

def _recipe_id(crew_name: str, force_tier: Optional[str], tool_names: list[str],
               task_hint: str) -> str:
    """Deterministic ID from the augmentation knobs.

    Same recipe (same crew + knobs) always collides on the same id —
    repeated discovery upserts the row rather than appending duplicates.
    """
    payload = json.dumps(
        {
            "crew": crew_name,
            "tier": force_tier or "",
            "tools": sorted(tool_names),
            "hint": task_hint.strip(),
        },
        sort_keys=True,
    )
    h = hashlib.sha256(payload.encode()).hexdigest()[:14]
    return f"recipe_{crew_name}_{h}"


def null_recipe_for(crew_name: str) -> AgentRecipe:
    """The control-arm recipe for a crew (factory defaults, no augmentation).

    Always available to the selector. Its uses/successes counters are
    persisted like any other recipe so the bandit can compare null vs
    augmented over the same task signatures.
    """
    rid = _recipe_id(crew_name, None, [], "")
    rec = get_recipe(rid)
    if rec is not None:
        return rec
    # First call for this crew — create with uses=0 so the selector
    # picks it up from the start.
    rec = AgentRecipe(
        id=rid,
        crew_name=crew_name,
        proposed_by="seed",
        notes="control arm — no augmentation, factory defaults",
    )
    upsert_recipe(rec)
    return rec


def upsert_recipe(recipe: AgentRecipe) -> bool:
    """Persist or update a recipe in both Postgres and ChromaDB.

    Postgres is the authoritative store; the ChromaDB row is rebuilt
    from it. If recipe.id is empty, a deterministic id is computed
    from the recipe's knobs (so re-discovery upserts).

    Returns True on success.
    """
    if not ensure_schema():
        return False

    if not recipe.id:
        recipe.id = _recipe_id(
            recipe.crew_name,
            recipe.force_tier,
            recipe.extra_tool_names,
            recipe.task_hint,
        )

    pg_ok = _upsert_recipe_pg(recipe)
    chroma_ok = _upsert_recipe_chroma(recipe)
    return pg_ok or chroma_ok  # at least one channel persisted


def _upsert_recipe_pg(recipe: AgentRecipe) -> bool:
    try:
        from app.control_plane.db import execute
        execute(
            """
            INSERT INTO meta_agent_recipes
                (id, crew_name, force_tier, extra_tool_names, task_hint,
                 max_execution_time, task_signature, proposed_by, notes,
                 uses, successes, last_used_at, created_at)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s,
                    NULLIF(%s, '')::timestamptz, NULLIF(%s, '')::timestamptz)
            ON CONFLICT (id) DO UPDATE SET
                task_signature = EXCLUDED.task_signature,
                notes = EXCLUDED.notes,
                uses = EXCLUDED.uses,
                successes = EXCLUDED.successes,
                last_used_at = EXCLUDED.last_used_at,
                max_execution_time = EXCLUDED.max_execution_time
            """,
            (
                recipe.id,
                recipe.crew_name,
                recipe.force_tier,
                json.dumps(list(recipe.extra_tool_names)),
                recipe.task_hint,
                recipe.max_execution_time,
                recipe.task_signature,
                recipe.proposed_by,
                recipe.notes,
                int(recipe.uses),
                int(recipe.successes),
                recipe.last_used_at,
                recipe.created_at,
            ),
        )
        return True
    except Exception as exc:
        logger.debug(f"meta_agent: upsert_recipe pg failed: {exc}")
        return False


def _upsert_recipe_chroma(recipe: AgentRecipe) -> bool:
    """Mirror the recipe into the ChromaDB similarity index.

    The embedded text is recipe.task_signature (the kind of task this
    recipe was tried on) — that's what the selector matches against.
    Recipes without a task_signature are still indexed under their
    crew_name + notes so the null recipe is reachable.
    """
    try:
        from app.memory.chromadb_manager import get_client, embed
        client = get_client()
        col = client.get_or_create_collection(
            RECIPES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        embed_text = recipe.task_signature or f"{recipe.crew_name}: {recipe.notes}"
        meta = {
            "crew_name": recipe.crew_name,
            "force_tier": recipe.force_tier or "",
            "uses": int(recipe.uses),
            "successes": int(recipe.successes),
            "is_null": bool(recipe.is_null),
        }
        col.upsert(
            ids=[recipe.id],
            documents=[json.dumps(recipe.to_dict())],
            metadatas=[meta],
            embeddings=[embed(embed_text)],
        )
        return True
    except Exception as exc:
        logger.debug(f"meta_agent: upsert_recipe chroma failed: {exc}")
        return False


def get_recipe(recipe_id: str) -> Optional[AgentRecipe]:
    """Fetch a recipe by id from Postgres (authoritative)."""
    if not ensure_schema():
        return None
    try:
        from app.control_plane.db import execute_one
        row = execute_one(
            """
            SELECT id, crew_name, force_tier, extra_tool_names, task_hint,
                   max_execution_time, task_signature, proposed_by, notes,
                   uses, successes,
                   COALESCE(to_char(last_used_at AT TIME ZONE 'UTC',
                                     'YYYY-MM-DD"T"HH24:MI:SS+00:00'), '') AS last_used_at,
                   to_char(created_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS+00:00') AS created_at
            FROM meta_agent_recipes WHERE id = %s
            """,
            (recipe_id,),
        )
        if not row:
            return None
        # JSONB comes back as a list already; defensive parse otherwise
        tools = row["extra_tool_names"]
        if isinstance(tools, str):
            try:
                tools = json.loads(tools)
            except Exception:
                tools = []
        row["extra_tool_names"] = list(tools or [])
        return AgentRecipe.from_dict(row)
    except Exception as exc:
        logger.debug(f"meta_agent: get_recipe failed: {exc}")
        return None


def list_recipes(
    crew_name: Optional[str] = None,
    *,
    limit: int = 100,
    include_null: bool = True,
) -> list[AgentRecipe]:
    """List recipes, optionally filtered by crew. Sorted by uses DESC.

    Used by the /cp/ops dashboard and by amendment.py to compute the
    "blocked recipe" diagnostic.
    """
    if not ensure_schema():
        return []
    try:
        from app.control_plane.db import execute
        if crew_name:
            rows = execute(
                """
                SELECT id, crew_name, force_tier, extra_tool_names, task_hint,
                       max_execution_time, task_signature, proposed_by, notes,
                       uses, successes,
                       COALESCE(to_char(last_used_at AT TIME ZONE 'UTC',
                                         'YYYY-MM-DD"T"HH24:MI:SS+00:00'), '') AS last_used_at,
                       to_char(created_at AT TIME ZONE 'UTC',
                               'YYYY-MM-DD"T"HH24:MI:SS+00:00') AS created_at
                FROM meta_agent_recipes
                WHERE crew_name = %s
                ORDER BY uses DESC, created_at DESC
                LIMIT %s
                """,
                (crew_name, int(limit)),
                fetch=True,
            ) or []
        else:
            rows = execute(
                """
                SELECT id, crew_name, force_tier, extra_tool_names, task_hint,
                       max_execution_time, task_signature, proposed_by, notes,
                       uses, successes,
                       COALESCE(to_char(last_used_at AT TIME ZONE 'UTC',
                                         'YYYY-MM-DD"T"HH24:MI:SS+00:00'), '') AS last_used_at,
                       to_char(created_at AT TIME ZONE 'UTC',
                               'YYYY-MM-DD"T"HH24:MI:SS+00:00') AS created_at
                FROM meta_agent_recipes
                ORDER BY uses DESC, created_at DESC
                LIMIT %s
                """,
                (int(limit),),
                fetch=True,
            ) or []

        out: list[AgentRecipe] = []
        for row in rows:
            tools = row["extra_tool_names"]
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except Exception:
                    tools = []
            row["extra_tool_names"] = list(tools or [])
            try:
                rec = AgentRecipe.from_dict(row)
                if include_null or not rec.is_null:
                    out.append(rec)
            except Exception:
                continue
        return out
    except Exception as exc:
        logger.debug(f"meta_agent: list_recipes failed: {exc}")
        return []


# ── Outcome ledger ───────────────────────────────────────────────────────────

def record_outcome(outcome: RecipeOutcome) -> bool:
    """Append a recipe outcome and update the recipe's denormalised counters.

    Outcomes are append-only; the recipe row's uses/successes get bumped
    transactionally so the selector can read both consistently.

    Note: we use the bulk control_plane.execute helper, which auto-commits
    each statement. Two writes (outcome insert + counter bump) without a
    single transaction means a process crash between them can leave the
    counters one behind reality. That's tolerable: the selector smooths
    by (uses+2) and the next run's bump compensates. We don't introduce
    a new transactional path for this — that would duplicate
    control_plane.db's pool semantics.
    """
    if not ensure_schema():
        return False
    if not outcome.id:
        outcome.id = f"out_{uuid.uuid4().hex[:14]}"
    try:
        from app.control_plane.db import execute
        execute(
            """
            INSERT INTO meta_agent_outcomes
                (id, recipe_id, crew_name, task_id, success, confidence,
                 duration_s, cost_estimate, error_signature, user_feedback,
                 task_signature, recorded_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    NULLIF(%s, '')::timestamptz)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                outcome.id,
                outcome.recipe_id,
                outcome.crew_name,
                outcome.task_id,
                bool(outcome.success),
                outcome.confidence,
                float(outcome.duration_s),
                float(outcome.cost_estimate),
                outcome.error_signature,
                outcome.user_feedback,
                outcome.task_signature,
                outcome.recorded_at,
            ),
        )
        execute(
            """
            UPDATE meta_agent_recipes
            SET uses        = uses + 1,
                successes   = successes + %s,
                last_used_at = NOW()
            WHERE id = %s
            """,
            (1 if outcome.success else 0, outcome.recipe_id),
        )
        return True
    except Exception as exc:
        logger.debug(f"meta_agent: record_outcome failed: {exc}")
        return False


def list_outcomes(
    recipe_id: Optional[str] = None,
    *,
    crew_name: Optional[str] = None,
    since_days: Optional[int] = None,
    limit: int = 100,
) -> list[RecipeOutcome]:
    """Read recent outcomes. Used by metrics + the /cp/ops dashboard."""
    if not ensure_schema():
        return []
    clauses: list[str] = []
    params: list = []
    if recipe_id:
        clauses.append("recipe_id = %s")
        params.append(recipe_id)
    if crew_name:
        clauses.append("crew_name = %s")
        params.append(crew_name)
    if since_days is not None and since_days > 0:
        clauses.append("recorded_at > NOW() - (%s || ' days')::interval")
        params.append(int(since_days))
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))

    try:
        from app.control_plane.db import execute
        rows = execute(
            f"""
            SELECT id, recipe_id, crew_name, task_id, success, confidence,
                   duration_s, cost_estimate, error_signature, user_feedback,
                   task_signature,
                   to_char(recorded_at AT TIME ZONE 'UTC',
                           'YYYY-MM-DD"T"HH24:MI:SS+00:00') AS recorded_at
            FROM meta_agent_outcomes
            {where}
            ORDER BY recorded_at DESC
            LIMIT %s
            """,
            tuple(params),
            fetch=True,
        ) or []
        return [RecipeOutcome.from_dict(r) for r in rows]
    except Exception as exc:
        logger.debug(f"meta_agent: list_outcomes failed: {exc}")
        return []


# ── Similarity search (ChromaDB) ─────────────────────────────────────────────

def similarity_search(
    crew_name: str,
    task_text: str,
    *,
    n_results: int = 5,
) -> list[tuple[AgentRecipe, float]]:
    """Find recipes whose task_signature is similar to ``task_text``.

    Returns up to ``n_results`` tuples of (recipe, cosine_distance).
    Lower distance = more similar. Filtered to ``crew_name`` so a
    coding recipe never bleeds into a research dispatch.

    Falls back to [] (caller uses null recipe) if Chroma is down.
    """
    try:
        from app.memory.chromadb_manager import get_client, embed
        client = get_client()
        col = client.get_or_create_collection(
            RECIPES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        result = col.query(
            query_embeddings=[embed(task_text)],
            n_results=int(n_results),
            where={"crew_name": crew_name},
        )
    except Exception as exc:
        logger.debug(f"meta_agent: similarity_search query failed: {exc}")
        return []

    out: list[tuple[AgentRecipe, float]] = []
    docs = (result.get("documents") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]
    for doc, dist in zip(docs, dists):
        try:
            payload = json.loads(doc)
            tools = payload.get("extra_tool_names", [])
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except Exception:
                    tools = []
            payload["extra_tool_names"] = list(tools or [])
            recipe = AgentRecipe.from_dict(payload)
            out.append((recipe, float(dist)))
        except Exception:
            continue
    return out


# ── Pruning (called from idle_scheduler if desired) ──────────────────────────

def prune_dead_recipes(
    *,
    min_uses_for_keep: int = 5,
    max_age_days: int = 90,
) -> int:
    """Remove recipes that never accumulated evidence.

    A recipe with zero outcomes that's been sitting around for > 90
    days is noise — drop it from both stores. Recipes with at least
    ``min_uses_for_keep`` outcomes are kept regardless of age.
    Returns number of recipes pruned.
    """
    if not ensure_schema():
        return 0
    try:
        from app.control_plane.db import execute
        rows = execute(
            """
            SELECT id FROM meta_agent_recipes
            WHERE uses < %s
              AND created_at < NOW() - (%s || ' days')::interval
            """,
            (int(min_uses_for_keep), int(max_age_days)),
            fetch=True,
        ) or []
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        execute(
            "DELETE FROM meta_agent_recipes WHERE id = ANY(%s)",
            (ids,),
        )
        # Mirror the delete in Chroma; best-effort
        try:
            from app.memory.chromadb_manager import get_client
            client = get_client()
            col = client.get_or_create_collection(RECIPES_COLLECTION)
            col.delete(ids=ids)
        except Exception:
            pass
        return len(ids)
    except Exception as exc:
        logger.debug(f"meta_agent: prune_dead_recipes failed: {exc}")
        return 0
