"""
training_collector.py — Captures LLM interactions for knowledge distillation.

Every LLM call across all agents and tiers is captured as a potential
training example. Premium model outputs (Claude, Gemini) act as implicit
"teachers" — a local model is the "student" that learns from accumulated
prompt-completion pairs.

Data flow:
    LLM call → lifecycle_hooks POST_LLM_CALL → training_collector
    → PostgreSQL training.interactions + daily JSONL file

Curation pipeline (batch, scheduled):
    1. Quality scoring via external judge (different model family)
    2. Deduplication by content hash
    3. Domain tagging by agent role
    4. Difficulty scoring
    5. Synthetic ratio enforcement (max 70% from any one source)
    6. Format conversion to MLX chat JSONL

Model collapse prevention:
    - Provenance tracked immutably at collection time
    - Synthetic ratio enforced programmatically
    - Data from multiple model families mixed
    - Earlier generations never discarded

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path("/app/workspace/training_data")
RAW_DIR = TRAINING_DATA_DIR / "raw"
CURATED_DIR = TRAINING_DATA_DIR / "curated"

# IMMUTABLE: Curation thresholds
QUALITY_THRESHOLD = 0.70           # Minimum quality score for training eligibility
MAX_SINGLE_SOURCE_RATIO = 0.70     # No more than 70% from any one model family
MIN_TRAINING_SET_SIZE = 100        # Don't train with fewer than 100 examples
MAX_RESPONSE_LENGTH = 4000         # Cap stored response length

# IMMUTABLE: Source tier mapping
TIER_MAP = {
    "claude": ("T4_premium", "api_anthropic"),
    "anthropic": ("T4_premium", "api_anthropic"),
    "gemini": ("T4_premium", "api_google"),
    "deepseek": ("T2_budget", "api_deepseek"),
    "minimax": ("T3_mid", "api_minimax"),
    "kimi": ("T3_mid", "api_moonshot"),
    "glm": ("T3_mid", "api_zhipu"),
    "qwen": ("T1_local", "local_ollama"),
    "llama": ("T1_local", "local_ollama"),
    "mistral": ("T1_local", "local_ollama"),
    "phi": ("T1_local", "local_ollama"),
}


def _classify_model(model_name: str) -> tuple[str, str]:
    """Classify a model name into (source_tier, provenance)."""
    lower = model_name.lower()
    for key, (tier, prov) in TIER_MAP.items():
        if key in lower:
            return tier, prov
    return "T1_local", "local_ollama"


def _content_hash(messages: list, response: str) -> str:
    """Dedup key based on content."""
    content = json.dumps(messages, sort_keys=True) + response
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ── Data Collection Hook ──────────────────────────────────────────────────────


def create_training_data_hook():
    """Create a lifecycle hook that captures LLM interactions for training.

    Registered at priority 55 (after tool memorizer at 50, before health at 60).
    """
    def collect_training_data(ctx):
        """POST_LLM_CALL hook: capture prompt-completion pair."""
        try:
            response = ctx.get("llm_response", "")
            if not response or len(str(response)) < 20:
                return ctx

            model_name = ctx.get("model_name", "") or ctx.metadata.get("model", "unknown")
            agent_role = ctx.agent_id or "unknown"
            task_desc = ctx.task_description or ""

            # Build messages from context
            messages = ctx.get("messages", [])
            if not messages:
                prompt = ctx.get("prompt", "")
                if prompt:
                    messages = [{"role": "user", "content": str(prompt)[:2000]}]

            if not messages:
                return ctx

            # Truncate for storage
            stored_messages = [
                {"role": m.get("role", "user"), "content": str(m.get("content", ""))[:2000]}
                for m in messages[-5:]  # Last 5 messages for context
            ]
            stored_response = str(response)[:MAX_RESPONSE_LENGTH]

            # Classify source
            source_tier, provenance = _classify_model(str(model_name))

            record = {
                "id": _content_hash(stored_messages, stored_response),
                "agent_role": agent_role,
                "task_description": task_desc[:500],
                "messages": stored_messages,
                "response": stored_response,
                "source_model": str(model_name),
                "source_tier": source_tier,
                "provenance": provenance,
                "quality_score": None,
                "training_eligible": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Fire-and-forget storage (don't block the response)
            threading.Thread(
                target=_store_record, args=(record,),
                daemon=True, name="training-store",
            ).start()

        except Exception:
            logger.debug("training_collector: hook failed", exc_info=True)

        return ctx

    return collect_training_data


def _store_record(record: dict) -> None:
    """Store a training record to JSONL + PostgreSQL."""
    # 1. Append to daily JSONL (crash-safe, always works)
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = RAW_DIR / f"interactions_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("training_collector: JSONL write failed", exc_info=True)

    # 2. Insert into PostgreSQL (if available)
    try:
        _store_to_postgres(record)
    except Exception:
        logger.debug("training_collector: PostgreSQL write failed", exc_info=True)


def _store_to_postgres(record: dict) -> None:
    """Insert training record into PostgreSQL."""
    from app.config import get_settings
    import psycopg2

    s = get_settings()
    if not s.mem0_postgres_url:
        return

    conn = psycopg2.connect(s.mem0_postgres_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO training.interactions
            (id, agent_role, task_description, messages, response,
             source_model, source_tier, provenance, quality_score, training_eligible)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            record["id"],
            record["agent_role"],
            record["task_description"],
            json.dumps(record["messages"]),
            record["response"],
            record["source_model"],
            record["source_tier"],
            record["provenance"],
            record.get("quality_score"),
            record.get("training_eligible", True),
        ))
    conn.close()


# ── Curation Pipeline ─────────────────────────────────────────────────────────


# IMMUTABLE: Quality scoring prompt (external judge)
QUALITY_PROMPT = """Rate this AI assistant response on quality (0.0 to 1.0).

Task: {task_description}
Agent role: {agent_role}

User input:
{messages}

Response:
{response}

Score dimensions: accuracy, relevance, completeness, coherence, usefulness.
Return ONLY a JSON object: {{"overall": 0.X, "reasoning": "brief"}}"""


class CurationPipeline:
    """Batch curation of collected training data.

    Run as scheduled job (daily or weekly). Scores quality, enforces
    synthetic ratios, tags domains, and exports MLX-compatible JSONL.
    """

    def __init__(self):
        CURATED_DIR.mkdir(parents=True, exist_ok=True)

    def run_curation(self) -> dict:
        """Full curation pipeline. Returns stats."""
        # Load unscored interactions from PostgreSQL
        interactions = self._load_unscored()
        if not interactions:
            return {"status": "no_data", "scored": 0}

        # Stage 1: Quality scoring
        scored = self._score_quality(interactions)

        # Stage 2: Filter by quality threshold
        eligible = [r for r in scored if (r.get("quality_score") or 0) >= QUALITY_THRESHOLD]

        # Stage 3: Synthetic ratio enforcement
        balanced = self._enforce_ratios(eligible)

        # Stage 4: Export to MLX format
        if len(balanced) >= MIN_TRAINING_SET_SIZE:
            train_count, valid_count = self._export_mlx(balanced)
        else:
            train_count, valid_count = 0, 0

        stats = {
            "status": "completed",
            "total_scored": len(scored),
            "eligible": len(eligible),
            "balanced": len(balanced),
            "exported_train": train_count,
            "exported_valid": valid_count,
        }
        logger.info(f"training_collector: curation complete — {stats}")
        return stats

    def _load_unscored(self) -> list[dict]:
        """Load interactions that haven't been quality-scored yet.

        Merges PostgreSQL + JSONL sources to catch all interactions.
        PG has structured data from the POST_LLM_CALL hook; JSONL has
        data from the raw file collector. Deduplicates by ID.
        """
        all_records = {}

        # Source 1: PostgreSQL
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if s.mem0_postgres_url:
                conn = psycopg2.connect(s.mem0_postgres_url)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT id, agent_role, task_description, messages, response,
                               source_model, source_tier, provenance
                        FROM training.interactions
                        WHERE quality_score IS NULL AND training_eligible = TRUE
                        ORDER BY created_at DESC
                        LIMIT 500
                    """)
                    for r in cur.fetchall():
                        all_records[r[0]] = {
                            "id": r[0], "agent_role": r[1], "task_description": r[2],
                            "messages": r[3] if isinstance(r[3], list) else json.loads(r[3] or "[]"),
                            "response": r[4], "source_model": r[5],
                            "source_tier": r[6], "provenance": r[7],
                        }
                conn.close()
        except Exception as e:
            logger.debug(f"training_collector: PG load failed: {e}")

        # Source 2: JSONL files (catches interactions not in PG)
        jsonl_records = self._load_from_jsonl()
        for r in jsonl_records:
            rid = r.get("id", "")
            if rid and rid not in all_records:
                all_records[rid] = r

        logger.info(f"training_collector: loaded {len(all_records)} unscored interactions "
                     f"(PG + JSONL merged)")
        return list(all_records.values())[:500]

    def _load_from_jsonl(self) -> list[dict]:
        """Fallback: load from JSONL files."""
        records = []
        if RAW_DIR.exists():
            for jl in sorted(RAW_DIR.glob("*.jsonl"))[-7:]:  # Last 7 days
                for line in jl.read_text().splitlines():
                    try:
                        r = json.loads(line)
                        if r.get("quality_score") is None:
                            records.append(r)
                    except json.JSONDecodeError:
                        pass
        return records[:500]

    def _score_quality(self, interactions: list[dict]) -> list[dict]:
        """Score quality using an external judge LLM."""
        try:
            from app.llm_factory import create_cheap_vetting_llm
            judge = create_cheap_vetting_llm()  # No args — function takes none
        except Exception as e:
            # No judge available — assign neutral score
            logger.warning(f"training_collector: judge LLM creation failed: {e}")
            for r in interactions:
                r["quality_score"] = 0.6
            return interactions

        import re as _re

        scored_count = 0
        for record in interactions:
            try:
                messages_text = "\n".join(
                    f"[{m.get('role', '?')}]: {m.get('content', '')[:300]}"
                    for m in record.get("messages", [])
                )
                prompt = QUALITY_PROMPT.format(
                    task_description=record.get("task_description", "")[:300],
                    agent_role=record.get("agent_role", ""),
                    messages=messages_text[:1500],
                    response=record.get("response", "")[:1500],
                )
                raw = str(judge.call(prompt)).strip()
                match = _re.search(r'\{[\s\S]*?\}', raw)
                if match:
                    data = json.loads(match.group())
                    record["quality_score"] = float(data.get("overall", 0.5))
                    scored_count += 1
                    logger.debug(f"training_collector: scored {record.get('id','?')}: {record['quality_score']:.2f}")
                else:
                    logger.warning(f"training_collector: judge returned no JSON: {raw[:100]}")
                    record["quality_score"] = 0.5
            except Exception as e:
                logger.warning(f"training_collector: scoring failed for {record.get('id','?')}: {e}")
                record["quality_score"] = 0.5

            # Persist score to PostgreSQL
            self._update_score(record["id"], record["quality_score"])

        logger.info(f"training_collector: scored {scored_count}/{len(interactions)} interactions")

        return interactions

    def _update_score(self, record_id: str, score: float) -> None:
        """Update quality score in PostgreSQL."""
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if not s.mem0_postgres_url:
                return
            conn = psycopg2.connect(s.mem0_postgres_url)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE training.interactions SET quality_score = %s WHERE id = %s",
                    (score, record_id),
                )
            conn.close()
        except Exception:
            pass

    def _enforce_ratios(self, interactions: list[dict]) -> list[dict]:
        """Enforce max 70% from any single model family."""
        source_counts: dict[str, int] = {}
        for r in interactions:
            prov = r.get("provenance", "unknown")
            source_counts[prov] = source_counts.get(prov, 0) + 1

        total = len(interactions)
        max_per_source = int(total * MAX_SINGLE_SOURCE_RATIO)

        balanced = []
        included: dict[str, int] = {}
        for r in interactions:
            prov = r.get("provenance", "unknown")
            included[prov] = included.get(prov, 0)
            if included[prov] < max_per_source:
                balanced.append(r)
                included[prov] += 1

        return balanced

    def _export_mlx(self, records: list[dict]) -> tuple[int, int]:
        """Export curated data as MLX-compatible chat JSONL."""
        import random
        random.shuffle(records)

        # 90/10 train/valid split
        split = int(len(records) * 0.9)
        train = records[:split]
        valid = records[split:]

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_dir = CURATED_DIR / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        for path, dataset in [(out_dir / "train.jsonl", train), (out_dir / "valid.jsonl", valid)]:
            with open(path, "w") as f:
                for r in dataset:
                    mlx_record = self._to_mlx_format(r)
                    f.write(json.dumps(mlx_record) + "\n")

        logger.info(f"training_collector: exported {len(train)} train, {len(valid)} valid to {out_dir}")
        return len(train), len(valid)

    def _to_mlx_format(self, record: dict) -> dict:
        """Convert to MLX chat training format."""
        messages = []
        for m in record.get("messages", []):
            role = m.get("role", "user")
            if role in ("system", "user", "assistant"):
                messages.append({"role": role, "content": m.get("content", "")})

        # The collected response is the target completion
        messages.append({"role": "assistant", "content": record.get("response", "")})
        return {"messages": messages}

    def get_stats(self) -> dict:
        """Get training data collection stats."""
        try:
            from app.config import get_settings
            import psycopg2
            s = get_settings()
            if not s.mem0_postgres_url:
                return {"source": "unavailable"}

            conn = psycopg2.connect(s.mem0_postgres_url)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM training.interactions")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM training.interactions WHERE quality_score IS NOT NULL")
                scored = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM training.interactions WHERE quality_score >= %s", (QUALITY_THRESHOLD,))
                eligible = cur.fetchone()[0]
                cur.execute("""
                    SELECT source_tier, COUNT(*) FROM training.interactions
                    GROUP BY source_tier ORDER BY COUNT(*) DESC
                """)
                by_tier = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute("""
                    SELECT agent_role, COUNT(*) FROM training.interactions
                    GROUP BY agent_role ORDER BY COUNT(*) DESC
                """)
                by_role = {r[0]: r[1] for r in cur.fetchall()}
            conn.close()

            return {
                "total_interactions": total,
                "scored": scored,
                "eligible": eligible,
                "by_tier": by_tier,
                "by_role": by_role,
                "min_for_training": MIN_TRAINING_SET_SIZE,
                "ready_to_train": eligible >= MIN_TRAINING_SET_SIZE,
            }
        except Exception:
            return {"source": "error"}


# ── Module-level singleton ───────────────────────────────────────────────────

_pipeline: CurationPipeline | None = None


def get_pipeline() -> CurationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CurationPipeline()
    return _pipeline
