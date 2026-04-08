"""
llm_discovery.py — Automatic LLM model discovery, benchmarking, and promotion.

Full pipeline:
  1. SCAN: Query OpenRouter /models API for new/updated models
  2. FILTER: Keep models matching our tier/capability criteria
  3. BENCHMARK: Run eval_set_score against standardized test tasks
  4. COMPARE: Compare benchmark vs current model for the same role
  5. PROPOSE: If new model outperforms, create governance approval request
  6. PROMOTE: On approval, add to runtime catalog + assign to roles

Runs as an idle scheduler job. Discovered models stored in PostgreSQL
(control_plane.discovered_models). Promotions require governance approval
unless the model is free tier.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum thresholds for a model to be considered
MIN_CONTEXT_WINDOW = 8_000
MAX_COST_OUTPUT_PER_M = 20.0  # $20/M tokens max (excludes ultra-premium)

# Tier classification by cost
TIER_THRESHOLDS = {
    "free": 0.0,
    "budget": 1.0,      # ≤ $1/M output
    "mid": 5.0,          # ≤ $5/M output
    "premium": 20.0,     # ≤ $20/M output
}

# Provider-specific model ID prefixes for our catalog
PROVIDER_PREFIXES = {
    "openrouter": "openrouter/",
    "ollama": "ollama_chat/",
}

# Roles to benchmark new models against
BENCHMARK_ROLES = ["research", "coding", "writing"]


# ── OpenRouter Scanner ───────────────────────────────────────────────────────

def scan_openrouter() -> list[dict]:
    """Query OpenRouter /models API for all available models.

    Returns list of model dicts with standardized fields.
    """
    import httpx

    try:
        from app.config import get_settings
        s = get_settings()
        api_key = s.openrouter_api_key.get_secret_value()
    except Exception:
        api_key = os.getenv("OPENROUTER_API_KEY", "")

    if not api_key:
        logger.warning("llm_discovery: no OpenRouter API key")
        return []

    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning(f"llm_discovery: OpenRouter API returned {resp.status_code}")
            return []

        data = resp.json()
        models = data.get("data", [])
        logger.info(f"llm_discovery: OpenRouter returned {len(models)} models")
        return models

    except Exception as e:
        logger.warning(f"llm_discovery: OpenRouter scan failed: {e}")
        return []


def scan_ollama() -> list[dict]:
    """Query local Ollama for available models."""
    import httpx

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return [
                {
                    "id": f"ollama_chat/{m['name']}",
                    "name": m["name"],
                    "context_length": 32768,  # Default, Ollama doesn't always report this
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text"},
                    "provider": "ollama",
                }
                for m in models
            ]
    except Exception:
        pass
    return []


# ── Filter + Normalize ───────────────────────────────────────────────────────

def _normalize_model(raw: dict, provider: str = "openrouter") -> dict | None:
    """Normalize a raw API model to our standard format. Returns None if filtered out."""
    model_id = raw.get("id", "")
    name = raw.get("name", model_id)
    context = raw.get("context_length", 0) or 0

    # Extract pricing
    pricing = raw.get("pricing", {})
    cost_input = float(pricing.get("prompt", 0) or 0) * 1_000_000  # per-token → per-M
    cost_output = float(pricing.get("completion", 0) or 0) * 1_000_000

    # Filter
    if context < MIN_CONTEXT_WINDOW:
        return None
    if cost_output > MAX_COST_OUTPUT_PER_M:
        return None

    # Detect capabilities
    arch = raw.get("architecture", {})
    modality = arch.get("modality", "text")
    multimodal = "image" in modality or "multimodal" in modality

    # Classify tier
    tier = "premium"
    for tier_name, threshold in sorted(TIER_THRESHOLDS.items(), key=lambda x: x[1]):
        if cost_output <= threshold:
            tier = tier_name
            break

    # Build catalog-compatible model_id
    if provider == "openrouter":
        catalog_id = f"openrouter/{model_id}" if not model_id.startswith("openrouter/") else model_id
    elif provider == "ollama":
        catalog_id = model_id  # Already prefixed
    else:
        catalog_id = model_id

    return {
        "model_id": catalog_id,
        "provider": provider,
        "display_name": name,
        "context_window": context,
        "cost_input_per_m": round(cost_input, 6),
        "cost_output_per_m": round(cost_output, 6),
        "multimodal": multimodal,
        "tool_calling": True,  # Assume true for OpenRouter models
        "tier": tier,
        "raw_metadata": raw,
    }


# ── Database Operations ──────────────────────────────────────────────────────

def _get_known_model_ids() -> set[str]:
    """Get all model_ids already in discovered_models table."""
    from app.control_plane.db import execute
    rows = execute(
        "SELECT model_id FROM control_plane.discovered_models",
        fetch=True,
    )
    return {r["model_id"] for r in (rows or [])}


def _get_catalog_model_ids() -> set[str]:
    """Get all model_ids already in the static catalog."""
    from app.llm_catalog import CATALOG
    ids = set()
    for name, info in CATALOG.items():
        ids.add(info.get("model_id", ""))
        ids.add(name)
    return ids


def _store_discovered(model: dict) -> bool:
    """Store a newly discovered model in PostgreSQL."""
    from app.control_plane.db import execute
    try:
        execute(
            """INSERT INTO control_plane.discovered_models
               (model_id, provider, display_name, context_window,
                cost_input_per_m, cost_output_per_m, multimodal, tool_calling,
                source, raw_metadata, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'discovered')
               ON CONFLICT (model_id) DO UPDATE SET
                cost_input_per_m = EXCLUDED.cost_input_per_m,
                cost_output_per_m = EXCLUDED.cost_output_per_m,
                context_window = EXCLUDED.context_window,
                updated_at = NOW()""",
            (
                model["model_id"], model["provider"], model["display_name"],
                model["context_window"], model["cost_input_per_m"],
                model["cost_output_per_m"], model["multimodal"],
                model.get("tool_calling", True), "openrouter_api",
                json.dumps(model.get("raw_metadata", {})),
            ),
        )
        return True
    except Exception as e:
        logger.debug(f"llm_discovery: store failed: {e}")
        return False


def _update_benchmark(model_id: str, score: float, role: str) -> None:
    """Update benchmark score for a discovered model."""
    from app.control_plane.db import execute
    execute(
        """UPDATE control_plane.discovered_models
           SET benchmark_score = %s, benchmark_role = %s,
               benchmarked_at = NOW(), status = 'benchmarking',
               updated_at = NOW()
           WHERE model_id = %s""",
        (score, role, model_id),
    )


def _promote_model(model_id: str, tier: str, roles: list[str], reviewer: str = "system") -> None:
    """Mark a model as promoted."""
    from app.control_plane.db import execute
    execute(
        """UPDATE control_plane.discovered_models
           SET status = 'promoted', promoted_tier = %s,
               promoted_roles = %s, promoted_at = NOW(),
               reviewed_by = %s, updated_at = NOW()
           WHERE model_id = %s""",
        (tier, roles, reviewer, model_id),
    )


# ── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_model(model_id: str, role: str = "research", sample_size: int = 2) -> float:
    """Run standardized benchmark against a model.

    Uses eval_set tasks for the given role, scores with external judge.
    Returns 0.0-1.0 score, or -1.0 on failure.
    """
    try:
        from app.llm_factory import _cached_llm
        from app.config import get_settings

        s = get_settings()
        or_key = s.openrouter_api_key.get_secret_value()
        if not or_key:
            return -1.0

        # Create LLM for the candidate model
        if model_id.startswith("openrouter/"):
            candidate_llm = _cached_llm(
                model_id, max_tokens=1024,
                base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
        elif model_id.startswith("ollama_chat/"):
            candidate_llm = _cached_llm(model_id, max_tokens=1024)
        else:
            return -1.0

        # Test tasks per role
        test_tasks = {
            "research": [
                "What are the key differences between REST and GraphQL APIs?",
                "Explain the CAP theorem in 3 sentences.",
            ],
            "coding": [
                "Write a Python function to check if a number is prime.",
                "Implement a simple LRU cache class in Python.",
            ],
            "writing": [
                "Write a professional email declining a meeting invitation.",
                "Write release notes for a software version adding dark mode.",
            ],
        }

        tasks = test_tasks.get(role, test_tasks["research"])[:sample_size]

        # Create judge (different model — DGM compliant)
        from app.llm_factory import create_cheap_vetting_llm
        judge = create_cheap_vetting_llm()

        import re
        scores = []
        for task in tasks:
            try:
                response = str(candidate_llm.call(task)).strip()
                if not response or len(response) < 20:
                    scores.append(0.2)
                    continue

                judge_prompt = (
                    f"Score this AI response 0.0-1.0 on accuracy, completeness, clarity.\n"
                    f"Task: {task}\nResponse: {response[:2000]}\n\n"
                    f'Reply ONLY: {{"score": 0.X}}'
                )
                raw = str(judge.call(judge_prompt)).strip()
                match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
                if match:
                    scores.append(min(1.0, max(0.0, float(match.group(1)))))
                else:
                    scores.append(0.5)
            except Exception:
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"llm_discovery: benchmark {model_id} on {role}: {avg:.3f}")
        return avg

    except Exception as e:
        logger.warning(f"llm_discovery: benchmark failed for {model_id}: {e}")
        return -1.0


# ── Comparison + Promotion ───────────────────────────────────────────────────

def _get_current_model_score(role: str) -> tuple[str, float]:
    """Get the current model assigned to a role and its last benchmark score."""
    from app.llm_catalog import ROLE_DEFAULTS
    defaults = ROLE_DEFAULTS.get("balanced", {})
    current_model = defaults.get(role, defaults.get("default", "deepseek-v3.2"))

    # Get benchmark score from DB or use a default
    from app.control_plane.db import execute_scalar
    score = execute_scalar(
        """SELECT benchmark_score FROM control_plane.discovered_models
           WHERE model_id LIKE %s AND benchmark_role = %s
           ORDER BY benchmarked_at DESC LIMIT 1""",
        (f"%{current_model}%", role),
    )
    return current_model, float(score) if score else 0.7  # Default baseline


def propose_promotion(model: dict, benchmark_score: float, role: str) -> dict | None:
    """Create a governance request to promote a discovered model.

    Free models auto-promote. Others need human approval.
    """
    tier = model.get("tier", "budget")
    model_id = model["model_id"]

    current_model, current_score = _get_current_model_score(role)

    # Only propose if significantly better (5%+ improvement)
    if benchmark_score <= current_score * 1.05:
        logger.info(f"llm_discovery: {model_id} ({benchmark_score:.3f}) doesn't beat "
                     f"{current_model} ({current_score:.3f}) for {role}")
        return None

    # Free models auto-promote (no cost risk)
    if tier == "free":
        _promote_model(model_id, tier, [role], reviewer="auto")
        _add_to_runtime_catalog(model, [role])
        logger.info(f"llm_discovery: auto-promoted free model {model_id} for {role}")
        return {"status": "auto_promoted", "model": model_id, "role": role}

    # Others need governance approval
    try:
        from app.control_plane.governance import get_governance
        from app.control_plane.projects import get_projects
        gate = get_governance()
        pid = get_projects().get_active_project_id()

        req = gate.request_approval(
            project_id=pid,
            request_type="model_promotion",
            requested_by="llm_discovery",
            title=f"New model: {model.get('display_name', model_id)} for {role}",
            detail={
                "model_id": model_id,
                "tier": tier,
                "role": role,
                "benchmark_score": benchmark_score,
                "current_model": current_model,
                "current_score": current_score,
                "improvement": f"{((benchmark_score/current_score)-1)*100:.1f}%",
                "cost": f"${model.get('cost_output_per_m', 0):.4f}/M output",
            },
        )
        logger.info(f"llm_discovery: governance request created for {model_id}")
        return {"status": "pending_approval", "request_id": str(req.get("id", "")), "model": model_id}
    except Exception as e:
        logger.warning(f"llm_discovery: governance request failed: {e}")
        return None


def _add_to_runtime_catalog(model: dict, roles: list[str]) -> None:
    """Add a discovered model to the runtime catalog (in-memory overlay).

    The static CATALOG in llm_catalog.py is not modified — this adds
    models to a dynamic overlay that's checked alongside the static catalog.
    """
    from app.llm_catalog import CATALOG

    name = model["model_id"].split("/")[-1] if "/" in model["model_id"] else model["model_id"]

    # Estimate strengths from benchmark
    base_score = model.get("benchmark_score", 0.5) if isinstance(model.get("benchmark_score"), (int, float)) else 0.5
    strengths = {r: round(base_score, 2) for r in roles}
    strengths["general"] = round(base_score * 0.9, 2)

    entry = {
        "tier": model.get("tier", "budget"),
        "provider": model.get("provider", "openrouter"),
        "model_id": model["model_id"],
        "context": model.get("context_window", 32768),
        "multimodal": model.get("multimodal", False),
        "cost_input_per_m": model.get("cost_input_per_m", 0),
        "cost_output_per_m": model.get("cost_output_per_m", 0),
        "tool_use_reliability": 0.70,
        "description": f"Auto-discovered: {model.get('display_name', name)}",
        "strengths": strengths,
        "_discovered": True,  # Marker for dynamic models
    }

    # Add to catalog (runtime only — not persisted to .py file)
    CATALOG[name] = entry
    logger.info(f"llm_discovery: added {name} to runtime catalog (tier={entry['tier']})")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def run_discovery_cycle(max_benchmarks: int = 3) -> dict:
    """Full discovery pipeline. Called by idle scheduler.

    Returns: {scanned, new_found, benchmarked, promoted, proposals}
    """
    result = {
        "scanned": 0, "new_found": 0, "benchmarked": 0,
        "promoted": 0, "proposals": 0, "errors": [],
    }

    # Step 1: Scan OpenRouter
    raw_models = scan_openrouter()
    result["scanned"] = len(raw_models)

    if not raw_models:
        return result

    # Step 2: Filter + normalize
    known_ids = _get_known_model_ids()
    catalog_ids = _get_catalog_model_ids()

    new_models = []
    for raw in raw_models:
        normalized = _normalize_model(raw, provider="openrouter")
        if not normalized:
            continue
        mid = normalized["model_id"]
        if mid not in known_ids and mid not in catalog_ids:
            new_models.append(normalized)

    result["new_found"] = len(new_models)

    # Step 3: Store all new discoveries
    for model in new_models:
        _store_discovered(model)

    if not new_models:
        logger.info(f"llm_discovery: scanned {result['scanned']} models, no new discoveries")
        return result

    logger.info(f"llm_discovery: found {len(new_models)} new models")

    # Step 4: Benchmark top candidates (cheapest + most capable first)
    # Sort by cost (prefer cheap) then by context window (prefer large)
    candidates = sorted(
        new_models,
        key=lambda m: (m["cost_output_per_m"], -m["context_window"]),
    )[:max_benchmarks]

    for model in candidates:
        # Pick the most relevant role to benchmark
        role = "research"  # Default
        if model.get("multimodal"):
            role = "research"  # Multimodal models are best tested on research

        score = benchmark_model(model["model_id"], role=role, sample_size=2)
        if score < 0:
            result["errors"].append(f"Benchmark failed for {model['model_id']}")
            continue

        _update_benchmark(model["model_id"], score, role)
        model["benchmark_score"] = score
        result["benchmarked"] += 1

        # Step 5: Compare and propose
        proposal = propose_promotion(model, score, role)
        if proposal:
            if proposal.get("status") == "auto_promoted":
                result["promoted"] += 1
            else:
                result["proposals"] += 1

    # Audit trail
    try:
        from app.control_plane.audit import get_audit
        get_audit().log(
            actor="llm_discovery",
            action="discovery.cycle",
            detail=result,
        )
    except Exception:
        pass

    logger.info(f"llm_discovery: cycle complete — {result}")
    return result


def get_discovered_models(status: str = None, limit: int = 20) -> list[dict]:
    """Get discovered models for dashboard/Signal display."""
    from app.control_plane.db import execute
    if status:
        return execute(
            """SELECT model_id, display_name, provider, context_window,
                      cost_output_per_m, benchmark_score, benchmark_role,
                      status, promoted_tier, discovered_at
               FROM control_plane.discovered_models
               WHERE status = %s
               ORDER BY discovered_at DESC LIMIT %s""",
            (status, limit), fetch=True,
        ) or []
    return execute(
        """SELECT model_id, display_name, provider, context_window,
                  cost_output_per_m, benchmark_score, benchmark_role,
                  status, promoted_tier, discovered_at
           FROM control_plane.discovered_models
           ORDER BY discovered_at DESC LIMIT %s""",
        (limit,), fetch=True,
    ) or []


def format_discovery_report() -> str:
    """Human-readable discovery status for Signal."""
    models = get_discovered_models(limit=10)
    if not models:
        return "🔍 No models discovered yet. Run discovery with: discover models"

    lines = ["🔍 LLM Discovery:"]
    for m in models:
        status_icon = {"discovered": "🆕", "benchmarking": "⏳", "approved": "✅",
                       "promoted": "🚀", "rejected": "❌", "retired": "🗄️"}.get(m.get("status"), "?")
        score = f" score={m['benchmark_score']:.2f}" if m.get("benchmark_score") else ""
        cost = f" ${m.get('cost_output_per_m', 0):.3f}/Mo" if m.get("cost_output_per_m") else " free"
        lines.append(f"  {status_icon} {m.get('display_name', '?')[:40]}{score}{cost}")
    return "\n".join(lines)
