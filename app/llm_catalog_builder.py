"""
llm_catalog_builder.py — Self-populating LLM catalog.

The static CATALOG in `app.llm_catalog` is now a 3-entry survival bootstrap.
Everything beyond that is merged in at runtime from live sources:

    OpenRouter /models          pricing, context, modality, supported_parameters
    Artificial Analysis v2      intelligence_index, coding_index, math_index, speed
    Ollama /api/tags            local model inventory

``build_snapshot`` assembles a normalised catalog dict, ``derive_strengths``
maps AA evaluation columns to our nine canonical task types, and ``refresh``
persists a JSON snapshot under ``workspace/cache/llm_catalog_snapshot.json``
so the next startup can hydrate without a live network round-trip.

The builder is fail-open: any source can be missing and the remaining
sources still populate the catalog. If everything is missing, CATALOG
keeps its bootstrap contents — the system still boots.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_SNAPSHOT_PATH = Path("/app/workspace/cache/llm_catalog_snapshot.json")
_SNAPSHOT_TTL_HOURS_DEFAULT = 24.0

_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
_AA_URL = "https://artificialanalysis.ai/api/v2/data/llms/models"

# OpenRouter publishes router / meta models that aren't real LLMs
# (``auto`` picks for you, ``bodybuilder`` is a prompt template, etc.)
# They show up with bogus prices like ``-1000000`` — exclude by id/slug.
_OPENROUTER_PSEUDO_SLUGS: frozenset[str] = frozenset({
    "auto", "bodybuilder", "openrouter/auto", "openrouter/bodybuilder",
})

# Cost sanity window. Anything outside [0, 500] $/M is assumed junk —
# covers free, paid, and priority-tier pricing without admitting the
# negative-sentinel or accidental unit-mismatch rows we've seen from OR.
_COST_SANITY_MIN_PER_M = 0.0
_COST_SANITY_MAX_PER_M = 500.0

# Tier classification by output-$/M token. Matches llm_discovery's ladder.
_TIER_BY_COST: tuple[tuple[float, str], ...] = (
    (0.0, "free"),
    (1.0, "budget"),
    (5.0, "mid"),
    (30.0, "premium"),
)

# AA variant slugs we strip before catalog-key resolution (same suffixes as
# llm_external_ranks so both modules agree on base-model identity).
_AA_VARIANT_SUFFIXES: tuple[str, ...] = (
    "-adaptive", "-thinking", "-reasoning",
    "-xhigh", "-high", "-medium", "-low",
    "-max-effort", "-max", "-min-effort", "-min",
    "-non-reasoning-low-effort",
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _norm_index(value) -> float:
    """Normalise an AA index to 0..1.

    AA's intelligence / coding / math indices run on a 0..100 scale but
    frontier models score in the 55..80 band — a naive ``/100`` divide
    leaves even the strongest model under 0.80, which pushes AA-backed
    premium entries below the 0.88 tier-default we use for fallbacks.
    Scale by 70 (roughly the 95th-percentile frontier score) so Opus-
    class entries map to ~0.82 and non-AA tier-default fallbacks stay
    comparable.
    """
    if value is None:
        return 0.0
    try:
        return max(0.0, min(1.0, float(value) / 70.0))
    except (TypeError, ValueError):
        return 0.0


def _tier_for_cost(cost_output_per_m: float) -> str:
    """Map an output $/M price to our tier ladder."""
    for ceiling, tier in _TIER_BY_COST:
        if cost_output_per_m <= ceiling:
            return tier
    return "premium"


def _strip_aa_variant(slug: str) -> str:
    """Strip reasoning / effort-level suffixes from an AA slug so multiple
    variants collapse onto the same base model key."""
    s = slug
    for sfx in _AA_VARIANT_SUFFIXES:
        if s.endswith(sfx):
            return s[: -len(sfx)]
    return s


def _canonical_key(slug: str) -> str:
    """Convert a dash-separated AA slug to our catalog key style.

    AA emits ``deepseek-v3-2``; our catalog uses ``deepseek-v3.2``.
    The safe heuristic: a lone ``digit-dash-digit`` pattern is almost
    always a version delimiter and should become ``digit.digit``.
    """
    import re
    return re.sub(r"(\d)-(\d)", r"\1.\2", slug)


# ── Strength derivation ──────────────────────────────────────────────────

def derive_strengths(
    aa_row: dict | None,
    *,
    is_multimodal: bool,
    tier: str,
) -> dict[str, float]:
    """Map AA evaluations to our nine canonical task types.

    When AA has no row for the model we fall back to tier-weighted
    defaults, but we discount them by a confidence factor so an
    AA-measured frontier model is preferred over a tier-default
    fallback of equal nominal strength. Models we've actually
    benchmarked should win over ones we haven't.
    """
    if not aa_row:
        base = {
            "premium": 0.82, "mid": 0.76, "budget": 0.70,
            "free": 0.62, "local": 0.66,
        }.get(tier, 0.66)
        # Confidence discount: no AA row → we don't know how strong this
        # model really is. Better to assume "plausibly good" than "top of
        # its tier", otherwise unmeasured premium entries unfairly beat
        # AA-measured ones in the scoring function.
        base *= 0.85
        out = {
            "coding": base, "debugging": base, "architecture": base,
            "research": base, "writing": base, "reasoning": base,
            "vetting": base, "general": base,
        }
        out["multimodal"] = 1.0 if is_multimodal else 0.0
        return out

    evals = aa_row.get("evaluations") or {}
    intel = _norm_index(evals.get("artificial_analysis_intelligence_index"))
    coding = _norm_index(evals.get("artificial_analysis_coding_index"))
    math = _norm_index(evals.get("artificial_analysis_math_index"))

    # 0..1 already (raw probabilities)
    gpqa = float(evals.get("gpqa") or 0.0)
    livecodebench = float(evals.get("livecodebench") or 0.0)
    ifbench = float(evals.get("ifbench") or 0.0)

    # Coding score: prefer the dedicated AA coding index when present;
    # livecodebench provides an extra floor for contest-style coding;
    # fall back to overall intelligence so coding-capable models that
    # AA hasn't scored on the coding axis still rank sensibly.
    code_score = max(coding, livecodebench) if (coding or livecodebench) else intel

    reasoning_score = (
        0.4 * math + 0.3 * gpqa + 0.3 * intel
        if math else
        0.5 * gpqa + 0.5 * intel
        if gpqa else
        intel
    )

    return {
        "coding":       round(code_score, 3),
        "debugging":    round(code_score * 0.95, 3),
        "architecture": round(0.6 * intel + 0.4 * code_score, 3),
        "research":     round(intel, 3),
        "writing":      round(intel * 0.95, 3),
        "reasoning":    round(reasoning_score, 3),
        "multimodal":   1.0 if is_multimodal else 0.0,
        "vetting":      round(0.6 * intel + 0.2 * gpqa + 0.2 * ifbench, 3),
        "general":      round(intel, 3),
    }


def derive_tool_use_reliability(entry: dict) -> float:
    """Best-effort tool-use score until vetting-feedback telemetry
    accumulates enough signal to replace it (Phase 4 already wired).
    """
    if entry.get("supports_tools") is False:
        return 0.0
    tier_floor = {
        "premium": 0.90, "mid": 0.82, "budget": 0.76,
        "free": 0.60, "local": 0.65,
    }
    return tier_floor.get(entry.get("tier", "budget"), 0.70)


# ── Source fetchers ──────────────────────────────────────────────────────

def _fetch_openrouter() -> list[dict]:
    """Return the OpenRouter /models payload (empty list on any failure)."""
    try:
        from app.config import get_settings
        api_key = get_settings().openrouter_api_key.get_secret_value()
    except Exception:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = httpx.get(
            _OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("data", []) or []
    except Exception as exc:
        logger.debug(f"catalog_builder: OpenRouter fetch failed: {exc}")
        return []


def _fetch_artificial_analysis() -> list[dict]:
    """Return the AA /v2/data/llms/models payload (empty list on failure)."""
    try:
        from app.config import get_settings
        s = get_settings()
        api_key = (
            s.artificial_analysis_api_key.get_secret_value()
            if hasattr(s, "artificial_analysis_api_key") else ""
        ) or os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "") or os.getenv("AA_API_KEY", "")
    except Exception:
        api_key = os.getenv("ARTIFICIAL_ANALYSIS_API_KEY", "") or os.getenv("AA_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = httpx.get(
            _AA_URL, headers={"x-api-key": api_key}, timeout=30,
        )
        if resp.status_code != 200:
            return []
        return resp.json().get("data", []) or []
    except Exception as exc:
        logger.debug(f"catalog_builder: AA fetch failed: {exc}")
        return []


def _fetch_ollama_tags() -> list[dict]:
    """Return the local Ollama model list (empty list when Ollama is off)."""
    url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=5)
        if resp.status_code != 200:
            return []
        return resp.json().get("models", []) or []
    except Exception:
        return []


# ── Entry construction ──────────────────────────────────────────────────

def _tool_calling_from_openrouter(or_row: dict) -> bool:
    sp = or_row.get("supported_parameters") or []
    if isinstance(sp, (list, tuple, set)) and sp:
        return any(p in sp for p in ("tools", "tool_choice", "function_call"))
    return True  # OpenRouter default if the field is absent


def _build_openrouter_entry(
    catalog_key: str,
    or_row: dict,
    aa_row: dict | None,
) -> dict | None:
    # Skip router/meta/pseudo models — they show up with nonsense
    # pricing like -$1M/M and aren't real LLMs.
    raw_id = or_row.get("id", "") or ""
    raw_slug = raw_id.split("/")[-1].lower()
    if raw_slug in _OPENROUTER_PSEUDO_SLUGS or raw_id.lower() in _OPENROUTER_PSEUDO_SLUGS:
        return None

    pricing = or_row.get("pricing") or {}
    cost_input = float(pricing.get("prompt") or 0) * 1_000_000
    cost_output = float(pricing.get("completion") or 0) * 1_000_000

    # Cost sanity: reject rows with negative or absurd prices.
    if (
        cost_output < _COST_SANITY_MIN_PER_M
        or cost_output > _COST_SANITY_MAX_PER_M
        or cost_input < _COST_SANITY_MIN_PER_M
        or cost_input > _COST_SANITY_MAX_PER_M
    ):
        return None

    arch = or_row.get("architecture") or {}
    modality = arch.get("modality", "text") or ""
    is_mm = "image" in modality or "multimodal" in modality
    tier = _tier_for_cost(cost_output)

    entry = {
        "tier": tier,
        "provider": "openrouter",
        "model_id": or_row.get("id", ""),
        "context": int(or_row.get("context_length") or 0),
        "multimodal": is_mm,
        "cost_input_per_m": round(cost_input, 6),
        "cost_output_per_m": round(cost_output, 6),
        "supports_tools": _tool_calling_from_openrouter(or_row),
        "description": (or_row.get("name") or catalog_key)[:140],
        "_auto": True,
        "_source": "openrouter" + ("+aa" if aa_row else ""),
    }
    entry["model_id"] = _prefix_model_id(entry["provider"], entry["model_id"])
    entry["strengths"] = derive_strengths(aa_row, is_multimodal=is_mm, tier=tier)
    entry["tool_use_reliability"] = derive_tool_use_reliability(entry)
    return entry


def _build_anthropic_entry(
    catalog_key: str,
    aa_row: dict,
) -> dict:
    """Entries for Anthropic models come entirely from AA because Anthropic
    has no public listing endpoint. Pricing comes from AA's pricing block."""
    pricing = aa_row.get("pricing") or {}
    cost_input = float(pricing.get("price_1m_input_tokens") or 0)
    cost_output = float(pricing.get("price_1m_output_tokens") or 0)
    tier = _tier_for_cost(cost_output)
    # AA slug is like "claude-opus-4-7"; the Anthropic SDK needs the same
    # dashed form as the model_id, so we pass it through unchanged.
    raw_slug = _strip_aa_variant(aa_row.get("slug") or "")

    entry = {
        "tier": tier,
        "provider": "anthropic",
        "model_id": f"anthropic/{raw_slug}",
        # AA doesn't expose context length. Fall back to Anthropic's current
        # 1M floor so selection isn't forced to prefer other providers.
        "context": 1_000_000,
        "multimodal": True,  # Anthropic 4.x family is uniformly multimodal
        "cost_input_per_m": round(cost_input, 6),
        "cost_output_per_m": round(cost_output, 6),
        "supports_tools": True,
        "description": (aa_row.get("name") or catalog_key)[:140],
        "_auto": True,
        "_source": "aa_anthropic",
    }
    entry["strengths"] = derive_strengths(aa_row, is_multimodal=True, tier=tier)
    entry["tool_use_reliability"] = derive_tool_use_reliability(entry)
    return entry


def _build_local_entry(local_row: dict) -> dict:
    name = local_row.get("name") or local_row.get("model") or ""
    if not name:
        return {}
    # Embedding-only models (e.g. nomic-embed-text, mxbai-embed-large) must
    # NOT enter the chat catalog. Once present, the selector / cascade /
    # auto-promotion can pick them for a chat role and litellm.completion
    # then 400s with "does not support chat" on every call.
    if "embed" in name.lower():
        return {}
    size_bytes = int(local_row.get("size") or 0)
    size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else 0
    entry = {
        "tier": "local",
        "provider": "ollama",
        "model_id": f"ollama_chat/{name}",
        "size_gb": size_gb or 0,
        "ram_gb": int(size_gb * 1.3) if size_gb else 0,
        "speed": "medium",
        "context": 32_768,
        "multimodal": False,
        "cost_input_per_m": 0.0,
        "cost_output_per_m": 0.0,
        "supports_tools": True,  # Ollama default; codestral-style exceptions
                                 # handled by bootstrap or manual overlay
        "description": f"Ollama local: {name}",
        "_auto": True,
        "_source": "ollama",
    }
    entry["strengths"] = derive_strengths(None, is_multimodal=False, tier="local")
    entry["tool_use_reliability"] = derive_tool_use_reliability(entry)
    return entry


def _prefix_model_id(provider: str, raw_id: str) -> str:
    if not raw_id:
        return raw_id
    if provider == "openrouter" and not raw_id.startswith("openrouter/"):
        return f"openrouter/{raw_id}"
    return raw_id


# ── Merge logic ─────────────────────────────────────────────────────────

def build_snapshot() -> dict[str, dict]:
    """Fetch every source and assemble the merged catalog dict.

    Resolution order per model key:
      1. Anthropic entries via AA (authoritative for Claude family).
      2. OpenRouter entries merged with AA rows when the key matches.
      3. OpenRouter entries alone when AA has no match.
      4. Ollama local entries (non-conflicting, separate namespace).
    """
    openrouter_raw = _fetch_openrouter()
    aa_raw = _fetch_artificial_analysis()
    ollama_raw = _fetch_ollama_tags()

    # Index AA by canonical-key'd base slug → row.
    aa_by_key: dict[str, dict] = {}
    for m in aa_raw:
        base = _strip_aa_variant(m.get("slug", ""))
        key = _canonical_key(base)
        if not key:
            continue
        # Keep the row with highest intelligence (variants of the same
        # base model compete; strongest variant wins for the catalog
        # entry but the slug itself is preserved through the chain).
        current = aa_by_key.get(key)
        if current is None:
            aa_by_key[key] = m
        else:
            cur_intel = (
                (current.get("evaluations") or {})
                .get("artificial_analysis_intelligence_index") or -1
            )
            new_intel = (
                (m.get("evaluations") or {})
                .get("artificial_analysis_intelligence_index") or -1
            )
            if new_intel > cur_intel:
                aa_by_key[key] = m

    # Index OpenRouter by catalog key (last segment of slashed id).
    or_by_key: dict[str, dict] = {}
    for m in openrouter_raw:
        mid = m.get("id", "")
        if not mid:
            continue
        key = mid.split("/")[-1]
        or_by_key[_canonical_key(key)] = m

    snapshot: dict[str, dict] = {}

    # 1. Anthropic from AA (authoritative).
    for key, aa_row in aa_by_key.items():
        creator = (aa_row.get("model_creator") or {}).get("slug") or ""
        if creator != "anthropic":
            continue
        entry = _build_anthropic_entry(key, aa_row)
        if entry:
            snapshot[key] = entry

    # 2. OpenRouter entries (merged with AA where we have a match).
    for key, or_row in or_by_key.items():
        if key in snapshot:
            continue  # already covered by Anthropic/AA branch
        aa_row = aa_by_key.get(key)
        entry = _build_openrouter_entry(key, or_row, aa_row)
        if entry:
            snapshot[key] = entry

    # 3. Ollama locals — separate namespace, never collides with API models.
    for local in ollama_raw:
        entry = _build_local_entry(local)
        if entry:
            name = local.get("name", "")
            if name:
                snapshot[name] = entry

    snapshot["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "catalog_builder: built snapshot — "
        f"{len([k for k in snapshot if not k.startswith('_')])} models "
        f"(openrouter={len(openrouter_raw)}, aa={len(aa_raw)}, ollama={len(ollama_raw)})"
    )
    return snapshot


def merge_into_catalog(snapshot: dict[str, dict]) -> int:
    """Apply a snapshot to the live CATALOG in place. Returns number of
    entries added or updated. Bootstrap entries are preserved; live data
    overrides their auto-derived fields but never removes them.
    """
    from app.llm_catalog import CATALOG, _BOOTSTRAP_CATALOG
    added = 0
    for key, entry in snapshot.items():
        if key.startswith("_"):
            continue
        prior = CATALOG.get(key)
        if prior and not prior.get("_auto"):
            # Bootstrap entry — keep it, but refresh cost/strengths from
            # live data so the hand-coded numbers don't go stale.
            for field in (
                "cost_input_per_m", "cost_output_per_m",
                "strengths", "tool_use_reliability", "context", "multimodal",
            ):
                if field in entry:
                    prior[field] = entry[field]
            prior["_source"] = f"bootstrap+{entry.get('_source', '')}"
            continue
        CATALOG[key] = entry
        added += 1
    return added


# ── Persistence + entry point ───────────────────────────────────────────

def _load_snapshot() -> dict | None:
    try:
        if _SNAPSHOT_PATH.exists():
            return json.loads(_SNAPSHOT_PATH.read_text())
    except Exception as exc:
        logger.debug(f"catalog_builder: snapshot load failed: {exc}")
    return None


def _persist_snapshot(snapshot: dict) -> None:
    try:
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2))
    except Exception as exc:
        logger.debug(f"catalog_builder: snapshot persist failed: {exc}")


def _age_hours(snapshot: dict) -> float:
    ts = snapshot.get("_fetched_at")
    if not ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return float("inf")


def refresh(
    force: bool = False,
    ttl_hours: float = _SNAPSHOT_TTL_HOURS_DEFAULT,
) -> dict:
    """Load or rebuild the snapshot, merge it into CATALOG, return summary.

    - If the on-disk snapshot is younger than ``ttl_hours`` and ``force``
      is False, it's used as-is without hitting the network.
    - Otherwise all three sources are queried, a new snapshot is
      persisted, and the live CATALOG is mutated in place.
    """
    snap = _load_snapshot() if not force else None
    if not snap or _age_hours(snap) > ttl_hours:
        snap = build_snapshot()
        _persist_snapshot(snap)

    added = merge_into_catalog(snap)

    # Re-apply governance-approved overrides (model_id_remap, retired) so
    # freshly merged entries from the live sources don't overwrite approved
    # remaps.  Without this, every 24h refresh would undo an approved remap
    # until the next benchmark cycle re-detected the dead ID.
    from app.llm_catalog import CATALOG, _apply_overrides_to_catalog
    try:
        _apply_overrides_to_catalog(CATALOG)
    except Exception:
        pass

    return {
        "catalog_size": len([k for k in CATALOG if not k.startswith("_")]),
        "added_or_updated": added,
        "fetched_at": snap.get("_fetched_at"),
        "source_counts": {
            k: sum(1 for e in snap.values() if isinstance(e, dict) and e.get("_source", "").startswith(k))
            for k in ("openrouter", "aa_anthropic", "ollama")
        },
    }


def format_refresh_summary(summary: dict) -> str:
    """Human-readable refresh summary for Signal output."""
    from app.llm_catalog import CATALOG
    lines = [
        f"🗂  Catalog refresh — {summary['added_or_updated']} added/updated, "
        f"{summary['catalog_size']} total",
        f"   fetched_at: {summary.get('fetched_at', 'n/a')}",
    ]
    counts = summary.get("source_counts", {})
    if counts:
        parts = [f"{k}={v}" for k, v in counts.items() if v]
        if parts:
            lines.append(f"   sources: {', '.join(parts)}")
    return "\n".join(lines)
