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

logger = logging.getLogger(__name__)

# Minimum thresholds for a model to be considered
MIN_CONTEXT_WINDOW = 8_000
# Raised to 30 so Opus-class frontier launches at $25/M still get seen.
# The tier buckets below still constrain where a model lands in the
# cascade; this is just the outer "worth evaluating at all" gate.
MAX_COST_OUTPUT_PER_M = 30.0

# Tier classification by cost
TIER_THRESHOLDS = {
    "free": 0.0,
    "budget": 1.0,       # ≤ $1/M output
    "mid": 5.0,          # ≤ $5/M output
    "premium": 30.0,     # ≤ $30/M output
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
            # Skip embedding-only models — they must not be persisted as
            # chat-capable. See _build_local_entry in llm_catalog_builder.py.
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
                if "embed" not in (m.get("name") or "").lower()
            ]
    except Exception:
        pass
    return []

# ── Filter + Normalize ───────────────────────────────────────────────────────

def _detect_tool_calling(raw: dict, provider: str) -> bool:
    """Infer whether a model supports tool calling.

    OpenRouter's ``/models`` payload exposes ``supported_parameters``
    which includes ``"tools"``/``"tool_choice"`` for models that accept
    tool-use arguments. Ollama doesn't report this, so we fall back to
    a conservative heuristic on the model family.
    """
    supported = raw.get("supported_parameters") or raw.get("supported_params") or []
    if isinstance(supported, (list, tuple, set)):
        if any(p in supported for p in ("tools", "tool_choice", "function_call")):
            return True
        if supported:  # payload is authoritative — no tools listed means none
            return False
    # Fallback heuristic when the field is absent
    name = (raw.get("id", "") + " " + raw.get("name", "")).lower()
    _TOOLLESS_HINTS = ("base", "completion", "codestral", "embed")
    if any(h in name for h in _TOOLLESS_HINTS):
        return False
    if provider == "ollama":
        return False  # Ollama path verifies at runtime via circuit breaker
    return True


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
    tool_calling = _detect_tool_calling(raw, provider)

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
        "tool_calling": tool_calling,
        "tier": tier,
        "raw_metadata": raw,
    }

# ── Database Operations ──────────────────────────────────────────────────────

def _get_known_model_ids() -> set[str]:
    """Get model_ids that are fully cataloged (not stubs).

    Rows with zero cost AND zero context are stubs (e.g. from tech_radar
    hints) — they are excluded so the authoritative scanner
    (OpenRouter/Ollama) can re-discover and enrich them on its next run
    via _store_discovered's ON CONFLICT.
    """
    from app.control_plane.db import execute
    rows = execute(
        "SELECT model_id FROM control_plane.discovered_models "
        "WHERE cost_output_per_m > 0 OR context_window > 0",
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

def _store_discovered(model: dict, source: str = "openrouter_api") -> bool:
    """Store a newly discovered model in PostgreSQL.

    ON CONFLICT enriches existing rows (e.g. stubs planted by tech_radar)
    with the authoritative scanner's data — but preserves `source`
    (attribution to the first discoverer) and `status` (don't reset
    benchmarking progress).
    """
    from app.control_plane.db import execute
    try:
        execute(
            """INSERT INTO control_plane.discovered_models
               (model_id, provider, display_name, context_window,
                cost_input_per_m, cost_output_per_m, multimodal, tool_calling,
                source, raw_metadata, status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'discovered')
               ON CONFLICT (model_id) DO UPDATE SET
                provider = EXCLUDED.provider,
                display_name = EXCLUDED.display_name,
                context_window = EXCLUDED.context_window,
                cost_input_per_m = EXCLUDED.cost_input_per_m,
                cost_output_per_m = EXCLUDED.cost_output_per_m,
                multimodal = EXCLUDED.multimodal,
                tool_calling = EXCLUDED.tool_calling,
                raw_metadata = EXCLUDED.raw_metadata,
                updated_at = NOW()""",
            (
                model["model_id"], model["provider"], model["display_name"],
                model["context_window"], model["cost_input_per_m"],
                model["cost_output_per_m"], model["multimodal"],
                model.get("tool_calling", True), source,
                json.dumps(model.get("raw_metadata", {})),
            ),
        )
        return True
    except Exception as e:
        logger.debug(f"llm_discovery: store failed: {e}")
        return False


def _store_stub(
    model_id: str,
    provider: str,
    display_name: str,
    source: str,
    metadata: dict | None = None,
) -> bool:
    """Plant a minimal-info row for a model hinted by a non-authoritative source.

    Used by tech_radar when it hears about a model in the news but doesn't
    have pricing/context/capability data. The row stays invisible to
    _get_known_model_ids() (zero cost, zero context) so the authoritative
    scanner (OpenRouter/Ollama) can re-discover and enrich it on its next
    cycle. ON CONFLICT DO NOTHING prevents overwriting real data if the
    row already exists.
    """
    from app.control_plane.db import execute
    try:
        execute(
            """INSERT INTO control_plane.discovered_models
               (model_id, provider, display_name, context_window,
                cost_input_per_m, cost_output_per_m, multimodal, tool_calling,
                source, raw_metadata, status)
               VALUES (%s, %s, %s, 0, 0, 0, FALSE, TRUE, %s, %s, 'discovered')
               ON CONFLICT (model_id) DO NOTHING""",
            (
                model_id, provider, display_name,
                source, json.dumps(metadata or {}),
            ),
        )
        return True
    except Exception as e:
        logger.debug(f"llm_discovery: stub insert failed: {e}")
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
    """Mark a model as promoted in ``discovered_models``.

    ``promoted_roles`` is **merged** with the existing array — each
    call appends to the set of roles the model is endorsed for.
    Previously this was an outright replace, which meant a second
    approval for a different role clobbered the first.
    """
    from app.control_plane.db import execute
    execute(
        """
        UPDATE control_plane.discovered_models
           SET status = 'promoted',
               promoted_tier = %s,
               promoted_roles = ARRAY(
                   SELECT DISTINCT unnest(
                       COALESCE(promoted_roles, ARRAY[]::text[]) || %s::text[]
                   )
               ),
               promoted_at = COALESCE(promoted_at, NOW()),
               reviewed_by = %s,
               updated_at = NOW()
         WHERE model_id = %s
        """,
        (tier, roles, reviewer, model_id),
    )

# ── Benchmarking ─────────────────────────────────────────────────────────────

# The judge rotation is computed dynamically from the current catalog:
# ``_discover_judges`` picks the highest-intelligence catalog entry in each
# of three different provider families. This way a freshly-launched stronger
# model (e.g. Opus 4.8, Gemini 4) automatically becomes the reference judge
# the week after it lands — no hand-curated list to maintain.
#
# The previous hand-coded tuple is kept here commented-out as a reference
# for what the fallback-of-last-resort looks like when the catalog is in
# bootstrap-only state.
#   DEFAULT_JUDGES = (
#       ("claude-sonnet-4.6", "anthropic"),
#       ("gemini-3.1-pro",    "google"),
#       ("deepseek-v3.2",     "deepseek"),
#   )


def _discover_judges(
    target_families: int = 3,
    min_strength: float = 0.70,
) -> tuple[tuple[str, str], ...]:
    """Pick the strongest catalog entries across distinct provider families.

    Resolution order per family:
      1. **Manual pin** in ``control_plane.judge_pins`` — operator
         override; takes precedence regardless of strength.
      2. **Dynamic** pick — highest-``reasoning``-strength catalog entry
         in that family (the closest proxy to "good at judging another
         model's output").

    Scans the live CATALOG so a freshly-launched stronger model
    (e.g. Opus 4.8, Gemini 4) automatically becomes the reference judge
    the week after it lands — no hand-curated list to maintain.

    Falls back to whatever the bootstrap provides if the catalog hasn't
    been refreshed yet — survival mode still benchmarks, just with one
    judge instead of three.
    """
    from app.llm_catalog import CATALOG
    # ── Pinned overrides ────────────────────────────────────────────
    # A row in ``judge_pins`` forces a specific model for its family,
    # regardless of intelligence ranking. Models that aren't in the
    # live CATALOG are silently ignored (the dashboard write path
    # validates this, but defensive read-side too).
    pinned_by_family: dict[str, str] = {}
    try:
        from app.llm_judge_pins import list_pins as _list_judge_pins
        for fam, model in _list_judge_pins().items():
            if model in CATALOG:
                pinned_by_family[fam] = model
    except Exception:
        pass  # graceful — falls through to dynamic-only behaviour

    # ── Rank every catalog entry by judging-relevant strength ───────
    scored: list[tuple[str, str, float]] = []
    for name, entry in CATALOG.items():
        if entry.get("supports_tools") is False:
            continue  # must be able to return structured JSON verdicts
        strengths = entry.get("strengths", {})
        # reasoning is the best single proxy for judge ability;
        # fall back to vetting or general if missing.
        s = float(
            strengths.get("reasoning")
            or strengths.get("vetting")
            or strengths.get("general")
            or 0.0
        )
        if s < min_strength:
            continue
        scored.append((name, _provider_family(entry.get("model_id", name)), s))
    scored.sort(key=lambda t: -t[2])

    # One winner per family, up to target_families. Pinned families
    # use their pin first; everything else uses the top dynamic pick.
    picks: list[tuple[str, str]] = []
    seen_families: set[str] = set()

    # First pass: emit pinned families (preserves operator intent).
    for fam, model in pinned_by_family.items():
        if fam in seen_families:
            continue
        seen_families.add(fam)
        picks.append((model, fam))
        if len(picks) >= target_families:
            return tuple(picks)

    # Second pass: fill remaining slots dynamically.
    for name, family, _score in scored:
        if family in seen_families:
            continue
        seen_families.add(family)
        picks.append((name, family))
        if len(picks) >= target_families:
            break
    return tuple(picks)


def _provider_family(model_id: str) -> str:
    """Infer the provider family from a catalog model_id or key.

    Used to exclude judges whose family matches the candidate. The
    classification is intentionally coarse — a family boundary is
    enough to catch the same-lab scoring bias without getting bogged
    down in taxonomy.
    """
    s = (model_id or "").lower()
    if "claude" in s or "anthropic" in s:
        return "anthropic"
    if "gemini" in s or "google/gemma" in s or "gemma-" in s or s.startswith("gemma"):
        return "google"
    if "deepseek" in s:
        return "deepseek"
    if "gpt-" in s or "openai" in s:
        return "openai"
    if "mistral" in s or "codestral" in s:
        return "mistral"
    # qwen (Alibaba) comes before llama because Ollama model paths
    # like "ollama_chat/qwen3:30b-a3b" contain the substring "llama"
    # via the "ollama" prefix.
    if "qwen" in s or "alibaba" in s:
        return "alibaba"
    if "llama" in s or "meta/" in s:
        return "meta"
    if "kimi" in s or "moonshot" in s:
        return "moonshot"
    if "minimax" in s:
        return "minimax"
    if "glm" in s or "zhipu" in s:
        return "zhipu"
    if "xiaomi" in s or "mimo" in s:
        return "xiaomi"
    if "nemotron" in s or "nvidia" in s:
        return "nvidia"
    if "stepfun" in s or "step-" in s:
        return "stepfun"
    if "arcee" in s or "trinity" in s:
        return "arcee"
    return "unknown"


class _FallbackLLM:
    """Wrap a primary judge LLM with one OpenRouter fallback.

    Catches **specifically** the credit/auth/billing classes of error
    (out-of-credits on the direct provider API) and retries through
    OpenRouter, which proxies to multiple upstream providers and rarely
    runs out of credits in lockstep with one specific vendor account.

    Generic errors (timeout, rate limit, model_not_found) are NOT
    swapped — those should surface so they're visible in the
    benchmark / discovery logs, not silently get a different model.

    The wrapper records which path executed in
    ``self.last_used_fallback`` so the telemetry layer can flag panels
    where the OpenRouter fallback fired.
    """

    # Fragments that mean "the direct API ran out of money" — checked
    # case-insensitively against the exception's stringified form.
    _CREDIT_TOKENS = (
        "credit", "billing", "insufficient", "quota exceeded",
        "no credit", "out of credits", "402", "payment required",
    )
    _AUTH_TOKENS = ("401", "unauthorized", "authentication", "invalid api key")

    def __init__(self, primary, fallback, name: str = "") -> None:
        self._primary = primary
        self._fallback = fallback
        self._name = name
        self.last_used_fallback: bool = False

    def _should_fallback(self, exc: BaseException) -> bool:
        if self._fallback is None:
            return False
        msg = str(exc).lower()
        if any(tok in msg for tok in self._CREDIT_TOKENS):
            return True
        if any(tok in msg for tok in self._AUTH_TOKENS):
            return True
        # litellm BadRequestError on Anthropic with HTTP 402 surfaces
        # as a status_code attribute on the exception:
        code = getattr(exc, "status_code", None)
        if code in (401, 402):
            return True
        return False

    def call(self, *args, **kwargs):
        try:
            self.last_used_fallback = False
            return self._primary.call(*args, **kwargs)
        except BaseException as exc:
            if self._should_fallback(exc):
                logger.warning(
                    "_FallbackLLM[%s]: primary failed (%s) — "
                    "swapping to OpenRouter fallback",
                    self._name, type(exc).__name__,
                )
                self.last_used_fallback = True
                return self._fallback.call(*args, **kwargs)
            raise

    def __getattr__(self, name):
        # Delegate everything else (.model, .provider, etc.) to the primary.
        return getattr(self._primary, name)


def _openrouter_equivalent_model_id(catalog_key: str, entry: dict | None) -> str | None:
    """Best-effort lookup of an OpenRouter-routable model_id for ``catalog_key``.

    Strategy:
      1. If the entry already lives on OpenRouter, return its model_id
         (no fallback needed — same provider).
      2. If the catalog records ``openrouter_model_id`` (the builder
         populates this when the same canonical key is published by both
         AA and OpenRouter), return it.
      3. Otherwise infer: many vendors publish ``vendor/model`` ids on
         OpenRouter with the same shape (``anthropic/claude-opus-4-7``,
         ``google/gemini-2.5-pro``, etc.). When the entry's model_id
         already has a vendor prefix we accept it as a candidate; the
         actual probe happens at call time and fails harmlessly through
         the wrapper if the guess is wrong.

    Returns ``None`` when no candidate is found.
    """
    if not entry:
        return None
    provider = entry.get("provider")
    model_id = entry.get("model_id") or catalog_key
    if provider == "openrouter":
        return model_id
    # Builder may populate this hint on dual-published entries.
    or_hint = entry.get("openrouter_model_id")
    if or_hint:
        return or_hint
    # Vendor-prefixed strings ("anthropic/claude-…") are usable by
    # OpenRouter as-is. Bare names (e.g. "claude-sonnet-4-6") aren't.
    if "/" in model_id:
        return model_id
    return None


def _build_primary_llm(catalog_key: str, entry: dict):
    """Build the direct-provider LLM for a judge key (no fallback)."""
    from app.llm_factory import _cached_llm
    from app.config import get_settings, get_anthropic_api_key
    provider = entry.get("provider")
    if provider == "anthropic":
        key = get_anthropic_api_key()
        if not key:
            return None
        return _cached_llm(entry["model_id"], max_tokens=256, api_key=key)
    if provider == "openrouter":
        or_key = get_settings().openrouter_api_key.get_secret_value()
        if not or_key:
            return None
        return _cached_llm(
            entry["model_id"], max_tokens=256,
            base_url="https://openrouter.ai/api/v1", api_key=or_key,
        )
    if provider == "ollama":
        # Local models are their own fallback — no OpenRouter route.
        return _cached_llm(entry["model_id"], max_tokens=256)
    return None


def _build_openrouter_fallback(or_model_id: str):
    """Build an OpenRouter LLM for the given model_id, or None on failure."""
    try:
        from app.llm_factory import _cached_llm
        from app.config import get_settings
        or_key = get_settings().openrouter_api_key.get_secret_value()
        if not or_key:
            return None
        return _cached_llm(
            or_model_id, max_tokens=256,
            base_url="https://openrouter.ai/api/v1", api_key=or_key,
        )
    except Exception:
        return None


def _build_judge_llm(catalog_key: str):
    """Instantiate a judge LLM with OpenRouter credit fallback.

    Behaviour:
      * Build the primary LLM using whatever provider the catalog says
        (typically the vendor's direct API).
      * Look up an OpenRouter equivalent (when the entry isn't already
        OpenRouter-routed); build a second LLM there.
      * Wrap as ``_FallbackLLM(primary, openrouter)`` so a 402 / 401 /
        billing / "out of credits" error on the primary auto-routes
        through OpenRouter without losing the panel slot.

    Returns ``None`` when neither path can be built (no API keys,
    catalog key missing). All other errors are caught and produce
    ``None`` so judge-panel construction never throws.
    """
    try:
        from app.llm_catalog import get_model
        entry = get_model(catalog_key)
        if not entry:
            return None
        primary = _build_primary_llm(catalog_key, entry)
        if primary is None:
            return None
        # Local-tier judges don't need an OpenRouter fallback.
        if entry.get("provider") == "openrouter" or entry.get("tier") == "local":
            return primary
        or_model_id = _openrouter_equivalent_model_id(catalog_key, entry)
        if not or_model_id:
            return primary  # no fallback available; use primary alone
        fallback = _build_openrouter_fallback(or_model_id)
        if fallback is None:
            return primary
        return _FallbackLLM(primary, fallback, name=catalog_key)
    except Exception:
        return None
    return None


def _select_judges(
    candidate_model_id: str,
    judges: list[str] | None = None,
) -> list[tuple[str, str, object]]:
    """Return up to 2 callable judges whose provider family differs
    from the candidate's.

    Rotation source:
      * Explicit ``judges`` list if provided (for tests / manual rebenchmark).
      * Otherwise :func:`_discover_judges` — scans the live CATALOG and
        returns one representative per provider family ranked by judging
        ability. Automatically tracks the current strongest models.

    Returns a list of (catalog_key, family, llm) tuples. Callers should
    average verdicts across the returned judges. When none are eligible
    (the candidate shares a family with every available judge, or all
    keys are missing), returns [].
    """
    if judges:
        rotation = [
            (k, _provider_family(get_model(k)["model_id"] if get_model(k) else k))
            for k in judges
        ]
    else:
        rotation = list(_discover_judges())

    candidate_family = _provider_family(candidate_model_id)
    ok: list[tuple[str, str, object]] = []
    for key, fam in rotation:
        if fam == candidate_family:
            continue
        llm = _build_judge_llm(key)
        if llm is None:
            continue
        ok.append((key, fam, llm))
        if len(ok) >= 2:
            break
    return ok


# ── Model-ID error classification & naming-convention recovery ─────────────
# Providers occasionally retire IDs or change naming conventions:
#   Anthropic: claude-3.5-haiku → claude-3-5-haiku (dots→dashes); older
#              claude-4-sonnet became claude-sonnet-4-6 (word order + version).
#   OpenAI:    gpt-4-turbo  (legacy) → gpt-4o
# When a benchmark hits such a dead ID, exceptions like "model_not_found",
# "decommissioned", "invalid_model" or 404s surface.  The old code lumped
# all these in with transient errors and returned 0.0 — which looks like
# quality drift and spammed the governance queue.
#
# New behaviour:
#   1. Classify the exception into a category (retired / transient / auth).
#   2. On "retired", generate a handful of plausible id variants (dot↔dash,
#      Anthropic word-order flip) and try them one by one.
#   3. If a variant works, return the score AND a canonical-id hint so
#      rebenchmark_incumbent() can update the catalog in place.
#   4. If no variant works, return the BENCH_RETIRED sentinel so callers
#      can mark the entry as retired instead of treating it as drift.

BENCH_RETIRED = -2.0   # model ID is dead — don't retry, mark as retired
BENCH_FAILED = -1.0    # generic failure (auth, no key, no judges, etc.)

# Module-level remap table populated when _probe_model_id discovers that a
# retired model_id has a working naming-convention variant.  Keyed by the
# dead ID, value is the working replacement.  rebenchmark_incumbent() reads
# this after benchmarking to persist the remap into the catalog.
_RETIRED_REMAPS: dict[str, str] = {}

# Phrases that indicate the model ID itself is the problem (not transient).
_RETIRED_MARKERS = (
    "model_not_found", "model not found", "does not exist",
    "decommissioned", "retired", "deprecated",
    "invalid_model", "invalid model", "unknown model",
    "not available", "no longer available", "404", "not_found",
)


def _classify_llm_error(exc: Exception) -> str:
    """Return one of: 'retired', 'auth', 'transient', 'other'."""
    msg = (str(exc) + " " + str(getattr(exc, "response", ""))).lower()
    if any(m in msg for m in _RETIRED_MARKERS):
        return "retired"
    if any(m in msg for m in (
        "authentication", "invalid_api_key", "unauthorized", "403",
        "insufficient_quota", "credit", "billing",
    )):
        return "auth"
    if any(m in msg for m in (
        "overloaded", "529", "timeout", "connection", "503",
        "502", "rate limit", "429",
    )):
        return "transient"
    return "other"


def _generate_id_variants(model_id: str) -> list[str]:
    """Return plausible variants of a model_id for retired-ID recovery.

    Strategies (in order of likelihood, most useful first):
      1. Dot ↔ dash swap in the version portion (claude-3.5-haiku → -3-5-haiku)
      2. Reverse the swap (claude-3-5-haiku → -3.5-haiku)
      3. Anthropic word-order flip: version-in-the-middle vs version-at-end
         (claude-3.7-sonnet → claude-sonnet-3-7; claude-4-opus → claude-opus-4-6)
      4. Strip provider prefix variants (openrouter/anthropic/x ↔ anthropic/x)

    Deduplicated; excludes the original.  Caller should try these in order.
    """
    import re as _re
    variants: list[str] = []
    seen = {model_id}

    def _add(v: str) -> None:
        if v and v not in seen:
            seen.add(v)
            variants.append(v)

    # Split provider prefix if present
    if "/" in model_id:
        prefix, _, bare = model_id.rpartition("/")
    else:
        prefix, bare = "", model_id

    def _with_prefix(v: str) -> str:
        return f"{prefix}/{v}" if prefix else v

    # Strategy 1: swap dots ↔ dashes in version-like segments
    # Match digit.digit or digit.digit.digit sequences
    dot_to_dash = _re.sub(r"(\d+)\.(\d+)", r"\1-\2", bare)
    _add(_with_prefix(dot_to_dash))

    dash_to_dot = _re.sub(r"-(\d)-(\d)", r"-\1.\2", bare)
    _add(_with_prefix(dash_to_dot))

    # Strategy 2: Anthropic-style word reordering.  Matches patterns like
    # claude-<version>-<variant> and claude-<variant>-<version>.
    m = _re.match(r"^(claude)-(\d[\d\.-]*)-(opus|sonnet|haiku)$", bare, _re.IGNORECASE)
    if m:
        # e.g. claude-3.5-haiku → claude-haiku-3-5 AND claude-haiku-3.5
        fam, ver, variant = m.group(1), m.group(2), m.group(3)
        ver_dashes = ver.replace(".", "-")
        ver_dots = ver.replace("-", ".")
        _add(_with_prefix(f"{fam}-{variant}-{ver_dashes}"))
        _add(_with_prefix(f"{fam}-{variant}-{ver_dots}"))

    m2 = _re.match(r"^(claude)-(opus|sonnet|haiku)-(\d[\d\.-]*)$", bare, _re.IGNORECASE)
    if m2:
        # e.g. claude-opus-4-6 → claude-4-6-opus AND claude-4.6-opus
        fam, variant, ver = m2.group(1), m2.group(2), m2.group(3)
        ver_dashes = ver.replace(".", "-")
        ver_dots = ver.replace("-", ".")
        _add(_with_prefix(f"{fam}-{ver_dashes}-{variant}"))
        _add(_with_prefix(f"{fam}-{ver_dots}-{variant}"))

    # Strategy 3: provider prefix variant (openrouter/anthropic/X ↔ anthropic/X)
    if model_id.startswith("openrouter/anthropic/"):
        _add(model_id.replace("openrouter/anthropic/", "anthropic/", 1))
    elif model_id.startswith("anthropic/"):
        _add(model_id.replace("anthropic/", "openrouter/anthropic/", 1))

    # Cap at 4 variants — more than that burns judge calls on long-shots
    return variants[:4]


def _probe_model_id(model_id: str) -> tuple[bool, str]:
    """Cheap reachability probe: send a 1-token ping and classify the result.

    Returns (reachable, classification).  `classification` is one of
    'ok', 'retired', 'auth', 'transient', 'other'.  Used to decide whether
    a naming-convention variant should be tried before giving up.
    """
    try:
        from app.llm_factory import _cached_llm
        from app.config import get_settings, get_anthropic_api_key
        s = get_settings()
        if model_id.startswith("anthropic/"):
            key = get_anthropic_api_key()
            llm = _cached_llm(model_id, max_tokens=8, api_key=key) if key else None
        elif model_id.startswith("openrouter/"):
            key = s.openrouter_api_key.get_secret_value()
            llm = _cached_llm(
                model_id, max_tokens=8,
                base_url="https://openrouter.ai/api/v1", api_key=key,
            ) if key else None
        elif model_id.startswith("ollama_chat/"):
            llm = _cached_llm(model_id, max_tokens=8)
        else:
            return False, "other"
        if llm is None:
            return False, "auth"
        _ = str(llm.call("hi")).strip()
        return True, "ok"
    except Exception as exc:
        return False, _classify_llm_error(exc)


def benchmark_model(
    model_id: str,
    role: str = "research",
    sample_size: int = 2,
    judges: list[str] | None = None,
) -> float:
    """Run a standardised benchmark and return a 0.0-1.0 score.

    Multi-judge with family exclusion:
      - Selects up to 2 judges from DEFAULT_JUDGES whose provider
        family differs from the candidate's (so a new DeepSeek model
        isn't scored by DeepSeek V3.2).
      - Each judge scores every task independently; the task score is
        the mean of the eligible judges' scores.
      - Final score is the mean of task scores.
      - Returns BENCH_FAILED (-1.0) on setup failure (no key, no judges).
      - Returns BENCH_RETIRED (-2.0) if the model ID appears dead AND
        naming-convention variants also failed — caller should mark the
        catalog entry as retired.
    """
    try:
        from app.llm_factory import _cached_llm
        from app.config import get_settings

        s = get_settings()
        or_key = s.openrouter_api_key.get_secret_value()

        # Candidate LLM
        if model_id.startswith("openrouter/"):
            if not or_key:
                return BENCH_FAILED
            candidate_llm = _cached_llm(
                model_id, max_tokens=1024,
                base_url="https://openrouter.ai/api/v1", api_key=or_key,
            )
        elif model_id.startswith("ollama_chat/"):
            candidate_llm = _cached_llm(model_id, max_tokens=1024)
        elif model_id.startswith("anthropic/"):
            from app.config import get_anthropic_api_key
            key = get_anthropic_api_key()
            if not key:
                return BENCH_FAILED
            candidate_llm = _cached_llm(model_id, max_tokens=1024, api_key=key)
        else:
            return BENCH_FAILED

        # Cheap reachability probe BEFORE running full benchmark.  If the
        # ID is dead, we'll save ~20 LLM calls (sample_size × judges × tasks)
        # and can try naming-convention variants instead.
        reachable, kind = _probe_model_id(model_id)
        if not reachable and kind == "retired":
            logger.warning(
                f"llm_discovery: {model_id} probe says retired, trying variants"
            )
            for alt_id in _generate_id_variants(model_id):
                ok, alt_kind = _probe_model_id(alt_id)
                if ok:
                    logger.warning(
                        f"llm_discovery: {model_id} appears dead but "
                        f"{alt_id} works — recording as canonical remap"
                    )
                    # Store the remap on the module so rebenchmark_incumbent()
                    # can surface it to the catalog.  A sidecar dict keyed by
                    # the dead ID.
                    _RETIRED_REMAPS[model_id] = alt_id
                    # Recurse once with the alt ID
                    return benchmark_model(alt_id, role=role, sample_size=sample_size, judges=judges)
            logger.warning(
                f"llm_discovery: {model_id} appears retired and no working "
                f"variant found — returning BENCH_RETIRED"
            )
            return BENCH_RETIRED
        if not reachable and kind == "auth":
            return BENCH_FAILED
        # "transient" / "other" / "ok" → proceed; benchmark handles per-task retry

        # Judges — distinct provider families from the candidate.
        eligible = _select_judges(model_id, judges=judges)
        if not eligible:
            logger.warning(f"llm_discovery: no eligible judges for {model_id}")
            return BENCH_FAILED

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

        import re
        task_scores: list[float] = []
        # Telemetry: persist one row per (task × judge panel) so the
        # dashboard can show inter-rater agreement + which panels hit
        # the OpenRouter fallback (out-of-credits on the direct API).
        try:
            from app.llm_judge_telemetry import record_evaluation as _record_judge_eval
        except Exception:
            _record_judge_eval = None
        for task in tasks:
            try:
                response = str(candidate_llm.call(task)).strip()
                if not response or len(response) < 20:
                    task_scores.append(0.2)
                    continue

                judge_prompt = (
                    f"Score this AI response 0.0-1.0 on accuracy, completeness, clarity.\n"
                    f"Task: {task}\nResponse: {response[:2000]}\n\n"
                    f'Reply ONLY: {{"score": 0.X}}'
                )
                judge_scores: list[float] = []
                judge_keys_used: list[str] = []
                fallback_flags: list[bool] = []
                for jkey, _fam, judge_llm in eligible:
                    try:
                        raw = str(judge_llm.call(judge_prompt)).strip()
                        match = re.search(r'"score"\s*:\s*([\d.]+)', raw)
                        if match:
                            judge_scores.append(
                                min(1.0, max(0.0, float(match.group(1)))),
                            )
                            judge_keys_used.append(jkey)
                            # ``last_used_fallback`` is only present on
                            # the _FallbackLLM wrapper. For raw LLMs
                            # default to False (no wrapper = direct call).
                            fallback_flags.append(
                                bool(getattr(judge_llm, "last_used_fallback", False)),
                            )
                    except Exception:
                        continue
                task_scores.append(
                    sum(judge_scores) / len(judge_scores) if judge_scores else 0.5,
                )
                if _record_judge_eval and judge_keys_used:
                    try:
                        _record_judge_eval(
                            candidate_model=model_id,
                            judges=judge_keys_used,
                            scores=judge_scores,
                            used_fallback=fallback_flags,
                            rubric="benchmark:accuracy_completeness_clarity",
                            task_description=task[:500],
                        )
                    except Exception:
                        pass  # telemetry must never break the benchmark
            except Exception:
                task_scores.append(0.0)

        avg = sum(task_scores) / len(task_scores) if task_scores else 0.0
        judge_keys = ",".join(k for k, _, _ in eligible)
        logger.info(
            f"llm_discovery: benchmark {model_id} on {role} "
            f"via [{judge_keys}]: {avg:.3f}"
        )
        return avg

    except Exception as e:
        logger.warning(f"llm_discovery: benchmark failed for {model_id}: {e}")
        return -1.0


# ── Incumbent drift detection ────────────────────────────────────────────────

# Relative quality drop that triggers a governance alert. 0.20 = 20%.
INCUMBENT_DRIFT_ALERT_THRESHOLD = 0.20


def rebenchmark_incumbent(
    model_name: str,
    *,
    roles: list[str] | None = None,
    sample_size: int = 2,
) -> dict:
    """Re-run benchmarks against a catalog incumbent and refresh its
    strengths columns in place.

    Detects silent drift (e.g. a provider swapping in a cheaper quant
    under the same name, a mid-life quality regression) that the
    selection pipeline would otherwise miss because CATALOG's
    strengths values are static string-literal estimates.

    Returns a dict:
        {
          "model": name,
          "old_scores": {role: float, ...},   # prior strengths
          "new_scores": {role: float, ...},   # fresh benchmark
          "drift": {role: float, ...},        # new - old, per role
          "alerted": bool,                    # drift triggered gov alert
        }
    Missing models or API-unreachable candidates return a summary with
    an ``error`` key instead.
    """
    from app.llm_catalog import CATALOG

    entry = CATALOG.get(model_name)
    if not entry:
        return {"model": model_name, "error": "not in catalog"}

    roles = roles or BENCHMARK_ROLES
    old_scores = {r: float(entry.get("strengths", {}).get(r, 0.5)) for r in roles}

    new_scores: dict[str, float] = {}
    retired_detected = False
    for role in roles:
        score = benchmark_model(entry["model_id"], role=role, sample_size=sample_size)
        if score == BENCH_RETIRED:
            retired_detected = True
            break  # no point trying other roles — the ID is dead
        if score >= 0:
            new_scores[role] = score

    # If the ID is retired, check if benchmark_model found a working variant
    # via naming-convention recovery.  If yes, update the catalog's model_id
    # in place (preserves the catalog KEY so existing references still work).
    original_id = entry.get("model_id", "")
    if original_id in _RETIRED_REMAPS:
        new_id = _RETIRED_REMAPS.pop(original_id)
        entry["model_id"] = new_id
        logger.warning(
            f"llm_discovery: catalog remap '{model_name}': "
            f"{original_id} → {new_id} (naming convention changed upstream)"
        )
        try:
            from app.control_plane.governance import get_governance
            from app.control_plane.projects import get_projects
            get_governance().request_approval(
                project_id=get_projects().get_active_project_id(),
                request_type="model_id_remap",
                requested_by="llm_discovery",
                title=f"Model ID remapped: {model_name}",
                detail={
                    "catalog_key": model_name,
                    "old_model_id": original_id,
                    "new_model_id": new_id,
                    "reason": "original ID returned retired/not-found; "
                              "variant probe succeeded",
                },
            )
        except Exception:
            pass
        # Fall through: new_scores may already be populated from the recursive
        # benchmark call with the working ID.

    if retired_detected and not new_scores:
        logger.warning(
            f"llm_discovery: {model_name} ({original_id}) appears retired — "
            f"no naming-convention variant worked.  Marking as retired."
        )
        # Flag the catalog entry so the selector stops picking it.
        entry["_retired"] = True
        entry["_retired_at"] = datetime.now(timezone.utc).isoformat()
        try:
            from app.control_plane.governance import get_governance
            from app.control_plane.projects import get_projects
            get_governance().request_approval(
                project_id=get_projects().get_active_project_id(),
                request_type="model_retired",
                requested_by="llm_discovery",
                title=f"Model retired: {model_name}",
                detail={
                    "catalog_key": model_name,
                    "model_id": original_id,
                    "reason": "benchmark probe returned not_found / "
                              "decommissioned / deprecated",
                },
            )
        except Exception:
            pass
        return {
            "model": model_name,
            "error": "retired",
            "old_scores": old_scores,
            "model_id": original_id,
        }

    if not new_scores:
        return {
            "model": model_name,
            "error": "no scores produced",
            "old_scores": old_scores,
        }

    # Update strengths in place (runtime only; discovered_models gets
    # an insert-if-missing row for historical tracking).
    strengths = dict(entry.get("strengths", {}))
    for role, score in new_scores.items():
        strengths[role] = round(score, 2)
    entry["strengths"] = strengths

    # Drift analysis
    drift = {
        role: round(new_scores[role] - old_scores[role], 3)
        for role in new_scores
    }
    alerted = False
    worst = min(drift.values()) if drift else 0.0

    # Guard against benchmark-failure-misclassified-as-drift.  When the
    # benchmark can't reach a model (auth error, rate limit, timeout) it
    # often returns 0.0 for every role.  The drift detector would then see
    # a -100% drop and flood the governance queue with false positives —
    # as happened in April 2026 when 9 Anthropic models simultaneously got
    # "Quality drift" alerts because the bench runner couldn't reach the
    # API.  If the fresh benchmark score is effectively zero across every
    # role, treat it as a benchmark failure instead of a drift signal.
    bench_average = (
        sum(new_scores.values()) / len(new_scores) if new_scores else 0.0
    )
    benchmark_failed = (
        bench_average < 0.05 and all(
            old_scores.get(r, 0.0) > 0.1 for r in new_scores
        )
    )
    if benchmark_failed:
        logger.warning(
            f"llm_discovery: skipping drift alert for {model_name} — "
            f"all new scores ≈0 (benchmark likely failed to reach the model). "
            f"Previous scores {old_scores} preserved in catalog."
        )
        # Don't update strengths either — a failed benchmark should not
        # overwrite the historical estimate with zeros.
        return {
            "model": model_name,
            "old_scores": old_scores,
            "new_scores": new_scores,
            "drift": drift,
            "alerted": False,
            "benchmark_failed": True,
        }

    if worst <= -INCUMBENT_DRIFT_ALERT_THRESHOLD:
        alerted = _raise_drift_alert(model_name, old_scores, new_scores, drift)

    # Persist into discovered_models so drift history is queryable
    try:
        best_role = max(new_scores, key=new_scores.get)
        _upsert_incumbent_benchmark(model_name, entry, new_scores[best_role], best_role)
    except Exception as exc:
        logger.debug(f"rebenchmark: persist failed: {exc}")

    return {
        "model": model_name,
        "old_scores": old_scores,
        "new_scores": new_scores,
        "drift": drift,
        "alerted": alerted,
    }


def _upsert_incumbent_benchmark(
    model_name: str, entry: dict, score: float, role: str,
) -> None:
    """Store a rebenchmark row in discovered_models so drift history is
    queryable. Acts as INSERT when the incumbent has never flowed
    through discovery; UPDATE otherwise.
    """
    try:
        from app.control_plane.db import execute
        execute(
            """
            INSERT INTO control_plane.discovered_models
                   (model_id, provider, display_name, context_window,
                    cost_input_per_m, cost_output_per_m, multimodal,
                    tool_calling, source, raw_metadata, status,
                    benchmark_score, benchmark_role, benchmarked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    'catalog_incumbent', '{}', 'promoted',
                    %s, %s, NOW())
            ON CONFLICT (model_id) DO UPDATE SET
                benchmark_score = EXCLUDED.benchmark_score,
                benchmark_role  = EXCLUDED.benchmark_role,
                benchmarked_at  = NOW(),
                updated_at      = NOW()
            """,
            (
                entry["model_id"], entry.get("provider", "unknown"),
                entry.get("description", model_name)[:100],
                int(entry.get("context", 0)),
                float(entry.get("cost_input_per_m", 0)),
                float(entry.get("cost_output_per_m", 0)),
                bool(entry.get("multimodal", False)),
                bool(entry.get("supports_tools", True)),
                score, role,
            ),
        )
    except Exception as exc:
        logger.debug(f"rebenchmark: upsert incumbent failed: {exc}")


def _raise_drift_alert(
    model_name: str,
    old_scores: dict[str, float],
    new_scores: dict[str, float],
    drift: dict[str, float],
) -> bool:
    """Emit a governance request when a catalog incumbent shows
    significant quality drift. Returns True on success.
    """
    try:
        from app.control_plane.governance import get_governance
        from app.control_plane.projects import get_projects
        gate = get_governance()
        pid = get_projects().get_active_project_id()
        gate.request_approval(
            project_id=pid,
            request_type="incumbent_drift",
            requested_by="llm_discovery",
            title=f"Quality drift detected for {model_name}",
            detail={
                "model": model_name,
                "old_scores": old_scores,
                "new_scores": new_scores,
                "drift": drift,
                "threshold": INCUMBENT_DRIFT_ALERT_THRESHOLD,
            },
        )
        logger.warning(
            f"llm_discovery: DRIFT alert on {model_name} — drift={drift}",
        )
        return True
    except Exception as exc:
        logger.debug(f"rebenchmark: drift alert failed: {exc}")
        return False


def pick_incumbent_to_rebenchmark() -> str | None:
    """Return the catalog key of the next incumbent due for rebenchmark.

    Picks the model with the oldest ``benchmarked_at`` (or never
    benchmarked). Skips discovered entries — those flow through the
    normal discovery pipeline. Returns None when every incumbent has
    been benchmarked within the last week.
    """
    from app.llm_catalog import CATALOG
    try:
        from app.control_plane.db import execute
    except Exception:
        return None

    candidates = [
        name for name, info in CATALOG.items()
        if info.get("tier") in ("budget", "mid", "premium")
        and not info.get("_discovered")
        and not info.get("_retired")  # skip entries marked retired by benchmark
    ]
    if not candidates:
        return None

    # Look up last benchmarked_at for each catalog incumbent
    rows = execute(
        """
        SELECT model_id, benchmarked_at
          FROM control_plane.discovered_models
         WHERE source = 'catalog_incumbent'
        """,
        (),
        fetch=True,
    ) or []
    last_seen = {
        r["model_id"]: r["benchmarked_at"] for r in rows
    }

    # Match catalog entries by model_id
    dated: list[tuple[str, object]] = []
    for name in candidates:
        mid = CATALOG[name].get("model_id", "")
        dated.append((name, last_seen.get(mid)))

    # Never-benchmarked incumbents come first (None sorts as oldest).
    dated.sort(key=lambda x: (x[1] is not None, x[1]))
    return dated[0][0] if dated else None

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

def _dominates_incumbent(model: dict, role: str, cost_mode: str) -> bool:
    """Pareto-style check: does ``model`` outperform the current role
    default on both quality and cost for the given cost_mode?

    Used to decide whether an auto-promotion should take over the role
    assignment in a particular cost mode. A new model wins only if it
    is *both* cheaper-or-equal AND of higher benchmark score than the
    incumbent. For the incumbent's score we read ``strengths[role]``
    first, falling back to ``strengths["general"]`` — never the raw
    0.5 floor, which would let any newcomer with a generic score
    unfairly unseat a strong but non-role-tagged incumbent.
    """
    try:
        from app.llm_catalog import CATALOG, ROLE_DEFAULTS
        mode_defaults = ROLE_DEFAULTS.get(cost_mode, ROLE_DEFAULTS["balanced"])
        incumbent_key = mode_defaults.get(role, mode_defaults.get("default", ""))
        incumbent = CATALOG.get(incumbent_key)
        if not incumbent:
            return True
        incumbent_cost = float(incumbent.get("cost_output_per_m", 0))
        strengths = incumbent.get("strengths", {})
        incumbent_score = float(
            strengths.get(role)
            if role in strengths
            else strengths.get("general", 0.5)
        )
        my_cost = float(model.get("cost_output_per_m", 0))
        my_score = float(model.get("benchmark_score", 0.0))
        return my_cost <= incumbent_cost and my_score > incumbent_score
    except Exception:
        return False


def _add_to_runtime_catalog(model: dict, roles: list[str]) -> None:
    """Add a discovered model to the runtime catalog and (where it
    dominates the incumbent) install role-assignment overlays.

    - Catalog insert mirrors the previous behaviour so in-memory lookups
      succeed on the new model immediately.
    - Overlay writes go to ``control_plane.role_assignments`` so the
      selector picks the new model on its next call, persists across
      restarts, and is rehydrated by ``llm_rehydrate.rehydrate_catalog``.
    """
    from app.llm_catalog import CATALOG

    name = model["model_id"].split("/")[-1] if "/" in model["model_id"] else model["model_id"]

    # Estimate strengths from benchmark
    base_score = model.get("benchmark_score", 0.5) if isinstance(model.get("benchmark_score"), (int, float)) else 0.5
    # Per-role scores come from the benchmark when available; otherwise
    # inherit the base score. `per_role_scores` is set by the multi-role
    # benchmarking path (see run_discovery_cycle).
    per_role = model.get("per_role_scores") or {}
    strengths = {
        r: round(float(per_role.get(r, base_score)), 2)
        for r in roles
    }
    strengths["general"] = round(base_score * 0.9, 2)

    entry = {
        "tier": model.get("tier", "budget"),
        "provider": model.get("provider", "openrouter"),
        "model_id": model["model_id"],
        "context": model.get("context_window", 32768),
        "multimodal": model.get("multimodal", False),
        "cost_input_per_m": model.get("cost_input_per_m", 0),
        "cost_output_per_m": model.get("cost_output_per_m", 0),
        "tool_use_reliability": 0.80 if model.get("tool_calling", False) else 0.0,
        "supports_tools": bool(model.get("tool_calling", True)),
        "description": f"Auto-discovered: {model.get('display_name', name)}",
        "strengths": strengths,
        "_discovered": True,  # Marker for dynamic models
    }

    # Add to catalog (runtime only — not persisted to .py file)
    CATALOG[name] = entry
    logger.info(f"llm_discovery: added {name} to runtime catalog (tier={entry['tier']})")

    # Install role-assignment overlays for cost modes where the new
    # model Pareto-dominates the incumbent. Skips the write quietly if
    # the DB is unreachable — next discovery cycle will retry.
    try:
        from app.llm_role_assignments import set_assignment
        cost_modes = ("budget", "balanced", "quality")
        for role in roles:
            role_score = float(per_role.get(role, base_score))
            dominated_modes = [
                m for m in cost_modes if _dominates_incumbent(
                    {**model, "benchmark_score": role_score}, role, m,
                )
            ]
            for mode in dominated_modes:
                set_assignment(
                    role=role, cost_mode=mode, model=name,
                    source="auto_promotion",
                    reason=(
                        f"bench={role_score:.2f} "
                        f"${model.get('cost_output_per_m', 0):.3f}/Mo "
                        f"dominates default in {mode} mode"
                    ),
                    assigned_by="llm_discovery",
                    priority=150,
                )
    except Exception as exc:
        logger.debug(f"llm_discovery: role overlay write failed: {exc}")

# ── Main Pipeline ────────────────────────────────────────────────────────────

def _benchmark_all_roles(model_id: str, sample_size: int = 2) -> dict[str, float]:
    """Run the discovery benchmark across every role in BENCHMARK_ROLES.

    Returns a ``{role: score}`` map for roles that produced a valid
    score. A negative return from ``benchmark_model`` is treated as a
    skip rather than a zero (so transient judge errors don't kill a
    model's chances).
    """
    scores: dict[str, float] = {}
    for role in BENCHMARK_ROLES:
        s = benchmark_model(model_id, role=role, sample_size=sample_size)
        if s >= 0:
            scores[role] = s
    return scores


def run_discovery_cycle(max_benchmarks: int = 3) -> dict:
    """Full discovery pipeline. Called by idle scheduler.

    Returns: {scanned, new_found, benchmarked, promoted, proposals}
    """
    result = {
        "scanned": 0, "new_found": 0, "benchmarked": 0,
        "promoted": 0, "proposals": 0, "errors": [],
    }

    # Step 1: Scan sources — OpenRouter remote + local Ollama
    raw_openrouter = scan_openrouter()
    raw_ollama = scan_ollama()
    result["scanned"] = len(raw_openrouter) + len(raw_ollama)

    if not raw_openrouter and not raw_ollama:
        return result

    # Step 2: Filter + normalize (per provider — keeps the catalog key
    # prefix logic consistent)
    known_ids = _get_known_model_ids()
    catalog_ids = _get_catalog_model_ids()

    new_models: list[dict] = []
    for raw in raw_openrouter:
        normalized = _normalize_model(raw, provider="openrouter")
        if not normalized:
            continue
        mid = normalized["model_id"]
        if mid not in known_ids and mid not in catalog_ids:
            new_models.append(normalized)
    for raw in raw_ollama:
        normalized = _normalize_model(raw, provider="ollama")
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

    # Step 4: Benchmark top candidates across every BENCHMARK_ROLE.
    # Sort by cost (prefer cheap) then by context window (prefer large).
    candidates = sorted(
        new_models,
        key=lambda m: (m["cost_output_per_m"], -m["context_window"]),
    )[:max_benchmarks]

    for model in candidates:
        per_role = _benchmark_all_roles(model["model_id"])
        if not per_role:
            result["errors"].append(f"Benchmark failed for {model['model_id']}")
            continue

        # Best-scoring role acts as the primary benchmark for the
        # legacy single-column discovered_models row. The full score
        # map travels alongside the model dict into promotion logic.
        best_role, best_score = max(per_role.items(), key=lambda kv: kv[1])
        _update_benchmark(model["model_id"], best_score, best_role)
        model["benchmark_score"] = best_score
        model["benchmark_role"] = best_role
        model["per_role_scores"] = per_role
        result["benchmarked"] += 1

        # Step 5: Propose promotion for EACH role where the model
        # actually outperforms the incumbent. Free models auto-promote;
        # others queue a governance request.
        for role, score in per_role.items():
            proposal = propose_promotion(model, score, role)
            if proposal:
                if proposal.get("status") == "auto_promoted":
                    result["promoted"] += 1
                else:
                    result["proposals"] += 1

    # Step 6: Scan the upstream Ollama registry for available-but-not-pulled
    # variants of tracked families. Surfaces models like the qwen3.5 release
    # that scan_ollama() can't see (it only lists already-pulled tags).
    # Generates governance proposals — never auto-pulls (pulls are 5-50 GB).
    try:
        registry_proposals = _scan_registry_and_propose(raw_ollama)
        result["registry_proposals"] = registry_proposals
        if registry_proposals:
            result["proposals"] += registry_proposals
            logger.info(
                f"llm_discovery: registry scan surfaced {registry_proposals} "
                f"pull candidates"
            )
    except Exception as exc:
        logger.debug(f"llm_discovery: registry scan failed: {exc}", exc_info=True)
        result.setdefault("errors", []).append(f"registry_scan: {exc}")

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


def _scan_registry_and_propose(local_ollama_raw: list[dict]) -> int:
    """Scan ollama.com for new variants and emit pull proposals.

    Returns the number of governance requests created. Local-tag dedupe
    uses the same name field that scan_ollama() pulls from /api/tags so
    a candidate already on disk is never re-proposed.

    Idempotent: skips candidates that already have an open governance
    request of type ``local_model_pull`` for the same model.
    """
    from app.llm_registry_scanner import (
        scan_ollama_registry,
        diff_against_local,
        filter_dominated_by_installed,
        filter_quant_dominated,
        filter_recently_rejected,
    )
    candidates = scan_ollama_registry()
    if not candidates:
        return 0

    local_names = [m.get("name", "") for m in local_ollama_raw]
    new_candidates = diff_against_local(candidates, local_names)
    if not new_candidates:
        logger.debug("registry_scan: no new candidates beyond local /api/tags")
        return 0

    # Three layered filters (added 2026-04-30 after the user rejected 9/9
    # proposals in two idle cycles — same-family smaller siblings of an
    # already-installed larger model). Order matters: cheapest checks first.
    pre = len(new_candidates)
    new_candidates = filter_dominated_by_installed(new_candidates, local_names)
    new_candidates = filter_quant_dominated(new_candidates, local_names)
    new_candidates = filter_recently_rejected(new_candidates)
    if len(new_candidates) != pre:
        logger.info(
            "registry_scan: filtered %d → %d after dominance/quant/rejection checks",
            pre, len(new_candidates),
        )
    if not new_candidates:
        logger.debug(
            "registry_scan: all %d candidates filtered out (dominated, quant-dominated, or recently rejected)",
            pre,
        )
        return 0

    # Dedupe against existing open proposals so an idle cycle that runs
    # every few minutes doesn't flood the governance queue.
    pending_models = _existing_pull_proposal_models()

    proposals_made = 0
    # Cap at top 3 per cycle so a fresh-install scan doesn't dump 50
    # proposals at once. The user can re-run discovery to see more.
    for cand in new_candidates[:3]:
        if cand.full_name in pending_models:
            continue
        if _create_pull_proposal(cand):
            proposals_made += 1
    return proposals_made


def _existing_pull_proposal_models() -> set[str]:
    """Set of model names that already have an open ``local_model_pull``
    governance request — used to dedupe across cycles."""
    try:
        from app.control_plane.db import execute
        rows = execute(
            """SELECT detail_json FROM control_plane.governance_requests
               WHERE request_type = %s AND status = 'pending'""",
            ("local_model_pull",),
        ) or []
        seen: set[str] = set()
        for r in rows:
            detail = r.get("detail_json") or {}
            if isinstance(detail, str):
                import json as _j
                try:
                    detail = _j.loads(detail)
                except Exception:
                    detail = {}
            name = detail.get("model") or detail.get("model_id") or ""
            if name:
                seen.add(name)
        return seen
    except Exception:
        return set()


def _create_pull_proposal(cand) -> bool:
    """File a governance approval request for one registry candidate.

    Returns True on success. Errors are logged but not raised — registry
    scan is a best-effort enhancer, not a gate.

    The proposal detail includes a host-capacity probe result so the
    user sees the safety budget that was used to decide this model
    fits — relevant context after the 2026-04-25 SIGKILL spiral that
    was triggered by exactly this kind of "is it safe to load this?"
    question being answered with a hardcoded constant.
    """
    try:
        from app.control_plane.governance import get_governance
        from app.control_plane.projects import get_projects
        from app.llm_registry_scanner import probe_host_capacity
        gate = get_governance()
        pid = get_projects().get_active_project_id()
        capacity = probe_host_capacity()

        feature_summary = ", ".join(cand.features) if cand.features else "standard"
        # Annotate title with fit verdict so users can scan the queue
        # at a glance ("comfortable" vs "marginal").
        fit_marker = ""
        if capacity:
            cap = capacity.max_model_size_gb
            if cand.size_gb <= cap * 0.75:
                fit_marker = " ✓"
            elif cand.size_gb <= cap:
                fit_marker = " ~"
        title = (
            f"Pull local model: {cand.full_name} "
            f"({cand.size_gb:.1f} GB, {feature_summary}){fit_marker}"
        )
        gate.request_approval(
            project_id=pid,
            request_type="local_model_pull",
            requested_by="llm_discovery",
            title=title,
            detail=cand.to_proposal_detail(capacity),
        )
        logger.info(f"registry_scan: filed pull proposal for {cand.full_name}")
        return True
    except Exception as exc:
        logger.debug(
            f"registry_scan: proposal for {cand.full_name} failed: {exc}",
            exc_info=True,
        )
        return False


# ── Governance consumer ──────────────────────────────────────────────────────

def consume_approved_promotions(limit: int = 10) -> dict:
    """Apply every governance request approved since the last run.

    ``propose_promotion`` files a ``model_promotion`` governance request
    for non-free candidates. A human approves the request in the
    dashboard / Signal. This function consumes those approvals:
      - adds the model to the runtime catalog (if not already)
      - installs role assignment overlays
      - marks the discovered_models row as promoted
      - marks the governance_requests row as consumed

    Returns a summary dict suitable for logging/Signal display.
    """
    summary = {"applied": 0, "skipped": 0, "errors": 0}
    try:
        from app.control_plane.db import execute
    except Exception:
        return summary

    rows = execute(
        """
        SELECT id, detail_json, reviewed_at, reviewed_by
          FROM control_plane.governance_requests
         WHERE request_type = 'model_promotion'
           AND status = 'approved'
           AND consumed_at IS NULL
      ORDER BY reviewed_at ASC
         LIMIT %s
        """,
        (limit,),
        fetch=True,
    ) or []

    if not rows:
        return summary

    for row in rows:
        try:
            detail = row.get("detail_json") or {}
            if isinstance(detail, str):
                detail = json.loads(detail)
            model_id = detail.get("model_id")
            role = detail.get("role") or "research"
            tier = detail.get("tier", "budget")
            if not model_id:
                summary["skipped"] += 1
                continue

            # Pull the full discovered row so _add_to_runtime_catalog
            # gets real cost/context/multimodal metadata rather than
            # only the governance detail blob.
            disc = execute(
                """
                SELECT model_id, provider, display_name, context_window,
                       cost_input_per_m, cost_output_per_m, multimodal,
                       tool_calling, benchmark_score
                  FROM control_plane.discovered_models
                 WHERE model_id = %s
                """,
                (model_id,),
                fetch=True,
            ) or []
            if not disc:
                summary["skipped"] += 1
                continue
            disc_row = disc[0]

            model_payload = {
                "model_id":          disc_row["model_id"],
                "provider":          disc_row.get("provider", "openrouter"),
                "display_name":      disc_row.get("display_name", model_id),
                "context_window":    int(disc_row.get("context_window") or 0),
                "cost_input_per_m":  float(disc_row.get("cost_input_per_m") or 0),
                "cost_output_per_m": float(disc_row.get("cost_output_per_m") or 0),
                "multimodal":        bool(disc_row.get("multimodal")),
                "tool_calling":      bool(disc_row.get("tool_calling")),
                "tier":              tier,
                "benchmark_score":   float(disc_row.get("benchmark_score") or 0.5),
            }
            _add_to_runtime_catalog(model_payload, [role])
            _promote_model(
                model_id, tier, [role],
                reviewer=row.get("reviewed_by") or "governance",
            )

            # Three-layer authority wiring:
            #   - discovered_models.status = 'promoted' (above)
            #   - model_promotions row     (below) → resolver prefers it
            #   - rehydrate_catalog pulls the entry into live CATALOG so
            #     the resolver sees it on the very next selection.
            catalog_key = (
                model_id.split("/")[-1] if "/" in model_id else model_id
            )
            try:
                from app.llm_promotions import promote
                promote(
                    catalog_key,
                    promoted_by=f"governance:{row.get('reviewed_by') or 'user'}",
                    reason=f"approved for {role} via governance request {row.get('id')}",
                )
            except Exception as exc:
                logger.debug(f"llm_discovery: model_promotions insert failed: {exc}")

            try:
                from app.llm_rehydrate import rehydrate_catalog
                rehydrate_catalog(force=True)
            except Exception as exc:
                logger.debug(f"llm_discovery: rehydrate after promotion failed: {exc}")

            execute(
                """
                UPDATE control_plane.governance_requests
                   SET consumed_at = NOW()
                 WHERE id = %s
                """,
                (row["id"],),
            )
            summary["applied"] += 1
            logger.info(
                "llm_discovery: applied governance promotion model=%s role=%s tier=%s",
                model_id, role, tier,
            )
        except Exception as exc:
            summary["errors"] += 1
            logger.warning(f"llm_discovery: promotion consumer failed on row {row.get('id')}: {exc}")

    return summary

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
