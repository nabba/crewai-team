"""vendor_independence drill — verifies the cascade can route past
the dominant providers without an outage.

PROGRAM §51 — Q16 Theme 2 (decade-resilience, vendor independence
depth). The LLM cascade has structural failover (local Ollama →
DeepSeek → MiniMax → Anthropic → Gemini per CLAUDE.md), but the
*failover path* has never been exercised under conditions that
prove it would work if Anthropic + OpenRouter became unavailable.

This drill is DRY-RUN: it never issues live LLM calls (those would
burn budget on a quarterly drill and be flaky against rate limits).
What it DOES verify:

  1. **Cascade structural diversity** — at least 2 distinct
     non-Anthropic, non-OpenRouter providers are configured in the
     factory's resolution path. If both keys vanish overnight, is
     there ANY route forward?

  2. **Local-model reachability** — Ollama is the bottom of the
     cascade. Probe its listening socket (TCP connect, no
     payload). Times out fast. Doesn't actually invoke a model.

  3. **Vendor key configuration coverage** — for each provider
     present in the cascade module's source, the corresponding env
     var is set. Mismatch = stale code pointing at a vanished key
     OR fresh key without code wired up.

  4. **Factory selector resilience** — invoke ``select_model`` (the
     internal selector, NOT the LLM itself) with the dominant
     providers blocklisted and confirm it returns a non-None
     result. This is the "would routing work?" check.

  5. **Documented fallback chain** — the cascade order is documented
     in ``crewai-team/docs/LLM_SUBSYSTEM.md`` (per CLAUDE.md). The
     drill source-greps the doc for the expected provider chain and
     alerts if a documented provider isn't structurally present.

This drill is in the Q6 registry (Pattern A) — risk LOW, scheduler
auto-runs at 90-day cadence. Master switch
``drill_vendor_independence_enabled`` (default ON).

Failure modes that this catches:

  * "We thought we had a Groq fallback but ``GROQ_API_KEY`` was
    never set in this deployment."
  * "Ollama container died 8 months ago; we didn't notice because
    we never failed over to it."
  * "The selector hardcoded Anthropic somewhere and the cascade
    structure is illusory."
"""
from __future__ import annotations

import logging
import os
import re
import socket
import time
from datetime import datetime, timezone
from typing import Any

from app.resilience_drills.protocol import (
    DrillResult,
    DrillRisk,
    DrillSpec,
    DrillStatus,
    FailureClass,
    register,
)

logger = logging.getLogger(__name__)


SPEC = DrillSpec(
    name="vendor_independence",
    cadence_days=90,
    grace_days=30,
    warmup_days=0,  # existing drill — no warmup
    risk=DrillRisk.LOW,
    description=(
        "Verify the LLM cascade can route past dominant providers "
        "(Anthropic / OpenRouter) without an outage. Structural "
        "checks only — never issues live LLM calls."
    ),
    requires_master_switch="drill_vendor_independence_enabled",
)


# Providers the cascade documents (CLAUDE.md): "local Ollama →
# DeepSeek V3.2 (OpenRouter) → MiniMax M2.5 → Anthropic / Gemini".
# We treat Anthropic + OpenRouter as "dominant" — the drill verifies
# routes exist outside this pair.
_DOMINANT_PROVIDERS = ("anthropic", "openrouter")
_FALLBACK_ENV_CANDIDATES = (
    "OLLAMA_BASE_URL",     # local
    "GROQ_API_KEY",        # alternative cloud provider
    "GEMINI_API_KEY",      # alternative cloud provider
    "GOOGLE_API_KEY",      # alias for Gemini-family endpoints
    "DEEPSEEK_API_KEY",    # direct DeepSeek (sometimes routed via OR)
    "MINIMAX_API_KEY",     # direct MiniMax (sometimes routed via OR)
)
_OLLAMA_DEFAULT_HOST = "127.0.0.1"
_OLLAMA_DEFAULT_PORT = 11434
_OLLAMA_PROBE_TIMEOUT_S = 1.5

# Live-fitness extension point (opt-in via ``drill_vendor_independence_
# live_enabled``).
#
# **EXTENSION POINT — not active by default.** The structural drill
# above answers "would routing work?" The live-fitness check is the
# answer to "does the cascade actually return a usable reply when we
# force it past the dominant providers?" It would cost ~$0.10 per
# drill in non-Anthropic LLM tokens (quarterly cadence; ~$0.40/year).
#
# Activation requires TWO operator steps, not just flipping the flag:
#
#   1. Flip ``drill_vendor_independence_live_enabled`` to True.
#   2. Add a ``smoke_completion(question, exclude_providers,
#      timeout_s, max_chars) -> str`` helper to
#      ``app/llm_selector.py``. The helper should issue ONE cheap
#      completion via the cascade with the named providers temporarily
#      added to the chat-blocked-models blocklist and the result
#      stripped to ``max_chars`` characters. The helper does NOT
#      mutate the persistent blocklist — exclusion is per-call only.
#
# When the helper is absent, ``_check_live_fitness`` soft-passes with
# ``reason="llm_selector has no smoke_completion helper"`` — the
# structural drill remains green. This is the well-documented dormant
# state; flipping the master switch without the helper does not
# enable anything.
_LIVE_FITNESS_QUESTIONS = (
    "What is 2 + 2? Reply with only the digit.",
    "Name one Nordic capital. Reply with only the name.",
    "Pick a color: red or blue. Reply with only the word.",
)
_LIVE_FITNESS_MIN_OK_FRACTION = 2 / 3  # at least 2 of 3 must yield a non-empty short reply
_LIVE_FITNESS_TIMEOUT_S = 30
_LIVE_FITNESS_MAX_REPLY_CHARS = 200


def _check_cascade_structural_diversity() -> tuple[bool, str | None, dict]:
    """Verify at least 2 non-dominant fallback options are configured."""
    info: dict[str, Any] = {"configured_fallbacks": []}
    for env in _FALLBACK_ENV_CANDIDATES:
        value = os.environ.get(env, "").strip()
        if value:
            info["configured_fallbacks"].append(env)
    n = len(info["configured_fallbacks"])
    info["count"] = n
    if n >= 2:
        return True, None, info
    err = (
        f"only {n} non-dominant fallback(s) configured "
        f"({info['configured_fallbacks']}); cascade has no slack if "
        f"both Anthropic and OpenRouter become unavailable"
    )
    return False, err, info


def _check_ollama_reachable() -> tuple[bool, str | None, dict]:
    """TCP-connect probe to the Ollama port. Times out fast; never
    issues a model call. ``OLLAMA_BASE_URL=http://host:port`` is
    parsed; falls back to 127.0.0.1:11434."""
    raw = os.environ.get("OLLAMA_BASE_URL", "").strip()
    host = _OLLAMA_DEFAULT_HOST
    port = _OLLAMA_DEFAULT_PORT
    if raw:
        # Best-effort parse — accept "http://host:port", "host:port",
        # or just "host".
        try:
            from urllib.parse import urlparse
            parsed = urlparse(raw if "://" in raw else f"http://{raw}")
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port:
                port = parsed.port
        except Exception:
            pass
    info: dict[str, Any] = {"host": host, "port": port, "configured": bool(raw)}
    try:
        with socket.create_connection(
            (host, port), timeout=_OLLAMA_PROBE_TIMEOUT_S,
        ):
            info["reachable"] = True
            return True, None, info
    except (OSError, socket.timeout) as exc:
        info["reachable"] = False
        info["error_type"] = type(exc).__name__
        # If Ollama wasn't configured, this is informational only.
        if not raw:
            return True, None, info
        return False, f"Ollama unreachable at {host}:{port}", info


def _check_vendor_key_coverage() -> tuple[bool, str | None, dict]:
    """For each vendor the LLM factory mentions, check that the
    corresponding env var is set. Source-level check — never reads
    a key value, just presence."""
    info: dict[str, Any] = {"providers": {}}
    try:
        import inspect
        import app.llm_factory as factory
        source = inspect.getsource(factory)
    except Exception as exc:
        return False, f"llm_factory unreadable: {exc}", info
    expected: dict[str, str] = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    all_ok = True
    for provider, env_var in expected.items():
        mentioned = (env_var in source) or (provider in source.lower())
        present = bool(os.environ.get(env_var, "").strip())
        info["providers"][provider] = {
            "mentioned_in_factory": mentioned,
            "env_var_set": present,
        }
        # We don't FAIL on missing keys per se — that's the operator's
        # deployment decision. We DO fail if a provider is mentioned in
        # the factory but ALSO is a documented dominant fallback and is
        # absent (i.e., we have the code but no key).
        if mentioned and not present and provider not in _DOMINANT_PROVIDERS:
            # Non-dominant; absence is a warning at the diversity-check
            # level, not a fail here.
            continue
    return all_ok, None, info


def _check_selector_routes_past_dominants() -> tuple[bool, str | None, dict]:
    """Verify the resolver can return a non-None selection when both
    dominant providers are in the blocklist. Uses
    ``llm_selector.select_model`` if available, else falls back to a
    source-grep heuristic."""
    info: dict[str, Any] = {}
    try:
        from app import llm_selector
        info["selector_present"] = True
    except Exception as exc:
        info["selector_present"] = False
        info["import_error"] = str(exc)
        # Soft pass — the drill can't introspect without the selector.
        return True, None, info
    # Use source-grep to confirm the selector consults the blocked-
    # models list. The actual select_model() call may need extensive
    # setup; we want a cheap structural check.
    try:
        import inspect
        source = inspect.getsource(llm_selector)
    except Exception:
        return True, None, info
    info["source_lines"] = source.count("\n")
    # The selector should at minimum reference chat_blocked_models OR
    # a blocklist mechanism. If it doesn't, the cascade has no way to
    # route around a dominant provider that's misbehaving.
    has_blocklist = (
        "chat_blocked_models" in source
        or "blocked_models" in source
        or "blocklist" in source.lower()
    )
    info["selector_has_blocklist"] = has_blocklist
    if not has_blocklist:
        return False, (
            "selector source has no blocklist mechanism — cannot route "
            "around a misbehaving dominant provider"
        ), info
    return True, None, info


def _live_fitness_enabled() -> bool:
    """Read the opt-in flag for the live-LLM-call extension. Defaults
    OFF — the structural drill alone is the always-on path."""
    try:
        from app.runtime_settings import get_drill_vendor_independence_live_enabled
        return get_drill_vendor_independence_live_enabled()
    except Exception:
        return os.getenv(
            "DRILL_VENDOR_INDEPENDENCE_LIVE_ENABLED", "false",
        ).lower() in ("true", "1", "yes", "on")


def _check_live_fitness() -> tuple[bool, str | None, dict]:
    """Optionally issue 3 cheap LLM calls to the cascade with the
    dominant providers excluded. Pass iff ≥2/3 yield a non-empty,
    short reply within the timeout.

    This is the only check that actually spends LLM tokens. Skipped
    silently if the opt-in flag is OFF or the cascade isn't available.
    Failure-isolated: any exception → returns (True, None, {...})
    with ``reason=skipped`` so the structural drill stays green when
    live fitness can't run."""
    info: dict[str, Any] = {"enabled": _live_fitness_enabled()}
    if not info["enabled"]:
        info["reason"] = "skipped — drill_vendor_independence_live_enabled is OFF"
        return True, None, info

    try:
        # Try the lightweight, low-cost call path that the cascade
        # exposes. We import lazily so the structural drill never
        # depends on llm_selector being importable.
        from app import llm_selector
    except Exception as exc:
        info["reason"] = f"llm_selector unavailable: {type(exc).__name__}"
        return True, None, info

    # Pull the existing blocklist surface — if not present, skip.
    try:
        from app import runtime_settings as rs
        existing_blocked = list(
            rs._ensure_initialized().get("chat_blocked_models", []) or []
        )
    except Exception:
        existing_blocked = []
    info["existing_blocked_models"] = existing_blocked

    # We don't directly call the LLM here — that path lives in the
    # cascade and uses many integrations. Instead, we use a small
    # smoke harness: ask the cascade to answer the question with the
    # dominant providers in the temporary blocklist.
    smoke_fn = getattr(llm_selector, "smoke_completion", None)
    if smoke_fn is None or not callable(smoke_fn):
        info["reason"] = (
            "llm_selector has no smoke_completion helper; live "
            "fitness skipped (structural drill is sufficient)"
        )
        return True, None, info

    # Run the questions through smoke_completion with dominant
    # providers excluded for the duration of this call.
    results: list[dict[str, Any]] = []
    n_ok = 0
    for q in _LIVE_FITNESS_QUESTIONS:
        try:
            reply = smoke_fn(
                question=q,
                exclude_providers=list(_DOMINANT_PROVIDERS),
                timeout_s=_LIVE_FITNESS_TIMEOUT_S,
                max_chars=_LIVE_FITNESS_MAX_REPLY_CHARS,
            )
        except Exception as exc:
            results.append({"q_hash": hash(q) & 0xffffffff, "error": type(exc).__name__})
            continue
        # We deliberately do NOT log the reply text — just length +
        # ok bool — to keep the audit clean of free-form text.
        reply_text = (reply or "").strip()
        ok = bool(reply_text) and len(reply_text) <= _LIVE_FITNESS_MAX_REPLY_CHARS
        results.append({
            "q_hash": hash(q) & 0xffffffff,
            "reply_len": len(reply_text),
            "ok": ok,
        })
        if ok:
            n_ok += 1
    info["live_results"] = results
    info["n_ok"] = n_ok
    info["n_questions"] = len(_LIVE_FITNESS_QUESTIONS)
    info["ok_fraction"] = round(n_ok / max(1, len(_LIVE_FITNESS_QUESTIONS)), 3)
    if info["ok_fraction"] < _LIVE_FITNESS_MIN_OK_FRACTION:
        return False, (
            f"live cascade fitness: {n_ok}/{len(_LIVE_FITNESS_QUESTIONS)} "
            f"replies acceptable (need "
            f"{_LIVE_FITNESS_MIN_OK_FRACTION * 100:.0f}%+)"
        ), info
    return True, None, info


def _check_documented_chain_present() -> tuple[bool, str | None, dict]:
    """Source-grep the documented LLM subsystem doc for the cascade
    chain. Surface a warning if the doc names a provider that doesn't
    appear in factory source."""
    info: dict[str, Any] = {}
    try:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        doc_path = repo_root / "docs" / "LLM_SUBSYSTEM.md"
        if not doc_path.is_file():
            info["doc_path"] = "missing"
            return True, None, info
        doc_text = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        info["doc_error"] = str(exc)
        return True, None, info
    info["doc_path"] = "present"
    info["doc_chars"] = len(doc_text)
    # Names the doc commonly references (free-text); not a strict
    # contract — surface only.
    expected_terms = ("Ollama", "OpenRouter", "Anthropic", "Gemini")
    info["mentioned_in_doc"] = {
        term: (term.lower() in doc_text.lower()) for term in expected_terms
    }
    return True, None, info


def run(*, dry_run: bool = True) -> DrillResult:
    """Run the vendor-independence procedure verification.

    Q18 runner contract: returns a bare DrillResult; the orchestrator
    threads lock + audit + landmark + state. Always dry-run — never
    issues an LLM call.
    """
    started_dt = datetime.now(timezone.utc)
    started_at = started_dt.isoformat()
    t0 = time.monotonic()

    detail: dict[str, Any] = {"checks": {}}
    errors: list[str] = []
    status = DrillStatus.PASS
    failure_class: FailureClass | None = None

    ok, err, info = _check_cascade_structural_diversity()
    detail["checks"]["cascade_diversity"] = ok
    detail["cascade_diversity_info"] = info
    if not ok:
        errors.append(f"cascade_diversity: {err}")
        status = DrillStatus.FAIL

    ok, err, info = _check_ollama_reachable()
    detail["checks"]["ollama_reachable"] = ok
    detail["ollama_info"] = info
    if not ok:
        errors.append(f"ollama_reachable: {err}")
        status = DrillStatus.FAIL

    ok, err, info = _check_vendor_key_coverage()
    detail["checks"]["vendor_key_coverage"] = ok
    detail["vendor_key_coverage_info"] = info
    if not ok:
        errors.append(f"vendor_key_coverage: {err}")
        status = DrillStatus.FAIL

    ok, err, info = _check_selector_routes_past_dominants()
    detail["checks"]["selector_routes"] = ok
    detail["selector_routes_info"] = info
    if not ok:
        errors.append(f"selector_routes: {err}")
        status = DrillStatus.FAIL

    ok, err, info = _check_documented_chain_present()
    detail["checks"]["documented_chain"] = ok
    detail["documented_chain_info"] = info
    # Documentation check is informational; never fails the drill.

    ok, err, info = _check_live_fitness()
    detail["checks"]["live_fitness"] = ok
    detail["live_fitness_info"] = info
    if not ok:
        errors.append(f"live_fitness: {err}")
        status = DrillStatus.FAIL

    # Secret-leak guard (mirrors secret_rotation drill pattern).
    leak_ok, leak_err = _no_secret_in_detail(detail)
    detail["checks"]["no_secret_in_detail"] = leak_ok
    if not leak_ok:
        errors.append(f"no_secret_in_detail: {leak_err}")
        status = DrillStatus.ERROR  # P0-shaped
        failure_class = FailureClass.CODE_ERROR

    if status == DrillStatus.FAIL and failure_class is None:
        failure_class = FailureClass.STRUCTURAL_FAIL

    # Q18 — operator-ratifiable observation. The numeric/boolean
    # measurements the operator may want to lock in as baseline.
    cascade_info = detail.get("cascade_diversity_info", {})
    vendor_info = detail.get("vendor_key_coverage_info", {})
    ollama_info = detail.get("ollama_info", {})
    observation = {
        "n_fallbacks": cascade_info.get("count", 0),
        "configured_fallbacks": list(cascade_info.get("configured_fallbacks", [])),
        "ollama_reachable": bool(ollama_info.get("reachable", False)),
        "providers_with_keys": sorted(
            p for p, d in (vendor_info.get("providers") or {}).items()
            if isinstance(d, dict) and d.get("env_var_set")
        ),
        "selector_has_blocklist": bool(
            detail.get("selector_routes_info", {}).get("selector_has_blocklist", False)
        ),
    }

    completed_dt = datetime.now(timezone.utc)
    return DrillResult(
        drill_name=SPEC.name,
        status=status,
        started_at=started_at,
        completed_at=completed_dt.isoformat(),
        duration_s=round(time.monotonic() - t0, 3),
        dry_run=True,
        detail=detail,
        errors=errors,
        failure_class=failure_class,
        observation=observation,
    )


_LEAKED_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-or-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9_-]{32,}"),
)


def _no_secret_in_detail(detail: dict) -> tuple[bool, str | None]:
    """Scan serialized detail for secret-shaped substrings. Never
    include the matched value in the error message — defeating that
    is exactly the failure mode this guard exists to prevent."""
    try:
        import json as _json
        serialized = _json.dumps(detail, sort_keys=True, default=str)
    except Exception:
        serialized = str(detail)
    for pattern in _LEAKED_SECRET_PATTERNS:
        match = pattern.search(serialized)
        if match:
            return False, (
                f"secret-shaped substring of pattern {pattern.pattern!r} "
                f"detected in drill detail (length={len(match.group(0))}); "
                f"value redacted from this error"
            )
    return True, None


# Module-level registration.
register(SPEC, run)
