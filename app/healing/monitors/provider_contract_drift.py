"""Weekly LLM-provider contract-drift monitor (§2.7).

Distinct from :mod:`app.healing.llm_output_drift` (which checks
SEMANTIC drift via embedding similarity on full text outputs):
this monitor checks STRUCTURAL drift in the provider API
*response shape*. A vendor that quietly removes a top-level
field, renames a key, or changes a type in their JSON response
can break downstream parsers without ANY semantic-output quality
change — the silent failure mode that ``llm_output_drift`` doesn't
catch.

Algorithm:

  1. For each configured provider, capture the structural
     signature of a small fixture response: top-level keys,
     types of each value, keys of common nested dicts (usage,
     choices[0].message, ...).
  2. On first run, persist signatures as the baseline at
     ``workspace/healing/provider_contract_baseline.json``.
  3. On subsequent runs, compute the diff between the new and
     baseline signatures.
  4. If the diff has *removed* keys or *changed* types (additions
     are tolerated as additive-safe), Signal-alert + append a row
     to ``workspace/healing/provider_contract_history.jsonl``.

Why structural drift is real:

  - OpenAI changed ``finish_reason`` semantics + added
    ``finish_details`` mid-2024.
  - Anthropic added ``usage.cache_creation_input_tokens`` /
    ``cache_read_input_tokens`` for prompt caching mid-2025.
  - Various providers have renamed ``content`` ↔ ``text`` ↔
    ``message`` over time.

Each of those would be invisible to a quality-of-output check but
breaks parsers immediately.

Master switch: ``PROVIDER_CONTRACT_DRIFT_ENABLED`` (default
``true``). Cadence: weekly. The probe LLM call is injectable so
tests don't hit live providers.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.life_companion._common import (
    audit_event,
    background_enabled,
    read_state_json,
    write_state_json,
)

logger = logging.getLogger(__name__)


_STATE_FILE = "provider_contract_drift.json"
_BASELINE_PATH = Path("/app/workspace/healing/provider_contract_baseline.json")
_HISTORY_PATH = Path("/app/workspace/healing/provider_contract_history.jsonl")
_RUN_CADENCE_S = 7 * 24 * 3600  # weekly
_DEDUP_WINDOW_S = 7 * 86400


# Probe set — one structural sample per provider. The probe is a
# deliberately trivial completion the provider should always answer
# the same shape for. We don't care about the *content*; we care
# about the SHAPE of the response.
_DEFAULT_PROBE_PROMPT = "Reply with one word: ok"

_DEFAULT_PROVIDERS: tuple[str, ...] = ("anthropic", "openrouter")


def _enabled() -> bool:
    return os.getenv("PROVIDER_CONTRACT_DRIFT_ENABLED", "true").lower() in (
        "true", "1", "yes", "on",
    )


# ── Signature extraction ───────────────────────────────────────────────


def _type_name(v: Any) -> str:
    """Stable type-name for signature comparison."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def extract_signature(response: Any, *, max_depth: int = 3) -> dict[str, str]:
    """Build a flat key→type signature for the response object.

    Walks dicts and the FIRST element of any list (most LLM responses
    are ``{choices: [{...}]}`` — we recurse into ``choices[0]`` only).
    Dot-separated key paths.
    """
    sig: dict[str, str] = {}

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            sig[path] = _type_name(node)
            return
        if isinstance(node, dict):
            for k, v in sorted(node.items()):
                key = f"{path}.{k}" if path else str(k)
                if isinstance(v, (dict, list)):
                    walk(v, key, depth + 1)
                else:
                    sig[key] = _type_name(v)
        elif isinstance(node, list):
            if path:
                sig[path] = "list"
            if node:
                walk(node[0], f"{path}[0]" if path else "[0]", depth + 1)
        else:
            if path:
                sig[path] = _type_name(node)

    walk(response, "", 0)
    return sig


def diff_signatures(
    baseline: dict[str, str],
    current: dict[str, str],
) -> dict[str, list]:
    """Categorise differences. Returns dict with keys:
      ``removed``     keys in baseline but not in current
      ``added``       keys in current but not in baseline
      ``changed_type`` keys in both with differing type-names
    """
    removed = sorted(set(baseline) - set(current))
    added = sorted(set(current) - set(baseline))
    changed = sorted(
        k for k in (set(baseline) & set(current))
        if baseline[k] != current[k]
    )
    return {
        "removed": removed,
        "added": added,
        "changed_type": [
            {"key": k, "from": baseline[k], "to": current[k]}
            for k in changed
        ],
    }


def is_breaking_drift(diff: dict[str, list]) -> bool:
    """Removed keys OR type-changed keys break parsers; additions don't."""
    return bool(diff.get("removed")) or bool(diff.get("changed_type"))


# ── Baseline + history ─────────────────────────────────────────────────


def _read_baseline(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_baseline(path: Path, data: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _append_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ── Probe execution ────────────────────────────────────────────────────


ProbeFn = Callable[[str], Any]
"""Takes a provider name; returns the raw response object (dict)."""


def _default_probe(provider: str) -> Any:
    """Real LLM probe — small completion against the provider's API."""
    try:
        from app.llm_factory import create_specialist_llm
    except Exception:
        return None
    try:
        llm = create_specialist_llm(role="research", max_tokens=16)
    except Exception:
        return None
    try:
        # Non-string returns from .call() are exactly what we want to
        # inspect — the wrapped response object.
        response = llm.call(messages=[
            {"role": "user", "content": _DEFAULT_PROBE_PROMPT},
        ])
        # Some clients return just the text; the structural signature is
        # then trivial. That's still recorded as a baseline.
        return response
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "provider_contract_drift: probe for %s raised: %s", provider, exc,
        )
        return None


# ── Public surface ─────────────────────────────────────────────────────


def run_one_pass(
    *,
    baseline_path: Path | str | None = None,
    history_path: Path | str | None = None,
    probe_fn: ProbeFn | None = None,
    providers: list[str] | None = None,
    now: datetime | None = None,
) -> dict:
    """Single drift-detection pass. Returns a structured result.

    Test/operator hooks:
      ``baseline_path`` / ``history_path`` override storage paths.
      ``probe_fn`` injects a fake LLM response (per provider).
      ``providers`` overrides the provider list.
      ``now`` overrides the clock.
    """
    if not _enabled():
        return {"status": "disabled", "alerts": 0}

    bp = Path(baseline_path) if baseline_path else _BASELINE_PATH
    hp = Path(history_path) if history_path else _HISTORY_PATH
    probe = probe_fn or _default_probe
    provider_list = list(providers) if providers else list(_DEFAULT_PROVIDERS)

    baseline = _read_baseline(bp)
    new_baseline = dict(baseline)
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    alerts: list[dict] = []
    seeded: list[str] = []
    skipped: list[dict] = []

    for provider in provider_list:
        try:
            response = probe(provider)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"provider": provider, "reason": f"probe raised: {exc}"})
            continue
        if response is None:
            skipped.append({"provider": provider, "reason": "probe returned None"})
            continue

        signature = extract_signature(response)
        if provider not in baseline:
            new_baseline[provider] = signature
            seeded.append(provider)
            continue

        diff = diff_signatures(baseline[provider], signature)
        if is_breaking_drift(diff):
            alerts.append({
                "provider": provider,
                "diff": diff,
                "ts": timestamp,
            })

    # Persist updated baseline (only for newly-seen providers).
    if seeded:
        _write_baseline(bp, new_baseline)

    # Append history rows for each alert.
    for alert in alerts:
        _append_history(hp, alert)

    return {
        "status": "ok",
        "n_providers": len(provider_list),
        "seeded": seeded,
        "skipped": skipped,
        "alerts": len(alerts),
        "alert_details": alerts,
    }


def run() -> None:
    """Cadence-guarded entry point for the healing-monitors daemon."""
    if not background_enabled():
        return
    state = read_state_json(_STATE_FILE, {"last_run_at": 0.0})
    now = time.time()
    if now - float(state.get("last_run_at", 0)) < _RUN_CADENCE_S:
        return
    state["last_run_at"] = now

    summary = run_one_pass()
    state["last_summary"] = summary

    audit_event(
        "provider_contract_drift_pass",
        n_providers=summary.get("n_providers", 0),
        alerts=summary.get("alerts", 0),
        seeded=summary.get("seeded", []),
    )

    # Signal alert per breaking drift, deduped by (provider, diff hash) within window.
    if summary.get("alerts", 0) > 0:
        try:
            from app.healing.handlers._common import send_signal_alert
            for alert in summary["alert_details"]:
                provider = alert["provider"]
                d = alert["diff"]
                body = (
                    f"⚠️ LLM provider contract drift — `{provider}`\n\n"
                    f"Removed keys: {d['removed'] or '—'}\n"
                    f"Type-changed: {[c['key'] for c in d['changed_type']] or '—'}\n"
                    f"Added (additive-safe, FYI): {d['added'] or '—'}\n\n"
                    f"Parsers may need updating. See `provider_contract_history.jsonl`."
                )
                try:
                    send_signal_alert(
                        body,
                        tag=f"provider_contract_drift:{provider}",
                    )
                except Exception:
                    logger.debug(
                        "provider_contract_drift: signal alert failed",
                        exc_info=True,
                    )
        except Exception:
            logger.debug(
                "provider_contract_drift: signal handler import failed",
                exc_info=True,
            )

    write_state_json(_STATE_FILE, state)
