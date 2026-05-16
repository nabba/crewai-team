"""kb_contradiction — epistemic-claim contradiction probe (Q17.6).

Weekly probe. Samples 200 claims, groups by subject, runs pairwise
negation-pair check + numeric-comparator check. Surfaces structural
contradictions to Signal + ledger.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


NAME = "kb_contradiction"
CADENCE_SECONDS = 24 * 3600
MASTER_SWITCH_KEY = "kb_contradiction_monitor_enabled"
_INTERNAL_CADENCE_S = 7 * 24 * 3600

_DEFAULT_SAMPLE_SIZE = 200
_MIN_SHARED_TOKENS = 3
_MAX_PAIRS_REPORTED = 20

_NEGATION_PAIRS = [
    ("is", "isn't"), ("is", "is not"),
    ("are", "aren't"), ("are", "are not"),
    ("was", "wasn't"), ("was", "was not"),
    ("were", "weren't"), ("were", "were not"),
    ("has", "hasn't"), ("has", "has not"),
    ("have", "haven't"), ("have", "have not"),
    ("does", "doesn't"), ("does", "does not"),
    ("can", "can't"), ("can", "cannot"),
    ("will", "won't"), ("will", "will not"),
    ("should", "shouldn't"), ("should", "should not"),
]

_COMPARATOR_PAIRS = [(">", "<"), (">=", "<="), (">=", "<"), (">", "<=")]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _claim_sources() -> list[Path]:
    root = _workspace_root()
    return [root / "epistemic" / "claims.jsonl", root / "epistemic" / "claims" / "claims.jsonl"]


def _load_claims(sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in _claim_sources():
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    if len(out) <= sample_size:
        return out
    return rng.sample(out, sample_size)


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "")}


def _claim_text(c: dict[str, Any]) -> str:
    for k in ("text", "claim", "statement", "predicate", "body"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _claim_subject(c: dict[str, Any]) -> str:
    for k in ("subject", "topic", "entity", "subject_key"):
        v = c.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    text = _claim_text(c)
    toks = _TOKEN_RE.findall(text)
    return (" ".join(toks[:3])).lower() or text[:40].lower()


def _is_negation_pair(text_a: str, text_b: str) -> str | None:
    a_low = " " + text_a.lower() + " "
    b_low = " " + text_b.lower() + " "
    for pos, neg in _NEGATION_PAIRS:
        in_a_pos = f" {pos} " in a_low
        in_a_neg = f" {neg} " in a_low
        in_b_pos = f" {pos} " in b_low
        in_b_neg = f" {neg} " in b_low
        if in_a_pos and in_b_neg and not in_a_neg and not in_b_pos:
            return f"{pos}↔{neg}"
        if in_a_neg and in_b_pos and not in_a_pos and not in_b_neg:
            return f"{pos}↔{neg}"
    for gt, lt in _COMPARATOR_PAIRS:
        if gt in text_a and lt in text_b:
            return f"{gt}↔{lt}"
        if lt in text_a and gt in text_b:
            return f"{gt}↔{lt}"
    return None


def _shared_tokens(a: dict[str, Any], b: dict[str, Any]) -> int:
    return len(_tokens(_claim_text(a)) & _tokens(_claim_text(b)))


def _find_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        by_subject.setdefault(_claim_subject(c), []).append(c)
    out: list[dict[str, Any]] = []
    for subj, group in by_subject.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if _shared_tokens(a, b) < _MIN_SHARED_TOKENS:
                    continue
                hit = _is_negation_pair(_claim_text(a), _claim_text(b))
                if hit is None:
                    continue
                out.append({
                    "subject": subj, "pattern": hit,
                    "claim_a": _claim_text(a)[:280],
                    "claim_b": _claim_text(b)[:280],
                    "claim_a_id": a.get("id") or a.get("claim_id"),
                    "claim_b_id": b.get("id") or b.get("claim_id"),
                    "ts_a": a.get("ts") or a.get("timestamp"),
                    "ts_b": b.get("ts") or b.get("timestamp"),
                })
                if len(out) >= _MAX_PAIRS_REPORTED:
                    return out
    return out


def _state_path() -> Path:
    return _workspace_root() / "healing" / "kb_contradiction_state.json"


def _log_path() -> Path:
    return _workspace_root() / "healing" / "kb_contradictions.jsonl"


def _read_state() -> dict[str, Any]:
    p = _state_path()
    if not p.exists():
        return {"last_run": 0, "n_seen": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"last_run": 0, "n_seen": 0}


def _write_state(s: dict[str, Any]) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(s, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.debug("kb_contradiction: state write failed", exc_info=True)


def _append_log(rows: list[dict[str, Any]]) -> None:
    p = _log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(p, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **r}, sort_keys=True) + "\n")
    except OSError:
        logger.debug("kb_contradiction: log append failed", exc_info=True)


def _emit_alert(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    body = [f"Found {len(rows)} candidate KB contradiction(s). Review at /cp/monitor.", ""]
    for r in rows[:5]:
        body.append(f"  • {r['subject'][:40]:40s} [{r['pattern']}]")
        body.append(f"      A: {r['claim_a'][:100]}")
        body.append(f"      B: {r['claim_b'][:100]}")
    try:
        from app.notify import notify
        notify(title="🧩 KB contradiction candidates", body="\n".join(body), url="/cp/monitor", topic="kb_contradiction", critical=False, arbitrate=True)
    except Exception:
        logger.debug("kb_contradiction: notify failed", exc_info=True)
    try:
        from app.identity.continuity_ledger import record_event
        record_event(kind="q17_landmark", actor="kb_contradiction", summary=f"{len(rows)} KB contradictions surfaced", detail={"subsystem": "kb_contradiction", "n": len(rows)})
    except Exception:
        logger.debug("kb_contradiction: ledger emit failed", exc_info=True)


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_kb_contradiction_monitor_enabled
        return get_kb_contradiction_monitor_enabled()
    except Exception:
        return True


def _cadence_due(state: dict[str, Any]) -> bool:
    last = float(state.get("last_run") or 0)
    return (datetime.now(timezone.utc).timestamp() - last) >= _INTERNAL_CADENCE_S


def run(*, sample_size: int = _DEFAULT_SAMPLE_SIZE, rng_seed: int | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"checked": False, "n_sampled": 0, "n_contradictions": 0, "errors": 0}
    if not _enabled():
        summary["skipped"] = True
        return summary
    state = _read_state()
    if not _cadence_due(state):
        summary["skipped_cadence"] = True
        return summary
    try:
        rng = random.Random(rng_seed)
        claims = _load_claims(sample_size, rng)
        summary["n_sampled"] = len(claims)
        contradictions = _find_contradictions(claims)
        summary["n_contradictions"] = len(contradictions)
        summary["contradictions"] = contradictions
        if contradictions:
            _append_log(contradictions)
            _emit_alert(contradictions)
        state["last_run"] = datetime.now(timezone.utc).timestamp()
        state["n_seen"] = int(state.get("n_seen") or 0) + len(contradictions)
        _write_state(state)
        summary["checked"] = True
    except Exception:
        logger.debug("kb_contradiction: probe failed", exc_info=True)
        summary["errors"] += 1
    return summary
