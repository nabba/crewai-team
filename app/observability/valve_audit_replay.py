"""
valve_audit_replay.py — Daily replay job for the reducing-valve audit.

Reads the prior day's valve_audit.jsonl, samples rejections per filter,
and re-evaluates each sample two ways:

  (A) Loose-replay (deterministic, no LLM): re-checks the same predicate
      at a relaxed threshold. Yields a "would_pass_at_relaxed" verdict.
      Cheap, falsifiable. Produces the *disagreement rate* (DR).

  (B) Second-opinion LLM (optional, gated by VALVE_AUDIT_LLM_REPLAY=1):
      Sends the rejected text + reason to a cheap-vetting LLM and asks
      "would this have been useful given the original task?" Cross-family
      so the same family's judge can't whitewash itself. Produces the
      *false rejection rate* (FRR) — the headline metric.

Decision criterion (applied by the consolidating summary):
    A filter "needs review" when FRR ≥ 0.15 over a 7-day rolling window
    with ≥ 100 sampled rejections. DR is reported alongside as soft
    evidence.

Cost ceiling: when VALVE_AUDIT_LLM_REPLAY=1, we cap LLM calls at
VALVE_AUDIT_LLM_BUDGET_USD per day (default $1). When the cap is hit,
sampling shrinks; the loose-replay pass continues.

Outputs:
    /app/workspace/logs/valve_audit_verdicts.jsonl  — one line per replay
    /app/workspace/logs/valve_audit_summary.jsonl   — one line per day, all filters
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Tunables ─────────────────────────────────────────────────────────────────

_PER_FILTER_SAMPLE = 50               # max sampled rejections per filter per day
_LLM_BUDGET_ENV = "VALVE_AUDIT_LLM_BUDGET_USD"
_LLM_REPLAY_ENV = "VALVE_AUDIT_LLM_REPLAY"
_LLM_DEFAULT_BUDGET_USD = 1.0

# Loose-replay threshold relaxations. Keyed by (filter_id, reason).
# When the original predicate failed at score < threshold, the replay
# would_pass if score >= relaxed_threshold.
_RELAXED_THRESHOLDS: dict[tuple[str, str], float] = {
    ("F1", "density_below_threshold"): 0.001,
    ("F1", "confidence_below_threshold"): 0.50,
    ("F4", "too_short"): 10.0,
    ("F4", "meta_commentary"): 600.0,
    ("F4", "coding_no_code_block"): 50.0,
    ("F8", "below_novelty"): 0.5,
    ("F8", "below_quality"): 0.5,
    ("F8", "below_panel"): 0.4,
}

# Filters where loose-replay doesn't apply (categorical rejections like
# "no_text", "cooldown", or pattern-match rejections). We still count
# them in the totals but skip the would_pass check.
_NO_REPLAY_REASONS: set[tuple[str, str]] = {
    ("F4", "quality_failure_pattern"),
    ("F8", "no_text"),
    ("F8", "cooldown"),
}


_LLM_PROMPT = """You are reviewing a filter decision in an AI system.

A filter rejected the following text with reason: "{reason}".
Filter id: {filter_id}

Text:
{text}

Question: would this text have been useful to the user / pipeline if the
filter had let it through? Consider: is the rejection reason actually
warranted, or does the text contain useful signal the filter dropped?

Reply with a single line of JSON only (no preamble, no markdown):
{{"verdict": "useful" | "not_useful" | "unclear", "rationale": "<≤200 chars>"}}"""


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class _Verdict:
    rejection_id: str
    filter_id: str
    reason: str
    loose_would_pass: bool | None     # None when N/A
    llm_verdict: str | None           # "useful" | "not_useful" | "unclear" | None
    llm_rationale: str | None
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rejection_id": self.rejection_id,
            "filter_id": self.filter_id,
            "reason": self.reason,
            "loose_would_pass": self.loose_would_pass,
            "llm_verdict": self.llm_verdict,
            "llm_rationale": self.llm_rationale,
            "cost_usd": round(self.cost_usd, 6),
        }


# ── Public entry point ───────────────────────────────────────────────────────

def run_daily_replay(target_date: str | None = None) -> dict[str, Any]:
    """Replay yesterday's valve_audit.jsonl. Persist verdicts + summary.

    Args:
        target_date: ISO date "YYYY-MM-DD". Defaults to yesterday (UTC).
            Used by the idle-scheduler hook (defaults) and by tests
            (explicit date).

    Returns the daily summary dict.
    """
    from app.observability import valve_audit
    from app.paths import WORKSPACE_ROOT

    if target_date is None:
        target_date = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

    log_path = valve_audit.log_path()
    rejections = _load_rejections_for_date(log_path, target_date)
    if not rejections:
        logger.info("valve_audit_replay: no rejections for %s", target_date)
        return _empty_summary(target_date)

    sampled = _stratified_sample(rejections, _PER_FILTER_SAMPLE)
    llm_enabled = _llm_replay_enabled()
    budget = _llm_budget_usd()

    verdicts: list[_Verdict] = []
    spent = 0.0
    for r in sampled:
        verdict = _replay_one(r, llm_enabled and spent < budget)
        spent += verdict.cost_usd
        verdicts.append(verdict)

    logs_dir = WORKSPACE_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _append_verdicts(logs_dir / "valve_audit_verdicts.jsonl", target_date, verdicts)

    summary = _summarise(target_date, rejections, verdicts, spent)
    _append_summary(logs_dir / "valve_audit_summary.jsonl", summary)

    logger.info(
        "valve_audit_replay: date=%s rejections=%d sampled=%d cost=$%.4f",
        target_date, len(rejections), len(sampled), spent,
    )
    return summary


# ── Sampling + loading ───────────────────────────────────────────────────────

def _load_rejections_for_date(log_path: Path, target_date: str) -> list[dict]:
    if not log_path.exists():
        return []
    rows: list[dict] = []
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = (row.get("ts") or "")[:10]
                if ts == target_date:
                    rows.append(row)
    except Exception:
        logger.debug("valve_audit_replay: load failed", exc_info=True)
    return rows


def _stratified_sample(rejections: list[dict], per_filter_cap: int) -> list[dict]:
    """Per filter_id, take min(N, per_filter_cap). Deterministic seed
    derived from the filter_id so re-runs of the same day are stable."""
    by_filter: dict[str, list[dict]] = defaultdict(list)
    for r in rejections:
        by_filter[r.get("filter_id", "?")].append(r)
    out: list[dict] = []
    for filter_id, items in by_filter.items():
        if len(items) <= per_filter_cap:
            out.extend(items)
        else:
            rng = random.Random(filter_id)
            out.extend(rng.sample(items, per_filter_cap))
    return out


# ── Per-rejection replay ─────────────────────────────────────────────────────

def _replay_one(rejection: dict, run_llm: bool) -> _Verdict:
    filter_id = rejection.get("filter_id", "?")
    reason = rejection.get("reason", "")
    rid = _rejection_id(rejection)

    loose = _loose_replay(filter_id, reason, rejection)

    llm_verdict: str | None = None
    llm_rationale: str | None = None
    cost = 0.0
    if run_llm:
        llm_verdict, llm_rationale, cost = _llm_second_opinion(rejection)

    return _Verdict(
        rejection_id=rid,
        filter_id=filter_id,
        reason=reason,
        loose_would_pass=loose,
        llm_verdict=llm_verdict,
        llm_rationale=llm_rationale,
        cost_usd=cost,
    )


def _loose_replay(filter_id: str, reason: str, rejection: dict) -> bool | None:
    """Return True iff the rejection would pass at the relaxed threshold.
    Returns None when loose-replay doesn't apply (categorical rejections)."""
    key = (filter_id, reason)
    if key in _NO_REPLAY_REASONS:
        return None
    if key not in _RELAXED_THRESHOLDS:
        return None
    relaxed = _RELAXED_THRESHOLDS[key]
    score = rejection.get("score")
    threshold = rejection.get("threshold")
    if score is None or threshold is None:
        return None
    # For "below_X" reasons the score must rise above relaxed.
    # For "too_short" / "coding_no_code_block" the score is a length;
    # would_pass means length >= relaxed (which is < original threshold).
    return float(score) >= float(relaxed)


def _llm_second_opinion(rejection: dict) -> tuple[str | None, str | None, float]:
    """Cross-family judge call. Best-effort. On failure returns (None, None, 0)."""
    try:
        from app.llm_factory import create_cheap_vetting_llm
        llm = create_cheap_vetting_llm()
    except Exception:
        return None, None, 0.0

    text = rejection.get("input_text", "")
    if not text:
        return None, None, 0.0
    prompt = _LLM_PROMPT.format(
        filter_id=rejection.get("filter_id", "?"),
        reason=rejection.get("reason", ""),
        text=text[:1500],
    )
    raw: str | None = None
    try:
        response = llm.invoke(prompt) if hasattr(llm, "invoke") else llm.call(prompt)
        raw = getattr(response, "content", None) or str(response)
    except Exception:
        logger.debug("valve_audit_replay: LLM call failed", exc_info=True)
        return None, None, 0.0

    parsed = _parse_llm_verdict(raw or "")
    if parsed is None:
        return None, None, _approx_cost(prompt, raw or "")
    verdict, rationale = parsed
    return verdict, rationale, _approx_cost(prompt, raw or "")


def _parse_llm_verdict(raw: str) -> tuple[str, str] | None:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    verdict = obj.get("verdict")
    if verdict not in {"useful", "not_useful", "unclear"}:
        return None
    rationale = (obj.get("rationale") or "")[:200]
    return verdict, rationale


def _approx_cost(prompt: str, response: str) -> float:
    """Rough cost estimate. The cheap-vetting tier is ~$0.0001/1k tokens
    each way; we use 4 chars/token as the standard heuristic."""
    in_tokens = len(prompt) / 4
    out_tokens = len(response) / 4
    return (in_tokens + out_tokens) * 0.0001 / 1000.0


# ── Summary ──────────────────────────────────────────────────────────────────

def _summarise(
    date: str,
    rejections: list[dict],
    verdicts: list[_Verdict],
    cost_usd: float,
) -> dict[str, Any]:
    by_filter_rejections: dict[str, int] = defaultdict(int)
    for r in rejections:
        by_filter_rejections[r.get("filter_id", "?")] += 1

    by_filter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for v in verdicts:
        bucket = by_filter[v.filter_id]
        bucket["sampled"] += 1
        if v.loose_would_pass is True:
            bucket["loose_would_pass"] += 1
        elif v.loose_would_pass is None:
            bucket["loose_replay_na"] += 1
        if v.llm_verdict == "useful":
            bucket["llm_useful"] += 1
        elif v.llm_verdict == "not_useful":
            bucket["llm_not_useful"] += 1
        elif v.llm_verdict == "unclear":
            bucket["llm_unclear"] += 1
        elif v.llm_verdict is None:
            bucket["llm_skipped"] += 1

    per_filter: list[dict[str, Any]] = []
    for filter_id, total in by_filter_rejections.items():
        b = by_filter.get(filter_id, {})
        sampled = b.get("sampled", 0)
        loose_eligible = sampled - b.get("loose_replay_na", 0)
        dr = (
            b.get("loose_would_pass", 0) / loose_eligible
            if loose_eligible > 0 else None
        )
        llm_ans = b.get("llm_useful", 0) + b.get("llm_not_useful", 0)
        frr = (
            b.get("llm_useful", 0) / llm_ans
            if llm_ans > 0 else None
        )
        per_filter.append({
            "filter_id": filter_id,
            "rejections_total": total,
            "sampled": sampled,
            "disagreement_rate": round(dr, 4) if dr is not None else None,
            "false_rejection_rate": round(frr, 4) if frr is not None else None,
            "estimated_useful_lost_per_day": (
                round(total * frr, 1) if frr is not None else None
            ),
            "needs_review": bool(frr is not None and frr >= 0.15 and total >= 50),
        })
    per_filter.sort(key=lambda d: -(d["rejections_total"] or 0))

    return {
        "date": date,
        "rejections_total": len(rejections),
        "sampled_total": len(verdicts),
        "cost_usd": round(cost_usd, 4),
        "filters": per_filter,
        "criterion": {
            "frr_threshold": 0.15,
            "min_samples_for_review": 50,
        },
    }


def _empty_summary(date: str) -> dict[str, Any]:
    return {
        "date": date,
        "rejections_total": 0,
        "sampled_total": 0,
        "cost_usd": 0.0,
        "filters": [],
        "criterion": {"frr_threshold": 0.15, "min_samples_for_review": 50},
    }


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _rejection_id(row: dict) -> str:
    """Stable id derived from ts + input_hash + filter_id."""
    h = (row.get("input_hash") or "")[:16]
    return f"{(row.get('ts') or '')}|{row.get('filter_id', '?')}|{h}"


def _append_verdicts(path: Path, date: str, verdicts: list[_Verdict]) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            for v in verdicts:
                row = {"replay_date": date, **v.to_dict()}
                f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        logger.debug("valve_audit_replay: verdicts write failed", exc_info=True)


def _append_summary(path: Path, summary: dict) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, default=str) + "\n")
    except Exception:
        logger.debug("valve_audit_replay: summary write failed", exc_info=True)


def _llm_replay_enabled() -> bool:
    val = os.environ.get(_LLM_REPLAY_ENV, "0").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _llm_budget_usd() -> float:
    raw = os.environ.get(_LLM_BUDGET_ENV)
    if not raw:
        return _LLM_DEFAULT_BUDGET_USD
    try:
        return float(raw)
    except ValueError:
        return _LLM_DEFAULT_BUDGET_USD
