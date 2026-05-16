"""agreement_ledger — operator-response self-model (Q17.5)."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


KNOWN_CATEGORIES = frozenset({
    "proactive_briefing", "proposal_bridge", "tier3_amendment", "library_radar",
    "paper_pipeline", "brainstorm_idea", "threads_consultation", "health_alert",
    "person_suggestion", "cross_modal_pattern", "code_change", "other",
})


class AgreementResponse(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    IGNORED = "ignored"
    DEFERRED = "deferred"


_ROLLING_WINDOW_DAYS = 90
_IGNORE_AFTER_DAYS = 7
_LOCK = threading.Lock()


def _workspace_root() -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        return Path(WORKSPACE_ROOT)
    except Exception:
        return Path(os.environ.get("WORKSPACE_ROOT", "/app/workspace"))


def _ledger_path() -> Path:
    return _workspace_root() / "self_model" / "agreement_ledger.jsonl"


def _enabled() -> bool:
    try:
        from app.runtime_settings import get_agreement_ledger_enabled
        return get_agreement_ledger_enabled()
    except Exception:
        return True


def _now_iso(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def _append(row: dict[str, Any]) -> None:
    if not _enabled():
        return
    p = _ledger_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True)
    with _LOCK:
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            logger.debug("agreement_ledger: append failed", exc_info=True)


def record_suggestion(
    *,
    suggestion_id: str | None = None,
    category: str = "other",
    source_module: str = "",
    summary: str = "",
    now: datetime | None = None,
) -> str:
    cat = category if category in KNOWN_CATEGORIES else "other"
    sid = suggestion_id or f"sg_{uuid.uuid4().hex[:12]}"
    _append({
        "ts": _now_iso(now),
        "suggestion_id": sid,
        "category": cat,
        "source_module": source_module[:64],
        "summary": summary[:280],
        "response": AgreementResponse.PENDING.value,
    })
    return sid


def record_response(suggestion_id: str, response: AgreementResponse | str, *, note: str = "", now: datetime | None = None) -> None:
    if isinstance(response, str):
        try:
            response = AgreementResponse(response.lower())
        except ValueError:
            return
    _append({"ts": _now_iso(now), "suggestion_id": suggestion_id, "response": response.value, "note": note[:280]})


def _read_all() -> list[dict[str, Any]]:
    p = _ledger_path()
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
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
        return []
    return out


def _final_state(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    creation: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = r.get("suggestion_id")
        if not sid:
            continue
        if "category" in r and sid not in creation:
            creation[sid] = r
        prev = by_id.get(sid)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            by_id[sid] = r
    out: dict[str, dict[str, Any]] = {}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_IGNORE_AFTER_DAYS)).isoformat()
    for sid, latest in by_id.items():
        base = creation.get(sid, {})
        merged: dict[str, Any] = dict(base)
        merged.update({k: v for k, v in latest.items() if v is not None})
        resp = merged.get("response", AgreementResponse.PENDING.value)
        if resp == AgreementResponse.PENDING.value and base.get("ts", "") < cutoff:
            resp = AgreementResponse.IGNORED.value
        merged["response"] = resp
        out[sid] = merged
    return out


def rolling_rate(category: str | None = None, *, window_days: int = _ROLLING_WINDOW_DAYS, now: datetime | None = None) -> dict[str, Any]:
    cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=window_days)).isoformat()
    rows = _read_all()
    final = _final_state(rows)
    selected: list[dict[str, Any]] = []
    for sid, r in final.items():
        if r.get("ts", "") < cutoff:
            continue
        if category and r.get("category") != category:
            continue
        selected.append(r)
    counts: dict[str, int] = {}
    for r in selected:
        c = r.get("response", AgreementResponse.PENDING.value)
        counts[c] = counts.get(c, 0) + 1
    n = len(selected)
    rates = {k: round(v / n, 3) for k, v in counts.items()} if n else {}
    first = min((r.get("ts", "") for r in selected), default=None)
    last = max((r.get("ts", "") for r in selected), default=None)
    return {"window_days": window_days, "category": category, "n": n, "by_response": counts, "rates": rates, "first_ts": first, "last_ts": last}


def summary_for_briefing(*, window_days: int = _ROLLING_WINDOW_DAYS) -> dict[str, Any]:
    rows = _read_all()
    final = _final_state(rows)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    by_cat: dict[str, dict[str, Any]] = {}
    n_pending = 0
    n_total = 0
    for r in final.values():
        if r.get("ts", "") < cutoff:
            continue
        n_total += 1
        cat = r.get("category", "other")
        cur = by_cat.setdefault(cat, {"n": 0, "by_response": {}})
        cur["n"] += 1
        resp = r.get("response", AgreementResponse.PENDING.value)
        cur["by_response"][resp] = cur["by_response"].get(resp, 0) + 1
        if resp == AgreementResponse.PENDING.value:
            n_pending += 1
    for cat, blob in by_cat.items():
        n = blob["n"]
        blob["accepted_rate"] = round(blob["by_response"].get("accepted", 0) / n, 3) if n else 0
        blob["rejected_rate"] = round(blob["by_response"].get("rejected", 0) / n, 3) if n else 0
        blob["ignored_rate"] = round(blob["by_response"].get("ignored", 0) / n, 3) if n else 0
    return {"window_days": window_days, "n_total": n_total, "by_category": by_cat, "n_pending": n_pending}


def briefing_section() -> str:
    s = summary_for_briefing()
    if s["n_total"] == 0:
        return ""
    lines = [f"🧭 Suggestion agreement (rolling {s['window_days']}d, n={s['n_total']}):"]
    rows = sorted(s["by_category"].items(), key=lambda kv: kv[1]["n"], reverse=True)
    for cat, blob in rows[:6]:
        lines.append(f"  • {cat:24s} n={blob['n']:>3d}  ✓{blob['accepted_rate']:.0%} ✗{blob['rejected_rate']:.0%} …{blob['ignored_rate']:.0%}")
    if s["n_pending"]:
        lines.append(f"  ({s['n_pending']} still pending)")
    return "\n".join(lines)
