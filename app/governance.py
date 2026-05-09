"""
governance.py — Unified promotion governance across all improvement systems.

Provides a shared promotion protocol so evolution, feedback modification,
training pipeline, and ATLAS all use consistent:
  - Safety gates (0.95 minimum — zero tolerance for safety regression)
  - Quality gates (0.70 minimum across all systems)
  - Regression checks (max 15% drop on any dimension)
  - Audit trail (governance.promotions table in PostgreSQL)
  - Rollback rules (recorded for unified querying)

Each improvement system still owns its own trigger logic and metrics
collection. Governance only gates the final promotion decision and
provides unified audit across all systems.

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Universal promotion thresholds ────────────────────────────────
# These apply to ALL improvement systems identically.
#
# ── Floors / ceiling (post-bootstrap safety contract, 2026-05-09) ────
# The ratchet protocol in app/governance_ratchet/ can RAISE the
# effective minimums above these floors as the system earns trust,
# and (with operator typed-phrase confirmation) RELAX them back down
# — but never below FLOOR. The ``effective_*_minimum()`` functions
# below clamp via ``max(FLOOR, ratcheted)`` so even a corrupted
# ratchet state file can't drop below the safety contract.
#
# Operators authorized this Tier-3 edit on 2026-05-09.

SAFETY_MINIMUM_FLOOR = 0.95       # Hard gate: never below this
QUALITY_MINIMUM_FLOOR = 0.70      # Minimum quality score across all systems
MAX_REGRESSION = 0.15             # No dimension drops > 15% from baseline
MAX_PROMOTIONS_PER_DAY = 20       # Rate limit across all systems combined

# Backwards-compat aliases for any external readers that imported the
# constants directly. These reflect the FLOOR — readers that care about
# the live effective value should call ``effective_safety_minimum()`` /
# ``effective_quality_minimum()`` instead.
SAFETY_MINIMUM = SAFETY_MINIMUM_FLOOR
QUALITY_MINIMUM = QUALITY_MINIMUM_FLOOR


def threshold_floor(name: str) -> float:
    """Floor lookup for the ratchet protocol. Read from
    ``app.governance_ratchet.protocol._floor`` — keeps the floors
    physically inside this TIER_IMMUTABLE file.
    """
    if name == "safety_minimum":
        return SAFETY_MINIMUM_FLOOR
    if name == "quality_minimum":
        return QUALITY_MINIMUM_FLOOR
    raise ValueError(f"unknown threshold {name!r}")


def effective_safety_minimum() -> float:
    """The safety threshold ``evaluate_promotion`` actually enforces.
    Equals ``max(FLOOR, ratcheted)`` so operator ratcheting raises
    the bar but a corrupted state file can never drop below FLOOR.
    """
    try:
        from app.governance_ratchet import effective_value
        return float(effective_value("safety_minimum"))
    except Exception:
        # Ratchet unavailable / state corrupted → fall back to FLOOR.
        return SAFETY_MINIMUM_FLOOR


def effective_quality_minimum() -> float:
    """The quality threshold ``evaluate_promotion`` actually enforces.
    See ``effective_safety_minimum`` for the floor invariant.
    """
    try:
        from app.governance_ratchet import effective_value
        return float(effective_value("quality_minimum"))
    except Exception:
        return QUALITY_MINIMUM_FLOOR


# ── Goodhart hard gate (Wave 3 #2, 2026-05-09 — operator-authorized) ──────


def _goodhart_hard_gate_disabled() -> bool:
    """Emergency disable. When true, the goodhart gate is skipped entirely
    — for incident response when a buggy detector is blocking promotions.
    """
    import os
    return os.getenv("GOODHART_HARD_GATE_DISABLED", "false").lower() in (
        "true", "1", "yes",
    )


def _goodhart_hard_gate_enforcing() -> bool:
    """When true, severity='high' BLOCKS promotion. Default OFF: ship
    the gate in advisory mode for ~2 weeks before enforcing so operators
    can characterise false-positive rates first.
    """
    import os
    return os.getenv("GOODHART_HARD_GATE_ENFORCING", "false").lower() in (
        "true", "1", "yes",
    )


def _evaluate_goodhart_gate() -> dict:
    """Read the recent goodhart severity and decide whether to block.

    Returns a dict shaped for inclusion in ``PromotionResult.gate_results``::

        {
          "phase": "advisory" | "enforcing" | "disabled",
          "severity": "none|low|medium|high",
          "description": "<sample if any>",
          "block": bool,
          "passed": bool,
        }

    Never raises — failure to read goodhart state degrades to "none".
    """
    if _goodhart_hard_gate_disabled():
        return {
            "phase": "disabled",
            "severity": "none",
            "description": "",
            "block": False,
            "passed": True,
        }

    try:
        from app.goodhart_guard import recent_signal_summary
        summary = recent_signal_summary(lookback_hours=24)
    except Exception:
        # Detector unavailable → fail OPEN (don't block on our own bugs).
        return {
            "phase": "advisory",
            "severity": "none",
            "description": "(detector unavailable)",
            "block": False,
            "passed": True,
        }

    severity = summary.get("highest_severity", "none")
    description = summary.get("highest_description", "")

    enforcing = _goodhart_hard_gate_enforcing()
    block = enforcing and severity == "high"

    return {
        "phase": "enforcing" if enforcing else "advisory",
        "severity": severity,
        "description": description,
        "counts": summary.get("counts", {}),
        "block": block,
        "passed": not block,
    }


# ── Promotion Protocol ───────────────────────────────────────────────────────


@dataclass
class PromotionRequest:
    """Universal promotion request submitted by any improvement system.

    Phase E4: ``__post_init__`` runs strict shape validation. Catches
    the silent-None / wrong-type class of bug that the cross-area audit
    flagged — historically a missing field could slip through and the
    safety/quality comparison would raise a confusing TypeError deep
    inside :func:`evaluate_promotion`. Now every request must satisfy:

      - ``system``, ``target``, ``proposed_by`` are non-empty strings
      - ``quality_score`` and ``safety_score`` are floats in [0.0, 1.0]
      - ``metrics``, ``baseline_scores``, ``artifacts`` are dicts

    Any violation raises ``ValueError`` at construction with a precise
    message; the caller (evolution.py / modification_engine.py / etc.)
    can log the malformed payload instead of polluting the governance
    audit trail with garbage.
    """
    system: str              # "evolution" | "modification" | "training" | "atlas"
    target: str              # What's being promoted (role name, adapter name, skill name, etc.)
    proposed_by: str         # Which subsystem or agent proposed this
    quality_score: float     # Normalized 0.0-1.0 (MANDATORY)
    safety_score: float      # Normalized 0.0-1.0 (MANDATORY)
    metrics: dict = field(default_factory=dict)       # System-specific scores
    baseline_scores: dict = field(default_factory=dict)  # Previous version metrics
    artifacts: dict = field(default_factory=dict)      # System-specific (prompt text, adapter path, etc.)
    reason: str = ""         # Human-readable explanation

    def __post_init__(self) -> None:
        # Strings must be non-empty (governance audit needs them).
        for fname in ("system", "target", "proposed_by"):
            v = getattr(self, fname)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"PromotionRequest.{fname} must be a non-empty string, got {v!r}"
                )
        # Scores must be numeric and in [0, 1] — the gate comparisons
        # depend on this. None or out-of-range here would create
        # malformed audit rows.
        for fname in ("quality_score", "safety_score"):
            v = getattr(self, fname)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(
                    f"PromotionRequest.{fname} must be a float, got {type(v).__name__}={v!r}"
                )
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(
                    f"PromotionRequest.{fname} must be in [0.0, 1.0], got {v!r}"
                )
            # Coerce ints to float so downstream `>=` is consistent.
            object.__setattr__(self, fname, float(v))
        # Containers must be dicts (audit serialization assumes this).
        for fname in ("metrics", "baseline_scores", "artifacts"):
            v = getattr(self, fname)
            if not isinstance(v, dict):
                raise ValueError(
                    f"PromotionRequest.{fname} must be a dict, got {type(v).__name__}"
                )


@dataclass
class PromotionResult:
    """Result of a promotion gate evaluation."""
    approved: bool
    reason: str
    promotion_id: str = ""
    gate_results: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.promotion_id:
            self.promotion_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ── Gate Evaluation ──────────────────────────────────────────────────────────


def evaluate_promotion(request: PromotionRequest) -> PromotionResult:
    """Apply universal promotion gates to a request from any improvement system.

    Gate order (fail-fast):
      1. Safety hard gate — any safety score below 0.95 is an immediate veto
      2. Quality minimum — must meet 0.70 floor
      3. Regression check — no dimension drops > 15% from baseline
      4. Rate limit — prevent runaway promotion loops

    Returns PromotionResult with gate_results showing which gates passed/failed.
    All evaluations are recorded in the unified audit trail.
    """
    gates = {}

    # Resolve LIVE effective minimums via the ratchet protocol — these
    # equal max(FLOOR, ratcheted_value), so the FLOOR (post-bootstrap
    # safety contract) is always honoured even with a corrupted state
    # file. The ratchet itself is operator-controlled (no agent path
    # in V1) via the React /cp/settings UI.
    _safety_min = effective_safety_minimum()
    _quality_min = effective_quality_minimum()

    # Gate 0: Goodhart hard gate (Wave 3 #2, 2026-05-09 — operator-authorized).
    #
    # Two phases controlled by env flags:
    #   * Advisory (default ON): every promotion records the current
    #     gaming-signal severity in gates["goodhart"], but does NOT
    #     block. Lets operators watch for false-positives over a
    #     period before flipping to enforcing.
    #   * Enforcing (GOODHART_HARD_GATE_ENFORCING=true, default OFF):
    #     promotions are BLOCKED when the recent-window severity is
    #     "high".
    #   * Emergency disable (GOODHART_HARD_GATE_DISABLED=true): both
    #     phases skipped — for incident response when a buggy
    #     detector is blocking everything.
    _goodhart_gate_result = _evaluate_goodhart_gate()
    gates["goodhart"] = _goodhart_gate_result
    if _goodhart_gate_result.get("block"):
        result = PromotionResult(
            approved=False,
            reason=(
                f"Goodhart hard gate blocked: severity="
                f"{_goodhart_gate_result.get('severity')!r} — "
                f"{_goodhart_gate_result.get('description', '')[:140]}"
            ),
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # Gate 1: Safety (hard veto)
    safety_ok = request.safety_score >= _safety_min
    gates["safety"] = {
        "passed": safety_ok,
        "score": request.safety_score,
        "threshold": _safety_min,
        "floor": SAFETY_MINIMUM_FLOOR,
    }
    if not safety_ok:
        result = PromotionResult(
            approved=False,
            reason=f"Safety gate failed: {request.safety_score:.3f} < {_safety_min}",
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # Gate 2: Quality minimum
    quality_ok = request.quality_score >= _quality_min
    gates["quality"] = {
        "passed": quality_ok,
        "score": request.quality_score,
        "threshold": _quality_min,
        "floor": QUALITY_MINIMUM_FLOOR,
    }
    if not quality_ok:
        result = PromotionResult(
            approved=False,
            reason=f"Quality gate failed: {request.quality_score:.3f} < {_quality_min}",
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # Gate 3: Regression check (if baseline provided)
    regression_ok = True
    regression_detail = {}
    if request.baseline_scores:
        for dim, baseline_val in request.baseline_scores.items():
            current_val = request.metrics.get(dim, baseline_val)
            if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                if baseline_val > 0:
                    drop = (baseline_val - current_val) / baseline_val
                    if drop > MAX_REGRESSION:
                        regression_ok = False
                        regression_detail[dim] = {
                            "baseline": baseline_val,
                            "current": current_val,
                            "drop": round(drop, 4),
                        }
    gates["regression"] = {
        "passed": regression_ok,
        "threshold": MAX_REGRESSION,
        "details": regression_detail,
    }
    if not regression_ok:
        dims = ", ".join(f"{d}: {v['drop']:.0%} drop" for d, v in regression_detail.items())
        result = PromotionResult(
            approved=False,
            reason=f"Regression gate failed: {dims}",
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # Gate 4: Rate limit (prevent runaway loops)
    rate_ok = _check_rate_limit(request.system)
    gates["rate_limit"] = {"passed": rate_ok, "max_per_day": MAX_PROMOTIONS_PER_DAY}
    if not rate_ok:
        result = PromotionResult(
            approved=False,
            reason=f"Rate limit exceeded: >{MAX_PROMOTIONS_PER_DAY} promotions/day",
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # All gates passed
    result = PromotionResult(
        approved=True,
        reason="All promotion gates passed",
        gate_results=gates,
    )
    _record_promotion(request, result)
    logger.info(
        f"governance: APPROVED promotion [{request.system}] target={request.target} "
        f"quality={request.quality_score:.3f} safety={request.safety_score:.3f}"
    )
    return result


# ── Audit Trail ──────────────────────────────────────────────────────────────


def _record_promotion(request: PromotionRequest, result: PromotionResult) -> None:
    """Write promotion decision to unified governance.promotions table."""
    try:
        from app.config import get_settings
        import psycopg2
        s = get_settings()
        if not s.mem0_postgres_url:
            return
        conn = psycopg2.connect(s.mem0_postgres_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            # Ensure schema exists
            cur.execute("CREATE SCHEMA IF NOT EXISTS governance")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS governance.promotions (
                    id UUID PRIMARY KEY,
                    system TEXT NOT NULL,
                    target TEXT NOT NULL,
                    proposed_by TEXT,
                    quality_score FLOAT,
                    safety_score FLOAT,
                    approved BOOLEAN,
                    reason TEXT,
                    gate_results JSONB,
                    metrics JSONB,
                    baseline_scores JSONB,
                    artifacts JSONB,
                    created_at TIMESTAMPTZ DEFAULT now()
                )
            """)
            cur.execute("""
                INSERT INTO governance.promotions
                (id, system, target, proposed_by, quality_score, safety_score,
                 approved, reason, gate_results, metrics, baseline_scores, artifacts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result.promotion_id,
                request.system,
                request.target,
                request.proposed_by,
                request.quality_score,
                request.safety_score,
                result.approved,
                result.reason,
                json.dumps(result.gate_results),
                json.dumps(request.metrics),
                json.dumps(request.baseline_scores),
                json.dumps(request.artifacts),
            ))
        conn.close()
    except Exception:
        logger.debug("governance: failed to record promotion", exc_info=True)


def _check_rate_limit(system: str) -> bool:
    """Check if we've exceeded daily promotion limit for this system."""
    try:
        from app.config import get_settings
        import psycopg2
        s = get_settings()
        if not s.mem0_postgres_url:
            return True  # No DB = no rate tracking, allow
        conn = psycopg2.connect(s.mem0_postgres_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM governance.promotions
                WHERE system = %s AND approved = true
                AND created_at > now() - interval '24 hours'
            """, (system,))
            count = cur.fetchone()[0]
        conn.close()
        return count < MAX_PROMOTIONS_PER_DAY
    except Exception:
        return True  # Fail open on DB errors (don't block promotions on infra issues)


# ── Query Helpers ────────────────────────────────────────────────────────────


def get_recent_promotions(system: str = None, limit: int = 20) -> list[dict]:
    """Query recent promotion decisions across all or a specific system."""
    try:
        from app.config import get_settings
        import psycopg2
        s = get_settings()
        if not s.mem0_postgres_url:
            return []
        conn = psycopg2.connect(s.mem0_postgres_url)
        with conn.cursor() as cur:
            if system:
                cur.execute("""
                    SELECT id, system, target, approved, reason, quality_score,
                           safety_score, created_at
                    FROM governance.promotions
                    WHERE system = %s
                    ORDER BY created_at DESC LIMIT %s
                """, (system, limit))
            else:
                cur.execute("""
                    SELECT id, system, target, approved, reason, quality_score,
                           safety_score, created_at
                    FROM governance.promotions
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
        conn.close()
        return [
            {
                "id": str(r[0]), "system": r[1], "target": r[2],
                "approved": r[3], "reason": r[4],
                "quality_score": r[5], "safety_score": r[6],
                "created_at": r[7].isoformat() if r[7] else "",
            }
            for r in rows
        ]
    except Exception:
        return []


def format_governance_report() -> str:
    """Human-readable summary of recent governance activity."""
    recent = get_recent_promotions(limit=10)
    if not recent:
        return "No promotion decisions recorded yet."

    lines = ["📋 Governance — Recent Promotions"]
    approved = sum(1 for r in recent if r["approved"])
    rejected = len(recent) - approved
    lines.append(f"   Last 10: {approved} approved, {rejected} rejected\n")

    for r in recent[:5]:
        status = "✅" if r["approved"] else "❌"
        lines.append(
            f"   {status} [{r['system']}] {r['target']} — "
            f"q={r.get('quality_score', 0):.2f} s={r.get('safety_score', 0):.2f}"
        )
        if not r["approved"]:
            lines.append(f"      Reason: {r['reason'][:80]}")

    return "\n".join(lines)
