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

SAFETY_MINIMUM = 0.95       # Hard gate: all systems use same safety floor
QUALITY_MINIMUM = 0.70      # Minimum quality score across all systems
MAX_REGRESSION = 0.15       # No dimension drops > 15% from baseline
MAX_PROMOTIONS_PER_DAY = 20 # Rate limit across all systems combined


# ── Promotion Protocol ───────────────────────────────────────────────────────


@dataclass
class PromotionRequest:
    """Universal promotion request submitted by any improvement system."""
    system: str              # "evolution" | "modification" | "training" | "atlas"
    target: str              # What's being promoted (role name, adapter name, skill name, etc.)
    proposed_by: str         # Which subsystem or agent proposed this
    quality_score: float     # Normalized 0.0-1.0 (MANDATORY)
    safety_score: float      # Normalized 0.0-1.0 (MANDATORY)
    metrics: dict = field(default_factory=dict)       # System-specific scores
    baseline_scores: dict = field(default_factory=dict)  # Previous version metrics
    artifacts: dict = field(default_factory=dict)      # System-specific (prompt text, adapter path, etc.)
    reason: str = ""         # Human-readable explanation


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

    # Gate 1: Safety (hard veto)
    safety_ok = request.safety_score >= SAFETY_MINIMUM
    gates["safety"] = {
        "passed": safety_ok,
        "score": request.safety_score,
        "threshold": SAFETY_MINIMUM,
    }
    if not safety_ok:
        result = PromotionResult(
            approved=False,
            reason=f"Safety gate failed: {request.safety_score:.3f} < {SAFETY_MINIMUM}",
            gate_results=gates,
        )
        _record_promotion(request, result)
        return result

    # Gate 2: Quality minimum
    quality_ok = request.quality_score >= QUALITY_MINIMUM
    gates["quality"] = {
        "passed": quality_ok,
        "score": request.quality_score,
        "threshold": QUALITY_MINIMUM,
    }
    if not quality_ok:
        result = PromotionResult(
            approved=False,
            reason=f"Quality gate failed: {request.quality_score:.3f} < {QUALITY_MINIMUM}",
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
