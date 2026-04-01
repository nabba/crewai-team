"""
modification_engine.py — Proposes targeted prompt/config changes based on
diagnosed feedback patterns.

NOT an agent — a plain Python service function that calls an LLM.  This is
deliberate: agents should not modify their own prompts, even indirectly.

The modification engine:
  1. Reads triggered patterns from the feedback pipeline
  2. Checks rate limits (IMMUTABLE constants)
  3. Generates a hypothesis (proposed prompt change) using a mid-tier LLM
  4. Routes by tier: Tier 1 → eval sandbox, Tier 2 → Signal approval
  5. Records all attempts in PostgreSQL modification schema

Rate limits and tier boundaries are IMMUTABLE — defined as module-level
constants that cannot be changed by any agent.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

# ── IMMUTABLE: Rate limits ────────────────────────────────────────────────
# These are infrastructure-level constants.  No agent, no config file,
# no environment variable can override them.
TIER1_MAX_PER_DAY = 10
TIER1_MAX_PER_WEEK = 30
TIER2_MAX_PER_DAY = 3
TIER2_MAX_PER_WEEK = 10
REJECTION_COOLDOWN_HOURS = 1

# ── IMMUTABLE: Tier routing rules ─────────────────────────────────────────
# Parameters that can be modified autonomously (Tier 1)
TIER1_PARAMETERS = frozenset({
    "system_prompt", "few_shot_examples", "style_params",
    "knowledge_base", "tool_defaults",
})

# Parameters that require human approval (Tier 2)
TIER2_PARAMETERS = frozenset({
    "workflow_graph", "agent_roles", "tool_permissions",
    "delegation_policies", "inter_agent_communication",
})

# ── IMMUTABLE: Modification strategies ────────────────────────────────────
STRATEGIES = [
    "additive_instruction",     # Add new instruction/constraint to prompt
    "example_injection",        # Add input-output example pair
    "instruction_refinement",   # Modify existing instruction to be more specific
    "constraint_addition",      # Add hard "never/always" constraint
    "persona_calibration",      # Adjust tone/formality/style parameters
]

# ── Hypothesis generation prompt ──────────────────────────────────────────
HYPOTHESIS_PROMPT = """You are a prompt engineering specialist for an AI agent system.
Based on user feedback, propose a specific modification to the agent's prompt.

CURRENT PROMPT (for role: {role}):
---
{current_prompt}
---

FEEDBACK PATTERN:
- Category: {category}
- Direction: {direction}
- Event count: {event_count}
- Severity: {severity}

EXAMPLE INTERACTIONS THAT TRIGGERED THIS FEEDBACK:
{examples}

AVAILABLE STRATEGIES:
1. additive_instruction — Add a new instruction or constraint
2. example_injection — Add an input-output example demonstrating desired behavior
3. instruction_refinement — Modify an existing instruction to be more specific
4. constraint_addition — Add a hard "never do X" or "always do Y" rule
5. persona_calibration — Adjust tone, formality, or style parameters

RULES:
- Make the MINIMAL change necessary to address the feedback
- Preserve all existing safety constraints and constitutional principles
- Do not remove existing instructions unless they directly contradict the feedback
- Explain your reasoning clearly

Respond with JSON:
{{
  "strategy": "one of the 5 strategies above",
  "new_prompt": "the COMPLETE modified prompt (not just the diff)",
  "explanation": "what you changed and why",
  "predicted_impact": "what improvement you expect"
}}"""


class ModificationEngine:
    """Proposes and routes prompt modifications based on feedback patterns."""

    def __init__(self, db_url: str, prompt_registry, feedback_pipeline,
                 eval_sandbox=None):
        self._db_url = db_url
        self._registry = prompt_registry
        self._feedback = feedback_pipeline
        self._eval_sandbox = eval_sandbox
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(self._db_url, pool_size=2, max_overflow=1)
            except Exception:
                logger.warning("modification_engine: PostgreSQL unavailable", exc_info=True)
        return self._engine

    def _execute(self, query: str, params: dict = None) -> list:
        engine = self._get_engine()
        if not engine:
            return []
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                if result.returns_rows:
                    return [dict(row._mapping) for row in result]
                conn.commit()
                return []
        except Exception:
            logger.warning("modification_engine: query failed", exc_info=True)
            return []

    def process_triggered_patterns(self) -> list[dict]:
        """Main entry point.  Called by idle scheduler.

        For each triggered pattern:
        1. Check rate limits
        2. Generate hypothesis
        3. Route by tier
        """
        from app import idle_scheduler
        patterns = self._feedback.get_triggered_patterns()
        results = []

        for pattern in patterns:
            if idle_scheduler.should_yield():
                break

            target_param = pattern.get("target_parameter", "")
            target_role = pattern.get("target_role", "commander")
            tier = self._determine_tier(target_param)

            if not self._check_rate_limit(tier, target_role):
                logger.info(f"modification_engine: rate limit hit for {tier}/{target_role}")
                continue

            # Check cooldown
            if self._is_in_cooldown(target_role):
                logger.info(f"modification_engine: {target_role} is in cooldown")
                continue

            try:
                hypothesis = self._generate_hypothesis(pattern, target_role)
                if not hypothesis:
                    continue

                attempt_id = self._record_attempt(pattern, tier, target_role,
                                                   target_param, hypothesis)

                if tier == "tier1":
                    result = self._route_tier1(attempt_id, target_role, hypothesis)
                else:
                    result = self._route_tier2(attempt_id, target_role, hypothesis)

                results.append(result)
                self._feedback.mark_pattern_resolved(pattern["id"], attempt_id)

            except Exception:
                logger.warning(f"modification_engine: failed processing pattern {pattern.get('id')}",
                              exc_info=True)

        return results

    def _determine_tier(self, target_parameter: str) -> str:
        """Determine modification tier based on target parameter."""
        if target_parameter in TIER1_PARAMETERS:
            return "tier1"
        elif target_parameter in TIER2_PARAMETERS:
            return "tier2"
        else:
            return "tier1"  # default to Tier 1 for unknown parameters

    def _check_rate_limit(self, tier: str, target_role: str) -> bool:
        """Check if we're within rate limits.  Uses IMMUTABLE constants."""
        now = datetime.now(timezone.utc)
        day_ago = (now - timedelta(days=1)).isoformat()
        week_ago = (now - timedelta(days=7)).isoformat()

        if tier == "tier1":
            max_day, max_week = TIER1_MAX_PER_DAY, TIER1_MAX_PER_WEEK
        else:
            max_day, max_week = TIER2_MAX_PER_DAY, TIER2_MAX_PER_WEEK

        day_count = self._execute(
            """SELECT COUNT(*) as cnt FROM modification.attempts
               WHERE tier = :tier AND created_at >= :since""",
            {"tier": tier, "since": day_ago}
        )
        if day_count and day_count[0].get("cnt", 0) >= max_day:
            return False

        week_count = self._execute(
            """SELECT COUNT(*) as cnt FROM modification.attempts
               WHERE tier = :tier AND created_at >= :since""",
            {"tier": tier, "since": week_ago}
        )
        if week_count and week_count[0].get("cnt", 0) >= max_week:
            return False

        return True

    def _is_in_cooldown(self, target_role: str) -> bool:
        """Check if a role is in cooldown after a rejection."""
        rows = self._execute(
            """SELECT cooldown_until FROM modification.attempts
               WHERE target_role = :role AND cooldown_until IS NOT NULL
               ORDER BY created_at DESC LIMIT 1""",
            {"role": target_role}
        )
        if rows and rows[0].get("cooldown_until"):
            cooldown = rows[0]["cooldown_until"]
            if isinstance(cooldown, str):
                cooldown = datetime.fromisoformat(cooldown)
            if cooldown.tzinfo is None:
                cooldown = cooldown.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < cooldown
        return False

    def _generate_hypothesis(self, pattern: dict, target_role: str) -> dict | None:
        """Use mid-tier LLM to propose a concrete prompt change."""
        try:
            from app.llm_factory import create_cheap_vetting_llm

            current_prompt = self._registry.get_active_prompt(target_role)
            if not current_prompt:
                logger.warning(f"modification_engine: no active prompt for {target_role}")
                return None

            # Get example interactions
            examples = self._feedback.get_recent_feedback_for_role(target_role, n=5)
            examples_text = "\n".join([
                f"- Task: {e.get('original_task', '')[:200]}\n"
                f"  Response: {e.get('original_response', '')[:200]}\n"
                f"  Feedback: {e.get('direction', '')}"
                for e in examples[:3]
            ]) if examples else "No specific examples available."

            prompt = HYPOTHESIS_PROMPT.format(
                role=target_role,
                current_prompt=current_prompt[:3000],
                category=pattern.get("category", ""),
                direction=pattern.get("direction", ""),
                event_count=pattern.get("event_count", 0),
                severity="moderate",
                examples=examples_text,
            )

            llm = create_cheap_vetting_llm()
            raw = str(llm.call(prompt)).strip()

            if "{" in raw:
                json_str = raw[raw.index("{"):raw.rindex("}") + 1]
                result = json.loads(json_str)
                if "new_prompt" in result:
                    return result
        except Exception:
            logger.warning("modification_engine: hypothesis generation failed", exc_info=True)
        return None

    def _record_attempt(self, pattern: dict, tier: str, target_role: str,
                         target_param: str, hypothesis: dict) -> str:
        """Record a modification attempt in PostgreSQL."""
        attempt_id = str(uuid.uuid4())
        current_version = self._registry.get_active_version(target_role)

        # Propose the new version in the registry (but don't promote yet)
        proposed_version = self._registry.propose_version(
            target_role,
            hypothesis["new_prompt"],
            hypothesis.get("explanation", "Feedback-driven modification"),
        )

        self._execute(
            """INSERT INTO modification.attempts
               (id, pattern_id, tier, target_role, target_parameter, strategy,
                current_version, proposed_version, proposed_content, explanation, status)
               VALUES (:id, :pid, :tier, :role, :param, :strategy,
                       :curr, :prop, :content, :explanation, :status)""",
            {
                "id": attempt_id,
                "pid": pattern.get("id"),
                "tier": tier,
                "role": target_role,
                "param": target_param,
                "strategy": hypothesis.get("strategy", ""),
                "curr": current_version,
                "prop": proposed_version,
                "content": hypothesis["new_prompt"][:10000],
                "explanation": hypothesis.get("explanation", "")[:2000],
                "status": "pending",
            }
        )

        self._log_action(attempt_id, tier, target_role, "proposed",
                          hypothesis.get("explanation", ""))
        return attempt_id

    def _route_tier1(self, attempt_id: str, target_role: str,
                      hypothesis: dict) -> dict:
        """Submit directly to eval sandbox (autonomous modification)."""
        logger.info(f"modification_engine: Tier 1 — evaluating {target_role} modification")

        if self._eval_sandbox:
            # Get the attempt details
            attempts = self._execute(
                "SELECT * FROM modification.attempts WHERE id = :id",
                {"id": attempt_id}
            )
            if not attempts:
                return {"attempt_id": attempt_id, "status": "error"}

            attempt = attempts[0]
            eval_result = self._eval_sandbox.evaluate_modification(
                attempt_id,
                attempt["current_version"],
                attempt["proposed_version"],
                target_role,
            )

            if eval_result.get("approved"):
                self._promote(attempt_id, target_role, attempt["proposed_version"])
                return {"attempt_id": attempt_id, "status": "promoted"}
            else:
                self._reject(attempt_id, target_role, eval_result.get("reason", ""))
                return {"attempt_id": attempt_id, "status": "rejected"}
        else:
            # No eval sandbox — promote directly (bootstrap mode)
            attempts = self._execute(
                "SELECT proposed_version FROM modification.attempts WHERE id = :id",
                {"id": attempt_id}
            )
            if attempts:
                self._promote(attempt_id, target_role, attempts[0]["proposed_version"])
            return {"attempt_id": attempt_id, "status": "promoted_no_eval"}

    def _route_tier2(self, attempt_id: str, target_role: str,
                      hypothesis: dict) -> dict:
        """Send diff to owner via Signal for approval."""
        logger.info(f"modification_engine: Tier 2 — requesting approval for {target_role}")

        self._execute(
            "UPDATE modification.attempts SET status = 'awaiting_approval' WHERE id = :id",
            {"id": attempt_id}
        )

        # Send approval request via Signal
        try:
            from app.signal_client import SignalClient
            from app.config import get_settings
            s = get_settings()
            client = SignalClient()

            diff_preview = hypothesis.get("explanation", "No explanation")[:500]
            msg = (
                f"🔧 Prompt modification proposal (Tier 2)\n\n"
                f"Role: {target_role}\n"
                f"Strategy: {hypothesis.get('strategy', 'unknown')}\n"
                f"Change: {diff_preview}\n\n"
                f"Reply 'approve {attempt_id[:8]}' to accept or "
                f"'reject {attempt_id[:8]}' to decline."
            )
            client._send_sync(s.signal_owner_number, msg)
        except Exception:
            logger.warning("modification_engine: failed to send Tier 2 approval request", exc_info=True)

        self._log_action(attempt_id, "tier2", target_role, "awaiting_approval",
                          hypothesis.get("explanation", ""))
        return {"attempt_id": attempt_id, "status": "awaiting_approval"}

    def approve_tier2(self, attempt_id_prefix: str) -> bool:
        """Called when owner approves via Signal reply."""
        attempts = self._execute(
            """SELECT * FROM modification.attempts
               WHERE id::text LIKE :prefix AND status = 'awaiting_approval'
               ORDER BY created_at DESC LIMIT 1""",
            {"prefix": f"{attempt_id_prefix}%"}
        )
        if not attempts:
            return False

        attempt = attempts[0]
        if self._eval_sandbox:
            eval_result = self._eval_sandbox.evaluate_modification(
                str(attempt["id"]),
                attempt["current_version"],
                attempt["proposed_version"],
                attempt["target_role"],
            )
            if eval_result.get("approved"):
                self._promote(str(attempt["id"]), attempt["target_role"],
                              attempt["proposed_version"])
                return True
            else:
                self._reject(str(attempt["id"]), attempt["target_role"],
                              eval_result.get("reason", ""))
                return False
        else:
            self._promote(str(attempt["id"]), attempt["target_role"],
                          attempt["proposed_version"])
            return True

    def reject_tier2(self, attempt_id_prefix: str, reason: str = "") -> bool:
        """Called when owner rejects via Signal reply."""
        attempts = self._execute(
            """SELECT * FROM modification.attempts
               WHERE id::text LIKE :prefix AND status = 'awaiting_approval'
               ORDER BY created_at DESC LIMIT 1""",
            {"prefix": f"{attempt_id_prefix}%"}
        )
        if not attempts:
            return False
        self._reject(str(attempts[0]["id"]), attempts[0]["target_role"], reason)
        return True

    def _promote(self, attempt_id: str, target_role: str, version: int) -> None:
        """Promote a proposed version to active."""
        self._registry.promote_version(target_role, version)
        self._execute(
            """UPDATE modification.attempts
               SET status = 'promoted', promoted_at = now()
               WHERE id = :id""",
            {"id": attempt_id}
        )
        self._log_action(attempt_id, "", target_role, "promoted",
                          f"v{version:03d} is now active")
        logger.info(f"modification_engine: promoted {target_role} to v{version:03d}")

    def _reject(self, attempt_id: str, target_role: str, reason: str) -> None:
        """Reject a modification and apply cooldown."""
        cooldown = datetime.now(timezone.utc) + timedelta(hours=REJECTION_COOLDOWN_HOURS)
        self._execute(
            """UPDATE modification.attempts
               SET status = 'rejected', evaluated_at = now(),
                   cooldown_until = :cooldown
               WHERE id = :id""",
            {"id": attempt_id, "cooldown": cooldown.isoformat()}
        )
        self._log_action(attempt_id, "", target_role, "rejected", reason)
        logger.info(f"modification_engine: rejected {target_role} — {reason[:80]}")

    def _log_action(self, attempt_id: str, tier: str, target_role: str,
                     action: str, detail: str) -> None:
        """Record an action in the modification log."""
        self._execute(
            """INSERT INTO modification.log (attempt_id, tier, target_role, action, detail)
               VALUES (:aid, :tier, :role, :action, :detail)""",
            {
                "aid": attempt_id,
                "tier": tier,
                "role": target_role,
                "action": action,
                "detail": (detail or "")[:2000],
            }
        )
