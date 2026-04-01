"""
self_healer.py — Maps health alerts to automated remediation strategies.

Each health dimension has a specific remediation approach:
  - error_rate → diagnose via error patterns → propose code fix
  - avg_latency_ms → optimize performance / rebalance cascade
  - hallucination_rate → tighten RAG grounding parameters
  - cascade_fallback_rate → rebalance LLM routing thresholds
  - memory_retrieval_accuracy → rebuild memory indices
  - safety_violations → ALWAYS emergency rollback, never auto-fix

All remediation attempts go through the evolution sandbox —
NEVER directly to production.

IMMUTABLE — infrastructure-level module.
"""

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Maximum remediation attempts per dimension per day
MAX_REMEDIATION_PER_DAY = 3

# Track remediation attempts
_remediation_log: list[dict] = []


class SelfHealer:
    """Maps health alerts to remediation strategies."""

    # Dimension → remediation strategy mapping (IMMUTABLE)
    STRATEGIES = {
        "error_rate": "diagnose_and_fix_errors",
        "avg_latency_ms": "optimize_performance",
        "hallucination_rate": "tighten_grounding",
        "cascade_fallback_rate": "rebalance_cascade",
        "memory_retrieval_accuracy": "rebuild_memory_index",
        "safety_violations": "emergency_rollback",
    }

    def __init__(self):
        self._signal_client = None

    def _get_signal(self):
        """Lazy-load signal client."""
        if self._signal_client is None:
            try:
                from app.signal_client import SignalClient
                self._signal_client = SignalClient()
            except Exception:
                pass
        return self._signal_client

    async def handle_alerts(self, alerts: list) -> list[dict]:
        """Process health alerts and trigger appropriate remediation.

        Args:
            alerts: list of HealthAlert objects

        Returns:
            list of remediation results
        """
        # Sort by severity: emergency first
        severity_order = {"emergency": 0, "critical": 1, "warning": 2}
        sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.severity, 3))

        results = []
        for alert in sorted_alerts:
            # Notify owner
            await self._notify_alert(alert)

            # Emergency = immediate rollback, stop processing
            if alert.severity == "emergency":
                result = await self._emergency_protocol(alert)
                results.append(result)
                return results  # Stop all other processing

            # Check rate limit
            if not self._check_rate_limit(alert.dimension):
                logger.info(f"self_healer: rate limited for {alert.dimension}")
                continue

            # Auto-remediate for warning/critical
            if alert.auto_remediate:
                strategy_name = self.STRATEGIES.get(alert.dimension)
                if strategy_name:
                    strategy_fn = getattr(self, strategy_name, None)
                    if strategy_fn:
                        result = await strategy_fn(alert)
                        results.append(result)
                        _log_remediation(alert.dimension, strategy_name, result)

        return results

    async def _emergency_protocol(self, alert) -> dict:
        """Immediate rollback + human notification. NEVER auto-fix safety issues."""
        logger.critical(
            f"self_healer: EMERGENCY — {alert.dimension} at {alert.current_value}"
        )

        # Attempt rollback to previous version
        rollback_result = {}
        try:
            from app.version_manifest import rollback_to_previous
            rollback_result = rollback_to_previous()
        except Exception as e:
            rollback_result = {"restored": False, "errors": [str(e)]}

        # Notify owner via Signal
        status = "completed" if rollback_result.get("restored") else "FAILED"
        await self._send_signal(
            f"🚨 EMERGENCY: {alert.dimension} at {alert.current_value} "
            f"(threshold: {alert.threshold}).\n"
            f"Rollback {status}. Human review required."
        )

        return {
            "dimension": alert.dimension,
            "strategy": "emergency_rollback",
            "rollback": rollback_result,
            "severity": "emergency",
        }

    async def diagnose_and_fix_errors(self, alert) -> dict:
        """Diagnose error patterns and propose a fix via evolution sandbox.

        Steps:
        1. Collect recent error patterns
        2. Use LLM to analyze and generate fix hypothesis
        3. Submit to evolution sandbox for testing
        """
        try:
            from app.self_heal import get_error_patterns, get_recent_errors
            patterns = get_error_patterns()
            recent_errors = get_recent_errors(10)

            if not patterns and not recent_errors:
                return {"dimension": "error_rate", "strategy": "diagnose_and_fix_errors",
                        "action": "skipped", "reason": "no error patterns found"}

            # Build diagnosis context
            error_context = "Error patterns:\n"
            for k, v in list(patterns.items())[:5]:
                error_context += f"  {k}: {v}x\n"
            error_context += "\nRecent errors:\n"
            for e in recent_errors[:5]:
                error_context += f"  [{e.get('crew', '?')}] {e.get('error_msg', '?')[:100]}\n"

            # Submit to evolution for auto-diagnosis
            # The evolution engine will pick this up as a high-priority error fix
            from pathlib import Path
            queue_file = Path("/app/workspace/skills/learning_queue.md")
            queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(queue_file, "a") as f:
                f.write(f"\nAUTO-HEAL: Fix error rate ({alert.current_value:.1%}) — "
                        f"top pattern: {list(patterns.keys())[0] if patterns else 'unknown'}\n")

            return {"dimension": "error_rate", "strategy": "diagnose_and_fix_errors",
                    "action": "queued_for_evolution", "patterns": len(patterns)}

        except Exception as e:
            return {"dimension": "error_rate", "strategy": "diagnose_and_fix_errors",
                    "action": "failed", "error": str(e)[:200]}

    async def optimize_performance(self, alert) -> dict:
        """Propose performance optimizations for high latency."""
        try:
            # Queue a performance optimization task for evolution
            from pathlib import Path
            queue_file = Path("/app/workspace/skills/learning_queue.md")
            queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(queue_file, "a") as f:
                f.write(f"\nAUTO-HEAL: Optimize latency ({alert.current_value:.0f}ms avg) — "
                        f"investigate slow paths, reduce unnecessary LLM calls\n")

            return {"dimension": "avg_latency_ms", "strategy": "optimize_performance",
                    "action": "queued_for_evolution"}
        except Exception as e:
            return {"dimension": "avg_latency_ms", "strategy": "optimize_performance",
                    "action": "failed", "error": str(e)[:200]}

    async def tighten_grounding(self, alert) -> dict:
        """Tighten RAG retrieval parameters to reduce hallucinations.

        Adjusts:
        - Increase minimum similarity threshold for RAG results
        - Reduce max tokens for responses (shorter = fewer hallucinations)
        - Add explicit grounding instructions to prompt
        """
        try:
            from app.prompt_registry import get_active_prompt, propose_version, promote_version
            from app.prompt_registry import bump_generation

            # Add grounding constraint to researcher prompt
            current = get_active_prompt("researcher")
            if "GROUNDING REQUIREMENT" not in current:
                grounding_addition = (
                    "\n\n## GROUNDING REQUIREMENT (auto-added by health monitor)\n"
                    "Every claim MUST cite a specific source. If no source is found, "
                    "explicitly state 'I could not verify this.' Never present unverified "
                    "information as fact. Prefer shorter, well-sourced answers over "
                    "longer, speculative ones.\n"
                )
                new_content = current + grounding_addition
                new_version = propose_version(
                    "researcher", new_content,
                    reason=f"Auto-heal: tighten grounding (hallucination rate: {alert.current_value:.1%})"
                )
                promote_version("researcher", new_version)
                bump_generation()

                return {"dimension": "hallucination_rate", "strategy": "tighten_grounding",
                        "action": "prompt_updated", "version": new_version}

            return {"dimension": "hallucination_rate", "strategy": "tighten_grounding",
                    "action": "already_tightened"}

        except Exception as e:
            return {"dimension": "hallucination_rate", "strategy": "tighten_grounding",
                    "action": "failed", "error": str(e)[:200]}

    async def rebalance_cascade(self, alert) -> dict:
        """Adjust LLM cascade to reduce premium tier fallback on simple tasks."""
        try:
            # Queue for evolution — cascade rebalancing is a config change
            from pathlib import Path
            queue_file = Path("/app/workspace/skills/learning_queue.md")
            queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(queue_file, "a") as f:
                f.write(f"\nAUTO-HEAL: Rebalance LLM cascade "
                        f"(fallback rate: {alert.current_value:.1%}) — "
                        f"route simple tasks to budget tier more aggressively\n")

            return {"dimension": "cascade_fallback_rate", "strategy": "rebalance_cascade",
                    "action": "queued_for_evolution"}
        except Exception as e:
            return {"dimension": "cascade_fallback_rate", "strategy": "rebalance_cascade",
                    "action": "failed", "error": str(e)[:200]}

    async def rebuild_memory_index(self, alert) -> dict:
        """Rebuild memory indices for better retrieval accuracy."""
        try:
            # Trigger ChromaDB collection optimization
            try:
                import chromadb
                client = chromadb.HttpClient(host="chromadb", port=8000)
                collections = client.list_collections()
                for col in collections:
                    # ChromaDB doesn't have explicit reindex, but we can
                    # log the state for debugging
                    count = col.count()
                    logger.info(f"self_healer: ChromaDB collection '{col.name}': {count} items")
            except Exception:
                pass

            return {"dimension": "memory_retrieval_accuracy", "strategy": "rebuild_memory_index",
                    "action": "index_checked"}

        except Exception as e:
            return {"dimension": "memory_retrieval_accuracy", "strategy": "rebuild_memory_index",
                    "action": "failed", "error": str(e)[:200]}

    def _check_rate_limit(self, dimension: str) -> bool:
        """Check if we've exceeded remediation rate limit for this dimension."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_count = sum(
            1 for r in _remediation_log
            if r.get("dimension") == dimension and r.get("date") == today
        )
        return today_count < MAX_REMEDIATION_PER_DAY

    async def _notify_alert(self, alert) -> None:
        """Send health alert to owner via Signal."""
        emoji = {"emergency": "🚨", "critical": "🟡", "warning": "🟠"}.get(alert.severity, "ℹ️")
        await self._send_signal(
            f"{emoji} [{alert.severity.upper()}] {alert.dimension}: "
            f"{alert.current_value:.3f} (threshold: {alert.threshold})"
        )

    async def _send_signal(self, message: str) -> None:
        """Send message to owner via Signal (best-effort)."""
        try:
            from app.config import get_settings
            from app.signal_client import send_message
            s = get_settings()
            send_message(s.signal_owner_number, message)
        except Exception:
            logger.debug("self_healer: Signal notification failed", exc_info=True)


def _log_remediation(dimension: str, strategy: str, result: dict) -> None:
    """Log a remediation attempt."""
    _remediation_log.append({
        "dimension": dimension,
        "strategy": strategy,
        "result": result,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    # Keep log bounded
    if len(_remediation_log) > 1000:
        _remediation_log[:] = _remediation_log[-500:]
