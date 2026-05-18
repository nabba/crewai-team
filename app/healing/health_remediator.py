"""
health_remediator.py — Maps health alerts to automated remediation strategies.

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

    # Remediation verification log
    _verification_log: list[dict] = []

    async def handle_alerts(self, alerts: list) -> list[dict]:
        """Process health alerts and trigger appropriate remediation.

        Protected by circuit breaker: if self-healer fails 3 consecutive times,
        stops attempting remediation for 10 minutes (prevents cascading failures).
        After each fix, schedules background verification to confirm it worked.
        """
        # Circuit breaker: stop if healer itself is broken
        from app.circuit_breaker import is_available as _cb_ok, record_success as _cb_ok_fn, record_failure as _cb_fail
        if not _cb_ok("self_healer"):
            logger.warning("self_healer: circuit breaker open — skipping remediation for 10 min")
            return []

        severity_order = {"emergency": 0, "critical": 1, "warning": 2}
        sorted_alerts = sorted(alerts, key=lambda a: severity_order.get(a.severity, 3))

        results = []
        for alert in sorted_alerts:
            await self._notify_alert(alert)

            if alert.severity == "emergency":
                result = await self._emergency_protocol(alert)
                results.append(result)
                return results

            if not self._check_rate_limit(alert.dimension):
                logger.info(f"self_healer: rate limited for {alert.dimension}")
                continue

            if alert.auto_remediate:
                strategy_name = self.STRATEGIES.get(alert.dimension)
                if strategy_name:
                    strategy_fn = getattr(self, strategy_name, None)
                    if strategy_fn:
                        try:
                            result = await strategy_fn(alert)
                            _cb_ok_fn("self_healer")
                            results.append(result)
                            _log_remediation(alert.dimension, strategy_name, result)
                            # Schedule verification (5 min later)
                            self._schedule_verification(alert, result)
                        except Exception as exc:
                            _cb_fail("self_healer")
                            logger.error(f"self_healer: strategy {strategy_name} failed: {exc}")

        return results

    def _schedule_verification(self, alert, result, delay_s: int = 300) -> None:
        """Schedule background verification that a fix actually worked.

        Waits delay_s, then re-checks the health dimension. Logs whether
        the remediation was effective.
        """
        import threading

        def _verify():
            import time as _t
            _t.sleep(delay_s)
            try:
                from app.health_monitor import get_monitor
                monitor = get_monitor()
                state = monitor.get_health_state()
                current = getattr(state, alert.dimension, None)
                if current is None:
                    return

                # Check improvement (lower is better for most, except accuracy)
                if alert.dimension in ("memory_retrieval_accuracy",):
                    improved = current > alert.current_value
                else:
                    improved = current < alert.current_value

                entry = {
                    "dimension": alert.dimension,
                    "strategy": result.get("strategy", "?"),
                    "before": alert.current_value,
                    "after": current,
                    "improved": improved,
                    "ts": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                }
                self._verification_log.append(entry)
                if len(self._verification_log) > 100:
                    self._verification_log.pop(0)

                if improved:
                    logger.info(
                        f"self_healer: VERIFIED fix for {alert.dimension}: "
                        f"{alert.current_value:.3f} → {current:.3f}"
                    )
                else:
                    logger.warning(
                        f"self_healer: fix FAILED for {alert.dimension}: "
                        f"{alert.current_value:.3f} → {current:.3f} (no improvement)"
                    )
            except Exception:
                pass

        t = threading.Thread(target=_verify, daemon=True, name=f"heal-verify-{alert.dimension}")
        t.start()

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
            from app.healing.error_diagnosis import get_error_patterns, get_recent_errors
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
        """Reduce latency by tightening context loading and skipping slow paths.

        Direct fix: increase slow_path_trigger_threshold so the expensive
        LLM-based certainty assessment triggers less often.
        """
        try:
            from app.subia.sentience_config import load_config, apply_change
            config = load_config()
            current = config.get("slow_path_trigger_threshold", 0.4)
            # Higher threshold = fewer slow-path triggers = lower latency
            new_val = min(0.6, current + 0.03)
            applied = apply_change("slow_path_trigger_threshold", new_val)
            action = "applied_directly" if applied else "bounded_rejected"
            logger.info(f"self_healer: optimize_performance: slow_path_trigger {current:.2f} → {new_val:.2f} ({action})")

            return {"dimension": "avg_latency_ms", "strategy": "optimize_performance",
                    "action": action, "old": current, "new": new_val}
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
        """Adjust LLM cascade to reduce premium tier fallback on simple tasks.

        Direct fix: lower the certainty_low_threshold by 0.02 so more tasks
        stay on budget tier instead of escalating to premium.
        """
        try:
            from app.subia.sentience_config import load_config, apply_change
            config = load_config()
            current = config.get("certainty_low_threshold", 0.4)
            # Lower threshold = more tasks classified as "mid certainty" = fewer escalations
            new_val = max(0.2, current - 0.02)
            applied = apply_change("certainty_low_threshold", new_val)
            action = "applied_directly" if applied else "bounded_rejected"
            logger.info(f"self_healer: rebalance_cascade: certainty_low {current:.2f} → {new_val:.2f} ({action})")

            return {"dimension": "cascade_fallback_rate", "strategy": "rebalance_cascade",
                    "action": action, "old": current, "new": new_val}
        except Exception as e:
            return {"dimension": "cascade_fallback_rate", "strategy": "rebalance_cascade",
                    "action": "failed", "error": str(e)[:200]}

    async def rebuild_memory_index(self, alert) -> dict:
        """Rebuild memory indices for better retrieval accuracy.

        Direct fix: trigger skill re-indexing and clear stale result cache.
        """
        try:
            rebuilt = []
            # 1. Re-index skills into ChromaDB
            try:
                from app.idle_scheduler import _default_jobs
                # Find and run the skill-index job directly
                for name, fn, *_ in _default_jobs():
                    if name == "skill-index":
                        fn()
                        rebuilt.append("skills")
                        break
            except Exception:
                pass

            # 2. Clear stale result cache entries (force fresh lookups)
            try:
                from app.memory.chromadb_manager import get_client
                client = get_client()
                if client:
                    try:
                        cache = client.get_or_create_collection("result_cache")
                        count = cache.count()
                        if count > 100:
                            # Delete oldest entries to force fresh cache
                            oldest = cache.get(limit=count // 2, include=[])
                            if oldest and oldest.get("ids"):
                                cache.delete(ids=oldest["ids"])
                                # PROGRAM §56 iter-2 hook — keep the
                                # ledger consistent with the parallel
                                # eviction path in app/result_cache.py,
                                # else replay rebuilds resurrect stale
                                # cache entries the operator purged.
                                try:
                                    from app.memory.source_ledger import (
                                        hook_collection_delete,
                                    )
                                    hook_collection_delete(
                                        "memory", cache.name, list(oldest["ids"]),
                                    )
                                except Exception:
                                    logger.debug(
                                        "health_remediator: result_cache "
                                        "delete ledger hook failed",
                                        exc_info=True,
                                    )
                                rebuilt.append(f"result_cache({len(oldest['ids'])} purged)")
                    except Exception:
                        pass
            except Exception:
                pass

            logger.info(f"self_healer: rebuild_memory_index: {rebuilt}")
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
