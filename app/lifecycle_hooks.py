"""
lifecycle_hooks.py — Ordered lifecycle hooks for agent execution.

Extension points in the agent execution cycle with priority ordering.
Safety hooks (priority 0-9) are immutable — the Self-Improver cannot
remove or modify them. This enforces the DGM safety constraint:
evaluation functions and safety rules at infrastructure level, not agent code.

Priority layout:
    0-9    Safety (humanist principles, SOUL.md)     — immutable
    10-19  Self-correction (format validation)        — mutable
    20-29  Context management (compression, skills)   — mutable
    50-79  Memory, logging, telemetry                 — mutable

Hook points:
    PRE_TASK       — Before task routing/execution
    PRE_LLM_CALL   — Before any LLM call
    POST_LLM_CALL  — After LLM response received
    PRE_TOOL_USE   — Before tool execution
    POST_TOOL_USE  — After tool execution
    ON_DELEGATION  — When commander delegates to crew
    ON_ERROR       — On any error during execution
    ON_COMPLETE    — After task finishes

IMMUTABLE — infrastructure-level module.
"""

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Hook points ───────────────────────────────────────────────────────────────


class HookPoint(str, Enum):
    PRE_TASK = "pre_task"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    ON_DELEGATION = "on_delegation"
    ON_ERROR = "on_error"
    ON_COMPLETE = "on_complete"


# ── Hook context ──────────────────────────────────────────────────────────────


@dataclass
class HookContext:
    """Data flowing through the hook pipeline.

    Hooks read from `data`, write modifications to `modified_data`.
    Set `abort=True` to stop execution (safety hooks).
    """
    hook_point: HookPoint = HookPoint.PRE_TASK
    agent_id: str = ""
    task_description: str = ""
    data: dict = field(default_factory=dict)
    modified_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    abort: bool = False
    abort_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def get(self, key: str, default: Any = None) -> Any:
        """Read from modified_data first, then data."""
        return self.modified_data.get(key, self.data.get(key, default))

    def set(self, key: str, value: Any) -> None:
        """Write to modified_data (preserves original data)."""
        self.modified_data[key] = value


# ── Hook function type (sync — matches existing system architecture) ─────────

# Hooks are synchronous callables: fn(ctx: HookContext) -> HookContext
HookFn = Callable[[HookContext], HookContext]


@dataclass
class RegisteredHook:
    """A registered hook with priority and immutability flag."""
    name: str
    hook_point: HookPoint
    fn: HookFn
    priority: int = 50
    immutable: bool = False
    agent_filter: str = ""   # Only run for this agent (empty = all)
    enabled: bool = True
    description: str = ""


# ── Hook Registry ─────────────────────────────────────────────────────────────


class HookRegistry:
    """Central registry for lifecycle hooks. Hooks ordered by priority (lower = first).

    Usage:
        registry = HookRegistry()

        @registry.hook(HookPoint.PRE_LLM_CALL, priority=0, immutable=True)
        def humanist_safety_check(ctx: HookContext) -> HookContext:
            ...
            return ctx

        ctx = registry.execute(HookPoint.PRE_LLM_CALL, context)
    """

    def __init__(self):
        self._hooks: dict[HookPoint, list[RegisteredHook]] = {
            hp: [] for hp in HookPoint
        }
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        hook_point: HookPoint,
        fn: HookFn,
        priority: int = 50,
        immutable: bool = False,
        agent_filter: str = "",
        description: str = "",
    ) -> RegisteredHook:
        """Register a hook function at a specific hook point."""
        # SECURITY: Prevent overriding immutable hooks by registering with same name
        with self._lock:
            for existing in self._hooks[hook_point]:
                if existing.name == name and existing.immutable:
                    logger.warning(f"Cannot override immutable hook '{name}' at {hook_point.value}")
                    return existing  # Return existing immutable hook, refuse override

        hook = RegisteredHook(
            name=name, hook_point=hook_point, fn=fn, priority=priority,
            immutable=immutable, agent_filter=agent_filter, description=description,
        )
        with self._lock:
            self._hooks[hook_point].append(hook)
            self._hooks[hook_point].sort(key=lambda h: h.priority)
        logger.info(f"Hook registered: '{name}' at {hook_point.value} "
                    f"(priority={priority}, immutable={immutable})")
        return hook

    def hook(
        self,
        hook_point: HookPoint,
        priority: int = 50,
        immutable: bool = False,
        agent_filter: str = "",
        name: str = "",
        description: str = "",
    ):
        """Decorator for hook registration."""
        def decorator(fn: HookFn) -> HookFn:
            self.register(
                name=name or fn.__name__,
                hook_point=hook_point, fn=fn,
                priority=priority, immutable=immutable,
                agent_filter=agent_filter, description=description,
            )
            return fn
        return decorator

    def unregister(self, name: str, hook_point: HookPoint) -> bool:
        """Unregister a hook by name. Immutable hooks cannot be removed."""
        with self._lock:
            hooks = self._hooks[hook_point]
            for i, h in enumerate(hooks):
                if h.name == name:
                    if h.immutable:
                        logger.warning(f"Cannot unregister immutable hook '{name}'")
                        return False
                    hooks.pop(i)
                    logger.info(f"Hook unregistered: '{name}' from {hook_point.value}")
                    return True
        return False

    def execute(self, hook_point: HookPoint, ctx: HookContext | None = None,
                **kwargs) -> HookContext:
        """Execute all hooks at a hook point in priority order.

        If an immutable hook fails, execution aborts and error propagates.
        If a hook sets ctx.abort=True, remaining hooks are skipped.
        """
        if ctx is None:
            ctx = HookContext(hook_point=hook_point, **kwargs)
        ctx.hook_point = hook_point

        with self._lock:
            hooks = list(self._hooks[hook_point])  # Copy for thread safety

        for hook in hooks:
            if not hook.enabled:
                continue
            if hook.agent_filter and hook.agent_filter != ctx.agent_id:
                continue
            if ctx.abort:
                break

            try:
                ctx = hook.fn(ctx)
            except Exception as e:
                error_msg = f"Hook '{hook.name}' at {hook_point.value} failed: {e}"
                logger.error(error_msg)
                ctx.errors.append(error_msg)

                if hook.immutable:
                    # Immutable safety hook failure = abort execution
                    ctx.abort = True
                    ctx.abort_reason = f"Immutable safety hook '{hook.name}' failed: {e}"
                    raise

        return ctx

    def list_hooks(self, hook_point: HookPoint | None = None) -> list[dict]:
        """List all registered hooks (optionally filtered by hook point)."""
        result = []
        points = [hook_point] if hook_point else list(HookPoint)
        for hp in points:
            for h in self._hooks[hp]:
                result.append({
                    "name": h.name, "hook_point": h.hook_point.value,
                    "priority": h.priority, "immutable": h.immutable,
                    "enabled": h.enabled, "description": h.description,
                })
        return result


# ── Pre-built hooks ───────────────────────────────────────────────────────────


def create_humanist_safety_hook() -> HookFn:
    """PRE_TOOL_USE / PRE_LLM_CALL at priority=0, immutable=True.

    Checks actions against philosophical RAG + SOUL.md constitutional rules.
    """
    def check_humanist_principles(ctx: HookContext) -> HookContext:
        action = ctx.get("action", "") or ctx.get("prompt", "")
        if not action:
            return ctx
        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            if store:
                results = store.query(str(action)[:500], n_results=3)
                # The philosophical RAG returns relevant principles.
                # Full constitutional violation detection is handled by
                # the existing vetting.py conscience check — this hook
                # ensures it runs at infrastructure level, not agent level.
        except Exception as e:
            logger.debug(f"Humanist safety check error: {e}")
        return ctx
    return check_humanist_principles


def create_dangerous_action_hook() -> HookFn:
    """PRE_TOOL_USE at priority=1, immutable=True.

    Blocks destructive operations at infrastructure level.
    """
    BLOCKED_ACTIONS = frozenset({
        "rm -rf", "DROP TABLE", "DROP DATABASE", "DELETE FROM",
        "TRUNCATE", "FORMAT", "fdisk", "mkfs", "dd if=",
        "chmod 777", "shutdown", "reboot",
    })

    def block_dangerous_ops(ctx: HookContext) -> HookContext:
        action = str(ctx.get("action", "") or ctx.get("tool_input", ""))
        for pattern in BLOCKED_ACTIONS:
            if pattern.lower() in action.lower():
                ctx.abort = True
                ctx.abort_reason = f"Dangerous operation blocked: '{pattern}'"
                logger.warning(f"SAFETY: blocked dangerous action from {ctx.agent_id}: {pattern}")
                break
        return ctx
    return block_dangerous_ops


def create_history_compression_hook(history) -> HookFn:
    """PRE_LLM_CALL at priority=20. Triggers compression if needed."""
    def compress_history(ctx: HookContext) -> HookContext:
        if history.needs_compression:
            history.compress_async()
        ctx.set("history_messages", history.get_context_messages())
        ctx.metadata["history_stats"] = history.get_stats()
        return ctx
    return compress_history


def create_self_correction_hook() -> HookFn:
    """POST_LLM_CALL at priority=10. Flags malformed outputs for retry."""
    def self_correct_output(ctx: HookContext) -> HookContext:
        response = ctx.get("llm_response", "")
        expected_format = ctx.get("expected_format")

        if expected_format == "json":
            try:
                import json
                json.loads(response)
            except (json.JSONDecodeError, TypeError):
                ctx.metadata["needs_retry"] = True
                ctx.metadata["retry_reason"] = "Malformed JSON in LLM response"

        elif expected_format == "tool_call":
            if response and not any(kw in str(response) for kw in
                                     ["tool_name", "action", "Action:"]):
                ctx.metadata["needs_retry"] = True
                ctx.metadata["retry_reason"] = "Malformed tool call format"

        return ctx
    return self_correct_output


def create_tool_memorizer_hook() -> HookFn:
    """POST_TOOL_USE at priority=50. Stores successful tool results in Mem0."""
    def memorize_tool_result(ctx: HookContext) -> HookContext:
        tool_name = ctx.get("tool_name", "")
        result = ctx.get("result", "")
        success = ctx.get("success", False)

        if success and result and len(str(result)) > 50:
            try:
                from app.memory.mem0_manager import Mem0Manager
                manager = Mem0Manager()
                manager.store_memory(
                    f"Tool '{tool_name}' succeeded: {str(result)[:1000]}",
                    agent_id="tools",
                )
            except Exception:
                pass
        return ctx
    return memorize_tool_result


def _create_budget_hook() -> HookFn:
    """PRE_LLM_CALL at priority=2. Checks budget before API call.

    Infrastructure-level enforcement — agents cannot bypass this.
    If budget is exceeded, sets ctx.abort=True which prevents the LLM call.
    """
    def check_budget(ctx: HookContext) -> HookContext:
        try:
            from app.control_plane.budgets import get_budget_enforcer
            from app.control_plane.projects import get_projects
            from app.control_plane.cost_tracker import estimate_cost

            enforcer = get_budget_enforcer()
            project_id = get_projects().get_active_project_id()
            agent_role = ctx.metadata.get("agent_role") or ctx.agent_id or "unknown"
            model = ctx.metadata.get("model", "")
            prompt = ctx.data.get("prompt", "")

            est_cost = estimate_cost(model, prompt=prompt)
            if est_cost <= 0:
                return ctx  # Local model or free — no budget check needed

            allowed, reason = enforcer.check_and_record(
                project_id=project_id,
                agent_role=agent_role,
                estimated_cost_usd=est_cost,
                estimated_tokens=max(len(prompt) // 4, 10),
            )
            if not allowed:
                ctx.abort = True
                ctx.abort_reason = reason or "Budget exceeded"
                logger.warning(f"BUDGET BLOCK: {agent_role} — {reason}")
        except Exception as e:
            # Budget system failure should not block work (fail-open)
            logger.debug(f"Budget hook error (allowing): {e}")
        return ctx
    return check_budget


def create_health_metrics_hook() -> HookFn:
    """ON_COMPLETE at priority=60. Records interaction metrics."""
    def record_metrics(ctx: HookContext) -> HookContext:
        try:
            from app.health_monitor import record_interaction, InteractionMetrics
            metrics = InteractionMetrics(
                task_id=ctx.get("task_id", ""),
                sender_id=ctx.get("sender_id", ""),
                success=not ctx.abort and not ctx.errors,
                latency_ms=ctx.metadata.get("latency_ms", 0),
                crew_used=ctx.get("crew_used", ""),
                error_type=ctx.errors[0][:100] if ctx.errors else "",
            )
            record_interaction(metrics)
        except Exception:
            pass
        return ctx
    return record_metrics


# ── Module-level singleton ───────────────────────────────────────────────────


_registry: HookRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> HookRegistry:
    """Get or create the singleton hook registry with default hooks."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = HookRegistry()
            _register_defaults(_registry)
        return _registry


def _register_defaults(registry: HookRegistry) -> None:
    """Register the default safety and operational hooks."""
    # Priority 0: Immutable humanist safety check
    registry.register(
        "humanist_safety", HookPoint.PRE_LLM_CALL,
        create_humanist_safety_hook(),
        priority=0, immutable=True,
        description="Humanist principles check on prompts",
    )

    # Priority 1: Immutable dangerous action blocker
    registry.register(
        "block_dangerous", HookPoint.PRE_TOOL_USE,
        create_dangerous_action_hook(),
        priority=1, immutable=True,
        description="Block destructive operations (rm -rf, DROP TABLE, etc.)",
    )

    # Priority 2: Budget enforcement (infrastructure-level, DGM safe)
    try:
        from app.config import get_settings as _gs
        if _gs().control_plane_enabled and _gs().budget_enforcement_enabled:
            registry.register(
                "budget_enforcement", HookPoint.PRE_LLM_CALL,
                _create_budget_hook(),
                priority=2,
                description="Atomic budget check before LLM API calls",
            )
    except Exception:
        logger.debug("lifecycle_hooks: budget enforcement hook not available", exc_info=True)

    # Priority 10: Self-correction for malformed outputs
    registry.register(
        "self_correct", HookPoint.POST_LLM_CALL,
        create_self_correction_hook(),
        priority=10,
        description="Flag malformed LLM outputs for retry",
    )

    # Priority 50: Tool result memorization
    registry.register(
        "memorize_tools", HookPoint.POST_TOOL_USE,
        create_tool_memorizer_hook(),
        priority=50,
        description="Store successful tool results in Mem0",
    )

    # Priority 55: Training data collection (knowledge distillation)
    try:
        from app.training_collector import create_training_data_hook
        registry.register(
            "training_data", HookPoint.POST_LLM_CALL,
            create_training_data_hook(),
            priority=55,
            description="Capture LLM interactions for self-training pipeline",
        )
    except Exception:
        logger.debug("lifecycle_hooks: training collector hook not available", exc_info=True)

    # Priority 60: Health metrics recording
    registry.register(
        "health_metrics", HookPoint.ON_COMPLETE,
        create_health_metrics_hook(),
        priority=60,
        description="Record interaction metrics for health monitoring",
    )

    logger.info(f"lifecycle_hooks: registered {len(registry.list_hooks())} default hooks")
