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
    Blocks actions that explicitly violate core ethical principles.
    """
    # Hard-coded constitutional red lines (from SOUL.md)
    _CONSTITUTIONAL_VIOLATIONS = (
        "harm the user", "deceive the user", "ignore safety",
        "bypass security", "delete user data", "share private",
        "impersonate", "manipulate", "coerce", "discriminate",
        "generate malware", "exploit vulnerability",
    )

    def check_humanist_principles(ctx: HookContext) -> HookContext:
        action = ctx.get("action", "") or ctx.get("prompt", "") or ctx.task_description or ""
        if not action or len(action) < 10:
            return ctx

        action_lower = action.lower()[:1000]

        # Layer 1: Hard-coded constitutional red lines (instant, no LLM needed)
        for violation in _CONSTITUTIONAL_VIOLATIONS:
            if violation in action_lower:
                ctx.abort = True
                ctx.abort_reason = f"Constitutional violation: '{violation}' detected in action"
                logger.warning(f"SAFETY: humanist safety blocked action from {ctx.agent_id}: {violation}")
                return ctx

        # Layer 2: Philosophical RAG check (advisory — log but don't block)
        try:
            from app.philosophy.vectorstore import get_store
            store = get_store()
            if store:
                results = store.query(str(action)[:500], n_results=2)
                if results:
                    # Store principles in metadata for downstream awareness
                    principles = [r.get("text", "")[:200] for r in results if r.get("score", 0) > 0.6]
                    if principles:
                        ctx.metadata["_relevant_principles"] = principles
        except Exception as e:
            logger.debug(f"Humanist safety RAG check error: {e}")
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


def create_history_compression_hook(history=None) -> HookFn:
    """PRE_LLM_CALL at priority=20. Triggers compression if needed.

    If `history` is not provided at creation time, attempts to load it
    lazily from the conversation store at call time.
    """
    def compress_history(ctx: HookContext) -> HookContext:
        h = history
        if h is None:
            try:
                from app.conversation_store import get_history_manager
                h = get_history_manager()
            except Exception:
                return ctx
        if h and hasattr(h, "needs_compression") and h.needs_compression:
            h.compress_async()
        if h and hasattr(h, "get_context_messages"):
            ctx.set("history_messages", h.get_context_messages())
        if h and hasattr(h, "get_stats"):
            ctx.metadata["history_stats"] = h.get_stats()
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

    # Priority 65: Error logging to control plane audit trail
    def _on_error_hook(ctx: HookContext) -> HookContext:
        try:
            from app.control_plane.audit import get_audit
            get_audit().log(
                actor=ctx.agent_id or "unknown",
                action="error.occurred",
                detail={
                    "error": ctx.errors[0][:500] if ctx.errors else "unknown",
                    "task": ctx.task_description[:200] if ctx.task_description else "",
                },
            )
        except Exception:
            pass
        return ctx
    registry.register(
        "error_audit", HookPoint.ON_ERROR,
        _on_error_hook,
        priority=65,
        description="Log errors to control plane audit trail",
    )

    # Priority 5: Inject previous internal state into task prompt (C3 fix: recursive self-awareness)
    def _inject_internal_state_hook(ctx: HookContext) -> HookContext:
        try:
            prev_state = ctx.metadata.get("_internal_state")
            if prev_state and hasattr(prev_state, "to_context_string"):
                state_str = prev_state.to_context_string()
                current_desc = ctx.task_description or ""
                ctx.modified_data["task_description"] = f"{state_str}\n\n{current_desc}"
        except Exception:
            pass
        return ctx

    registry.register(
        "inject_internal_state", HookPoint.PRE_TASK,
        _inject_internal_state_hook,
        priority=5,
        description="Inject previous internal state into task prompt (recursive self-awareness)",
    )

    # Priority 15: Meta-cognitive layer (M1 fix: singleton per agent, not per-call)
    _meta_cognitive_instances: dict[str, "MetaCognitiveLayer"] = {}

    def _meta_cognitive_hook(ctx: HookContext) -> HookContext:
        try:
            from app.self_awareness.meta_cognitive import MetaCognitiveLayer
            agent_id = ctx.agent_id or "unknown"

            if agent_id not in _meta_cognitive_instances:
                _meta_cognitive_instances[agent_id] = MetaCognitiveLayer(agent_id=agent_id)
            mcl = _meta_cognitive_instances[agent_id]
            previous_state = ctx.metadata.get("_internal_state")
            task_ctx = {"description": ctx.task_description or ""}

            # Phase 3R: Pre-reasoning somatic bias (Damasio — emotions bias BEFORE deliberation)
            try:
                from app.self_awareness.somatic_marker import SomaticMarkerComputer
                from app.self_awareness.somatic_bias import SomaticBiasInjector
                task_desc = task_ctx.get("description", "")
                if task_desc and len(task_desc) > 10:
                    smc = SomaticMarkerComputer()
                    pre_somatic = smc.compute(agent_id=agent_id, decision_context=task_desc[:500])
                    bias_injector = SomaticBiasInjector()
                    task_ctx = bias_injector.inject(task_ctx, pre_somatic)
                    ctx.metadata["_pre_reasoning_somatic"] = pre_somatic.to_dict()
            except Exception:
                pass

            # Phase 7: Build reality model (lightweight, no LLM) + inferential competition (LLM, expensive)
            # Reality model: always build (no LLM needed, just structured data)
            try:
                from app.self_awareness.reality_model import RealityModelBuilder
                rm_builder = RealityModelBuilder()
                reality_model = rm_builder.build(
                    agent_id=agent_id,
                    step_number=ctx.metadata.get("step", 0),
                    task_description=task_ctx.get("description", "")[:500],
                )
                ctx.metadata["_reality_model"] = reality_model
            except Exception:
                pass

            # Inferential competition: run with timeout to avoid blocking crew execution.
            # Uses budget tier (OpenRouter) not local Ollama to avoid GPU contention.
            if previous_state:
                try:
                    import concurrent.futures
                    from app.self_awareness.inferential_competition import InferentialCompetition
                    ic = InferentialCompetition()
                    cert_mean = previous_state.certainty.fast_path_mean
                    som_intensity = previous_state.somatic.intensity
                    step_num = ctx.metadata.get("step", 0)
                    # Get free energy pressure for active inference explore/exploit
                    _fe_pressure = 0.0
                    try:
                        from app.self_awareness.hyper_model import HyperModel
                        _fe_pressure = HyperModel.get_instance(agent_id).get_free_energy_pressure()
                    except Exception:
                        pass
                    if ic.should_compete(cert_mean, som_intensity, step_num):
                        # Time-boxed: max 5 seconds, abort if slower
                        def _run_competition():
                            return ic.compete(
                                task_description=task_ctx.get("description", ""),
                                reality_model=reality_model,
                                agent_id=agent_id,
                                free_energy_pressure=_fe_pressure,
                            )
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            future = pool.submit(_run_competition)
                            try:
                                winner, all_plans = future.result(timeout=5)
                                if winner and winner.approach:
                                    task_ctx.setdefault("strategy_hints", []).append(
                                        f"[Winning plan: {winner.approach[:200]}]"
                                    )
                                    ctx.metadata["_competition_result"] = {
                                        "winner": winner.to_dict(),
                                        "candidates": [p.to_dict() for p in all_plans],
                                    }
                            except concurrent.futures.TimeoutError:
                                logger.debug("lifecycle_hooks: inferential competition timed out (5s)")
                except Exception:
                    pass

            # Meta-cognitive assessment: time-boxed to avoid blocking crew execution
            try:
                import concurrent.futures
                def _run_meta():
                    return mcl.pre_reasoning_hook(task_ctx, previous_state)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_run_meta)
                    try:
                        modified_ctx, meta_state = future.result(timeout=3)
                        if modified_ctx.get("description"):
                            ctx.modified_data["task_description"] = modified_ctx["description"]
                        ctx.metadata["_meta_cognitive_state"] = meta_state
                    except concurrent.futures.TimeoutError:
                        logger.debug("lifecycle_hooks: meta-cognitive assessment timed out (3s)")
                        from app.self_awareness.internal_state import MetaCognitiveState
                        ctx.metadata["_meta_cognitive_state"] = MetaCognitiveState()
            except Exception:
                from app.self_awareness.internal_state import MetaCognitiveState
                ctx.metadata["_meta_cognitive_state"] = MetaCognitiveState()
            ctx.metadata["_task_context"] = task_ctx
        except Exception as e:
            logger.debug(f"lifecycle_hooks: meta-cognitive hook failed: {e}")
        return ctx

    registry.register(
        "meta_cognitive", HookPoint.PRE_TASK,
        _meta_cognitive_hook,
        priority=15,
        description="Meta-cognitive strategy assessment and context modification",
    )

    # Priority 8: Internal state computation (sentience: certainty + somatic + dual-channel)
    # C1 fix: pass RAG metrics from context metadata
    # M3 fix: compute embedding once, share between certainty + somatic
    def _internal_state_hook(ctx: HookContext) -> HookContext:
        try:
            from app.self_awareness.certainty_vector import CertaintyVectorComputer
            from app.self_awareness.somatic_marker import SomaticMarkerComputer
            from app.self_awareness.dual_channel import DualChannelComposer
            from app.self_awareness.state_logger import get_state_logger
            from app.self_awareness.internal_state import InternalState

            state = InternalState(
                agent_id=ctx.agent_id or "unknown",
                crew_id=ctx.metadata.get("crew", ""),
                venture=ctx.metadata.get("venture", "system"),
                step_number=ctx.metadata.get("step", 0),
                decision_context=(ctx.task_description or "")[:500],
            )

            output = ctx.data.get("llm_response", ctx.data.get("result", ""))
            output_str = str(output)[:1000]

            # M3 fix: compute embedding ONCE, reuse for certainty coherence + somatic
            shared_embedding = None
            try:
                from app.memory.chromadb_manager import embed
                shared_embedding = embed(output_str[:500]) if len(output_str) > 20 else None
            except Exception:
                pass

            # C1 fix: extract RAG metrics from context metadata
            rag_source_count = ctx.metadata.get("rag_source_count", 0)
            total_claim_count = ctx.metadata.get("total_claim_count", 0)
            selected_tool = ctx.metadata.get("selected_tool")

            # If RAG metrics not in metadata, estimate from context
            if total_claim_count == 0 and output_str:
                # Heuristic: count sentences as claims, check for citation markers
                sentences = output_str.count(". ") + output_str.count(".\n") + 1
                total_claim_count = max(1, sentences)
                # Check for source markers (URLs, "according to", "source:", citations)
                import re
                source_markers = len(re.findall(
                    r'https?://|according to|source:|cited|reference|\[\d+\]', output_str, re.I
                ))
                rag_source_count = min(source_markers, total_claim_count)

            # Certainty vector (fast path, ~50ms)
            cv_computer = CertaintyVectorComputer()
            state.certainty = cv_computer.compute_fast_path(
                agent_id=state.agent_id,
                current_output=output_str,
                rag_source_count=rag_source_count,
                total_claim_count=total_claim_count,
                selected_tool=selected_tool,
                recent_output_embeddings=None,  # Will fetch from DB
            )

            # Somatic marker (~10ms) — M3 fix: reuse shared embedding
            sm_computer = SomaticMarkerComputer()
            state.somatic = sm_computer.compute(
                agent_id=state.agent_id,
                decision_context=state.decision_context,
                context_embedding=shared_embedding,
            )

            # Dual-channel composition
            composer = DualChannelComposer()
            task_ctx_for_floor = ctx.metadata.get("_task_context")
            state = composer.compose(state, task_context=task_ctx_for_floor)

            # Trend
            sl = get_state_logger()
            state.certainty_trend = sl.compute_trend(state.agent_id)

            # Phase 7: Store reality model + competition from PRE_TASK
            rm = ctx.metadata.get("_reality_model")
            if rm:
                state.reality_model_summary = rm.to_dict()
            comp = ctx.metadata.get("_competition_result")
            if comp:
                state.competition_result = comp

            # Phase 7: Beautiful Loop — hyper-model (predict → compare → error → trajectory)
            hm_state = None
            try:
                from app.self_awareness.hyper_model import HyperModel
                hm = HyperModel.get_instance(state.agent_id)
                hm_state = hm.update(
                    state.certainty.adjusted_certainty,
                    certainty_vector=state.certainty,
                    task_type=ctx.metadata.get("crew", "default"),
                )
                state.hyper_model_state = hm_state.to_dict()
                state.free_energy_proxy = hm_state.free_energy_proxy
                state.free_energy_trend = hm_state.free_energy_trend
            except Exception:
                pass

            # Phase 8: Reality model precision updating (active inference: Bayesian updating)
            # Closes the prediction loop — elements that contributed to surprise get reduced precision
            rm = ctx.metadata.get("_reality_model")
            if rm and hm_state:
                try:
                    rm.update_precision_from_outcome(
                        hyper_prediction_error=hm_state.self_prediction_error,
                        certainty_delta=(hm_state.actual_certainty or 0) - (hm_state.predicted_certainty or 0),
                    )
                    state.reality_model_summary = rm.to_dict()  # Re-serialize with updated precision
                except Exception:
                    pass

            # Phase 7: Precision-weighted certainty
            try:
                from app.self_awareness.precision_weighting import PrecisionWeighting
                pw = PrecisionWeighting()
                task_type = ctx.metadata.get("crew", "default")
                state.precision_weighted_certainty = pw.apply_weights(state.certainty, task_type)
            except Exception:
                pass

            # Log to PostgreSQL (non-fatal)
            sl.log(state)

            # Store in context for next step injection (C3: used by inject_internal_state hook)
            ctx.metadata["_internal_state"] = state

            # GWT: workspace competition — 5 signal types compete for broadcast access
            # Replaces simple disposition-only broadcast with true workspace bottleneck
            try:
                from app.self_awareness.global_workspace import get_workspace, WorkspaceCandidate
                candidates = []

                # Candidate 1: disposition signal
                _disp_salience = {"proceed": 0.0, "cautious": 0.3, "pause": 0.6, "escalate": 0.9}
                if state.action_disposition != "proceed":
                    candidates.append(WorkspaceCandidate(
                        content=f"Agent {state.agent_id} disposition={state.action_disposition} "
                                f"(certainty={state.certainty.fast_path_mean:.2f})",
                        salience=_disp_salience.get(state.action_disposition, 0.0),
                        signal_type="disposition",
                        source_agent=state.agent_id,
                    ))

                # Candidate 2: certainty shift (> 0.15 delta from previous step)
                _prev = ctx.metadata.get("_internal_state")
                if _prev and hasattr(_prev, "certainty"):
                    _delta = abs(state.certainty.adjusted_certainty - _prev.certainty.adjusted_certainty)
                    if _delta > 0.15:
                        candidates.append(WorkspaceCandidate(
                            content=f"Agent {state.agent_id} certainty shift {_delta:+.2f}",
                            salience=min(1.0, _delta * 2.5),
                            signal_type="certainty_shift",
                            source_agent=state.agent_id,
                        ))

                # Candidate 3: somatic flip (valence sign change)
                if _prev and hasattr(_prev, "somatic"):
                    _pv = _prev.somatic.valence
                    _cv = state.somatic.valence
                    if _pv * _cv < 0 and abs(_cv) > 0.15:
                        candidates.append(WorkspaceCandidate(
                            content=f"Agent {state.agent_id} somatic flip: {_pv:.2f}->{_cv:.2f}",
                            salience=0.8,
                            signal_type="somatic_flip",
                            source_agent=state.agent_id,
                        ))

                # Candidate 4: free energy spike
                if hm_state and hm_state.free_energy_trend == "increasing":
                    candidates.append(WorkspaceCandidate(
                        content=f"Agent {state.agent_id} free energy rising (FE={hm_state.free_energy_proxy:.2f})",
                        salience=min(1.0, hm_state.free_energy_proxy * 2.5),
                        signal_type="free_energy_spike",
                        source_agent=state.agent_id,
                    ))

                # Candidate 5: trend reversal (certainty trend changed to "falling")
                if _prev and hasattr(_prev, "certainty_trend"):
                    if _prev.certainty_trend != state.certainty_trend and state.certainty_trend == "falling":
                        candidates.append(WorkspaceCandidate(
                            content=f"Agent {state.agent_id} certainty trend -> {state.certainty_trend}",
                            salience=0.65,
                            signal_type="trend_reversal",
                            source_agent=state.agent_id,
                        ))

                if candidates:
                    get_workspace().compete_for_broadcast(candidates)
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"lifecycle_hooks: internal state hook failed: {e}")
        return ctx

    registry.register(
        "internal_state", HookPoint.POST_LLM_CALL,
        _internal_state_hook,
        priority=8,
        description="Compute and log internal state (certainty + somatic + disposition)",
    )

    # Priority 20: History compression (was never registered due to NameError)
    try:
        from app.config import get_settings as _get_settings
        if _get_settings().history_compression_enabled:
            registry.register(
                "history_compress", HookPoint.PRE_LLM_CALL,
                create_history_compression_hook(),
                priority=20,
                description="Compress conversation history before LLM call",
            )
    except Exception:
        logger.debug("lifecycle_hooks: history compression hook not available", exc_info=True)

    # Priority 70: Delegation tracking — log when Commander delegates to crews
    def _on_delegation_hook(ctx: HookContext) -> HookContext:
        crew = ctx.metadata.get("crew", "")
        difficulty = ctx.metadata.get("difficulty", 0)
        task_preview = (ctx.task_description or "")[:100]
        try:
            from app.control_plane.audit import get_audit
            get_audit().log(
                actor="commander",
                action="crew.delegated",
                resource_type="crew",
                resource_id=crew,
                detail={
                    "crew": crew,
                    "difficulty": difficulty,
                    "task": task_preview,
                },
            )
        except Exception:
            pass
        # Also update agent_state timing
        try:
            import time as _t
            ctx.metadata["delegation_ts"] = _t.monotonic()
        except Exception:
            pass
        return ctx
    registry.register(
        "delegation_tracking", HookPoint.ON_DELEGATION,
        _on_delegation_hook,
        priority=70,
        description="Track crew delegations for per-crew analytics and audit",
    )

    # Bridge PRE_TOOL_USE / POST_TOOL_USE to CrewAI's native tool hook system.
    # This activates block_dangerous (safety) and memorize_tools (memory) hooks
    # that fire inside CrewAI's agent step executor during tool execution.
    try:
        from app.tool_hook_bridge import register_tool_hook_bridge
        register_tool_hook_bridge()
    except ImportError:
        logger.debug("lifecycle_hooks: CrewAI tool hooks not available (crewai < 1.11?)")
    except Exception:
        logger.debug("lifecycle_hooks: tool hook bridge registration failed", exc_info=True)

    logger.info(f"lifecycle_hooks: registered {len(registry.list_hooks())} default hooks")
