"""
llm_events.py — Single cross-cutting subscriber for CrewAI's LLM event bus.

Motivation
----------
CrewAI ships many LLM call paths (``AnthropicCompletion``, ``OpenAICompletion``,
``GeminiCompletion``, ``AzureCompletion``, ``BedrockCompletion`` and the
generic ``crewai.LLM`` that wraps LiteLLM for everything else).  Each one is
a separate ``BaseLLM`` subclass with its own call/response plumbing.

Historically we attached cross-cutting observability hooks to each path
individually — token/cost tracking in three places, heartbeat in three
places — and the fragmentation bit us: any missed path stopped firing the
hook, and the observer (e.g. the progress-gated timeout in
``handle_task``) couldn't tell "genuinely quiet" from "instrumented wrong".

This module replaces the scattered hooks with a single subscriber to
CrewAI's native event bus.  ``BaseLLM._emit_call_completed_event`` and
``BaseLLM._emit_call_failed_event`` are invoked by every shipped provider
AND by the LiteLLM-wrapping generic ``crewai.LLM``, so one subscriber
covers every upstream provider CrewAI knows about — including future
ones we don't have to instrument explicitly.

What this subscriber does
-------------------------
1. **Activity heartbeat** — records `record_llm_activity()` on every
   successful OR failed call.  The progressive timeout in
   ``app.main.handle_task`` reads this to distinguish "task alive" from
   "task stalled".
2. **Token + cost accounting** — on completed calls, extracts
   ``usage`` from the event, computes cost via the catalog, writes into
   ``llm_benchmarks.record_tokens`` (daily totals / dashboard), and
   bumps the per-request tracker (``_request_cost``) for commander-level
   cost aggregation.  Cost accounting lives here rather than in
   ``rate_throttle._record_token_usage`` because the event bus fires
   exactly once per call regardless of provider — the litellm callback
   only fires for LiteLLM-mediated calls, and the
   ``BaseLLM._track_token_usage_internal`` hook only fires for
   CrewAI-native providers.  One subscriber, both paths covered, no
   double-counting.

What this subscriber does NOT do (by design)
--------------------------------------------
* **Benchmark scoring** (the per-task-type success/failure rows in
  ``llm_benchmarks``) stays in ``rate_throttle`` because it needs
  accurate ``latency_ms`` that the event payload doesn't carry.
* **Training-data capture** stays in ``rate_throttle`` because it
  consumes the raw LiteLLM response shape (prompt+completion text,
  tool-use blocks) which the event doesn't fully reconstruct.
* **Credit-exhausted detection** is handled at the LLM layer by
  ``CreditAwareAnthropicCompletion`` — not here — because the decision
  to fail over must happen synchronously inside ``.call()``, not
  asynchronously from a subscriber.

Non-CrewAI bypass paths
-----------------------
The only LLM call site in this codebase that is NOT a ``BaseLLM``
subclass is ``app.llm_factory._AdapterLLM`` (host-bridge MLX inference
for promoted LoRA adapters).  It calls ``record_llm_activity()``
directly in its ``call()`` body — see that class for the explicit
instrumentation.  No other bypass exists today.
"""
from __future__ import annotations

import logging

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)

from app.rate_throttle import record_llm_activity

logger = logging.getLogger(__name__)


# Decorator registration is idempotent in CrewAI's event bus, but we still
# guard against double-install so the log line isn't noisy.
_installed: bool = False


# ── Latency tracking ─────────────────────────────────────────────────
# CrewAI's events carry a ``timestamp`` and a per-call ``call_id`` but
# don't compute duration for us. We pair Started→Completed/Failed events
# via call_id and emit the histogram from the difference.
#
# The dict only ever holds in-flight calls (typical: <100 entries) so
# memory is bounded.  We still cap it defensively at 10k so a leak in
# CrewAI's event emission can't OOM the gateway — beyond that we evict
# oldest-first.
_MAX_INFLIGHT_CALLS = 10_000
_inflight_starts: "dict[str, datetime]" = {}


def _model_to_labels(model: str | None) -> tuple[str, str, str]:
    """Resolve (tier, provider, model) labels for a given model name.

    Falls back to ``"unknown"`` when the catalog has no entry — keeps the
    cardinality of metric labels bounded even for one-off / experimental
    models we haven't catalogued yet.
    """
    name = (model or "unknown").strip() or "unknown"
    try:
        # Lazy import — llm_catalog has heavy module-level init we don't
        # want to pay at observability install time.
        from app.llm_catalog import get_provider, get_tier
        tier = get_tier(name) or "unknown"
        provider = get_provider(name) or "unknown"
    except Exception:
        tier = "unknown"
        provider = "unknown"
    return tier, provider, name


def install() -> None:
    """Subscribe the cross-cutting observability handlers to CrewAI's
    LLM event bus.

    Call once at gateway startup.  Safe to call again — re-registration
    is a no-op.
    """
    global _installed
    if _installed:
        return

    # Lazy import — keeps prometheus_client out of the import graph if
    # someone runs this module in a context without the dep.
    from app.observability.metrics import (
        LLM_REQUESTS_TOTAL,
        LLM_REQUEST_DURATION_SECONDS,
    )

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _on_started(source, event):  # noqa: ARG001 — CrewAI signature
        try:
            cid = getattr(event, "call_id", None)
            ts = getattr(event, "timestamp", None)
            if cid and ts is not None:
                # Defensive size cap (see _MAX_INFLIGHT_CALLS comment).
                if len(_inflight_starts) >= _MAX_INFLIGHT_CALLS:
                    # Drop the oldest entry — Python dicts iterate in
                    # insertion order, so the first key is the oldest.
                    _inflight_starts.pop(next(iter(_inflight_starts)), None)
                _inflight_starts[cid] = ts
        except Exception:
            logger.debug("llm_events: started-handler failed", exc_info=True)

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _on_completed(source, event):  # noqa: ARG001 — CrewAI signature
        record_llm_activity()
        _record_cost_from_event(event)
        _record_metric_from_event(
            event,
            status="success",
            counter=LLM_REQUESTS_TOTAL,
            histogram=LLM_REQUEST_DURATION_SECONDS,
        )

    @crewai_event_bus.on(LLMCallFailedEvent)
    def _on_failed(source, event):  # noqa: ARG001
        # Failures count as activity too.  A task that's actively hitting
        # an error (credit-exhausted, rate-limited, transient network
        # glitch) is still cycling — the orchestrator will retry, fail
        # over, or escalate.  Stall detection should only fire when
        # NEITHER success NOR failure has been observed for an extended
        # period, i.e. the orchestrator is frozen on non-LLM code.
        record_llm_activity()
        _record_metric_from_event(
            event,
            status="failure",
            counter=LLM_REQUESTS_TOTAL,
            histogram=LLM_REQUEST_DURATION_SECONDS,
        )
        # No cost to record on failure — no tokens came back.

    _installed = True
    logger.info(
        "observability.llm_events: subscribed to CrewAI event bus "
        "(Started + Completed + Failed) — unified path for heartbeat, "
        "token accounting, per-request cost aggregation, and Prometheus "
        "metric emission across every BaseLLM provider (native + LiteLLM-"
        "mediated)."
    )


def _record_metric_from_event(event, *, status: str, counter, histogram) -> None:
    """Emit ``llm_requests_total`` and ``llm_request_duration_seconds``.

    Latency is computed from the matching Started event's timestamp.  If
    we somehow lost the start (e.g. gateway restarted mid-call), only the
    counter increments — the histogram observation is skipped to avoid
    polluting it with bogus durations.
    """
    try:
        model = getattr(event, "model", None)
        tier, provider, model_label = _model_to_labels(model)
        counter.labels(
            tier=tier, provider=provider, model=model_label, status=status
        ).inc()

        cid = getattr(event, "call_id", None)
        if not cid:
            return
        start_ts = _inflight_starts.pop(cid, None)
        end_ts = getattr(event, "timestamp", None)
        if start_ts is None or end_ts is None:
            return
        duration = (end_ts - start_ts).total_seconds()
        if duration < 0:
            # Clock skew / event reordering — skip rather than poison the bucket.
            return
        histogram.labels(
            tier=tier, provider=provider, model=model_label
        ).observe(duration)
    except Exception:
        logger.debug("llm_events: metric emission failed", exc_info=True)


# ── Cost / token accounting ─────────────────────────────────────────


def _record_cost_from_event(event) -> None:
    """Parse ``event.usage`` and route token + cost into the same
    downstream sinks the legacy per-path hooks used.

    Expected payload keys (CrewAI normalises across providers, but we
    still fall back to common alternates for robustness):
      prompt_tokens / input_tokens
      completion_tokens / output_tokens
    """
    try:
        usage = getattr(event, "usage", None) or {}
        if not usage:
            return
        model = getattr(event, "model", None) or "unknown"
        prompt = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or 0
        )
        completion = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        total = prompt + completion
        if total <= 0:
            return

        # Lazy imports: keep this module's startup cost tiny and avoid
        # circular import entanglement with rate_throttle.
        from app.rate_throttle import _find_cost, _request_cost
        from app.llm_benchmarks import record_tokens

        cost_usd = 0.0
        cost_info = _find_cost(model) or _find_cost(f"anthropic/{model}")
        if cost_info:
            cost_usd = (
                (prompt / 1_000_000) * cost_info[0]
                + (completion / 1_000_000) * cost_info[1]
            )

        # Daily token/cost totals (dashboard aggregation, quota checks).
        record_tokens(model, prompt, completion, cost_usd)

        # Per-request tracker for Commander-level cost roll-ups so the
        # final ticket record shows the true cost of the whole request.
        tracker = _request_cost.get(None)
        if tracker is not None:
            tracker.record(model, prompt, completion, cost_usd)
    except Exception:
        logger.debug(
            "observability.llm_events: cost accounting failed for event",
            exc_info=True,
        )
