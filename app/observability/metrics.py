"""
metrics.py — Prometheus metric primitives for the BotArmy gateway.

Two layers:

  1. **HTTP metrics** are emitted automatically by ``prometheus-fastapi-
     instrumentator`` once :func:`register_metrics` runs at startup.  This
     gives us::

         http_requests_total{method, handler, status}
         http_request_duration_seconds_bucket{method, handler, le}

     plus the standard ``process_*``/``python_*`` tree.  This is what the
     gateway request-rate / latency / 5xx panels in the BotArmy Grafana
     dashboard graph.

  2. **Application metrics** are defined as module-level singletons here and
     incremented by the rest of the codebase:

       * :data:`LLM_REQUESTS_TOTAL` — every LLM call, labelled by tier /
         provider / model / status.  Wired in
         :mod:`app.observability.llm_events`, which is the single subscriber
         for CrewAI's LLM event bus and therefore covers every provider
         (native + LiteLLM-mediated).
       * :data:`LLM_REQUEST_DURATION_SECONDS` — companion histogram, labels
         tier / provider / model.  Latency is computed by pairing the
         ``LLMCallStartedEvent`` ``timestamp`` with the
         ``LLMCallCompletedEvent`` (or Failed) ``timestamp`` via ``call_id``.
       * :data:`LLM_CASCADE_ALL_TIERS_FAILED_TOTAL` — bumped from
         :mod:`app.recovery.loop` whenever the recovery loop exhausts every
         alternative strategy without a successful completion.
       * :data:`MEM0_POSTGRES_CONNECTION_ERRORS_TOTAL` — bumped from
         :mod:`app.memory.mem0_manager` when init or a call against the
         Mem0 backend raises.

Why singletons?
---------------
``prometheus_client.Counter`` and ``Histogram`` are global by design — the
exposition format requires unique metric names per process.  Importing this
module from N places gives N references to the same object, which is what
we want.

The metrics are exposed on ``/metrics`` (plain-text exposition).  The
chart's ``ServiceMonitor`` (templates/servicemonitor.yaml, gated on
``monitoring.serviceMonitor.enabled``) tells Prometheus to scrape that
endpoint every 30 s.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import Counter, Histogram

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


# ── LLM cascade metrics ──────────────────────────────────────────────
# Labels are chosen to match what the BotArmy Grafana dashboard panels
# query against (`sum(rate(llm_requests_total[1m])) by (tier)`).  Adding
# `provider` + `model` is cheap (small label cardinality — one model per
# tier roughly) and lets us slice by provider when an outage hits one.

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM API calls, labelled by tier/provider/model/status.",
    ["tier", "provider", "model", "status"],
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "End-to-end LLM API call latency, sliced by tier/provider/model.",
    ["tier", "provider", "model"],
    # Buckets cover the spectrum from a fast Haiku token to a long Opus
    # tool-use chain.  Above 300 s we're already in timeout territory.
    buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300),
)

LLM_CASCADE_ALL_TIERS_FAILED_TOTAL = Counter(
    "llm_cascade_all_tiers_failed_total",
    "Recovery loop exhausted every alternative strategy without a successful "
    "completion. Triggers the BotArmyLLMCascadeAllFailing alert.",
)


# ── Memory backend metrics ───────────────────────────────────────────
# Cheap counter — only increments on the rare error path.  Used by the
# BotArmyPostgresDown alert as the secondary signal (the primary is
# `up{job=...-postgres} == 0`, which only works for in-cluster Postgres;
# managed RDS / Cloud SQL has no in-cluster pod for that probe).

MEM0_POSTGRES_CONNECTION_ERRORS_TOTAL = Counter(
    "mem0_postgres_connection_errors_total",
    "Failures connecting to or operating against the Mem0 Postgres backend "
    "(init failures + per-call exceptions).",
)


# ── FastAPI integration ──────────────────────────────────────────────

_registered: bool = False


def register_metrics(app: "FastAPI") -> None:
    """Attach the prometheus-fastapi-instrumentator to ``app`` and expose
    ``/metrics``.

    Idempotent — re-registration is a no-op (matches the pattern in
    :mod:`app.observability.llm_events`).

    Call once after the FastAPI ``app`` instance is constructed.  Safe to
    call when prometheus-fastapi-instrumentator isn't installed: the
    function logs a warning and returns instead of crashing the gateway.
    """
    global _registered
    if _registered:
        return

    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError:
        logger.warning(
            "metrics: prometheus-fastapi-instrumentator not installed; "
            "/metrics endpoint will not be available. "
            "Install via `pip install prometheus-fastapi-instrumentator`."
        )
        return

    # Default config: skips paths starting with /metrics (avoids
    # self-recursion) and groups handlers by path template (so
    # /tasks/{task_id} doesn't explode the label space).
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,    # don't pollute the OpenAPI spec
        tags=["observability"],
    )

    _registered = True
    logger.info(
        "metrics: registered prometheus-fastapi-instrumentator on /metrics "
        "(http_requests_total + http_request_duration_seconds + "
        "process_* + python_*) and exposed application metrics defined in "
        "app.observability.metrics."
    )
