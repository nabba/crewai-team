"""Control-plane dashboard routes — ops_misc topic.

Operational telemetry catch-all: tasks, tokens, snapshots, tech-radar, chat, system-status, org-chart, research, health, signal-commands, idle, pool, notify.

Extracted from app/control_plane/dashboard_api.py as part of WP G
Phase 1 (productization plan, 2026-05-17). Pure code movement —
routes, classes, and helpers are verbatim. The parent router in
``dashboard_api.py`` re-attaches the ``/api/cp`` prefix and the
``require_gateway_auth`` dependency via ``include_router``, so the
URL surface and auth boundary are unchanged.
"""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# No prefix or dependencies here — the parent router in dashboard_api.py
# carries those, so every path below is identical to the original.
router = APIRouter()


# ── Module-level constants ──────────────────────────────────────────────
# Owned here for the system-status and chat handlers. ``_KNOWN_CREWS``
# is owned by ``dashboard_routes_budgets_costs`` (its primary consumer)
# and imported below for ``get_crew_tasks``.

from app.control_plane.dashboard_routes_budgets_costs import _KNOWN_CREWS  # noqa: F401

_PRIMARY_USER_ALIASES = frozenset({"andrus", "owner", "primary", "me", "user"})

_KNOWN_ERROR_HINTS: tuple[tuple[str, str], ...] = (
    ("connection refused", "Service is down or not listening"),
    ("timeout",            "Service did not respond in time"),
    ("name or service",    "DNS / hostname unresolved"),
    ("no route to host",   "Network unreachable"),
    ("unauthorized",       "Auth failed — check API key / credentials"),
    ("authentication",     "Auth failed — check API key / credentials"),
    ("permission denied",  "Auth / permission rejected"),
    ("forbidden",          "Forbidden — check API key / scopes"),
    ("insufficient",       "Out of credits — top up to continue"),
    ("payment required",   "Out of credits — top up to continue"),
    ("rate_limit",         "Provider rate-limited the request"),
    ("quota",              "Quota exhausted"),
    ("402",                "Out of credits — top up to continue"),
    ("401",                "Auth failed — check API key"),
    ("403",                "Forbidden — check API key / scopes"),
    ("404",                "Endpoint not found"),
    ("429",                "Rate-limited"),
    ("500",                "Upstream server error"),
    ("502",                "Upstream gateway error"),
    ("503",                "Upstream service unavailable"),
    ("504",                "Upstream gateway timeout"),
)


@router.get("/idle/jobs")
def list_idle_jobs():
    """Snapshot of every known idle job's current state.

    Closes the cross-area gap where the dashboard previously showed
    only Firebase heartbeats while the idle scheduler ran ~100 jobs
    invisibly. Each entry includes failure_count, in_cooldown,
    cooldown_until_ts, seconds_since_last_success/failure, and
    currently_running. Read-only; calling this never affects the
    scheduler's behaviour.

    The ``inbound_dlq`` block reports the load-shed dead-letter
    queue's active backend (memory or redis) and current depth, so
    the React pane has a single endpoint for "what's happening in
    the background?" without making operators inspect env vars.
    """
    from app.idle_scheduler import get_job_snapshot, is_enabled, is_idle
    out: dict = {
        "scheduler_enabled": is_enabled(),
        "scheduler_idle": is_idle(),
        "jobs": get_job_snapshot(),
    }
    try:
        from app.dead_letter_inbound import backend_info as _dlq_info
        out["inbound_dlq"] = _dlq_info()
    except Exception:
        # DLQ module is optional from the dashboard's POV — never block
        # the snapshot if the import or backend probe fails.
        pass
    return out


@router.get("/org-chart")
def get_org_chart_api():
    from app.control_plane.org_chart import get_org_chart
    return get_org_chart()


@router.get("/health")
def control_plane_health():
    """Aggregated system health for dashboard."""
    from app.control_plane.db import execute_scalar
    from app.control_plane.governance import get_governance
    ticket_count = execute_scalar("SELECT COUNT(*) FROM control_plane.tickets") or 0
    audit_count = execute_scalar("SELECT COUNT(*) FROM control_plane.audit_log") or 0
    pending = get_governance().pending_count()
    return {
        "status": "ok",
        "tickets_total": ticket_count,
        "audit_entries": audit_count,
        "governance_pending": pending,
    }


@router.get("/tech-radar")
def tech_radar(limit: int = Query(20, ge=1, le=100)):
    """Technology-discovery items collected during idle scans.

    Mirrors the parsing the Firestore publisher performs: memory items stored
    in ``scope_tech_radar`` follow ``[category] title: summary. Action: ...``.
    """
    import re as _re
    discoveries: list[dict] = []
    err: str | None = None
    try:
        from app.memory.scoped_memory import retrieve_operational
        items = retrieve_operational("scope_tech_radar", "technology discovery", n=limit) or []
        for item in items:
            text = item if isinstance(item, str) else str(item)
            m = _re.match(r'\[(\w+)\]\s*(.+?):\s*(.+?)(?:\.\s*Action:\s*(.+))?$', text, _re.DOTALL)
            if m:
                discoveries.append({
                    "category": m.group(1),
                    "title": m.group(2).strip(),
                    "summary": m.group(3).strip(),
                    "action": (m.group(4) or "").strip(),
                })
            else:
                discoveries.append({
                    "category": "unknown",
                    "title": text[:80],
                    "summary": text[:200],
                    "action": "",
                })
    except Exception as exc:
        err = str(exc)
        logger.debug("tech-radar endpoint: %s", exc)

    # Search backend health — lets the React app explain *why* the radar may
    # be quiet (e.g. "Brave quota exhausted, falling back to SearXNG").
    search_status: dict = {}
    try:
        from app.tools.web_search import get_search_status
        search_status = get_search_status()
    except Exception as exc:
        logger.debug("tech-radar endpoint: search status failed: %s", exc)

    return {
        "discoveries": discoveries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
        "search_status": search_status,
    }


@router.get("/tasks")
def get_crew_tasks(limit: int = Query(20, ge=1, le=200), project_id: str | None = Query(None)):
    """Return recent crew tasks + crew statuses + full agent roster.

    Reads the Control Plane `crew_tasks` table (Postgres).  The legacy
    Firestore `tasks` / `crews` collections are still mirrored by
    `app.firebase.crew_tracking` for backwards-observability but are no
    longer on the dashboard read path — no more 429 quota banners.

    - `tasks`  — last N tasks from crew_tasks (running or completed)
    - `crews`  — per-crew latest status derived from crew_tasks, merged
                 with the canonical registry so idle crews still show up
    - `agents` — the PostgreSQL org-chart roster so every agent /
                 subagent is represented even when idle
    """
    err: str | None = None
    tasks: list[dict] = []
    crews: list[dict] = []

    try:
        from app.control_plane.crew_tasks import list_recent, crew_statuses
        tasks = list_recent(limit=limit, project_id=project_id)
        crews = crew_statuses()
    except Exception as exc:
        err = f"tasks read: {exc}"
        logger.debug("tasks endpoint: %s", err)

    # Ensure every known crew appears in the list even if no tasks yet.
    # Each crew carries a "kind" tag so the dashboard can group user-
    # addressable crews separately from internal orchestration crews.
    known_kinds = {name: kind for name, kind in _KNOWN_CREWS}
    seen_crews = {c.get("name") for c in crews}
    for c in crews:
        name = c.get("name")
        if name in known_kinds and "kind" not in c:
            c["kind"] = known_kinds[name]
    for name, kind in _KNOWN_CREWS:
        if name not in seen_crews:
            crews.append({"name": name, "state": "idle", "kind": kind})

    agents: list[dict] = []
    try:
        from app.control_plane.org_chart import get_org_chart
        agents = get_org_chart() or []
    except Exception as exc:
        logger.debug("tasks endpoint: org_chart read failed: %s", exc)

    return {
        "tasks": tasks,
        "crews": crews,
        "agents": agents,
        "project_id": project_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": err,
    }


@router.get("/tasks/{task_id}/timeline")
def get_task_timeline(task_id: str):
    """Return the fine-grained execution flow of a single crew task.

    Response shape:
        {
          "task": { ...crew_tasks row... },
          "spans": [                          # nested tree
            {
              "id": 42,
              "span_type": "agent",
              "name": "Researcher",
              "started_at": "...",
              "completed_at": "...",
              "state": "running|completed|failed",
              "detail": {...},
              "children": [ ...spans... ]
            },
            ...
          ],
          "updated_at": "..."
        }

    Spans are populated by ``app.crews.span_events`` as CrewAI emits
    agent/tool/llm-call lifecycle events. Call this endpoint repeatedly
    (poll at 2s) while ``task.state == 'running'`` to watch the flow
    build; stop when state transitions to ``completed`` / ``failed``.
    """
    from app.control_plane.db import execute_one
    from app.control_plane.crew_task_spans import list_spans

    task: dict | None = None
    try:
        task = execute_one(
            """
            SELECT id, crew, project_id, state, summary, result_preview,
                   error, model, tokens_used, cost_usd,
                   parent_task_id, is_sub_agent,
                   delegated_from, delegated_to, delegation_reason,
                   started_at, completed_at, last_updated
              FROM control_plane.crew_tasks
             WHERE id = %s
            """,
            (task_id,),
        )
    except Exception as exc:
        logger.debug("tasks/timeline: crew_tasks lookup failed: %s", exc)

    if not task:
        raise HTTPException(status_code=404, detail=f"task {task_id!r} not found")

    spans_flat = list_spans(task_id)

    # Build the tree in-Python: group by parent_span_id, attach children
    # in started_at order. Root spans have parent_span_id IS NULL.
    by_id: dict[int, dict] = {}
    for s in spans_flat:
        s["children"] = []
        by_id[s["id"]] = s
    roots: list[dict] = []
    for s in spans_flat:
        parent_id = s.get("parent_span_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(s)
        else:
            roots.append(s)

    return {
        "task": task,
        "spans": roots,
        "span_count": len(spans_flat),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/observability/snapshots/{kind}/latest")
def observability_snapshot_latest(kind: str):
    """Return the most recent snapshot of the given ``kind``, or 404 if
    nothing has been recorded.  Shape: ``{ts, kind, payload}``."""
    from app.observability.snapshots import latest as _latest
    snap = _latest(kind)
    if snap is None:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot recorded for kind={kind!r}",
        )
    return {
        "ts": snap.ts.isoformat() if hasattr(snap.ts, "isoformat") else str(snap.ts),
        "kind": snap.kind,
        "payload": snap.payload,
    }


@router.get("/observability/snapshots/{kind}/recent")
def observability_snapshot_recent(
    kind: str,
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return the most recent ``limit`` snapshots for ``kind``, newest
    first.  Empty list if nothing has been recorded for that kind."""
    from app.observability.snapshots import recent as _recent
    snaps = _recent(kind, limit=limit)
    return {
        "kind": kind,
        "count": len(snaps),
        "items": [
            {
                "ts": s.ts.isoformat() if hasattr(s.ts, "isoformat") else str(s.ts),
                "payload": s.payload,
            }
            for s in snaps
        ],
    }


@router.get("/observability/snapshots")
def observability_snapshot_kinds():
    """Return every distinct ``kind`` currently recorded + its newest
    timestamp.  Useful for dashboards to discover what's available
    without enumerating hardcoded kinds.
    """
    try:
        from app.control_plane.db import execute
        rows = execute(
            """
            SELECT kind, MAX(ts) AS latest_ts, COUNT(*) AS count
              FROM observability_snapshots
          GROUP BY kind
          ORDER BY MAX(ts) DESC
            """,
            fetch=True,
        )
    except Exception:
        rows = []
    return {
        "kinds": [
            {
                "kind": r.get("kind") if isinstance(r, dict) else r[0],
                "latest_ts": (
                    (r.get("latest_ts") if isinstance(r, dict) else r[1]).isoformat()
                    if (r.get("latest_ts") if isinstance(r, dict) else r[1])
                    else None
                ),
                "count": r.get("count") if isinstance(r, dict) else r[2],
            }
            for r in (rows or [])
        ],
    }


@router.get("/research/adapters")
def research_adapter_status():
    """Show which paid research adapters are configured.

    Response shape::

      {
        "adapters": {
          "apollo":        {"configured": false, "env_key": "APOLLO_API_KEY"},
          "linkedin_data": {"configured": false, "env_key": "PROXYCURL_API_KEY"}
        },
        "source_priority_now": ["regulator", "company_site", "search"],
        "source_priority_full": ["regulator", "company_site", "apollo",
                                 "linkedin_data", "search"]
      }
    """
    try:
        from app.tools.research_orchestrator import (
            get_paid_adapter_status, default_source_priority,
        )
    except Exception as e:
        return {"error": f"orchestrator import failed: {e}"}
    status = get_paid_adapter_status()
    return {
        "adapters": {
            "apollo": {
                "configured": status.get("apollo", False),
                "env_key": "APOLLO_API_KEY",
                "provides": ["head_of_sales", "head_of_sales_linkedin",
                             "head_of_sales_email"],
                "homepage": "https://apollo.io/settings/api",
            },
            "linkedin_data": {
                "configured": status.get("linkedin_data", False),
                "env_key": "PROXYCURL_API_KEY",
                "provides": ["head_of_sales", "head_of_sales_linkedin"],
                "homepage": "https://nubela.co/proxycurl/api-keys",
            },
        },
        "source_priority_now": default_source_priority(),
        "source_priority_full": [
            "regulator", "company_site", "apollo", "linkedin_data", "search",
        ],
    }


class ChatSendBody(BaseModel):
    sender: str = "andrus"
    message: str


def _resolve_chat_sender(raw: str | None) -> str:
    """Resolve the chat-tab ``sender`` query value to the wire-level id
    Signal writes under so the React Chat tab and the Signal channel
    share one conversation thread.

    Signal messages are keyed on ``settings.signal_owner_number`` (a
    phone number); the React tab passes ``"andrus"`` for ergonomics.
    Without this translation the dashboard would read an empty bucket
    keyed on ``HMAC("andrus")`` while the real history sat under
    ``HMAC("+372...")``.
    """
    s = (raw or "").strip()
    if not s or s.lower() in _PRIMARY_USER_ALIASES:
        try:
            from app.config import get_settings
            owner = (get_settings().signal_owner_number or "").strip()
            if owner:
                return owner
        except Exception:
            pass
        return "andrus"
    return s


@router.get("/chat/messages")
def chat_messages(sender: str = Query("andrus"), limit: int = Query(50, ge=1, le=500)):
    """Recent conversation history with the given sender.

    Returns oldest-first so the React UI can append-render straight to a
    scroll area without reversing. Each entry: ``{role, content, ts}``.
    The ``sender`` query is resolved to the configured primary-user
    phone number when it equals "andrus" / "owner" / "me" so the React
    Chat tab sees the same thread Signal writes to.
    """
    resolved = _resolve_chat_sender(sender)
    try:
        from app.conversation_store import get_recent_messages
        rows = get_recent_messages(resolved, limit=limit)
    except Exception as exc:
        logger.debug("chat_messages: read failed: %s", exc)
        return {"sender": sender, "messages": [], "error": str(exc)}
    rows.reverse()  # store returns DESC; chat wants ASC for append
    return {"sender": sender, "messages": rows}


@router.post("/chat/send")
def chat_send(body: ChatSendBody):
    """Send a chat message — same dispatch path as a Signal message.

    Routes through ``Commander.handle()`` so every existing slash
    command, recovery loop, project router, and lifecycle hook fires
    exactly as if the message had arrived from Signal. Sender defaults
    to ``andrus`` and is resolved to ``settings.signal_owner_number``
    so the conversation thread is shared with Signal.
    """
    text = (body.message or "").strip()
    if not text:
        raise HTTPException(400, "empty message")
    sender = _resolve_chat_sender(body.sender)
    try:
        from app.agents.commander import Commander
    except Exception as exc:
        raise HTTPException(503, f"commander unavailable: {exc}")
    try:
        reply = Commander().handle(text, sender=sender) or ""
    except Exception as exc:
        logger.warning("chat_send: commander dispatch failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"dispatch failed: {exc}")
    return {"sender": sender, "message": text, "reply": str(reply)}


def _probe(name: str, category: str, fn, *, link: str | None = None) -> dict:
    """Run a probe with a soft deadline + uniform error surface."""
    import time as _t
    t0 = _t.monotonic()
    try:
        result = fn()
        latency_ms = int((_t.monotonic() - t0) * 1000)
        if isinstance(result, dict):
            result.setdefault("name", name)
            result.setdefault("category", category)
            result.setdefault("latency_ms", latency_ms)
            if link and "link" not in result:
                result["link"] = link
            return result
        return {
            "name": name, "category": category, "status": "ok",
            "message": str(result) if result else "responding",
            "latency_ms": latency_ms, "link": link,
        }
    except Exception as exc:
        return {
            "name": name, "category": category, "status": "error",
            "message": _interpret_error(exc),
            "link": _credit_link_for(exc) or link,
            "latency_ms": int((_t.monotonic() - t0) * 1000),
        }


def _interpret_error(exc: Exception | str) -> str:
    """Turn raw exceptions into a short human-readable hint."""
    raw = str(exc).strip()
    low = raw.lower()
    for needle, hint in _KNOWN_ERROR_HINTS:
        if needle in low:
            return f"{hint} — {raw[:160]}"
    return raw[:200] or type(exc).__name__ if isinstance(exc, Exception) else raw[:200]


def _credit_link_for(exc: Exception | str) -> str | None:
    """If the exception looks like credit exhaustion, return the
    provider's top-up URL so the dashboard can render a 'Top up' button."""
    try:
        from app.firebase.publish import detect_credit_error, _CREDIT_URLS
        provider = detect_credit_error(exc)
        if provider:
            return _CREDIT_URLS.get(provider)
    except Exception:
        pass
    return None


@router.get("/pool/diagnostics")
def pool_diagnostics():
    """Read-only pool diagnostics for the control_plane Postgres pool.

    Returns the thread-safe counter snapshot from
    ``app.control_plane.db.get_pool_diagnostics()`` plus the
    configured pool capacity so the dashboard can show
    "current_borrows / maxconn" at a glance.

    Counters track total acquires, slow acquires (> 0.5s), current and
    peak concurrent borrows, and per-failure-kind counts
    (pool_unavailable / pool_exhausted / pool_other / connection_error).
    These were added in PR 1 (2026-05-16) to investigate the dominant
    "connection pool exhausted" pattern in errors.jsonl without changing
    pool behavior.

    NB: counters are in-process. A gateway restart zeroes them. Use
    this endpoint together with a load window (e.g. poll for 15 min
    and watch peak_borrows / failures_pool_exhausted climb) to
    diagnose pressure, NOT for long-term aggregate trends.
    """
    import os

    try:
        from app.control_plane.db import get_pool_diagnostics
        diag = get_pool_diagnostics()
    except Exception as exc:
        return {"error": str(exc), "counters": None}

    # Surface the configured pool ceiling alongside the live counters
    # so the dashboard can show utilisation as a ratio without making a
    # second call. Mirrors the same env logic used by _create_pool_with_retry.
    try:
        maxconn = int(os.environ.get("CONTROL_PLANE_POOL_MAX", "24"))
    except (TypeError, ValueError):
        maxconn = 24

    utilisation = (
        diag["current_borrows"] / maxconn if maxconn else 0.0
    )
    peak_utilisation = (
        diag["peak_borrows"] / maxconn if maxconn else 0.0
    )

    return {
        "counters": diag,
        "maxconn": maxconn,
        "utilisation": round(utilisation, 3),
        "peak_utilisation": round(peak_utilisation, 3),
    }


@router.get("/system-status")
def system_status():
    """Aggregated monitoring view used by the React /cp/monitor page.

    Each check is a row with ``status ∈ {ok, warn, error}``, a
    one-line ``message``, an optional ``link`` (used for credit-
    top-up CTAs), and a measured ``latency_ms``. Probes have a soft
    timeout and never raise — failures land as ``error`` rows.
    """
    checks: list[dict] = []

    # ── Containers ─────────────────────────────────────────────
    def _pg():
        from app.control_plane.db import execute_scalar
        n = execute_scalar("SELECT COUNT(*) FROM control_plane.budgets")
        return {"status": "ok", "message": f"connected · {n} budget rows"}
    checks.append(_probe("PostgreSQL (control plane)", "Containers", _pg))

    def _chroma():
        from app.memory.chromadb_manager import get_client
        client = get_client()
        cols = client.list_collections() if client else []
        return {"status": "ok", "message": f"connected · {len(cols)} collections"}
    checks.append(_probe("ChromaDB", "Containers", _chroma))

    def _neo4j():
        from app.subia.belief import neo4j_mirror
        drv = neo4j_mirror._get_driver()
        if drv is None:
            return {"status": "warn", "message": "driver not configured (NEO4J_URL unset?)"}
        with drv.session() as s:
            s.run("RETURN 1").consume()
        return {"status": "ok", "message": "connected"}
    checks.append(_probe("Neo4j", "Containers", _neo4j))

    def _gateway():
        return {"status": "ok", "message": "responding (you're reading me)"}
    checks.append(_probe("Gateway HTTP", "Containers", _gateway))

    # ── Gateways / messaging ───────────────────────────────────
    def _signal():
        """Probe vanilla signal-cli's --http JSON-RPC endpoint.

        Pre-2026-05-10 fix this hit GET /v1/about, which is the path
        used by the *separate* signal-cli-rest-api Docker wrapper —
        not vanilla signal-cli.  Vanilla signal-cli's --http mode is
        JSON-RPC at POST /api/v1/rpc.  The 404 from /v1/about
        triggered a false "Endpoint not found" ERROR in the System
        Monitor for any deployment using upstream signal-cli (the
        host-native dev path).
        """
        import json as _json
        import urllib.request as _u
        from app.config import get_settings
        s = get_settings()
        url = (s.signal_http_url or "").rstrip("/")
        if not url:
            return {"status": "warn", "message": "signal_http_url not configured"}
        body = _json.dumps({
            "jsonrpc": "2.0", "method": "version", "id": 1,
        }).encode("utf-8")
        req = _u.Request(
            f"{url}/api/v1/rpc",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _u.urlopen(req, timeout=2) as r:
            payload = _json.loads(r.read().decode("utf-8"))
        version = (payload.get("result") or {}).get("version") or "?"
        if "error" in payload:
            return {
                "status": "error",
                "message": f"rpc error: {payload['error']}",
            }
        return {
            "status": "ok",
            "message": f"daemon responding ({s.signal_owner_number} · v{version})",
        }
    checks.append(_probe("Signal-cli daemon", "Messaging", _signal))

    def _bridge():
        """Probe the host bridge by exercising the same code path the
        rest of the system uses to decide if it's wired.

        Pre-2026-05-10 fix this gated on the legacy ``bridge_enabled``
        setting (env ``BRIDGE_ENABLED=1``).  Since §27.3 the actual
        wiring uses per-agent ``BRIDGE_TOKEN_<AGENT>`` tokens via
        ``app.bridge_client.get_bridge(...)`` — ``BRIDGE_ENABLED`` is
        no longer the source of truth.  The probe was reporting
        "host bridge disabled" while the bridge was fully functional.
        """
        # Try the modern per-agent path first.  ``change_requests`` is
        # the canonical agent token used by the React /cp/changes
        # surface; if it's wired, the bridge is operational.
        try:
            from app.bridge_client import get_bridge
            client = get_bridge("change_requests")
            if client and client.is_available():
                return {
                    "status": "ok",
                    "message": "responding via per-agent token (change_requests)",
                }
        except Exception:
            pass

        # Fall back to the legacy enable check + raw /health probe so
        # laptop dev setups with BRIDGE_ENABLED=1 + BRIDGE_SHARED_SECRET
        # still surface a sensible status.
        import urllib.request as _u
        from app.config import get_settings
        s = get_settings()
        if not s.bridge_enabled:
            return {
                "status": "warn",
                "message": (
                    "host bridge: no per-agent BRIDGE_TOKEN_* tokens "
                    "configured AND BRIDGE_ENABLED=0"
                ),
            }
        url = f"http://{s.bridge_host}:{s.bridge_port}/health"
        with _u.urlopen(url, timeout=2) as r:
            ok = r.status == 200
        return {
            "status": "ok" if ok else "error",
            "message": f"port {s.bridge_port}: http {r.status}",
        }
    checks.append(_probe("Host bridge", "Messaging", _bridge))

    # ── Internal subsystems ────────────────────────────────────
    def _idle():
        from app.idle_scheduler import is_enabled, _currently_running_job
        running = _currently_running_job
        if not is_enabled():
            return {"status": "warn", "message": "background tasks OFF (Settings → Background tasks)"}
        msg = f"running · current: {running}" if running else "running · idle"
        return {"status": "ok", "message": msg}
    checks.append(_probe("Idle scheduler", "Internal", _idle))

    def _self_heal():
        """Self-heal journal status — count entries within a real time
        window, not the whole FIFO buffer.

        Pre-2026-05-10 fix this called ``get_recent_errors(50)`` which
        returns the last 50 entries regardless of timestamp.  So any
        install with historical errors (most have weeks-old residue
        from before fixes landed) was permanently in WARN — even when
        zero errors had been recorded in the last 24 h.  The "50
        recent errors, top: coding:BadRequestError×16" alert was
        technically true but misleading: those 50 entries spanned
        2026-04-02 → 2026-04-28 with NO entries from the past 24 h.

        Now: filter by ``ts`` against a 24 h window.  Only entries
        actually fired recently count toward the WARN status.  When
        the journal has historical residue but no fresh entries, the
        operator sees ``"no errors in last 24h (N historical in
        journal)"`` instead of a phantom WARN.
        """
        from datetime import datetime, timedelta, timezone

        from app.healing.error_diagnosis import get_recent_errors
        # Pull a generous slice so the time-window cut isn't truncated
        # by FIFO ordering on a busy day.
        all_recent = list(get_recent_errors(200) or [])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        def _ts_in_window(entry: dict) -> bool:
            ts_str = entry.get("ts") or ""
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return False
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts >= cutoff

        recent_24h = [e for e in all_recent if _ts_in_window(e)]
        if not recent_24h:
            total = len(all_recent)
            if total > 0:
                msg = f"no errors in last 24h ({total} historical in journal)"
            else:
                msg = "no recent errors"
            return {"status": "ok", "message": msg}

        # Compute top pattern over the WINDOW, not all-time, so the
        # operator sees what's actually firing now.
        windowed_patterns: dict[str, int] = {}
        for e in recent_24h:
            key = f"{e.get('crew', '?')}:{e.get('error_type', '?')}"
            windowed_patterns[key] = windowed_patterns.get(key, 0) + 1
        top = sorted(
            windowed_patterns.items(), key=lambda kv: -kv[1],
        )[:1]
        top_str = f", top: {top[0][0]}×{top[0][1]}" if top else ""
        return {
            "status": "warn",
            "message": f"{len(recent_24h)} errors in last 24h{top_str}",
        }
    checks.append(_probe("Self-heal journal", "Internal", _self_heal))

    def _budget_reconcile():
        from app.control_plane.db import execute_one
        row = execute_one(
            "SELECT MAX(updated_at) AS last_ts FROM control_plane.budgets WHERE spent_usd > 0"
        )
        if not row or not row.get("last_ts"):
            return {"status": "warn", "message": "no recent reconcile activity"}
        return {"status": "ok", "message": f"last write {row['last_ts']}"}
    checks.append(_probe("Budget reconcile", "Internal", _budget_reconcile))

    def _person_correlation():
        """Q4.2.2#6 — surface person-correlation idle-job health.
        Skipped silently when master switch is OFF (the stack is
        invisible to the operator until they opt in)."""
        try:
            from app.runtime_settings import get_person_correlation_enabled
            if not get_person_correlation_enabled():
                return {"status": "ok", "message": "disabled (L1 master OFF)"}
        except Exception:
            return {"status": "warn", "message": "master switch unreadable"}
        try:
            from app.companion.person_model import current_profile
            prof = current_profile() or {}
        except Exception:
            return {"status": "warn", "message": "profile read failed"}
        n = len(prof.get("people") or [])
        # Probe the history file age as a liveness signal.
        try:
            from app.companion.person_model import _default_history_path
            hp = _default_history_path()
            if hp.exists():
                from datetime import datetime, timezone
                mtime = datetime.fromtimestamp(hp.stat().st_mtime, tz=timezone.utc)
                age = datetime.now(timezone.utc) - mtime
                age_hours = age.total_seconds() / 3600.0
                if age_hours > 36:
                    return {
                        "status": "warn",
                        "message": (
                            f"{n} tracked · last compile "
                            f"{int(age_hours)}h ago (cadence 12h)"
                        ),
                    }
                return {
                    "status": "ok",
                    "message": (
                        f"{n} tracked · last compile {int(age_hours)}h ago"
                    ),
                }
        except Exception:
            pass
        return {"status": "ok", "message": f"{n} people tracked"}
    checks.append(_probe("Person correlation", "Internal", _person_correlation))

    # ── External services / credit alerts ──────────────────────
    try:
        from app.firebase.publish import _active_alerts, _CREDIT_URLS
        for provider, alert in _active_alerts.items():
            checks.append({
                "name": f"{provider.capitalize()} credit",
                "category": "External services",
                "status": "error",
                "message": (alert.get("error") or "credit exhausted")[:200],
                "link": alert.get("url") or _CREDIT_URLS.get(provider),
                "since": alert.get("ts"),
                "latency_ms": 0,
            })
        if not _active_alerts:
            checks.append({
                "name": "LLM provider credits",
                "category": "External services",
                "status": "ok",
                "message": "no active credit alerts",
                "link": None,
                "latency_ms": 0,
            })
    except Exception as exc:
        checks.append({
            "name": "Credit alerts feed",
            "category": "External services",
            "status": "warn",
            "message": _interpret_error(exc),
            "link": None,
            "latency_ms": 0,
        })

    # ── Roll-up summary (counts + worst status) ────────────────
    by_cat: dict[str, dict[str, int]] = {}
    worst = "ok"
    rank = {"ok": 0, "warn": 1, "error": 2}
    for c in checks:
        cat = c.get("category", "Other")
        st = c.get("status", "ok")
        bucket = by_cat.setdefault(cat, {"ok": 0, "warn": 0, "error": 0})
        bucket[st] = bucket.get(st, 0) + 1
        if rank.get(st, 0) > rank.get(worst, 0):
            worst = st

    return {
        "checks": checks,
        "by_category": by_cat,
        "overall": worst,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/signal-commands")
def signal_commands():
    """Hand-curated catalogue of every Signal slash / NL command.

    Source of truth: ``app/agents/commander/command_registry.py``.
    Used by the React /cp/chat sidebar so the user can scan the full
    surface without scrolling through ``commands.py`` source.
    """
    from app.agents.commander.command_registry import to_payload, categories
    return {
        "categories": categories(),
        "commands": to_payload(),
    }


