"""Signal voting for architecture requests.

Mirrors :mod:`app.change_requests.signal` at the package-granularity
tier. An ASK message is sent to the configured Signal owner with the
proposal summary + 👍/👎 prompt; the message timestamp is recorded
on the request so the reaction handler in :mod:`app.main` can
correlate.

Reaction-handling lives in ``app/main.py`` (Signal endpoint
``/signal/inbound``). When the user reacts on a message timestamp
that matches a pending architecture request, the handler dispatches
to :func:`app.architecture_requests.lifecycle.approve` or
:func:`reject`. After approval, the operator typically also clicks
"scaffold" in the React UI to materialise the stub package — that's
a separate step from approval, since scaffolding produces files the
operator may want to review before per-file CRs are filed.

Message body shape:

  - Header: intent + requestor (always full).
  - Package + file-layout summary (capped at 6 files; "+N more" if
    longer — full layout in /api/cp/architecture-requests/{id}).
  - Integration points + env switches (capped).
  - Footer: 👍/👎 prompt + request id.
"""
from __future__ import annotations

import logging

from app.architecture_requests.models import ArchitectureRequest

logger = logging.getLogger(__name__)


_MAX_FILE_LAYOUT_LINES = 6
_MAX_INTEGRATION_LINES = 4
_MAX_ENV_SWITCH_LINES = 4
_MAX_INTENT_CHARS = 200
_MAX_MOTIVATION_CHARS = 600


def build_ask_body(req: ArchitectureRequest) -> str:
    """Compose the Signal message body for an architecture-request ASK."""
    intent = req.intent[:_MAX_INTENT_CHARS]
    motivation = req.motivation[:_MAX_MOTIVATION_CHARS]
    motivation_truncated = (
        "" if len(req.motivation) <= _MAX_MOTIVATION_CHARS
        else "…\n[motivation truncated]"
    )

    files_block = _render_file_layout(req)
    ip_block = _render_integration_points(req)
    env_block = _render_env_switches(req)

    body = (
        f"🏛 ARCHITECTURE REQUEST · {req.package_path}\n\n"
        f"From: {req.requestor}\n"
        f"Intent: {intent}\n\n"
        f"Motivation:\n{motivation}{motivation_truncated}\n\n"
        f"{files_block}\n"
        f"{ip_block}\n"
        f"{env_block}\n"
        f"Test plan: {req.test_plan[:300]}"
        f"{'…' if len(req.test_plan) > 300 else ''}\n\n"
        f"👍 to approve the design  ·  👎 to reject\n"
        f"id: {req.id}"
    )
    return body


def _render_file_layout(req: ArchitectureRequest) -> str:
    if not req.file_layout:
        return "Files: (empty)"
    lines = ["Files:"]
    for fs in req.file_layout[:_MAX_FILE_LAYOUT_LINES]:
        lines.append(f"  • {fs.path} — {fs.purpose}")
    extra = len(req.file_layout) - _MAX_FILE_LAYOUT_LINES
    if extra > 0:
        lines.append(f"  • (+{extra} more — see /api/cp/architecture-requests/{req.id})")
    return "\n".join(lines)


def _render_integration_points(req: ArchitectureRequest) -> str:
    if not req.integration_points:
        return "Integration points: (none)"
    lines = ["Integration points:"]
    for ip in req.integration_points[:_MAX_INTEGRATION_LINES]:
        lines.append(f"  • {ip.kind} → {ip.target_module}")
    extra = len(req.integration_points) - _MAX_INTEGRATION_LINES
    if extra > 0:
        lines.append(f"  • (+{extra} more)")
    return "\n".join(lines)


def _render_env_switches(req: ArchitectureRequest) -> str:
    if not req.env_switches:
        return "Env switches: (none)"
    lines = ["Env switches:"]
    items = list(req.env_switches.items())
    for name, default in items[:_MAX_ENV_SWITCH_LINES]:
        lines.append(f"  • {name} (default: {default})")
    extra = len(items) - _MAX_ENV_SWITCH_LINES
    if extra > 0:
        lines.append(f"  • (+{extra} more)")
    return "\n".join(lines)


def send_ask(request_id: str) -> int | None:
    """Send the ASK to Signal; record the message ts on the request.

    Returns the Signal message timestamp on success; None on any
    failure (config missing, send failed, store missing). The caller
    leaves the request in PROPOSED for manual operator approval via
    the React surface.
    """
    try:
        from app.architecture_requests import lifecycle, store
        from app.config import get_settings
        from app.signal_client import send_message_blocking
    except Exception as exc:  # noqa: BLE001
        logger.warning("architecture_requests.signal: imports failed: %s", exc)
        return None

    req = store.get(request_id)
    if req is None:
        logger.warning("architecture_requests.signal: %r not found", request_id)
        return None

    settings = get_settings()
    recipient = (getattr(settings, "signal_owner_number", "") or "").strip()
    if not recipient:
        logger.warning(
            "architecture_requests.signal: SIGNAL_OWNER_NUMBER not set; "
            "ASK not sent for %s. Operator must approve via React.",
            request_id,
        )
        return None

    body = build_ask_body(req)
    try:
        ts = send_message_blocking(recipient, body)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "architecture_requests.signal: send raised for %s: %s",
            request_id, exc,
        )
        return None
    if ts is None:
        logger.warning(
            "architecture_requests.signal: send returned None for %s", request_id,
        )
        return None

    try:
        lifecycle.attach_signal_ts(request_id, ts)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "architecture_requests.signal: attach_signal_ts failed for %s: %s",
            request_id, exc,
        )
    return ts


def find_request_by_signal_ts(signal_ts: int) -> str | None:
    """Resolver used by the Signal reaction handler in :mod:`app.main`.
    Returns the request id, or None if nothing matches."""
    try:
        from app.architecture_requests import store
    except Exception:
        return None
    return store.find_by_signal_ts(signal_ts)
