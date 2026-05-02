"""
signal_attachment.py — agent-callable Signal attachment delivery.

Exposes a single CrewAI BaseTool, ``signal_send_attachment``, that
sends one or more files from ``/app/workspace/output/`` to the
configured Signal owner number. Agents pair this with
``pdf_compose`` to deliver real artifacts after composing them —
closing the loop between "agent generates a report" and "user
opens the PDF in Signal."

Why this exists
---------------
Pre-2026-05-03 agents could write files to disk but had no way to
hand them to the user. Reports lived in workspace/output/ and
required manual `docker cp` + Signal-CLI invocation by a human
operator. Failure mode in production: agents wrote PDFs, declared
"the file is at /app/workspace/output/...", and the user never saw
them because Signal Bridge runs on the host, not in the container.

Design
------
* Recipient is HARD-PINNED to ``settings.signal_owner_number``. No
  arbitrary recipients — the tool can't be coerced into spamming
  third parties with agent-generated files.
* Files MUST live under ``/app/workspace/output/``. Anything else
  is rejected (path-traversal guard + explicit allowlist of the
  workspace subtree). Agents can only deliver what they could
  plausibly have authored.
* Container → host path translation is handled here: signal-cli
  runs on the host and needs the host-side absolute path (e.g.
  ``/Users/andrus/BotArmy/.../workspace/output/x.pdf``); the
  container sees ``/app/workspace/output/x.pdf``. We translate via
  ``settings.workspace_host_path`` and reject when the host-path
  setting is empty (otherwise signal-cli would silently fail to
  find the attachment).
* Best-effort delivery — Signal send failures are surfaced to the
  agent in the tool's return string so it can retry / fall back to
  describing the file inline. Never raises.

Safety properties
-----------------
* HARD recipient pin (no override). The configured owner is the
  only person who ever receives.
* HARD path scope (only workspace/output/ subtree). No arbitrary
  filesystem read.
* Per-call cap of 5 attachments + 25 MB total. signal-cli rejects
  larger payloads anyway; we surface the cap upfront so the agent
  doesn't write 30 PDFs and then silently lose them.
* Body text is bounded to 2000 chars. Long context goes IN the
  PDF, not in the Signal message.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Allowlisted output directory inside the container.
_ALLOWED_DIR = Path("/app/workspace/output").resolve()

# Per-call hard caps — signal-cli + Signal-protocol limits, plus
# UX safety (a 30-attachment dump is almost certainly a bug).
_MAX_ATTACHMENTS = 5
_MAX_TOTAL_BYTES = 25 * 1024 * 1024  # 25 MB
_MAX_BODY_CHARS = 2000


def _validate_attachments(paths: list[str]) -> tuple[list[Path], str | None]:
    """Resolve + validate caller-provided paths.

    Returns ``(valid_paths, error_or_None)``. On any rejection,
    error is a short human-readable string and valid_paths is the
    set that DID pass — caller can decide to send the partial set
    or abort.
    """
    if not paths:
        return [], "no attachments provided"
    if len(paths) > _MAX_ATTACHMENTS:
        return [], f"too many attachments ({len(paths)} > cap {_MAX_ATTACHMENTS})"

    valid: list[Path] = []
    errors: list[str] = []
    total_bytes = 0
    for p_str in paths:
        try:
            p = Path(str(p_str)).resolve()
        except (OSError, ValueError) as exc:
            errors.append(f"  {p_str}: cannot resolve ({exc})")
            continue

        # Must live under the allowed dir
        try:
            p.relative_to(_ALLOWED_DIR)
        except ValueError:
            errors.append(f"  {p}: outside {_ALLOWED_DIR}")
            continue

        if not p.exists():
            errors.append(f"  {p}: does not exist")
            continue

        if not p.is_file():
            errors.append(f"  {p}: not a regular file")
            continue

        try:
            size = p.stat().st_size
        except OSError as exc:
            errors.append(f"  {p}: stat failed ({exc})")
            continue

        if total_bytes + size > _MAX_TOTAL_BYTES:
            errors.append(
                f"  {p}: would exceed total size cap "
                f"({(total_bytes + size) / 1024 / 1024:.1f} MB > "
                f"{_MAX_TOTAL_BYTES / 1024 / 1024:.0f} MB)"
            )
            continue

        valid.append(p)
        total_bytes += size

    if errors and not valid:
        return [], "all attachments rejected:\n" + "\n".join(errors)
    if errors:
        # Mixed — some valid, some not. Caller might still want to send.
        return valid, "some attachments rejected:\n" + "\n".join(errors)
    return valid, None


def _container_to_host(container_paths: list[Path], workspace_host: str) -> list[str]:
    """Map ``/app/workspace/...`` → ``<host_workspace>/...`` so signal-cli
    (which runs on the host, not in the container) can read the files.
    """
    out: list[str] = []
    container_root = "/app/workspace"
    host_root = workspace_host.rstrip("/")
    for p in container_paths:
        s = str(p)
        if s.startswith(container_root):
            out.append(host_root + s[len(container_root):])
        else:
            # Shouldn't happen given the resolve() + relative_to() guard,
            # but if it does, surface as-is so the failure is loud.
            out.append(s)
    return out


# ── Public factory ──────────────────────────────────────────────────

def create_signal_attachment_tools(agent_id: str = "coder") -> list:
    """Build the Signal-attachment delivery tool for an agent.

    Returns ``[]`` when the configured Signal stack isn't reachable
    (no owner number, no host-workspace mapping). Returns
    ``[SignalSendAttachmentTool]`` when fully configured.
    """
    try:
        from app.config import get_settings
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        return []

    settings = get_settings()
    if not getattr(settings, "signal_owner_number", "").strip():
        logger.debug("signal_attachment: SIGNAL_OWNER_NUMBER not set — tool unavailable")
        return []
    if not getattr(settings, "workspace_host_path", "").strip():
        logger.debug("signal_attachment: WORKSPACE_HOST_PATH not set — tool unavailable")
        return []

    class _SendAttachmentInput(BaseModel):
        body: str = Field(
            description=(
                "Short text message accompanying the attachments. "
                "Max 2000 chars. Use this for a 1-3 sentence summary; "
                "long context goes IN the PDF, not in the Signal "
                "message body."
            ),
        )
        attachments: list[str] = Field(
            description=(
                "List of absolute paths under /app/workspace/output/. "
                "Up to 5 files, 25 MB total. Container paths "
                "(starting /app/workspace/...) are auto-translated "
                "to host paths for signal-cli. Anything outside "
                "/app/workspace/output/ is rejected."
            ),
        )

    class SignalSendAttachmentTool(BaseTool):
        name: str = "signal_send_attachment"
        description: str = (
            "Send a Signal message with one or more file attachments "
            "to the configured owner. USE THIS to deliver PDFs / "
            "CSVs / images / reports the agent has written to "
            "/app/workspace/output/. The user receives the files as "
            "Signal attachments they can open on their phone.\n\n"
            "Recipient is HARD-PINNED to the configured Signal owner "
            "number — there is no `to` parameter; do not try to "
            "specify one. Files MUST live under /app/workspace/output/; "
            "writes to anywhere else are rejected. Per-call caps: "
            "5 attachments, 25 MB total, 2000 chars body.\n\n"
            "Pattern (after pdf_compose returns the path):\n"
            "  signal_send_attachment(\n"
            "      body='Estonia forest report — 2001-2024 Hansen data.',\n"
            "      attachments=['/app/workspace/output/estonia_forest.pdf'],\n"
            "  )\n\n"
            "Returns: success/failure summary + delivered file list."
        )
        args_schema: Type[BaseModel] = _SendAttachmentInput

        def _run(self, body: str, attachments: list[str]) -> str:
            # 1. Validate body
            if not body or not body.strip():
                return "ERROR: body is required (1-3 sentence summary)"
            body = body.strip()
            if len(body) > _MAX_BODY_CHARS:
                body = body[:_MAX_BODY_CHARS] + "…"

            # 2. Validate attachments
            valid_paths, err = _validate_attachments(attachments or [])
            if not valid_paths:
                return f"ERROR — no valid attachments to send.\n{err}"

            # 3. Translate container paths → host paths for signal-cli
            settings = get_settings()
            recipient = settings.signal_owner_number.strip()
            host_paths = _container_to_host(valid_paths, settings.workspace_host_path)

            # 4. Send
            try:
                from app.signal_client import send_message
                send_message(recipient, body, attachments=host_paths)
            except Exception as exc:
                return (
                    f"ERROR — signal_client.send_message raised: "
                    f"{type(exc).__name__}: {exc}\n"
                    f"Files were valid; delivery failed at the "
                    f"signal-cli layer."
                )

            lines = ["Signal message sent."]
            lines.append(f"  recipient: {recipient[:6]}***{recipient[-4:]}")
            lines.append(f"  attachments ({len(valid_paths)}):")
            for p in valid_paths:
                size = p.stat().st_size
                lines.append(f"    - {p.name}  ({size:,} bytes)")
            if err:
                lines.append(f"\nNOTE: some attachments rejected:\n{err}")
            return "\n".join(lines)

    return [SignalSendAttachmentTool()]
