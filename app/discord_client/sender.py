"""
discord_client.sender — outbound DM delivery.

``send_via_discord(user_id, body, attachment_paths)`` posts a message to
the Discord user identified by ``user_id``. Used by:
  - the gateway reply path when a sender starts with ``discord:``
  - the React /cp/files "Send via Discord" button

Discord's per-message text cap is 2000 characters; longer payloads get
chunked at sentence boundaries (same helper Signal uses). Attachments
are limited to 8 MB each on the free tier — we cap at 5 files per call
to match the Signal surface.

Returns ``(ok: bool, detail: str)`` so REST handlers can surface a
single-line status. Never raises.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_MAX_TEXT = 2000
_MAX_FILES = 5
_MAX_FILE_BYTES = 8 * 1024 * 1024


def send_via_discord(
    user_id: str,
    body: str = "",
    attachment_paths: Iterable[str | Path] = (),
) -> tuple[bool, str]:
    """Post a DM to the configured owner. Sync wrapper around the async path."""
    from app.discord_client.bot import get_client

    client = get_client()
    if client is None:
        return False, "Discord bot not running"
    if not user_id or not user_id.strip():
        return False, "no Discord user_id"

    paths = [Path(p) for p in attachment_paths]
    err = _validate_paths(paths)
    if err:
        return False, err

    try:
        loop = client.loop
    except AttributeError:
        return False, "Discord client has no event loop"

    fut = asyncio.run_coroutine_threadsafe(
        _async_send(client, int(user_id), body, paths),
        loop,
    )
    try:
        return fut.result(timeout=20)
    except Exception as exc:
        return False, f"Discord send failed: {type(exc).__name__}: {exc}"


def _validate_paths(paths: list[Path]) -> str:
    if len(paths) > _MAX_FILES:
        return f"too many attachments ({len(paths)} > {_MAX_FILES})"
    for p in paths:
        if not p.exists() or not p.is_file():
            return f"attachment not found: {p}"
        if p.stat().st_size > _MAX_FILE_BYTES:
            return f"{p.name} exceeds {_MAX_FILE_BYTES // (1024 * 1024)} MB Discord cap"
    return ""


async def _async_send(client, user_id: int, body: str, paths: list[Path]) -> tuple[bool, str]:
    """Resolve the user, open a DM, send text + attachments."""
    try:
        import discord
    except ImportError:
        return False, "discord.py not installed"

    try:
        user = client.get_user(user_id) or await client.fetch_user(user_id)
    except Exception as exc:
        return False, f"could not resolve user {user_id}: {exc}"

    files = [discord.File(str(p), filename=p.name) for p in paths]
    chunks = _chunk(body, _MAX_TEXT) if body else [""]

    try:
        first = True
        for chunk in chunks:
            payload = chunk if chunk else None
            if first and files:
                await user.send(content=payload, files=files)
            elif first:
                await user.send(content=payload or "​")
            else:
                await user.send(content=chunk)
            first = False
    except Exception as exc:
        return False, f"Discord send error: {type(exc).__name__}: {exc}"

    detail = "Discord delivered"
    if files:
        detail += f" ({len(files)} attachment{'s' if len(files) != 1 else ''})"
    return True, detail


def _chunk(text: str, limit: int) -> list[str]:
    """Sentence-aware chunking matching the Signal helper's behaviour."""
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            out.append(remaining)
            break
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut > limit // 3:
            out.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip("\n")
            continue
        cut = window.rfind(". ")
        if cut > limit // 3:
            out.append(remaining[:cut + 1])
            remaining = remaining[cut + 2:]
            continue
        cut = window.rfind("\n")
        if cut > limit // 3:
            out.append(remaining[:cut])
            remaining = remaining[cut + 1:]
            continue
        out.append(remaining[:limit])
        remaining = remaining[limit:]
    return out
