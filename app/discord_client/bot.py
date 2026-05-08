"""
discord_client.bot — discord.py client running as a gateway background task.

Inbound flow:
  1. ``on_message`` fires for every message the bot can see.
  2. We accept ONLY direct messages (no guild context) AND only from
     the configured ``DISCORD_OWNER_ID``. Everything else is silently
     ignored — this is a personal assistant, not a public bot.
  3. The owner's message is forwarded to ``handle_task`` (the same
     entry point Signal uses) with sender = ``discord:<user_id>``.
  4. The reply path inspects the sender prefix and routes outbound to
     ``send_via_discord`` instead of ``signal_client.send``.

The bot is opt-in via DISCORD_ENABLED + a populated DISCORD_BOT_TOKEN.
When either is missing, ``start_bot`` is a no-op and ``is_running``
reports False; the rest of the system stays uninvolved.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.config import get_discord_bot_token, get_settings

logger = logging.getLogger(__name__)

# Module-level handles so the lifespan handler can shut down cleanly.
_client = None  # type: ignore[assignment]
_task: Optional[asyncio.Task] = None
_running = False


def is_running() -> bool:
    return _running


def get_client():
    """Return the live discord.Client instance for use by sender.py.

    Returns None when the bot isn't running.
    """
    return _client if _running else None


async def start_bot(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Kick off the Discord bot as a background asyncio task.

    Idempotent — calling twice in the same process leaves the existing
    task in place. Silently no-ops when the bot isn't configured.
    """
    global _client, _task, _running
    if _task is not None and not _task.done():
        return

    s = get_settings()
    if not s.discord_enabled:
        logger.debug("discord_client: DISCORD_ENABLED is false")
        return
    token = get_discord_bot_token()
    if not token:
        logger.warning("discord_client: DISCORD_BOT_TOKEN not set; bot disabled")
        return
    if not (s.discord_owner_id or "").strip():
        logger.warning("discord_client: DISCORD_OWNER_ID not set; bot disabled")
        return

    try:
        import discord
    except ImportError:
        logger.warning("discord_client: discord.py not installed; bot disabled")
        return

    intents = discord.Intents.default()
    # Required to see DM contents (privileged in v2 only when bot has
    # MESSAGE CONTENT intent toggled on the developer portal).
    intents.message_content = True
    intents.dm_messages = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:  # type: ignore[unused-ignore]
        global _running
        _running = True
        logger.info(f"discord_client: connected as {client.user}")

    @client.event
    async def on_disconnect() -> None:  # type: ignore[unused-ignore]
        global _running
        _running = False
        logger.info("discord_client: disconnected")

    @client.event
    async def on_message(message) -> None:  # type: ignore[unused-ignore]
        try:
            await _route_inbound(message, client)
        except Exception:
            logger.exception("discord_client: on_message handler raised")

    _client = client
    if loop is None:
        loop = asyncio.get_running_loop()
    _task = loop.create_task(_run_with_logging(client, token), name="discord-bot")
    logger.info("discord_client: bot task scheduled")


async def stop_bot() -> None:
    """Close the bot connection cleanly. Safe to call when not running."""
    global _client, _task, _running
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            logger.debug("discord_client.stop_bot: close raised", exc_info=True)
    _running = False
    if _task is not None and not _task.done():
        try:
            await asyncio.wait_for(_task, timeout=5)
        except (asyncio.TimeoutError, Exception):
            _task.cancel()
    _client = None
    _task = None


# ── Internals ─────────────────────────────────────────────────────────────

async def _run_with_logging(client, token: str) -> None:
    """Wrap client.start so an unhandled error is logged, not silent."""
    try:
        await client.start(token)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("discord_client: bot task crashed")


async def _route_inbound(message, client) -> None:
    """Validate the sender + DM context, forward to the gateway's handle_task."""
    # Ignore the bot's own messages.
    if message.author == client.user:
        return
    # Only DMs (no guild) — refuse channel messages outright.
    if getattr(message, "guild", None) is not None:
        return
    # Owner check — case the IDs as ints to defend against int/str config drift.
    s = get_settings()
    try:
        owner_id = int(s.discord_owner_id)
    except (TypeError, ValueError):
        return
    if int(message.author.id) != owner_id:
        return

    text = (message.content or "").strip()
    if not text:
        return

    # Re-use the existing rate limiter on the synthetic Discord sender.
    from app.security import is_within_rate_limit, _redact_number
    sender = f"discord:{message.author.id}"
    if not is_within_rate_limit(sender):
        try:
            await message.channel.send(
                "Rate limit hit — try again in a minute.",
            )
        except Exception:
            pass
        return

    logger.info(
        f"discord inbound from {_redact_number(sender)}: {len(text)} chars"
    )

    # Dispatch through the same handler Signal uses, on a thread so the
    # bot's event loop isn't blocked by a long crew run.
    from main import handle_task  # late import — main has its own deps
    asyncio.create_task(handle_task(sender, text))
