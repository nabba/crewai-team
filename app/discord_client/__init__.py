"""
app.discord_client — second messaging surface alongside Signal.

Architecture:
  - The bot runs as an asyncio task on the gateway's event loop, started
    in main.py's lifespan when DISCORD_ENABLED is true.
  - Inbound DMs from the configured owner ID get routed through
    ``handle_task`` exactly like Signal messages, with sender prefixed
    "discord:" so conversation history stays namespaced.
  - Outbound replies are dispatched via ``send_via_discord`` (see
    sender.py) which the main reply path picks based on the sender prefix.

Public surface:

    start_bot(loop)              kick off the bot as a background task
    stop_bot()                   close the connection cleanly on shutdown
    is_running()                 healthcheck for the dashboard
    send_via_discord(...)        outbound DM (re-exported from sender.py)
"""
from __future__ import annotations

from app.discord_client.bot import start_bot, stop_bot, is_running
from app.discord_client.sender import send_via_discord

__all__ = ["start_bot", "stop_bot", "is_running", "send_via_discord"]
