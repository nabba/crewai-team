import json
import asyncio
import logging
import socket
import requests as http_requests

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Reusable HTTP session for signal-cli calls (connection pooling)
_http_session = http_requests.Session()
_http_session.headers["Content-Type"] = "application/json"

MAX_SIGNAL_LENGTH = 1500
_MAX_RESPONSE_BYTES = 65536


def _chunk_at_sentences(text: str, max_len: int) -> list[str]:
    """Split text into chunks at sentence/paragraph boundaries (Q12).

    Avoids breaking mid-word or mid-URL. Falls back to hard cut if no
    good boundary is found within the chunk.
    """
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        # Try to find a good cut point within the budget
        window = remaining[:max_len]
        # Prefer paragraph boundary
        cut = window.rfind("\n\n")
        if cut > max_len // 3:
            chunks.append(remaining[:cut].rstrip())
            remaining = remaining[cut:].lstrip("\n")
            continue
        # Then sentence boundary (". " followed by uppercase or newline)
        cut = window.rfind(". ")
        if cut > max_len // 3:
            chunks.append(remaining[:cut + 1])
            remaining = remaining[cut + 2:]
            continue
        # Then any newline
        cut = window.rfind("\n")
        if cut > max_len // 3:
            chunks.append(remaining[:cut])
            remaining = remaining[cut + 1:]
            continue
        # Hard cut at max_len (last resort)
        chunks.append(remaining[:max_len])
        remaining = remaining[max_len:]
    return chunks


class SignalClient:
    async def react(self, recipient: str, emoji: str,
                    target_author: str, target_timestamp: int):
        """Send an emoji reaction to a specific message.

        Args:
            recipient: Phone number of the conversation
            emoji: Emoji character (e.g. "👀")
            target_author: Phone number of the message author being reacted to
            target_timestamp: Timestamp (ms since epoch) of the message to react to
        """
        if recipient.strip() != settings.signal_owner_number.strip():
            logger.error("Blocked reaction to non-owner recipient")
            return
        if not target_timestamp:
            logger.warning("Cannot react: no target timestamp")
            return
        await asyncio.to_thread(
            self._react_sync, recipient, emoji, target_author, target_timestamp
        )

    def _react_sync(self, recipient: str, emoji: str,
                    target_author: str, target_timestamp: int):
        """Send reaction via HTTP first, fall back to Unix socket."""
        http_url = getattr(settings, "signal_http_url", "")
        if http_url:
            if self._react_http(http_url, recipient, emoji, target_author, target_timestamp):
                return
            logger.warning("signal-cli HTTP reaction failed, trying Unix socket fallback")
        self._react_socket(recipient, emoji, target_author, target_timestamp)

    def _react_http(self, base_url: str, recipient: str, emoji: str,
                    target_author: str, target_timestamp: int) -> bool:
        """Send reaction via signal-cli HTTP JSON-RPC."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendReaction",
                "params": {
                    "recipient": [recipient],
                    "emoji": emoji,
                    "target-author": target_author,
                    "target-timestamp": target_timestamp,
                },
            }
            resp = _http_session.post(
                base_url.rstrip("/") + "/api/v1/rpc",
                json=payload,
                timeout=10,
            )
            data = resp.json()
            if "error" in data:
                logger.error(f"signal-cli reaction HTTP error: {data['error'].get('message', '')}")
                return False
            logger.info(f"Reaction {emoji} sent via HTTP")
            return True
        except Exception:
            logger.error("signal-cli reaction HTTP failed", exc_info=True)
            return False

    def _react_socket(self, recipient: str, emoji: str,
                      target_author: str, target_timestamp: int):
        """Send reaction via signal-cli Unix socket."""
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(settings.signal_socket_path)
            sock.settimeout(10)

            request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendReaction",
                "params": {
                    "recipient": [recipient],
                    "emoji": emoji,
                    "target-author": target_author,
                    "target-timestamp": target_timestamp,
                },
            }) + "\n"

            sock.sendall(request.encode())

            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > _MAX_RESPONSE_BYTES:
                    break

            if data:
                try:
                    resp = json.loads(data.split(b"\n")[0])
                    if "error" in resp:
                        logger.error("signal-cli reaction socket error")
                    else:
                        logger.info(f"Reaction {emoji} sent via socket")
                except json.JSONDecodeError:
                    logger.error("signal-cli reaction returned invalid JSON")
        except Exception:
            logger.error("signal-cli reaction socket failed")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    async def send(self, recipient: str, text: str, attachments: list[str] | None = None):
        """Send a message back to the user's iPhone via signal-cli.

        Args:
            recipient: Phone number to send to (must be owner)
            text: Message text
            attachments: Optional list of absolute file paths on the HOST filesystem
                         to attach to the message. signal-cli reads these from the host.
        """
        if recipient.strip() != settings.signal_owner_number.strip():
            logger.error("Blocked attempt to send to non-owner recipient")
            return

        # If no attachments, chunk long messages at sentence boundaries (Q12)
        # (parallel sends via gather don't guarantee delivery order)
        if not attachments:
            chunks = _chunk_at_sentences(text, MAX_SIGNAL_LENGTH)
            for chunk in chunks:
                await asyncio.to_thread(self._send_sync, recipient, chunk)
        else:
            # With attachments, send a single message (text + files)
            await asyncio.to_thread(
                self._send_sync, recipient, text[:MAX_SIGNAL_LENGTH], attachments
            )

    def _send_sync(self, recipient: str, text: str,
                   attachments: list[str] | None = None) -> int | None:
        """Try HTTP first (works from inside Docker), fall back to Unix socket.

        Returns the Signal message timestamp of the sent message so callers
        can correlate later reactions back to this specific message.  Returns
        None if delivery failed.
        """
        http_url = getattr(settings, "signal_http_url", "")
        if http_url:
            ts = self._send_http(http_url, recipient, text, attachments)
            if ts is not None:
                return ts
            logger.warning("signal-cli HTTP failed, trying Unix socket fallback")

        return self._send_socket(recipient, text, attachments)

    def _send_http(self, base_url: str, recipient: str, text: str,
                   attachments: list[str] | None = None) -> int | None:
        """Send via signal-cli's HTTP JSON-RPC endpoint.

        Returns the Signal message timestamp on success, or None on failure.
        """
        try:
            params = {
                "recipient": [recipient],
                "message": text,
            }
            if attachments:
                params["attachments"] = attachments
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "send",
                "params": params,
            }
            resp = _http_session.post(
                base_url.rstrip("/") + "/api/v1/rpc",
                json=payload,
                timeout=15,
            )
            data = resp.json()
            if "error" in data:
                logger.error(f"signal-cli HTTP RPC error: {data['error'].get('message','')}")
                return None
            att_info = f" (+{len(attachments)} attachment(s))" if attachments else ""
            logger.info(f"Message sent via signal-cli HTTP{att_info}")
            # signal-cli returns {"result": {"timestamp": N, ...}} — capture
            # the timestamp so callers can correlate future reactions.
            result = data.get("result") or {}
            ts = result.get("timestamp")
            return int(ts) if isinstance(ts, (int, float)) else 0
        except Exception:
            logger.error("signal-cli HTTP request failed", exc_info=True)
            return None

    def _send_socket(self, recipient: str, text: str,
                     attachments: list[str] | None = None) -> int | None:
        """Send via signal-cli Unix socket (works only on same host, not from Docker VM).

        Returns the Signal message timestamp on success, or None on failure.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(settings.signal_socket_path)
            sock.settimeout(10)

            params = {
                "recipient": [recipient],
                "message": text,
            }
            if attachments:
                params["attachments"] = attachments

            request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "send",
                "params": params,
            }) + "\n"

            sock.sendall(request.encode())

            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(data) > _MAX_RESPONSE_BYTES:
                    logger.error("signal-cli response exceeded buffer limit")
                    return None

            if data:
                try:
                    resp = json.loads(data.split(b"\n")[0])
                    if "error" in resp:
                        logger.error("signal-cli RPC error")
                        return None
                    att_info = f" (+{len(attachments)} attachment(s))" if attachments else ""
                    logger.info(f"Message sent via signal-cli socket{att_info}")
                    result = resp.get("result") or {}
                    ts = result.get("timestamp")
                    return int(ts) if isinstance(ts, (int, float)) else 0
                except json.JSONDecodeError:
                    logger.error("signal-cli returned invalid JSON")
                    return None
            else:
                logger.error("No response from signal-cli socket")
                return None

        except Exception:
            logger.error("signal-cli socket communication failed")
            return None
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass


# ── Module-level convenience function ────────────────────────────────────────
# Used by: healing.health_remediator, healing.error_diagnosis, auditor,
#          auto_deployer, evolution, workspace_versioning, llm_factory —
#          all call send_message() synchronously.

def send_message(recipient: str, text: str, attachments: list | None = None) -> None:
    """Send a Signal message (synchronous wrapper for async SignalClient.send).

    Called by self-healing, escalation, and alerting modules throughout the system.
    Non-fatal — silently logs on failure (alerting must never crash the caller).
    """
    try:
        import asyncio
        client = SignalClient()

        # Try to get a running event loop (if called from async context)
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — schedule as task
            loop.create_task(client.send(recipient, text, attachments))
        except RuntimeError:
            # No running loop — create one (sync context, e.g. background threads)
            asyncio.run(client.send(recipient, text, attachments))
    except Exception as e:
        logger.warning(f"send_message failed (non-fatal): {e}")


def send_message_blocking(
    recipient: str, text: str, attachments: list | None = None,
) -> int | None:
    """Send a Signal message synchronously and return the Signal timestamp.

    Unlike send_message() (fire-and-forget), this blocks until signal-cli
    responds and returns the message's timestamp so callers can correlate
    future reactions back to this message.

    Returns the timestamp (int) on success, or None on failure.  Used by
    proposal notifications so 👍 / 👎 reactions can be mapped back to the
    originating proposal.
    """
    try:
        client = SignalClient()
        # Call _send_sync directly — it already returns the timestamp
        return client._send_sync(recipient, text, attachments)
    except Exception as e:
        logger.warning(f"send_message_blocking failed (non-fatal): {e}")
        return None


async def send_durable(
    recipient: str, text: str, attachments: list | None = None,
    reply_to_id: int | None = None,
) -> int | None:
    """Persist the send to outbound_queue FIRST, then actually send.

    If the container dies between enqueue and the signal-cli call (or
    during the call itself), the row stays 'queued' and the next startup
    replays it.  On success the row is marked 'sent' with the returned
    Signal timestamp; on failure it's marked 'failed' so a poison payload
    can't loop forever (cap is 3 attempts in get_pending_outbound).

    Use this instead of SignalClient().send() for any reply that MUST be
    delivered (user-facing handle_task responses, especially those with
    .md attachments that a mid-rebuild would otherwise lose).

    Returns the Signal timestamp on success, None if the send failed but
    the row is now queued for replay.
    """
    from app.conversation_store import (
        enqueue_outbound, mark_outbound_sent, mark_outbound_failed,
    )
    import asyncio as _asyncio

    qid = enqueue_outbound(recipient, text, attachments, reply_to_id)
    try:
        client = SignalClient()
        # Route through the async send() method so the caller stays off
        # the main thread; _send_sync returns the timestamp.
        if recipient.strip() != settings.signal_owner_number.strip():
            logger.error("Blocked attempt to send to non-owner recipient")
            if qid:
                mark_outbound_failed(qid, "non-owner recipient blocked")
            return None

        if not attachments:
            # Chunked text-only send — grab the last chunk's timestamp.
            chunks = _chunk_at_sentences(text, MAX_SIGNAL_LENGTH)
            last_ts: int | None = None
            for chunk in chunks:
                ts = await _asyncio.to_thread(client._send_sync, recipient, chunk)
                if ts:
                    last_ts = ts
            if last_ts is None and qid:
                mark_outbound_failed(qid, "send returned no timestamp")
                return None
            if qid:
                mark_outbound_sent(qid, last_ts)
            return last_ts
        else:
            ts = await _asyncio.to_thread(
                client._send_sync, recipient,
                text[:MAX_SIGNAL_LENGTH], attachments,
            )
            if ts is None and qid:
                mark_outbound_failed(qid, "send returned no timestamp")
                return None
            if qid:
                mark_outbound_sent(qid, ts)
            return ts
    except Exception as exc:
        if qid:
            mark_outbound_failed(qid, f"{type(exc).__name__}: {exc}")
        logger.warning(f"send_durable failed: {exc}")
        return None


def replay_pending_outbound_sync() -> int:
    """Re-send any queued outbound messages from a previous container
    instance.  Call once at startup, after inbound queue replay.  Returns
    the count of rows re-dispatched (not the count sent successfully —
    some may fail again and stay in 'failed' state)."""
    from app.conversation_store import (
        get_pending_outbound, mark_outbound_sent, mark_outbound_failed,
    )
    pending = get_pending_outbound(max_attempts=3)
    if not pending:
        return 0
    client = SignalClient()
    logger.warning(f"Replaying {len(pending)} unfinished outbound sends")
    for row in pending:
        try:
            attachments = row.get("attachments") or None
            ts = client._send_sync(
                row["recipient"], row["message"] or "",
                attachments if attachments else None,
            )
            if ts:
                mark_outbound_sent(row["id"], ts)
                logger.info(f"outbound replay id={row['id']} sent ts={ts}")
            else:
                mark_outbound_failed(row["id"], "replay got no timestamp")
        except Exception as exc:
            mark_outbound_failed(row["id"], f"replay: {type(exc).__name__}: {exc}")
    return len(pending)
