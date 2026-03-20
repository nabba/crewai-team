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

        # If no attachments, chunk long messages and send sequentially
        # (parallel sends via gather don't guarantee delivery order)
        if not attachments:
            chunks = [
                text[i : i + MAX_SIGNAL_LENGTH]
                for i in range(0, len(text), MAX_SIGNAL_LENGTH)
            ]
            for chunk in chunks:
                await asyncio.to_thread(self._send_sync, recipient, chunk)
        else:
            # With attachments, send a single message (text + files)
            await asyncio.to_thread(
                self._send_sync, recipient, text[:MAX_SIGNAL_LENGTH], attachments
            )

    def _send_sync(self, recipient: str, text: str, attachments: list[str] | None = None):
        """Try HTTP first (works from inside Docker), fall back to Unix socket."""
        http_url = getattr(settings, "signal_http_url", "")
        if http_url:
            if self._send_http(http_url, recipient, text, attachments):
                return
            logger.warning("signal-cli HTTP failed, trying Unix socket fallback")

        self._send_socket(recipient, text, attachments)

    def _send_http(self, base_url: str, recipient: str, text: str,
                   attachments: list[str] | None = None) -> bool:
        """Send via signal-cli's HTTP JSON-RPC endpoint."""
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
                return False
            att_info = f" (+{len(attachments)} attachment(s))" if attachments else ""
            logger.info(f"Message sent via signal-cli HTTP{att_info}")
            return True
        except Exception:
            logger.error("signal-cli HTTP request failed", exc_info=True)
            return False

    def _send_socket(self, recipient: str, text: str,
                     attachments: list[str] | None = None):
        """Send via signal-cli Unix socket (works only on same host, not from Docker VM)."""
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
                    return

            if data:
                try:
                    resp = json.loads(data.split(b"\n")[0])
                    if "error" in resp:
                        logger.error("signal-cli RPC error")
                    else:
                        att_info = f" (+{len(attachments)} attachment(s))" if attachments else ""
                        logger.info(f"Message sent via signal-cli socket{att_info}")
                except json.JSONDecodeError:
                    logger.error("signal-cli returned invalid JSON")
            else:
                logger.error("No response from signal-cli socket")

        except Exception:
            logger.error("signal-cli socket communication failed")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
