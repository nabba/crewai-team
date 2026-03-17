import json
import asyncio
import logging
import socket
import requests as http_requests

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

MAX_SIGNAL_LENGTH = 1500
_MAX_RESPONSE_BYTES = 65536


class SignalClient:
    async def send(self, recipient: str, text: str):
        """Send a message back to the user's iPhone via signal-cli."""
        if recipient.strip() != settings.signal_owner_number.strip():
            logger.error("Blocked attempt to send to non-owner recipient")
            return

        chunks = [
            text[i : i + MAX_SIGNAL_LENGTH]
            for i in range(0, len(text), MAX_SIGNAL_LENGTH)
        ]
        for chunk in chunks:
            await asyncio.to_thread(self._send_sync, recipient, chunk)

    def _send_sync(self, recipient: str, text: str):
        """Try HTTP first (works from inside Docker), fall back to Unix socket."""
        http_url = getattr(settings, "signal_http_url", "")
        if http_url:
            if self._send_http(http_url, recipient, text):
                return
            logger.warning("signal-cli HTTP failed, trying Unix socket fallback")

        self._send_socket(recipient, text)

    def _send_http(self, base_url: str, recipient: str, text: str) -> bool:
        """Send via signal-cli's HTTP JSON-RPC endpoint."""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "send",
                "params": {
                    "recipient": [recipient],
                    "message": text,
                },
            }
            resp = http_requests.post(
                base_url.rstrip("/") + "/api/v1/rpc",
                json=payload,
                timeout=15,
            )
            data = resp.json()
            if "error" in data:
                logger.error(f"signal-cli HTTP RPC error: {data['error'].get('message','')}")
                return False
            logger.info("Message sent via signal-cli HTTP")
            return True
        except Exception:
            logger.error("signal-cli HTTP request failed", exc_info=True)
            return False

    def _send_socket(self, recipient: str, text: str):
        """Send via signal-cli Unix socket (works only on same host, not from Docker VM)."""
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(settings.signal_socket_path)
            sock.settimeout(10)

            request = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "send",
                "params": {
                    "recipient": [recipient],
                    "message": text,
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
                    logger.error("signal-cli response exceeded buffer limit")
                    return

            if data:
                try:
                    resp = json.loads(data.split(b"\n")[0])
                    if "error" in resp:
                        logger.error("signal-cli RPC error")
                    else:
                        logger.info("Message sent via signal-cli socket")
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
