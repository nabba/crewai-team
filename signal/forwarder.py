"""
Forwards inbound Signal messages from signal-cli JSON-RPC to the FastAPI gateway.
Run this alongside signal-cli as a separate service.
"""
import socket
import json
import requests
import os
import time

SOCKET_PATH = os.environ.get("SIGNAL_SOCKET_PATH", "/tmp/signal-cli.sock")
GATEWAY_URL = "http://127.0.0.1:8765/signal/inbound"
GATEWAY_SECRET = os.environ.get("GATEWAY_SECRET", "")


def log(msg):
    print(f"[forwarder] {msg}", flush=True)


def listen():
    log(f"Connecting to signal-cli socket at {SOCKET_PATH}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(SOCKET_PATH)
        log("Connected. Waiting for incoming messages...")

        subscribe = json.dumps({
            "jsonrpc": "2.0",
            "id": "sub1",
            "method": "subscribeReceive",
        }) + "\n"
        sock.sendall(subscribe.encode())
        log("Sent subscribeReceive request")

        buffer = b""

        while True:
            data = sock.recv(4096)
            if not data:
                log("Socket closed by signal-cli, reconnecting...")
                break
            buffer += data

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                try:
                    msg = json.loads(line)

                    envelope = msg.get("params", {}).get("envelope", msg.get("params", {}))
                    data_msg = envelope.get("dataMessage", {})

                    if data_msg and (data_msg.get("message") or data_msg.get("attachments")):
                        sender = envelope.get("source") or envelope.get("sourceNumber")
                        message = data_msg.get("message", "")
                        # Extract attachment metadata for the gateway
                        attachments = []
                        for att in data_msg.get("attachments", []):
                            attachments.append({
                                "contentType": att.get("contentType", ""),
                                "filename": att.get("filename", ""),
                                "id": att.get("id", ""),
                                "size": att.get("size", 0),
                            })
                        att_info = f", {len(attachments)} attachment(s)" if attachments else ""
                        log(f"Incoming message ({len(message)} chars{att_info})")

                        payload = {
                            "sender": sender,
                            "message": message,
                            "attachments": attachments,
                        }
                        headers = {}
                        if GATEWAY_SECRET:
                            headers["Authorization"] = f"Bearer {GATEWAY_SECRET}"

                        try:
                            resp = requests.post(
                                GATEWAY_URL,
                                json=payload,
                                headers=headers,
                                timeout=30,
                            )
                            log(f"Forwarded to gateway: {resp.status_code}")
                        except Exception as e:
                            log(f"Failed to forward: {e}")
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    log(f"Error processing message: {e}")
    finally:
        sock.close()


def main():
    if not GATEWAY_SECRET:
        log("WARNING: GATEWAY_SECRET not set — requests will be rejected by gateway")

    while True:
        try:
            listen()
        except ConnectionRefusedError:
            log("Connection refused, retrying in 5s...")
        except FileNotFoundError:
            log(f"Socket not found at {SOCKET_PATH}, retrying in 5s...")
        except Exception as e:
            log(f"Error: {e}, retrying in 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
