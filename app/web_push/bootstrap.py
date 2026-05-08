"""
web_push.bootstrap — generate a VAPID key pair, print the env values to add.

Run once on the host:

    python -m app.web_push.bootstrap

Output is two strings to copy into ``.env``:

    VAPID_PUBLIC_KEY=...
    VAPID_PRIVATE_KEY=...

The public key is exposed read-only via ``GET /config/vapid_public_key`` so
the React PWA can subscribe browsers without a rebuild. The private key
stays on the gateway and signs every push delivery.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import base64
        from py_vapid import Vapid01
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        print("py_vapid / cryptography not installed — `pip install pywebpush` first.")
        return 1

    vapid = Vapid01()
    vapid.generate_keys()

    # Public key: uncompressed-point base64url (what browsers expect for
    # PushManager.subscribe applicationServerKey).
    pub_raw = vapid.public_key.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    pub_b64 = base64.urlsafe_b64encode(pub_raw).decode().rstrip("=")

    # Private key: PEM string. pywebpush accepts either a PEM file path or a
    # PEM string in vapid_private_key, so we store the PEM directly.
    priv_pem = vapid.private_pem().decode("ascii")
    # Single-line for .env: replace newlines with literal \n; pywebpush
    # restores them when it parses the key.
    priv_oneline = priv_pem.replace("\n", "\\n")

    print("Add these to .env:\n")
    print(f"VAPID_PUBLIC_KEY={pub_b64}")
    print(f'VAPID_PRIVATE_KEY="{priv_oneline}"')
    print()
    print("Optional — set the contact email Web Push servers can reach you at:")
    print("VAPID_CONTACT_EMAIL=andrus@raudsalu.com")
    print()
    print("After saving, restart the gateway to load the keys, then open")
    print("/cp/settings to enable browser push notifications.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
