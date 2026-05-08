"""
google_workspace.bootstrap — one-time consent flow for the Google account.

Run once on the host (or anywhere with a browser) after setting
``GOOGLE_OAUTH_CLIENT_ID`` and ``GOOGLE_OAUTH_CLIENT_SECRET`` in .env:

    python -m app.google_workspace.bootstrap

The script opens a browser tab against Google's consent page, captures
the authorization code on a local loopback server, exchanges it for a
refresh token, and writes the result to ``workspace/google_token.json``.

The expected account is ``andrus@raudsalu.com`` — the script warns
(but doesn't reject) when a different account ends up consenting, so
multi-account testing is still possible.
"""
from __future__ import annotations

import sys

from app.google_workspace.auth import (
    SCOPES, TOKEN_PATH, get_credentials, run_bootstrap_flow,
)


_EXPECTED_EMAIL = "andrus@raudsalu.com"


def main() -> int:
    print("Google Workspace bootstrap")
    print(f"Token will be saved to: {TOKEN_PATH}")
    print("Scopes requested:")
    for scope in SCOPES:
        print(f"  - {scope}")
    print()

    if TOKEN_PATH.exists():
        print(f"Existing token found at {TOKEN_PATH}.")
        choice = input("Re-run consent and overwrite? [y/N]: ").strip().lower()
        if choice not in ("y", "yes"):
            print("Aborted — existing token preserved.")
            return 0

    ok = run_bootstrap_flow()
    if not ok:
        print("Bootstrap failed — see logs and ensure GOOGLE_OAUTH_CLIENT_ID "
              "and GOOGLE_OAUTH_CLIENT_SECRET are set in .env.")
        return 1

    # Smoke-test: pull the authenticated user's email and warn if it doesn't
    # match the expected account.
    creds = get_credentials()
    if creds is None:
        print("Token saved but credentials failed to reload — investigate logs.")
        return 1

    try:
        from googleapiclient.discovery import build
        oauth2 = build("oauth2", "v2", credentials=creds, cache_discovery=False)
        info = oauth2.userinfo().get().execute()
        email = info.get("email", "<unknown>")
        print(f"Authenticated as: {email}")
        if email != _EXPECTED_EMAIL:
            print(
                f"⚠️  Expected {_EXPECTED_EMAIL}; the saved token belongs to a "
                "different account. Re-run if this is wrong."
            )
        else:
            print(f"✓ Account matches expected ({_EXPECTED_EMAIL}).")
    except Exception as exc:  # pragma: no cover — informational only
        print(f"Could not verify account email (non-fatal): {exc}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
