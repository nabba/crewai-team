"""
auth_patterns.py — Reusable authentication strategy templates.

Not hardcoded connectors — reusable auth *strategies* that the API Scout
instantiates with API-specific parameters. The system identifies which
pattern an API uses from documentation and fills in the template.

Patterns:
  - api_key_header: Authorization: Bearer {key} or custom header
  - api_key_query: ?api_key={key}
  - oauth2_client_credentials: client_id + secret → token endpoint → bearer
  - oauth2_device_code: device flow for headless environments
  - session_cookie: login endpoint → session cookie → attach
  - webhook_signature: HMAC verification for incoming webhooks
  - basic_auth: username:password base64-encoded

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ── Auth pattern definitions ─────────────────────────────────────────────────


@dataclass
class AuthPattern:
    """A reusable authentication strategy template."""
    pattern_id: str
    name: str
    description: str
    required_params: list[str]  # what the user must provide
    optional_params: list[str] = field(default_factory=list)
    code_template: str = ""     # Python code template
    detection_signals: list[str] = field(default_factory=list)  # signals in API docs


# ── Built-in patterns ────────────────────────────────────────────────────────


PATTERNS: dict[str, AuthPattern] = {

    "api_key_header": AuthPattern(
        pattern_id="api_key_header",
        name="API Key (Header)",
        description="API key sent in a request header (Authorization: Bearer or custom header)",
        required_params=["api_key"],
        optional_params=["header_name", "prefix"],
        detection_signals=[
            "api key", "api_key", "apikey", "Authorization: Bearer",
            "X-API-Key", "x-api-key", "api-key header",
        ],
        code_template='''
import httpx

class AuthApiKeyHeader:
    """API Key authentication via request header."""

    def __init__(self, api_key: str, header_name: str = "Authorization", prefix: str = "Bearer"):
        self._api_key = api_key
        self._header_name = header_name
        self._prefix = prefix

    def get_headers(self) -> dict[str, str]:
        if self._prefix:
            return {self._header_name: f"{self._prefix} {self._api_key}"}
        return {self._header_name: self._api_key}

    def apply(self, client: httpx.Client) -> httpx.Client:
        client.headers.update(self.get_headers())
        return client
''',
    ),

    "api_key_query": AuthPattern(
        pattern_id="api_key_query",
        name="API Key (Query Parameter)",
        description="API key sent as a URL query parameter",
        required_params=["api_key"],
        optional_params=["param_name"],
        detection_signals=[
            "api_key=", "apikey=", "key=", "query parameter", "?key=",
        ],
        code_template='''
import httpx

class AuthApiKeyQuery:
    """API Key authentication via query parameter."""

    def __init__(self, api_key: str, param_name: str = "api_key"):
        self._api_key = api_key
        self._param_name = param_name

    def get_params(self) -> dict[str, str]:
        return {self._param_name: self._api_key}

    def apply_to_url(self, url: str) -> str:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{self._param_name}={self._api_key}"
''',
    ),

    "oauth2_client_credentials": AuthPattern(
        pattern_id="oauth2_client_credentials",
        name="OAuth2 Client Credentials",
        description="OAuth2 client credentials flow: exchange client_id + secret for access token",
        required_params=["client_id", "client_secret", "token_url"],
        optional_params=["scopes", "audience"],
        detection_signals=[
            "oauth2", "oauth 2", "client_credentials", "client credentials",
            "client_id", "client_secret", "token endpoint", "/oauth/token",
            "grant_type=client_credentials",
        ],
        code_template='''
import httpx
import time
import threading

class AuthOAuth2ClientCredentials:
    """OAuth2 Client Credentials flow with automatic token refresh."""

    def __init__(self, client_id: str, client_secret: str, token_url: str,
                 scopes: list[str] | None = None, audience: str = ""):
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._scopes = scopes or []
        self._audience = audience
        self._access_token: str = ""
        self._expires_at: float = 0
        self._lock = threading.Lock()

    def get_token(self) -> str:
        with self._lock:
            if self._access_token and time.time() < self._expires_at - 60:
                return self._access_token
            return self._refresh_token()

    def _refresh_token(self) -> str:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scopes:
            data["scope"] = " ".join(self._scopes)
        if self._audience:
            data["audience"] = self._audience

        resp = httpx.post(self._token_url, data=data, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        self._access_token = body["access_token"]
        self._expires_at = time.time() + body.get("expires_in", 3600)
        return self._access_token

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}
''',
    ),

    "oauth2_device_code": AuthPattern(
        pattern_id="oauth2_device_code",
        name="OAuth2 Device Code",
        description="OAuth2 device authorization flow for headless environments (requires human consent)",
        required_params=["client_id", "device_auth_url", "token_url"],
        optional_params=["scopes"],
        detection_signals=[
            "device code", "device flow", "device authorization",
            "urn:ietf:params:oauth:grant-type:device_code",
        ],
        code_template='''
import httpx
import time

class AuthOAuth2DeviceCode:
    """OAuth2 Device Code flow — requires human-in-the-loop for authorization."""

    def __init__(self, client_id: str, device_auth_url: str, token_url: str,
                 scopes: list[str] | None = None):
        self._client_id = client_id
        self._device_auth_url = device_auth_url
        self._token_url = token_url
        self._scopes = scopes or []

    def start_authorization(self) -> dict:
        """Start device auth. Returns {user_code, verification_uri, device_code}."""
        data = {"client_id": self._client_id}
        if self._scopes:
            data["scope"] = " ".join(self._scopes)

        resp = httpx.post(self._device_auth_url, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def poll_for_token(self, device_code: str, interval: int = 5,
                       max_attempts: int = 60) -> str | None:
        """Poll token endpoint until user authorizes or timeout."""
        for _ in range(max_attempts):
            time.sleep(interval)
            resp = httpx.post(self._token_url, data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": self._client_id,
            }, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("access_token")
            body = resp.json()
            if body.get("error") == "authorization_pending":
                continue
            elif body.get("error") == "slow_down":
                interval += 2
            else:
                break
        return None
''',
    ),

    "basic_auth": AuthPattern(
        pattern_id="basic_auth",
        name="HTTP Basic Authentication",
        description="Username and password sent as base64-encoded Authorization header",
        required_params=["username", "password"],
        detection_signals=[
            "basic auth", "basic authentication", "Authorization: Basic",
            "username and password", "HTTP Basic",
        ],
        code_template='''
import base64
import httpx

class AuthBasic:
    """HTTP Basic Authentication."""

    def __init__(self, username: str, password: str):
        self._credentials = base64.b64encode(
            f"{username}:{password}".encode()
        ).decode()

    def get_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self._credentials}"}
''',
    ),

    "webhook_signature": AuthPattern(
        pattern_id="webhook_signature",
        name="Webhook Signature Verification",
        description="HMAC signature verification for incoming webhooks",
        required_params=["secret"],
        optional_params=["header_name", "hash_algorithm"],
        detection_signals=[
            "webhook signature", "hmac", "X-Hub-Signature", "X-Signature",
            "webhook secret", "signature verification", "signing secret",
        ],
        code_template='''
import hashlib
import hmac

class WebhookSignatureVerifier:
    """HMAC signature verification for incoming webhooks."""

    def __init__(self, secret: str, header_name: str = "X-Hub-Signature-256",
                 hash_algorithm: str = "sha256"):
        self._secret = secret.encode()
        self._header_name = header_name
        self._algorithm = hash_algorithm

    def verify(self, payload: bytes, signature: str) -> bool:
        expected = hmac.new(
            self._secret, payload, getattr(hashlib, self._algorithm)
        ).hexdigest()
        # Handle "sha256=..." prefix
        if "=" in signature:
            signature = signature.split("=", 1)[1]
        return hmac.compare_digest(expected, signature)
''',
    ),
}


# ── Detection ─────────────────────────────────────────────────────────────────


def detect_auth_pattern(api_docs_text: str) -> list[tuple[str, float]]:
    """Detect which auth patterns an API likely uses from its documentation text.

    Returns list of (pattern_id, confidence) sorted by confidence descending.
    """
    text_lower = api_docs_text.lower()
    results = []

    for pattern_id, pattern in PATTERNS.items():
        matches = 0
        for signal in pattern.detection_signals:
            if signal.lower() in text_lower:
                matches += 1
        if matches > 0:
            confidence = min(1.0, matches / max(2, len(pattern.detection_signals) * 0.5))
            results.append((pattern_id, confidence))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_pattern(pattern_id: str) -> Optional[AuthPattern]:
    """Get an auth pattern by ID."""
    return PATTERNS.get(pattern_id)


def get_pattern_code(pattern_id: str) -> str:
    """Get the code template for an auth pattern."""
    pattern = PATTERNS.get(pattern_id)
    return pattern.code_template.strip() if pattern else ""


def list_patterns() -> list[str]:
    """List all available pattern IDs."""
    return list(PATTERNS.keys())
