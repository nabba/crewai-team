from crewai.tools import tool
from urllib.parse import urlparse
import ipaddress
import socket
import trafilatura
import requests
from app.audit import log_tool_blocked

# Blocked hostnames and schemes for SSRF protection
_BLOCKED_HOSTS = {
    "localhost", "chromadb", "gateway", "docker-proxy",
    "0.0.0.0", "127.0.0.1", "::", "::1",
    "metadata.google.internal",        # GCP metadata
    "metadata.google.internal.",
    "169.254.169.254",                  # AWS/Azure/GCP metadata endpoint
}
_ALLOWED_SCHEMES = {"http", "https"}


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL is safe to fetch (SSRF protection)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = parsed.hostname or ""
    if hostname.lower() in _BLOCKED_HOSTS:
        return False, f"Blocked host: {hostname}"

    # Resolve hostname and check for private/internal IPs
    try:
        for info in socket.getaddrinfo(hostname, parsed.port or 443):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"Blocked private/internal IP: {addr}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    return True, ""


@tool("web_fetch")
def web_fetch(url: str) -> str:
    """
    Fetch and extract clean text content from a URL.
    Strips ads, navigation, and boilerplate. Returns plain text up to 8000 tokens.
    Only allows public HTTP/HTTPS URLs (no internal or private network access).
    """
    safe, reason = _is_safe_url(url)
    if not safe:
        log_tool_blocked("web_fetch", "unknown", reason)
        return f"URL blocked: {reason}"

    try:
        # Always fetch through requests so we can inspect redirect targets
        # (trafilatura.fetch_url has its own HTTP client that bypasses our SSRF check)
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CrewAI-Bot/1.0)"},
            allow_redirects=True,
        )
        response.raise_for_status()

        # Check final URL after redirects for SSRF
        final_safe, final_reason = _is_safe_url(response.url)
        if not final_safe:
            log_tool_blocked("web_fetch", "unknown", f"redirect: {final_reason}")
            return f"Redirect blocked: {final_reason}"

        # Extract with trafilatura from already-fetched HTML (no second request)
        text = trafilatura.extract(response.text)
        if text:
            return text[:32000]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:32000]
    except Exception:
        return "Fetch error: unable to retrieve URL content."
