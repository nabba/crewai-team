from crewai.tools import tool
from urllib.parse import urlparse
import ipaddress
import socket
import trafilatura
import requests
from requests.adapters import HTTPAdapter
from app.audit import log_tool_blocked

# Reusable HTTP session with connection pooling
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (compatible; CrewAI-Bot/1.0)"
_session.mount("https://", HTTPAdapter(pool_connections=5, pool_maxsize=5))
_session.mount("http://", HTTPAdapter(pool_connections=2, pool_maxsize=2))

# Blocked hostnames and schemes for SSRF protection
_BLOCKED_HOSTS = {
    "localhost", "chromadb", "gateway", "docker-proxy",
    "0.0.0.0", "127.0.0.1", "::", "::1",
    "metadata.google.internal",        # GCP metadata
    "metadata.google.internal.",
    "169.254.169.254",                  # AWS/Azure/GCP metadata endpoint
    "host.docker.internal",             # Docker Desktop host gateway
    "kubernetes.default",               # Kubernetes API
    "kubernetes.default.svc",
}
# Only HTTPS for external fetches — HTTP content can be intercepted/modified
_ALLOWED_SCHEMES = {"https"}

# Max response size (10 MB) — prevent OOM from huge downloads
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024


def _is_private_ip(addr: str) -> bool:
    """Check if an IP address is private, loopback, link-local, or reserved."""
    try:
        ip = ipaddress.ip_address(addr)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_reserved or ip.is_multicast
        )
    except ValueError:
        return True  # If we can't parse it, block it


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Validate URL is safe to fetch (SSRF protection)."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"Blocked scheme: {parsed.scheme}"

    hostname = parsed.hostname or ""
    if not hostname:
        return False, "Missing hostname"

    # M2: Strip IPv6 brackets for comparison (urlparse keeps brackets on hostname)
    hostname_clean = hostname.strip("[]").lower()

    # Block dotless hostnames (e.g., "chromadb" or single-word hosts)
    if "." not in hostname_clean and ":" not in hostname_clean:
        return False, f"Blocked internal hostname: {hostname}"

    if hostname_clean in _BLOCKED_HOSTS:
        return False, f"Blocked host: {hostname}"

    # Block numeric IPs that look like internal addresses (e.g., http://2130706433)
    # urlparse may parse "http://2130706433" with hostname="2130706433"
    try:
        ip = ipaddress.ip_address(hostname)
        if _is_private_ip(str(ip)):
            return False, f"Blocked private IP: {hostname}"
    except ValueError:
        pass  # Not a raw IP, continue with DNS resolution

    # M3: Resolve hostname with timeout and check for private/internal IPs.
    # E8: Use thread-scoped timeout instead of global socket.setdefaulttimeout()
    # which is thread-unsafe and affects all concurrent network operations.
    import concurrent.futures
    def _resolve():
        resolved = []
        for info in socket.getaddrinfo(hostname, parsed.port or 443):
            addr = info[4][0]
            if _is_private_ip(addr):
                raise ValueError(f"Blocked private/internal IP: {addr}")
            resolved.append(addr)
        return resolved

    try:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            resolved_addrs = pool.submit(_resolve).result(timeout=5)
        if not resolved_addrs:
            return False, f"No addresses resolved for: {hostname}"
    except ValueError as e:
        return False, str(e)
    except concurrent.futures.TimeoutError:
        return False, f"DNS resolution timed out for: {hostname}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    return True, ""


def _check_response_ip(response) -> tuple[bool, str]:
    """Check the actual IP the connection was made to (DNS rebinding defense)."""
    try:
        sock = response.raw._connection.sock
        if sock is None:
            return True, ""  # Can't inspect, allow (pre-check already ran)
        peername = sock.getpeername()
        if peername and _is_private_ip(peername[0]):
            return False, f"DNS rebinding detected: connected to private IP {peername[0]}"
    except (AttributeError, OSError):
        pass  # Socket already closed or unavailable
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
        response = _session.get(
            url,
            timeout=15,
            allow_redirects=True,
            stream=True,  # Stream so we can check size before loading
        )
        response.raise_for_status()

        # Check content length to prevent OOM
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > _MAX_RESPONSE_BYTES:
            response.close()
            log_tool_blocked("web_fetch", "unknown", f"Response too large: {content_length} bytes")
            return "URL blocked: response too large."

        # Read in chunks with hard size limit — prevents OOM even when
        # Content-Length is absent or lying (stream=True avoids buffering)
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=65536):
            downloaded += len(chunk)
            if downloaded > _MAX_RESPONSE_BYTES:
                break
            chunks.append(chunk)
        response.close()
        content = b"".join(chunks)
        html_text = content.decode("utf-8", errors="replace")

        # Check final URL after redirects for SSRF
        final_safe, final_reason = _is_safe_url(response.url)
        if not final_safe:
            log_tool_blocked("web_fetch", "unknown", f"redirect: {final_reason}")
            return f"Redirect blocked: {final_reason}"

        # DNS rebinding defense: check actual connection IP
        ip_safe, ip_reason = _check_response_ip(response)
        if not ip_safe:
            log_tool_blocked("web_fetch", "unknown", ip_reason)
            return f"URL blocked: {ip_reason}"

        # Extract with trafilatura from already-fetched HTML (no second request)
        # Q13: Limit to 12000 chars (~3K tokens) — keeps room in the model's
        # context window for system prompt, tools, task, and other search results.
        # Original 32K could fill a 32K-context local model entirely.
        _MAX_TEXT_CHARS = 12000
        text = trafilatura.extract(html_text)
        if text:
            return text[:_MAX_TEXT_CHARS]

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:_MAX_TEXT_CHARS]
    except Exception:
        return "Fetch error: unable to retrieve URL content."
