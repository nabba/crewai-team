"""
api_scout.py — Autonomous API discovery, analysis, and client generation.

Pipeline: Discovery → Analysis → Connector Build → Test → Register

When the system encounters an unfamiliar API ("connect to Airtable"), API Scout:
  1. Searches for API docs, OpenAPI specs, SDKs, tutorials
  2. Parses docs into structured API knowledge (endpoints, auth, schemas)
  3. Generates a typed Python client with auth handling and retry logic
  4. Tests the client in sandbox
  5. Registers as a reusable skill in the library

Spec-first with fallback:
  1. OpenAPI/Swagger spec (highest confidence)
  2. Official SDK / typed client library
  3. Official documentation pages
  4. Community tutorials / blog posts
  5. Trial and error (lowest confidence)

IMMUTABLE — infrastructure-level module.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── API Knowledge Model ──────────────────────────────────────────────────────


@dataclass
class EndpointSpec:
    """Specification for a single API endpoint."""
    path: str = ""
    method: str = "GET"
    description: str = ""
    parameters: list[dict] = field(default_factory=list)  # [{name, type, required, in}]
    request_body: dict = field(default_factory=dict)       # {format, fields, example}
    response_schema: dict = field(default_factory=dict)    # {format, fields, example}
    rate_limit: str = ""
    requires_auth: bool = True


@dataclass
class APIKnowledge:
    """Structured knowledge about an API."""
    name: str = ""
    base_url: str = ""
    version: str = ""
    description: str = ""
    auth_type: str = ""          # pattern_id from auth_patterns.py
    auth_details: dict = field(default_factory=dict)   # token_url, scopes, etc.
    endpoints: list[EndpointSpec] = field(default_factory=list)
    rate_limits: dict = field(default_factory=dict)    # global rate limit info
    error_codes: dict = field(default_factory=dict)    # {code: description}
    doc_sources: list[dict] = field(default_factory=list)  # [{url, type, confidence}]
    confidence: float = 0.0
    discovered_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["endpoints"] = [asdict(e) for e in self.endpoints]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "APIKnowledge":
        endpoints = [EndpointSpec(**e) for e in d.pop("endpoints", [])]
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        obj = cls(**valid)
        obj.endpoints = endpoints
        return obj


# ── API Scout Pipeline ───────────────────────────────────────────────────────


# Discovery search priority order
SEARCH_PRIORITY = [
    "{api_name} openapi spec yaml json",
    "{api_name} api documentation developer",
    "{api_name} python sdk client library pypi",
    "{api_name} api tutorial example",
    "{api_name} rest api authentication",
]

# LLM prompt for structured API extraction (IMMUTABLE)
API_EXTRACTION_PROMPT = """Analyze this API documentation and extract structured information.

API Name: {api_name}
Documentation Content:
{doc_content}

Extract the following as a JSON object:
{{
  "name": "API name",
  "base_url": "https://api.example.com/v1",
  "version": "v1 or latest version",
  "description": "Brief description",
  "auth_type": "one of: api_key_header, api_key_query, oauth2_client_credentials, oauth2_device_code, basic_auth, webhook_signature, none",
  "auth_details": {{"token_url": "...", "scopes": [...], "header_name": "..."}},
  "endpoints": [
    {{
      "path": "/resource",
      "method": "GET",
      "description": "What it does",
      "parameters": [{{"name": "id", "type": "string", "required": true, "in": "path"}}],
      "request_body": {{"format": "json", "fields": {{}}, "example": {{}}}},
      "response_schema": {{"format": "json", "fields": {{}}, "example": {{}}}},
      "rate_limit": "100 req/min"
    }}
  ],
  "rate_limits": {{"requests_per_minute": 100, "requests_per_second": 3}},
  "error_codes": {{"401": "Unauthorized", "429": "Rate limited"}}
}}

Be thorough — include ALL endpoints you can find. For auth_type, choose the most
appropriate pattern. Return ONLY valid JSON."""

# LLM prompt for client code generation (IMMUTABLE)
CLIENT_GENERATION_PROMPT = """Generate a production-quality Python client for this API.

API Knowledge:
{api_knowledge}

Auth Pattern Code:
{auth_pattern_code}

Requirements:
1. Use httpx for HTTP calls (async-friendly)
2. Include proper error handling with custom exceptions
3. Include retry logic with exponential backoff for 429/5xx
4. Include rate limiting (respect the API's limits)
5. Type hints on all methods
6. Docstrings on all public methods
7. Return parsed JSON (dict) from all methods, not raw responses
8. Include a constructor that takes credentials as parameters

Generate ONLY the Python code. No markdown fences, no explanation.
The code should be a single file that can be imported and used directly."""


class APIScout:
    """Autonomous API discovery, analysis, and client generation pipeline."""

    def __init__(self):
        self._knowledge_cache: dict[str, APIKnowledge] = {}
        self._knowledge_dir = Path("/app/workspace/atlas/api_knowledge")
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._load_cache()

    def _load_cache(self) -> None:
        """Load previously discovered API knowledge from disk."""
        for path in self._knowledge_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                knowledge = APIKnowledge.from_dict(data)
                if knowledge.name:
                    self._knowledge_cache[knowledge.name.lower()] = knowledge
            except Exception:
                pass

    def discover_api(self, api_name: str, hints: str = "") -> APIKnowledge:
        """Full pipeline: discover → analyze → generate knowledge for an API.

        Args:
            api_name: Name of the API (e.g., "Notion", "Stripe", "Airtable")
            hints: Optional hints about the API (docs URL, etc.)

        Returns:
            Structured API knowledge
        """
        logger.info(f"api_scout: starting discovery for '{api_name}'")

        # Check cache first
        cached = self._knowledge_cache.get(api_name.lower())
        if cached and cached.confidence >= 0.7:
            logger.info(f"api_scout: using cached knowledge for '{api_name}' "
                        f"(confidence={cached.confidence:.2f})")
            return cached

        # Step 1: Search for documentation
        doc_sources = self._search_for_docs(api_name, hints)

        # Step 2: Fetch and parse documentation
        doc_content = self._fetch_docs(doc_sources)

        # Step 3: Extract structured API knowledge via LLM
        knowledge = self._extract_knowledge(api_name, doc_content, doc_sources)

        # Step 4: Detect auth pattern
        if not knowledge.auth_type:
            from app.atlas.auth_patterns import detect_auth_pattern
            full_doc = " ".join(d.get("content", "") for d in doc_content)
            patterns = detect_auth_pattern(full_doc)
            if patterns:
                knowledge.auth_type = patterns[0][0]

        # Cache and persist
        self._knowledge_cache[api_name.lower()] = knowledge
        self._persist_knowledge(knowledge)

        logger.info(f"api_scout: discovered '{api_name}' — "
                    f"{len(knowledge.endpoints)} endpoints, auth={knowledge.auth_type}, "
                    f"confidence={knowledge.confidence:.2f}")
        return knowledge

    def generate_client(self, knowledge: APIKnowledge) -> str:
        """Generate a Python client from API knowledge.

        Returns: Python source code for the client.
        """
        logger.info(f"api_scout: generating client for '{knowledge.name}'")

        # Get auth pattern code
        auth_code = ""
        if knowledge.auth_type:
            from app.atlas.auth_patterns import get_pattern_code
            auth_code = get_pattern_code(knowledge.auth_type)

        # Generate via LLM
        prompt = CLIENT_GENERATION_PROMPT.format(
            api_knowledge=json.dumps(knowledge.to_dict(), indent=2)[:4000],
            auth_pattern_code=auth_code,
        )

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=4096, role="coding")
            raw = str(llm.call(prompt)).strip()

            # Clean up — remove markdown fences if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            return raw
        except Exception as e:
            logger.error(f"api_scout: client generation failed: {e}")
            return ""

    def generate_tests(self, knowledge: APIKnowledge, client_code: str) -> str:
        """Generate test code for an API client.

        Returns: Python test code (pytest format).
        """
        prompt = f"""Generate pytest tests for this API client code.

Client Code:
{client_code[:3000]}

API Endpoints:
{json.dumps([asdict(e) for e in knowledge.endpoints[:10]], indent=2)[:2000]}

Requirements:
1. Use httpx mock (respx or unittest.mock) — do NOT make real API calls
2. Test each public method at least once
3. Test error handling (401, 429, 500 responses)
4. Test rate limiting behavior
5. Test auth token refresh (if applicable)

Generate ONLY Python test code. No markdown fences."""

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=2048, role="coding")
            raw = str(llm.call(prompt)).strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return raw
        except Exception:
            return ""

    def build_and_register(self, api_name: str, hints: str = "") -> dict:
        """Full end-to-end: discover API → generate client → register as skill.

        Returns: {success, skill_id, confidence, endpoints_count, error}
        """
        try:
            # Discover
            knowledge = self.discover_api(api_name, hints)
            if not knowledge.endpoints:
                return {"success": False, "error": "No endpoints discovered"}

            # Generate client
            client_code = self.generate_client(knowledge)
            if not client_code:
                return {"success": False, "error": "Client generation failed"}

            # Generate tests
            test_code = self.generate_tests(knowledge, client_code)

            # Register as skill
            from app.atlas.skill_library import get_library
            library = get_library()

            skill_id = f"apis/{api_name.lower().replace(' ', '_')}/client"
            manifest = library.register_skill(
                skill_id=skill_id,
                name=f"{knowledge.name} API Client",
                category="apis",
                code=client_code,
                description=knowledge.description,
                source_type=self._primary_source_type(knowledge.doc_sources),
                source_urls=[s.get("url", "") for s in knowledge.doc_sources],
                auth_pattern=knowledge.auth_type,
                test_code=test_code,
                tags=[api_name.lower(), "api", "client", knowledge.auth_type],
            )

            result = {
                "success": True,
                "skill_id": skill_id,
                "confidence": manifest.effective_confidence(),
                "endpoints_count": len(knowledge.endpoints),
                "auth_type": knowledge.auth_type,
            }
            # Audit trail
            try:
                from app.atlas.audit_log import log_external_call
                log_external_call(
                    agent="api_scout", action="build_and_register",
                    target=api_name, method="discover+generate+register",
                    result="success",
                )
            except Exception:
                pass
            return result

        except Exception as e:
            logger.error(f"api_scout: build_and_register failed for '{api_name}': {e}")
            try:
                from app.atlas.audit_log import log_external_call
                log_external_call(
                    agent="api_scout", action="build_and_register",
                    target=api_name, result="failure",
                )
            except Exception:
                pass
            return {"success": False, "error": str(e)[:500]}

    # ── Private helpers ───────────────────────────────────────────────────

    def _search_for_docs(self, api_name: str, hints: str) -> list[dict]:
        """Search the web for API documentation."""
        sources = []

        # If hints include a URL, use it directly
        if hints and ("http" in hints or "www" in hints):
            sources.append({"url": hints.strip(), "type": "hint", "confidence": 0.9})

        # Search using Brave Search (available in the system)
        try:
            from app.tools.web_search import search_brave
            for template in SEARCH_PRIORITY[:3]:  # Top 3 search queries
                query = template.format(api_name=api_name)
                results = search_brave(query, count=3)
                for r in results:
                    url = r.get("url", "")
                    title = r.get("title", "").lower()

                    # Classify source type
                    source_type = "community"
                    confidence = 0.50
                    if "openapi" in title or "swagger" in title:
                        source_type = "openapi_spec"
                        confidence = 0.95
                    elif "developer" in url or "docs" in url or "api" in url:
                        source_type = "official_docs"
                        confidence = 0.85
                    elif "pypi" in url or "github.com" in url:
                        source_type = "official_sdk"
                        confidence = 0.90
                    elif "youtube" in url:
                        source_type = "youtube_tutorial"
                        confidence = 0.60

                    sources.append({
                        "url": url,
                        "title": r.get("title", ""),
                        "type": source_type,
                        "confidence": confidence,
                    })
        except Exception:
            logger.debug("api_scout: web search unavailable", exc_info=True)

        # Deduplicate by URL
        seen = set()
        unique = []
        for s in sources:
            if s["url"] not in seen:
                seen.add(s["url"])
                unique.append(s)

        # Sort by confidence
        unique.sort(key=lambda s: s["confidence"], reverse=True)
        return unique[:10]  # Top 10 sources

    def _fetch_docs(self, sources: list[dict]) -> list[dict]:
        """Fetch content from documentation sources."""
        fetched = []
        for source in sources[:5]:  # Fetch top 5
            url = source.get("url", "")
            if not url:
                continue
            try:
                import httpx
                resp = httpx.get(url, timeout=15, follow_redirects=True)
                if resp.status_code == 200:
                    content = resp.text[:10000]  # Limit size
                    # Try to parse as OpenAPI spec
                    if source["type"] == "openapi_spec":
                        try:
                            spec = json.loads(content)
                            source["parsed_spec"] = spec
                        except json.JSONDecodeError:
                            try:
                                import yaml
                                spec = yaml.safe_load(content)
                                source["parsed_spec"] = spec
                            except Exception:
                                pass

                    fetched.append({
                        "url": url,
                        "type": source["type"],
                        "content": content,
                        "confidence": source["confidence"],
                    })
            except Exception:
                logger.debug(f"api_scout: failed to fetch {url}", exc_info=True)

        return fetched

    def _extract_knowledge(
        self, api_name: str, doc_content: list[dict], doc_sources: list[dict]
    ) -> APIKnowledge:
        """Use LLM to extract structured API knowledge from documentation."""
        # Check if we have a parsed OpenAPI spec — direct extraction
        for doc in doc_content:
            if doc.get("type") == "openapi_spec" and "parsed_spec" in doc_sources:
                knowledge = self._parse_openapi_spec(api_name, doc_sources[0].get("parsed_spec", {}))
                if knowledge and knowledge.endpoints:
                    knowledge.doc_sources = doc_sources
                    knowledge.confidence = 0.95
                    return knowledge

        # Fall back to LLM extraction
        combined_content = "\n\n---\n\n".join(
            d.get("content", "")[:3000] for d in doc_content[:3]
        )

        if not combined_content:
            return APIKnowledge(
                name=api_name,
                discovered_at=datetime.now(timezone.utc).isoformat(),
                doc_sources=doc_sources,
                confidence=0.2,
            )

        prompt = API_EXTRACTION_PROMPT.format(
            api_name=api_name,
            doc_content=combined_content[:6000],
        )

        try:
            from app.llm_factory import create_specialist_llm
            llm = create_specialist_llm(max_tokens=3000, role="research")
            raw = str(llm.call(prompt)).strip()

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]+\}', raw)
            if json_match:
                data = json.loads(json_match.group())
                knowledge = APIKnowledge.from_dict(data)
                knowledge.discovered_at = datetime.now(timezone.utc).isoformat()
                knowledge.doc_sources = doc_sources

                # Compute confidence from source quality
                if doc_sources:
                    max_source_conf = max(s.get("confidence", 0) for s in doc_sources)
                    knowledge.confidence = min(0.90, max_source_conf)
                else:
                    knowledge.confidence = 0.50

                return knowledge
        except Exception:
            logger.debug("api_scout: LLM extraction failed", exc_info=True)

        return APIKnowledge(
            name=api_name,
            discovered_at=datetime.now(timezone.utc).isoformat(),
            doc_sources=doc_sources,
            confidence=0.3,
        )

    def _parse_openapi_spec(self, api_name: str, spec: dict) -> Optional[APIKnowledge]:
        """Parse an OpenAPI/Swagger spec directly into APIKnowledge."""
        if not spec or not isinstance(spec, dict):
            return None

        try:
            info = spec.get("info", {})
            servers = spec.get("servers", [])
            base_url = servers[0].get("url", "") if servers else ""

            endpoints = []
            paths = spec.get("paths", {})
            for path, methods in paths.items():
                for method, details in methods.items():
                    if method in ("get", "post", "put", "delete", "patch"):
                        params = []
                        for p in details.get("parameters", []):
                            params.append({
                                "name": p.get("name", ""),
                                "type": p.get("schema", {}).get("type", "string"),
                                "required": p.get("required", False),
                                "in": p.get("in", "query"),
                            })
                        endpoints.append(EndpointSpec(
                            path=path,
                            method=method.upper(),
                            description=details.get("summary", details.get("description", "")),
                            parameters=params,
                        ))

            # Detect auth from security schemes
            auth_type = ""
            security_schemes = spec.get("components", {}).get("securitySchemes", {})
            for scheme_name, scheme in security_schemes.items():
                stype = scheme.get("type", "")
                if stype == "apiKey":
                    loc = scheme.get("in", "header")
                    auth_type = "api_key_header" if loc == "header" else "api_key_query"
                elif stype == "oauth2":
                    auth_type = "oauth2_client_credentials"
                elif stype == "http" and scheme.get("scheme") == "bearer":
                    auth_type = "api_key_header"
                elif stype == "http" and scheme.get("scheme") == "basic":
                    auth_type = "basic_auth"

            return APIKnowledge(
                name=api_name,
                base_url=base_url,
                version=info.get("version", ""),
                description=info.get("description", "")[:500],
                auth_type=auth_type,
                endpoints=endpoints,
                confidence=0.95,
                discovered_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception:
            return None

    def _persist_knowledge(self, knowledge: APIKnowledge) -> None:
        """Save API knowledge to disk."""
        filename = knowledge.name.lower().replace(" ", "_")
        path = self._knowledge_dir / f"{filename}.json"
        path.write_text(json.dumps(knowledge.to_dict(), indent=2))

    def _primary_source_type(self, sources: list[dict]) -> str:
        """Determine the primary source type from doc sources."""
        if not sources:
            return "trial_and_error"
        # Return the type of the highest-confidence source
        best = max(sources, key=lambda s: s.get("confidence", 0))
        type_map = {
            "openapi_spec": "openapi_spec",
            "official_docs": "official_docs",
            "official_sdk": "official_sdk",
            "youtube_tutorial": "youtube_tutorial",
            "community": "blog_post",
            "hint": "official_docs",
        }
        return type_map.get(best.get("type", ""), "trial_and_error")

    def get_known_apis(self) -> list[str]:
        """Return list of APIs we have knowledge about."""
        return list(self._knowledge_cache.keys())

    def get_api_knowledge(self, api_name: str) -> Optional[APIKnowledge]:
        """Look up cached API knowledge."""
        return self._knowledge_cache.get(api_name.lower())


# ── Module-level singleton ───────────────────────────────────────────────────

_scout: APIScout | None = None


def get_scout() -> APIScout:
    """Get or create the singleton API scout."""
    global _scout
    if _scout is None:
        _scout = APIScout()
    return _scout
