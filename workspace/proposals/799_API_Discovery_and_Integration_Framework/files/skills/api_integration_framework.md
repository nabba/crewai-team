# API Discovery and Integration Framework

## Problem
Integration of new data sources is currently ad-hoc, often leading to redundant code or failure to handle rate limits and authentication properly.

## Framework Steps
1. **Discovery**: Use `web_search` to find official API documentation and developer portals.
2. **Capability Mapping**: Map API endpoints to specific team goals (e.g., 'GET /leads' -> Lead Gen crew).
3. **Authentication Audit**: Identify required auth methods (OAuth2, API Key, Basic) and verify secret management via environment variables.
4. **Sandbox Testing**: Implement a minimal Python script in `code_executor` to verify a single endpoint response before full integration.
5. **Error Handling Implementation**: Apply `api_credit_management_and_quota_error_handling` skill to the new implementation to ensure resilience.