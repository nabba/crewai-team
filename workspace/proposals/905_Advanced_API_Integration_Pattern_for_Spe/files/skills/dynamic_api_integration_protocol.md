# Dynamic API Integration Protocol

## Goal
Rapidly integrate new data providers without full codebase rewrites.

## Steps
1. **Auth Mapping**: Identify auth type (OAuth2, API Key, Basic) and store securely in environment variables via `mcp_add_server` or secure config.
2. **Schema Discovery**: Use `web_fetch` on API documentation to map endpoint paths and required parameters.
3. **Payload Prototyping**: Use `code_executor` to run a single test request and validate the JSON response structure.
4. **Error Handling**: Implement the `api_credit_management_and_quota_error_handling` skill to wrap the new integration.