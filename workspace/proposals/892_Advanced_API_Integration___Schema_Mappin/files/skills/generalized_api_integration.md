# Generalized API Integration Framework

## Protocol for New API Integration
1. **Discovery**: Use `web_search` to find the official API documentation and authentication method (OAuth2, API Key, etc.).
2. **Schema Analysis**: Use `web_fetch` to retrieve JSON schema or OpenAPI specifications. Identify mandatory vs optional parameters.
3. **Prototyping**: Use `code_executor` to run a minimal connectivity test (ping/auth check).
4. **Mapping**: Create a mapping table between the user's natural language request and the API endpoint parameters.
5. **Error Handling**: Implement exponential backoff for 429 (Rate Limit) and 5xx errors as per the `api_credit_management` skill.