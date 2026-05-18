# Automated API Client Generation

## Workflow
1. **Fetch Specification**: Use `web_fetch` to obtain the `swagger.json` or `openapi.json` from the target service.
2. **Schema Analysis**: Identify base URLs, authentication methods (Bearer, API Key), and endpoint structures.
3. **Boilerplate Generation**: Use the `code_executor` to generate a Python class using `requests` or `httpx` that maps every endpoint to a method.
4. **Validation**: Run a connectivity test on a single 'GET' endpoint to verify the generated client.

## Best Practices
- Always include retry logic and timeout handling.
- Use Pydantic models for request/response validation to ensure data integrity before passing to the Writing crew.