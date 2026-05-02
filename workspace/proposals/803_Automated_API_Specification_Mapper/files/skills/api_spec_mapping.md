# API Specification Mapping Protocol

## Objective
Reduce coding errors by creating a formal mapping of API endpoints before implementation.

## Process
1. **Discovery**: Use `web_search` to find the latest official API documentation.
2. **Extraction**: Use `web_fetch` to extract:
   - Base URL
   - Authentication method (OAuth2, API Key, etc.)
   - Endpoint paths and HTTP methods
   - Required vs Optional parameters
   - Expected response schemas
3. **Schema Generation**: Create a `spec.json` file in the workspace containing these details.
4. **Validation**: The coding crew must reference `spec.json` rather than raw documentation to prevent hallucinated parameters.