# API Integration Framework

## Step 1: Discovery
- Use `web_search` to find official API documentation.
- Identify authentication methods (API Key, OAuth2, etc.).
- Determine rate limits and quota constraints.

## Step 2: Prototyping
- Use `code_executor` to perform a 'Hello World' request.
- Map the JSON response structure to a flat internal schema.

## Step 3: Toolization
- Wrap the API call in a robust Python function with error handling for 429 (Too Many Requests) and 500 (Server Error).
- Document the tool's arguments and expected output for other agents.