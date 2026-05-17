# Multi-Step Tool Chain Verification

## Objective
Reduce runtime errors in the `code_executor` by validating external data retrieved via `web_search` or `web_fetch`.

## Verification Workflow
1. **Extraction Phase**: Use `web_fetch` to get content.
2. **Validation Phase**: Before passing data to `code_executor`, the agent must:
   - Check for empty responses.
   - Identify the data format (JSON, HTML table, Plain Text).
   - Clean noise (headers, footers, ads) using a regex or simple Python script.
3. **Execution Phase**: Pass the *cleaned* and *validated* snippet to the coding crew.

## Failure Recovery
- If `code_executor` throws a `TypeError` or `ValueError` related to data format, the agent must immediately return to the **Validation Phase** to refine the data cleaning logic rather than blindly retrying the code.