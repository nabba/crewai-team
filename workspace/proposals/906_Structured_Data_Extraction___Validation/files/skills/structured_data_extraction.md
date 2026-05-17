# Structured Data Extraction & Validation

## Objective
Transform raw `web_fetch` and `web_search` results into machine-readable, validated formats before passing them to the writing or coding crews.

## The EVF Pattern (Extract, Validate, Format)
1. **Extract**: Use the LLM to identify specific entities (dates, prices, names, specs) from raw text.
2. **Validate**: Check for contradictions. If the search results provide conflicting dates for the same event, flag it for the research crew to resolve.
3. **Format**: Output the final data as a JSON object or Markdown table.

## Implementation Guidelines
- Always define a schema first (e.g., "I need a list of PSPs with: Name, Region, License Type, API Availability").
- Use the `code_executor` to run a quick Python script to validate JSON schema integrity before storing in `team_memory_store`.