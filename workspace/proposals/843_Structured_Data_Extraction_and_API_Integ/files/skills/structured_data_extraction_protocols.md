# Structured Data Extraction Protocols

## Objective
Convert unstructured web content into machine-readable formats to enable automated analysis and database insertion.

## Workflow
1. **Schema Definition**: Before fetching, define the required fields (e.g., `company_name`, `api_endpoint`, `pricing_tier`).
2. **Pattern Identification**: Use `web_fetch` to identify HTML table structures or JSON-LD blocks.
3. **Extraction Strategy**:
    - For Tables: Use `code_executor` with Pandas to parse HTML tables.
    - For JSON-LD: Extract script tags with `type="application/ld+json"`.
    - For Text: Use LLM-based entity extraction with a strict JSON output format.
4. **Validation**: Verify extracted data against the schema to ensure no missing values or type mismatches.

## Error Handling
- Handle pagination by detecting 'Next' buttons via `web_fetch` content analysis.
- Manage rate limits by implementing exponential backoff in the `code_executor` scripts.