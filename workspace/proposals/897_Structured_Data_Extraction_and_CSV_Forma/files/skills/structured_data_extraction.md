# Structured Data Extraction & Formatting

## Objective
Convert unstructured text from `web_search` and `web_fetch` into structured tabular data for lead lists, market analysis, and reporting.

## Protocol
1. **Schema Definition**: Define required columns (e.g., Company Name, Website, Contact, Location, Service Type) before extraction.
2. **Extraction Loop**: Iterate through research sources, extracting entities that match the schema.
3. **Validation**: Cross-reference extracted data points to ensure consistency.
4. **Formatting**: Output data in a clean CSV format using the `code_executor` to ensure correct escaping and encoding.

## Tool Integration
- Use `code_executor` to generate `.csv` files in the `/app/workspace/` directory.
- Pass CSV paths to the `writing` crew for final report synthesis.