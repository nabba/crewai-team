# Structured Data Extraction & Schema Mapping

## Objective
Ensure consistent, machine-readable output when transitioning data from `web_fetch` to the `coding` crew.

## Workflow
1. **Schema Definition**: Before executing `web_fetch`, the agent must define a JSON schema (keys and types) for the desired data.
2. **Targeted Extraction**: Use LLM prompting to extract specifically for that schema, avoiding conversational filler.
3. **Validation**: Pass the output through a validation step to ensure no missing required fields.
4. **Normalization**: Convert dates, currencies, and units to a standard format (ISO 8601, USD, Metric) during extraction.

## Example Prompting Pattern
'Extract the following entities from the text: [Entity List]. Format the output as a JSON array strictly following this schema: { "company_name": "string", "hq_location": "string" }.'