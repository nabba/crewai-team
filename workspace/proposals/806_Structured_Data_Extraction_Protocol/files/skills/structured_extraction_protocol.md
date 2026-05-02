# Structured Data Extraction Protocol

## Objective
Convert unstructured web content into machine-readable formats (JSON/CSV) with 100% schema adherence.

## Workflow
1. **Schema Definition**: Define the target JSON schema explicitly before extraction.
2. **Chunked Processing**: For long pages, extract entities per section to avoid context window truncation.
3. **Validation Step**: Use the coding crew to validate that the output is valid JSON and matches the required keys.
4. **Confidence Scoring**: Mark fields as 'null' if not explicitly found rather than guessing.

## Prompt Pattern
'Extract the following entities [List] from the text. Output ONLY a JSON array. If a value is missing, use null. Schema: {key: type}'