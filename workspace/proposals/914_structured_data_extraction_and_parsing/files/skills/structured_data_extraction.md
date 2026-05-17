# Structured Data Extraction Protocol

## Problem
Raw output from `web_fetch` is often noisy, containing boilerplate and unstructured text that makes it difficult for the coding crew to process or for the writing crew to tabulate.

## Solution
Implement a 'Fetch-Extract-Validate' pipeline:
1. **Fetch**: Use `web_fetch` or `browser_fetch` to get raw content.
2. **Extract**: Pass the content to the LLM with a strict JSON schema definition. Use few-shot prompting to ensure the model extracts only requested entities (e.g., names, dates, prices).
3. **Validate**: Use the coding crew to verify the JSON is well-formed and contains no hallucinations by cross-referencing the raw text.
4. **Store**: Save the structured data to a file via `file_manager` for downstream use.