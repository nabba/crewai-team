# Proposal #805: Structured Data Extraction & Schema Mapping

**Type:** skill  
**Created:** 2026-05-02T18:32:34.330597+00:00  

## Why this is useful

The current team has powerful web search and fetch capabilities but lacks a standardized protocol for converting unstructured HTML/text into structured JSON formats (e.g., for the Lead Generation skill). This often leads to inconsistent data formats between the research and coding crews. I propose a skill that defines a systematic approach to 'Schema-First Extraction', requiring agents to define a target JSON schema before fetching data to ensure 100% compatibility with downstream databases.

## What will change

- Modifies `skills/structured_data_extraction_protocol.md`

## Potential risks to other subsystems

- Uncategorised (skills): impact scope unclear

## Files touched

- `skills/structured_data_extraction_protocol.md`

## Original description

The current team has powerful web search and fetch capabilities but lacks a standardized protocol for converting unstructured HTML/text into structured JSON formats (e.g., for the Lead Generation skill). This often leads to inconsistent data formats between the research and coding crews. I propose a skill that defines a systematic approach to 'Schema-First Extraction', requiring agents to define a target JSON schema before fetching data to ensure 100% compatibility with downstream databases.

---

**To decide:** react 👍 to the Signal notification to approve, or 👎 to reject.  
Or reply `approve 805` / `reject 805` via Signal.
