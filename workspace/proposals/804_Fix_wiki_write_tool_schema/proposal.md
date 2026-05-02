# Proposal #804: Fix wiki_write tool schema

**Type:** code  
**Created:** 2026-05-02T11:31:43.299968+00:00  

## Why this is useful

Diagnosis: The `wiki_write` tool definition has an invalid JSON schema where the 'title' parameter is listed as required but is missing from the properties definition, or the 'required' array is improperly formatted for the Azure provider.

Fix: Update the tool definition for 'wiki_write' to ensure the 'required' array exactly matches the keys defined in the 'properties' section of the JSON schema.

## What will change

- (no file changes)

## Potential risks to other subsystems

- Requires `docker compose up -d --build gateway` to take effect

## Files touched

None

## Original description

Diagnosis: The `wiki_write` tool definition has an invalid JSON schema where the 'title' parameter is listed as required but is missing from the properties definition, or the 'required' array is improperly formatted for the Azure provider.

Fix: Update the tool definition for 'wiki_write' to ensure the 'required' array exactly matches the keys defined in the 'properties' section of the JSON schema.

---

**To decide:** react 👍 to the Signal notification to approve, or 👎 to reject.  
Or reply `approve 804` / `reject 804` via Signal.
