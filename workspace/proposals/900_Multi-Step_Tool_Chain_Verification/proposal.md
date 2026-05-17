# Proposal #900: Multi-Step Tool Chain Verification

**Type:** skill  
**Created:** 2026-05-17T01:11:58.098133+00:00  

## Why this is useful

Problem: Complex tasks involving both `web_search` and `code_executor` often fail because the agent assumes the search result is perfectly formatted for the code. Solution: A verification skill that mandates a 'Data Validation' step before passing external web data into the coding sandbox.

## What will change

- Modifies `skills/tool_chain_verification.md`

## Potential risks to other subsystems

- Uncategorised (skills): impact scope unclear

## Files touched

- `skills/tool_chain_verification.md`

## Original description

Problem: Complex tasks involving both `web_search` and `code_executor` often fail because the agent assumes the search result is perfectly formatted for the code. Solution: A verification skill that mandates a 'Data Validation' step before passing external web data into the coding sandbox.

---

**To decide:** react 👍 to the Signal notification to approve, or 👎 to reject.  
Or reply `approve 900` / `reject 900` via Signal.
