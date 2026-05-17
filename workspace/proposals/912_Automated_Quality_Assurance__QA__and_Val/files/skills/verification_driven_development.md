# Verification-Driven Development (VDD) Protocol

## Objective
Ensure all code and reports are validated against a set of 'Acceptance Criteria' before delivery.

## Workflow
1. **Requirement Extraction**: Before coding, define 3-5 binary (Pass/Fail) criteria.
2. **Test Generation**: Create a `test_validation.py` script in the `code_executor` to verify outputs.
3. **Cross-Crew Audit**: The `research` crew must validate the facts in the `writing` crew's draft using `web_fetch` on primary sources.
4. **Failure Loop**: If a test fails, the agent must return to the 'coding' or 'research' phase without notifying the user until the fix is verified.