# QA Verification Protocol

## Objective
Ensure all technical and creative outputs are verified against requirements before final submission.

## Workflow
1. **Requirement Mapping**: Cross-reference the final output against every explicit constraint in the original prompt.
2. **Technical Validation**:
   - For Code: Run test cases in the Docker sandbox covering edge cases.
   - For Research: Verify claims via a second, independent web search query.
3. **Consistency Check**: Ensure terminology is consistent across the entire document/codebase.
4. **Failure Analysis**: If an error is found, document the 'Failure Mode' and update the system prompt to prevent recurrence.