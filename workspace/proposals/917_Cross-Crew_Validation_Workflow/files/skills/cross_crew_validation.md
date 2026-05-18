# Cross-Crew Validation Workflow

## Objective
Eliminate propagation of errors between research, coding, and writing crews.

## Workflow
1. **Research Output**: Research crew marks findings as 'PROVISIONAL'.
2. **Coding Verification**: Coding crew writes a quick validation script in the `code_executor` to check the consistency of the data (e.g., checking if dates are sequential or if URLs are active).
3. **Verification Stamp**: If validated, the coding crew updates the status to 'VERIFIED'.
4. **Writing Trigger**: Writing crew is prohibited from using 'PROVISIONAL' data for final report generation.
5. **Feedback Loop**: If verification fails, the task is routed back to the research crew with a specific error log.