# Cross-Crew Quality Assurance Protocol

## Objective
Ensure 100% factual alignment between Research, Coding, and Writing outputs.

## Workflow
1. **Source Mapping**: The Writing crew must create a hidden 'Verification Table' mapping every quantitative claim to a source file (e.g., `research_notes.txt` or `execution_output.log`).
2. **Discrepancy Flagging**: If the Writer finds a contradiction between Research and Coding output, they must trigger a 'Reconciliation Task' sent back to the Research lead.
3. **Final Validation**: Before final output, the Research crew must perform a 'Blind Review' of the final report to ensure no hallucinations were introduced during the synthesis phase.

## Success Criteria
- No claims without a corresponding source.
- All data points verified against the latest code_executor output.