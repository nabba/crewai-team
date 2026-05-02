# Cross-Crew Validation Protocol

## Problem
Data degradation occurs when moving from the Coding/Research phase to the Writing phase, as the writer may misinterpret raw data outputs.

## Solution
Implement a mandatory 'Validation Artifact' for every data-heavy task.

### Workflow
1. **Coding Crew**: After generating a dataset, create a `validation_summary.json` containing:
   - Total records processed.
   - Key aggregates (mean, max, min).
   - A sample of 3 'gold standard' rows.
2. **Writing Crew**: Before drafting, the writer must load the `validation_summary.json` and use it as a constraint for all claims made in the document.
3. **Self-Improvement Crew**: Audit the delta between the `validation_summary.json` and the final text to score accuracy.

### Success Metric
Zero discrepancy between raw code output aggregates and written report statistics.