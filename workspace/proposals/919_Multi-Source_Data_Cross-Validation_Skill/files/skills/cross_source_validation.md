# Multi-Source Data Cross-Validation Protocol

## Objective
Ensure factual accuracy when multiple tools/sources return conflicting data.

## Workflow
1. **Source Tiering**: Assign trust levels (e.g., Official API > News Article > Blog).
2. **Conflict Detection**: Flag discrepancies in quantitative values or key dates.
3. **Resolution Logic**:
   - If Source A (Tier 1) conflicts with Source B (Tier 2), prioritize Tier 1.
   - If Tier 1 sources conflict, search for a third 'Tie-breaker' source.
   - If no resolution is found, explicitly report the discrepancy as 'Contested Information'.
4. **Verification**: Cross-reference the resolved value with a known constant or secondary metric.