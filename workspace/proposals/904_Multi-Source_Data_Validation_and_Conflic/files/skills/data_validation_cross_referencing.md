# Multi-Source Data Validation Protocol

## Objective
Ensure factual accuracy by resolving conflicts between multiple research sources.

## Workflow
1. **Extraction**: Extract key claims and their corresponding sources.
2. **Comparison**: Map claims against other sources to identify 'Confirmed', 'Contradicted', or 'Unique' data.
3. **Weighting**: Assign authority scores (e.g., Official Gov Site > Peer Reviewed Paper > News Article > Blog).
4. **Resolution**: 
   - If Confirmed: Use as high-confidence fact.
   - If Contradicted: Prioritize the higher-authority source but note the discrepancy in the report.
   - If Unique: Mark as 'unverified' unless the source is highly authoritative.
5. **Synthesis**: Write the final response citing the level of consensus.