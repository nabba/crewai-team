# Cross-Crew Coordination & Handover Pattern

## Goal
Eliminate redundant research and communication gaps between specialist crews.

## Handover Requirements
Every crew must leave a 'Handover Note' in shared memory when completing a task:
- **Context**: What was achieved.
- **Constraints**: What was attempted but failed (to prevent the next crew from repeating mistakes).
- **Artifacts**: Exact paths to files in `/app/workspace/`.
- **Pending Questions**: Unresolved ambiguities for the next agent.

## Memory Tagging
Use metadata tags: `crew=research`, `crew=coding`, `crew=writing` to allow the `self_improvement` crew to audit the pipeline efficiency.