# Cross-Crew State Synchronization

## Problem
Information loss occurs during hand-offs between the research crew (finding data) and the writing crew (formatting data), leading to inaccuracies in final reports.

## Solution
Maintain a `project_state.json` in the workspace:
- **Research Crew**: Must update the `findings` section of the manifest whenever a new fact is confirmed.
- **Coding Crew**: Must update the `assets` section (paths to generated plots/CSV) when scripts finish execution.
- **Writing Crew**: Must read the `project_state.json` as the primary context window before starting a draft.
- **Protocol**: No crew should rely solely on chat history for critical data; they must refer to the manifest.