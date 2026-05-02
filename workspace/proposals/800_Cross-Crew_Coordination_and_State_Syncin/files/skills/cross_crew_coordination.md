# Cross-Crew Coordination Protocol

## Purpose
Ensure seamless knowledge transfer between Research, Coding, and Writing crews.

## The Shared State Ledger
Every crew must update the `team_memory_store` at these critical junctions:
1. **Research -> Coding**: When a technical specification or API endpoint is verified.
2. **Coding -> Writing**: When a prototype's limitations or specific implementation details are finalized.
3. **Writing -> Research**: When a gap in the narrative is identified that requires further data.

## Sync Commands
- `sync_start`: Retrieve all relevant team memory tags before starting a task.
- `sync_update`: Push a summary of findings using tags like `#technical_blocker` or `#verified_fact`.