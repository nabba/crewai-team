# Multi-Crew Orchestration Protocol

## Purpose
To prevent context degradation during hand-offs between Research, Coding, and Writing crews.

## State Document Requirement
Every project must maintain a `project_state.json` file containing:
- `current_phase`: (e.g., Research -> Analysis -> Drafting)
- `completed_milestones`: List of verified outputs.
- `pending_dependencies`: What the next crew needs before starting.
- `context_summary`: A concise summary of findings to prevent redundant searches.

## Hand-off Procedure
1. **Outbound**: The departing crew updates `project_state.json` and explicitly calls out the next crew's entry point.
2. **Inbound**: The arriving crew reads the state file first to synchronize their internal memory with the project's actual progress.