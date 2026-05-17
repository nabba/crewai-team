# Cross-Crew Dependency Management

## Objective
Synchronize state across research, coding, and writing crews to prevent redundancy and data loss.

## The Handoff Manifest
Every single task must maintain a `manifest.json` in the workspace:
```json
{
  "task_id": "unique_id",
  "status": "in_progress|completed",
  "artifacts": {
    "research": "/app/workspace/research_notes.md",
    "data": "/app/workspace/cleaned_data.csv",
    "visuals": ["/app/workspace/output/chart1.png"]
  },
  "dependencies": ["research_complete", "data_validated"]
}
```

## Crew Responsibilities
- **Research Crew**: Initializes the manifest and populates the `research` artifact.
- **Coding Crew**: Reads the manifest, processes the research artifact, and adds the `data` and `visuals` artifacts.
- **Writing Crew**: Only begins synthesis once all dependencies in the manifest are marked as completed.