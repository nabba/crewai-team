# Cross-Crew Synchronization Protocol

## Objective
Minimize information decay during transitions between Research -> Coding -> Writing crews.

## The Handoff Artifact
Every crew must produce a `handoff_artifact.json` containing:
1. **Core Findings**: Key data points verified by tools.
2. **Assumptions**: Unverified hypotheses that require checking.
3. **Constraints**: Technical or stylistic boundaries identified.
4. **Reference Map**: List of URLs and file paths used.

## Workflow
- **Research**: Creates the artifact after completing the data gathering phase.
- **Coding**: Uses the artifact to define function signatures and data schemas before implementation.
- **Writing**: Uses the artifact to ensure all technical claims are mapped back to research sources.