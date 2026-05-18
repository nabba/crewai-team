# Cross-Crew Synchronization Protocol

## Objective
Prevent information loss during hand-offs between Research, Coding, and Writing crews.

## The Handover Manifest
Every crew must produce a `HANDOVER.md` file when transitioning a task. It must contain:
1. **Core Findings**: 3-5 bullet points of the most critical discoveries.
2. **Technical Constraints**: Specific limits, API quirks, or bugs encountered (Crucial for Coding -> Writing).
3. **Verification Status**: What has been proven vs. what is an assumption.
4. **Next Action Required**: Explicit instructions for the receiving crew.

## Workflow
- **Research -> Coding**: Include API endpoints, data schemas, and required libraries.
- **Coding -> Writing**: Include execution logs, performance metrics, and 'gotchas' in the logic.
- **Writing -> Self-Improvement**: Include user feedback gaps and areas where the AI struggled to articulate the solution.