# Cross-Crew Coordination Protocol

## Objective
Ensure seamless data flow between Research -> Coding -> Writing crews to prevent redundant work and information loss.

## Handover Process
1. **Research Phase**: Research crew must store a 'Findings Manifest' in shared memory containing: 
   - Verified facts
   - Source URLs
   - Unresolved gaps
2. **Coding Phase**: Coding crew must reference the Manifest before generating scripts to ensure data structures match research findings.
3. **Writing Phase**: Writing crew must validate the final draft against the 'Findings Manifest' for factual accuracy.

## State Tracking
Use a status flag in team memory: `[TASK_ID]: {status: 'RESEARCHING|CODING|WRITING|REVIEWING', lead: 'agent_name'}`.