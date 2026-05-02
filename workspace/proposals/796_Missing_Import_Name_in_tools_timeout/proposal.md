# Proposal #796: Missing Import Name in tools_timeout

**Type:** code  
**Created:** 2026-05-02T06:56:20.643550+00:00  

## Why this is useful

Diagnosis: The function 'seconds_since_last_tool_activity' is being imported in 'app/main.py' at line 1487, but it is not defined or exported in 'app/tools_timeout.py'.

Fix: Verify if 'seconds_since_last_tool_activity' was renamed or deleted in 'app/tools_timeout.py'. Either restore the function definition in 'app/tools_timeout.py' or update the import statement in 'app/main.py' to use the correct function name.

## What will change

- (no file changes)

## Potential risks to other subsystems

- Requires `docker compose up -d --build gateway` to take effect

## Files touched

None

## Original description

Diagnosis: The function 'seconds_since_last_tool_activity' is being imported in 'app/main.py' at line 1487, but it is not defined or exported in 'app/tools_timeout.py'.

Fix: Verify if 'seconds_since_last_tool_activity' was renamed or deleted in 'app/tools_timeout.py'. Either restore the function definition in 'app/tools_timeout.py' or update the import statement in 'app/main.py' to use the correct function name.

---

**To decide:** react 👍 to the Signal notification to approve, or 👎 to reject.  
Or reply `approve 796` / `reject 796` via Signal.
