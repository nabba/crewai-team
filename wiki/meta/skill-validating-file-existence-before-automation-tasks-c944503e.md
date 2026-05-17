---
aliases:
- skill validating file existence before automation tasks c944503e
author: idle_scheduler.wiki_synthesis
confidence: medium
created_at: '2026-05-16T22:59:48Z'
date: '2026-05-16'
related: []
relationships: []
section: meta
source: workspace/skills/skill__validating_file_existence_before_automation_tasks__c944503e.md
status: active
tags:
- self-improvement
- skills
- auto-synthesised
title: 'SKILL: Validating File Existence Before Automation Tasks'
updated_at: '2026-05-16T22:59:48Z'
version: 1
---

<!-- generated-by: self_improvement.integrator -->
# SKILL: Validating File Existence Before Automation Tasks

*kb: experiential | id: skill_experiential_fcfcad48c944503e | status: active | usage: 0 | created: 2026-05-04T14:57:19+00:00*

# SKILL: Validating File Existence Before Automation Tasks

**Topic:** Verify file accessibility and location before invoking automated processes to prevent blocked execution.

**When to use:**
- Before running crew/agent automation tasks
- When delegating work to external systems
- When file paths are assumed but not confirmed
- In sandbox/restricted environments with limited tool access

**Procedure:**

1. **Map available tools & constraints**
   - List attached tools and their scope (e.g., `file_manager` restricted to `output/`, `skills/`, `proposals/`)
   - Note which tools can execute host code vs. sandbox operations
   - Identify any environment restrictions (firewall, directory permissions)

2. **Verify source file location**
   - Query workspace structure: `file_manager(action='list', path='')`
   - Search recursively for target file: `grep -rln "search_term"` (if host execution available)
   - Confirm file exists *and* is in an accessible directory

3. **Check tool-to-resource alignment**
   - Does an available tool have permission to read/write the required path?
   - If file is outside sandbox, can any attached tool reach it (e.g., SSH, host executor)?
   - If blocked: escalate or request tool attachment before proceeding

4. **Fail fast with diagnostic output**
   - Document what was attempted, what succeeded, what failed
   - Provide the *exact* fix (shell commands, file paths, tool requirements)
   - Don't retry with same constraints—request missing access

5. **Distill into reusable knowledge**
   - Save discovery steps as a SKILL for future similar tasks
   - Include constraints encountered and resolution method

**Pitfalls:**
- ❌ Assuming file locations without validation → blocked tasks, wasted retries
- ❌ Trying different paths with same limited tool → diminishing returns
- ❌ Proceeding without identifying missing tool access
- ✅ Validate *early*, document *thoroughly*, escalate *clearly*
