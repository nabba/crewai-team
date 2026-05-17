---
aliases:
- skill validating file existence before automation tasks 6bc96b93
author: idle_scheduler.wiki_synthesis
confidence: medium
created_at: '2026-05-16T22:22:06Z'
date: '2026-05-16'
related: []
relationships: []
section: meta
source: workspace/skills/skill__validating_file_existence_before_automation_tasks__6bc96b93.md
status: active
tags:
- self-improvement
- skills
- auto-synthesised
title: 'SKILL: Validating File Existence Before Automation Tasks'
updated_at: '2026-05-16T22:22:06Z'
version: 1
---

<!-- generated-by: self_improvement.integrator -->
# SKILL: Validating File Existence Before Automation Tasks

*kb: experiential | id: skill_experiential_61d9c0c56bc96b93 | status: active | usage: 0 | created: 2026-05-02T22:44:30+00:00*

# SKILL: Validating File Existence Before Automation Tasks

## Topic
Pre-execution verification to prevent failed automation workflows due to missing files or unavailable tools.

## When to use
- Before attempting any file attachment, export, or signal operation
- When automation depends on specific output files existing in predictable locations
- When task instructions reference files that may not have been generated yet
- Before committing to multi-step workflows with external dependencies

## Procedure

1. **Inventory requested assets**: List all files the task expects (names, paths, formats)

2. **Search workspace systematically**: Check `output/`, `data/`, and project root directories; note actual files present with timestamps

3. **Identify naming mismatches**: Compare requested names against actual files (e.g., `2012_2025` vs. `2026-05-02` timestamps suggest regeneration happened)

4. **Audit available tools**: Verify the actual tool list attached to session matches task requirements (e.g., desktop control, file I/O, API access)

5. **Report blocking issues clearly**: Document missing files/tools with alternatives found; halt execution until user confirms which existing file to use or provides missing dependency

## Pitfalls

- **Assumption trap**: Don't assume files exist just because they were mentioned in setup instructions—regeneration may have changed names or paths
- **Timestamp confusion**: Recent file dates don't guarantee the *correct* file is present; verify content headers, not just timestamps
- **Silent tool gaps**: Never assume tool capabilities match persona descriptions—always reference the actual attached toolset
- **Proceeding anyway**: Attempting workarounds without user confirmation wastes tokens and creates misleading "success" reports on failed tasks

**Golden rule**: Honest blocking-issue reports save more time than optimistic execution of broken workflows.
