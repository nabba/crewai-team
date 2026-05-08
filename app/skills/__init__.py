"""
app.skills — Hermes-style "save this workflow" registry.

A `Skill` is a named, parameterized task description. ``/skill save morning
briefing: Today's calendar + top 3 urgent emails + weather`` persists a row
that ``/skill run morning briefing`` (or the React /cp/skills page) replays.

Skills are intentionally lightweight (one JSON file, no DB) so they survive
restarts even when the meta-agent layer is OFF. When the meta-agent IS on,
the runner records each run as a `RecipeOutcome` so the bandit picks up
data on the routing hint over time.

Public surface:

    save_skill(name, task_template, ...)     persist a skill
    get_skill(name)                          fetch by name
    list_skills()                            sorted by recent use
    delete_skill(name)                       remove
    expand(task_template, args)              substitute {placeholder}s
    run_skill(name, args, sender, commander) execute via commander.handle
"""
from __future__ import annotations

from app.skills.registry import (
    Skill, save_skill, get_skill, list_skills, delete_skill,
    record_run_result, extract_placeholders,
)
from app.skills.runner import expand, run_skill

__all__ = [
    "Skill",
    "save_skill", "get_skill", "list_skills", "delete_skill",
    "record_run_result", "extract_placeholders",
    "expand", "run_skill",
]
