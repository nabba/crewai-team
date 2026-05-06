#!/usr/bin/env python3
"""Introspector parity panel — Stage 1 → Stage 2 validation harness.

Drives 25 representative introspector tasks against both the legacy
plain-Agent path AND the LoadableAgent path, then reports the metrics
that gate the default flip per docs/TOOL_REGISTRY_PHASE_5.md §3:

  * Live parity: ≥0.90× legacy success rate
  * Token usage: validated separately via analyze_telemetry()
  * No new failure modes (operator review of any pair flagged below)

Per-task output goes to workspace/observability/parity_introspector.jsonl
so the operator can diff legacy vs loadable outputs side-by-side.

Usage (from the host — recommended; the container path also works):

    docker exec -it crewai-team-gateway-1 \
        python /app/scripts/parity_introspector.py

    # Subset (smoke test before the full panel):
    docker exec -it crewai-team-gateway-1 \
        sh -c 'PARITY_TASKS=3 python /app/scripts/parity_introspector.py'

    # Dry run (prints task list + categories, no LLM calls):
    docker exec -it crewai-team-gateway-1 \
        sh -c 'PARITY_DRY_RUN=1 python /app/scripts/parity_introspector.py'

After the run, validate token economics on the loadable path:

    docker exec crewai-team-gateway-1 python -c \
        'from app.tool_runtime.telemetry import analyze_telemetry; \
         import json; print(json.dumps(analyze_telemetry(agent_id="introspector"), indent=2))'

Env knobs:
    PARITY_TASKS    int — run only the first N tasks (default: all 25)
    PARITY_MODE     legacy|loadable|both — which path to run (default: both)
    PARITY_DRY_RUN  1 to print + skip LLM calls
    PARITY_OUTPUT   path for JSONL results
                    (default: workspace/observability/parity_introspector.jsonl)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Task panel ──────────────────────────────────────────────────────────────
#
# 25 tasks in three categories:
#   A (10) — eager tools sufficient. Exercises parity: both paths
#            should produce equivalent output.
#   B (10) — naturally needs reads-knowledge-base / searches-web /
#            reads-file. Exercises load_tool: loadable should discover
#            and load; legacy should fail or improvise without.
#   C  (5) — edge cases (empty / malformed / conflicting / unknown
#            tool / long input). Tests robustness of both paths.

_EXPECTED_OUTPUT = (
    "A concrete improvement policy with three labeled fields: "
    "TRIGGER (when the policy applies), ACTION (what to do differently), "
    "EVIDENCE (what in the trace prompted it)."
)

TASKS: list[dict] = [
    # ── Category A — eager-tools-only ────────────────────────────────────
    {"id": "A01", "category": "A",
     "prompt": "Review this self-report from the researcher: 'Spent 3 "
               "iterations searching for company X, found nothing.' "
               "Propose a policy.",
     "needs_loadable": False},
    {"id": "A02", "category": "A",
     "prompt": "A reflection note says the writer agent often produces "
               "overly long outputs. Propose a policy.",
     "needs_loadable": False},
    {"id": "A03", "category": "A",
     "prompt": "Look up your memory for prior policies on handling "
               "missing data. Summarize patterns.",
     "needs_loadable": False},
    {"id": "A04", "category": "A",
     "prompt": "Recall the most recent reflection stored in your "
               "memory. Propose one improvement based on it.",
     "needs_loadable": False},
    {"id": "A05", "category": "A",
     "prompt": "A self-report shows the coder produced runnable but "
               "uncommented code. Propose a policy.",
     "needs_loadable": False},
    {"id": "A06", "category": "A",
     "prompt": "Two consecutive reflections mention vetting failures. "
               "Identify the pattern and propose a single policy.",
     "needs_loadable": False},
    {"id": "A07", "category": "A",
     "prompt": "Check existing policies in your memory before proposing "
               "a new one for handling refusals. Avoid duplicates.",
     "needs_loadable": False},
    {"id": "A08", "category": "A",
     "prompt": "Write a reflection on whether last week's policies "
               "addressed observed failures.",
     "needs_loadable": False},
    {"id": "A09", "category": "A",
     "prompt": "Summarize the top 3 recurring patterns from your "
               "stored memories.",
     "needs_loadable": False},
    {"id": "A10", "category": "A",
     "prompt": "Propose a policy for an agent that repeatedly times "
               "out without producing output.",
     "needs_loadable": False},

    # ── Category B — naturally needs discoverable tool ───────────────────
    {"id": "B01", "category": "B",
     "prompt": "Search the knowledge base for entries on 'meta-cognitive "
               "failure modes' and propose a policy informed by what you find.",
     "needs_loadable": True},
    {"id": "B02", "category": "B",
     "prompt": "Look up published guidance on agent reflection patterns "
               "and propose a related introspection policy.",
     "needs_loadable": True},
    {"id": "B03", "category": "B",
     "prompt": "Read the file at workspace/skills/learning_queue.md. "
               "Propose a policy that addresses one of its open entries.",
     "needs_loadable": True},
    {"id": "B04", "category": "B",
     "prompt": "Search the KB for prior policies tagged 'recovery'. "
               "Propose one that does NOT duplicate existing entries.",
     "needs_loadable": True},
    {"id": "B05", "category": "B",
     "prompt": "Find a recent paper on meta-cognitive AI in your "
               "knowledge base and summarize how it informs an "
               "introspection policy.",
     "needs_loadable": True},
    {"id": "B06", "category": "B",
     "prompt": "Read the AwareBench reference in the KB. Propose one "
               "policy informed by a specific finding from it.",
     "needs_loadable": True},
    {"id": "B07", "category": "B",
     "prompt": "Search the web for 'self-improvement agent loop'. "
               "Cite one finding when proposing your policy.",
     "needs_loadable": True},
    {"id": "B08", "category": "B",
     "prompt": "Read workspace/recovery/refusal_frequency.json. "
               "Propose a policy addressing the most-frequent gap.",
     "needs_loadable": True},
    {"id": "B09", "category": "B",
     "prompt": "Search the KB for prior introspection traces. Propose "
               "a meta-policy that aggregates patterns across them.",
     "needs_loadable": True},
    {"id": "B10", "category": "B",
     "prompt": "Look up CrewAI multi-agent guidance in the KB. Propose "
               "an introspection policy informed by it.",
     "needs_loadable": True},

    # ── Category C — edge cases ──────────────────────────────────────────
    {"id": "C01", "category": "C",
     "prompt": "",
     "needs_loadable": False},
    {"id": "C02", "category": "C",
     "prompt": "Self-report (malformed): {agent: 'researcher', status: "
               "INCOMPLETE_JSON_HERE — propose a policy anyway.",
     "needs_loadable": False},
    {"id": "C03", "category": "C",
     "prompt": "Self-report A says vetting works reliably. Self-report "
               "B from the same hour says vetting failed 4 of 6 times. "
               "Propose a single policy reconciling both.",
     "needs_loadable": False},
    {"id": "C04", "category": "C",
     "prompt": "The agent used the 'time_machine' tool to handle the "
               "task (this tool does not exist in the registry). "
               "Propose a policy.",
     "needs_loadable": False},
    {"id": "C05", "category": "C",
     "prompt": ("A long execution trace follows. " + (
         "The researcher began at 14:02 with a query for PSP companies "
         "in CEE, used web_search returning 12 hits, filtered to 4, "
         "wrote draft, was vetted as wrong-crew, recovery loop fired. "
     ) * 40) + " Propose a single policy that addresses the dominant "
              "pattern.",
     "needs_loadable": False},
]


# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = ROOT / "workspace" / "observability" / "parity_introspector.jsonl"
OUTPUT_PATH = Path(os.environ.get("PARITY_OUTPUT", str(DEFAULT_OUTPUT)))
TASK_LIMIT = int(os.environ.get("PARITY_TASKS", str(len(TASKS))))
MODE = os.environ.get("PARITY_MODE", "both").lower()
DRY_RUN = os.environ.get("PARITY_DRY_RUN", "").strip() == "1"


# ── Runner ───────────────────────────────────────────────────────────────────

def _structural_signals(text: str) -> dict:
    """Heuristic check for the introspector's expected output shape."""
    upper = text.upper()
    return {
        "has_trigger": "TRIGGER" in upper,
        "has_action": "ACTION" in upper,
        "has_evidence": "EVIDENCE" in upper,
        "non_empty": bool(text.strip()),
        "length_chars": len(text),
    }


def _run_task_on_agent(agent, prompt: str) -> tuple[str | None, str | None]:
    """Drive one task through a CrewAI agent. Returns (output, error)."""
    try:
        from crewai import Crew, Task
    except Exception as exc:
        return None, f"crewai import failed: {exc}"
    try:
        task = Task(
            description=prompt or "(no prompt provided)",
            expected_output=_EXPECTED_OUTPUT,
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = crew.kickoff()
        return str(result), None
    except Exception:
        return None, traceback.format_exc(limit=4)


def _run_path(label: str, task_dict: dict) -> dict:
    """Build the introspector with the appropriate flag, run the task,
    capture output + duration. Restores env on exit."""
    saved = os.environ.get("LOADABLE_INTROSPECTOR")
    os.environ["LOADABLE_INTROSPECTOR"] = "1" if label == "loadable" else "0"
    started = time.time()
    output, error = None, None
    try:
        from app.agents.introspector import create_introspector
        agent = create_introspector()
        if not DRY_RUN:
            output, error = _run_task_on_agent(agent, task_dict["prompt"])
        else:
            output = f"(dry run — task {task_dict['id']} would have executed)"
    except Exception:
        error = traceback.format_exc(limit=4)
    finally:
        if saved is None:
            os.environ.pop("LOADABLE_INTROSPECTOR", None)
        else:
            os.environ["LOADABLE_INTROSPECTOR"] = saved
    elapsed = time.time() - started
    return {
        "label": label,
        "output": output,
        "error": error,
        "elapsed_s": round(elapsed, 2),
        "signals": _structural_signals(output or ""),
    }


def _run_one(task_dict: dict) -> dict:
    """Run one task on the configured paths."""
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "id": task_dict["id"],
        "category": task_dict["category"],
        "prompt": task_dict["prompt"][:300],
        "needs_loadable": task_dict["needs_loadable"],
    }
    if MODE in ("legacy", "both"):
        row["legacy"] = _run_path("legacy", task_dict)
    if MODE in ("loadable", "both"):
        row["loadable"] = _run_path("loadable", task_dict)
    return row


# ── Summary ──────────────────────────────────────────────────────────────────

def _success(path: dict | None) -> bool:
    """A run is 'successful' if it returned non-empty structured output
    without an exception. The operator still reviews failures manually."""
    if not path:
        return False
    if path.get("error"):
        return False
    sig = path.get("signals") or {}
    if not sig.get("non_empty"):
        return False
    # Structural signal: at least one of the three policy fields present.
    return any(sig.get(k) for k in ("has_trigger", "has_action", "has_evidence"))


def _summarize(rows: list[dict]) -> str:
    out = ["# Introspector parity panel — summary", ""]
    out.append(f"Tasks run: {len(rows)} (mode={MODE}, dry_run={DRY_RUN})")
    out.append("")

    by_cat: dict[str, dict] = {}
    legacy_pass = loadable_pass = 0
    for r in rows:
        cat = r["category"]
        bucket = by_cat.setdefault(cat, {"total": 0, "legacy_pass": 0, "loadable_pass": 0})
        bucket["total"] += 1
        if _success(r.get("legacy")):
            bucket["legacy_pass"] += 1
            legacy_pass += 1
        if _success(r.get("loadable")):
            bucket["loadable_pass"] += 1
            loadable_pass += 1

    out.append("| Category | Total | Legacy pass | Loadable pass |")
    out.append("|---|---:|---:|---:|")
    for cat in sorted(by_cat):
        b = by_cat[cat]
        out.append(f"| {cat} | {b['total']} | {b['legacy_pass']} | {b['loadable_pass']} |")
    out.append("")

    total = sum(b["total"] for b in by_cat.values())
    if MODE == "both" and legacy_pass:
        ratio = loadable_pass / legacy_pass
        verdict = "PASS" if ratio >= 0.90 else "FAIL"
        out.append(f"**Stage 1 → Stage 2 acceptance criterion 1**:")
        out.append(f"  loadable_pass / legacy_pass = {loadable_pass} / {legacy_pass} "
                   f"= {ratio:.2f} → **{verdict}** (≥0.90 required)")
    elif MODE == "both":
        out.append(f"**Cannot compute parity ratio** — legacy_pass = 0.")
    out.append("")

    out.append(f"Total legacy success: {legacy_pass}/{total}")
    out.append(f"Total loadable success: {loadable_pass}/{total}")
    out.append("")
    out.append("Operator review queue (any task where exactly one path passed):")
    for r in rows:
        leg_ok = _success(r.get("legacy"))
        load_ok = _success(r.get("loadable"))
        if leg_ok != load_ok:
            out.append(f"  - {r['id']} ({r['category']}): legacy={leg_ok} loadable={load_ok}")
    out.append("")
    out.append(f"Per-task JSONL: {OUTPUT_PATH}")
    return "\n".join(out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tasks = TASKS[:TASK_LIMIT]
    print(f"# Introspector parity panel — {len(tasks)} tasks, mode={MODE}, "
          f"dry_run={DRY_RUN}")
    print(f"# Output: {OUTPUT_PATH}")
    print()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with OUTPUT_PATH.open("w") as fh:
        for i, task in enumerate(tasks, 1):
            print(f"[{i:>2}/{len(tasks)}] {task['id']} ({task['category']}) ...",
                  flush=True)
            row = _run_one(task)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows.append(row)

    print()
    print(_summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
