"""parity.py — side-by-side comparison: stock vs LoadableAgent.

Phase 2 validation harness. Runs the same task panel against both
the legacy and Phase-2 paths, captures token usage + success
indicators, reports a comparison.

Two execution modes:

  * **dry**: doesn't make real LLM calls. Uses the analytical model
    from Phase 1c (``measure.compare_stock_vs_loadable``). Fast,
    deterministic, free. Default mode.
  * **live**: makes real LLM calls. Captures actual ``usage`` from
    Anthropic responses via the telemetry module. Costs money. Use
    when validating the Phase 1c gate post-merge.

CLI::

    docker exec crewai-team-gateway-1 python -m app.tool_runtime.parity
    # → dry-mode default report

    docker exec crewai-team-gateway-1 python -m app.tool_runtime.parity --live --runs 5
    # → 5 live calls per agent, real cache validation

Programmatic::

    from app.tool_runtime.parity import run_parity_panel
    report = run_parity_panel(mode="dry", runs=1)
    assert report["verdict"] == "GO"

The 50-task panel referenced in the Phase 1c memo lives in operator
hands — this module gives them the runner; they decide what tasks
exercise their pilot's behavior.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Default task panel — small, fast, illustrative. Operators can
# override by passing their own list to ``run_parity_panel``.
DEFAULT_PANEL: list[dict[str, Any]] = [
    {
        "name": "trivial_self_report",
        "description": (
            "Generate a one-paragraph self-report on what you'd do "
            "if asked to find a recurring failure pattern in 5 task "
            "execution traces. Use ONLY your reasoning, no tools."
        ),
        "expected_iterations": 1,  # no tool calls expected
        "expected_loads": 0,
    },
    {
        "name": "memory_lookup",
        "description": (
            "Retrieve any prior introspection notes you have stored "
            "in memory under the topic 'recurring_failures'. Summarize "
            "the top 3 in 2 sentences each. Use memory tools."
        ),
        "expected_iterations": 2,
        "expected_loads": 0,
    },
    {
        "name": "policy_synthesis",
        "description": (
            "Given the trace summary 'agents repeatedly hit timeout "
            "on long-running tools', synthesize one policy with "
            "TRIGGER + ACTION + EVIDENCE. Store it in scoped memory."
        ),
        "expected_iterations": 3,
        "expected_loads": 0,
    },
    {
        "name": "knowledge_assisted",
        "description": (
            "Look up 'meta-cognitive policy' in the knowledge base "
            "and integrate one relevant finding into a new policy. "
            "If you don't have the right tool, search the catalog."
        ),
        "expected_iterations": 4,
        "expected_loads": 1,  # may need to load knowledge_base tool
    },
    {
        "name": "web_grounded",
        "description": (
            "Synthesize a policy that references one current best "
            "practice for AI introspection. If you need fresh "
            "information, the catalog has tools for it."
        ),
        "expected_iterations": 4,
        "expected_loads": 1,  # web_search probably needs to load
    },
]


@dataclass
class TaskResult:
    name: str
    success: bool
    iterations: int
    effective_tokens: float
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "iterations": self.iterations,
            "effective_tokens": round(self.effective_tokens, 1),
            "error": self.error,
        }


# ── Dry mode (analytical) ───────────────────────────────────────────


def _dry_run_task(task: dict, agent_factory: Callable, *, agent_id: str) -> TaskResult:
    """Don't actually execute — just compute the modeled token cost
    given the task's expected iterations + loads."""
    from app.tool_runtime.measure import (
        _build_loadable_and_extract,
        compare_stock_vs_loadable,
        extract_prompt_shape,
    )

    iters = max(1, int(task.get("expected_iterations", 1)))
    loads = max(0, int(task.get("expected_loads", 0)))
    is_loadable = "loadable" in agent_id.lower()

    # Use the Phase 1c model with task-specific iter+load counts.
    report = compare_stock_vs_loadable(
        num_iterations=iters,
        num_mid_task_loads=min(loads, iters - 1),
    )
    tokens = report["loadable_total"] if is_loadable else report["stock_total"]
    return TaskResult(
        name=task["name"],
        success=True,
        iterations=iters,
        effective_tokens=tokens,
    )


# ── Live mode ───────────────────────────────────────────────────────


def _live_run_task(task: dict, agent_factory: Callable, *, agent_id: str) -> TaskResult:
    """Real LLM call. Wraps any exception → success=False so the
    panel keeps running across tasks."""
    from crewai import Task
    try:
        agent = agent_factory()
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            name=task["name"], success=False, iterations=0,
            effective_tokens=0.0, error=f"agent build: {exc}",
        )

    crew_task = Task(
        description=task["description"],
        expected_output="A concise response, ≤300 words.",
        agent=agent,
    )
    try:
        agent.execute_task(crew_task)
    except Exception as exc:  # noqa: BLE001
        return TaskResult(
            name=task["name"], success=False, iterations=0,
            effective_tokens=0.0, error=f"execute: {exc}",
        )

    # Pull telemetry for this agent's last call set.
    from app.tool_runtime.telemetry import analyze_telemetry
    summary = analyze_telemetry(agent_id=agent_id)
    return TaskResult(
        name=task["name"],
        success=True,
        iterations=summary.get("calls", 0),
        effective_tokens=summary.get("effective_input_tokens", 0.0),
    )


# ── Public API ──────────────────────────────────────────────────────


def run_parity_panel(
    *,
    panel: list[dict] | None = None,
    mode: str = "dry",
    runs: int = 1,
) -> dict[str, Any]:
    """Run the panel on stock + LoadableAgent. Returns a comparison.

    Args:
        panel: Task list. Each entry is {name, description,
            expected_iterations, expected_loads}. Defaults to
            ``DEFAULT_PANEL``.
        mode: "dry" (default) or "live".
        runs: Number of times to repeat each task. Live-mode default
            of 1 keeps API cost minimal; bump for statistical
            confidence in cache-hit rates.

    Returns:
        Report dict with per-agent totals + the headline ratio.
    """
    panel = panel or DEFAULT_PANEL

    def stock_factory():
        # Force the legacy path explicitly.
        import os
        prior = os.environ.get("LOADABLE_AGENT_EXPERIMENTAL")
        os.environ["LOADABLE_AGENT_EXPERIMENTAL"] = "0"
        try:
            from app.agents.introspector import create_introspector
            return create_introspector()
        finally:
            if prior is None:
                os.environ.pop("LOADABLE_AGENT_EXPERIMENTAL", None)
            else:
                os.environ["LOADABLE_AGENT_EXPERIMENTAL"] = prior

    def loadable_factory():
        import os
        os.environ["LOADABLE_AGENT_EXPERIMENTAL"] = "1"
        from app.agents.introspector import create_introspector
        return create_introspector()

    runner = _live_run_task if mode == "live" else _dry_run_task

    stock_results: list[TaskResult] = []
    loadable_results: list[TaskResult] = []
    for task in panel:
        for _ in range(runs):
            stock_results.append(runner(task, stock_factory, agent_id="stock_introspector"))
            loadable_results.append(runner(task, loadable_factory, agent_id="loadable_introspector"))

    stock_total = sum(r.effective_tokens for r in stock_results)
    loadable_total = sum(r.effective_tokens for r in loadable_results)
    ratio = loadable_total / max(stock_total, 1.0)

    stock_success_rate = sum(1 for r in stock_results if r.success) / len(stock_results)
    loadable_success_rate = sum(1 for r in loadable_results if r.success) / len(loadable_results)

    return {
        "mode": mode,
        "runs_per_task": runs,
        "tasks": [t["name"] for t in panel],
        "stock_results": [r.to_dict() for r in stock_results],
        "loadable_results": [r.to_dict() for r in loadable_results],
        "stock_total": round(stock_total, 1),
        "loadable_total": round(loadable_total, 1),
        "ratio": round(ratio, 3),
        "stock_success_rate": round(stock_success_rate, 3),
        "loadable_success_rate": round(loadable_success_rate, 3),
        # GO if ratio passes the gate AND parity success is no worse.
        "verdict": (
            "GO" if (ratio <= 0.50 and loadable_success_rate >= stock_success_rate * 0.90)
            else "NO-GO"
        ),
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Phase 2 — parity panel ({report['mode']} mode)",
        "",
        f"Runs per task: {report['runs_per_task']}",
        f"Tasks: {', '.join(report['tasks'])}",
        "",
        "## Per-task results",
        "",
        "| Task | Stock tokens | Loadable tokens | Stock OK | Loadable OK |",
        "|------|------------:|----------------:|:--------:|:-----------:|",
    ]
    by_task: dict[str, list] = {}
    for r in report["stock_results"]:
        by_task.setdefault(r["name"], [None, None])[0] = r
    for r in report["loadable_results"]:
        by_task.setdefault(r["name"], [None, None])[1] = r
    for name, (sr, lr) in by_task.items():
        stock_tokens = sr["effective_tokens"] if sr else 0
        loadable_tokens = lr["effective_tokens"] if lr else 0
        stock_ok = "✓" if sr and sr["success"] else "✗"
        loadable_ok = "✓" if lr and lr["success"] else "✗"
        lines.append(
            f"| {name} | {stock_tokens:,} | {loadable_tokens:,} | "
            f"{stock_ok} | {loadable_ok} |"
        )

    lines += [
        "",
        f"## Totals",
        "",
        f"* Stock total:    {report['stock_total']:,} effective tokens",
        f"* Loadable total: {report['loadable_total']:,} effective tokens",
        f"* Ratio:          {report['ratio']:.1%}",
        f"* Stock success rate: {report['stock_success_rate']:.0%}",
        f"* Loadable success rate: {report['loadable_success_rate']:.0%}",
        f"* **Verdict: {report['verdict']}**",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry", "live"], default="dry")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", action="store_true",
                        help="Emit raw JSON instead of markdown")
    args = parser.parse_args()
    report = run_parity_panel(mode=args.mode, runs=args.runs)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))


if __name__ == "__main__":
    main()
