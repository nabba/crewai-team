"""phase5_check.py — readiness check for Phase 5 cleanup.

Operator-side smoke test that verifies the LoadableAgent path can
construct each migrated agent without error, reports per-agent
readiness, and surfaces any drift between legacy and loadable
toolsets.

This script does NOT make real LLM calls. It exercises the agent
construction paths only — what runs in production when an env flag
is flipped. If this passes for an agent, you have high confidence
that flipping that agent's per-agent default to ON in code would
not introduce a *construction-time* regression.

Live behavior parity (does the agent actually solve tasks correctly?)
is operator-driven via a real task panel — see
``docs/TOOL_REGISTRY_PHASE_5.md``. This CLI is the construction-time
prerequisite for that.

CLI::

    docker exec crewai-team-gateway-1 python -m app.tool_runtime.phase5_check
    # → markdown report

    docker exec crewai-team-gateway-1 python -m app.tool_runtime.phase5_check --json
    # → JSON for programmatic consumption

Returns exit code 0 if ALL migrated agents pass construction +
parity; non-zero (1) otherwise.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentCheck:
    """Result of construction + parity check for one migrated agent."""

    name: str
    legacy_constructed: bool
    legacy_tool_count: int
    legacy_error: str | None
    loadable_constructed: bool
    loadable_tool_count: int
    loadable_error: str | None
    eager_parity: bool
    eager_parity_diff: dict | None  # {missing_in_loadable, extra_in_loadable}
    catalog_populated: bool
    catalog_size: int

    @property
    def ready(self) -> bool:
        """All construction-time checks pass; flag flip would be safe
        from a structural standpoint."""
        return (
            self.legacy_constructed
            and self.loadable_constructed
            and self.eager_parity
            and self.catalog_populated
        )

    def to_dict(self) -> dict:
        return {
            "agent": self.name,
            "ready": self.ready,
            "legacy": {
                "constructed": self.legacy_constructed,
                "tool_count": self.legacy_tool_count,
                "error": self.legacy_error,
            },
            "loadable": {
                "constructed": self.loadable_constructed,
                "tool_count": self.loadable_tool_count,
                "error": self.loadable_error,
                "catalog_size": self.catalog_size,
            },
            "parity": {
                "eager_set_matches": self.eager_parity,
                "diff": self.eager_parity_diff,
            },
        }


def _construct_via_factory(create_fn, **kwargs):
    """Try constructing an agent; capture any exception cleanly."""
    try:
        agent = create_fn(**kwargs)
        return agent, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def check_agent(name: str) -> AgentCheck:
    """Run the readiness check for one migrated agent.

    Builds the agent on both paths, compares eager toolsets, and
    reports a verdict. The flag is flipped temporarily for the
    loadable build, then restored.
    """
    create_fn, build_kwargs = _agent_factory_for(name)

    # Save current env state.
    var_per_agent = f"LOADABLE_{name.upper()}"
    saved_per_agent = os.environ.get(var_per_agent)
    saved_master = os.environ.get("LOADABLE_AGENT_EXPERIMENTAL")

    try:
        # ── legacy path
        os.environ[var_per_agent] = "0"
        os.environ.pop("LOADABLE_AGENT_EXPERIMENTAL", None)
        legacy, legacy_err = _construct_via_factory(create_fn, **build_kwargs)
        legacy_count = len(legacy.tools) if legacy is not None else 0
        legacy_names = {t.name for t in (legacy.tools or [])} if legacy else set()

        # ── loadable path
        os.environ[var_per_agent] = "1"
        loadable, loadable_err = _construct_via_factory(create_fn, **build_kwargs)
        loadable_count = len(loadable.tools) if loadable is not None else 0
        loadable_names = {t.name for t in (loadable.tools or [])} if loadable else set()

        # Catalog (only meaningful for the loadable agent)
        catalog_size = 0
        if loadable is not None and hasattr(loadable, "binder"):
            catalog_size = len(loadable.binder.catalog_names())

    finally:
        # Restore env state regardless of what happened.
        if saved_per_agent is None:
            os.environ.pop(var_per_agent, None)
        else:
            os.environ[var_per_agent] = saved_per_agent
        if saved_master is None:
            os.environ.pop("LOADABLE_AGENT_EXPERIMENTAL", None)
        else:
            os.environ["LOADABLE_AGENT_EXPERIMENTAL"] = saved_master

    # Parity check: loadable should differ from legacy by exactly the
    # 2 binder control tools, no missing legacy tools.
    parity = False
    parity_diff: dict | None = None
    if legacy is not None and loadable is not None:
        added = loadable_names - legacy_names
        missing = legacy_names - loadable_names
        parity = added == {"load_tool", "list_available_tools"} and not missing
        if not parity:
            parity_diff = {
                "added_in_loadable": sorted(added),
                "missing_in_loadable": sorted(missing),
                "expected_added": ["load_tool", "list_available_tools"],
            }

    return AgentCheck(
        name=name,
        legacy_constructed=legacy is not None,
        legacy_tool_count=legacy_count,
        legacy_error=legacy_err,
        loadable_constructed=loadable is not None,
        loadable_tool_count=loadable_count,
        loadable_error=loadable_err,
        eager_parity=parity,
        eager_parity_diff=parity_diff,
        catalog_populated=catalog_size > 0,
        catalog_size=catalog_size,
    )


def _agent_factory_for(name: str):
    """Return ``(factory_callable, build_kwargs)`` for a migrated agent."""
    if name == "introspector":
        from app.agents.introspector import create_introspector
        return create_introspector, {}
    if name == "researcher":
        from app.agents.researcher import create_researcher
        # Test the FULL path (the one that actually migrates) — light
        # path always uses legacy regardless of flag.
        return create_researcher, {"light": False}
    if name == "writer":
        from app.agents.writer import create_writer
        return create_writer, {}
    if name == "coder":
        from app.agents.coder import create_coder
        return create_coder, {}
    raise ValueError(f"unknown agent: {name}")


# ── Entry point ─────────────────────────────────────────────────────


_MIGRATED_AGENTS = ("introspector", "researcher", "writer", "coder")


def run_full_check() -> dict:
    """Run the readiness check for all migrated agents.

    Returns a dict with per-agent results + an aggregate verdict.
    """
    # Boot the registry so capability resolution works in loadable paths.
    try:
        from app.tool_registry.boot import boot_registry
        boot_registry(snapshot_to_postgres=False, index_to_chromadb=False, sync_forge=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("phase5_check: registry boot failed: %s", exc)

    results: list[AgentCheck] = []
    for name in _MIGRATED_AGENTS:
        results.append(check_agent(name))

    all_ready = all(r.ready for r in results)
    return {
        "verdict": "READY" if all_ready else "NOT-READY",
        "agents": [r.to_dict() for r in results],
        "summary": {
            "total": len(results),
            "ready": sum(1 for r in results if r.ready),
            "not_ready": sum(1 for r in results if not r.ready),
        },
    }


def render_report(report: dict) -> str:
    lines = [
        "# Phase 5 readiness check",
        "",
        f"**Verdict: {report['verdict']}**",
        "",
        "## Per-agent results",
        "",
        "| Agent | Ready | Legacy | Loadable | Eager parity | Catalog |",
        "|-------|:-----:|:------:|:--------:|:------------:|--------:|",
    ]
    for entry in report["agents"]:
        ready = "✓" if entry["ready"] else "✗"
        legacy_ok = "✓" if entry["legacy"]["constructed"] else "✗"
        loadable_ok = "✓" if entry["loadable"]["constructed"] else "✗"
        parity_ok = "✓" if entry["parity"]["eager_set_matches"] else "✗"
        catalog = entry["loadable"]["catalog_size"]
        lines.append(
            f"| {entry['agent']} | {ready} | "
            f"{legacy_ok} ({entry['legacy']['tool_count']}) | "
            f"{loadable_ok} ({entry['loadable']['tool_count']}) | "
            f"{parity_ok} | {catalog} |"
        )
    lines += [
        "",
        f"## Summary",
        "",
        f"* {report['summary']['ready']} of {report['summary']['total']} agents READY",
        f"* {report['summary']['not_ready']} NOT-READY",
        "",
    ]
    # Detail any failures.
    for entry in report["agents"]:
        if entry["ready"]:
            continue
        lines.append(f"### Issue: {entry['agent']}")
        lines.append("")
        if not entry["legacy"]["constructed"]:
            lines.append(f"  - Legacy construction failed: `{entry['legacy']['error']}`")
        if not entry["loadable"]["constructed"]:
            lines.append(f"  - Loadable construction failed: `{entry['loadable']['error']}`")
        if not entry["parity"]["eager_set_matches"] and entry["parity"]["diff"]:
            lines.append("  - Eager toolset parity mismatch:")
            diff = entry["parity"]["diff"]
            if diff.get("missing_in_loadable"):
                lines.append(
                    f"    - Missing in loadable: {diff['missing_in_loadable']}"
                )
            if diff.get("added_in_loadable") and \
               set(diff["added_in_loadable"]) != {"load_tool", "list_available_tools"}:
                lines.append(
                    f"    - Unexpected in loadable: "
                    f"{set(diff['added_in_loadable']) - {'load_tool', 'list_available_tools'}}"
                )
        if entry["loadable"]["catalog_size"] == 0:
            lines.append("  - Discoverable catalog is empty — no capabilities resolved.")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 readiness check.")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of a markdown report.")
    args = parser.parse_args()

    report = run_full_check()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_report(report))
    return 0 if report["verdict"] == "READY" else 1


if __name__ == "__main__":
    sys.exit(main())
