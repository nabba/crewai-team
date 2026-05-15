"""Resilience drills CLI entry point.

PROGRAM §44.4 — Q6.4 P2#9. Lets operators run a drill from a shell
without going through the gateway REST surface. Useful when the
gateway is sick (the very situation a recovery drill might want to
verify) or during operator debugging.

Usage:
    python -m app.resilience_drills list
    python -m app.resilience_drills run <name>
    python -m app.resilience_drills run <name> --dry-run
    python -m app.resilience_drills posture
    python -m app.resilience_drills audit [--limit N] [--drill NAME]

For ``kill_the_gateway`` the CLI runs ONLY the pre-drill check
(same as REST + scheduler). The disruptive LIVE drill remains
gated to ``scripts/drills/kill_the_gateway.sh`` outside the
gateway, with the typed-phrase confirmation.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_list(_args: argparse.Namespace) -> int:
    import app.resilience_drills.drills  # noqa: F401 — populate registry
    from app.resilience_drills.protocol import get_registry, drill_enabled
    from app.resilience_drills.audit import (
        days_since_last_success, last_result_for,
    )
    reg = get_registry()
    rows = []
    for spec in reg.list_specs():
        last = last_result_for(spec.name) or {}
        rows.append({
            "name": spec.name,
            "cadence_days": spec.cadence_days,
            "risk": getattr(spec.risk, "value", str(spec.risk)),
            "enabled": drill_enabled(spec),
            "last_status": last.get("status"),
            "last_run_at": last.get("started_at"),
            "days_since_last_success": days_since_last_success(spec.name),
        })
    print(json.dumps({"drills": rows}, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import app.resilience_drills.drills  # noqa: F401
    from app.resilience_drills.protocol import get_registry, drill_enabled
    reg = get_registry()
    spec = reg.get(args.name)
    if spec is None:
        print(f"unknown drill: {args.name}", file=sys.stderr)
        return 2
    if not drill_enabled(spec):
        print(
            f"drill {args.name} is not enabled (check master switch)",
            file=sys.stderr,
        )
        return 3
    runner = reg.runner_for(args.name)
    if runner is None:
        print(f"no runner for drill {args.name}", file=sys.stderr)
        return 2
    result = runner(dry_run=args.dry_run)
    payload: dict[str, Any] = result.to_dict()
    print(json.dumps(payload, indent=2))
    status = payload.get("status", "")
    return 0 if status == "pass" else 1


def _cmd_posture(_args: argparse.Namespace) -> int:
    from app.resilience_drills.posture import POSTURE
    print(json.dumps({
        "ha_enabled": POSTURE.ha_enabled,
        "rationale_short": POSTURE.rationale_short,
        "target_recovery_minutes": POSTURE.target_recovery_minutes,
        "off_host_targets": list(POSTURE.off_host_targets),
        "target_backup_age_days": POSTURE.target_backup_age_days,
        "quarterly_drills": list(POSTURE.quarterly_drills),
    }, indent=2))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from app.resilience_drills.audit import iter_results
    rows = list(iter_results())
    rows.sort(key=lambda r: r.get("started_at", ""), reverse=True)
    if args.drill:
        rows = [r for r in rows if r.get("drill_name") == args.drill]
    rows = rows[: args.limit]
    print(json.dumps({"results": rows}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.resilience_drills",
        description="Resilience drill CLI (PROGRAM §44).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered drills with last-run state")

    p_run = sub.add_parser("run", help="Run a drill")
    p_run.add_argument("name", help="drill name (e.g. backup_restore)")
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="dry-run mode (default for LOW-risk; required for HIGH-risk)",
    )

    sub.add_parser("posture", help="Show the resilience posture decision")

    p_audit = sub.add_parser("audit", help="Show recent drill audit entries")
    p_audit.add_argument("--limit", type=int, default=20)
    p_audit.add_argument("--drill", default=None, help="filter by drill name")

    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "run":
        return _cmd_run(args)
    if args.cmd == "posture":
        return _cmd_posture(args)
    if args.cmd == "audit":
        return _cmd_audit(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
