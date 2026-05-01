"""CLI entry point for the autotuner.

Usage:

  python -m app.epistemic                    # default: 7-day analysis
  python -m app.epistemic --window-days 14   # custom window
  python -m app.epistemic --no-persist       # dry-run; print only

Prints a summary table per proposal and the YAML patch text. Persists
to ``epistemic_tuning_proposals`` unless ``--no-persist`` is given.

For accepting a proposal and opening a PR, use the dashboard at
``/epistemic`` or the explicit helpers ``apply_proposal_to_disk`` /
``open_pr_for_proposal`` in :mod:`app.epistemic.autotune`. The CLI
does not automate the apply step — that's by design (humans gate
YAML changes).
"""
from __future__ import annotations

import argparse
import json
import sys

from app.epistemic.autotune import (
    DEFAULT_WINDOW_DAYS,
    ProposalKind,
    run_full_analysis,
)


_KIND_LABELS: dict[ProposalKind, str] = {
    ProposalKind.SEVERITY_DOWNGRADE: "downgrade",
    ProposalKind.SEVERITY_UPGRADE: "upgrade",
    ProposalKind.RETIREMENT_CANDIDATE: "retire",
    ProposalKind.VERIFIER_RETIREMENT: "verifier-retire",
}


def _format_proposal(p) -> str:
    label = _KIND_LABELS.get(p.kind, p.kind.value)
    return (
        f"  [{label}] {p.target_kind}/{p.target_id}  "
        f"conf={p.confidence:.2f}\n"
        f"    {p.rationale}\n"
        f"    metric_evidence={json.dumps(dict(p.metric_evidence), default=str)}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.epistemic",
        description=(
            "Generate tuning proposals for the bias library and "
            "verifier registry. See "
            "crewai-team/docs/SELF_REFLECTION.md §11 for the safety "
            "boundary (proposals only; humans approve via PR)."
        ),
    )
    parser.add_argument(
        "--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
        help=f"observation window (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--no-persist", action="store_true",
        help="dry-run: compute proposals but do not write to the DB",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit the full proposal list as JSON instead of a summary",
    )
    args = parser.parse_args(argv)

    proposals = run_full_analysis(
        window_days=args.window_days,
        persist=not args.no_persist,
    )

    if args.json:
        print(json.dumps(
            [p.as_jsonable() for p in proposals],
            indent=2, default=str,
        ))
        return 0

    if not proposals:
        print(
            f"No tuning proposals over the last {args.window_days} day(s). "
            "Either the system is well-calibrated or there is too "
            "little evidence yet — try again after more activity."
        )
        return 0

    print(
        f"Generated {len(proposals)} tuning proposal(s) over the last "
        f"{args.window_days} day(s):"
    )
    print()
    for p in proposals:
        print(_format_proposal(p))
    print(
        "Open the dashboard at /epistemic to review and accept/reject. "
        "Accepted proposals still require a CODEOWNERS PR."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
