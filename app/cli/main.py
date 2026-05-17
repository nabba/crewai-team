"""Argparse dispatcher for ``python -m app.cli``.

Layout: each subcommand is wired to a ``cmd_*`` function in :mod:`app.cli.commands`.
Global flags (``--endpoint``, ``--bearer``, ``--json``, ``--quiet``) are shared
via ``parents=`` so they're accepted at every level (``aai --json status`` and
``aai status --json`` both parse).

Exit code contract:
    0  ok
    1  user error (bad args / not found)
    2  transport / auth error
    3  gateway returned non-2xx
    130 keyboard interrupt

Pass-through subcommands (``brainstorm`` / ``drill`` / ``bootstrap`` / ``advisory``)
accept any tail args via ``argparse.REMAINDER`` and forward to the existing
``python -m <module>`` entry. Global flags on those must precede the verb.
"""
from __future__ import annotations

import argparse
import sys

from app.cli import commands


def _common_parent() -> argparse.ArgumentParser:
    # ``default=argparse.SUPPRESS`` is load-bearing: parents= subparsers
    # otherwise OVERWRITE namespace values with their own defaults, so a flag
    # passed before the verb would be reset by the leaf parser's re-parse.
    # SUPPRESS means the attribute is only set when the user actually passes
    # the flag — preserved across parent → leaf transitions.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--endpoint",
        default=argparse.SUPPRESS,
        help="Endpoint URL or named alias (local|tailnet|funnel). "
             "Default: $AAI_ENDPOINT or local.",
    )
    p.add_argument(
        "--bearer",
        default=argparse.SUPPRESS,
        help="Bearer token. Default: $AAI_BEARER or $GATEWAY_SECRET.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit JSON.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Suppress informational output; errors still go to stderr.",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    common = _common_parent()
    parser = argparse.ArgumentParser(
        prog="aai",
        description="Operator CLI for Andrus AI. Recovery / consolidation / scripting.",
        parents=[common],
    )
    subs = parser.add_subparsers(dest="verb", required=True, metavar="<verb>")

    def add(name: str, **kwargs):
        kwargs.setdefault("parents", [common])
        return subs.add_parser(name, **kwargs)

    def sub_with_common(p_obj):
        """Return an add_subparsers handle that inherits the common parent."""
        nested = p_obj.add_subparsers(dest="subverb", required=True)
        original_add = nested.add_parser

        def wrapped(name: str, **kwargs):
            kwargs.setdefault("parents", [common])
            return original_add(name, **kwargs)

        nested.add_parser = wrapped  # type: ignore[method-assign]
        return nested

    # ---- status ----------------------------------------------------------- #
    p = add("status", help="System status (mirrors /cp/monitor).")
    p.set_defaults(func=commands.cmd_status)

    # ---- healing run <monitor> -------------------------------------------- #
    p = add("healing", help="Healing monitor controls.")
    hs = sub_with_common(p)
    h_run = hs.add_parser("run", help="Force-probe a single monitor.")
    h_run.add_argument("monitor", help="Module name under app.healing.monitors.")
    h_run.set_defaults(func=commands.cmd_healing_run)

    # ---- logs tail -------------------------------------------------------- #
    p = add("logs", help="Log inspection.")
    ls = sub_with_common(p)
    l_tail = ls.add_parser("tail", help="Tail recent log lines.")
    l_tail.add_argument("-n", "--lines", type=int, default=50)
    l_tail.add_argument("--path", help="Override target log path.")
    l_tail.set_defaults(func=commands.cmd_logs_tail)

    # ---- recall ----------------------------------------------------------- #
    p = add("recall", help="Search conversation memory.")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--window-months", type=int, default=24)
    p.set_defaults(func=commands.cmd_recall)

    # ---- briefing --------------------------------------------------------- #
    p = add("briefing", help="Compose a daily briefing.")
    p.add_argument(
        "flavour",
        choices=["morning", "evening", "weekly"],
        nargs="?",
        default="morning",
    )
    p.set_defaults(func=commands.cmd_briefing)

    # ---- ledger ----------------------------------------------------------- #
    p = add("ledger", help="Identity continuity ledger.")
    ls2 = sub_with_common(p)
    l_tail = ls2.add_parser("tail", help="Print the most recent events.")
    l_tail.add_argument("-n", "--lines", type=int, default=50)
    l_tail.add_argument("--kind", action="append",
                        help="Filter by event kind (repeatable).")
    l_tail.set_defaults(func=commands.cmd_ledger_tail)

    # ---- threads ---------------------------------------------------------- #
    p = add("threads", help="Long-horizon question threads.")
    ts = sub_with_common(p)
    t_list = ts.add_parser("list", help="List threads.")
    t_list.add_argument("--status", help="Filter by status (open|in_progress|blocked|resolved|abandoned).")
    t_list.set_defaults(func=commands.cmd_threads_list)
    t_show = ts.add_parser("show", help="Show one thread by id (8-char prefix ok).")
    t_show.add_argument("thread_id")
    t_show.set_defaults(func=commands.cmd_threads_show)

    # ---- files ------------------------------------------------------------ #
    p = add("files", help="Operator-visible files.")
    fs = sub_with_common(p)
    f_list = fs.add_parser("list", help="List files under output / skills / notes.")
    f_list.add_argument("--root", choices=["output", "skills", "notes"])
    f_list.set_defaults(func=commands.cmd_files_list)
    f_send = fs.add_parser("send", help="Send a file via signal|email|discord.")
    f_send.add_argument("path")
    f_send.add_argument("--via", choices=["signal", "email", "discord"], required=True)
    f_send.add_argument("--body", help="Optional message body to accompany the file.")
    f_send.set_defaults(func=commands.cmd_files_send)

    # ---- notes ------------------------------------------------------------ #
    p = add("notes", help="Notes shortcuts.")
    ns = sub_with_common(p)
    n_save = ns.add_parser("save", help="Save a note. Body from --body or stdin.")
    n_save.add_argument("title", nargs="?", help="Filename (without .md). Default: 'note'.")
    n_save.add_argument("--body", help="Body text. Default: read stdin.")
    n_save.add_argument("--overwrite", action="store_true")
    n_save.set_defaults(func=commands.cmd_notes_save)

    # ---- skills ----------------------------------------------------------- #
    p = add("skills", help="Skill registry read-side.")
    ss = sub_with_common(p)
    s_list = ss.add_parser("list", help="List registered skills.")
    s_list.set_defaults(func=commands.cmd_skills_list)
    s_show = ss.add_parser("show", help="Show one skill by name.")
    s_show.add_argument("name")
    s_show.set_defaults(func=commands.cmd_skills_show)

    # ---- cr (changes) ----------------------------------------------------- #
    p = add("cr", help="Change-request read-side.")
    cs = sub_with_common(p)
    c_list = cs.add_parser("list", help="List change requests.")
    c_list.add_argument("--state", help="Filter by state (pending|approved|applied|rejected|...).")
    c_list.set_defaults(func=commands.cmd_changes_list)
    c_show = cs.add_parser("show", help="Show one CR with diff.")
    c_show.add_argument("request_id")
    c_show.set_defaults(func=commands.cmd_changes_show)

    # ---- amendments (Tier-3) --------------------------------------------- #
    p = add("amendments", help="Tier-3 amendment read-side.")
    asub = sub_with_common(p)
    a_list = asub.add_parser("list", help="List Tier-3 amendments.")
    a_list.set_defaults(func=commands.cmd_amendments_list)

    # ---- wrappers (passthrough — globals must precede the verb) ---------- #
    for verb, fn, help_text in (
        ("brainstorm", commands.cmd_brainstorm,
         "Interactive brainstorming (wraps python -m app.brainstorm)."),
        ("drill", commands.cmd_drill,
         "Resilience drills (wraps python -m app.resilience_drills)."),
    ):
        wp = subs.add_parser(verb, help=help_text)
        wp.add_argument("passthrough", nargs=argparse.REMAINDER,
                        help="Arguments forwarded verbatim.")
        wp.set_defaults(func=fn)

    bp = subs.add_parser(
        "bootstrap",
        help="Bootstrap a subsystem (google|web-push|browse|warm-spare).",
    )
    bp.add_argument("target", choices=["google", "web-push", "browse", "warm-spare"])
    bp.add_argument("passthrough", nargs=argparse.REMAINDER)
    bp.set_defaults(func=commands.cmd_bootstrap)

    ap = subs.add_parser("advisory", help="Observability advisory reports.")
    asub2 = ap.add_subparsers(dest="subverb", required=True)
    g = asub2.add_parser("goodhart",
                         help="Goodhart-gate advisory report (wraps python -m).")
    g.add_argument("passthrough", nargs=argparse.REMAINDER)
    g.set_defaults(func=commands.cmd_advisory_goodhart)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    try:
        return int(func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
