"""Subcommand implementations for ``python -m app.cli``.

One function per leaf subcommand, all named ``cmd_<verb>_<subverb>``
where applicable. Each takes a single ``argparse.Namespace`` and returns
an int exit code.

Conventions:

* HTTP-routed commands use :mod:`app.cli.transport` against the gateway.
* In-repo commands use direct imports (only viable when invoked from the
  repo). Import lazily inside the function so a missing module doesn't
  break unrelated subcommands.
* Output mode (``text`` / ``json`` / ``quiet``) is read from ``args.output``.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path
from typing import Any

from app.cli import output, transport
from app.cli.config import resolve

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _cfg(args: argparse.Namespace):
    return resolve(
        endpoint=getattr(args, "endpoint", None),
        bearer=getattr(args, "bearer", None),
    )


def _mode(args: argparse.Namespace) -> str:
    if getattr(args, "quiet", False):
        return "quiet"
    if getattr(args, "json", False):
        return "json"
    return "text"


def _http_or_die(fn, args: argparse.Namespace) -> int:
    """Run an HTTP-bound subcommand, mapping TransportError → exit code."""
    try:
        return fn(args)
    except transport.TransportError as exc:
        sys.stderr.write(f"{exc}\n")
        return exc.exit_code


# --------------------------------------------------------------------------- #
# Recovery / diagnostic
# --------------------------------------------------------------------------- #


def cmd_status(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        payload = transport.get(cfg, "/api/cp/system-status", timeout=15.0)
        mode = _mode(a)

        def render(p):
            lines = [f"endpoint: {cfg.endpoint}"]
            if isinstance(p, dict):
                for section, items in p.items():
                    if isinstance(items, list):
                        lines.append(f"\n[{section}]")
                        for item in items:
                            if isinstance(item, dict):
                                name = item.get("name") or item.get("label") or "?"
                                st = item.get("status") or item.get("state") or "?"
                                lines.append(f"  {name}: {st}")
                            else:
                                lines.append(f"  {item}")
                    else:
                        lines.append(f"{section}: {items}")
            else:
                lines.append(str(p))
            return "\n".join(lines)

        output.render(payload, mode=mode, text_renderer=render)
        return 0

    return _http_or_die(_run, args)


def cmd_healing_run(args: argparse.Namespace) -> int:
    """Force-probe a single healing monitor by module name.

    In-repo only — imports ``app.healing.monitors.<name>`` and calls ``run()``.
    """
    try:
        import importlib
        mod = importlib.import_module(f"app.healing.monitors.{args.monitor}")
    except ImportError as exc:
        return output.die(f"unknown monitor: {args.monitor} ({exc})", code=1)
    if not hasattr(mod, "run"):
        return output.die(f"monitor {args.monitor} has no run() entry point", code=1)
    try:
        mod.run()
    except Exception as exc:  # pragma: no cover — surface ad-hoc invocation errors
        return output.die(f"monitor {args.monitor} raised: {exc}", code=1)
    if _mode(args) == "json":
        output.render({"monitor": args.monitor, "status": "ran"}, mode="json")
    elif _mode(args) != "quiet":
        sys.stdout.write(f"{args.monitor}: ran\n")
    return 0


def cmd_logs_tail(args: argparse.Namespace) -> int:
    """Tail the last N lines from ``workspace/errors.jsonl`` (or another path).

    Falls back to ``audit.log`` if errors.jsonl is absent. Both are in
    workspace, so this is in-repo only.
    """
    workspace = Path(os.environ.get("WORKSPACE_ROOT") or "workspace")
    candidates = [workspace / "errors.jsonl", workspace / "audit.log"]
    if args.path:
        candidates = [Path(args.path)]
    target = next((c for c in candidates if c.exists()), None)
    if target is None:
        return output.die(f"no log file found (tried {', '.join(map(str, candidates))})", code=1)

    n = args.lines
    try:
        with target.open("r", encoding="utf-8", errors="replace") as fh:
            tail = fh.readlines()[-n:]
    except OSError as exc:
        return output.die(f"cannot read {target}: {exc}", code=1)

    if _mode(args) == "json":
        # Best-effort JSONL parse; non-JSON lines pass through as strings.
        import json as _json
        parsed = []
        for line in tail:
            line = line.rstrip("\n")
            try:
                parsed.append(_json.loads(line))
            except _json.JSONDecodeError:
                parsed.append({"raw": line})
        output.render({"path": str(target), "lines": parsed}, mode="json")
    else:
        for line in tail:
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
    return 0


# --------------------------------------------------------------------------- #
# Recall / inspection (in-repo + HTTP mix)
# --------------------------------------------------------------------------- #


def cmd_recall(args: argparse.Namespace) -> int:
    try:
        from app.conversation_memory.retrieval import recall
    except ImportError as exc:
        return output.die(f"conversation_memory unavailable: {exc}", code=1)

    results = recall(args.query, top_k=args.top_k, window_months=args.window_months)

    def render(items):
        if not items:
            return "(no matches)"
        lines = []
        for r in items:
            ts = getattr(r, "ts_iso", None) or getattr(r, "timestamp", "?")
            text = getattr(r, "text", "") or getattr(r, "preview", "")
            kind = getattr(r, "kind", "")
            text = text.replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "..."
            tag = f" [{kind}]" if kind else ""
            lines.append(f"{ts}{tag}\n  {text}")
        return "\n\n".join(lines)

    # JSON-render via dataclass __dict__ where available
    if _mode(args) == "json":
        payload = []
        for r in results:
            if hasattr(r, "__dict__"):
                payload.append({k: v for k, v in r.__dict__.items() if not k.startswith("_")})
            else:
                payload.append(str(r))
        output.render(payload, mode="json")
    else:
        output.render(results, mode="text", text_renderer=render)
    return 0


def cmd_briefing(args: argparse.Namespace) -> int:
    try:
        from app.life_companion import daily_briefing
    except ImportError as exc:
        return output.die(f"life_companion unavailable: {exc}", code=1)

    flavour = args.flavour
    composer = {
        "morning": getattr(daily_briefing, "_compose_morning", None),
        "evening": getattr(daily_briefing, "_compose_evening", None),
        "weekly": getattr(daily_briefing, "_compose_weekly", None),
    }.get(flavour)
    if composer is None:
        return output.die(f"unknown briefing flavour: {flavour}", code=1)

    try:
        text, _ts_set = composer()
    except Exception as exc:  # pragma: no cover — composer surface is rich
        return output.die(f"briefing composer failed: {exc}", code=1)

    if _mode(args) == "json":
        output.render({"flavour": flavour, "text": text}, mode="json")
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
    return 0


def cmd_ledger_tail(args: argparse.Namespace) -> int:
    try:
        from app.identity.continuity_ledger import list_events
    except ImportError as exc:
        return output.die(f"continuity_ledger unavailable: {exc}", code=1)

    kinds = set(args.kind) if args.kind else None
    events = list_events(kinds=kinds)
    tail = events[-args.lines:] if args.lines else events

    if _mode(args) == "json":
        payload = [
            {k: v for k, v in (e.__dict__ if hasattr(e, "__dict__") else {}).items()}
            for e in tail
        ]
        output.render(payload, mode="json")
    else:
        for e in tail:
            ts = getattr(e, "ts_iso", None) or getattr(e, "ts", "?")
            kind = getattr(e, "kind", "?")
            summary = getattr(e, "summary", "") or getattr(e, "description", "")
            sys.stdout.write(f"{ts}  {kind:24}  {summary}\n")
    return 0


def cmd_threads_list(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        params = {}
        if a.status:
            params["status"] = a.status
        payload = transport.get(cfg, "/api/cp/threads", params=params)
        threads = payload.get("threads", payload) if isinstance(payload, dict) else payload

        def render(items):
            if not items:
                return "(no threads)"
            rows = []
            for t in items:
                rows.append({
                    "id": (t.get("id") or "")[:8],
                    "status": t.get("status", ""),
                    "title": (t.get("title") or "")[:60],
                })
            return output.table(rows, ["id", "status", "title"])

        output.render(threads, mode=_mode(a), text_renderer=render)
        return 0

    return _http_or_die(_run, args)


def cmd_threads_show(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        payload = transport.get(cfg, f"/api/cp/threads/{a.thread_id}")

        def render(t):
            if not isinstance(t, dict):
                return str(t)
            lines = [
                f"id:     {t.get('id', '')}",
                f"title:  {t.get('title', '')}",
                f"status: {t.get('status', '')}",
            ]
            for key in ("blockers", "unblock_hints", "sub_questions", "notes"):
                items = t.get(key) or []
                if items:
                    lines.append(f"\n{key}:")
                    for item in items:
                        if isinstance(item, dict):
                            lines.append(f"  - {item.get('text') or item}")
                        else:
                            lines.append(f"  - {item}")
            return "\n".join(lines)

        output.render(payload, mode=_mode(a), text_renderer=render)
        return 0

    return _http_or_die(_run, args)


# --------------------------------------------------------------------------- #
# Files / notes / skills
# --------------------------------------------------------------------------- #


def cmd_files_list(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        params = {"root": a.root} if a.root else None
        payload = transport.get(cfg, "/api/cp/files", params=params)

        def render(p):
            if isinstance(p, dict) and "files" in p:
                files = p["files"]
            else:
                files = p if isinstance(p, list) else []
            rows = [
                {
                    "root": f.get("root", ""),
                    "path": f.get("path", "")[:60],
                    "size": f.get("size", ""),
                }
                for f in files
            ]
            return output.table(rows, ["root", "path", "size"]) or "(no files)"

        output.render(payload, mode=_mode(a), text_renderer=render)
        return 0

    return _http_or_die(_run, args)


def cmd_files_send(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        body = {"path": a.path, "via": a.via}
        if a.body:
            body["body"] = a.body
        payload = transport.post(cfg, "/api/cp/files/send", body=body)
        if _mode(a) == "quiet":
            return 0
        output.render(payload, mode=_mode(a),
                      text_renderer=lambda p: f"sent: {p}" if p else "sent")
        return 0

    return _http_or_die(_run, args)


def cmd_notes_save(args: argparse.Namespace) -> int:
    """Drop a note into ``workspace/notes/`` so the files API picks it up.

    Reads body from stdin if no positional ``body`` was given. Title becomes
    the filename (with ``.md`` suffix added if absent).
    """
    workspace = Path(os.environ.get("WORKSPACE_ROOT") or "workspace")
    notes_dir = workspace / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    title = args.title or "note"
    if not title.endswith(".md"):
        title += ".md"
    # Sanitize: only allow safe filename chars
    safe = "".join(c for c in title if c.isalnum() or c in "-_. ").strip()
    if not safe or safe == ".md":
        return output.die("note title must contain at least one safe character", code=1)
    target = notes_dir / safe
    if target.exists() and not args.overwrite:
        return output.die(f"refusing to overwrite {target} (use --overwrite)", code=1)

    if args.body is not None:
        text = args.body
    else:
        text = sys.stdin.read()

    target.write_text(text, encoding="utf-8")
    if _mode(args) == "json":
        output.render({"path": str(target), "bytes": len(text.encode("utf-8"))}, mode="json")
    elif _mode(args) != "quiet":
        sys.stdout.write(f"wrote {target} ({len(text.encode('utf-8'))} bytes)\n")
    return 0


def cmd_skills_list(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        payload = transport.get(cfg, "/api/cp/skills")

        def render(p):
            items = p if isinstance(p, list) else (p.get("skills") if isinstance(p, dict) else [])
            rows = [
                {
                    "name": s.get("name", ""),
                    "uses": s.get("uses", ""),
                    "summary": (s.get("description") or s.get("summary") or "")[:60],
                }
                for s in (items or [])
            ]
            return output.table(rows, ["name", "uses", "summary"]) or "(no skills)"

        output.render(payload, mode=_mode(a), text_renderer=render)
        return 0

    return _http_or_die(_run, args)


def cmd_skills_show(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        payload = transport.get(cfg, f"/api/cp/skills/{a.name}")
        output.render(payload, mode=_mode(a))
        return 0

    return _http_or_die(_run, args)


# --------------------------------------------------------------------------- #
# Governance read-side
# --------------------------------------------------------------------------- #


def cmd_changes_list(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        params = {"state": a.state} if a.state else None
        payload = transport.get(cfg, "/api/cp/changes", params=params)

        def render(p):
            items = p if isinstance(p, list) else (p.get("requests") if isinstance(p, dict) else [])
            rows = [
                {
                    "id": (c.get("id") or "")[:10],
                    "state": c.get("state", ""),
                    "requestor": c.get("requestor", ""),
                    "path": (c.get("path") or "")[:50],
                }
                for c in (items or [])
            ]
            return output.table(rows, ["id", "state", "requestor", "path"]) or "(no change requests)"

        output.render(payload, mode=_mode(a), text_renderer=render)
        return 0

    return _http_or_die(_run, args)


def cmd_changes_show(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        payload = transport.get(cfg, f"/api/cp/changes/{a.request_id}")
        output.render(payload, mode=_mode(a))
        return 0

    return _http_or_die(_run, args)


def cmd_amendments_list(args: argparse.Namespace) -> int:
    def _run(a):
        cfg = _cfg(a)
        # The Tier-3 surface mounted variously; try the known paths and fall through.
        for path in ("/api/cp/amendments", "/api/cp/governance/pending"):
            try:
                payload = transport.get(cfg, path)
                break
            except transport.GatewayError:
                continue
        else:
            raise transport.GatewayError("no amendments endpoint responded")
        output.render(payload, mode=_mode(a))
        return 0

    return _http_or_die(_run, args)


# --------------------------------------------------------------------------- #
# Wrappers for existing python -m entries
# --------------------------------------------------------------------------- #


def _passthrough(module_name: str, args: argparse.Namespace) -> int:
    """Re-invoke ``python -m <module_name>`` with the passthrough args.

    Reuses the current Python interpreter via :func:`runpy.run_module` so we
    don't fork a new process. Replaces ``sys.argv`` for the duration.
    """
    saved = sys.argv
    sys.argv = [module_name, *(args.passthrough or [])]
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0 if exc.code is None else 1
    finally:
        sys.argv = saved
    return 0


def cmd_brainstorm(args: argparse.Namespace) -> int:
    return _passthrough("app.brainstorm", args)


def cmd_drill(args: argparse.Namespace) -> int:
    return _passthrough("app.resilience_drills", args)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    target = args.target
    module = {
        "google": "app.google_workspace.bootstrap",
        "web-push": "app.web_push.bootstrap",
        "browse": "app.browse.host_collector",
        "warm-spare": "app.warm_spare.manifest",
    }.get(target)
    if module is None:
        return output.die(f"unknown bootstrap target: {target}", code=1)
    return _passthrough(module, args)


def cmd_advisory_goodhart(args: argparse.Namespace) -> int:
    return _passthrough("app.observability.goodhart_advisory_report", args)
