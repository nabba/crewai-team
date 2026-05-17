"""Output rendering helpers.

Three modes, set globally on the namespace before subcommand dispatch:

* ``text`` — human-readable, default. Color when isatty.
* ``json`` — stable schema, ``jq``-friendly.
* ``quiet`` — primary value to stdout, errors to stderr. ``aai recall X --quiet | head`` works.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Iterable


def is_tty() -> bool:
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def render(payload: Any, *, mode: str = "text", text_renderer=None) -> None:
    """Print ``payload`` in the chosen mode.

    ``text_renderer`` is a function that takes the payload and returns a
    str (or yields lines). If absent, text mode falls back to pretty-JSON.
    """
    if mode == "json":
        sys.stdout.write(json.dumps(payload, indent=2, default=str))
        sys.stdout.write("\n")
        return
    if text_renderer is None:
        # Fall back to compact JSON-on-one-line for unknown structures.
        if isinstance(payload, (dict, list)):
            sys.stdout.write(json.dumps(payload, default=str))
        else:
            sys.stdout.write(str(payload))
        sys.stdout.write("\n")
        return
    out = text_renderer(payload)
    if isinstance(out, str):
        sys.stdout.write(out)
        if not out.endswith("\n"):
            sys.stdout.write("\n")
    else:
        for line in out:
            sys.stdout.write(line)
            if not line.endswith("\n"):
                sys.stdout.write("\n")


def die(msg: str, *, code: int = 1) -> int:
    sys.stderr.write(msg)
    if not msg.endswith("\n"):
        sys.stderr.write("\n")
    return code


def table(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    """Render an aligned text table. Empty input → empty string."""
    rows = list(rows)
    if not rows:
        return ""
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join(
        "  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns)
        for row in rows
    )
    return f"{header}\n{sep}\n{body}"
