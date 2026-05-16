"""destructive_advisory — discipline for monitors that recommend destructive actions.

PROGRAM (2026-05-16 post-incident hardening).

Two incidents on 2026-05-16 followed the same pattern: a healing monitor
emitted an alert recommending a destructive action against load-bearing
data, the operator acted on it, and only an ad-hoc snapshot saved the
data:

  * migration_drill alert pointed at the drill shell script, which had a
    bind-mount race that corrupted the live postgres database.
  * chromadb_hygiene alert pointed at deleting "orphan segment" dirs,
    but the monitor's classification was wrong (queried `collections.id`
    instead of `segments.id`); 43 live segment dirs across 5 KBs were
    deleted and only the pre-delete tar snapshot saved the data.

Both incidents share a structural failure: there's no enforced
discipline that a destructive recommendation must (a) snapshot
beforehand, (b) tell the operator where the snapshot lives, (c) include
a schema verification step the operator can run BEFORE acting, and
(d) include the explicit undo command.

This module is the guardrail. A monitor whose alert recommends a
destructive action constructs a ``DestructiveAdvisory`` instance and
calls :func:`emit`. The dataclass refuses construction if any of the
four discipline fields is missing or if the snapshot file is not yet
on disk — so a monitor author cannot accidentally ship an unsafe
advisory that compiles and runs.

The helper does NOT auto-apply anything. It does not even invoke the
snapshot itself — the monitor takes the snapshot first (because only
the monitor knows what to snapshot), and passes the result to the
advisory. The advisory's role is purely to enforce the discipline of
the alert message itself.

Usage::

    from app.healing.destructive_advisory import (
        DestructiveAdvisory, snapshot_paths, emit,
    )

    targets = [Path("workspace/memory/orphan-uuid")]
    snap = snapshot_paths(
        targets,
        dest_dir=Path("workspace/.snapshots"),
        label="chromadb_orphans",
    )
    if snap is None:
        # Snapshot failed — DO NOT proceed with the alert. The class
        # would refuse to construct anyway.
        return

    adv = DestructiveAdvisory(
        monitor_name="chromadb_hygiene",
        summary="ChromaDB orphan segments — 8 dirs, 22.2 MB",
        targets=targets,
        snapshot_path=snap,
        apply_command=(
            "docker compose stop chromadb && "
            "rm -rf <targets> && "
            "docker compose start chromadb"
        ),
        undo_command=f"tar -xzf {snap} -C .",
        verify_command=(
            "python3 -c \"import sqlite3; "
            "conn=sqlite3.connect('workspace/memory/chroma.sqlite3'); "
            "print(set(r[0] for r in conn.execute('SELECT id FROM segments')))\""
        ),
        schema_assumption=(
            "Orphan = on-disk UUID dir not referenced by `segments.id` "
            "(NOT `collections.id` — see 2026-05-16 incident)."
        ),
    )
    emit(adv)
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Snapshot helper
# ─────────────────────────────────────────────────────────────────────────


def snapshot_paths(
    targets: Sequence[Path],
    dest_dir: Path,
    label: str,
    *,
    now: Optional[float] = None,
) -> Optional[Path]:
    """Tar ``targets`` into ``dest_dir/{label}_<ts>.tar.gz``.

    Returns the snapshot path on success, or ``None`` on failure. Callers
    MUST NOT proceed to emit a destructive advisory if this returns
    ``None`` — there's no recovery path without the snapshot.

    The label is sanitised (alphanumeric + underscore only) so a
    monitor author can't accidentally create a file in an unintended
    location.

    Paths inside ``targets`` are passed verbatim to tar; the caller is
    responsible for ensuring they're inside the workspace.
    """
    if not targets:
        logger.warning("snapshot_paths: empty target list, refusing to snapshot")
        return None

    # Sanitise label to avoid traversal / odd filenames.
    safe_label = "".join(
        c if c.isalnum() or c == "_" else "_" for c in (label or "snapshot")
    )[:64]
    if not safe_label:
        safe_label = "snapshot"

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now if now is not None else time.time()))
    dest_dir.mkdir(parents=True, exist_ok=True)
    snap_path = dest_dir / f"{safe_label}_{ts}.tar.gz"

    # Verify every target exists before invoking tar — tar would fail
    # noisily and partially anyway, but explicit pre-check produces
    # clearer logs.
    missing = [str(t) for t in targets if not Path(t).exists()]
    if missing:
        logger.warning(
            "snapshot_paths: %d target(s) missing, refusing to snapshot: %s",
            len(missing), missing[:5],
        )
        return None

    # Use a file-list to handle long arg lists + paths with spaces safely.
    list_file = snap_path.with_suffix(".tar.gz.filelist.tmp")
    try:
        list_file.write_text(
            "\n".join(str(Path(t)) for t in targets),
            encoding="utf-8",
        )
        rc = subprocess.call(
            ["tar", "-czf", str(snap_path), "-T", str(list_file)],
        )
        if rc != 0:
            logger.error(
                "snapshot_paths: tar exited %d for label=%s", rc, safe_label,
            )
            if snap_path.exists():
                try:
                    snap_path.unlink()
                except OSError:
                    pass
            return None
    except Exception:
        logger.exception("snapshot_paths: unexpected failure")
        return None
    finally:
        try:
            list_file.unlink()
        except OSError:
            pass

    # Sanity: a non-empty archive should be at least a few hundred bytes
    # even for empty inputs (tar headers).
    try:
        size = snap_path.stat().st_size
        if size < 100:
            logger.warning(
                "snapshot_paths: produced suspiciously small snapshot (%d bytes); "
                "treating as failure",
                size,
            )
            snap_path.unlink()
            return None
    except OSError:
        return None

    return snap_path


# ─────────────────────────────────────────────────────────────────────────
# Advisory dataclass
# ─────────────────────────────────────────────────────────────────────────


_REQUIRED_NON_EMPTY_FIELDS = (
    "monitor_name",
    "summary",
    "apply_command",
    "undo_command",
    "verify_command",
    "schema_assumption",
)


@dataclass(frozen=True)
class DestructiveAdvisory:
    """A recommendation to the operator to take a destructive action.

    Construction validates the four discipline fields (post-2026-05-16):

      * ``snapshot_path`` — the pre-action snapshot MUST already be on
        disk. Pass the return value of :func:`snapshot_paths`.
      * ``verify_command`` — a concrete shell snippet the operator can
        run BEFORE acting to validate the monitor's classification
        against the current state of the world.
      * ``undo_command`` — a concrete shell snippet to reverse the
        action if it produces unexpected results.
      * ``schema_assumption`` — one line declaring what assumption the
        classification depends on, so the next maintainer can audit it.

    A ``ValueError`` at construction time is the discipline boundary:
    a malformed advisory can't be emitted by accident.
    """

    monitor_name: str
    summary: str
    targets: Sequence[Path]
    snapshot_path: Path
    apply_command: str
    undo_command: str
    verify_command: str
    schema_assumption: str

    def __post_init__(self) -> None:
        for fname in _REQUIRED_NON_EMPTY_FIELDS:
            v = getattr(self, fname)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    f"DestructiveAdvisory: required field {fname!r} is empty"
                )
        if not self.targets:
            raise ValueError(
                "DestructiveAdvisory: targets must contain at least one path"
            )
        snap = Path(self.snapshot_path)
        if not snap.is_file():
            raise ValueError(
                f"DestructiveAdvisory: snapshot_path {snap} does not exist. "
                "Take the snapshot BEFORE constructing the advisory."
            )
        if snap.stat().st_size < 100:
            raise ValueError(
                f"DestructiveAdvisory: snapshot at {snap} is suspiciously "
                f"small ({snap.stat().st_size} bytes). Refusing — there's "
                f"no recovery path."
            )

    def format(self) -> str:
        """Format the advisory into a Signal-alert body with the
        discipline visible at every step. Intentionally verbose: the
        operator should pause and read it."""
        n_targets = len(self.targets)
        target_preview = ", ".join(str(p) for p in list(self.targets)[:3])
        if n_targets > 3:
            target_preview += f", … (+{n_targets - 3} more)"
        return (
            f"🛑 {self.summary}\n"
            f"\n"
            f"This advisory recommends a destructive action. Before "
            f"acting, run the schema check below — the monitor's "
            f"classification depends on an assumption that may not hold.\n"
            f"\n"
            f"Schema check (validate FIRST):\n"
            f"  {self.verify_command}\n"
            f"\n"
            f"If the check confirms the issue, apply:\n"
            f"  {self.apply_command}\n"
            f"\n"
            f"If the action produces unexpected results, undo:\n"
            f"  {self.undo_command}\n"
            f"\n"
            f"Snapshot already taken (before this alert was sent):\n"
            f"  {self.snapshot_path}\n"
            f"\n"
            f"Targets ({n_targets}): {target_preview}\n"
            f"Classification assumption: {self.schema_assumption}"
        )


# ─────────────────────────────────────────────────────────────────────────
# Emission helper
# ─────────────────────────────────────────────────────────────────────────


def emit(
    advisory: DestructiveAdvisory,
    *,
    send_fn: Optional[Callable[..., None]] = None,
    audit_fn: Optional[Callable[..., None]] = None,
    tag: Optional[str] = None,
) -> None:
    """Emit a destructive advisory via the Signal alert path.

    By default uses the canonical ``send_signal_alert`` + ``audit_event``
    helpers from ``app.healing.handlers._common``. Injectable for tests.

    The tag defaults to ``destructive_advisory:<monitor_name>`` so the
    arbiter can dedup repeat advisories.
    """
    body = advisory.format()
    final_tag = tag or f"destructive_advisory:{advisory.monitor_name}"

    if send_fn is None:
        try:
            from app.healing.handlers._common import send_signal_alert as _send
            send_fn = _send
        except Exception:
            logger.exception("destructive_advisory: send_signal_alert unavailable")
            send_fn = None

    if audit_fn is None:
        try:
            from app.healing.handlers._common import audit_event as _audit
            audit_fn = _audit
        except Exception:
            audit_fn = None

    if send_fn is not None:
        try:
            send_fn(body, tag=final_tag)
        except Exception:
            logger.exception(
                "destructive_advisory: send_fn raised for monitor=%s",
                advisory.monitor_name,
            )

    if audit_fn is not None:
        try:
            audit_fn(
                "destructive_advisory_emitted",
                monitor=advisory.monitor_name,
                summary=advisory.summary,
                n_targets=len(advisory.targets),
                snapshot_path=str(advisory.snapshot_path),
                tag=final_tag,
            )
        except Exception:
            logger.exception(
                "destructive_advisory: audit_fn raised for monitor=%s",
                advisory.monitor_name,
            )
