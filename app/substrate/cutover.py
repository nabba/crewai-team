"""
cutover — operator-decided commit of a successful cloud migration.

Productization plan WP D Phase 4. Migration (Phase 3) produces a
verified-ready cloud copy alongside the still-running local instance.
Cutover is the separate, operator-decided moment when the cloud copy
becomes canonical and the local instance steps down.

Critically separate from migrate:
  * Migrate validates and provisions but does NOT change which instance
    receives traffic. The operator can keep the cloud copy as a tested
    cold-spare for weeks.
  * Cutover commits the change. Mirrors the warm_spare module's
    operator-driven design — automated split-brain resolution is too
    costly for a single-operator system, so the human is in the loop.

What cutover actually does:
  1. Validates the migration_run_id exists + was a successful live run
  2. Probes the cloud instance health (via kubectl)
  3. Demotes the LOCAL warm_spare state to DEMOTED
  4. Claims canonical on the CLOUD warm_spare state via kubectl exec
  5. Emits ``cloud_migration:cutover_completed`` to identity ledger
  6. Prints the manual checklist for Signal-device handoff

What cutover deliberately does NOT do:
  * Migrate Signal-cli device identity (Signal Protocol assigns
    unique device tokens; impossible to move automatically)
  * Update DNS, Tailscale Funnel, or other routing — operator's call
  * Stop the local gateway process — operator's call

These deliberate omissions are why cutover prints a checklist instead
of fully automating. The two state-machine transitions (DEMOTED on
local, CANONICAL on cloud) are the auditable record of the operator's
decision.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

CloudTarget = Literal["gcp", "aws"]

TYPED_PHRASE_REQUIRED = "CUTOVER TO GCP"
WARM_SPARE_CLAIM_PHRASE = "CLAIM CANONICAL"


# ── Gate result + cutover run dataclasses ───────────────────────────


class CutoverGateFailure(Exception):
    """Raised when a pre-flight gate refuses cutover.

    Distinct from a step failure: gates run before any state-machine
    transition and represent operator-fixable problems (wrong typed
    phrase, missing migration report, stale migration).
    """


@dataclass
class CutoverStep:
    name: str
    status: str   # "ok" | "warn" | "fail" | "skipped"
    detail: str = ""
    duration_s: float = 0.0
    output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CutoverRun:
    """A single cutover attempt. Persisted at
    workspace/migrations/<migration_run_id>/cutover.json."""
    migration_run_id: str
    started_at: str
    target: str
    project_id: str | None
    steps: list[CutoverStep] = field(default_factory=list)
    completed_at: str = ""
    duration_s: float = 0.0
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    succeeded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "migration_run_id": self.migration_run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": round(self.duration_s, 3),
            "target": self.target,
            "project_id": self.project_id,
            "steps": [s.to_dict() for s in self.steps],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "succeeded": self.succeeded,
        }


# ── Pre-flight gates ────────────────────────────────────────────────


def _load_migration_report(run_id: str) -> dict[str, Any] | None:
    """Return the persisted migration report or None if absent/unreadable."""
    try:
        from app.paths import WORKSPACE_ROOT
        report = Path(WORKSPACE_ROOT) / "migrations" / run_id / "report.json"
        if not report.exists():
            return None
        return json.loads(report.read_text())
    except Exception:
        logger.debug("cutover: failed to read migration report", exc_info=True)
        return None


@dataclass(frozen=True)
class CutoverGateResult:
    name: str
    passed: bool
    detail: str


def evaluate_cutover_gates(
    *,
    target: CloudTarget,
    migration_run_id: str,
    confirm_phrase: str,
    max_migration_age_days: float = 30.0,
) -> list[CutoverGateResult]:
    """Run every refuse-on-fail gate before any state-machine change.

    Pure function — no I/O beyond reading the migration report. Caller
    asserts ``all(g.passed)`` before executing the cutover.
    """
    gates: list[CutoverGateResult] = []

    # G1 — typed-phrase confirmation
    expected = TYPED_PHRASE_REQUIRED if target == "gcp" else f"CUTOVER TO {target.upper()}"
    gates.append(CutoverGateResult(
        name="typed_phrase",
        passed=(confirm_phrase == expected),
        detail=(
            "ok" if confirm_phrase == expected
            else f"--confirm must equal exactly {expected!r}"
        ),
    ))

    # G2 — migration run-id must exist
    if not migration_run_id:
        gates.append(CutoverGateResult(
            name="migration_run_id",
            passed=False,
            detail="--to <run-id> required",
        ))
        # Skip the rest — they all read the report
        return gates

    report = _load_migration_report(migration_run_id)
    if report is None:
        gates.append(CutoverGateResult(
            name="migration_run_id",
            passed=False,
            detail=f"no migration report at workspace/migrations/{migration_run_id}/report.json",
        ))
        return gates

    gates.append(CutoverGateResult(
        name="migration_run_id",
        passed=True,
        detail=f"report found for {migration_run_id}",
    ))

    # G3 — must have been a live run (not dry-run)
    is_live = not bool(report.get("dry_run", True))
    gates.append(CutoverGateResult(
        name="migration_was_live",
        passed=is_live,
        detail=(
            "ok (live run)" if is_live
            else "migration was a dry-run — cutover requires a live migrate first"
        ),
    ))

    # G4 — migration must have succeeded
    ready = bool(report.get("ready_for_live", False))
    gates.append(CutoverGateResult(
        name="migration_succeeded",
        passed=ready,
        detail=(
            "ok" if ready
            else "migration was not ready_for_live — cutover refused"
        ),
    ))

    # G5 — migration not stale (default 30 days). A long-stale migration
    # likely means the cluster is gone or out-of-sync with the current
    # workspace state.
    started_at = report.get("started_at", "")
    try:
        started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - started).total_seconds() / 86400
        fresh = age_days <= max_migration_age_days
        gates.append(CutoverGateResult(
            name="migration_freshness",
            passed=fresh,
            detail=(
                f"ok ({age_days:.1f} days old)" if fresh
                else f"migration is {age_days:.1f} days old (> {max_migration_age_days}) — re-run `botarmy migrate --live` first"
            ),
        ))
    except Exception:
        gates.append(CutoverGateResult(
            name="migration_freshness",
            passed=False,
            detail=f"could not parse started_at={started_at!r}",
        ))

    # G6 — target match: report's target must match the requested cutover target
    report_target = report.get("target", "")
    gates.append(CutoverGateResult(
        name="target_match",
        passed=(report_target == target),
        detail=(
            "ok" if report_target == target
            else f"migration was for target={report_target!r}; --to {target} does not match"
        ),
    ))

    return gates


# ── Step helpers ────────────────────────────────────────────────────


def _record(run: CutoverRun, step: CutoverStep) -> None:
    run.steps.append(step)


def _time_step(name: str, fn) -> CutoverStep:
    started = time.monotonic()
    try:
        step = fn()
        if not isinstance(step, CutoverStep):
            step = CutoverStep(name=name, status="fail",
                               detail=f"step returned {type(step).__name__}")
    except Exception as exc:
        logger.exception("cutover step %s crashed", name)
        step = CutoverStep(
            name=name,
            status="fail",
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
    step.duration_s = round(time.monotonic() - started, 3)
    return step


# ── Live-mode subprocess plumbing ───────────────────────────────────
# Cutover shells out to ``kubectl exec`` for the cloud-side state-machine
# update. Same execute-gate pattern as ``migration.py``:
#   - require execute=True arg OR BOTARMY_MIGRATE_LIVE_EXECUTE=1 env
#     (re-uses the migrate env var since cutover is part of the same
#      migration flow)


import os
import subprocess


def _shell(
    argv: list[str], *, timeout: float, execute: bool = False,
) -> tuple[int, str, str]:
    # Single source of truth — see cloud_prep.is_live_execute_enabled.
    # Consults runtime_settings first (React-toggleable) then env var.
    from app.substrate.cloud_prep import is_live_execute_enabled
    if not (execute or is_live_execute_enabled()):
        return 0, f"<dry: {' '.join(argv)}>", ""
    cmd_name = argv[0] if argv else "<empty>"
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd_name}: command not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd_name}: timed out after {timeout}s"
    except Exception as exc:
        return 1, "", f"{cmd_name}: {type(exc).__name__}: {exc}"


# ── Step implementations ────────────────────────────────────────────


def _step_probe_cloud_health() -> CutoverStep:
    """Confirm the cloud gateway pod is responsive before cutover.

    A stale or broken cluster (e.g. cloud was provisioned weeks ago and
    the node group scaled to zero, or the pod was rebooted into a bad
    state) would mean cutover transitions the local instance to DEMOTED
    while the cloud can't actually serve. We probe before demoting.
    """
    namespace = "botarmy"
    probe_cmd = [
        "kubectl", "-n", namespace, "exec", "deploy/gateway", "--",
        "python", "-c",
        "from app.subia.integrity import verify_integrity; "
        "r = verify_integrity(strict=False); "
        "import json; print(json.dumps({'ok': r.ok, 'n_files': r.n_files}))",
    ]
    rc, out, err = _shell(probe_cmd, timeout=120.0)
    if rc != 0:
        return CutoverStep(
            name="cloud_health",
            status="fail",
            detail=f"kubectl exec failed: {(err or out)[:200]}",
            output={"command": " ".join(probe_cmd)},
        )
    # Parse the printed JSON
    try:
        last_line = (out or "").strip().split("\n")[-1]
        parsed = json.loads(last_line)
        if not parsed.get("ok"):
            return CutoverStep(
                name="cloud_health",
                status="warn",
                detail=f"cloud responsive but SubIA integrity drifted: {parsed}",
                output=parsed,
            )
        return CutoverStep(
            name="cloud_health",
            status="ok",
            detail=f"cloud responsive, subia ok ({parsed.get('n_files')} files)",
            output=parsed,
        )
    except (json.JSONDecodeError, IndexError):
        # In dry-shell mode the output is "<dry: ...>" which won't parse.
        # That's a warn, not a fail — operator running with --really-do-it
        # would see real JSON.
        return CutoverStep(
            name="cloud_health",
            status="warn",
            detail=f"probe ran but output unparseable: {out[:120]}",
            output={"stdout_tail": out[-300:] if out else ""},
        )


def _step_demote_local() -> CutoverStep:
    """Move local warm_spare state to DEMOTED.

    No subprocess — this is a direct Python call. The operator's local
    instance reads its own state file and learns it's DEMOTED next time
    anything checks (e.g. the host_substrate_health monitor).
    """
    try:
        from app.warm_spare.failover import demote, current_state
        prior = current_state()
        result = demote()
        return CutoverStep(
            name="demote_local",
            status="ok",
            detail=f"local: {prior.get('state', 'UNKNOWN')} → {result.get('state')}",
            output={"prior": prior, "after": result},
        )
    except Exception as exc:
        return CutoverStep(
            name="demote_local",
            status="fail",
            detail=f"warm_spare.demote() failed: {type(exc).__name__}: {exc}",
        )


def _step_claim_canonical_cloud() -> CutoverStep:
    """Tell the cloud gateway it is now canonical.

    Runs `kubectl exec` to call `claim_canonical(...)` inside the cloud
    pod. Best-effort: if kubectl is unavailable the local DEMOTED state
    still records the operator's intent and the operator can complete
    the cloud-side claim manually.
    """
    namespace = "botarmy"
    claim_cmd = [
        "kubectl", "-n", namespace, "exec", "deploy/gateway", "--",
        "python", "-c",
        f"from app.warm_spare.failover import claim_canonical; "
        f"import json; "
        f"r = claim_canonical({WARM_SPARE_CLAIM_PHRASE!r}); "
        f"print(json.dumps(r))",
    ]
    rc, out, err = _shell(claim_cmd, timeout=60.0)
    if rc != 0:
        # Local demote already happened; mark this as warn (not fail) so
        # the cutover run still completes. Operator finishes manually.
        return CutoverStep(
            name="claim_canonical_cloud",
            status="warn",
            detail=(
                f"kubectl exec exit {rc}: {(err or out)[:120]} — "
                f"run claim_canonical({WARM_SPARE_CLAIM_PHRASE!r}) on the cloud manually"
            ),
            output={"command": " ".join(claim_cmd), "stderr_tail": err[-300:] if err else ""},
        )
    try:
        last_line = (out or "").strip().split("\n")[-1]
        parsed = json.loads(last_line)
        accepted = bool(parsed.get("accepted"))
        return CutoverStep(
            name="claim_canonical_cloud",
            status="ok" if accepted else "warn",
            detail=(
                f"cloud claim accepted → {parsed.get('state')}"
                if accepted else
                f"cloud refused claim: {parsed.get('reason', 'unknown')}"
            ),
            output=parsed,
        )
    except (json.JSONDecodeError, IndexError):
        return CutoverStep(
            name="claim_canonical_cloud",
            status="warn",
            detail=f"claim ran but output unparseable: {out[:120]}",
            output={"stdout_tail": out[-300:] if out else ""},
        )


# ── Roll-up + persistence ───────────────────────────────────────────


def _finalize(run: CutoverRun) -> None:
    for s in run.steps:
        if s.status == "fail":
            run.blockers.append(f"{s.name}: {s.detail}")
        elif s.status == "warn":
            run.warnings.append(f"{s.name}: {s.detail}")
    # Success = no fails. Warnings are tolerated (e.g. cloud claim
    # needs manual finish).
    run.succeeded = not run.blockers


def _write_report(run: CutoverRun) -> Path:
    try:
        from app.paths import WORKSPACE_ROOT
        out_dir = Path(WORKSPACE_ROOT) / "migrations" / run.migration_run_id
    except Exception:
        out_dir = Path("/tmp") / "botarmy_cutover" / run.migration_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "cutover.json"
    report.write_text(
        json.dumps(run.to_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _emit_ledger_event(run: CutoverRun, phase: str) -> None:
    """Record cutover phases on the identity continuity ledger."""
    try:
        from app.identity.continuity_ledger import record_event
        summary = (
            f"cutover {phase} → {run.target} "
            f"(migration_run_id={run.migration_run_id[:12]})"
        )
        record_event(
            kind="cloud_migration",
            actor="botarmy_cutover",
            summary=summary,
            detail={
                "phase": phase,
                "migration_run_id": run.migration_run_id,
                "target": run.target,
                "project_id": run.project_id,
                "succeeded": run.succeeded,
                "n_blockers": len(run.blockers),
                "n_warnings": len(run.warnings),
            },
        )
    except Exception:
        logger.debug("cutover: ledger emit failed (non-fatal)", exc_info=True)


# ── Public entry ────────────────────────────────────────────────────


def run_cutover(
    *,
    migration_run_id: str,
    confirm_phrase: str,
    target: CloudTarget = "gcp",
    project_id: str | None = None,
    execute_subprocess: bool = False,
) -> CutoverRun:
    """Execute the cutover for a verified migration. Operator-driven.

    Gates (CutoverGateFailure raised before any state change):
      * typed phrase ``--confirm "CUTOVER TO GCP"``
      * migration_run_id resolves to an existing report
      * report.dry_run is False (was a live migrate)
      * report.ready_for_live is True (migration succeeded)
      * report not older than 30 days
      * report.target matches the requested target

    Pipeline (halts on first failure):
      1. cloud_health        — kubectl exec probe
      2. demote_local        — warm_spare.demote() locally
      3. claim_canonical_cloud — kubectl exec calls claim_canonical()

    Identity-ledger landmarks: cutover_started, cutover_completed
    (success) OR cutover_failed (with reason).
    """
    # 1. Evaluate gates
    gates = evaluate_cutover_gates(
        target=target,
        migration_run_id=migration_run_id,
        confirm_phrase=confirm_phrase,
    )
    failed = [g for g in gates if not g.passed]
    if failed:
        raise CutoverGateFailure(
            "cutover refused — "
            + "; ".join(f"{g.name}: {g.detail}" for g in failed)
        )

    # 2. Project id resolved from migration report when not supplied
    if not project_id:
        report = _load_migration_report(migration_run_id) or {}
        project_id = report.get("project_id")

    started_mono = time.monotonic()
    run = CutoverRun(
        migration_run_id=migration_run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        target=target,
        project_id=project_id,
    )
    _emit_ledger_event(run, phase="cutover_started")

    # 3. Patch _shell at module level so execute_subprocess propagates
    global _shell
    _orig_shell = _shell

    def _gated_shell(argv, *, timeout, execute=False):
        return _orig_shell(argv, timeout=timeout, execute=execute_subprocess or execute)

    _shell = _gated_shell
    try:
        # Halt-on-fail pipeline (cloud_health failure means we don't
        # demote local; demote_local failure means we don't claim cloud)
        steps_to_run = [
            ("cloud_health", _step_probe_cloud_health),
            ("demote_local", _step_demote_local),
            ("claim_canonical_cloud", _step_claim_canonical_cloud),
        ]
        for name, fn in steps_to_run:
            step = _time_step(name, fn)
            _record(run, step)
            if step.status == "fail":
                break
    finally:
        _shell = _orig_shell

    _finalize(run)
    run.completed_at = datetime.now(timezone.utc).isoformat()
    run.duration_s = round(time.monotonic() - started_mono, 3)

    _write_report(run)
    _emit_ledger_event(
        run,
        phase="cutover_completed" if run.succeeded else "cutover_failed",
    )
    return run


# ── Operator-facing checklist + formatter ───────────────────────────


_POST_CUTOVER_CHECKLIST_GCP = """
After successful cutover, complete the manual Signal-device handoff:

  1. STOP the local gateway: `docker compose stop gateway` (or kill
     the run_host.py process). The host_bridge can remain running —
     it's the signal-cli on the new instance that needs to claim
     the device identity, not the bridge.

  2. RE-REGISTER signal-cli on the cloud instance with your Signal
     phone number. This is a one-time SMS verification:
         kubectl -n botarmy exec deploy/gateway -- \\
             signal-cli -u +<NUMBER> register
         # then enter the SMS code:
         kubectl -n botarmy exec deploy/gateway -- \\
             signal-cli -u +<NUMBER> verify <CODE>

  3. VERIFY the cloud instance now receives Signal messages:
     Send a test message to the AndrusAI Signal number and confirm
     the cloud gateway processes it.

  4. UPDATE Tailscale Funnel / DNS / dashboard URL to point at the
     cloud ingress (if the operator uses these surfaces).

  5. OBSERVE the local instance's host_substrate_health monitor
     will detect the DEMOTED state and emit a substrate_migration
     event on its next probe (~6h cadence). The annual reflection
     auto-surfaces both events.

The local instance now serves as a warm spare. To bring it back to
canonical, run `claim_canonical("CLAIM CANONICAL")` on the LOCAL
instance after the cloud heartbeat goes silent for 15+ minutes.
""".strip()


def format_cutover_run(run: CutoverRun) -> str:
    """Human-readable summary, suitable for the CLI."""
    lines: list[str] = []
    lines.append(
        f"=== Cutover for migration {run.migration_run_id[:12]} — "
        f"{run.target} ==="
    )
    lines.append(f"  started:  {run.started_at}")
    if run.project_id:
        lines.append(f"  project:  {run.project_id}")
    lines.append("")

    glyph = {"ok": "✓", "warn": "⚠", "fail": "✗", "skipped": "·"}
    name_max = max(len(s.name) for s in run.steps) if run.steps else 0
    for s in run.steps:
        lines.append(
            f"  {glyph.get(s.status, '?')} {s.name:<{name_max}}  "
            f"{s.status:<6} {s.detail}  ({s.duration_s:.2f}s)"
        )
    lines.append("")

    if run.succeeded:
        lines.append("  → CUTOVER COMPLETE.")
        lines.append("")
        lines.append(_POST_CUTOVER_CHECKLIST_GCP)
    else:
        lines.append("  → CUTOVER FAILED.")
        for b in run.blockers:
            lines.append(f"     • {b}")
        lines.append(
            "\n     Recovery: local warm_spare state has been preserved if demote\n"
            "     did not run. Inspect workspace/migrations/{}/cutover.json,\n"
            "     resolve the failure, then re-run.".format(run.migration_run_id),
        )

    if run.warnings:
        lines.append("")
        for w in run.warnings:
            lines.append(f"     • warning: {w}")

    return "\n".join(lines)
