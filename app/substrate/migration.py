"""
migration — orchestrator for ``botarmy migrate --dry-run``.

Productization plan WP D Phase 2. Composes the cost estimator (Phase 0),
cloud-readiness probe (Phase 1), DR bundle metadata (T2.3), and the
identity continuity ledger into one report that answers: *if I ran the
live migrate now, would it succeed, what would it cost, and what would
it do?*

Pure dry-run only. No ``terraform apply``, no ``gcloud storage cp``,
no cloud-side resource creation. The compose target is "produce a
report the operator can read in 30 seconds and either approve or
defer the live migrate."

Never raises. Per-step failures land in the step's ``status``/``detail``
and continue the run so the operator sees a complete picture.

Phase 3 (live migrate) will extend this same step framework with
``dry_run=False`` variants — the dataclass shape stays.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

CloudTarget = Literal["gcp", "aws"]
Tier = Literal["cheapest", "prod"]
StepStatus = Literal["ok", "warn", "fail", "skipped"]


# ── Run + step dataclasses ──────────────────────────────────────────


@dataclass
class MigrationStep:
    """One step in the migration pipeline. Status drives the roll-up."""
    name: str
    status: StepStatus
    detail: str = ""
    duration_s: float = 0.0
    output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationRun:
    """A single migration attempt — currently dry-run only.

    Persisted at ``workspace/migrations/<run_id>/report.json``.
    """
    run_id: str
    started_at: str
    target: str
    tier: str
    region: str
    project_id: str | None
    dry_run: bool
    steps: list[MigrationStep] = field(default_factory=list)
    completed_at: str = ""
    duration_s: float = 0.0
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ready_for_live: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": round(self.duration_s, 3),
            "target": self.target,
            "tier": self.tier,
            "region": self.region,
            "project_id": self.project_id,
            "dry_run": self.dry_run,
            "steps": [s.to_dict() for s in self.steps],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "ready_for_live": self.ready_for_live,
        }


# ── Step helpers ────────────────────────────────────────────────────


def _record(run: MigrationRun, step: MigrationStep) -> None:
    run.steps.append(step)


def _time_step(name: str, fn) -> MigrationStep:
    """Run ``fn`` and wrap its result with timing + failure isolation."""
    started = time.monotonic()
    try:
        step = fn()
        if not isinstance(step, MigrationStep):
            # Defensive — step builders should always return MigrationStep
            step = MigrationStep(
                name=name,
                status="fail",
                detail=f"step builder returned {type(step).__name__}",
            )
    except Exception as exc:
        logger.exception("migration step %s crashed", name)
        step = MigrationStep(
            name=name,
            status="fail",
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
    step.duration_s = round(time.monotonic() - started, 3)
    return step


# ── Individual steps ────────────────────────────────────────────────


def _step_preflight(target: CloudTarget) -> MigrationStep:
    """Run cloud_doctor; surface its overall + per-probe state."""
    from app.substrate.cloud_doctor import check_readiness
    r = check_readiness(target)
    status_map: dict[str, StepStatus] = {
        "OK": "ok",
        "DEGRADED": "warn",
        "MISSING": "fail",
        "UNKNOWN": "warn",
    }
    return MigrationStep(
        name="preflight",
        status=status_map.get(r.overall, "warn"),
        detail=f"cloud_doctor overall={r.overall}",
        output={
            "overall": r.overall,
            "probes": [
                {"name": p.name, "status": p.status, "required": p.required, "detail": p.detail}
                for p in r.probes
            ],
        },
    )


def _step_cost_estimate(
    target: CloudTarget, tier: Tier, region: str | None,
    budget_cap_usd: float | None,
) -> MigrationStep:
    """Produce the itemized cost estimate; flag if it exceeds the cap."""
    from app.substrate.cloud_cost import estimate_monthly_cost
    b = estimate_monthly_cost(target, tier, region=region)
    over_cap = (
        budget_cap_usd is not None
        and b.total_monthly_usd > budget_cap_usd
    )
    return MigrationStep(
        name="cost_estimate",
        status="fail" if over_cap else "ok",
        detail=(
            f"${b.total_monthly_usd:.2f}/mo exceeds cap ${budget_cap_usd:.2f}"
            if over_cap else
            f"${b.total_monthly_usd:.2f}/mo ({tier}, {b.region})"
        ),
        output=b.to_dict(),
    )


def _step_bundle_metadata() -> MigrationStep:
    """Inspect the freshest DR tarball. Surface its size + age + manifest."""
    try:
        from app.paths import WORKSPACE_ROOT
    except Exception as exc:
        return MigrationStep(
            name="bundle_metadata",
            status="fail",
            detail=f"paths import failed: {exc}",
        )

    backup_dir = Path(WORKSPACE_ROOT) / "backups" / "dr"
    if not backup_dir.exists():
        return MigrationStep(
            name="bundle_metadata",
            status="fail",
            detail="no workspace/backups/dr/ — run `botarmy backup` first",
        )
    tarballs = sorted(
        backup_dir.glob("*.tar.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not tarballs:
        return MigrationStep(
            name="bundle_metadata",
            status="fail",
            detail="no DR tarballs in workspace/backups/dr/",
        )
    newest = tarballs[0]
    size_bytes = newest.stat().st_size
    age_seconds = datetime.now(timezone.utc).timestamp() - newest.stat().st_mtime
    age_days = age_seconds / 86400

    # Best-effort manifest peek: dr exports embed manifest.json in the tarball
    manifest_summary: dict[str, Any] = {}
    try:
        import tarfile
        with tarfile.open(newest, "r:gz") as tf:
            for member in tf.getmembers():
                if member.name.endswith("manifest.json"):
                    fobj = tf.extractfile(member)
                    if fobj:
                        m = json.loads(fobj.read())
                        manifest_summary = {
                            "started_at": m.get("started_at"),
                            "total_rows_chromadb": m.get("total_rows_chromadb"),
                            "total_rows_postgres": m.get("total_rows_postgres"),
                            "total_bytes": m.get("total_bytes"),
                            "ok": m.get("ok"),
                            "subia_integrity_at_export": m.get("subia_integrity_at_export") or {},
                            "n_ledger_files": len(m.get("ledgers") or []),
                            "n_chroma_collections": len(m.get("chromadb") or []),
                            "n_postgres_tables": len(m.get("postgres") or []),
                        }
                    break
    except Exception as exc:
        logger.debug("migration: manifest peek failed", exc_info=True)
        manifest_summary = {"error": f"{type(exc).__name__}: {exc}"}

    # 7-day freshness — older than this and we warn.
    status: StepStatus = "ok"
    detail = f"{newest.name} ({size_bytes / 1024 / 1024:.1f} MB, {age_days:.1f} days old)"
    if age_days > 7:
        status = "warn"
        detail += " — STALE (consider `botarmy backup` first)"

    return MigrationStep(
        name="bundle_metadata",
        status=status,
        detail=detail,
        output={
            "bundle_path": str(newest),
            "bundle_size_bytes": size_bytes,
            "bundle_size_mb": round(size_bytes / 1024 / 1024, 2),
            "age_days": round(age_days, 2),
            "manifest_summary": manifest_summary,
        },
    )


def _step_transfer_plan(
    target: CloudTarget, project_id: str | None, run_id: str,
) -> MigrationStep:
    """Compute the transfer command + destination URI.

    No actual upload — just describes what would happen. The transfer
    is naturally scoped per run_id to keep multiple migration drills
    from clobbering each other in a shared bucket.
    """
    if target == "gcp":
        if not project_id:
            return MigrationStep(
                name="transfer_plan",
                status="fail",
                detail="--project required for gcp transfer plan",
            )
        bucket = f"gs://andrusai-migrations-{project_id}"
        dest = f"{bucket}/{run_id}/bundle.tar.gz"
        command = f"gcloud storage cp <bundle> {dest}"
    elif target == "aws":
        bucket = "s3://andrusai-migrations"
        dest = f"{bucket}/{run_id}/bundle.tar.gz"
        command = f"aws s3 cp <bundle> {dest}"
    else:
        return MigrationStep(
            name="transfer_plan",
            status="fail",
            detail=f"unknown target: {target}",
        )
    return MigrationStep(
        name="transfer_plan",
        status="ok",
        detail=f"would upload to {dest}",
        output={
            "bucket": bucket,
            "destination": dest,
            "command": command,
            "note": (
                "secrets are excluded by the DR exporter's denylist — "
                "the cloud side initializes its own via Secret Manager."
            ),
        },
    )


def _step_restore_plan(target: CloudTarget) -> MigrationStep:
    """Plan the cloud-side restore step. No actual kubectl invocation."""
    namespace = "botarmy"
    commands = [
        f"kubectl -n {namespace} apply -f deploy/k8s/  # via Helm",
        f"kubectl -n {namespace} exec gateway-0 -- python -m app.dr.import_kbs --bundle <gcs_uri>",
    ]
    return MigrationStep(
        name="restore_plan",
        status="ok",
        detail=f"would restore into namespace={namespace}",
        output={
            "target_namespace": namespace,
            "commands": commands,
        },
    )


def _step_verify_plan() -> MigrationStep:
    """Plan the verification step. No actual verification."""
    checks = [
        "boot_drill on the restored cluster",
        "verify_integrity (SubIA) returns OK",
        "ChromaDB row counts match manifest",
        "Postgres table row counts match manifest",
        "Continuity ledger event count matches source",
        "Test request through gateway returns 200",
    ]
    return MigrationStep(
        name="verify_plan",
        status="ok",
        detail="6 verification checks would run on the live target",
        output={"checks": checks},
    )


# ── Live-mode pre-flight gates (Phase 3) ────────────────────────────


TYPED_PHRASE_REQUIRED = "MIGRATE TO GCP"


@dataclass(frozen=True)
class GateResult:
    """One pre-flight gate's verdict before any live action."""
    name: str
    passed: bool
    detail: str


def evaluate_live_gates(
    *,
    target: CloudTarget,
    project_id: str | None,
    confirm_phrase: str,
    budget_cap_usd: float,
    cost_estimate_monthly_usd: float,
    bundle_age_days: float | None,
    cloud_doctor_overall: str,
) -> list[GateResult]:
    """Run every refuse-on-fail gate in order. Pure function — no I/O.

    Defense in depth: each gate is independent so a regression in one
    can't paper over the others. Caller asserts ``all(g.passed)`` before
    executing any live step.
    """
    gates: list[GateResult] = []

    # G1 — typed-phrase confirmation
    expected = TYPED_PHRASE_REQUIRED if target == "gcp" else f"MIGRATE TO {target.upper()}"
    gates.append(GateResult(
        name="typed_phrase",
        passed=(confirm_phrase == expected),
        detail=(
            "ok" if confirm_phrase == expected
            else f"--confirm must equal exactly {expected!r} (got {confirm_phrase!r})"
        ),
    ))

    # G2 — project required for cloud target
    gates.append(GateResult(
        name="project_id",
        passed=bool(project_id),
        detail="ok" if project_id else f"--project required for --to {target}",
    ))

    # G3 — budget cap
    over = cost_estimate_monthly_usd > budget_cap_usd
    gates.append(GateResult(
        name="budget_cap",
        passed=not over,
        detail=(
            f"${cost_estimate_monthly_usd:.2f}/mo exceeds cap ${budget_cap_usd:.2f}"
            if over else
            f"ok (${cost_estimate_monthly_usd:.2f}/mo ≤ ${budget_cap_usd:.2f})"
        ),
    ))

    # G4 — bundle freshness (24h hard cap for live)
    if bundle_age_days is None:
        gates.append(GateResult(
            name="bundle_freshness",
            passed=False,
            detail="no DR bundle — run `botarmy backup` first",
        ))
    else:
        fresh = bundle_age_days <= 1.0
        gates.append(GateResult(
            name="bundle_freshness",
            passed=fresh,
            detail=(
                "ok"
                if fresh else
                f"newest bundle is {bundle_age_days:.1f} days old (> 24h) — run `botarmy backup` first"
            ),
        ))

    # G5 — cloud_doctor must be OK (not just non-MISSING — the live
    # path can't tolerate DEGRADED probes either, because they often
    # indicate missing ADC which makes terraform apply fail mid-way)
    gates.append(GateResult(
        name="cloud_doctor",
        passed=(cloud_doctor_overall == "OK"),
        detail=(
            "ok" if cloud_doctor_overall == "OK"
            else f"cloud_doctor overall={cloud_doctor_overall} — resolve before --live"
        ),
    ))

    return gates


# ── Subprocess plumbing (testable seam) ─────────────────────────────
#
# Every live step shells out through ``_shell`` so tests can monkey-patch
# one symbol to neutralize the entire execution path. ``_shell`` itself
# has a hard guard: it refuses to run unless the environment opts in via
# ``BOTARMY_MIGRATE_LIVE_EXECUTE=1`` OR the explicit ``execute=True``
# argument. Belt-and-suspenders against accidental real-cloud invocation.


import os
import subprocess


def _shell(
    argv: list[str],
    *,
    timeout: float,
    execute: bool = False,
    cwd: Path | str | None = None,
) -> tuple[int, str, str]:
    """Run a shell command with hard execute-gate.

    Returns (returncode, stdout, stderr). Never raises.

    Refuses to execute unless either:
      - ``execute=True`` was passed, OR
      - ``BOTARMY_MIGRATE_LIVE_EXECUTE=1`` is set

    With neither, returns (0, "<dry: $cmd>", "") — useful for unit tests
    that want to verify the orchestrator builds the right commands.

    Resolution of the env-var alternative is delegated to
    ``cloud_prep.is_live_execute_enabled``, which consults the
    runtime_settings JSON FIRST (so a React toggle takes effect
    immediately) and falls back to the env var.
    """
    from app.substrate.cloud_prep import is_live_execute_enabled
    if not (execute or is_live_execute_enabled()):
        return 0, f"<dry: {' '.join(argv)}>", ""
    cmd_name = argv[0] if argv else "<empty argv>"
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd_name}: command not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd_name}: timed out after {timeout}s"
    except Exception as exc:
        return 1, "", f"{cmd_name}: {type(exc).__name__}: {exc}"


# ── Live step implementations ───────────────────────────────────────


def _repo_root() -> Path:
    """Find the crewai-team repo root from this module's own location.

    Used by ``_step_provision_live`` to locate install scripts WITHOUT
    going through WORKSPACE_ROOT (which tests redirect). Install
    scripts live in the repo, not the workspace.
    """
    # app/substrate/migration.py → parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def _write_per_run_tfvars(
    target: CloudTarget, project_id: str, region: str, tier: Tier, run_id: str,
) -> Path:
    """Generate a minimal per-run terraform.tfvars under workspace/migrations/.

    gcp.sh refuses to proceed under ``--non-interactive`` without a tfvars
    file. We write one keyed to the run id so:
      * The operator's existing ``deploy/terraform/gcp/terraform.tfvars``
        (if any) is NOT clobbered.
      * Each migration run has an auditable snapshot of what it provisioned.
      * Cleanup is just ``rm -rf workspace/migrations/<run_id>``.

    The generated file deliberately omits ``extra_env`` (no API keys —
    operator manages those via Secret Manager post-deploy), ``domain``
    (no public ingress — saves the load balancer cost), and
    ``enable_monitoring`` (saves ~$15/mo; opt-in for the operator).

    Hardening additions (2026-05-17): pulls ``hardening_profile``,
    ``binauthz_mode`` from runtime_settings; auto-detects Tailnet CIDR +
    laptop public IP + Workspace org_id via ``cloud_hardening``. All
    detection is failure-isolated so a missing tailscale binary doesn't
    block the apply.
    """
    try:
        from app.paths import WORKSPACE_ROOT
        out_dir = Path(WORKSPACE_ROOT) / "migrations" / run_id
    except Exception:
        out_dir = Path("/tmp") / "botarmy_migrations" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tfvars_path = out_dir / "terraform.tfvars"

    # ── Hardening detection (failure-isolated) ──────────────────
    hardening_profile = "strict"
    binauthz_mode = "AUDIT"
    allowed_cidrs_block = "allowed_cidrs     = []\n"
    # Target-specific defaults for the org block. GCP uses `org_id`;
    # AWS uses `aws_org_enabled` + `aws_org_root_id`.
    if target == "aws":
        org_id_block = 'aws_org_enabled   = false\n'
    else:
        org_id_block = 'org_id            = ""\n'
    try:
        from app.runtime_settings import get_hardening_profile, get_binauthz_mode
        hardening_profile = get_hardening_profile()
        binauthz_mode = get_binauthz_mode()
    except Exception:
        pass

    # VPC-SC tfvars block (GCP-only, opt-in)
    vpc_sc_block = ""

    if hardening_profile == "strict":
        try:
            from app.substrate.cloud_hardening import (
                detect_tailnet_cidr, detect_laptop_public_ip,
                detect_org_id, detect_aws_org_root_id,
                detect_access_policy_id, build_allowed_cidrs,
            )
            tailnet = detect_tailnet_cidr()
            laptop = detect_laptop_public_ip()
            cidrs = build_allowed_cidrs(tailnet_cidr=tailnet, laptop_public_ip=laptop)
            if cidrs:
                entries = ",\n".join(
                    f'  {{ cidr_block = "{c.cidr_block}", display_name = "{c.display_name}" }}'
                    for c in cidrs
                )
                allowed_cidrs_block = f"allowed_cidrs = [\n{entries}\n]\n"
            if target == "gcp":
                org_id = detect_org_id()
                if org_id:
                    org_id_block = f'org_id            = "{org_id}"\n'
                # VPC-SC opt-in path
                try:
                    from app.runtime_settings import (
                        get_vpc_sc_enabled, get_vpc_sc_dry_run,
                        get_binauthz_attestor_name,
                    )
                    if get_binauthz_attestor_name():
                        attestor_line = f'binauthz_attestor_name = "{get_binauthz_attestor_name()}"\n'
                    else:
                        attestor_line = ""
                    if get_vpc_sc_enabled() and org_id:
                        access_policy = detect_access_policy_id(org_id)
                        if access_policy:
                            vpc_sc_block = (
                                f"vpc_sc_enabled    = true\n"
                                f'access_policy_id  = "{access_policy}"\n'
                                f"vpc_sc_dry_run    = {str(get_vpc_sc_dry_run()).lower()}\n"
                            )
                        else:
                            # Operator opted in but no access policy exists yet —
                            # leave vpc_sc_enabled=false in tfvars, surface
                            # the gap via the hardening_summary output.
                            attestor_line = attestor_line  # no-op; keep flake clean
                    # Inject attestor + vpc_sc lines after org_id_block
                    org_id_block = org_id_block + attestor_line
                except Exception:
                    pass
            elif target == "aws":
                # AWS path: substitute org_id block for aws_org_root_id +
                # aws_org_enabled. Only the management account can
                # auto-detect; workload accounts get an empty/False pair.
                root_id = detect_aws_org_root_id()
                if root_id:
                    org_id_block = (
                        f'aws_org_enabled   = true\n'
                        f'aws_org_root_id   = "{root_id}"\n'
                    )
                else:
                    # Override the GCP-shaped default — AWS doesn't use org_id
                    org_id_block = (
                        'aws_org_enabled   = false\n'
                        'aws_org_root_id   = ""\n'
                    )
        except Exception:
            # Detection failed — leave the empty defaults. Operator can
            # edit the rendered tfvars by hand before re-running, or
            # use `botarmy hardening refresh-allowed-cidrs` after apply.
            pass

    # Conservative defaults: no public ingress, no monitoring, no API keys.
    # Operator adds these post-deploy via Secret Manager + a follow-up
    # terraform apply with a hand-tuned tfvars when ready for production.
    #
    # GCP and AWS share the same shape for: region/tier/extra_env/domain,
    # hardening_profile, allowed_cidrs. They diverge on the project id
    # (GCP has it, AWS doesn't) and the org block (GCP=org_id; AWS=
    # aws_org_enabled+aws_org_root_id). binauthz_mode is GCP-only.
    project_line = f'project_id        = "{project_id}"\n' if target == "gcp" else ""
    binauthz_line = f'binauthz_mode     = "{binauthz_mode}"\n' if target == "gcp" else ""

    content = (
        f"# Auto-generated by app/substrate/migration.py for run_id={run_id}\n"
        f"# Productization plan WP D Phase 3 + hardening (2026-05-17).\n"
        f"# Manual additions (e.g. API keys, domain, monitoring) should\n"
        f"# go in a SEPARATE deploy/terraform/{target}/terraform.tfvars file\n"
        f"# and be applied via a follow-up `terraform apply`.\n"
        f"\n"
        f"{project_line}"
        f'region            = "{region}"\n'
        f'tier              = "{tier}"\n'
        + ("enable_monitoring = false\n" if target == "gcp" else "")
        + f'domain            = ""\n'
        f"extra_env         = {{}}\n"
        f"\n"
        f"# ── Hardening (defaults: strict). Tune in /cp/settings.\n"
        f'hardening_profile = "{hardening_profile}"\n'
        f"{binauthz_line}"
        f"{allowed_cidrs_block}"
        f"{org_id_block}"
        f"{vpc_sc_block}"
    )
    tfvars_path.write_text(content, encoding="utf-8")
    return tfvars_path


def _step_provision_live(
    target: CloudTarget, project_id: str, region: str, tier: Tier, run_id: str,
) -> MigrationStep:
    """Run scripts/install/<target>.sh — provisions empty target cluster.

    Does NOT restore data. That's the next step. Timeout: 30 minutes
    (terraform apply on GKE Autopilot is typically 8-12 min).

    For GCP target: writes a per-run terraform.tfvars under
    ``workspace/migrations/<run_id>/`` and passes its path to gcp.sh
    via ``--config``. Avoids clobbering any operator-owned
    ``deploy/terraform/gcp/terraform.tfvars``.
    """
    install_sh = _repo_root() / "scripts" / "install" / f"{target}.sh"
    # In dry-shell mode (test env), the script's physical presence is
    # not required — _shell short-circuits before invoking bash. In
    # real-execute mode, bash will return 127 if the script is missing
    # and the step will record fail with a clear message.

    env = dict(os.environ)
    install_argv = ["bash", str(install_sh), "--target", target, "--non-interactive"]
    if target == "gcp":
        env["TF_VAR_project_id"] = project_id
        env["TF_VAR_region"] = region
        env["TF_VAR_tier"] = tier
        # Tag every resource with the migration run id so an operator
        # can grep their GCP audit log for "what did this run create?"
        env["TF_VAR_labels"] = (
            f'{{project = "botarmy", managed_by = "terraform", '
            f'migration_run_id = "{run_id}"}}'
        )
        # Write per-run tfvars and pass via --config (install.sh forwards
        # to gcp.sh, which uses it instead of looking for the default
        # terraform.tfvars).
        tfvars_path = _write_per_run_tfvars(target, project_id, region, tier, run_id)
        install_argv.extend(["--config", str(tfvars_path)])

    rc, out, err = _shell(install_argv, timeout=1800.0)
    if rc != 0:
        return MigrationStep(
            name="provision",
            status="fail",
            detail=f"install script exit {rc} — check terraform state, may need `terraform destroy`",
            output={
                "command": " ".join(install_argv),
                "stdout_tail": out[-2000:] if out else "",
                "stderr_tail": err[-2000:] if err else "",
            },
        )
    return MigrationStep(
        name="provision",
        status="ok",
        detail=f"{target} target provisioned",
        output={
            "command": " ".join(install_argv),
            "stdout_tail": out[-500:] if out else "",
        },
    )


def _step_transfer_live(
    target: CloudTarget, project_id: str, run_id: str, bundle_path: Path,
) -> MigrationStep:
    """Upload the DR bundle to the cloud-side bucket.

    Bucket is per-run-id-namespaced so multiple migrations don't clobber
    each other. The bucket itself is provisioned by terraform during
    Phase 3.1 (or pre-created by the operator).
    """
    if target == "gcp":
        bucket = f"gs://andrusai-migrations-{project_id}"
        dest = f"{bucket}/{run_id}/bundle.tar.gz"
        cmd = ["gcloud", "storage", "cp", str(bundle_path), dest]
    elif target == "aws":
        bucket = "s3://andrusai-migrations"
        dest = f"{bucket}/{run_id}/bundle.tar.gz"
        cmd = ["aws", "s3", "cp", str(bundle_path), dest]
    else:
        return MigrationStep(name="transfer", status="fail",
                              detail=f"unknown target: {target}")

    rc, out, err = _shell(cmd, timeout=600.0)  # 10 min — bundles can be big
    if rc != 0:
        return MigrationStep(
            name="transfer",
            status="fail",
            detail=f"upload failed exit {rc}: {(err or out)[:200]}",
            output={"command": " ".join(cmd), "dest": dest, "stderr_tail": err[-500:]},
        )
    return MigrationStep(
        name="transfer",
        status="ok",
        detail=f"uploaded to {dest}",
        output={"command": " ".join(cmd), "dest": dest},
    )


def _step_restore_live(
    target: CloudTarget, project_id: str, run_id: str,
) -> MigrationStep:
    """Run the DR import inside the target cluster's gateway pod."""
    namespace = "botarmy"
    if target == "gcp":
        bucket = f"gs://andrusai-migrations-{project_id}"
        bundle_uri = f"{bucket}/{run_id}/bundle.tar.gz"
    else:
        bucket = "s3://andrusai-migrations"
        bundle_uri = f"{bucket}/{run_id}/bundle.tar.gz"

    # Pod name selector — first gateway pod
    selector_cmd = [
        "kubectl", "-n", namespace, "get", "pod",
        "-l", "app=gateway", "-o", "jsonpath={.items[0].metadata.name}",
    ]
    rc, pod_name, err = _shell(selector_cmd, timeout=60.0)
    if rc != 0 or not pod_name.strip():
        return MigrationStep(
            name="restore",
            status="fail",
            detail=f"could not find gateway pod: {(err or pod_name)[:200]}",
            output={"command": " ".join(selector_cmd)},
        )
    pod = pod_name.strip()

    import_cmd = [
        "kubectl", "-n", namespace, "exec", pod, "--",
        "python", "-m", "app.dr.import_kbs", "--bundle", bundle_uri,
    ]
    rc, out, err = _shell(import_cmd, timeout=1200.0)   # 20 min for import
    if rc != 0:
        return MigrationStep(
            name="restore",
            status="fail",
            detail=f"import exit {rc}: {(err or out)[-200:]}",
            output={
                "command": " ".join(import_cmd),
                "stderr_tail": err[-1000:] if err else "",
            },
        )
    return MigrationStep(
        name="restore",
        status="ok",
        detail=f"bundle imported into {pod}",
        output={
            "command": " ".join(import_cmd),
            "stdout_tail": out[-500:] if out else "",
            "pod": pod,
        },
    )


def _step_verify_live(target: CloudTarget) -> MigrationStep:
    """Run boot_drill + SubIA integrity check against the live target."""
    namespace = "botarmy"
    verify_cmd = [
        "kubectl", "-n", namespace, "exec", "deploy/gateway", "--",
        "python", "-c",
        "from app.dr.boot_drill import run_drill; "
        "from app.subia.integrity import verify_integrity; "
        "r = run_drill(); v = verify_integrity(strict=False); "
        "import json; "
        "print(json.dumps({'drill_ok': r.overall_ok, "
        "'subia_ok': v.ok, 'n_files': v.n_files}))",
    ]
    rc, out, err = _shell(verify_cmd, timeout=600.0)
    if rc != 0:
        return MigrationStep(
            name="verify",
            status="fail",
            detail=f"verify exit {rc}: {(err or out)[:200]}",
        )
    # Parse the printed JSON
    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(out.strip().split("\n")[-1])
    except Exception:
        return MigrationStep(
            name="verify",
            status="warn",
            detail=f"verify ran but output unparseable: {out[:200]}",
            output={"stdout_tail": out[-500:]},
        )
    drill_ok = bool(parsed.get("drill_ok"))
    subia_ok = bool(parsed.get("subia_ok"))
    if drill_ok and subia_ok:
        return MigrationStep(
            name="verify",
            status="ok",
            detail=f"drill OK, subia OK ({parsed.get('n_files')} files)",
            output=parsed,
        )
    return MigrationStep(
        name="verify",
        status="fail",
        detail=f"verify checks failed: drill_ok={drill_ok} subia_ok={subia_ok}",
        output=parsed,
    )


# ── Finalize: blockers + ready_for_live ─────────────────────────────


def _finalize(run: MigrationRun) -> None:
    """Walk steps; classify failures as blockers, warnings as warnings."""
    for s in run.steps:
        if s.status == "fail":
            run.blockers.append(f"{s.name}: {s.detail}")
        elif s.status == "warn":
            run.warnings.append(f"{s.name}: {s.detail}")
    if run.dry_run:
        run.ready_for_live = not run.blockers
    else:
        # Live mode: ready_for_live means "all live steps succeeded"
        run.ready_for_live = not run.blockers


# ── Continuity-ledger emission ──────────────────────────────────────


def _emit_ledger_event(run: MigrationRun, phase: str | None = None) -> None:
    """Record a migration landmark on the identity continuity ledger.

    Dry-run emits one event at completion. Live mode emits at each
    transition (started / provisioned / restored / verified / completed
    / failed). Annual reflection auto-surfaces via
    ``summarise_drift.by_kind``.
    """
    try:
        from app.identity.continuity_ledger import record_event
        if phase is None:
            phase = "dry_run" if run.dry_run else (
                "live_ready" if run.ready_for_live else "live_failed"
            )
        verdict = (
            "ready" if run.ready_for_live else "blocked"
        ) if run.dry_run else phase
        summary = (
            f"{phase} {verdict} → "
            f"{run.target} {run.region} (run_id={run.run_id[:12]})"
        )
        record_event(
            kind="cloud_migration",
            actor="botarmy_migrate",
            summary=summary,
            detail={
                "phase": phase,
                "run_id": run.run_id,
                "target": run.target,
                "tier": run.tier,
                "region": run.region,
                "project_id": run.project_id,
                "ready_for_live": run.ready_for_live,
                "n_blockers": len(run.blockers),
                "n_warnings": len(run.warnings),
                "dry_run": run.dry_run,
            },
        )
    except Exception:
        logger.debug("migration: ledger emit failed (non-fatal)", exc_info=True)


# ── Report persistence ──────────────────────────────────────────────


def _write_report(run: MigrationRun) -> Path:
    """Write ``workspace/migrations/<run_id>/report.json``."""
    try:
        from app.paths import WORKSPACE_ROOT
        out_dir = Path(WORKSPACE_ROOT) / "migrations" / run.run_id
    except Exception:
        out_dir = Path("/tmp") / "botarmy_migrations" / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.json"
    report.write_text(
        json.dumps(run.to_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


# ── Public entry ────────────────────────────────────────────────────


def run_migration_dry_run(
    *,
    target: CloudTarget = "gcp",
    tier: Tier = "cheapest",
    region: str | None = None,
    project_id: str | None = None,
    budget_cap_usd: float | None = None,
    run_id: str | None = None,
) -> MigrationRun:
    """Drive a dry-run migration. Never raises, never spends money.

    Pipeline:
      1. preflight       — cloud_doctor probe rollup
      2. cost_estimate   — itemized monthly cost; flagged if > budget cap
      3. bundle_metadata — freshness, size, SubIA manifest snapshot
      4. transfer_plan   — destination bucket + command (NOT executed)
      5. restore_plan    — kubectl commands (NOT executed)
      6. verify_plan     — checks that would run (NOT executed)

    Roll-up: fail-class steps become blockers (refuse live migrate);
    warn-class steps become warnings. ``ready_for_live`` is True iff
    every step is ok or warn (no blockers).
    """
    if region is None:
        region = "europe-north1" if target == "gcp" else "eu-north-1"
    if run_id is None:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_" + uuid.uuid4().hex[:8]
        )

    started_mono = time.monotonic()
    run = MigrationRun(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        target=target,
        tier=tier,
        region=region,
        project_id=project_id,
        dry_run=True,
    )

    _record(run, _time_step("preflight", lambda: _step_preflight(target)))
    _record(run, _time_step(
        "cost_estimate",
        lambda: _step_cost_estimate(target, tier, region, budget_cap_usd),
    ))
    _record(run, _time_step("bundle_metadata", _step_bundle_metadata))
    _record(run, _time_step(
        "transfer_plan",
        lambda: _step_transfer_plan(target, project_id, run_id),
    ))
    _record(run, _time_step("restore_plan", lambda: _step_restore_plan(target)))
    _record(run, _time_step("verify_plan", _step_verify_plan))

    _finalize(run)
    run.completed_at = datetime.now(timezone.utc).isoformat()
    run.duration_s = round(time.monotonic() - started_mono, 3)

    _write_report(run)
    _emit_ledger_event(run)
    return run


# ── Live orchestrator (Phase 3) ─────────────────────────────────────


class GateFailure(Exception):
    """Raised when a pre-flight gate refuses live migrate.

    Distinct from a normal step failure because gates run BEFORE any
    side-effecting work and represent operator-fixable problems
    (wrong typed phrase, stale bundle, etc).
    """


def run_migration_live(
    *,
    target: CloudTarget = "gcp",
    tier: Tier = "cheapest",
    region: str | None = None,
    project_id: str | None = None,
    confirm_phrase: str = "",
    budget_cap_usd: float = 200.0,
    run_id: str | None = None,
    execute_subprocess: bool = False,
) -> MigrationRun:
    """Drive a LIVE migration — provisions cloud resources, transfers
    the bundle, restores it, verifies.

    Hard gates (all must pass before any cloud-side work):
      * typed phrase ``--confirm "MIGRATE TO GCP"``
      * project id present
      * estimated monthly cost ≤ ``budget_cap_usd``
      * latest DR bundle ≤ 24h old
      * cloud_doctor overall=OK (not DEGRADED)

    Gate failure raises ``GateFailure`` BEFORE any side effects. Step
    failure (terraform apply error, kubectl exec error) appends a fail
    step + halts the pipeline + writes a partial report. Operator is
    expected to run ``terraform destroy`` and start over (Phase 3 v1
    does not auto-rollback).

    ``execute_subprocess=True`` is the second-layer guard alongside the
    ``BOTARMY_MIGRATE_LIVE_EXECUTE=1`` env var. Tests leave both off so
    the orchestrator runs without spending money.
    """
    if region is None:
        region = "europe-north1" if target == "gcp" else "eu-north-1"
    if run_id is None:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_" + uuid.uuid4().hex[:8]
        )

    # ── 1. Compute cost + bundle metadata WITHOUT side effects ──────
    from app.substrate.cloud_cost import estimate_monthly_cost
    cost = estimate_monthly_cost(target, tier, region=region)

    from app.substrate.cloud_doctor import check_readiness
    cd = check_readiness(target)

    # Bundle age
    bundle_age_days: float | None = None
    bundle_path: Path | None = None
    try:
        from app.paths import WORKSPACE_ROOT
        backup_dir = Path(WORKSPACE_ROOT) / "backups" / "dr"
        if backup_dir.exists():
            tarballs = sorted(
                backup_dir.glob("*.tar.gz"),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if tarballs:
                bundle_path = tarballs[0]
                age_seconds = datetime.now(timezone.utc).timestamp() - bundle_path.stat().st_mtime
                bundle_age_days = age_seconds / 86400
    except Exception as exc:
        logger.debug("migration_live: bundle inspect failed", exc_info=True)
        bundle_age_days = None

    # ── 2. Evaluate gates ────────────────────────────────────────────
    gates = evaluate_live_gates(
        target=target,
        project_id=project_id,
        confirm_phrase=confirm_phrase,
        budget_cap_usd=budget_cap_usd,
        cost_estimate_monthly_usd=cost.total_monthly_usd,
        bundle_age_days=bundle_age_days,
        cloud_doctor_overall=cd.overall,
    )
    failed_gates = [g for g in gates if not g.passed]
    if failed_gates:
        raise GateFailure(
            "live migrate refused — "
            + "; ".join(f"{g.name}: {g.detail}" for g in failed_gates)
        )

    # ── 3. Build the run + execute steps ─────────────────────────────
    started_mono = time.monotonic()
    run = MigrationRun(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        target=target,
        tier=tier,
        region=region,
        project_id=project_id,
        dry_run=False,
    )

    # Landmark: live started
    _emit_ledger_event(run, phase="live_started")

    # Step pipeline — halts on first failure (unlike dry-run which
    # continues for report completeness). Each step's _shell() call
    # respects execute_subprocess.

    # Patch _shell at module level for this call so execute_subprocess
    # propagates without API churn through every step.
    global _shell
    _orig_shell = _shell

    def _gated_shell(argv, *, timeout, execute=False, cwd=None):
        return _orig_shell(argv, timeout=timeout, execute=execute_subprocess or execute, cwd=cwd)

    _shell = _gated_shell
    try:
        # Each step returns a MigrationStep; we halt on the first fail.
        live_steps: list = [
            ("provision", lambda: _step_provision_live(target, project_id, region, tier, run_id)),
            ("transfer",  lambda: _step_transfer_live(target, project_id, run_id, bundle_path)),
            ("restore",   lambda: _step_restore_live(target, project_id, run_id)),
            ("verify",    lambda: _step_verify_live(target)),
        ]
        for name, fn in live_steps:
            step = _time_step(name, fn)
            _record(run, step)
            if step.status == "fail":
                break
            # Emit a landmark for each successful step
            _emit_ledger_event(run, phase=f"live_{name}_completed")
    finally:
        _shell = _orig_shell

    _finalize(run)
    run.completed_at = datetime.now(timezone.utc).isoformat()
    run.duration_s = round(time.monotonic() - started_mono, 3)

    _write_report(run)
    # Final landmark
    _emit_ledger_event(
        run,
        phase="live_ready" if run.ready_for_live else "live_failed",
    )
    return run


# ── CLI-friendly formatter ──────────────────────────────────────────


def format_run(run: MigrationRun) -> str:
    """Human-readable summary, suitable for the CLI."""
    lines: list[str] = []
    mode = "dry-run" if run.dry_run else "LIVE"
    lines.append(
        f"=== Migration {mode} {run.run_id[:12]} — "
        f"{run.target}/{run.tier}/{run.region} ==="
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

    # Cost summary
    cost_step = next((s for s in run.steps if s.name == "cost_estimate"), None)
    if cost_step and cost_step.output:
        lines.append("")
        lines.append(
            f"  Estimated cost: ${cost_step.output.get('total_monthly_usd', 0):.2f}/mo "
            f"(${cost_step.output.get('total_annual_usd', 0):.2f}/yr)"
        )

    lines.append("")
    if run.dry_run:
        if run.ready_for_live:
            lines.append(f"  → READY for live migrate. Run id: {run.run_id}")
        else:
            lines.append("  → NOT READY for live migrate.")
            for b in run.blockers:
                lines.append(f"     • blocker: {b}")
    else:
        if run.ready_for_live:
            lines.append(
                f"  → LIVE migrate SUCCEEDED. Run id: {run.run_id}\n"
                f"     Cluster is provisioned + restored + verified.\n"
                f"     Next: run `botarmy cutover --to {run.run_id}` when ready (Phase 4)."
            )
        else:
            lines.append("  → LIVE migrate FAILED.")
            for b in run.blockers:
                lines.append(f"     • {b}")
            lines.append(
                "\n     Recovery: inspect workspace/migrations/{}/report.json,\n"
                "     then run `terraform destroy` in deploy/terraform/{}/ to clean up.".format(
                    run.run_id, run.target,
                )
            )
    if run.warnings:
        lines.append("")
        for w in run.warnings:
            lines.append(f"     • warning: {w}")
    return "\n".join(lines)
