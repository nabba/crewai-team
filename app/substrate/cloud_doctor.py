"""
cloud_doctor — probes the operator's local tooling for cloud-install readiness.

Productization plan WP D Phase 1. Asks: *does this workstation have
what it needs to run* ``botarmy migrate --to <target>`` *successfully?*

Per-probe model:
  * Every probe shells out with a 10-second timeout. Network probes
    (gcloud projects list) shell out longer because they round-trip to
    the cloud, but still bounded.
  * Per-probe status: ``OK`` / ``MISSING`` / ``STALE`` / ``UNKNOWN``.
  * Overall status: ``OK`` only when every probe is OK; ``MISSING`` if
    any probe missing required tooling; ``DEGRADED`` if optional
    tooling absent.

Never raises — a broken probe records an UNKNOWN with the exception
message. Never spends money: no probe creates cloud resources.

Composition: ``check_readiness()`` is the entry point used by both the
CLI (``botarmy cloud-doctor``) and the upcoming ``botarmy migrate``
preflight (Phase 2). Refuse-on-MISSING is the migrate preflight's job;
this module just reports.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

CloudTarget = Literal["gcp", "aws"]
ProbeStatus = Literal["OK", "MISSING", "STALE", "UNKNOWN"]


@dataclass
class ProbeResult:
    name: str
    status: ProbeStatus
    detail: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CloudReadiness:
    target: str
    timestamp: str
    overall: str   # OK | MISSING | DEGRADED | UNKNOWN
    probes: list[ProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "timestamp": self.timestamp,
            "overall": self.overall,
            "probes": [p.to_dict() for p in self.probes],
        }


# ── Generic shell helper ────────────────────────────────────────────


def _run(argv: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a subprocess; return ``(rc, stdout, stderr)``. Never raises."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"{argv[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{argv[0]}: timed out after {timeout}s"
    except Exception as exc:
        return 1, "", f"{argv[0]}: {type(exc).__name__}: {exc}"


def _probe_tool_present(name: str, argv: list[str], required: bool = True) -> ProbeResult:
    """Generic 'is this CLI installed and runnable?' probe."""
    if shutil.which(argv[0]) is None:
        return ProbeResult(
            name=name,
            status="MISSING",
            detail=f"{argv[0]} not on PATH",
            required=required,
        )
    rc, out, err = _run(argv)
    if rc != 0:
        return ProbeResult(
            name=name,
            status="UNKNOWN",
            detail=(err or out or f"exit {rc}")[:200],
            required=required,
        )
    # Truncate version banners for readable output
    detail = (out or err).splitlines()[0] if (out or err) else "present"
    return ProbeResult(
        name=name,
        status="OK",
        detail=detail[:160],
        required=required,
    )


# ── Cross-cloud probes ──────────────────────────────────────────────


def _probe_terraform() -> ProbeResult:
    return _probe_tool_present("terraform", ["terraform", "version"], required=True)


def _probe_kubectl() -> ProbeResult:
    return _probe_tool_present(
        "kubectl",
        ["kubectl", "version", "--client=true", "--output=yaml"],
        required=True,
    )


def _probe_helm() -> ProbeResult:
    return _probe_tool_present("helm", ["helm", "version", "--short"], required=True)


def _probe_docker() -> ProbeResult:
    """Docker is required for building the gateway image before push."""
    return _probe_tool_present("docker", ["docker", "--version"], required=True)


# ── GCP-specific probes ─────────────────────────────────────────────


def _probe_gcp_cli() -> ProbeResult:
    return _probe_tool_present("gcloud", ["gcloud", "--version"], required=True)


def _probe_gcp_auth() -> ProbeResult:
    """Verify there is at least one active gcloud account."""
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="gcloud auth",
            status="MISSING",
            detail="gcloud not installed",
            required=True,
        )
    rc, out, err = _run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
    )
    if rc != 0:
        return ProbeResult(
            name="gcloud auth",
            status="UNKNOWN",
            detail=(err or out)[:200],
            required=True,
        )
    if not out:
        return ProbeResult(
            name="gcloud auth",
            status="MISSING",
            detail="no active account — run `gcloud auth login`",
            required=True,
        )
    # Truncate to first account line
    account = out.splitlines()[0]
    return ProbeResult(
        name="gcloud auth",
        status="OK",
        detail=f"active account: {account}",
        required=True,
    )


def _probe_gcp_project() -> ProbeResult:
    """Verify a default project is set."""
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="gcloud project",
            status="MISSING",
            detail="gcloud not installed",
            required=True,
        )
    rc, out, err = _run(["gcloud", "config", "get", "core/project"])
    if rc != 0:
        return ProbeResult(
            name="gcloud project",
            status="UNKNOWN",
            detail=(err or out)[:200],
            required=True,
        )
    # gcloud emits "(unset)" when no project is configured
    if not out or "(unset)" in out:
        return ProbeResult(
            name="gcloud project",
            status="MISSING",
            detail="no default project — run `gcloud config set project <id>`",
            required=True,
        )
    return ProbeResult(
        name="gcloud project",
        status="OK",
        detail=f"default project: {out}",
        required=True,
    )


def _probe_gcp_application_default_creds() -> ProbeResult:
    """Application Default Credentials — needed by Terraform's google
    provider."""
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="ADC",
            status="MISSING",
            detail="gcloud not installed",
            required=False,
        )
    rc, out, err = _run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        timeout=15.0,
    )
    if rc == 0 and out:
        return ProbeResult(
            name="ADC",
            status="OK",
            detail="application-default credentials present",
            required=False,
        )
    return ProbeResult(
        name="ADC",
        status="MISSING",
        detail="run `gcloud auth application-default login` before terraform",
        required=False,
    )


# ── GCP permission probes (gap-3 fix, 2026-05-17) ──────────────────
#
# The original 4 GCP probes verified "an account exists, a project is
# set, ADC can mint a token." All three can be technically true while
# the active identity lacks the project-level IAM roles that terraform
# apply actually needs. Live cloud migrate would then fail
# mid-pipeline, leaving cost-accruing partial state.
#
# The probes below close that gap with cheap read-only checks that
# answer "can this identity actually do the work?":
#   * Active account type (user vs narrow service account)
#   * Project-level read access (proves at minimum Viewer)
#   * Required APIs enabled (terraform refuses to create resources
#     against disabled APIs; some can be auto-enabled by terraform
#     itself but only if serviceUsageAdmin role is present)
#   * ADC's credentials file has a populated account field


def _probe_gcp_active_account_type() -> ProbeResult:
    """Distinguish user accounts from narrow service accounts.

    Service accounts ending in ``.gserviceaccount.com`` are often
    purpose-scoped (e.g., gee-runner, ci-pipeline) and lack the broad
    IAM roles terraform apply needs. Warning-level: an operator may
    deliberately use an SA with the right permissions; we don't want
    to hard-fail, just surface the situation.
    """
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="gcloud account type",
            status="MISSING",
            detail="gcloud not installed",
            required=False,
        )
    rc, out, err = _run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
    )
    if rc != 0 or not out:
        return ProbeResult(
            name="gcloud account type",
            status="UNKNOWN",
            detail=(err or "no active account")[:200],
            required=False,
        )
    account = out.splitlines()[0].strip()
    if account.endswith(".gserviceaccount.com"):
        return ProbeResult(
            name="gcloud account type",
            status="STALE",   # not a fail, but worth flagging
            detail=(
                f"active is a service account ({account.split('@')[0]}) — "
                f"terraform apply usually needs a user identity with "
                f"project Owner/Editor. Run `gcloud config set account "
                f"<your-user>` if this SA doesn't have project-wide perms."
            ),
            required=False,
        )
    return ProbeResult(
        name="gcloud account type",
        status="OK",
        detail=f"user account: {account}",
        required=False,
    )


def _probe_gcp_project_access() -> ProbeResult:
    """Verify the active identity can READ the configured project.

    ``gcloud projects describe <id>`` returns 200 with at least
    project/viewer, 403 without. This is the cheapest possible signal
    that the identity↔project pairing is sane before any terraform
    invocation that needs Editor/Owner.
    """
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="gcloud project access",
            status="MISSING",
            detail="gcloud not installed",
            required=True,
        )
    # Read the configured project first
    rc, out, err = _run(["gcloud", "config", "get", "core/project"])
    if rc != 0 or not out or "(unset)" in out:
        return ProbeResult(
            name="gcloud project access",
            status="MISSING",
            detail="no default project configured",
            required=True,
        )
    project = out.splitlines()[0].strip()
    # Now probe project access
    rc, out, err = _run(
        ["gcloud", "projects", "describe", project, "--format=value(projectId)"],
        timeout=20.0,
    )
    if rc != 0:
        msg = (err or out or "describe failed").strip().splitlines()[0]
        # Common: 403 (no permission) or 404 (project doesn't exist)
        if "403" in msg or "permission" in msg.lower():
            return ProbeResult(
                name="gcloud project access",
                status="MISSING",
                detail=(
                    f"403 on {project} — active account lacks even Viewer "
                    f"role. Switch account or grant roles/viewer."
                ),
                required=True,
            )
        if "404" in msg or "not found" in msg.lower():
            return ProbeResult(
                name="gcloud project access",
                status="MISSING",
                detail=f"project {project!r} not found",
                required=True,
            )
        return ProbeResult(
            name="gcloud project access",
            status="UNKNOWN",
            detail=msg[:200],
            required=True,
        )
    return ProbeResult(
        name="gcloud project access",
        status="OK",
        detail=f"{project} readable",
        required=True,
    )


# Critical APIs that terraform apply needs already enabled. Some can
# be enabled by ``google_project_service.required`` terraform resources
# but that itself needs ``serviceusage.serviceUsageAdmin``. Cheaper to
# enable them up-front via ``gcloud services enable`` and skip the
# meta-permission problem.
_REQUIRED_GCP_APIS = (
    "container.googleapis.com",          # GKE
    "sqladmin.googleapis.com",           # Cloud SQL
    "cloudresourcemanager.googleapis.com",  # IAM bindings + project ops
    "servicenetworking.googleapis.com",  # CloudSQL private VPC connection
    "compute.googleapis.com",            # VPC + load balancer
    "artifactregistry.googleapis.com",   # gateway image registry
    "secretmanager.googleapis.com",      # gateway secrets
    "iam.googleapis.com",                # workload identity + SAs
    "storage.googleapis.com",            # migrations bucket
)


def _probe_gcp_required_apis() -> ProbeResult:
    """Critical APIs must be enabled before terraform apply.

    A subset of these CAN be auto-enabled by terraform's
    ``google_project_service.required`` resource — but only if the
    identity invoking terraform has ``serviceusage.serviceUsageAdmin``,
    which most service accounts don't. Safer to verify up-front.
    """
    if shutil.which("gcloud") is None:
        return ProbeResult(
            name="gcloud required APIs",
            status="MISSING",
            detail="gcloud not installed",
            required=True,
        )
    rc, out, err = _run(
        ["gcloud", "services", "list", "--enabled", "--format=value(config.name)"],
        timeout=30.0,
    )
    if rc != 0:
        msg = (err or out or "services list failed").strip().splitlines()[0]
        return ProbeResult(
            name="gcloud required APIs",
            status="UNKNOWN",
            detail=msg[:200],
            required=True,
        )
    enabled = set(line.strip() for line in out.splitlines() if line.strip())
    missing = [api for api in _REQUIRED_GCP_APIS if api not in enabled]
    if missing:
        return ProbeResult(
            name="gcloud required APIs",
            status="MISSING",
            detail=(
                f"{len(missing)}/{len(_REQUIRED_GCP_APIS)} APIs disabled — "
                f"run `gcloud services enable {' '.join(missing[:3])}"
                f"{' ...' if len(missing) > 3 else ''}` to enable."
            ),
            required=True,
        )
    return ProbeResult(
        name="gcloud required APIs",
        status="OK",
        detail=f"{len(_REQUIRED_GCP_APIS)} required APIs enabled",
        required=True,
    )


def _probe_gcp_adc_populated() -> ProbeResult:
    """ADC credentials file should carry an explicit ``account`` field.

    An empty account means ADC uses an anonymous OAuth client with
    no quota-project context. Terraform usually works, but quota
    accounting and certain API calls (e.g., billing-aware ones) fail
    in subtle ways. Warning-level — operator can ignore but should
    know.
    """
    home = os.path.expanduser("~")
    adc_path = Path(home) / ".config" / "gcloud" / "application_default_credentials.json"
    if not adc_path.exists():
        return ProbeResult(
            name="ADC account",
            status="MISSING",
            detail=f"{adc_path} not found — run `gcloud auth application-default login`",
            required=False,
        )
    try:
        import json
        data = json.loads(adc_path.read_text())
    except Exception as exc:
        return ProbeResult(
            name="ADC account",
            status="UNKNOWN",
            detail=f"could not parse ADC json: {exc}",
            required=False,
        )
    account = data.get("account") or ""
    if not account.strip():
        return ProbeResult(
            name="ADC account",
            status="STALE",
            detail=(
                "ADC has no explicit account — terraform may run but "
                "quota accounting is per-project rather than per-user. "
                "Run `gcloud auth application-default login` to bind a user."
            ),
            required=False,
        )
    return ProbeResult(
        name="ADC account",
        status="OK",
        detail=f"bound to: {account}",
        required=False,
    )


# ── AWS-specific probes ─────────────────────────────────────────────


def _probe_aws_cli() -> ProbeResult:
    return _probe_tool_present("aws", ["aws", "--version"], required=True)


def _probe_aws_identity() -> ProbeResult:
    """sts get-caller-identity — proves the AWS CLI can authenticate."""
    if shutil.which("aws") is None:
        return ProbeResult(
            name="aws identity",
            status="MISSING",
            detail="aws CLI not installed",
            required=True,
        )
    rc, out, err = _run(["aws", "sts", "get-caller-identity", "--output", "text"])
    if rc != 0:
        return ProbeResult(
            name="aws identity",
            status="MISSING",
            detail=(err or out or "no credentials configured")[:200],
            required=True,
        )
    # First field is account-id, last is the principal ARN
    parts = out.split()
    arn = parts[-1] if parts else out
    return ProbeResult(
        name="aws identity",
        status="OK",
        detail=f"caller: {arn}",
        required=True,
    )


# ── Continuity-readiness probes (shared across targets) ─────────────


def _probe_continuity_bundle_recent(max_age_days: float = 7.0) -> ProbeResult:
    """A migrate without a fresh backup is reckless. Surface the age of
    the freshest DR tarball so the operator knows to run ``botarmy
    backup`` if it's stale."""
    try:
        from app.paths import WORKSPACE_ROOT
        backup_dir = WORKSPACE_ROOT / "backups" / "dr"
        if not backup_dir.exists():
            return ProbeResult(
                name="continuity bundle",
                status="MISSING",
                detail="no workspace/backups/dr/ — run `botarmy backup` first",
                required=True,
            )
        tarballs = sorted(
            backup_dir.glob("*.tar.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not tarballs:
            return ProbeResult(
                name="continuity bundle",
                status="MISSING",
                detail="no DR tarballs in workspace/backups/dr/",
                required=True,
            )
        newest = tarballs[0]
        age_days = (datetime.now(timezone.utc).timestamp() - newest.stat().st_mtime) / 86400
        if age_days > max_age_days:
            return ProbeResult(
                name="continuity bundle",
                status="STALE",
                detail=f"newest bundle is {age_days:.1f} days old (> {max_age_days})",
                required=True,
            )
        return ProbeResult(
            name="continuity bundle",
            status="OK",
            detail=f"{newest.name} — {age_days:.1f} days old",
            required=True,
        )
    except Exception as exc:
        return ProbeResult(
            name="continuity bundle",
            status="UNKNOWN",
            detail=f"{type(exc).__name__}: {exc}",
            required=True,
        )


def _probe_gcp_project_exists() -> ProbeResult:
    """Verify the target project exists. Required when
    ``gcp_bootstrap_enabled=False`` (which is the default); when bootstrap
    is on, this probe degrades to optional (a missing project triggers
    Stage 0a instead of refusing the run).
    """
    project = os.environ.get("CLOUDSDK_CORE_PROJECT") or os.environ.get("GOOGLE_PROJECT", "")
    if not project:
        # Fall back to gcloud config
        rc, out, _ = _run(["gcloud", "config", "get-value", "project"], timeout=10.0)
        project = out.strip() if rc == 0 else ""
    if not project:
        return ProbeResult(
            name="gcloud project exists",
            status="MISSING",
            detail="no project_id set (gcloud config get-value project returned empty)",
            required=False,
        )

    rc, out, err = _run(
        ["gcloud", "projects", "describe", project, "--format=value(projectId,lifecycleState)"],
        timeout=15.0,
    )
    if rc == 0 and "ACTIVE" in out.upper():
        return ProbeResult(
            name="gcloud project exists",
            status="OK",
            detail=f"{project} is ACTIVE",
            required=False,
        )
    # gcp_bootstrap_enabled=True path: missing project is fine (we'll create it)
    try:
        from app.runtime_settings import get_gcp_bootstrap_enabled
        bootstrap_on = get_gcp_bootstrap_enabled()
    except Exception:
        bootstrap_on = False
    return ProbeResult(
        name="gcloud project exists",
        status="MISSING",
        detail=(
            f"project {project} not found — Stage 0a (gcp_bootstrap) will create it"
            if bootstrap_on else
            f"project {project} not found and gcp_bootstrap_enabled is OFF; "
            f"either create the project by hand or enable bootstrap in /cp/settings"
        ),
        required=not bootstrap_on,
    )


def _probe_tailnet_reachable() -> ProbeResult:
    """Heads-up probe — does NOT block on its own. When Tailnet is not
    reachable AND hardening_profile=strict, the operator either sets
    ``var.allowed_cidrs`` manually or runs the wizard with the laptop
    IP fallback. The probe just surfaces the state.
    """
    try:
        from app.substrate.cloud_hardening import tailnet_reachable
        ok = tailnet_reachable()
    except Exception as exc:
        return ProbeResult(
            name="tailnet reachable",
            status="UNKNOWN",
            detail=f"{type(exc).__name__}: {exc}",
            required=False,
        )
    return ProbeResult(
        name="tailnet reachable",
        status="OK" if ok else "MISSING",
        detail="100.64.0.0/10 will be added to master_authorized_networks" if ok else (
            "tailscale not installed or not joined to a Tailnet — laptop public IP will be used as the sole allowlist entry"
        ),
        required=False,
    )


def _probe_gcp_binauthz_signing_ready() -> ProbeResult:
    """Soft probe — does the operator have an attestor + signed-image
    pipeline wired so Binary Authorization ENFORCE mode is safe to flip?

    This is a HEURISTIC check, not a guarantee: it looks for the
    presence of a ``cosign.pub`` key file under
    ``deploy/k8s/binauthz/`` OR an env var ``BINAUTHZ_ATTESTOR``. The
    real proof comes from a signed image landing in Artifact Registry.

    Always optional. Surfaces a clear "stay in AUDIT mode" message when
    no signing pipeline is detected.
    """
    repo_root = os.environ.get("INSTALL_ROOT", "")
    if not repo_root:
        # Best-effort guess: assume cwd is the repo root
        repo_root = os.getcwd()
    cosign_pub = os.path.join(repo_root, "deploy", "k8s", "binauthz", "cosign.pub")
    attestor = os.environ.get("BINAUTHZ_ATTESTOR", "").strip()
    if os.path.isfile(cosign_pub) or attestor:
        return ProbeResult(
            name="binauthz signing ready",
            status="OK",
            detail=(
                f"attestor={attestor or 'cosign.pub'} — ENFORCE mode is safe to flip"
            ),
            required=False,
        )
    return ProbeResult(
        name="binauthz signing ready",
        status="MISSING",
        detail=(
            "no image-signing pipeline detected (no deploy/k8s/binauthz/cosign.pub "
            "and BINAUTHZ_ATTESTOR is empty). Keep binauthz_mode=AUDIT until your "
            "signing pipeline lands — ENFORCE will reject every unsigned image."
        ),
        required=False,
    )


def verify_hardening(
    run_dir: str, *, project_id: str, target: CloudTarget = "gcp",
) -> dict[str, Any]:
    """Read the post-apply ``hardening_summary`` terraform output and
    confirm each policy actually landed. Called by the migrate wizard's
    Step 6 outcome card."""
    if target != "gcp":
        return {"ok": True, "summary": {}, "reason": f"verify_hardening: {target} not yet implemented"}
    try:
        from app.substrate.cloud_hardening import verify_hardening as _verify
        return _verify(run_dir, project_id=project_id)
    except Exception as exc:
        return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _probe_subia_integrity() -> ProbeResult:
    """SubIA must verify clean before migration — drift in the source
    would propagate to the cloud copy."""
    try:
        from app.subia.integrity import verify_integrity
        r = verify_integrity(strict=False)
        if r.ok:
            return ProbeResult(
                name="subia integrity",
                status="OK",
                detail=f"{r.n_files} files clean",
                required=False,  # not blocking — operator may choose to migrate anyway
            )
        n_mis = len(r.mismatched or [])
        n_extra = len(r.extra or [])
        n_miss = len(r.missing or [])
        return ProbeResult(
            name="subia integrity",
            status="STALE",
            detail=f"drift: mismatched={n_mis} extra={n_extra} missing={n_miss}",
            required=False,
        )
    except Exception as exc:
        return ProbeResult(
            name="subia integrity",
            status="UNKNOWN",
            detail=f"{type(exc).__name__}: {exc}",
            required=False,
        )


# ── Public entry ────────────────────────────────────────────────────


def check_readiness(target: CloudTarget = "gcp") -> CloudReadiness:
    """Probe the operator's local tooling for cloud-install readiness.

    Args:
      target: ``gcp`` or ``aws``.

    Returns:
      ``CloudReadiness`` with one ProbeResult per check. ``overall``
      summarizes:
        * ``OK``       — every required probe passed
        * ``MISSING``  — at least one required probe missing/stale
        * ``DEGRADED`` — only optional probes failing
    """
    probes: list[ProbeResult] = []

    # Generic tooling
    probes.append(_probe_terraform())
    probes.append(_probe_kubectl())
    probes.append(_probe_helm())
    probes.append(_probe_docker())

    # Target-specific
    if target == "gcp":
        probes.append(_probe_gcp_cli())
        probes.append(_probe_gcp_auth())
        probes.append(_probe_gcp_project())
        probes.append(_probe_gcp_application_default_creds())
        # Gap-3 additions (productization plan, 2026-05-17): verify
        # the identity↔project↔API tuple is workable, not just that
        # gcloud is technically authenticated.
        probes.append(_probe_gcp_active_account_type())
        probes.append(_probe_gcp_project_access())
        probes.append(_probe_gcp_required_apis())
        probes.append(_probe_gcp_adc_populated())
        # Hardening additions (2026-05-17): observational. Don't block,
        # just surface so the React wizard knows whether to render the
        # 0a Bootstrap card vs go straight to Step 1, and whether to
        # warn about ENFORCE-without-signing-pipeline.
        probes.append(_probe_gcp_project_exists())
        probes.append(_probe_tailnet_reachable())
        probes.append(_probe_gcp_binauthz_signing_ready())
    elif target == "aws":
        probes.append(_probe_aws_cli())
        probes.append(_probe_aws_identity())
    else:
        # Defensive — caller is type-checked but be explicit
        raise ValueError(f"unknown target: {target!r}")

    # Continuity preconditions
    probes.append(_probe_continuity_bundle_recent())
    probes.append(_probe_subia_integrity())

    # Roll up
    required_problems = [
        p for p in probes
        if p.required and p.status in ("MISSING", "STALE", "UNKNOWN")
    ]
    optional_problems = [
        p for p in probes
        if not p.required and p.status in ("MISSING", "STALE", "UNKNOWN")
    ]
    if required_problems:
        overall = "MISSING"
    elif optional_problems:
        overall = "DEGRADED"
    else:
        overall = "OK"

    return CloudReadiness(
        target=target,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall=overall,
        probes=probes,
    )


def format_readiness(r: CloudReadiness) -> str:
    """Human-readable summary for the CLI."""
    lines: list[str] = []
    lines.append(f"=== Cloud readiness — target={r.target} — overall: {r.overall} ===")
    lines.append("")
    glyph = {"OK": "✓", "MISSING": "✗", "STALE": "⚠", "UNKNOWN": "?"}
    name_max = max(len(p.name) for p in r.probes) if r.probes else 0
    for p in r.probes:
        req = "" if p.required else " (optional)"
        lines.append(
            f"  {glyph.get(p.status, '?')} {p.name:<{name_max}}  "
            f"{p.status:<8} {p.detail}{req}"
        )
    lines.append("")
    if r.overall == "OK":
        lines.append("  → Ready to run `botarmy migrate`.")
    elif r.overall == "MISSING":
        missing = [p.name for p in r.probes if p.required and p.status != "OK"]
        lines.append(f"  → Resolve required probes before migrate: {', '.join(missing)}")
    else:
        lines.append("  → All required probes pass. Optional probes have warnings.")
    return "\n".join(lines)
