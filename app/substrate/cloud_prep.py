"""
cloud_prep — auto-unblock the GCP identity↔project↔API tuple so that
``botarmy migrate --live`` can run without the operator typing the
3-command unblock sequence.

Productization plan WP D Phase 5a (2026-05-17). The 3 commands the
operator would otherwise type:

  1. ``gcloud config set account <user>``       — switches active gcloud
                                                  identity
  2. ``gcloud services enable <apis...>``        — turns on the APIs
                                                  terraform needs
  3. ``gcloud auth application-default login``   — populates ADC

This module automates #1 and #2 directly. For #3, it sidesteps ADC
entirely by minting a short-lived OAuth access token from the active
identity and exporting it as ``GOOGLE_OAUTH_ACCESS_TOKEN`` — terraform's
google provider checks that env var BEFORE falling back to ADC, so
terraform works without the interactive browser OAuth.

Key design choices:

  * **No interactive OAuth.** We CAN'T pop a browser from a React
    button click. The active-account+access-token approach avoids
    needing ADC at all.

  * **Operator picks the user.** We don't hard-code an account email.
    Callers pass ``active_account`` (presumably from a React dropdown
    of available gcloud-stored identities).

  * **Mint-fresh-tokens contract.** ``mint_terraform_env`` returns
    fresh env vars. The token expires in ~1h; long terraform applies
    re-mint mid-run isn't supported in this version — operator's
    apply must finish inside the token's lifetime (typical ~12-15min
    for GKE Autopilot, well inside the 1h cap).

  * **Subprocess execute-gate.** Same pattern as migration._shell:
    refuses real subprocess unless ``BOTARMY_MIGRATE_LIVE_EXECUTE=1``
    OR ``execute=True`` passed. Tests run with neither, getting
    deterministic dry-shell behavior.
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Same set the doctor checks. Single source of truth would be cleaner;
# importing from cloud_doctor here would create a cycle when the doctor
# wants to import cloud_prep helpers. We accept the small duplication.
REQUIRED_GCP_APIS = (
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "servicenetworking.googleapis.com",
    "compute.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
)


@dataclass(frozen=True)
class PrepStep:
    name: str
    status: str   # "ok" | "skipped" | "fail"
    detail: str
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrepResult:
    """Outcome of a full cloud-prep run. Steps are recorded in order,
    even on failure, so the operator can see exactly where it stopped.
    """
    active_account: str | None = None
    project_id: str | None = None
    steps: list[PrepStep] = field(default_factory=list)
    terraform_env: dict[str, str] = field(default_factory=dict)
    succeeded: bool = False
    fail_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_account": self.active_account,
            "project_id": self.project_id,
            "steps": [s.to_dict() for s in self.steps],
            # Redact the actual token from any external serialization.
            # The env dict is needed in-process for terraform but must
            # never appear in logs / reports / the React payload.
            "terraform_env_keys": sorted(self.terraform_env.keys()),
            "succeeded": self.succeeded,
            "fail_reason": self.fail_reason,
        }


# ── Subprocess plumbing (testable seam) ─────────────────────────────


def is_live_execute_enabled() -> bool:
    """Single source of truth for the cloud-migrate execute-gate.

    Consulted by every ``_shell`` in the substrate (migration, cloud_prep,
    cutover) AND by the migrate REST API's ``post_start`` handler.

    Resolution — EITHER source enables the gate (OR semantics):
      * ``BOTARMY_MIGRATE_LIVE_EXECUTE`` env var — legacy CLI workflow.
        Setting this still works (and overrides a False
        runtime_settings) so existing shell-scripted workflows + tests
        continue to function.
      * ``runtime_settings.migrate_live_execute`` — file-backed,
        React-toggleable, survives gateway restart, identity-ledger-
        aware. The new path.

    OR semantics keep the env-var override functional while letting
    React flip it on independently. To disable both: unset the env
    var AND set runtime_settings to False.
    """
    env_on = os.environ.get(
        "BOTARMY_MIGRATE_LIVE_EXECUTE", "",
    ) in ("1", "true", "yes")
    if env_on:
        return True
    try:
        from app.runtime_settings import get_migrate_live_execute
        return bool(get_migrate_live_execute())
    except Exception:
        return False


def _shell(
    argv: list[str], *, timeout: float, execute: bool = False,
) -> tuple[int, str, str]:
    """Same execute-gate pattern as ``migration._shell``.

    Returns (rc, stdout, stderr). Refuses to run unless ``execute=True``
    OR ``is_live_execute_enabled()``. Tests leave neither and get a
    deterministic ``<dry: cmd>`` placeholder.
    """
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


# ── Discovery helpers ───────────────────────────────────────────────


def list_authenticated_accounts() -> list[dict[str, str]]:
    """Return every gcloud-stored account with ``type`` (user|service_account)
    and ``status`` (active|inactive). Pure read.

    React dropdown sources from this. Operator picks which identity
    to migrate as. Always executes for real (read-only, no state change,
    no money spent) regardless of the BOTARMY_MIGRATE_LIVE_EXECUTE gate.
    """
    rc, out, err = _shell(
        ["gcloud", "auth", "list", "--format=value(account,status)"],
        timeout=15.0,
        execute=True,
    )
    if rc != 0:
        return []
    accounts: list[dict[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "<dry:" in line:
            continue
        parts = line.split()
        if not parts:
            continue
        account = parts[0]
        is_active = len(parts) > 1 and "*" in parts[1]
        acct_type = (
            "service_account"
            if account.endswith(".gserviceaccount.com")
            else "user"
        )
        accounts.append({
            "account": account,
            "type": acct_type,
            "active": "yes" if is_active else "no",
        })
    return accounts


# ── Step implementations ────────────────────────────────────────────


def _step_set_active_account(active_account: str) -> PrepStep:
    import time
    started = time.monotonic()
    if not active_account:
        return PrepStep(
            name="set_active_account",
            status="fail",
            detail="empty active_account argument",
            duration_s=round(time.monotonic() - started, 3),
        )
    rc, out, err = _shell(
        ["gcloud", "config", "set", "account", active_account],
        timeout=10.0,
    )
    duration = round(time.monotonic() - started, 3)
    if rc != 0:
        return PrepStep(
            name="set_active_account",
            status="fail",
            detail=f"exit {rc}: {(err or out)[:200]}",
            duration_s=duration,
        )
    return PrepStep(
        name="set_active_account",
        status="ok",
        detail=f"active gcloud account set to {active_account}",
        duration_s=duration,
    )


def _step_set_project(project_id: str) -> PrepStep:
    import time
    started = time.monotonic()
    rc, out, err = _shell(
        ["gcloud", "config", "set", "core/project", project_id],
        timeout=10.0,
    )
    duration = round(time.monotonic() - started, 3)
    if rc != 0:
        return PrepStep(
            name="set_project",
            status="fail",
            detail=f"exit {rc}: {(err or out)[:200]}",
            duration_s=duration,
        )
    return PrepStep(
        name="set_project",
        status="ok",
        detail=f"active project set to {project_id}",
        duration_s=duration,
    )


def _step_enable_apis(project_id: str, apis: tuple[str, ...]) -> PrepStep:
    """Enable any of the required APIs that aren't already enabled.

    Idempotent — gcloud services enable is a no-op for already-enabled
    APIs. We still call it for all of them in one batch because
    listing-then-enabling-the-missing-subset costs an extra round-trip.
    """
    import time
    started = time.monotonic()
    cmd = ["gcloud", "services", "enable", *apis, f"--project={project_id}"]
    # APIs can take a minute or two to propagate
    rc, out, err = _shell(cmd, timeout=180.0)
    duration = round(time.monotonic() - started, 3)
    if rc != 0:
        msg = (err or out).strip().splitlines()[0] if (err or out).strip() else "no output"
        # Common: PERMISSION_DENIED if the identity lacks
        # serviceusage.serviceUsageAdmin
        if "PERMISSION_DENIED" in msg or "403" in msg:
            return PrepStep(
                name="enable_apis",
                status="fail",
                detail=(
                    f"permission denied — active identity lacks "
                    f"roles/serviceusage.serviceUsageAdmin on {project_id}. "
                    f"Switch to an account with Owner/Editor + try again."
                ),
                duration_s=duration,
            )
        return PrepStep(
            name="enable_apis",
            status="fail",
            detail=f"exit {rc}: {msg[:200]}",
            duration_s=duration,
        )
    return PrepStep(
        name="enable_apis",
        status="ok",
        detail=f"{len(apis)} APIs enabled (or already on)",
        duration_s=duration,
    )


def _step_mint_terraform_token() -> tuple[PrepStep, str]:
    """Mint a short-lived OAuth access token for terraform.

    Returns (PrepStep, token). The token is also stored in PrepResult
    so the caller's terraform_env dict can carry it.
    """
    import time
    started = time.monotonic()
    rc, out, err = _shell(
        ["gcloud", "auth", "print-access-token"],
        timeout=15.0,
    )
    duration = round(time.monotonic() - started, 3)
    if rc != 0:
        return (
            PrepStep(
                name="mint_terraform_token",
                status="fail",
                detail=f"exit {rc}: {(err or out)[:200]}",
                duration_s=duration,
            ),
            "",
        )
    token = out.strip().splitlines()[0] if out.strip() else ""
    if not token or token.startswith("<dry:"):
        # Dry-shell mode returns the placeholder; treat as ok-but-no-token.
        return (
            PrepStep(
                name="mint_terraform_token",
                status="skipped",
                detail="dry-shell mode — no real token minted",
                duration_s=duration,
            ),
            token,
        )
    return (
        PrepStep(
            name="mint_terraform_token",
            status="ok",
            detail=f"minted access token (length {len(token)}, ~1h validity)",
            duration_s=duration,
        ),
        token,
    )


# ── Public entry ────────────────────────────────────────────────────


def prepare_gcp_for_migrate(
    *,
    active_account: str,
    project_id: str,
    apis: tuple[str, ...] = REQUIRED_GCP_APIS,
) -> PrepResult:
    """Run the full 3-command unblock sequence programmatically.

    Order matters:
      1. set_active_account — every subsequent gcloud call inherits this
      2. set_project        — every subsequent gcloud call inherits this
      3. enable_apis        — needs the active account to have admin
      4. mint_terraform_token — replaces the need for ADC OAuth

    On failure, returns a PrepResult with succeeded=False, fail_reason
    set, and partial steps recorded. Caller decides whether to
    surface to React or retry with a different account.
    """
    result = PrepResult(active_account=active_account, project_id=project_id)

    step = _step_set_active_account(active_account)
    result.steps.append(step)
    if step.status == "fail":
        result.fail_reason = f"set_active_account: {step.detail}"
        return result

    step = _step_set_project(project_id)
    result.steps.append(step)
    if step.status == "fail":
        result.fail_reason = f"set_project: {step.detail}"
        return result

    step = _step_enable_apis(project_id, apis)
    result.steps.append(step)
    if step.status == "fail":
        result.fail_reason = f"enable_apis: {step.detail}"
        return result

    mint_step, token = _step_mint_terraform_token()
    result.steps.append(mint_step)
    if mint_step.status == "fail":
        result.fail_reason = f"mint_terraform_token: {mint_step.detail}"
        return result

    # Build the terraform env. ``GOOGLE_OAUTH_ACCESS_TOKEN`` is what
    # terraform's google provider checks before ADC fallback.
    if token and not token.startswith("<dry:"):
        result.terraform_env = {
            "GOOGLE_OAUTH_ACCESS_TOKEN": token,
            # Same project for redundancy — some terraform versions
            # respect this var directly.
            "GOOGLE_PROJECT": project_id,
            "GOOGLE_REGION": "europe-north1",  # default, callers can override
        }
    result.succeeded = True
    return result
