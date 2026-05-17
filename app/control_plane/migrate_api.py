"""Control-plane REST API for React-driven cloud migration.

Productization plan WP D Phase 5a (2026-05-17). Endpoints under
``/api/cp/migrate/*`` that map 1:1 to the React wizard steps:

  GET  /api/cp/migrate/accounts        → list gcloud-stored accounts
  GET  /api/cp/migrate/preflight       → cloud_doctor probe rollup
  POST /api/cp/migrate/cost            → cost estimate for the requested config
  POST /api/cp/migrate/start           → kick off async migration
  GET  /api/cp/migrate/runs            → list recent runs
  GET  /api/cp/migrate/runs/<run_id>   → poll one run's status
  POST /api/cp/migrate/runs/<run_id>/cancel → cooperative cancel

Auth: inherited via the dashboard_api parent router's
``require_gateway_auth`` dependency. In K8s production, every endpoint
needs ``Authorization: Bearer <gateway-secret>``.

Safety gates ENFORCED AT THE API LEVEL (since the CLI's env-var +
typed-phrase gates can't fire from a button click):

  * ``confirm_phrase`` must equal exactly ``"MIGRATE TO GCP"`` in
    /start request body (mirrors the CLI's --confirm flag).
  * ``budget_cap_usd`` is REQUIRED (no default) — operator must
    explicitly state a spend ceiling.
  * Hard ceiling: ``budget_cap_usd <= 500`` (caught here to prevent
    typo-induced four-figure spends).
  * In-flight migration check — start refuses if another run is active.
  * Migration only runs when ``BOTARMY_MIGRATE_LIVE_EXECUTE=1`` is set
    on the gateway process (the same env-var guard the CLI uses).
    Without it, /start runs the orchestrator but every subprocess
    returns a ``<dry: ...>`` placeholder.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.control_plane.auth_dep import require_gateway_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cp/migrate",
    tags=["control-plane", "migrate"],
    dependencies=[Depends(require_gateway_auth)],
)


# ── Request models ─────────────────────────────────────────────────


class _CostBody(BaseModel):
    target: str = Field(default="gcp", pattern="^(gcp|aws)$")
    tier: str = Field(default="cheapest", pattern="^(cheapest|prod)$")
    region: str | None = None
    enable_monitoring: bool = True
    has_domain: bool = False


class _StartBody(BaseModel):
    target: str = Field(default="gcp", pattern="^(gcp|aws)$")
    tier: str = Field(default="cheapest", pattern="^(cheapest|prod)$")
    region: str | None = None
    project_id: str = Field(..., min_length=1)
    active_account: str = Field(..., min_length=1)
    confirm_phrase: str = Field(..., min_length=1)
    # Hard ceiling of $500 is the API-level safety net. The orchestrator
    # also enforces the per-tier estimate ≤ budget_cap.
    budget_cap_usd: float = Field(..., gt=0, le=500)


# ── Endpoints ──────────────────────────────────────────────────────


@router.get("/accounts")
def list_accounts() -> dict[str, Any]:
    """List gcloud-stored accounts the operator can migrate as.

    React renders this as a dropdown on the wizard's first page.
    Pure read — no state change.
    """
    try:
        from app.substrate.cloud_prep import list_authenticated_accounts
        return {"accounts": list_authenticated_accounts()}
    except Exception as exc:
        logger.exception("migrate_api: list_accounts crashed")
        raise HTTPException(500, f"list_accounts: {type(exc).__name__}: {exc}")


@router.get("/preflight")
def get_preflight(target: str = Query("gcp", pattern="^(gcp|aws)$")) -> dict[str, Any]:
    """Run cloud_doctor and return its rollup.

    Composed with the rest of the wizard: React displays each probe,
    shows the operator-actionable details for any MISSING/STALE checks,
    and disables the 'Start migrate' button until overall=OK.
    """
    try:
        from app.substrate.cloud_doctor import check_readiness
        readiness = check_readiness(target=target)
        return readiness.to_dict()
    except Exception as exc:
        logger.exception("migrate_api: preflight crashed")
        raise HTTPException(500, f"preflight: {type(exc).__name__}: {exc}")


@router.post("/cost")
def post_cost(body: _CostBody) -> dict[str, Any]:
    """Itemized monthly cost estimate for the requested config.

    React shows this on the cost-review wizard page. The operator
    sees the same number that ``botarmy cost --target gcp`` prints
    on the CLI.
    """
    try:
        from app.substrate.cloud_cost import estimate_monthly_cost
        breakdown = estimate_monthly_cost(
            target=body.target,
            tier=body.tier,
            region=body.region,
            enable_monitoring=body.enable_monitoring,
            has_domain=body.has_domain,
        )
        return breakdown.to_dict()
    except Exception as exc:
        logger.exception("migrate_api: cost crashed")
        raise HTTPException(500, f"cost: {type(exc).__name__}: {exc}")


@router.post("/start", status_code=202)
def post_start(body: _StartBody) -> dict[str, Any]:
    """Kick off an async migration. Returns 202 Accepted + run_id.

    React polls /runs/<run_id> for progress.

    Refuses with:
      * 400 — confirm_phrase mismatch
      * 409 — another migration is already in flight
      * 422 — pydantic validation (budget over $500, etc.)
    """
    # Typed-phrase API-level gate. The orchestrator ALSO enforces this
    # (it's the same string compared in evaluate_live_gates), but
    # we surface it as 400 here for a cleaner React error path.
    from app.substrate.migration import TYPED_PHRASE_REQUIRED as _PHRASE
    expected = _PHRASE if body.target == "gcp" else f"MIGRATE TO {body.target.upper()}"
    if body.confirm_phrase != expected:
        raise HTTPException(
            400,
            f"confirm_phrase must equal exactly {expected!r}",
        )

    # Execute-gate visibility. We DON'T 503 if it's missing — we let
    # the migration run in dry-shell mode which produces a useful
    # report without spending money. React reads this back as
    # ``dry_shell_mode`` in the response so the wizard can show the
    # yellow banner.
    #
    # Resolution: runtime_settings.migrate_live_execute (React-toggleable)
    # → BOTARMY_MIGRATE_LIVE_EXECUTE env var (legacy fallback). The
    # single resolver lives in cloud_prep.is_live_execute_enabled.
    from app.substrate.cloud_prep import is_live_execute_enabled
    execute_subprocess = is_live_execute_enabled()

    try:
        from app.substrate.migration_runner import (
            start_async_migration, RunnerBusyError,
        )
        record = start_async_migration(
            target=body.target,
            tier=body.tier,
            region=body.region,
            project_id=body.project_id,
            active_account=body.active_account,
            confirm_phrase=body.confirm_phrase,
            budget_cap_usd=body.budget_cap_usd,
            execute_subprocess=execute_subprocess,
        )
        return {
            "run_id": record.run_id,
            "status": record.status,
            "execute_subprocess": execute_subprocess,
            # Friendly hint for the React UI: if execute is off, the
            # operator will get a clean report but no real cluster.
            "dry_shell_mode": not execute_subprocess,
        }
    except RunnerBusyError as exc:
        raise HTTPException(409, str(exc))
    except Exception as exc:
        logger.exception("migrate_api: start crashed")
        raise HTTPException(500, f"start: {type(exc).__name__}: {exc}")


@router.get("/runs")
def get_runs(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """List recent migration runs. Newest first."""
    try:
        from app.substrate.migration_runner import list_recent_runs, active_run_id
        recs = list_recent_runs(limit=limit)
        return {
            "active_run_id": active_run_id(),
            "runs": [r.to_dict() for r in recs],
        }
    except Exception as exc:
        logger.exception("migrate_api: runs crashed")
        raise HTTPException(500, f"runs: {type(exc).__name__}: {exc}")


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """Poll a single run's current state. React calls this every 2-5s
    during a migration to render the progress bar + current step."""
    try:
        from app.substrate.migration_runner import load_run_record
        record = load_run_record(run_id)
        if record is None:
            raise HTTPException(404, f"no run found with id={run_id!r}")
        return record.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("migrate_api: get_run crashed")
        raise HTTPException(500, f"get_run: {type(exc).__name__}: {exc}")


@router.get("/hardening-preview")
def get_hardening_preview(
    profile: str = Query("strict", pattern="^(off|basic|strict)$"),
    binauthz_mode: str = Query("AUDIT", pattern="^(AUDIT|ENFORCE)$"),
) -> dict[str, Any]:
    """One-shot detection for the Step 3.5 Hardening card.

    Returns:
      * ``tailnet_reachable`` / ``tailnet_cidr`` — auto-detected
      * ``laptop_public_ip`` — auto-detected via HTTPS probe
      * ``org_id`` — auto-detected via ``gcloud organizations list``
      * ``recommended_cidrs`` — composed allowlist (Tailnet + laptop /32)
      * ``notes`` — operator-visible warnings (e.g. "Tailnet not detected,
        run `botarmy hardening refresh-allowed-cidrs` if your IP changes")

    Pure read. Detection failures degrade silently — caller never sees an
    exception, just empty fields + an explanatory note.
    """
    try:
        from app.substrate.cloud_hardening import hardening_preview
        return hardening_preview(profile=profile, binauthz_mode=binauthz_mode).to_dict()
    except Exception as exc:
        logger.exception("migrate_api: hardening_preview crashed")
        raise HTTPException(500, f"hardening_preview: {type(exc).__name__}: {exc}")


class _BootstrapBody(BaseModel):
    project_id: str = Field(..., description="GCP project ID to create")
    billing_account: str = Field(..., description="XXXXXX-XXXXXX-XXXXXX")
    org_id: str | None = Field(None, description="Numeric org ID; None puts project under no-org")
    project_name: str | None = Field(None, description="Human-readable project name")
    confirm_phrase: str = Field(
        ..., description="Must equal 'CREATE GCP PROJECT' to authorize the create",
    )
    dry_run: bool = Field(False, description="Print actions instead of executing")


@router.post("/bootstrap-project", dependencies=[Depends(require_gateway_auth)])
def post_bootstrap_project(body: _BootstrapBody) -> dict[str, Any]:
    """Stage 0a — run ``scripts/install/gcp_bootstrap.sh`` to create the
    project + link billing + enable APIs.

    Three-layer safety:
      1. ``runtime_settings.gcp_bootstrap_enabled`` must be True
      2. Typed-phrase ``confirm_phrase == 'CREATE GCP PROJECT'``
      3. Bearer-auth on the route (matches the rest of /api/cp/migrate)

    Idempotent — the shell script short-circuits if the project already
    exists and is ACTIVE.
    """
    # Layer 1: master switch
    try:
        from app.runtime_settings import get_gcp_bootstrap_enabled
        if not get_gcp_bootstrap_enabled():
            raise HTTPException(
                403,
                "gcp_bootstrap_enabled is OFF — flip it in /cp/settings "
                "(typed-phrase confirmation required)",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("migrate_api: bootstrap_project runtime_settings read failed")
        raise HTTPException(500, f"bootstrap: {type(exc).__name__}: {exc}")

    # Layer 2: typed phrase
    if body.confirm_phrase != "CREATE GCP PROJECT":
        raise HTTPException(
            400, "confirm_phrase must be 'CREATE GCP PROJECT' (verbatim)",
        )

    # Run the shell script — subprocess.run() so output streams to the API
    # response. Use 5-minute timeout (enabling 16 APIs first-time runs ~3 min).
    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "install" / "gcp_bootstrap.sh"
    if not script.is_file():
        raise HTTPException(500, f"bootstrap script not found at {script}")
    argv = [
        "bash", str(script),
        "--project-id", body.project_id,
        "--billing-account", body.billing_account,
        "--confirm", "CREATE GCP PROJECT",
    ]
    if body.org_id:
        argv += ["--org-id", body.org_id]
    if body.project_name:
        argv += ["--project-name", body.project_name]
    if body.dry_run:
        argv.append("--dry-run")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300.0)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "bootstrap script timed out after 5 minutes")
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
        "dry_run": body.dry_run,
        "ok": proc.returncode == 0,
    }


class _AwsBootstrapBody(BaseModel):
    email: str = Field(..., description="Root email for the new AWS account (must be unique)")
    account_name: str | None = Field(None, description="Human-readable name; defaults to local part of email")
    role_name: str = Field("OrganizationAccountAccessRole", description="IAM role created in the new account")
    org_unit_id: str | None = Field(None, description="Destination OU id (e.g. ou-abcd-12345678); empty = org root")
    confirm_phrase: str = Field(
        ..., description="Must equal 'CREATE AWS ACCOUNT' to authorize the create",
    )
    dry_run: bool = Field(False, description="Print actions instead of executing")


@router.post("/bootstrap-aws-account", dependencies=[Depends(require_gateway_auth)])
def post_bootstrap_aws_account(body: _AwsBootstrapBody) -> dict[str, Any]:
    """Stage 0a for AWS — create an Organizations member account.

    Three-layer safety identical to ``post_bootstrap_project``:
      1. ``runtime_settings.aws_bootstrap_enabled`` must be True
      2. Typed-phrase ``confirm_phrase == 'CREATE AWS ACCOUNT'``
      3. Bearer-auth on the route

    Caller must be running against the Organizations MANAGEMENT account
    — the script refuses if invoked from a member account.
    """
    try:
        from app.runtime_settings import get_aws_bootstrap_enabled
        if not get_aws_bootstrap_enabled():
            raise HTTPException(
                403,
                "aws_bootstrap_enabled is OFF — flip it in /cp/settings "
                "(typed-phrase confirmation required)",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("migrate_api: bootstrap_aws runtime_settings read failed")
        raise HTTPException(500, f"bootstrap_aws: {type(exc).__name__}: {exc}")

    if body.confirm_phrase != "CREATE AWS ACCOUNT":
        raise HTTPException(
            400, "confirm_phrase must be 'CREATE AWS ACCOUNT' (verbatim)",
        )

    import subprocess
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "install" / "aws_bootstrap.sh"
    if not script.is_file():
        raise HTTPException(500, f"bootstrap script not found at {script}")
    argv = [
        "bash", str(script),
        "--email", body.email,
        "--confirm", "CREATE AWS ACCOUNT",
        "--role-name", body.role_name,
    ]
    if body.account_name:
        argv += ["--account-name", body.account_name]
    if body.org_unit_id:
        argv += ["--org-unit-id", body.org_unit_id]
    if body.dry_run:
        argv.append("--dry-run")
    try:
        # Wait up to 7 min — account-create polling is up to 5 min plus
        # buffer for OU move + the script's own setup work.
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=420.0)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "bootstrap script timed out after 7 minutes")
    # Last line of stdout is the new account id (or existing id on idempotent path).
    new_account_id = ""
    if proc.returncode == 0 and proc.stdout.strip():
        last_line = proc.stdout.strip().splitlines()[-1]
        if last_line.isdigit():
            new_account_id = last_line
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-2000:],
        "dry_run": body.dry_run,
        "ok": proc.returncode == 0,
        "new_account_id": new_account_id,
    }


@router.post("/runs/{run_id}/cancel")
def post_cancel(run_id: str) -> dict[str, Any]:
    """Request cooperative cancellation of an in-flight run.

    Cancel is checked between pipeline steps; a mid-step terraform
    apply will continue to completion (~15 min) before the cancel
    takes effect. For hard abort, operator runs ``terraform destroy``
    out-of-band.
    """
    try:
        from app.substrate.migration_runner import cancel_active, active_run_id
        current = active_run_id()
        if current is None:
            raise HTTPException(409, "no migration is currently in flight")
        if current != run_id:
            raise HTTPException(
                409,
                f"run {run_id!r} is not the active run (active: {current!r})",
            )
        cancel_active()
        return {"cancel_requested": True, "active_run_id": current}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("migrate_api: cancel crashed")
        raise HTTPException(500, f"cancel: {type(exc).__name__}: {exc}")
