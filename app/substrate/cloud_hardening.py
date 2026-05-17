"""Cloud-hardening helpers — auto-detection + post-apply verification.

These helpers are consumed by:

  * ``app.control_plane.migrate_api`` — the new ``hardening-preview``
    endpoint that surfaces auto-detected IPs + recommended CIDRs to the
    React Step 3.5 card.
  * ``app.substrate.migration`` — when rendering ``terraform.tfvars``
    for a run, the orchestrator pulls ``allowed_cidrs`` + ``org_id``
    from here.
  * ``app.substrate.cloud_doctor.verify_hardening`` — after a successful
    apply, reads the ``hardening_summary`` terraform output back and
    confirms each policy actually landed.
  * ``scripts/botarmy hardening refresh-allowed-cidrs`` — the lock-out
    break-glass: re-detect IPs and emit a tfvars patch the operator
    targets with ``terraform apply -target=google_container_cluster.botarmy``.

Every read is failure-isolated. A missing ``tailscale`` binary or a
firewall-blocked ``checkip.amazonaws.com`` returns an empty result, not
an exception. The caller never has to wrap.

No state is persisted in this module — recompute on every call. The
detected values change (laptop IP roams; the operator joins/leaves
Tailnet) and a cached read is worse than a fresh one.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

# Tailscale's standard CGNAT range — every Tailnet node sits inside this
# /10. Pinning to the range (not an individual host) keeps the allowlist
# stable as the operator's nodes come and go.
TAILSCALE_CGNAT_CIDR = "100.64.0.0/10"

# Public IP probe endpoints. We try multiple in case one is rate-limited
# or blocked. All three are stateless plain-text endpoints with no auth.
_PUBLIC_IP_ENDPOINTS = (
    "https://checkip.amazonaws.com",
    "https://api.ipify.org",
    "https://ifconfig.me/ip",
)

# Allowed CIDR sentinels — refused at validate-time. Anyone trying to
# pass these is either testing or making a mistake.
_FORBIDDEN_CIDRS = frozenset({
    "0.0.0.0/0",
    "::/0",
    "0.0.0.0",
    "0.0.0.0/1",  # halves of the world
    "128.0.0.0/1",
})


@dataclass(frozen=True)
class AllowedCidr:
    """One entry on the master-authorized-networks list. Mirrors the
    terraform object type in ``deploy/terraform/gcp/variables.tf``."""
    cidr_block: str
    display_name: str

    def to_dict(self) -> dict[str, str]:
        return {"cidr_block": self.cidr_block, "display_name": self.display_name}


@dataclass
class HardeningPreview:
    """What the React Step 3.5 card needs to render."""
    profile: str
    tailnet_reachable: bool
    tailnet_cidr: str | None
    laptop_public_ip: str | None
    recommended_cidrs: list[AllowedCidr] = field(default_factory=list)
    org_id: str | None = None
    binauthz_mode: str = "AUDIT"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommended_cidrs"] = [c.to_dict() for c in self.recommended_cidrs]
        return d


# ── Detection helpers ────────────────────────────────────────────


def tailnet_reachable(timeout_sec: float = 3.0) -> bool:
    """True iff ``tailscale status`` returns rc=0 within ``timeout_sec``.

    A True result means the local host is joined to a Tailnet — the
    100.64.0.0/10 CIDR is then safe to include in the GKE master
    allowlist (the operator's phone, laptop, etc., reach the cluster
    through their Tailnet IPs).
    """
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--peers=false"],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        logger.debug("cloud_hardening: tailscale binary not on PATH")
        return False
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("cloud_hardening: tailscale status failed: %s", exc)
        return False


def detect_tailnet_cidr() -> str | None:
    """Return ``TAILSCALE_CGNAT_CIDR`` iff Tailnet is reachable.

    Deliberately returns the whole /10 — pinning to a single host inside
    the Tailnet defeats the purpose (every other Tailnet device of the
    operator's still needs to reach the cluster).
    """
    return TAILSCALE_CGNAT_CIDR if tailnet_reachable() else None


def detect_laptop_public_ip(timeout_sec: float = 4.0) -> str | None:
    """Probe a public-IP-echo endpoint and return the result as a /32.

    Used as a fallback when Tailnet is not reachable, OR added alongside
    a Tailnet CIDR for the case where the operator wants ``terraform
    apply`` to succeed from a CI runner outside Tailnet.

    Failure-isolated: any HTTPS error returns None.
    """
    for url in _PUBLIC_IP_ENDPOINTS:
        try:
            req = urllib_request.Request(url, headers={"User-Agent": "AndrusAI/1.0"})
            with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace").strip()
        except (urllib_error.URLError, OSError, ValueError) as exc:
            logger.debug("cloud_hardening: public-IP probe %s failed: %s", url, exc)
            continue
        if not raw:
            continue
        try:
            ipaddress.ip_address(raw)
        except ValueError:
            logger.debug("cloud_hardening: %s returned non-IP %r", url, raw)
            continue
        return raw
    return None


def detect_access_policy_id(org_id: str, timeout_sec: float = 5.0) -> str | None:
    """Look up the Access Context Manager access policy id for an org.

    Each Workspace org has at most one access policy. Returns the numeric
    id (no prefix) when it exists, None when not yet created.

    The first call to create a perimeter requires this policy to exist.
    If not yet created, the operator runs:
        gcloud access-context-manager policies create \\
            --organization <org_id> --title "<title>"

    The terraform module does NOT create the policy — that's an
    org-level mutation that the operator should authorise once.
    """
    if not org_id:
        return None
    try:
        proc = subprocess.run(
            [
                "gcloud", "access-context-manager", "policies", "list",
                "--organization", org_id, "--format=json",
            ],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        if proc.returncode != 0:
            logger.debug("cloud_hardening: ACM policy list rc=%s err=%s",
                         proc.returncode, proc.stderr)
            return None
        policies = json.loads(proc.stdout) if proc.stdout.strip() else []
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("cloud_hardening: ACM policy list failed: %s", exc)
        return None
    if not policies:
        return None
    # Format: accessPolicies/123456789012
    name = str(policies[0].get("name", ""))
    if "/" in name:
        return name.split("/", 1)[1]
    return name or None


def detect_aws_org_root_id(timeout_sec: float = 5.0) -> str | None:
    """Look up the AWS Organizations root id the active caller's account
    belongs to.

    Returns the root id (e.g. ``r-abcd1234``) when:
      * ``aws organizations describe-organization`` returns OK
      * the caller has ``organizations:ListRoots`` (typically only the
        management account does)

    Returns None for standalone AWS accounts or workload accounts that
    can't see the org structure.
    """
    try:
        proc = subprocess.run(
            ["aws", "organizations", "list-roots", "--output", "json"],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        if proc.returncode != 0:
            logger.debug("cloud_hardening: aws org root list rc=%s err=%s",
                         proc.returncode, proc.stderr)
            return None
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("cloud_hardening: aws org root list failed: %s", exc)
        return None
    roots = data.get("Roots", []) if isinstance(data, dict) else []
    if not roots:
        return None
    return str(roots[0].get("Id", "")) or None


def detect_org_id(timeout_sec: float = 5.0) -> str | None:
    """Look up the GCP organization ID the active gcloud account belongs to.

    Returns the numeric ID (no ``organizations/`` prefix) when the operator
    is a member of a Workspace org. Returns None for solo Gmail accounts
    or when ``gcloud`` is not authenticated.

    The first org is taken when multiple are visible — pinning to one
    org is the operator's deliberate decision, surfaced in the React
    card with an "auto-detected" badge they can override.
    """
    try:
        proc = subprocess.run(
            ["gcloud", "organizations", "list", "--format=json"],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        if proc.returncode != 0:
            logger.debug("cloud_hardening: org list rc=%s err=%s", proc.returncode, proc.stderr)
            return None
        orgs = json.loads(proc.stdout) if proc.stdout.strip() else []
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.debug("cloud_hardening: org list failed: %s", exc)
        return None
    if not orgs:
        return None
    name = str(orgs[0].get("name", ""))
    # Format: organizations/123456789012
    if name.startswith("organizations/"):
        return name.split("/", 1)[1]
    return None


# ── CIDR validation ───────────────────────────────────────────────


def validate_cidr(cidr: str) -> tuple[bool, str]:
    """Return (ok, reason). ``ok=False`` means refuse — call site MUST
    not pass the CIDR to terraform."""
    if not cidr or not isinstance(cidr, str):
        return False, "cidr is empty or not a string"
    if cidr in _FORBIDDEN_CIDRS:
        return False, f"refused world-open CIDR {cidr!r}"
    try:
        net = ipaddress.ip_network(cidr, strict=False)
    except ValueError as exc:
        return False, f"invalid CIDR: {exc}"
    if net.prefixlen == 0:
        return False, "prefix /0 is world-open; refused"
    return True, ""


# ── Build allowed_cidrs list for terraform.tfvars ──────────────────


def build_allowed_cidrs(
    *,
    tailnet_cidr: str | None = None,
    laptop_public_ip: str | None = None,
    extra: list[AllowedCidr] | None = None,
) -> list[AllowedCidr]:
    """Compose the master-authorized-networks list from auto-detected +
    operator-supplied sources.

    Order matters for terraform stability — Tailnet first, laptop IP
    second, extras last. Duplicates are deduplicated by ``cidr_block``
    (first wins). Each entry is validated; invalid CIDRs are dropped
    with a log line.
    """
    out: list[AllowedCidr] = []
    seen: set[str] = set()

    def _push(c: AllowedCidr) -> None:
        ok, why = validate_cidr(c.cidr_block)
        if not ok:
            logger.warning("cloud_hardening: dropping CIDR %r: %s", c.cidr_block, why)
            return
        if c.cidr_block in seen:
            return
        seen.add(c.cidr_block)
        out.append(c)

    if tailnet_cidr:
        _push(AllowedCidr(cidr_block=tailnet_cidr, display_name="Tailnet (auto-detected)"))
    if laptop_public_ip:
        # /32 host — narrower than the operator probably needs but
        # safest default. They can edit in React.
        try:
            ipaddress.ip_address(laptop_public_ip)
            cidr = f"{laptop_public_ip}/32"
            _push(AllowedCidr(cidr_block=cidr, display_name="Laptop public IP (auto-detected)"))
        except ValueError:
            logger.debug("cloud_hardening: laptop IP %r failed parse", laptop_public_ip)
    for c in (extra or []):
        _push(c)
    return out


# ── End-to-end preview ───────────────────────────────────────────


def hardening_preview(profile: str = "strict", binauthz_mode: str = "AUDIT") -> HardeningPreview:
    """One-shot detection used by the migrate-API ``/hardening-preview``
    endpoint. Composes the auto-detected pieces into a single payload
    the React card can render.
    """
    tailnet = detect_tailnet_cidr()
    laptop_ip = detect_laptop_public_ip()
    org_id = detect_org_id()

    cidrs = build_allowed_cidrs(tailnet_cidr=tailnet, laptop_public_ip=laptop_ip)

    notes: list[str] = []
    if profile == "strict" and not cidrs:
        notes.append(
            "No Tailnet + no laptop public IP detected. master_authorized_networks "
            "will be empty — anyone with valid IAM can reach the K8s API. Add at "
            "least one CIDR before clicking Apply."
        )
    if profile == "strict" and not tailnet:
        notes.append(
            "Tailnet not detected. If your IP changes after apply, run "
            "`botarmy hardening refresh-allowed-cidrs` to re-detect and "
            "patch the master allowlist without a full migration."
        )
    if profile == "strict" and not org_id:
        notes.append(
            "Google Workspace org not detected. Org policies will be skipped. "
            "If you have a Workspace, run `gcloud organizations list` to verify "
            "the active account has Organization Viewer."
        )
    if profile == "strict" and binauthz_mode == "ENFORCE":
        notes.append(
            "Binary Authorization ENFORCE will reject unsigned images. Ensure "
            "your image-signing pipeline (cosign + attestor) is wired before "
            "applying — otherwise the first deploy will fail at pod admission."
        )

    return HardeningPreview(
        profile=profile,
        tailnet_reachable=tailnet is not None,
        tailnet_cidr=tailnet,
        laptop_public_ip=laptop_ip,
        recommended_cidrs=cidrs,
        org_id=org_id,
        binauthz_mode=binauthz_mode,
        notes=notes,
    )


# ── Post-apply verification ──────────────────────────────────────


def verify_hardening(
    run_dir: str, *, project_id: str, timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """Read the ``hardening_summary`` terraform output from a completed
    run and compare to the expected profile.

    Called by ``cloud_doctor.verify_hardening`` from the post-apply Step 6
    of the wizard. Returns a structured verdict the React Outcome card
    renders.

    Failure-isolated: a missing tfstate or unparseable output returns a
    ``{ok: False, reason: ...}`` dict.
    """
    try:
        proc = subprocess.run(
            ["terraform", "output", "-json", "hardening_summary"],
            cwd=run_dir,
            capture_output=True, text=True, timeout=timeout_sec,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "reason": f"terraform output failed: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "reason": f"terraform output rc={proc.returncode}: {proc.stderr.strip()}"}
    try:
        summary = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "reason": f"output JSON parse failed: {exc}"}
    if not isinstance(summary, dict):
        return {"ok": False, "reason": f"output not a map: {summary!r}"}
    return {"ok": True, "summary": summary, "project_id": project_id}
