# Hardening primitives applied across the module when hardening_profile
# is "basic" or "strict". Each section is independently flag-gated so a
# downgrade from strict→basic only removes the strict-only resources.
#
# Notes on what's NOT in this file (because Autopilot does it for us):
#   * Shielded GKE nodes — Autopilot enables them by default; we can't
#     turn them off. No terraform required.
#   * Network policy — Autopilot enables Dataplane V2 with network
#     policies enforced. No terraform required.
#   * Secure boot, integrity monitoring — Autopilot defaults.
#
# What IS in this file:
#   * Binary Authorization policy (strict only; mode toggleable AUDIT/ENFORCE)
#   * Master-authorized-networks helper (referenced from gke.tf)
#   * VPC firewall: deny all egress except documented allowlist (strict)

# ─── Binary Authorization ───────────────────────────────────────
# Cluster-level policy that gates pod admission on image attestations.
# Default mode = AUDIT (logs would-be-blocks; first deploy doesn't brick).
# Operator graduates to ENFORCE after wiring an attestor + signing pipeline.

resource "google_binary_authorization_policy" "botarmy" {
  count   = local.hardening_strict ? 1 : 0
  project = var.project_id

  default_admission_rule {
    # When ENFORCE mode is requested AND a cosign attestor has been
    # provisioned (by scripts/install/cosign_setup.sh), require
    # attestations from it. Otherwise fall back to ALWAYS_ALLOW —
    # ENFORCE without an attestor would brick every deploy.
    evaluation_mode = (
      var.binauthz_mode == "ENFORCE" && var.binauthz_attestor_name != ""
      ? "REQUIRE_ATTESTATION"
      : "ALWAYS_ALLOW"
    )
    enforcement_mode = var.binauthz_mode == "ENFORCE" ? "ENFORCED_BLOCK_AND_AUDIT_LOG" : "DRYRUN_AUDIT_LOG_ONLY"
    require_attestations_by = (
      var.binauthz_mode == "ENFORCE" && var.binauthz_attestor_name != ""
      ? ["projects/${var.project_id}/attestors/${var.binauthz_attestor_name}"]
      : []
    )
  }

  # Allowlist Google-published system images so kube-system / gke-system
  # pods (which the operator doesn't sign) keep running.
  admission_whitelist_patterns {
    name_pattern = "gcr.io/google_containers/*"
  }
  admission_whitelist_patterns {
    name_pattern = "gcr.io/google-containers/*"
  }
  admission_whitelist_patterns {
    name_pattern = "k8s.gcr.io/**"
  }
  admission_whitelist_patterns {
    name_pattern = "gke.gcr.io/**"
  }
  admission_whitelist_patterns {
    name_pattern = "registry.k8s.io/**"
  }
  admission_whitelist_patterns {
    name_pattern = "gcr.io/gke-release/**"
  }
  admission_whitelist_patterns {
    name_pattern = "gcr.io/stackdriver-agents/**"
  }
  # Allowlist our own Artifact Registry repo — the gateway image lives
  # here. When the signing pipeline lands we'll swap this for an
  # attestor-bound rule referencing a cluster_admission_rule.
  admission_whitelist_patterns {
    name_pattern = "${var.region}-docker.pkg.dev/${var.project_id}/${local.name}/**"
  }

  depends_on = [google_project_service.required]
}

# ─── Local for master-authorized-networks ────────────────────────
# Referenced from gke.tf's master_authorized_networks_config block.
locals {
  master_authorized_networks = local.hardening_strict ? var.allowed_cidrs : []
}

# ─── Deny-all egress (strict only, opt-in) ──────────────────────
# Disabled by default even at strict — locking down egress for
# Autopilot is tricky because Google's own services need to reach the
# cluster and Cloud NAT already constrains the egress surface.
# When operator wants tighter network policy they apply NetworkPolicy
# resources inside the cluster (handled in app/control_plane/k8s/, not
# here). This file deliberately does NOT create a deny-all firewall.

# ─── Outputs helper ────────────────────────────────────────────
locals {
  hardening_summary = {
    profile                    = var.hardening_profile
    cmek_enabled               = local.hardening_active
    binauthz_enabled           = local.hardening_strict
    binauthz_mode              = local.hardening_strict ? var.binauthz_mode : "n/a"
    binauthz_attestor_wired    = local.hardening_strict && var.binauthz_attestor_name != ""
    master_authorized_networks = length(local.master_authorized_networks)
    org_policies_enabled       = local.hardening_strict && var.org_id != ""
    audit_log_sink_enabled     = local.hardening_strict
    cloud_armor_enabled        = local.hardening_strict
    flow_logs_enabled          = local.hardening_active
    vpc_sc_enabled             = local.hardening_strict && var.vpc_sc_enabled && var.org_id != "" && var.access_policy_id != ""
    vpc_sc_dry_run             = var.vpc_sc_dry_run
  }
}
