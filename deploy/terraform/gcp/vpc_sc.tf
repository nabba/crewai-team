# VPC Service Controls — perimeter around the project preventing
# data exfiltration through Google-managed APIs (storage, CloudSQL,
# Secret Manager, etc.).
#
# Only applied when:
#   * hardening_profile = "strict"
#   * vpc_sc_enabled = true (opt-in even at strict)
#   * var.org_id is set
#   * var.access_policy_id is set
#
# Defaults to DRY_RUN mode so the first apply is observational.
# Switching to enforced mode is operator-decided in /cp/settings
# after reviewing dry-run logs.
#
# VPC-SC has the same lock-out class as master_authorized_networks:
# get the access level wrong and you can't even read your own GCS
# bucket from a laptop outside the allow-list. Recovery procedure
# documented in docs/CLOUD_LOCKOUT_RECOVERY.md.

locals {
  vpc_sc_active = (
    local.hardening_strict
    && var.vpc_sc_enabled
    && var.org_id != ""
    && var.access_policy_id != ""
  )
}

# ─── Access level — who is allowed INTO the perimeter ─────────
# Each entry under conditions is OR-ed: if any matches, the request
# passes. We seed with the same CIDR list used for master_authorized_
# _networks (Tailnet + laptop public IP).
resource "google_access_context_manager_access_level" "operator" {
  count  = local.vpc_sc_active ? 1 : 0
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/accessLevels/${local.name}_operator"
  title  = "${local.name} operator access"

  basic {
    combining_function = "OR"

    conditions {
      ip_subnetworks = [for c in var.allowed_cidrs : c.cidr_block]
    }

    # OPTIONAL: tighten further with device policy (requires Endpoint
    # Verification on the operator's laptop). Disabled by default —
    # operator can flip on once Endpoint Verification is installed.
  }
}

# ─── The perimeter itself ─────────────────────────────────────
# We use the standalone (non-dry-run) resource even in dry-run mode;
# the `use_explicit_dry_run_spec` switch determines whether the
# perimeter is enforced or merely logs would-be-blocks. The dry-run
# spec is identical to the enforced spec — flipping
# var.vpc_sc_dry_run = false simply moves the same configuration
# from `spec` to `status`.
resource "google_access_context_manager_service_perimeter" "botarmy" {
  count  = local.vpc_sc_active ? 1 : 0
  parent = "accessPolicies/${var.access_policy_id}"
  name   = "accessPolicies/${var.access_policy_id}/servicePerimeters/${local.name}"
  title  = "${local.name} service perimeter"

  perimeter_type = "PERIMETER_TYPE_REGULAR"

  use_explicit_dry_run_spec = var.vpc_sc_dry_run

  # spec is used in dry-run mode (would-be-block logs only).
  dynamic "spec" {
    for_each = var.vpc_sc_dry_run ? [1] : []
    content {
      resources           = ["projects/${data.google_project.this.number}"]
      restricted_services = var.vpc_sc_restricted_services
      access_levels = [
        google_access_context_manager_access_level.operator[0].name,
      ]
      ingress_policies {
        ingress_from {
          identity_type = "ANY_USER_ACCOUNT"
          sources {
            access_level = google_access_context_manager_access_level.operator[0].name
          }
        }
        ingress_to {
          resources = ["*"]
          dynamic "operations" {
            for_each = var.vpc_sc_restricted_services
            content {
              service_name = operations.value
              method_selectors {
                method = "*"
              }
            }
          }
        }
      }
    }
  }

  # status is used in enforced mode. Identical to spec.
  dynamic "status" {
    for_each = var.vpc_sc_dry_run ? [] : [1]
    content {
      resources           = ["projects/${data.google_project.this.number}"]
      restricted_services = var.vpc_sc_restricted_services
      access_levels = [
        google_access_context_manager_access_level.operator[0].name,
      ]
      ingress_policies {
        ingress_from {
          identity_type = "ANY_USER_ACCOUNT"
          sources {
            access_level = google_access_context_manager_access_level.operator[0].name
          }
        }
        ingress_to {
          resources = ["*"]
          dynamic "operations" {
            for_each = var.vpc_sc_restricted_services
            content {
              service_name = operations.value
              method_selectors {
                method = "*"
              }
            }
          }
        }
      }
    }
  }

  lifecycle {
    create_before_destroy = false
  }
}

# Export to the hardening summary so cloud_doctor.verify_hardening
# can attest the perimeter is wired.
locals {
  vpc_sc_summary = {
    vpc_sc_enabled   = local.vpc_sc_active
    vpc_sc_dry_run   = var.vpc_sc_dry_run
    vpc_sc_resources = local.vpc_sc_active ? 1 : 0
  }
}
