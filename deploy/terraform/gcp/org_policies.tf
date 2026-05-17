# Org-policy constraints applied at the Workspace org level. Only run
# when var.org_id is provided AND hardening_profile=strict.
#
# Caveats:
#   * Caller must have ``roles/orgpolicy.policyAdmin`` on the org.
#   * Constraints applied here are inherited by every project under the
#     org. Set surgically; running these against an org with existing
#     workloads can break apps that rely on default behaviour.
#   * Each policy uses the v2 API (google_org_policy_policy) — older
#     google_organization_policy is deprecated.
#
# The constraints encoded here are the high-value "wish I'd had this on
# day one" set. Add more as your security review demands.

locals {
  org_policies_active = local.hardening_strict && var.org_id != ""
}

# 1. Block raw service-account key creation (use Workload Identity instead).
resource "google_org_policy_policy" "disable_sa_key_creation" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/iam.disableServiceAccountKeyCreation"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

resource "google_org_policy_policy" "disable_sa_key_upload" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/iam.disableServiceAccountKeyUpload"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# 2. Skip the default network in every new project — forces explicit VPC.
resource "google_org_policy_policy" "skip_default_network" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/compute.skipDefaultNetworkCreation"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# 3. Require OS Login (SSH via IAM, no project-level SSH keys).
resource "google_org_policy_policy" "require_os_login" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/compute.requireOsLogin"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# 4. Deny external IPs on VMs by default (Cloud NAT handles egress).
resource "google_org_policy_policy" "vm_external_ip_deny" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/compute.vmExternalIpAccess"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      deny_all = "TRUE"
    }
  }
}

# 5. CloudSQL: restrict public IPs (private-IP only).
resource "google_org_policy_policy" "restrict_cloudsql_public_ip" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/sql.restrictPublicIp"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# 6. GCS: enforce uniform bucket-level access (no ACLs).
resource "google_org_policy_policy" "uniform_bucket_access" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/storage.uniformBucketLevelAccess"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}

# 7. Restrict IAM grants to identities in your Workspace org (no public).
# Without this constraint, anyone with the project IAM Admin role could
# grant access to ``allUsers`` or ``allAuthenticatedUsers``.
resource "google_org_policy_policy" "iam_allowed_domains" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/iam.allowedPolicyMemberDomains"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      values {
        # Set via post-create operator action — placeholder values list.
        # The customer-ID is the C-prefix identifier shown in Workspace
        # Admin Console → Account settings. It's safer to leave this
        # empty (default-deny) than to hardcode a wrong value.
        allowed_values = []
      }
    }
  }

  # IMPORTANT: we deliberately do NOT seed allowed_values from terraform.
  # Operator runs:
  #   gcloud resource-manager org-policies allow \
  #     iam.allowedPolicyMemberDomains \
  #     C0xxxxxxxx \
  #     --organization=<org_id>
  # after the initial apply, once they confirm the customer ID. The
  # alternative — seeding an empty allow-list — would lock the project
  # out of every external integration on apply.
  lifecycle {
    ignore_changes = [spec[0].rules[0].values]
  }
}

# 8. Require Shielded VMs (defense against pre-boot tampering).
resource "google_org_policy_policy" "require_shielded_vm" {
  count  = local.org_policies_active ? 1 : 0
  name   = "organizations/${var.org_id}/policies/compute.requireShieldedVm"
  parent = "organizations/${var.org_id}"

  spec {
    rules {
      enforce = "TRUE"
    }
  }
}
