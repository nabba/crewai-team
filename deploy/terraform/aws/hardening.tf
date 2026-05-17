# Hardening primitives applied across the module when hardening_profile
# is "basic" or "strict". Mirrors the GCP module's hardening.tf shape so
# the operator-facing surface is target-agnostic.
#
# Notes on AWS-specific differences from GCP:
#   * Shielded VMs equivalent → EC2 IMDSv2 enforcement (handled per-node
#     by the eks-managed-node-group module's default `metadata_options`).
#   * Workload Identity equivalent → IRSA (already wired in eks.tf).
#   * Cloud Armor equivalent → AWS WAFv2 (waf.tf).
#   * Binary Authorization equivalent → ECR image scanning + signing
#     (handled inline in ecr.tf — there's no AWS K8s admission webhook
#     for image-signing without a third-party operator).
#   * Org policies equivalent → AWS Organizations SCPs (scp.tf), only
#     when var.aws_org_enabled.

locals {
  hardening_active_aws = var.hardening_profile != "off"
  hardening_strict_aws = var.hardening_profile == "strict"

  # CIDRs the EKS public endpoint accepts when strict. Empty list passes
  # through to the EKS module's default (0.0.0.0/0 with IAM-gate).
  eks_public_access_cidrs = local.hardening_strict_aws ? [for c in var.allowed_cidrs : c.cidr_block] : []

  hardening_summary_aws = {
    profile                = var.hardening_profile
    cmek_enabled           = local.hardening_active_aws
    eks_endpoint_allowlist = length(local.eks_public_access_cidrs)
    waf_enabled            = local.hardening_strict_aws
    cloudtrail_enabled     = local.hardening_strict_aws
    guardduty_enabled      = local.hardening_strict_aws
    flow_logs_enabled      = local.hardening_active_aws
    scp_enabled            = local.hardening_strict_aws && var.aws_org_enabled
  }
}

# ─── GuardDuty + Security Hub (continuous detection) ────────────
# Both default-on for the account (us, not just region). $4-6/mo for
# personal-scale usage. Deliberately separate from CloudTrail so the
# operator can disable one without losing the audit trail.

resource "aws_guardduty_detector" "this" {
  count  = local.hardening_strict_aws ? 1 : 0
  enable = true

  datasources {
    s3_logs {
      enable = true
    }
  }

  tags = var.tags
}

resource "aws_securityhub_account" "this" {
  count = local.hardening_strict_aws ? 1 : 0

  # auto_enable_controls = true is recommended but expands the bill
  # ($0.0001/check × ~3000/mo); operator-flippable post-apply.
  auto_enable_controls = false
}
