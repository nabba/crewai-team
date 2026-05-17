# Service Control Policies — AWS Organizations equivalent of GCP org
# policies. Only created when both aws_org_enabled AND hardening_profile=strict.
#
# IMPORTANT: SCPs are attached at the Organizations level, NOT the member
# account. The terraform must run against management-account credentials.
# Most operators run terraform from the workload account — in that case
# this file is a no-op and the operator applies SCPs separately via the
# management account.

locals {
  scp_active = local.hardening_strict_aws && var.aws_org_enabled && var.aws_org_root_id != ""
}

# 1. Deny use of root account (defense against credential leak).
resource "aws_organizations_policy" "deny_root_actions" {
  count = local.scp_active ? 1 : 0
  name  = "${local.name}-deny-root-actions"
  type  = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "DenyRootActions"
      Effect   = "Deny"
      Action   = "*"
      Resource = "*"
      Condition = {
        StringLike = {
          "aws:PrincipalArn" = "arn:aws:iam::*:root"
        }
      }
    }]
  })
}

# 2. Require MFA for IAM operations.
resource "aws_organizations_policy" "require_mfa_for_iam" {
  count = local.scp_active ? 1 : 0
  name  = "${local.name}-require-mfa-for-iam"
  type  = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "RequireMFAForIAM"
      Effect = "Deny"
      Action = [
        "iam:CreateUser",
        "iam:CreateAccessKey",
        "iam:DeleteAccessKey",
        "iam:UpdateAccessKey",
        "iam:AttachUserPolicy",
        "iam:DetachUserPolicy",
      ]
      Resource = "*"
      Condition = {
        BoolIfExists = {
          "aws:MultiFactorAuthPresent" = "false"
        }
      }
    }]
  })
}

# 3. Deny disabling CloudTrail / GuardDuty / Security Hub (no log-tampering).
resource "aws_organizations_policy" "deny_security_disable" {
  count = local.scp_active ? 1 : 0
  name  = "${local.name}-deny-security-disable"
  type  = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "DenySecurityDisable"
      Effect = "Deny"
      Action = [
        "cloudtrail:StopLogging",
        "cloudtrail:DeleteTrail",
        "guardduty:DeleteDetector",
        "guardduty:DisassociateFromMasterAccount",
        "guardduty:UpdateDetector",
        "securityhub:DisableSecurityHub",
        "securityhub:DeleteInsight",
        "config:DeleteConfigurationRecorder",
        "config:DeleteDeliveryChannel",
        "config:StopConfigurationRecorder",
      ]
      Resource = "*"
    }]
  })
}

# 4. Deny region usage outside operator's allowed regions.
resource "aws_organizations_policy" "deny_unwanted_regions" {
  count = local.scp_active ? 1 : 0
  name  = "${local.name}-deny-unwanted-regions"
  type  = "SERVICE_CONTROL_POLICY"

  content = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "DenyUnwantedRegions"
      Effect = "Deny"
      NotAction = [
        # Global services + ones that ignore region
        "iam:*",
        "organizations:*",
        "support:*",
        "cloudfront:*",
        "route53:*",
        "globalaccelerator:*",
        "wafv2:*", # the global scope of WAFv2 is us-east-1 anyway
      ]
      Resource = "*"
      Condition = {
        StringNotEquals = {
          "aws:RequestedRegion" = [var.region]
        }
      }
    }]
  })
}

# Attach all four to the root (operator can re-target via console).
resource "aws_organizations_policy_attachment" "all" {
  for_each = local.scp_active ? {
    deny_root    = aws_organizations_policy.deny_root_actions[0].id
    require_mfa  = aws_organizations_policy.require_mfa_for_iam[0].id
    deny_disable = aws_organizations_policy.deny_security_disable[0].id
    deny_region  = aws_organizations_policy.deny_unwanted_regions[0].id
  } : {}

  policy_id = each.value
  target_id = var.aws_org_root_id
}
