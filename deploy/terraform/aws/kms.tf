# Customer-Managed KMS keys for EKS Secrets envelope encryption, RDS
# at-rest encryption, ECR repo encryption, and Secrets Manager wrap-key.
# Mirrors deploy/terraform/gcp/kms.tf shape.
# data.aws_caller_identity is declared in main.tf — reuse it here.

# One key per service so rotation can be staged independently.
resource "aws_kms_key" "eks_secrets" {
  count                    = local.hardening_active_aws ? 1 : 0
  description              = "${local.name} — EKS Secrets envelope encryption"
  deletion_window_in_days  = 30
  enable_key_rotation      = true
  key_usage                = "ENCRYPT_DECRYPT"
  customer_master_key_spec = var.kms_protection_level

  tags = merge(var.tags, { Purpose = "eks-secrets" })

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "eks_secrets" {
  count         = local.hardening_active_aws ? 1 : 0
  name          = "alias/${local.name}-eks-secrets"
  target_key_id = aws_kms_key.eks_secrets[0].key_id
}

resource "aws_kms_key" "rds" {
  count                    = local.hardening_active_aws ? 1 : 0
  description              = "${local.name} — RDS at-rest encryption"
  deletion_window_in_days  = 30
  enable_key_rotation      = true
  key_usage                = "ENCRYPT_DECRYPT"
  customer_master_key_spec = var.kms_protection_level

  tags = merge(var.tags, { Purpose = "rds" })

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "rds" {
  count         = local.hardening_active_aws ? 1 : 0
  name          = "alias/${local.name}-rds"
  target_key_id = aws_kms_key.rds[0].key_id
}

resource "aws_kms_key" "ecr" {
  count                    = local.hardening_active_aws ? 1 : 0
  description              = "${local.name} — ECR repo encryption"
  deletion_window_in_days  = 30
  enable_key_rotation      = true
  key_usage                = "ENCRYPT_DECRYPT"
  customer_master_key_spec = var.kms_protection_level

  tags = merge(var.tags, { Purpose = "ecr" })

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "ecr" {
  count         = local.hardening_active_aws ? 1 : 0
  name          = "alias/${local.name}-ecr"
  target_key_id = aws_kms_key.ecr[0].key_id
}

resource "aws_kms_key" "secrets_manager" {
  count                    = local.hardening_active_aws ? 1 : 0
  description              = "${local.name} — Secrets Manager wrap-key"
  deletion_window_in_days  = 30
  enable_key_rotation      = true
  key_usage                = "ENCRYPT_DECRYPT"
  customer_master_key_spec = var.kms_protection_level

  tags = merge(var.tags, { Purpose = "secrets-manager" })

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_kms_alias" "secrets_manager" {
  count         = local.hardening_active_aws ? 1 : 0
  name          = "alias/${local.name}-secrets-manager"
  target_key_id = aws_kms_key.secrets_manager[0].key_id
}
