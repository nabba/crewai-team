variable "region" {
  description = "AWS region. EU-North-1 (Stockholm) is closest to Helsinki/Tallinn."
  type        = string
  default     = "eu-north-1"
}

variable "cluster_name" {
  description = "EKS cluster name. Also used as resource prefix."
  type        = string
  default     = "botarmy"
}

variable "kubernetes_version" {
  description = "EKS Kubernetes version."
  type        = string
  default     = "1.30"
}

variable "tier" {
  description = "Sizing tier. 'cheapest' targets dev/lab budget (~$130/mo). 'prod' is multi-AZ HA (~$400/mo)."
  type        = string
  default     = "cheapest"
  validation {
    condition     = contains(["cheapest", "prod"], var.tier)
    error_message = "tier must be 'cheapest' or 'prod'."
  }
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC. Pick something that doesn't collide with your office network."
  type        = string
  default     = "10.42.0.0/16"
}

variable "domain" {
  description = "Domain to expose the gateway on (e.g. bot.example.com). Empty = no Ingress (port-forward only)."
  type        = string
  default     = ""
}

variable "acm_certificate_arn" {
  description = "ARN of an ACM certificate for the domain. Empty = cert-manager + Let's Encrypt at the cluster level."
  type        = string
  default     = ""
}

variable "namespace" {
  description = "Kubernetes namespace for BotArmy."
  type        = string
  default     = "botarmy"
}

variable "gateway_image_tag" {
  description = "Tag of the gateway image to deploy. Image is pushed to the created ECR repo."
  type        = string
  default     = "latest"
}

variable "deploy_helm_chart" {
  description = "Whether `terraform apply` should also run the BotArmy Helm chart. Set false to provision infra only and run helm separately."
  type        = bool
  default     = true
}

variable "extra_env" {
  description = "Map of extra env vars (API keys, model selectors) to inject into the gateway. Synced to AWS Secrets Manager + the k8s envFrom secret."
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "tags" {
  description = "AWS tags applied to all resources."
  type        = map(string)
  default = {
    Project   = "BotArmy"
    ManagedBy = "Terraform"
  }
}

# ─── Tier-derived sizing ──────────────────────────────────────
# These are computed from `tier` in main.tf locals. Override individually if
# you want a custom mix (e.g. cheapest cluster but prod-sized DB).

variable "node_instance_types" {
  description = "EC2 instance types for the managed node group. Empty = derived from tier."
  type        = list(string)
  default     = []
}

variable "node_desired_size" {
  description = "Desired worker node count. 0 = derived from tier."
  type        = number
  default     = 0
}

variable "rds_instance_class" {
  description = "RDS instance class. Empty = derived from tier."
  type        = string
  default     = ""
}

variable "rds_allocated_storage" {
  description = "Initial RDS storage in GB. 0 = derived from tier."
  type        = number
  default     = 0
}

variable "rds_multi_az" {
  description = "Whether RDS is multi-AZ. null = derived from tier."
  type        = bool
  default     = null
}

# ── External Secrets Operator (Phase C2 opt-in) ──────────────────────
variable "use_external_secrets" {
  description = <<-EOT
    Use External Secrets Operator (ESO) to sync the gateway env Secret
    from AWS Secrets Manager into the cluster, instead of Terraform
    writing a ``kubernetes_secret`` directly.

    Default false preserves the v1 behaviour (terraform-owned k8s
    Secret). When true:
      * Terraform NO LONGER creates ``kubernetes_secret.botarmy_env``.
      * Terraform creates a ``ClusterSecretStore`` backed by Secrets
        Manager and an ``ExternalSecret`` that ESO reconciles into the
        same Secret name (``botarmy-env``) the chart references.
      * The ESO controller must already be installed in the cluster
        (``helm install external-secrets external-secrets/external-secrets``).
      * IRSA bind: attach a role with secretsmanager:GetSecretValue
        on this secret to the external-secrets service account. See
        deploy/HARDENING.md for the IRSA HCL.

    Rotation becomes: update Secrets Manager → ESO syncs within
    ``external_secret_refresh_interval``. No ``terraform apply``.
  EOT
  type        = bool
  default     = false
}

variable "external_secret_refresh_interval" {
  description = "How often ESO re-syncs from Secrets Manager. Ignored when use_external_secrets=false."
  type        = string
  default     = "5m"
}

# ─── Hardening (2026-05-17 extension, AWS mirror of GCP) ─────────
variable "hardening_profile" {
  description = "Hardening profile: 'off' (no extra hardening), 'basic' (EKS secrets KMS + RDS KMS + ECR scanning + VPC flow logs), 'strict' (basic + EKS private endpoint allowlist + AWS WAFv2 + CloudTrail + GuardDuty + Security Hub + SCPs)."
  type        = string
  default     = "strict"
  validation {
    condition     = contains(["off", "basic", "strict"], var.hardening_profile)
    error_message = "hardening_profile must be one of: off, basic, strict."
  }
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to reach the EKS public API endpoint when hardening_profile=strict. Auto-detected by app.substrate.cloud_hardening: Tailnet (100.64.0.0/10) preferred + laptop public IP as fallback. Empty list = the cluster keeps its existing default 0.0.0.0/0 public endpoint."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "kms_protection_level" {
  description = "AWS KMS key type. SYMMETRIC_DEFAULT (free under 20k requests/mo) is recommended. CMK keys cost $1/month each."
  type        = string
  default     = "SYMMETRIC_DEFAULT"
}

variable "aws_org_enabled" {
  description = "Set true if you operate inside AWS Organizations + want Service Control Policies applied. Requires Organizations management-account credentials; the workload account can't self-apply SCPs."
  type        = bool
  default     = false
}

variable "aws_org_root_id" {
  description = "Organization root ID (e.g. r-abcd1234) for SCP attachment. Required when aws_org_enabled=true. Auto-detected by app.substrate.cloud_hardening when running under the migrate wizard."
  type        = string
  default     = ""
}

variable "cloudtrail_retention_days" {
  description = "S3 lifecycle retention for the CloudTrail audit-log bucket. 400 days is the operator's standard."
  type        = number
  default     = 400
}

variable "waf_rate_limit_per_5min" {
  description = "Per-IP rate-limit threshold on the AWS WAFv2. 3000 = 10rps."
  type        = number
  default     = 3000
}
