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
  type    = bool
  default = false
}

variable "external_secret_refresh_interval" {
  description = "How often ESO re-syncs from Secrets Manager. Ignored when use_external_secrets=false."
  type        = string
  default     = "5m"
}
