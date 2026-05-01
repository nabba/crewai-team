variable "project_id" {
  description = "GCP project ID. Must already exist with billing enabled."
  type        = string
}

variable "region" {
  description = "GCP region. europe-north1 (Hamina, Finland) is closest to Helsinki/Tallinn."
  type        = string
  default     = "europe-north1"
}

variable "zone" {
  description = "Default zone within the region (only relevant for zonal clusters / cheapest tier)."
  type        = string
  default     = "europe-north1-a"
}

variable "cluster_name" {
  description = "GKE cluster name. Also used as resource prefix."
  type        = string
  default     = "botarmy"
}

variable "kubernetes_version" {
  description = "GKE control-plane release channel target. Autopilot manages versions; pin only if you need to."
  type        = string
  default     = ""
}

variable "tier" {
  description = "Sizing tier. 'cheapest' = zonal Autopilot + db-g1-small (~$110/mo). 'prod' = regional Autopilot + HA Cloud SQL (~$420/mo)."
  type        = string
  default     = "cheapest"
  validation {
    condition     = contains(["cheapest", "prod"], var.tier)
    error_message = "tier must be 'cheapest' or 'prod'."
  }
}

variable "vpc_cidr" {
  description = "Primary subnet CIDR. Pick something that doesn't collide with your office network."
  type        = string
  default     = "10.43.0.0/20"
}

variable "pods_cidr" {
  description = "Secondary range for pod IPs (must not overlap vpc_cidr)."
  type        = string
  default     = "10.44.0.0/14"
}

variable "services_cidr" {
  description = "Secondary range for ClusterIP service IPs."
  type        = string
  default     = "10.48.0.0/20"
}

variable "domain" {
  description = "Domain to expose the gateway on (e.g. bot.example.com). Empty = no Ingress (port-forward only)."
  type        = string
  default     = ""
}

variable "managed_certificate" {
  description = "When true and domain is set, provision a Google-managed cert via the ManagedCertificate CRD. False = bring your own."
  type        = bool
  default     = true
}

variable "namespace" {
  description = "Kubernetes namespace for BotArmy."
  type        = string
  default     = "botarmy"
}

variable "gateway_image_tag" {
  description = "Tag of the gateway image to deploy. Image is pushed to the created Artifact Registry repo."
  type        = string
  default     = "latest"
}

variable "deploy_helm_chart" {
  description = "Whether `terraform apply` should also run the BotArmy Helm chart."
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Install kube-prometheus-stack + Grafana, expose ServiceMonitors. Adds ~$15/mo for storage + scrape resources."
  type        = bool
  default     = true
}

variable "extra_env" {
  description = "Map of extra env vars (API keys, model selectors). Synced to GCP Secret Manager + the in-cluster Secret."
  type        = map(string)
  default     = {}
  sensitive   = true
}

variable "labels" {
  description = "GCP labels applied to all resources."
  type        = map(string)
  default = {
    project    = "botarmy"
    managed_by = "terraform"
  }
}

# ─── Tier-derived sizing (overrides) ─────────────────────────
variable "cloudsql_tier" {
  description = "Cloud SQL instance tier. Empty = derived from var.tier."
  type        = string
  default     = ""
}

variable "cloudsql_disk_size" {
  description = "Cloud SQL disk size in GB. 0 = derived from tier."
  type        = number
  default     = 0
}

variable "cloudsql_high_availability" {
  description = "Whether Cloud SQL is regional (HA). null = derived from tier."
  type        = bool
  default     = null
}
