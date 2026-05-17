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

# ── External Secrets Operator (Phase C2 opt-in) ──────────────────────
variable "use_external_secrets" {
  description = <<-EOT
    Use External Secrets Operator (ESO) to sync the gateway env Secret
    from GCP Secret Manager into the cluster, instead of Terraform
    writing a ``kubernetes_secret`` directly.

    Default false preserves the v1 behaviour. When true:
      * Terraform NO LONGER creates ``kubernetes_secret.botarmy_env``.
      * Terraform creates a ``ClusterSecretStore`` backed by GCP Secret
        Manager and an ``ExternalSecret`` that ESO reconciles into the
        same Secret name the chart references.
      * The ESO controller must already be installed in the cluster.
      * Workload Identity bind: link the external-secrets KSA to a GSA
        with ``roles/secretmanager.secretAccessor`` on this secret.
        See deploy/HARDENING.md.

    Rotation becomes: update Secret Manager → ESO syncs within
    ``external_secret_refresh_interval``. No ``terraform apply``.
  EOT
  type        = bool
  default     = false
}

variable "external_secret_refresh_interval" {
  description = "How often ESO re-syncs from GCP Secret Manager. Ignored when use_external_secrets=false."
  type        = string
  default     = "5m"
}

# ─── Hardening (2026-05-17 extension) ────────────────────────
# Three-step profile applied across GKE, CloudSQL, network, Cloud Armor,
# Binary Authorization, KMS, audit-log sinks, and (when org_id is set)
# org-level constraints. "off" preserves the pre-hardening behaviour for
# users who want to roll their own.
variable "hardening_profile" {
  description = "Hardening profile: 'off' (no extra hardening), 'basic' (Shielded nodes, Workload Identity, VPC flow logs, CMEK, deletion_protection on prod), 'strict' (basic + master-authorized-networks, Cloud Armor, Binary Authorization, audit-log sinks, org policies)."
  type        = string
  default     = "strict"
  validation {
    condition     = contains(["off", "basic", "strict"], var.hardening_profile)
    error_message = "hardening_profile must be one of: off, basic, strict."
  }
}

variable "allowed_cidrs" {
  description = "CIDR blocks allowed to reach the GKE master endpoint when hardening_profile=strict. Auto-detected by app.substrate.cloud_hardening: Tailnet (100.64.0.0/10) preferred + laptop public IP as fallback. Empty list disables master_authorized_networks (use only with hardening_profile=off|basic)."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "binauthz_mode" {
  description = "Binary Authorization enforcement mode. AUDIT (default) logs would-be-blocks but allows; ENFORCE rejects unsigned images. Only honored when hardening_profile=strict. Promote to ENFORCE only after your image-signing pipeline is wired."
  type        = string
  default     = "AUDIT"
  validation {
    condition     = contains(["AUDIT", "ENFORCE"], var.binauthz_mode)
    error_message = "binauthz_mode must be AUDIT or ENFORCE."
  }
}

variable "binauthz_attestor_name" {
  description = "Binary Authorization attestor name (created by scripts/install/cosign_setup.sh). Empty = ALWAYS_ALLOW default rule even in ENFORCE mode. Set to the attestor short name (e.g. 'botarmy-attestor') after cosign_setup runs."
  type        = string
  default     = ""
}

variable "kms_protection_level" {
  description = "KMS key protection level for CMEK on CloudSQL, GKE etcd, Artifact Registry, Secret Manager. SOFTWARE (~free, FIPS 140-2 L1) or HSM (€$2-3/key/month, FIPS 140-2 L3)."
  type        = string
  default     = "SOFTWARE"
  validation {
    condition     = contains(["SOFTWARE", "HSM"], var.kms_protection_level)
    error_message = "kms_protection_level must be SOFTWARE or HSM."
  }
}

variable "org_id" {
  description = "Google Workspace organization ID (numeric, no 'organizations/' prefix). Set when you have a Workspace org and want org-policy constraints applied. Empty = skip org policies (still applies project-level hardening). Auto-detected by app.substrate.cloud_hardening.detect_org_id when running under the migrate wizard."
  type        = string
  default     = ""
}

variable "audit_log_retention_days" {
  description = "Retention for the audit-log GCS bucket. 400 days is the GCP default; longer keeps the trail across audits."
  type        = number
  default     = 400
}

variable "cloud_armor_rate_limit_rpm" {
  description = "Per-IP request-per-minute rate-limit threshold on the Cloud Armor security policy. 600 = 10 rps; reasonable for a personal-assistant workload."
  type        = number
  default     = 600
}

# ─── VPC Service Controls ───────────────────────────────────
variable "vpc_sc_enabled" {
  description = "Apply a VPC Service Controls perimeter around the project. Requires var.org_id + var.access_policy_id. Default OFF — VPC-SC can lock you out as easily as master_authorized_networks if misconfigured, so it's opt-in even at hardening_profile=strict."
  type        = bool
  default     = false
}

variable "access_policy_id" {
  description = "Existing access-context-manager access policy id (numeric, no prefix). Each org has at most one access policy — auto-detected by app.substrate.cloud_hardening.detect_access_policy_id when running under the migrate wizard. Empty when vpc_sc_enabled=false."
  type        = string
  default     = ""
}

variable "vpc_sc_dry_run" {
  description = "Apply the perimeter in DRY_RUN mode — logs would-be-blocks instead of enforcing. Default true so the first apply is observational; flip to false in /cp/settings after reviewing the dry-run logs for ~1 week."
  type        = bool
  default     = true
}

variable "vpc_sc_restricted_services" {
  description = "Google APIs the perimeter restricts. Default list covers the services BotArmy actually uses; add more (e.g. bigquery.googleapis.com) as needed."
  type        = list(string)
  default = [
    "storage.googleapis.com",
    "container.googleapis.com",
    "secretmanager.googleapis.com",
    "sqladmin.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudkms.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "containeranalysis.googleapis.com",
    "binaryauthorization.googleapis.com",
  ]
}
