provider "google" {
  project        = var.project_id
  region         = var.region
  zone           = var.zone
  default_labels = var.labels
}

provider "google-beta" {
  project        = var.project_id
  region         = var.region
  zone           = var.zone
  default_labels = var.labels
}

# ─── Required APIs ────────────────────────────────────────────
# These need to be on before any resource that uses them. Terraform can
# enable them, but the first apply will retry once or twice while they
# propagate (~30 s each).
locals {
  required_apis = [
    "compute.googleapis.com",
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "servicenetworking.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ]
}

resource "google_project_service" "required" {
  for_each                   = toset(local.required_apis)
  service                    = each.value
  disable_dependent_services = false
  disable_on_destroy         = false
}

# ─── Tier resolution ──────────────────────────────────────────
locals {
  tier_defaults = {
    cheapest = {
      cloudsql_tier              = "db-g1-small" # 1.7 GB · ~$25/mo
      cloudsql_disk_size         = 20
      cloudsql_high_availability = false
      cluster_regional           = false # zonal Autopilot
    }
    prod = {
      cloudsql_tier              = "db-custom-2-7680" # 2 vCPU / 7.5 GB · ~$140/mo
      cloudsql_disk_size         = 100
      cloudsql_high_availability = true
      cluster_regional           = true # regional Autopilot (HA)
    }
  }

  cloudsql_tier              = var.cloudsql_tier != "" ? var.cloudsql_tier : local.tier_defaults[var.tier].cloudsql_tier
  cloudsql_disk_size         = var.cloudsql_disk_size > 0 ? var.cloudsql_disk_size : local.tier_defaults[var.tier].cloudsql_disk_size
  cloudsql_high_availability = var.cloudsql_high_availability != null ? var.cloudsql_high_availability : local.tier_defaults[var.tier].cloudsql_high_availability
  cluster_regional           = local.tier_defaults[var.tier].cluster_regional

  name = var.cluster_name
}

# ─── Kubernetes + Helm providers (auth via gcloud) ────────────
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.botarmy.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.botarmy.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.botarmy.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.botarmy.master_auth[0].cluster_ca_certificate)
  }
}

# Postgres provider — talks to Cloud SQL through its private IP via the
# user's machine + VPN/peered network. Same caveat as the AWS module:
# CREATE EXTENSION needs network reachability from the Terraform host.
provider "postgresql" {
  host            = google_sql_database_instance.botarmy.private_ip_address
  port            = 5432
  database        = google_sql_database.mem0.name
  username        = google_sql_user.mem0.name
  password        = random_password.cloudsql.result
  sslmode         = "disable" # private IP, intra-VPC; SSL still recommended for prod
  connect_timeout = 30
  superuser       = false
}
