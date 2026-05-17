# Customer-Managed Encryption Keys (CMEK) for CloudSQL, GKE etcd,
# Artifact Registry, and Secret Manager. Created only when
# hardening_profile != "off". One keyring per cluster, one key per
# service so rotation can be staged independently.
#
# SOFTWARE protection (default) is free; HSM is €$2-3/key/month for
# FIPS 140-2 Level 3 compliance. Compose `var.kms_protection_level`
# at the variables layer.

locals {
  hardening_active = var.hardening_profile != "off"
  hardening_strict = var.hardening_profile == "strict"
  # GCP service agents that need keyEncrypter/Decrypter on each key.
  # google_project.botarmy.number is the project number; we read it via
  # data source to avoid hardcoding.
  service_agents = local.hardening_active ? {
    cloudsql          = "service-${data.google_project.this.number}@gcp-sa-cloud-sql.iam.gserviceaccount.com"
    gke               = "service-${data.google_project.this.number}@container-engine-robot.iam.gserviceaccount.com"
    artifact_registry = "service-${data.google_project.this.number}@gcp-sa-artifactregistry.iam.gserviceaccount.com"
    secret_manager    = "service-${data.google_project.this.number}@gcp-sa-secretmanager.iam.gserviceaccount.com"
  } : {}
}

data "google_project" "this" {
  project_id = var.project_id
}

resource "google_kms_key_ring" "botarmy" {
  count    = local.hardening_active ? 1 : 0
  name     = "${local.name}-keyring"
  location = var.region

  depends_on = [google_project_service.required]
}

resource "google_kms_crypto_key" "cloudsql" {
  count    = local.hardening_active ? 1 : 0
  name     = "${local.name}-cloudsql"
  key_ring = google_kms_key_ring.botarmy[0].id

  rotation_period = "7776000s" # 90 days

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = var.kms_protection_level
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "gke_etcd" {
  count    = local.hardening_active ? 1 : 0
  name     = "${local.name}-gke-etcd"
  key_ring = google_kms_key_ring.botarmy[0].id

  rotation_period = "7776000s" # 90 days

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = var.kms_protection_level
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "artifact_registry" {
  count    = local.hardening_active ? 1 : 0
  name     = "${local.name}-artifact-registry"
  key_ring = google_kms_key_ring.botarmy[0].id

  rotation_period = "7776000s"

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = var.kms_protection_level
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_kms_crypto_key" "secret_manager" {
  count    = local.hardening_active ? 1 : 0
  name     = "${local.name}-secret-manager"
  key_ring = google_kms_key_ring.botarmy[0].id

  rotation_period = "7776000s"

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = var.kms_protection_level
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ─── Service-agent IAM bindings ─────────────────────────────────
# Each GCP-managed service needs keyEncrypter/Decrypter on its specific
# key before it can use CMEK. Without this, the resource create fails.

resource "google_kms_crypto_key_iam_member" "cloudsql_sa" {
  count         = local.hardening_active ? 1 : 0
  crypto_key_id = google_kms_crypto_key.cloudsql[0].id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${local.service_agents.cloudsql}"
}

resource "google_kms_crypto_key_iam_member" "gke_sa" {
  count         = local.hardening_active ? 1 : 0
  crypto_key_id = google_kms_crypto_key.gke_etcd[0].id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${local.service_agents.gke}"
}

resource "google_kms_crypto_key_iam_member" "artifact_registry_sa" {
  count         = local.hardening_active ? 1 : 0
  crypto_key_id = google_kms_crypto_key.artifact_registry[0].id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${local.service_agents.artifact_registry}"
}

resource "google_kms_crypto_key_iam_member" "secret_manager_sa" {
  count         = local.hardening_active ? 1 : 0
  crypto_key_id = google_kms_crypto_key.secret_manager[0].id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${local.service_agents.secret_manager}"
}
