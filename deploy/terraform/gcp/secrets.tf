# Same shape as the AWS module:
#   1. Computed env (DB host, generated GATEWAY_SECRET, internal service URLs)
#      merged with var.extra_env (user-supplied API keys; overrides win).
#   2. Pushed to Secret Manager (one secret per env var, labelled botarmy=true)
#      so anything in the GCP project can read them.
#   3. Also synced into a Kubernetes Opaque Secret named `botarmy-env` in the
#      workload namespace, consumed by the chart via envFrom.
#
# Phase C2: see ``deploy/HARDENING.md`` for the GCP-specific ESO setup
# (ClusterSecretStore with Workload Identity binding to a GSA that has
# ``roles/secretmanager.secretAccessor`` on this secret) plus GKE
# Application-layer Secrets Encryption via ``google_kms_crypto_key``
# referenced from ``database_encryption`` on the cluster.

resource "random_password" "gateway_secret" {
  length  = 64
  special = false
}

# Neo4j runs in-cluster (no GCP-managed Neo4j equivalent). Generate a password
# here and propagate it via the env secret. The chart's neo4j.yaml composes
# `NEO4J_AUTH = neo4j/$(MEM0_NEO4J_PASSWORD)` at pod startup using k8s var
# expansion — so the value lives only in this secret, never in template literals.
resource "random_password" "neo4j" {
  length  = 32
  special = false
}

resource "kubernetes_namespace" "botarmy" {
  metadata { name = var.namespace }
}

locals {
  generated_env = {
    # Mem0 / pgvector
    MEM0_POSTGRES_HOST     = google_sql_database_instance.botarmy.private_ip_address
    MEM0_POSTGRES_PORT     = "5432"
    MEM0_POSTGRES_DB       = google_sql_database.mem0.name
    MEM0_POSTGRES_USER     = google_sql_user.mem0.name
    MEM0_POSTGRES_PASSWORD = random_password.cloudsql.result

    # Neo4j (in-cluster StatefulSet)
    MEM0_NEO4J_PASSWORD = random_password.neo4j.result
    MEM0_NEO4J_USER     = "neo4j"

    # Gateway
    GATEWAY_SECRET = random_password.gateway_secret.result
    GATEWAY_BIND   = "0.0.0.0"
    GATEWAY_PORT   = "8765"

    # In-cluster service DNS
    NEO4J_URI    = "bolt://${var.cluster_name}-neo4j:7687"
    CHROMADB_URL = "http://${var.cluster_name}-chromadb:8000"

    # K8s mode — host-only features off
    LOCAL_LLM_ENABLED     = "false"
    SIGNAL_BOT_NUMBER     = ""
    RECOVERY_LOOP_ENABLED = "false"
  }

  effective_env = merge(local.generated_env, var.extra_env)
}

# ─── Secret Manager (one secret per env var, all under one prefix) ─
# GCP Secret Manager is keyed differently from AWS SM — each value is its own
# Secret resource. We store the full payload as a JSON blob in `botarmy-env`
# (matches AWS SM shape) AND optionally individually for tools that prefer
# per-key secrets. Per-key adds a few cents/mo per secret; skipping for now.
resource "google_secret_manager_secret" "botarmy_env" {
  secret_id = "${local.name}-env"
  replication {
    auto {}
  }
  labels = var.labels

  depends_on = [google_project_service.required]
}

resource "google_secret_manager_secret_version" "botarmy_env" {
  secret      = google_secret_manager_secret.botarmy_env.id
  secret_data = jsonencode(local.effective_env)
}

# Allow the gateway's GCP service account to read the secret.
resource "google_secret_manager_secret_iam_member" "gateway_read" {
  secret_id = google_secret_manager_secret.botarmy_env.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.gateway.email}"
}

# ─── Kubernetes Secret consumed by the chart ──────────────────
# Same approach as AWS for v1 — TF writes the values directly. Trade-off:
# rotating a value means re-applying. ESO can be layered on later.
resource "kubernetes_secret" "botarmy_env" {
  metadata {
    name      = "botarmy-env"
    namespace = kubernetes_namespace.botarmy.metadata[0].name
  }
  data = local.effective_env
  type = "Opaque"
}
