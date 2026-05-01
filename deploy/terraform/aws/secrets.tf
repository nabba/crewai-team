# Secret flow:
#
#   1. Terraform creates the database password (random_password.rds in rds.tf).
#   2. Stores it + every key in var.extra_env into AWS Secrets Manager — that's
#      the canonical source of truth, queryable from anything in the AWS account.
#   3. Also creates a Kubernetes Secret named `botarmy-env` in the workload
#      namespace, populated from the same map. The Helm chart consumes it via
#      `envFrom: secretRef` (same shape as the local install).
#
# This avoids the External Secrets Operator dependency. Trade-off: rotating a
# value means re-running `terraform apply`, not just updating Secrets Manager.
# Acceptable for v1; ESO can be layered in later (see README → "Hardening").

# ─── Generated GATEWAY_SECRET (auth token between Signal forwarder + gateway) ─
resource "random_password" "gateway_secret" {
  length  = 64
  special = false
}

# Neo4j runs in-cluster (no RDS-equivalent for Neo4j). Generate a password
# here and propagate via the env secret. The chart's neo4j.yaml composes
# `NEO4J_AUTH = neo4j/$(MEM0_NEO4J_PASSWORD)` at pod startup.
resource "random_password" "neo4j" {
  length  = 32
  special = false
}

# ─── Required Kubernetes namespace ────────────────────────────
resource "kubernetes_namespace" "botarmy" {
  metadata { name = var.namespace }
}

# ─── Compose the env map ──────────────────────────────────────
# Database connection strings + computed secrets are merged with whatever the
# user passed in `var.extra_env` (their API keys etc.). User-supplied values
# WIN if they override (e.g. they want to bring their own GATEWAY_SECRET).
locals {
  generated_env = {
    # Mem0 / pgvector
    MEM0_POSTGRES_HOST     = aws_db_instance.botarmy.address
    MEM0_POSTGRES_PORT     = tostring(aws_db_instance.botarmy.port)
    MEM0_POSTGRES_DB       = aws_db_instance.botarmy.db_name
    MEM0_POSTGRES_USER     = aws_db_instance.botarmy.username
    MEM0_POSTGRES_PASSWORD = random_password.rds.result

    # Neo4j (in-cluster StatefulSet)
    MEM0_NEO4J_PASSWORD = random_password.neo4j.result
    MEM0_NEO4J_USER     = "neo4j"

    # Gateway
    GATEWAY_SECRET = random_password.gateway_secret.result
    GATEWAY_BIND   = "0.0.0.0"
    GATEWAY_PORT   = "8765"

    # In-cluster service DNS for the stateless dependencies that DO run in k8s.
    # Postgres is managed (RDS) so we override the in-cluster default.
    NEO4J_URI    = "bolt://${var.cluster_name}-neo4j:7687"
    CHROMADB_URL = "http://${var.cluster_name}-chromadb:8000"

    # K8s mode — disable host-only features
    LOCAL_LLM_ENABLED     = "false" # no host Ollama in cluster
    SIGNAL_BOT_NUMBER     = ""      # Signal interface skipped
    RECOVERY_LOOP_ENABLED = "false"
  }

  # User-supplied extra_env overrides anything generated.
  effective_env = merge(local.generated_env, var.extra_env)
}

# ─── AWS Secrets Manager (one secret, JSON-encoded payload) ───
resource "aws_secretsmanager_secret" "botarmy_env" {
  name                    = "${local.name}-env"
  description             = "BotArmy gateway environment (API keys + DB creds)"
  recovery_window_in_days = var.tier == "prod" ? 7 : 0 # immediate delete in dev
}

resource "aws_secretsmanager_secret_version" "botarmy_env" {
  secret_id     = aws_secretsmanager_secret.botarmy_env.id
  secret_string = jsonencode(local.effective_env)
}

# ─── Kubernetes Secret consumed by the chart ──────────────────
resource "kubernetes_secret" "botarmy_env" {
  metadata {
    name      = "botarmy-env"
    namespace = kubernetes_namespace.botarmy.metadata[0].name
  }

  # Opaque secret — flat key/value pairs, one per env var. Helm chart
  # references it via `envFrom: { secretRef: { name: botarmy-env } }`.
  data = local.effective_env
  type = "Opaque"
}
