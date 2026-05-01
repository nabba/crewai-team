# GKE-specific Helm releases:
#
#   - cert-manager (only when domain set + managed_certificate is false)
#   - kube-prometheus-stack (when var.enable_monitoring)
#   - the BotArmy chart itself
#
# GKE doesn't need an ALB controller installed — Ingress with className
# `gce` is handled by the platform. Google-managed certs go via the
# ManagedCertificate CRD which we create inline if the user opts in.

# ─── BotArmy chart ────────────────────────────────────────────
locals {
  botarmy_chart_path = "${path.module}/../../k8s"
}

resource "helm_release" "botarmy" {
  count = var.deploy_helm_chart ? 1 : 0

  name      = "botarmy"
  chart     = local.botarmy_chart_path
  namespace = kubernetes_namespace.botarmy.metadata[0].name

  # Don't block the apply on pod readiness — the gateway pod can't start
  # until the image has been pushed to Artifact Registry, which the
  # dispatcher does after terraform apply returns. The dispatcher's
  # `kubectl rollout status` call is the right readiness gate. (Verified
  # by an e2e test on 2026-05-01: with wait=true, helm timed out at 5m
  # on ImagePullBackOff.)
  wait    = false
  timeout = 600

  values = [yamlencode({
    image = {
      repository = "${local.artifact_registry_url}/gateway"
      tag        = var.gateway_image_tag
    }

    envSecretName = kubernetes_secret.botarmy_env.metadata[0].name

    # Bind the gateway's KSA to the GCP SA via Workload Identity.
    serviceAccount = {
      create = true
      name   = "botarmy-gateway"
      annotations = {
        "iam.gke.io/gcp-service-account" = google_service_account.gateway.email
      }
    }

    gateway = {
      service = { type = "ClusterIP" }
      ingress = var.domain == "" ? { enabled = false } : {
        enabled   = true
        className = "gce"
        host      = var.domain
        annotations = merge(
          {
            "kubernetes.io/ingress.global-static-ip-name" = google_compute_global_address.botarmy_ingress[0].name
          },
          var.managed_certificate ? {
            "networking.gke.io/managed-certificates" = "botarmy-cert"
          } : {}
        )
        # When using Google-managed cert, the chart-level TLS spec is a no-op
        # (the cert lives in a CRD, not a Secret). Still leave it on so other
        # paths (cert-manager) work if managed_certificate is false.
        tls = {
          enabled    = !var.managed_certificate
          secretName = "botarmy-tls"
        }
      }
    }

    # Cloud SQL is the source of truth — disable in-cluster Postgres.
    postgres = { enabled = false }

    # GKE's default StorageClass is `standard-rwo` (PD balanced). Use SSD
    # (`premium-rwo`) for the databases.
    neo4j    = { persistence = { storageClass = "premium-rwo" } }
    chromadb = { persistence = { storageClass = "premium-rwo" } }

    # Monitoring switches — chart templates (servicemonitor.yaml etc.) read these.
    monitoring = {
      serviceMonitor = { enabled = var.enable_monitoring }
      prometheusRule = { enabled = var.enable_monitoring }
      dashboards     = { enabled = var.enable_monitoring }
    }
  })]

  depends_on = [
    kubernetes_secret.botarmy_env,
    google_sql_user.mem0, # ensure the DB user exists before the gateway pod tries to use it
    helm_release.kube_prometheus_stack,
  ]
}

# ─── Static IP for the Ingress (so the LB IP doesn't churn) ──
resource "google_compute_global_address" "botarmy_ingress" {
  count = var.domain != "" ? 1 : 0
  name  = "${local.name}-ingress-ip"
}

# ─── Google-managed TLS cert ─────────────────────────────────
# This uses the ManagedCertificate CRD which is GKE-specific and not in
# the standard kubernetes provider schema, so we apply it via raw manifest.
resource "kubernetes_manifest" "managed_cert" {
  count = var.domain != "" && var.managed_certificate ? 1 : 0

  manifest = {
    apiVersion = "networking.gke.io/v1"
    kind       = "ManagedCertificate"
    metadata = {
      name      = "botarmy-cert"
      namespace = var.namespace
    }
    spec = {
      domains = [var.domain]
    }
  }

  depends_on = [kubernetes_namespace.botarmy]
}
