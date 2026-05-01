# Observability for GCP — same kube-prometheus-stack as the AWS module,
# plus a Cloud Monitoring dashboard for Cloud SQL + LB + GKE-published metrics
# that don't exist in Prometheus.
#
# Cost: ~$15/mo for the in-cluster Prometheus storage. Cloud Monitoring is
# free for the Google-published metrics we use (the dashboards have no
# custom metric ingestion).

# ─── Alert routing ────────────────────────────────────────────
# Identical shape to the AWS module's variables — see
# deploy/terraform/aws/observability.tf for full commentary.

variable "alertmanager_slack_webhook_url" {
  description = "Slack incoming-webhook URL for Alertmanager notifications. Empty = no Slack."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alertmanager_slack_channel" {
  description = "Slack channel for routed alerts."
  type        = string
  default     = "#botarmy-alerts"
}

variable "alertmanager_email_to" {
  description = "Email address to receive critical alerts. Empty = no email."
  type        = string
  default     = ""
}

variable "alertmanager_smtp_from" {
  description = "From-address for Alertmanager email."
  type        = string
  default     = "alerts@botarmy.local"
}

variable "alertmanager_smtp_smarthost" {
  description = "SMTP relay host:port. Required if alertmanager_email_to is set."
  type        = string
  default     = ""
}

variable "alertmanager_smtp_auth_username" {
  description = "SMTP relay auth username."
  type        = string
  default     = ""
}

variable "alertmanager_smtp_auth_password" {
  description = "SMTP relay auth password."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alertmanager_opsgenie_api_key" {
  description = "Opsgenie API key (Integration → API). Empty = no Opsgenie."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alertmanager_opsgenie_api_url" {
  description = "Opsgenie API base. Use https://api.eu.opsgenie.com/ for EU accounts."
  type        = string
  default     = "https://api.opsgenie.com/"
}

locals {
  alertmanager_routing_enabled = (
    var.alertmanager_slack_webhook_url != ""
    || var.alertmanager_email_to != ""
    || var.alertmanager_opsgenie_api_key != ""
  )

  alertmanager_config = {
    global = merge(
      var.alertmanager_slack_webhook_url != "" ? {
        slack_api_url = var.alertmanager_slack_webhook_url
      } : {},
      var.alertmanager_smtp_smarthost != "" ? {
        smtp_smarthost     = var.alertmanager_smtp_smarthost
        smtp_from          = var.alertmanager_smtp_from
        smtp_auth_username = var.alertmanager_smtp_auth_username
        smtp_auth_password = var.alertmanager_smtp_auth_password
        smtp_require_tls   = true
      } : {}
    )
    route = {
      receiver        = "default"
      group_by        = ["alertname", "severity"]
      group_wait      = "30s"
      group_interval  = "5m"
      repeat_interval = "4h"
      routes = [
        {
          matchers        = ["severity = critical"]
          receiver        = "critical"
          group_wait      = "10s"
          repeat_interval = "1h"
        },
        {
          matchers = ["severity = warning"]
          receiver = "warning"
        }
      ]
    }
    receivers = [
      { name = "default" },
      merge(
        { name = "critical" },
        var.alertmanager_slack_webhook_url != "" ? {
          slack_configs = [{
            channel       = var.alertmanager_slack_channel
            send_resolved = true
            title         = "[CRITICAL] {{ .GroupLabels.alertname }}"
            text          = "{{ range .Alerts }}*{{ .Annotations.summary }}*\n{{ .Annotations.description }}\n{{ end }}"
          }]
        } : {},
        var.alertmanager_email_to != "" ? {
          email_configs = [{
            to            = var.alertmanager_email_to
            send_resolved = true
            headers       = { Subject = "[CRITICAL] BotArmy: {{ .GroupLabels.alertname }}" }
          }]
        } : {},
        var.alertmanager_opsgenie_api_key != "" ? {
          opsgenie_configs = [{
            api_key       = var.alertmanager_opsgenie_api_key
            api_url       = var.alertmanager_opsgenie_api_url
            send_resolved = true
            priority      = "P1"
            message       = "{{ .GroupLabels.alertname }}"
            description   = "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}\n{{ end }}"
            source        = "botarmy"
            tags          = "botarmy,critical"
          }]
        } : {}
      ),
      merge(
        { name = "warning" },
        var.alertmanager_slack_webhook_url != "" ? {
          slack_configs = [{
            channel       = var.alertmanager_slack_channel
            send_resolved = true
            title         = "[warning] {{ .GroupLabels.alertname }}"
            text          = "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"
          }]
        } : {},
        var.alertmanager_opsgenie_api_key != "" ? {
          opsgenie_configs = [{
            api_key       = var.alertmanager_opsgenie_api_key
            api_url       = var.alertmanager_opsgenie_api_url
            send_resolved = true
            priority      = "P3"
            message       = "{{ .GroupLabels.alertname }}"
            description   = "{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}"
            source        = "botarmy"
            tags          = "botarmy,warning"
          }]
        } : {}
      )
    ]
  }
}

resource "helm_release" "kube_prometheus_stack" {
  count = var.enable_monitoring ? 1 : 0

  name             = "kube-prometheus-stack"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  version          = "61.3.0"
  namespace        = "monitoring"
  create_namespace = true
  timeout          = 600

  values = [yamlencode({
    fullnameOverride = "kube-prometheus-stack"

    prometheus = {
      prometheusSpec = {
        serviceMonitorSelectorNilUsesHelmValues = false
        ruleSelectorNilUsesHelmValues           = false
        retention                               = "7d"
        scrapeInterval                          = "30s"
        storageSpec = {
          volumeClaimTemplate = {
            spec = {
              accessModes      = ["ReadWriteOnce"]
              storageClassName = "premium-rwo"
              resources        = { requests = { storage = "10Gi" } }
            }
          }
        }
      }
    }

    grafana = {
      adminPassword            = random_password.grafana_admin[0].result
      defaultDashboardsEnabled = true
      sidecar = {
        dashboards = {
          enabled         = true
          searchNamespace = "ALL"
          label           = "grafana_dashboard"
        }
      }
      persistence = {
        enabled          = true
        size             = "5Gi"
        storageClassName = "premium-rwo"
      }
      service = { type = "ClusterIP" }
    }

    alertmanager = merge(
      {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                accessModes      = ["ReadWriteOnce"]
                storageClassName = "premium-rwo"
                resources        = { requests = { storage = "5Gi" } }
              }
            }
          }
        }
      },
      local.alertmanager_routing_enabled ? {
        config = local.alertmanager_config
      } : {}
    )
  })]

  depends_on = [google_container_cluster.botarmy]
}

resource "random_password" "grafana_admin" {
  count   = var.enable_monitoring ? 1 : 0
  length  = 24
  special = false
}

# Stash the Grafana password in Secret Manager.
resource "google_secret_manager_secret" "grafana_admin" {
  count     = var.enable_monitoring ? 1 : 0
  secret_id = "${local.name}-grafana-admin"
  replication {
    auto {}
  }
  labels = var.labels
}

resource "google_secret_manager_secret_version" "grafana_admin" {
  count       = var.enable_monitoring ? 1 : 0
  secret      = google_secret_manager_secret.grafana_admin[0].id
  secret_data = random_password.grafana_admin[0].result
}

# ─── Cloud Monitoring dashboard for managed pieces ────────────
resource "google_monitoring_dashboard" "botarmy" {
  count = var.enable_monitoring ? 1 : 0
  dashboard_json = jsonencode({
    displayName = "${local.name} — Infra"
    mosaicLayout = {
      columns = 12
      tiles = [
        {
          width  = 6
          height = 4
          widget = {
            title = "Cloud SQL CPU"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter = "resource.type=\"cloudsql_database\" AND resource.labels.database_id=\"${var.project_id}:${google_sql_database_instance.botarmy.name}\" AND metric.type=\"cloudsql.googleapis.com/database/cpu/utilization\""
                    aggregation = {
                      alignmentPeriod  = "60s"
                      perSeriesAligner = "ALIGN_MEAN"
                    }
                  }
                }
              }]
              yAxis = { label = "CPU", scale = "LINEAR" }
            }
          }
        },
        {
          xPos   = 6
          width  = 6
          height = 4
          widget = {
            title = "Cloud SQL connections"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter      = "resource.type=\"cloudsql_database\" AND resource.labels.database_id=\"${var.project_id}:${google_sql_database_instance.botarmy.name}\" AND metric.type=\"cloudsql.googleapis.com/database/postgresql/num_backends\""
                    aggregation = { alignmentPeriod = "60s", perSeriesAligner = "ALIGN_MEAN" }
                  }
                }
              }]
            }
          }
        },
        {
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "Cloud SQL disk usage"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter      = "resource.type=\"cloudsql_database\" AND resource.labels.database_id=\"${var.project_id}:${google_sql_database_instance.botarmy.name}\" AND metric.type=\"cloudsql.googleapis.com/database/disk/utilization\""
                    aggregation = { alignmentPeriod = "300s", perSeriesAligner = "ALIGN_MEAN" }
                  }
                }
              }]
            }
          }
        },
        {
          xPos   = 6
          yPos   = 4
          width  = 6
          height = 4
          widget = {
            title = "GKE container CPU (BotArmy namespace)"
            xyChart = {
              dataSets = [{
                timeSeriesQuery = {
                  timeSeriesFilter = {
                    filter      = "resource.type=\"k8s_container\" AND resource.labels.namespace_name=\"${var.namespace}\" AND metric.type=\"kubernetes.io/container/cpu/core_usage_time\""
                    aggregation = { alignmentPeriod = "60s", perSeriesAligner = "ALIGN_RATE", crossSeriesReducer = "REDUCE_SUM", groupByFields = ["resource.label.container_name"] }
                  }
                }
              }]
            }
          }
        }
      ]
    }
  })
}
