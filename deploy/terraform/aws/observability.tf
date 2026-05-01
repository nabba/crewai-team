# Observability — only installed when var.enable_monitoring is true.
#
# What you get:
#   - kube-prometheus-stack: Prometheus + Alertmanager + Grafana
#   - The BotArmy chart's ServiceMonitor + PrometheusRule + dashboard ConfigMap
#     get discovered automatically (release: kube-prometheus-stack labels)
#   - A CloudWatch dashboard for the AWS-managed pieces (RDS, ALB, NAT)
#
# Cost impact: ~$15/mo for the in-cluster Prometheus storage (10 GiB gp3 PVC).
# CloudWatch dashboard is free; CloudWatch metrics that drive it bill at
# ~$0.30/metric/month for custom metrics, but the ones we use are free
# AWS-published metrics.

variable "enable_monitoring" {
  description = "Install kube-prometheus-stack + provision a CloudWatch dashboard for infra metrics."
  type        = bool
  default     = true
}

variable "monitoring_storage_size" {
  description = "Persistent storage for Prometheus TSDB (in GiB). 10 GiB ≈ 7 days at 1k samples/s."
  type        = number
  default     = 10
}

# ─── Alert routing ────────────────────────────────────────────
# Both are optional. If neither is set, Alertmanager runs but routes alerts
# to a default null receiver (alerts visible in the UI, but no notifications).
#
# Get a Slack webhook at:
#   https://api.slack.com/messaging/webhooks
# Make two channels (e.g. #botarmy-critical and #botarmy-warning) and use
# one webhook for each, OR use a single webhook + let Alertmanager template
# the channel name from the severity label. We do the second.

variable "alertmanager_slack_webhook_url" {
  description = "Slack incoming-webhook URL for Alertmanager notifications. Empty = no Slack."
  type        = string
  default     = ""
  sensitive   = true
}

variable "alertmanager_slack_channel" {
  description = "Slack channel for routed alerts (single channel; severity goes in the message)."
  type        = string
  default     = "#botarmy-alerts"
}

variable "alertmanager_email_to" {
  description = "Email address to receive critical alerts. Empty = no email."
  type        = string
  default     = ""
}

variable "alertmanager_smtp_from" {
  description = "From-address for Alertmanager email. Required if alertmanager_email_to is set."
  type        = string
  default     = "alerts@botarmy.local"
}

variable "alertmanager_smtp_smarthost" {
  description = "SMTP relay host:port (e.g. smtp.sendgrid.net:587). Required if alertmanager_email_to is set."
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

# ── Opsgenie ─────────────────────────────────────────────────
# Standard pattern for production paging. Get a key at:
#   https://app.opsgenie.com/settings/integration/add/API
# (or the EU instance: https://app.eu.opsgenie.com/...).
# Critical alerts → P1 (paged, on-call wakes up).
# Warning alerts  → P3 (queued, no immediate notification).

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

# Computed Alertmanager config — composed from the variables above. Empty
# `receivers` list when no destinations are configured (Alertmanager still
# runs, just doesn't notify).
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
          matchers = ["severity = critical"]
          receiver = "critical"
          # Don't sit on a critical for the full group_wait — page sooner.
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
      {
        name = "default"
        # If nothing is configured, default is silent — alerts visible in the
        # Alertmanager UI but not pushed anywhere. Operators still notice via
        # the dashboards.
      },
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
            priority      = "P3" # Queued, no escalation
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

# ─── kube-prometheus-stack ────────────────────────────────────
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

    # Pick up our ServiceMonitors / PrometheusRules without label gymnastics.
    prometheus = {
      prometheusSpec = {
        serviceMonitorSelectorNilUsesHelmValues = false
        ruleSelectorNilUsesHelmValues           = false
        retention                               = "7d"
        storageSpec = {
          volumeClaimTemplate = {
            spec = {
              accessModes      = ["ReadWriteOnce"]
              storageClassName = "gp3"
              resources = {
                requests = { storage = "${var.monitoring_storage_size}Gi" }
              }
            }
          }
        }
        # Don't scrape kubelet's massive cAdvisor metric set on small clusters
        # — keeps memory bounded.
        scrapeInterval = "30s"
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
        storageClassName = "gp3"
      }
      # ALB-friendly: ClusterIP, expose via Ingress separately if needed.
      service = { type = "ClusterIP" }
    }

    alertmanager = merge(
      {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                accessModes      = ["ReadWriteOnce"]
                storageClassName = "gp3"
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

  depends_on = [module.eks]
}

resource "random_password" "grafana_admin" {
  count   = var.enable_monitoring ? 1 : 0
  length  = 24
  special = false
}

# Surface the Grafana password in outputs (and AWS Secrets Manager so it's
# discoverable in the AWS console too).
resource "aws_secretsmanager_secret" "grafana_admin" {
  count                   = var.enable_monitoring ? 1 : 0
  name                    = "${local.name}-grafana-admin"
  description             = "Grafana admin password (kube-prometheus-stack)"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "grafana_admin" {
  count         = var.enable_monitoring ? 1 : 0
  secret_id     = aws_secretsmanager_secret.grafana_admin[0].id
  secret_string = random_password.grafana_admin[0].result
}

# ─── CloudWatch dashboard for AWS-managed pieces ──────────────
# Covers what Prometheus can't see (and what AWS gives us for free): RDS
# CPU/connections, ALB request count, NAT GW bytes.
resource "aws_cloudwatch_dashboard" "botarmy" {
  count          = var.enable_monitoring ? 1 : 0
  dashboard_name = "${local.name}-infra"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "RDS CPU"
          region = var.region
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.botarmy.id]
          ]
          period = 60
          stat   = "Average"
          yAxis  = { left = { min = 0, max = 100 } }
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "RDS connections / free memory"
          region = var.region
          metrics = [
            ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", aws_db_instance.botarmy.id],
            [".", "FreeableMemory", ".", "."]
          ]
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "RDS read/write IOPS"
          region = var.region
          metrics = [
            ["AWS/RDS", "ReadIOPS", "DBInstanceIdentifier", aws_db_instance.botarmy.id],
            [".", "WriteIOPS", ".", "."]
          ]
          period = 60
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "NAT GW bytes (egress to internet)"
          region = var.region
          metrics = [
            ["AWS/NATGateway", "BytesOutToDestination", { "stat" = "Sum" }]
          ]
          period = 300
        }
      }
    ]
  })
}
