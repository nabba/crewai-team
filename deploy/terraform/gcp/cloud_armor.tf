# Cloud Armor security policy. Attached to the ingress backend service
# in helm.tf (BackendConfig CRD); only created at hardening_profile=strict.
#
# Layers (highest priority = lowest number):
#   1000 — Per-IP rate-limit (var.cloud_armor_rate_limit_rpm req/minute)
#   1100 — OWASP-style preconfigured WAF: SQL injection
#   1110 — OWASP-style preconfigured WAF: cross-site scripting
#   1120 — OWASP-style preconfigured WAF: local file inclusion
#   1130 — OWASP-style preconfigured WAF: remote code execution
#   2147483647 — Default rule: ALLOW (so legitimate traffic passes)
#
# Tuning notes:
#   * RATE_BASED_BAN with 10-minute ban window is enough to fend off a
#     casual scanner without paging the operator on every burst.
#   * preconfigured_waf_config sensitivity=1 keeps false-positives low
#     for a personal-assistant workload. Bump to 4 once you have
#     monitoring of triggered rules.

resource "google_compute_security_policy" "botarmy" {
  count = local.hardening_strict ? 1 : 0
  name  = "${local.name}-armor"
  type  = "CLOUD_ARMOR"

  # 1000 — rate limit
  rule {
    action   = "rate_based_ban"
    priority = 1000
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = var.cloud_armor_rate_limit_rpm
        interval_sec = 60
      }
      ban_duration_sec = 600
    }
    description = "Per-IP RPM throttle (default ${var.cloud_armor_rate_limit_rpm}/min, 10-min ban)"
  }

  # 1100 — SQL injection
  rule {
    action   = "deny(403)"
    priority = 1100
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('sqli-v33-stable', {'sensitivity': 1})"
      }
    }
    description = "OWASP SQL injection"
  }

  # 1110 — XSS
  rule {
    action   = "deny(403)"
    priority = 1110
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('xss-v33-stable', {'sensitivity': 1})"
      }
    }
    description = "OWASP XSS"
  }

  # 1120 — Local file inclusion
  rule {
    action   = "deny(403)"
    priority = 1120
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('lfi-v33-stable', {'sensitivity': 1})"
      }
    }
    description = "OWASP LFI"
  }

  # 1130 — Remote code execution
  rule {
    action   = "deny(403)"
    priority = 1130
    match {
      expr {
        expression = "evaluatePreconfiguredWaf('rce-v33-stable', {'sensitivity': 1})"
      }
    }
    description = "OWASP RCE"
  }

  # Default — required rule
  rule {
    action   = "allow"
    priority = 2147483647
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default: allow"
  }

  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
    }
  }

  depends_on = [google_project_service.required]
}
