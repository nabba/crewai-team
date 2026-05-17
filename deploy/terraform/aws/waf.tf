# AWS WAFv2 — REGIONAL scope (for ALB). Cloud Armor mirror for AWS.
# Only created at hardening_profile=strict. Attached via an ALB
# WebACL-association in helm.tf when an ingress exists.

resource "aws_wafv2_web_acl" "botarmy" {
  count = local.hardening_strict_aws ? 1 : 0

  name        = "${local.name}-waf"
  description = "BotArmy ingress WAF (rate-limit + AWS Managed Rules)"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  # Rule 1: per-IP rate-limit
  rule {
    name     = "rate-limit-${local.name}"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = var.waf_rate_limit_per_5min
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-rate-limit"
      sampled_requests_enabled   = true
    }
  }

  # Rule 2: AWS Managed Rules — Common Rule Set (OWASP top 10)
  rule {
    name     = "aws-common"
    priority = 10

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesCommonRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-common"
      sampled_requests_enabled   = true
    }
  }

  # Rule 3: AWS Managed Rules — Known bad inputs
  rule {
    name     = "aws-known-bad-inputs"
    priority = 20

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-known-bad-inputs"
      sampled_requests_enabled   = true
    }
  }

  # Rule 4: AWS Managed Rules — SQL injection
  rule {
    name     = "aws-sqli"
    priority = 30

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesSQLiRuleSet"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "${local.name}-sqli"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.name}-waf"
    sampled_requests_enabled   = true
  }

  tags = var.tags
}
