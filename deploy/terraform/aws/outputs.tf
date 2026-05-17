output "cluster_name" {
  description = "EKS cluster name. Use with `aws eks update-kubeconfig`."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS API server endpoint."
  value       = module.eks.cluster_endpoint
}

output "cluster_region" {
  description = "AWS region the cluster lives in."
  value       = var.region
}

output "kubeconfig_command" {
  description = "Run this to point your local kubectl at the new cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "ecr_repository_url" {
  description = "Push the gateway image here: docker tag <local> <this>:<tag> && docker push <this>:<tag>"
  value       = aws_ecr_repository.gateway.repository_url
}

output "rds_endpoint" {
  description = "Postgres endpoint. Internal only — only reachable from EKS nodes."
  value       = aws_db_instance.botarmy.address
}

output "rds_port" {
  value = aws_db_instance.botarmy.port
}

output "secrets_manager_arn" {
  description = "AWS Secrets Manager ARN holding the full env payload."
  value       = aws_secretsmanager_secret.botarmy_env.arn
}

output "namespace" {
  value = var.namespace
}

output "ingress_hostname" {
  description = "ALB hostname assigned by AWS once the ingress is reconciled. Empty until then."
  value = try(
    [for ing in data.kubernetes_ingress_v1.botarmy[*] : ing.status[0].load_balancer[0].ingress[0].hostname][0],
    ""
  )
}

# Side-channel data lookup so `ingress_hostname` populates after the ALB
# reconciles. Returns empty in the same apply that creates the ingress.
data "kubernetes_ingress_v1" "botarmy" {
  count = var.deploy_helm_chart && var.domain != "" ? 1 : 0
  metadata {
    name      = "botarmy-${var.cluster_name}-gateway"
    namespace = var.namespace
  }
  depends_on = [helm_release.botarmy]
}

# ─── Hardening summary ────────────────────────────────────────
output "hardening_summary" {
  description = "Map of hardening primitives → enabled state for this apply."
  value       = local.hardening_summary_aws
}

output "kms_eks_secrets_arn" {
  description = "KMS key encrypting EKS Kubernetes Secrets. Empty when hardening_profile=off."
  value       = try(aws_kms_key.eks_secrets[0].arn, "")
}

output "kms_rds_arn" {
  description = "KMS key encrypting the RDS instance at rest. Empty when hardening_profile=off."
  value       = try(aws_kms_key.rds[0].arn, "")
}

output "cloudtrail_bucket" {
  description = "S3 bucket holding the CloudTrail audit log. Empty when hardening_profile != strict."
  value       = try(aws_s3_bucket.cloudtrail[0].bucket, "")
}

output "guardduty_detector_id" {
  description = "GuardDuty detector ID. Empty when hardening_profile != strict."
  value       = try(aws_guardduty_detector.this[0].id, "")
}

output "waf_acl_arn" {
  description = "WAFv2 Web ACL ARN to attach to the ALB. Empty when hardening_profile != strict."
  value       = try(aws_wafv2_web_acl.botarmy[0].arn, "")
}

output "eks_public_access_cidrs" {
  description = "CIDR allowlist applied to the EKS public endpoint."
  value       = local.eks_public_access_cidrs
}
