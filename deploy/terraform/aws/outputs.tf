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
