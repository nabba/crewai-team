# Cluster-level helpers we install before the BotArmy chart:
#
#   - aws-load-balancer-controller — turns Kubernetes Ingress objects into AWS
#     ALBs. Required for any external traffic.
#   - cert-manager — provisions TLS certs via Let's Encrypt (when no
#     acm_certificate_arn is provided).
#
# Then the BotArmy chart itself (gated on var.deploy_helm_chart so users can
# do `terraform apply` then run helm separately for tighter iteration).

# ─── IRSA for the ALB controller ──────────────────────────────
module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name                              = "${local.name}-alb-controller"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
}

resource "kubernetes_service_account" "alb_controller" {
  metadata {
    name      = "aws-load-balancer-controller"
    namespace = "kube-system"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.alb_controller_irsa.iam_role_arn
    }
  }
}

resource "helm_release" "alb_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.8.1"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }
  set {
    name  = "serviceAccount.create"
    value = "false"
  }
  set {
    name  = "serviceAccount.name"
    value = kubernetes_service_account.alb_controller.metadata[0].name
  }
  set {
    name  = "region"
    value = var.region
  }
  set {
    name  = "vpcId"
    value = module.vpc.vpc_id
  }

  depends_on = [module.eks]
}

# ─── cert-manager (only when ACM cert isn't provided) ─────────
resource "helm_release" "cert_manager" {
  count            = var.acm_certificate_arn == "" && var.domain != "" ? 1 : 0
  name             = "cert-manager"
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  namespace        = "cert-manager"
  version          = "v1.15.1"
  create_namespace = true

  set {
    name  = "installCRDs"
    value = "true"
  }
}

# ─── The BotArmy chart ────────────────────────────────────────
# Path is computed relative to this terraform module.
locals {
  botarmy_chart_path = "${path.module}/../../k8s"
}

resource "helm_release" "botarmy" {
  count = var.deploy_helm_chart ? 1 : 0

  name      = "botarmy"
  chart     = local.botarmy_chart_path
  namespace = kubernetes_namespace.botarmy.metadata[0].name

  # Don't block the apply on pod readiness. The gateway pod can't be Ready
  # until the dispatcher has built + pushed the image to ECR — which happens
  # AFTER terraform apply returns. The dispatcher's `kubectl rollout status`
  # call after the image push is the right gate. (Verified by an e2e test on
  # 2026-05-01: with wait=true, helm timed out at 5m on ImagePullBackOff.)
  wait    = false
  timeout = 600

  # The chart's values get cloud-shaped overrides — image points at ECR,
  # in-cluster Postgres is disabled (we use RDS), ingress is wired up.
  values = [yamlencode({
    image = {
      repository = aws_ecr_repository.gateway.repository_url
      tag        = var.gateway_image_tag
    }

    envSecretName = kubernetes_secret.botarmy_env.metadata[0].name

    gateway = {
      service = { type = "ClusterIP" }
      ingress = var.domain == "" ? { enabled = false } : {
        enabled   = true
        className = "alb"
        host      = var.domain
        annotations = merge(
          {
            "alb.ingress.kubernetes.io/scheme"       = "internet-facing"
            "alb.ingress.kubernetes.io/target-type"  = "ip"
            "alb.ingress.kubernetes.io/listen-ports" = "[{\"HTTP\": 80}, {\"HTTPS\": 443}]"
            "alb.ingress.kubernetes.io/ssl-redirect" = "443"
          },
          var.acm_certificate_arn != "" ? {
            "alb.ingress.kubernetes.io/certificate-arn" = var.acm_certificate_arn
          } : {}
        )
        tls = {
          enabled    = var.acm_certificate_arn == ""
          secretName = "botarmy-tls"
        }
      }
    }

    # Disable the in-cluster Postgres — we use RDS.
    postgres = { enabled = false }

    # Storage class for cloud — gp3 is the AWS default for EBS CSI.
    neo4j    = { persistence = { storageClass = "gp3" } }
    chromadb = { persistence = { storageClass = "gp3" } }

    # Wire ServiceMonitor + dashboards iff kube-prometheus-stack is installed.
    monitoring = {
      serviceMonitor = { enabled = var.enable_monitoring }
      prometheusRule = { enabled = var.enable_monitoring }
      dashboards     = { enabled = var.enable_monitoring }
    }
  })]

  depends_on = [
    helm_release.alb_controller,
    helm_release.kube_prometheus_stack,
    kubernetes_secret.botarmy_env,
    aws_db_instance.botarmy, # ensure DB exists before gateway pod connects
  ]
}
