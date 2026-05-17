# EKS cluster + a single managed node group. Uses the canonical
# terraform-aws-modules/eks module — handles control-plane logging, OIDC
# provider for IRSA, security groups, and the addons (CoreDNS, kube-proxy,
# VPC CNI, EBS CSI driver) for us.

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.20"

  cluster_name    = local.name
  cluster_version = var.kubernetes_version

  # Public endpoint is on by default — gated by EKS auth (IAM-mapped users
  # only). Private-only is recommended for prod; expose via a bastion or VPN.
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Hardening (strict): lock the public endpoint to an operator-provided
  # CIDR allowlist (Tailnet + laptop public IP auto-detected by the
  # migrate wizard). Empty list keeps the default 0.0.0.0/0 with the
  # IAM gate as the only protection — fine for basic/off profiles.
  cluster_endpoint_public_access_cidrs = length(local.eks_public_access_cidrs) > 0 ? local.eks_public_access_cidrs : ["0.0.0.0/0"]

  # Allow the identity running terraform to admin the cluster. Without this,
  # `kubectl` against the cluster will fail with auth errors right after apply.
  enable_cluster_creator_admin_permissions = true

  # Envelope-encrypt Kubernetes Secrets with our own KMS key when
  # hardening is active. The EKS module wires the cluster role to the
  # key automatically when ``cluster_encryption_config`` is set.
  cluster_encryption_config = local.hardening_active_aws ? {
    provider_key_arn = aws_kms_key.eks_secrets[0].arn
    resources        = ["secrets"]
  } : {}

  # Send all control-plane audit-log types to CloudWatch Logs.
  cluster_enabled_log_types = local.hardening_active_aws ? [
    "api", "audit", "authenticator", "controllerManager", "scheduler",
  ] : []

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
  }

  eks_managed_node_group_defaults = {
    ami_type      = "AL2023_x86_64_STANDARD"
    capacity_type = "ON_DEMAND"
  }

  eks_managed_node_groups = {
    default = {
      name           = "${local.name}-default"
      instance_types = local.node_instance_types

      min_size     = max(local.node_desired_size - 1, 1)
      max_size     = local.node_desired_size + 2
      desired_size = local.node_desired_size

      disk_size = 50 # GiB. CrewAI's gateway image is ~3 GiB; leave headroom.
    }
  }
}

# IRSA role for the EBS CSI driver — required for PVC provisioning.
module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name             = "${local.name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}
