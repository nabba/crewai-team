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

  # Allow the identity running terraform to admin the cluster. Without this,
  # `kubectl` against the cluster will fail with auth errors right after apply.
  enable_cluster_creator_admin_permissions = true

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
