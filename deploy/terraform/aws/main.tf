provider "aws" {
  region = var.region
  default_tags { tags = var.tags }
}

data "aws_availability_zones" "available" {
  filter {
    name   = "opt-in-status"
    values = ["opt-in-not-required"]
  }
}

data "aws_caller_identity" "current" {}

# ─── Tier → concrete sizing ───────────────────────────────────
# Users can pick `tier = "cheapest"` and get sane defaults, or override any
# individual variable for a custom mix.
locals {
  tier_defaults = {
    cheapest = {
      node_instance_types   = ["t3.medium"] # 2 vCPU / 4 GiB · ~$30/mo on-demand
      node_desired_size     = 2
      rds_instance_class    = "db.t4g.micro" # 2 vCPU / 1 GiB · ~$13/mo
      rds_allocated_storage = 20
      rds_multi_az          = false
    }
    prod = {
      node_instance_types   = ["m5.large"] # 2 vCPU / 8 GiB · ~$70/mo on-demand
      node_desired_size     = 3
      rds_instance_class    = "db.m5.large" # 2 vCPU / 8 GiB · ~$130/mo single-AZ
      rds_allocated_storage = 100
      rds_multi_az          = true
    }
  }

  # Resolved sizing — explicit var > tier default
  node_instance_types   = length(var.node_instance_types) > 0 ? var.node_instance_types : local.tier_defaults[var.tier].node_instance_types
  node_desired_size     = var.node_desired_size > 0 ? var.node_desired_size : local.tier_defaults[var.tier].node_desired_size
  rds_instance_class    = var.rds_instance_class != "" ? var.rds_instance_class : local.tier_defaults[var.tier].rds_instance_class
  rds_allocated_storage = var.rds_allocated_storage > 0 ? var.rds_allocated_storage : local.tier_defaults[var.tier].rds_allocated_storage
  rds_multi_az          = var.rds_multi_az != null ? var.rds_multi_az : local.tier_defaults[var.tier].rds_multi_az

  # 3 AZs for prod, 2 for cheapest
  azs = slice(data.aws_availability_zones.available.names, 0, var.tier == "prod" ? 3 : 2)

  name = var.cluster_name
}

# ─── Kubernetes + Helm providers ──────────────────────────────
# Authenticated via the EKS cluster's CA + a short-lived token. The token
# refresh happens on each apply, so don't cache state for too long.
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.region]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.region]
    }
  }
}

# Note: no postgresql provider is configured. RDS sits in private subnets and
# isn't reachable from the Terraform host on most networks. Mem0's pgvector
# backend runs `CREATE EXTENSION vector` on first connect, which is more
# reliable than threading the extension call through Terraform. See rds.tf.
