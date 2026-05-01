# VPC built from the well-maintained terraform-aws-modules/vpc module.
# 2 AZs (cheapest) or 3 AZs (prod), with a /20 private subnet per AZ for
# the EKS nodes + RDS, and a /24 public subnet per AZ for the ALB.
#
# The module creates a single NAT gateway in cheapest mode (saves ~$30/mo per
# extra NAT GW × AZ) and one-per-AZ in prod (HA egress).

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.8"

  name = "${local.name}-vpc"
  cidr = var.vpc_cidr
  azs  = local.azs

  private_subnets = [for k, _ in local.azs : cidrsubnet(var.vpc_cidr, 4, k)]
  public_subnets  = [for k, _ in local.azs : cidrsubnet(var.vpc_cidr, 8, k + 48)]

  enable_nat_gateway     = true
  single_nat_gateway     = var.tier == "cheapest"
  one_nat_gateway_per_az = var.tier == "prod"

  enable_dns_hostnames = true
  enable_dns_support   = true

  # EKS / load-balancer-controller subnet discovery tags
  public_subnet_tags = {
    "kubernetes.io/role/elb"              = 1
    "kubernetes.io/cluster/${local.name}" = "shared"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"     = 1
    "kubernetes.io/cluster/${local.name}" = "shared"
  }
}
