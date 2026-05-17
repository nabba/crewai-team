# ECR repository for the gateway image. The install.sh dispatcher builds
# the image locally, tags it with the ECR URL, and pushes — Terraform doesn't
# do that itself (avoids burning a `terraform apply` cycle on every code change).

resource "aws_ecr_repository" "gateway" {
  name = "${local.name}-gateway"
  # Strict: immutable tags so a malicious push can't replace a known-good
  # image with the same tag. Basic/off keep MUTABLE for dev ergonomics.
  image_tag_mutability = local.hardening_strict_aws ? "IMMUTABLE" : "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # Customer-managed KMS when hardening is active. Defaults to AES256
  # (AWS-managed key) when off.
  encryption_configuration {
    encryption_type = local.hardening_active_aws ? "KMS" : "AES256"
    kms_key         = local.hardening_active_aws ? aws_kms_key.ecr[0].arn : null
  }
}

# Keep at most 20 images — drop the oldest so the registry doesn't grow forever.
resource "aws_ecr_lifecycle_policy" "gateway" {
  repository = aws_ecr_repository.gateway.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 20 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}
