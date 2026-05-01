# ECR repository for the gateway image. The install.sh dispatcher builds
# the image locally, tags it with the ECR URL, and pushes — Terraform doesn't
# do that itself (avoids burning a `terraform apply` cycle on every code change).

resource "aws_ecr_repository" "gateway" {
  name                 = "${local.name}-gateway"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
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
