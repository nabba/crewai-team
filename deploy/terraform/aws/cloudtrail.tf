# CloudTrail multi-region audit log → encrypted S3 bucket. Mirrors
# deploy/terraform/gcp/audit_logs.tf shape.
#
# Strict only — basic profile relies on EKS control-plane logging +
# RDS logs already configured by the modules.

resource "aws_s3_bucket" "cloudtrail" {
  count         = local.hardening_strict_aws ? 1 : 0
  bucket        = "${data.aws_caller_identity.current.account_id}-${local.name}-cloudtrail"
  force_destroy = false

  tags = var.tags
}

resource "aws_s3_bucket_versioning" "cloudtrail" {
  count  = local.hardening_strict_aws ? 1 : 0
  bucket = aws_s3_bucket.cloudtrail[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail" {
  count  = local.hardening_strict_aws ? 1 : 0
  bucket = aws_s3_bucket.cloudtrail[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail" {
  count  = local.hardening_strict_aws ? 1 : 0
  bucket = aws_s3_bucket.cloudtrail[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudtrail" {
  count  = local.hardening_strict_aws ? 1 : 0
  bucket = aws_s3_bucket.cloudtrail[0].id

  rule {
    id     = "expire"
    status = "Enabled"

    filter {}

    expiration {
      days = var.cloudtrail_retention_days
    }
  }
}

resource "aws_s3_bucket_policy" "cloudtrail" {
  count  = local.hardening_strict_aws ? 1 : 0
  bucket = aws_s3_bucket.cloudtrail[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.cloudtrail[0].arn
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.cloudtrail[0].arn}/AWSLogs/${data.aws_caller_identity.current.account_id}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
    ]
  })
}

resource "aws_cloudtrail" "this" {
  count          = local.hardening_strict_aws ? 1 : 0
  name           = "${local.name}-cloudtrail"
  s3_bucket_name = aws_s3_bucket.cloudtrail[0].id

  is_multi_region_trail         = true
  enable_log_file_validation    = true
  include_global_service_events = true

  # Data events on S3 buckets — operator can adjust per cost concern.
  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }

  tags = var.tags

  depends_on = [aws_s3_bucket_policy.cloudtrail]
}
