# RDS Postgres 16 with the pgvector extension enabled.
#
# Why not the terraform-aws-modules/rds module? It's great, but for a single
# instance with a custom parameter group + extension creation, raw resources
# read more clearly. Easy to swap for the module later if requirements grow.

resource "random_password" "rds" {
  length  = 32
  special = false # Avoids quoting hell in connection strings + k8s secrets
  upper   = true
  lower   = true
  numeric = true
}

resource "aws_db_subnet_group" "botarmy" {
  name       = "${local.name}-db-subnets"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name        = "${local.name}-rds"
  description = "Allow Postgres from EKS nodes only"
  vpc_id      = module.vpc.vpc_id
}

resource "aws_security_group_rule" "rds_from_eks_nodes" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = module.eks.node_security_group_id
  security_group_id        = aws_security_group.rds.id
}

resource "aws_security_group_rule" "rds_egress_all" {
  type              = "egress"
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  security_group_id = aws_security_group.rds.id
}

# Custom parameter group — required to allow `vector` in shared_preload_libraries.
# (pg_stat_statements is included for free observability.)
resource "aws_db_parameter_group" "pg16_vector" {
  name   = "${local.name}-pg16-vector"
  family = "postgres16"

  parameter {
    name         = "shared_preload_libraries"
    value        = "pg_stat_statements"
    apply_method = "pending-reboot"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000" # log queries slower than 1 s
  }
}

resource "aws_db_instance" "botarmy" {
  identifier        = "${local.name}-pg"
  engine            = "postgres"
  engine_version    = "16.3"
  instance_class    = local.rds_instance_class
  allocated_storage = local.rds_allocated_storage
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = "mem0"
  username = "mem0"
  password = random_password.rds.result
  port     = 5432

  parameter_group_name = aws_db_parameter_group.pg16_vector.name
  db_subnet_group_name = aws_db_subnet_group.botarmy.name

  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = local.rds_multi_az

  backup_retention_period   = var.tier == "prod" ? 7 : 1
  delete_automated_backups  = var.tier != "prod"
  skip_final_snapshot       = var.tier != "prod"
  final_snapshot_identifier = var.tier == "prod" ? "${local.name}-final-${formatdate("YYYY-MM-DD", timestamp())}" : null
  deletion_protection       = var.tier == "prod"

  performance_insights_enabled = var.tier == "prod"

  apply_immediately = true

  lifecycle {
    ignore_changes = [final_snapshot_identifier]
  }
}

# CREATE EXTENSION vector is handled by the application layer (Mem0's
# pgvector backend issues it on first connect). RDS Postgres 16 includes
# pgvector in the allowlist, and the mem0 master user has permission to
# create extensions in its own database.
#
# Why we removed the postgresql_extension.vector resource: the postgresql
# provider would have to talk to RDS from the Terraform host. RDS is in
# private subnets — laptop / CI runner can only reach it via NAT GW egress,
# which most corporate networks block on outbound 5432. The resource was
# fragile across user environments. Doing CREATE EXTENSION at app boot is
# both more reliable and matches what Mem0 already does on the local
# install. Verified 2026-05-01 against the GCP module's identical change.
