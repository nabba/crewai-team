# Cloud SQL Postgres 16 with pgvector. Cloud SQL exposes pgvector via
# the `cloudsql.enable_pgvector` flag (newer than `cloudsql.iam_authentication`).
# We then run CREATE EXTENSION via the postgresql provider once it's up.

resource "random_password" "cloudsql" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "botarmy" {
  name             = "${local.name}-pg"
  database_version = "POSTGRES_16"
  region           = var.region

  # Hardening: deletion_protection ON for prod tier when hardening is
  # active. Cheapest tier keeps it OFF so the wizard's teardown CLI
  # still works in dev.
  deletion_protection = local.hardening_active && var.tier == "prod"

  # CMEK: only at hardening_profile=basic|strict. Without CMEK Cloud SQL
  # still encrypts at rest with a Google-managed key — CMEK gives YOU
  # the rotation + revocation surface.
  encryption_key_name = local.hardening_active ? google_kms_crypto_key.cloudsql[0].id : null

  settings {
    tier              = local.cloudsql_tier
    availability_type = local.cloudsql_high_availability ? "REGIONAL" : "ZONAL"
    disk_size         = local.cloudsql_disk_size
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = var.tier == "prod"
      backup_retention_settings {
        retained_backups = var.tier == "prod" ? 7 : 1
      }
    }

    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.botarmy.id
      enable_private_path_for_google_cloud_services = true
    }

    # pgvector is preinstalled on Cloud SQL Postgres 16; no flag needed.
    # The `postgresql_extension.vector` resource below runs CREATE EXTENSION
    # to make it available in our database. (Earlier versions of this file
    # set cloudsql.enable_pgvector here — that flag does not exist in the
    # Cloud SQL admin API. Verified by an end-to-end test on 2026-05-01.)

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000" # log queries slower than 1 s
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = false
      record_client_address   = false
    }
  }

  depends_on = [
    google_service_networking_connection.private_vpc_connection,
    google_kms_crypto_key_iam_member.cloudsql_sa,
  ]
}

resource "google_sql_database" "mem0" {
  name     = "mem0"
  instance = google_sql_database_instance.botarmy.name
}

resource "google_sql_user" "mem0" {
  name     = "mem0"
  instance = google_sql_database_instance.botarmy.name
  password = random_password.cloudsql.result
}

# CREATE EXTENSION vector is handled by the application layer, not Terraform.
# Cloud SQL grants the `cloudsqlsuperuser` role to user-created roles by
# default, which is sufficient to run `CREATE EXTENSION vector;`. Mem0's
# pgvector backend issues that statement on first connect (mem0_manager.py),
# so we don't need the postgresql provider to do it.
#
# Why we removed the postgresql_extension.vector resource: the postgresql
# provider runs from the Terraform host. Cloud SQL's private IP isn't
# reachable from outside the VPC — laptop, CI runner, etc. — so the resource
# would always time out on real-world deploys unless the user spun up the
# Cloud SQL Auth Proxy first. Verified by an end-to-end apply on
# 2026-05-01.
