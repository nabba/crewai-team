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

  deletion_protection = false # toggle for prod manually

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

    database_flags {
      name  = "cloudsql.enable_pgvector"
      value = "on"
    }

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

  depends_on = [google_service_networking_connection.private_vpc_connection]
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

# CREATE EXTENSION vector — runs from the laptop / runner via the
# private IP. Requires that the Terraform host can route to the VPC
# (Cloud VPN, IAP TCP forwarding, or running TF from a Cloud Build job
# inside the project). See README troubleshooting if this step times out.
resource "postgresql_extension" "vector" {
  name     = "vector"
  database = google_sql_database.mem0.name

  depends_on = [
    google_sql_database.mem0,
    google_sql_user.mem0,
  ]
}
