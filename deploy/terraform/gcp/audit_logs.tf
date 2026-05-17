# Audit-log sink: send Cloud Audit Logs to a dedicated GCS bucket so the
# operator has a forensic trail independent of the live Logging service.
# Retention is controlled by var.audit_log_retention_days (default 400 d).
#
# Only created at hardening_profile=strict.
#
# Two sinks:
#   1. cloudaudit.googleapis.com/* (admin / data / system events) → GCS
#   2. all audit logs → BigQuery (queryable; ~$5/mo for a personal-scale
#      project's audit volume). Opt-out by setting var.audit_log_retention_days=0.

resource "google_storage_bucket" "audit_logs" {
  count                       = local.hardening_strict ? 1 : 0
  name                        = "${var.project_id}-${local.name}-audit-logs"
  location                    = var.region
  storage_class               = "STANDARD"
  force_destroy               = false
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = var.audit_log_retention_days
    }
  }

  retention_policy {
    retention_period = var.audit_log_retention_days * 86400 # days → seconds
  }

  labels = var.labels

  depends_on = [google_project_service.required]
}

resource "google_logging_project_sink" "audit_logs_gcs" {
  count                  = local.hardening_strict ? 1 : 0
  name                   = "${local.name}-audit-logs-gcs"
  destination            = "storage.googleapis.com/${google_storage_bucket.audit_logs[0].name}"
  filter                 = "logName:\"cloudaudit.googleapis.com\""
  unique_writer_identity = true
}

# Sink writer needs Storage Object Creator on the bucket — otherwise the
# sink runs but every log dropped.
resource "google_storage_bucket_iam_member" "audit_log_sink_writer" {
  count  = local.hardening_strict ? 1 : 0
  bucket = google_storage_bucket.audit_logs[0].name
  role   = "roles/storage.objectCreator"
  member = google_logging_project_sink.audit_logs_gcs[0].writer_identity
}

# ─── BigQuery dataset for querying audit logs ─────────────────────
resource "google_bigquery_dataset" "audit_logs" {
  count                       = local.hardening_strict ? 1 : 0
  dataset_id                  = "${replace(local.name, "-", "_")}_audit_logs"
  location                    = var.region
  description                 = "Audit-log sink target for ${local.name}"
  default_table_expiration_ms = var.audit_log_retention_days * 86400000
  labels                      = var.labels

  depends_on = [google_project_service.required]
}

resource "google_logging_project_sink" "audit_logs_bq" {
  count                  = local.hardening_strict ? 1 : 0
  name                   = "${local.name}-audit-logs-bq"
  destination            = "bigquery.googleapis.com/projects/${var.project_id}/datasets/${google_bigquery_dataset.audit_logs[0].dataset_id}"
  filter                 = "logName:\"cloudaudit.googleapis.com\""
  unique_writer_identity = true

  bigquery_options {
    use_partitioned_tables = true
  }
}

resource "google_bigquery_dataset_iam_member" "audit_log_sink_bq_writer" {
  count      = local.hardening_strict ? 1 : 0
  dataset_id = google_bigquery_dataset.audit_logs[0].dataset_id
  role       = "roles/bigquery.dataEditor"
  member     = google_logging_project_sink.audit_logs_bq[0].writer_identity
}
