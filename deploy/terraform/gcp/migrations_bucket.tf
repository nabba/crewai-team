# ─────────────────────────────────────────────────────────────────
# Migrations bundle bucket (Productization plan WP D, 2026-05-17)
# ─────────────────────────────────────────────────────────────────
#
# Holds DR continuity bundles uploaded by ``botarmy migrate --live``.
# Each run lands under ``<run_id>/bundle.tar.gz``. The gateway pod's
# Workload-Identity service account reads from here during the restore
# step; the operator's local gcloud auth writes to here from
# ``app/substrate/migration.py:_step_transfer_live``.
#
# Bucket name is DETERMINISTIC: ``andrusai-migrations-<project_id>``.
# This matches the URI ``app/substrate/migration.py:_step_transfer_live``
# constructs (``gs://andrusai-migrations-<project_id>/<run_id>/...``).
# Don't change the prefix without also updating the migration step.
#
# Cleanup: ``terraform destroy`` removes the bucket. A 30-day lifecycle
# rule auto-deletes bundles so an abandoned migration doesn't accrue
# storage cost indefinitely.

resource "google_storage_bucket" "migrations" {
  name                        = "andrusai-migrations-${var.project_id}"
  location                    = var.region
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = true # let terraform destroy clean up bundles too

  # Auto-delete bundles after 30 days. Operator can override by
  # promoting the bundle elsewhere before then.
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30
    }
  }

  # Versioning OFF — bundles are immutable by run-id, no edit history needed.
  versioning {
    enabled = false
  }

  labels = var.labels
}

# Grant the gateway's Workload-Identity SA read access so the restore
# step (`kubectl exec ... python -m app.dr.import_kbs --bundle gs://...`)
# can pull the bundle from the cluster side.
resource "google_storage_bucket_iam_member" "migrations_gateway_reader" {
  bucket = google_storage_bucket.migrations.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.gateway.email}"
}
