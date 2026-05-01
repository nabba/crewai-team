# Artifact Registry — replaces deprecated Container Registry. The dispatcher
# (scripts/install/gcp.sh) builds the gateway image locally and pushes here.

resource "google_artifact_registry_repository" "gateway" {
  location      = var.region
  repository_id = "${local.name}-gateway"
  description   = "BotArmy gateway container images"
  format        = "DOCKER"

  cleanup_policies {
    id     = "keep-last-20"
    action = "KEEP"
    most_recent_versions {
      keep_count = 20
    }
  }

  cleanup_policies {
    id     = "delete-old-untagged"
    action = "DELETE"
    condition {
      tag_state  = "UNTAGGED"
      older_than = "604800s" # 7 days
    }
  }

  depends_on = [google_project_service.required]
}

# Computed URL the dispatcher uses for `docker push`.
locals {
  artifact_registry_url = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.gateway.repository_id}"
}
