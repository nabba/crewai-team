# GKE Autopilot — cheapest path: no node groups to manage, charges per pod
# resource request. Good fit for BotArmy's modest, fairly steady workload.
#
# In `cheapest` tier: zonal cluster (single zone, control plane is free).
# In `prod` tier: regional cluster (multi-zone control plane, ~$73/mo).

resource "google_container_cluster" "botarmy" {
  provider = google-beta
  name     = local.name

  # Autopilot REQUIRES regional clusters — zonal Autopilot is rejected by the
  # API ("Autopilot clusters must be regional clusters"). Standard mode would
  # allow zonal at the cost of node management; we keep Autopilot for both
  # tiers to give a single operational model. Regional Autopilot's control
  # plane bills the same $0.10/hr as Standard regional, and GCP gives every
  # billing account a $74.40/mo credit that covers exactly one cluster.
  location = var.region

  enable_autopilot    = true
  deletion_protection = false # turn on for prod by hand if you want

  network    = google_compute_network.botarmy.id
  subnetwork = google_compute_subnetwork.botarmy.id

  ip_allocation_policy {
    cluster_secondary_range_name  = google_compute_subnetwork.botarmy.secondary_ip_range[0].range_name
    services_secondary_range_name = google_compute_subnetwork.botarmy.secondary_ip_range[1].range_name
  }

  # Workload Identity is on by default for Autopilot — exposes the GKE
  # metadata server to pods so they can mint short-lived GCP tokens.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Release channel — Autopilot picks versions; we just say "give me REGULAR".
  release_channel {
    channel = "REGULAR"
  }

  # Private nodes (no external IPs) but public control-plane endpoint
  # (still IAM-gated). Switch to fully private for prod.
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  # Force IPv4 — IPv6 dual-stack adds complexity without value here.
  datapath_provider = "ADVANCED_DATAPATH" # GKE Dataplane V2 (Cilium-based)

  depends_on = [
    google_project_service.required,
    google_service_networking_connection.private_vpc_connection,
  ]
}

# Service account that the gateway pod will impersonate via Workload Identity.
# Right now it has no permissions — add roles here if BotArmy needs to call
# GCS / Pub/Sub / etc.
resource "google_service_account" "gateway" {
  account_id   = "${local.name}-gateway"
  display_name = "BotArmy gateway pod identity"
}

# IAM binding so the Kubernetes ServiceAccount can act as the GCP SA.
# The Workload Identity pool `<project>.svc.id.goog` is created when the
# Autopilot cluster comes up — we therefore depend on the cluster explicitly.
# Without this depends_on, terraform parallelises and the IAM binding can
# race ahead, hitting "Identity Pool does not exist (<project>.svc.id.goog)".
resource "google_service_account_iam_member" "gateway_workload_identity" {
  service_account_id = google_service_account.gateway.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.namespace}/botarmy-gateway]"

  depends_on = [google_container_cluster.botarmy]
}
