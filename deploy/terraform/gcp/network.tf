# Custom VPC with secondary ranges for pods + services (required for
# IP aliasing on GKE). One subnet per region — Autopilot manages the
# nodes' AZ spread inside it.

resource "google_compute_network" "botarmy" {
  name                    = "${local.name}-vpc"
  auto_create_subnetworks = false
  routing_mode            = "REGIONAL"

  depends_on = [google_project_service.required]
}

resource "google_compute_subnetwork" "botarmy" {
  name                     = "${local.name}-subnet"
  ip_cidr_range            = var.vpc_cidr
  region                   = var.region
  network                  = google_compute_network.botarmy.id
  private_ip_google_access = true # nodes can hit Google APIs without external IP

  secondary_ip_range {
    range_name    = "${local.name}-pods"
    ip_cidr_range = var.pods_cidr
  }
  secondary_ip_range {
    range_name    = "${local.name}-services"
    ip_cidr_range = var.services_cidr
  }
}

# Cloud NAT so private nodes can pull images / hit the internet.
resource "google_compute_router" "botarmy" {
  name    = "${local.name}-router"
  region  = var.region
  network = google_compute_network.botarmy.id
}

resource "google_compute_router_nat" "botarmy" {
  name                               = "${local.name}-nat"
  router                             = google_compute_router.botarmy.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Private services connection — required for Cloud SQL via private IP.
# Without this, Cloud SQL only exposes a public IP and nothing in the VPC
# can reach it.
resource "google_compute_global_address" "private_ip_alloc" {
  name          = "${local.name}-cloudsql-private"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.botarmy.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.botarmy.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_alloc.name]

  depends_on = [google_project_service.required]
}
