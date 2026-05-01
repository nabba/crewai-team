output "cluster_name" {
  value = google_container_cluster.botarmy.name
}

output "cluster_location" {
  description = "Zone (cheapest tier) or region (prod tier) where the cluster lives."
  value       = google_container_cluster.botarmy.location
}

output "project_id" {
  value = var.project_id
}

output "kubeconfig_command" {
  description = "Run this to point your local kubectl at the new cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.botarmy.name} --location ${google_container_cluster.botarmy.location} --project ${var.project_id}"
}

output "artifact_registry_url" {
  description = "Push the gateway image here: docker tag <local> <this>/gateway:<tag> && docker push <this>/gateway:<tag>"
  value       = local.artifact_registry_url
}

output "cloudsql_instance" {
  value = google_sql_database_instance.botarmy.name
}

output "cloudsql_private_ip" {
  description = "Private IP — only reachable inside the VPC (or via Cloud SQL Auth Proxy)."
  value       = google_sql_database_instance.botarmy.private_ip_address
}

output "secret_manager_id" {
  value = google_secret_manager_secret.botarmy_env.id
}

output "namespace" {
  value = var.namespace
}

output "ingress_ip" {
  description = "Static IP attached to the Ingress (assign your DNS A record to this)."
  value       = try(google_compute_global_address.botarmy_ingress[0].address, "")
}
