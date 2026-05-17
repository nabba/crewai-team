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

# ─── Hardening summary ───────────────────────────────────────────
# Operator-readable map of every hardening primitive that's active in
# this plan. ``cloud_doctor.verify_hardening`` reads this output back
# after apply to confirm the policies actually landed.
output "hardening_summary" {
  description = "Map of hardening primitives → enabled state for this apply."
  value       = local.hardening_summary
}

output "cmek_keyring_name" {
  description = "Fully-qualified name of the CMEK keyring. Empty when hardening_profile=off."
  value       = try(google_kms_key_ring.botarmy[0].id, "")
}

output "audit_log_bucket" {
  description = "GCS bucket holding the cloud-audit-log sink. Empty when hardening_profile != strict."
  value       = try(google_storage_bucket.audit_logs[0].name, "")
}

output "audit_log_bq_dataset" {
  description = "BigQuery dataset for queryable audit logs. Empty when hardening_profile != strict."
  value       = try(google_bigquery_dataset.audit_logs[0].dataset_id, "")
}

output "cloud_armor_policy" {
  description = "Cloud Armor security policy attached to ingress. Empty when hardening_profile != strict."
  value       = try(google_compute_security_policy.botarmy[0].name, "")
}

output "binauthz_policy_mode" {
  description = "Effective Binary Authorization mode. 'disabled' when hardening_profile != strict."
  value       = local.hardening_strict ? var.binauthz_mode : "disabled"
}

output "master_authorized_cidrs" {
  description = "CIDR allowlist applied to the GKE master endpoint."
  value       = [for c in local.master_authorized_networks : c.cidr_block]
}
