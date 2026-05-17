"""Pinning tests for the GCP terraform hardening contract.

We deliberately don't run `terraform plan` here — that would require the
google provider + auth + a real project. Instead, we pin specific HCL
strings that must be present, which catches the class of regression
where someone refactors the module and accidentally drops a hardening
block. Composes with the `terraform validate` pass run by hand during
development.
"""
from __future__ import annotations

from pathlib import Path

import pytest

GCP_DIR = Path(__file__).resolve().parents[1] / "deploy" / "terraform" / "gcp"


def _read(name: str) -> str:
    p = GCP_DIR / name
    assert p.is_file(), f"missing terraform file: {p}"
    return p.read_text()


class TestHardeningFiles:
    def test_hardening_tf_exists(self):
        assert (GCP_DIR / "hardening.tf").is_file()

    def test_kms_tf_exists(self):
        assert (GCP_DIR / "kms.tf").is_file()

    def test_cloud_armor_tf_exists(self):
        assert (GCP_DIR / "cloud_armor.tf").is_file()

    def test_audit_logs_tf_exists(self):
        assert (GCP_DIR / "audit_logs.tf").is_file()

    def test_org_policies_tf_exists(self):
        assert (GCP_DIR / "org_policies.tf").is_file()


class TestVariables:
    def test_hardening_profile_variable_with_validation(self):
        text = _read("variables.tf")
        assert 'variable "hardening_profile"' in text
        # Refuses anything outside {off, basic, strict}
        assert 'contains(["off", "basic", "strict"]' in text

    def test_allowed_cidrs_variable_is_list_of_objects(self):
        text = _read("variables.tf")
        assert 'variable "allowed_cidrs"' in text
        assert "cidr_block" in text and "display_name" in text

    def test_binauthz_mode_variable_with_validation(self):
        text = _read("variables.tf")
        assert 'variable "binauthz_mode"' in text
        assert 'contains(["AUDIT", "ENFORCE"]' in text

    def test_org_id_variable_present(self):
        text = _read("variables.tf")
        assert 'variable "org_id"' in text


class TestGkeWiring:
    def test_master_authorized_networks_block(self):
        text = _read("gke.tf")
        assert "master_authorized_networks_config" in text
        # Must use the local computed from hardening.tf
        assert "local.master_authorized_networks" in text

    def test_etcd_cmek_block(self):
        text = _read("gke.tf")
        assert "database_encryption" in text
        assert "google_kms_crypto_key.gke_etcd" in text

    def test_binary_authorization_block(self):
        text = _read("gke.tf")
        assert "binary_authorization" in text
        assert "PROJECT_SINGLETON_POLICY_ENFORCE" in text

    def test_deletion_protection_only_for_prod(self):
        text = _read("gke.tf")
        # "hardening_active && var.tier == 'prod'" is the conjunction
        assert 'var.tier == "prod"' in text


class TestCloudSqlWiring:
    def test_cmek_reference(self):
        text = _read("cloudsql.tf")
        assert "encryption_key_name" in text
        assert "google_kms_crypto_key.cloudsql" in text


class TestCloudArmor:
    def test_security_policy_present(self):
        text = _read("cloud_armor.tf")
        assert 'google_compute_security_policy' in text
        # OWASP rules
        for rule in ("sqli-v33-stable", "xss-v33-stable", "lfi-v33-stable", "rce-v33-stable"):
            assert rule in text, f"missing OWASP rule {rule}"
        # Rate-limit + ban
        assert "rate_based_ban" in text
        assert "deny(429)" in text


class TestAuditLogs:
    def test_gcs_sink(self):
        text = _read("audit_logs.tf")
        assert 'google_storage_bucket' in text
        assert 'public_access_prevention    = "enforced"' in text
        assert 'cloudaudit.googleapis.com' in text

    def test_bigquery_sink(self):
        text = _read("audit_logs.tf")
        assert 'google_bigquery_dataset' in text
        assert 'google_logging_project_sink' in text


class TestOrgPolicies:
    def test_blocks_sa_key_creation(self):
        text = _read("org_policies.tf")
        assert "iam.disableServiceAccountKeyCreation" in text

    def test_requires_os_login(self):
        text = _read("org_policies.tf")
        assert "compute.requireOsLogin" in text

    def test_denies_external_vm_ips(self):
        text = _read("org_policies.tf")
        assert "compute.vmExternalIpAccess" in text
        assert 'deny_all = "TRUE"' in text

    def test_restricts_cloudsql_public_ip(self):
        text = _read("org_policies.tf")
        assert "sql.restrictPublicIp" in text

    def test_only_active_when_org_id_and_strict(self):
        text = _read("org_policies.tf")
        # The local that gates every policy
        assert "org_policies_active" in text
        assert 'local.hardening_strict' in text
        assert 'var.org_id != ""' in text


class TestKmsCmek:
    def test_keyring_and_four_keys(self):
        text = _read("kms.tf")
        assert "google_kms_key_ring" in text
        for key in ("cloudsql", "gke_etcd", "artifact_registry", "secret_manager"):
            assert f'"{key}"' in text, f"missing CMEK key: {key}"

    def test_keys_are_prevent_destroy(self):
        # Keys MUST set prevent_destroy or losing terraform state destroys
        # the operator's data-at-rest.
        text = _read("kms.tf")
        assert text.count("prevent_destroy = true") >= 4

    def test_keys_rotate_quarterly(self):
        text = _read("kms.tf")
        # 7_776_000 seconds == 90 days
        assert '"7776000s"' in text


class TestOutputs:
    def test_hardening_summary_output(self):
        text = _read("outputs.tf")
        assert 'output "hardening_summary"' in text
        assert "local.hardening_summary" in text

    def test_cmek_keyring_output(self):
        text = _read("outputs.tf")
        assert 'output "cmek_keyring_name"' in text

    def test_audit_log_outputs(self):
        text = _read("outputs.tf")
        assert 'output "audit_log_bucket"' in text
        assert 'output "audit_log_bq_dataset"' in text

    def test_binauthz_mode_output(self):
        text = _read("outputs.tf")
        assert 'output "binauthz_policy_mode"' in text
