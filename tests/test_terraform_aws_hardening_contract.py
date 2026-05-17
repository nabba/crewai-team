"""Pinning tests for the AWS terraform hardening contract. Mirrors
the GCP-side test_terraform_hardening_contract.py."""
from __future__ import annotations

from pathlib import Path

import pytest

AWS_DIR = Path(__file__).resolve().parents[1] / "deploy" / "terraform" / "aws"


def _read(name: str) -> str:
    p = AWS_DIR / name
    assert p.is_file(), f"missing terraform file: {p}"
    return p.read_text()


class TestHardeningFiles:
    def test_hardening_tf_exists(self):
        assert (AWS_DIR / "hardening.tf").is_file()

    def test_kms_tf_exists(self):
        assert (AWS_DIR / "kms.tf").is_file()

    def test_waf_tf_exists(self):
        assert (AWS_DIR / "waf.tf").is_file()

    def test_cloudtrail_tf_exists(self):
        assert (AWS_DIR / "cloudtrail.tf").is_file()

    def test_scp_tf_exists(self):
        assert (AWS_DIR / "scp.tf").is_file()


class TestVariables:
    def test_hardening_profile_validates(self):
        text = _read("variables.tf")
        assert 'variable "hardening_profile"' in text
        assert 'contains(["off", "basic", "strict"]' in text

    def test_allowed_cidrs_is_list_of_objects(self):
        text = _read("variables.tf")
        assert 'variable "allowed_cidrs"' in text
        assert "cidr_block" in text and "display_name" in text

    def test_aws_org_enabled_and_root_id(self):
        text = _read("variables.tf")
        assert 'variable "aws_org_enabled"' in text
        assert 'variable "aws_org_root_id"' in text


class TestEksWiring:
    def test_endpoint_allowlist_wired(self):
        text = _read("eks.tf")
        assert "cluster_endpoint_public_access_cidrs" in text
        assert "local.eks_public_access_cidrs" in text

    def test_secrets_encryption_with_cmek(self):
        text = _read("eks.tf")
        assert "cluster_encryption_config" in text
        assert "aws_kms_key.eks_secrets" in text

    def test_control_plane_logs_enabled(self):
        text = _read("eks.tf")
        assert "cluster_enabled_log_types" in text
        assert "audit" in text
        assert "authenticator" in text


class TestRdsWiring:
    def test_cmek_via_kms_key_id(self):
        text = _read("rds.tf")
        assert "kms_key_id" in text
        assert "aws_kms_key.rds" in text

    def test_iam_database_authentication(self):
        text = _read("rds.tf")
        assert "iam_database_authentication_enabled" in text

    def test_cloudwatch_logs_export(self):
        text = _read("rds.tf")
        assert "enabled_cloudwatch_logs_exports" in text


class TestEcrWiring:
    def test_immutable_tags_strict(self):
        text = _read("ecr.tf")
        assert "image_tag_mutability" in text
        assert "IMMUTABLE" in text

    def test_kms_encryption_when_hardening_active(self):
        text = _read("ecr.tf")
        assert "aws_kms_key.ecr" in text


class TestVpcWiring:
    def test_flow_logs_enabled(self):
        text = _read("vpc.tf")
        assert "enable_flow_log" in text
        assert "flow_log_cloudwatch_log_group_retention_in_days" in text


class TestWaf:
    def test_regional_wafv2_present(self):
        text = _read("waf.tf")
        assert "aws_wafv2_web_acl" in text
        assert 'scope       = "REGIONAL"' in text

    def test_rate_limit_rule(self):
        text = _read("waf.tf")
        assert "rate_based_statement" in text

    def test_aws_managed_rules(self):
        text = _read("waf.tf")
        for rule in (
            "AWSManagedRulesCommonRuleSet",
            "AWSManagedRulesKnownBadInputsRuleSet",
            "AWSManagedRulesSQLiRuleSet",
        ):
            assert rule in text, f"missing managed rule {rule}"


class TestCloudtrail:
    def test_multi_region_trail(self):
        text = _read("cloudtrail.tf")
        assert "aws_cloudtrail" in text
        assert "is_multi_region_trail" in text

    def test_log_file_validation(self):
        text = _read("cloudtrail.tf")
        assert "enable_log_file_validation" in text

    def test_s3_public_access_block(self):
        text = _read("cloudtrail.tf")
        assert "aws_s3_bucket_public_access_block" in text
        assert "restrict_public_buckets = true" in text


class TestGuardDuty:
    def test_detector(self):
        text = _read("hardening.tf")
        assert "aws_guardduty_detector" in text

    def test_securityhub(self):
        text = _read("hardening.tf")
        assert "aws_securityhub_account" in text


class TestScp:
    def test_deny_root(self):
        text = _read("scp.tf")
        assert "deny_root_actions" in text

    def test_require_mfa(self):
        text = _read("scp.tf")
        assert "require_mfa_for_iam" in text
        assert "aws:MultiFactorAuthPresent" in text

    def test_deny_security_disable(self):
        text = _read("scp.tf")
        assert "deny_security_disable" in text
        assert "cloudtrail:StopLogging" in text
        assert "guardduty:DeleteDetector" in text

    def test_deny_unwanted_regions(self):
        text = _read("scp.tf")
        assert "deny_unwanted_regions" in text


class TestKms:
    def test_four_keys(self):
        text = _read("kms.tf")
        for k in ("eks_secrets", "rds", "ecr", "secrets_manager"):
            assert f'"{k}"' in text, f"missing KMS key: {k}"

    def test_keys_are_prevent_destroy(self):
        text = _read("kms.tf")
        # 4 keys × prevent_destroy = 4 occurrences
        assert text.count("prevent_destroy = true") >= 4

    def test_key_rotation_enabled(self):
        text = _read("kms.tf")
        # terraform fmt normalizes whitespace; match on `=` rather than exact padding
        assert "enable_key_rotation" in text and "= true" in text
        # And every key should have it (one for each of the 4 KMS keys)
        assert text.count("enable_key_rotation") >= 4


class TestOutputs:
    def test_hardening_summary_output(self):
        text = _read("outputs.tf")
        assert 'output "hardening_summary"' in text
        assert "local.hardening_summary_aws" in text

    def test_kms_outputs(self):
        text = _read("outputs.tf")
        assert 'output "kms_eks_secrets_arn"' in text
        assert 'output "kms_rds_arn"' in text

    def test_cloudtrail_output(self):
        text = _read("outputs.tf")
        assert 'output "cloudtrail_bucket"' in text

    def test_waf_output(self):
        text = _read("outputs.tf")
        assert 'output "waf_acl_arn"' in text

    def test_eks_public_access_cidrs_output(self):
        text = _read("outputs.tf")
        assert 'output "eks_public_access_cidrs"' in text
