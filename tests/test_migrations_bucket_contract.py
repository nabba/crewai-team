"""Pinning test: the migrations-bundle bucket name must match across
the terraform that creates it and the Python that uploads to it.

Productization plan WP D Gap-2 fix (2026-05-17).

Without this pin, somebody could rename either side and the live
migrate's transfer step would fail post-`terraform apply` (leaving a
running cluster + nowhere to put the bundle). The exact contract:

    gs://andrusai-migrations-<project_id>/<run_id>/bundle.tar.gz
    ────────────────────────  ────────────  ───────────────────
    terraform bucket prefix    fan-out      object name

Both ends must agree on the prefix ``andrusai-migrations-`` and the
``<project_id>`` interpolation.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


REPO = Path(__file__).resolve().parent.parent
TF_FILE = REPO / "deploy" / "terraform" / "gcp" / "migrations_bucket.tf"
MIGRATION_PY = REPO / "app" / "substrate" / "migration.py"


def test_terraform_bucket_resource_exists():
    """The gap-2 fix added a google_storage_bucket. It must still be there."""
    assert TF_FILE.exists(), (
        f"{TF_FILE} missing — gap-2 fix regressed. The transfer step in "
        f"migration.py uploads to a bucket; without this terraform "
        f"resource, the bucket doesn't exist and live migrate fails."
    )
    src = TF_FILE.read_text()
    assert 'resource "google_storage_bucket" "migrations"' in src


def test_terraform_bucket_name_prefix():
    """Bucket name must be 'andrusai-migrations-<project_id>'."""
    src = TF_FILE.read_text()
    # Match: name = "andrusai-migrations-${var.project_id}"
    m = re.search(r'name\s*=\s*"andrusai-migrations-\$\{var\.project_id\}"', src)
    assert m is not None, (
        "bucket name must be exactly 'andrusai-migrations-${var.project_id}' "
        "— changing this string would break the contract with the "
        "_step_transfer_live URI in app/substrate/migration.py"
    )


def test_migration_py_uses_same_prefix():
    """``_step_transfer_live`` constructs gs://andrusai-migrations-<project_id>/..."""
    src = MIGRATION_PY.read_text()
    # The line in question:
    #   bucket = f"gs://andrusai-migrations-{project_id}"
    assert 'f"gs://andrusai-migrations-{project_id}"' in src, (
        "_step_transfer_live's GCP bucket URI must use exactly the same "
        "prefix the terraform creates. If you change one side, change "
        "both."
    )


def test_gateway_has_read_access_via_workload_identity():
    """The gateway pod's WI service account must be able to read bundles
    so the restore step (`python -m app.dr.import_kbs --bundle gs://...`)
    can pull from inside the cluster."""
    src = TF_FILE.read_text()
    assert 'google_storage_bucket_iam_member' in src, (
        "missing IAM binding — gateway pod cannot read the bundle "
        "without an objectViewer role on the bucket"
    )
    assert 'roles/storage.objectViewer' in src
    assert 'google_service_account.gateway.email' in src, (
        "IAM member must reference google_service_account.gateway, "
        "the existing Workload Identity SA used by the gateway pod"
    )


def test_bucket_lifecycle_cleanup_present():
    """30-day auto-delete keeps storage cost bounded if an operator
    abandons a migration. Pin this so a future edit doesn't silently
    remove the cleanup rule."""
    src = TF_FILE.read_text()
    assert "lifecycle_rule" in src
    # The age threshold should be defined; current default is 30 days.
    assert re.search(r"age\s*=\s*30", src), (
        "30-day lifecycle rule missing — bundles will accrue cost forever"
    )


def test_bucket_in_same_region_as_cluster():
    """Bucket region must follow var.region so the cluster + bucket are
    co-located (egress fee = 0 for in-region transfer)."""
    src = TF_FILE.read_text()
    assert re.search(r"location\s*=\s*var\.region", src), (
        "bucket location should be var.region for in-region egress savings"
    )
