"""
s3.py — S3-compatible off-host uploader for the per-KB source ledger.

PROGRAM §56 (2026-05-17). Works against any S3-compatible service —
AWS S3, Backblaze B2, Wasabi, MinIO, Cloudflare R2, etc. The shape
deliberately avoids AWS-specific features (no SSE-KMS, no lifecycle
rules baked in); plain PUT semantics that every S3 clone supports.

Configuration via env (operator-managed, never auto-set)::

    LEDGER_S3_BUCKET           required
    LEDGER_S3_ENDPOINT_URL     optional — for non-AWS providers
    LEDGER_S3_REGION           optional, defaults to "auto"
    LEDGER_S3_ACCESS_KEY_ID    fall back to env-default AWS chain
    LEDGER_S3_SECRET_ACCESS_KEY
    LEDGER_HOST_TAG            optional, prefixes the object key

Activated by ``runtime_settings.chromadb_ledger_s3_upload_enabled``
(default OFF). When OFF, every function in this module is a no-op
that returns a ``skipped=True`` result.

Failure modes
=============

* No bucket configured → returns ``skipped`` (operator hasn't wired
  the credentials yet; not an error).
* boto3 not installed → returns ``skipped`` (the package isn't a
  hard requirement; operator installs when they enable the feature).
* Upload error (network, 403, etc.) → returns ``ok=False`` with the
  error; the next pass retries from the same offset.

All errors are logged but never re-raised. Off-host is best-effort
on top of Q17.1 warm-spare which is the primary recovery surface.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from app.memory.source_ledger_offhost.common import (
    OffhostUploadResult,
    canonical_object_key,
    incremental_payload,
    record_upload,
)

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    """Master gate via runtime_settings. Default OFF.

    Setting this ON without configuring ``LEDGER_S3_BUCKET`` is OK;
    the uploader silently skips with ``skipped=True``.
    """
    try:
        from app.runtime_settings import get_chromadb_ledger_s3_upload_enabled
        return bool(get_chromadb_ledger_s3_upload_enabled())
    except Exception:
        return False


def _config() -> dict:
    """Read S3 connection config from env. Empty bucket = "not
    configured" — silent skip rather than failure."""
    return {
        "bucket": os.getenv("LEDGER_S3_BUCKET", "").strip(),
        "endpoint_url": os.getenv("LEDGER_S3_ENDPOINT_URL", "").strip() or None,
        "region_name": os.getenv("LEDGER_S3_REGION", "auto") or "auto",
        "access_key": os.getenv("LEDGER_S3_ACCESS_KEY_ID", "").strip() or None,
        "secret_key": os.getenv("LEDGER_S3_SECRET_ACCESS_KEY", "").strip() or None,
    }


def upload_kb_ledger(kb_name: str) -> OffhostUploadResult:
    """Upload today's incremental window for one KB.

    Returns a result dict suitable for the daemon's per-pass log.
    """
    result = OffhostUploadResult(ok=False, kb_name=kb_name, destination="s3")
    started = time.monotonic()

    if not _enabled():
        result.skipped = True
        result.error = "disabled"
        return result

    cfg = _config()
    if not cfg["bucket"]:
        result.skipped = True
        result.error = "no_bucket_configured"
        return result

    try:
        import boto3  # type: ignore
        from botocore.exceptions import ClientError  # type: ignore
    except Exception:
        result.skipped = True
        result.error = "boto3_not_installed"
        return result

    try:
        payload, n_rows, start_off, end_off = incremental_payload(kb_name, "s3")
    except Exception as exc:
        result.error = f"payload_build_failed: {exc}"
        return result
    if not payload:
        result.skipped = True
        result.ok = True
        result.error = "no_new_rows"
        return result

    try:
        client_kwargs: dict = {"region_name": cfg["region_name"]}
        if cfg["endpoint_url"]:
            client_kwargs["endpoint_url"] = cfg["endpoint_url"]
        if cfg["access_key"] and cfg["secret_key"]:
            client_kwargs["aws_access_key_id"] = cfg["access_key"]
            client_kwargs["aws_secret_access_key"] = cfg["secret_key"]
        # Otherwise rely on the default boto3 credential chain
        # (instance profile, env, ~/.aws/credentials).
        client = boto3.client("s3", **client_kwargs)
        key = canonical_object_key(kb_name)
        client.put_object(
            Bucket=cfg["bucket"],
            Key=key,
            Body=payload,
            ContentType="application/gzip",
            ContentEncoding="gzip",
            Metadata={
                "kb": kb_name,
                "rows": str(n_rows),
                "byte_offset_start": str(start_off),
                "byte_offset_end": str(end_off),
            },
        )
        record_upload(kb_name, "s3", end_off, key)
        result.ok = True
        result.bytes_uploaded = len(payload)
        result.rows_uploaded = n_rows
        result.object_key = f"s3://{cfg['bucket']}/{key}"
    except ClientError as exc:
        result.error = f"s3_client_error: {exc}"
    except Exception as exc:
        result.error = f"unexpected: {type(exc).__name__}: {exc}"
    result.duration_s = time.monotonic() - started
    return result
