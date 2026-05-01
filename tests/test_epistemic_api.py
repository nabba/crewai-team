"""Tests for app.epistemic.api FastAPI router shapes."""
from __future__ import annotations

import os
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# ── Stub heavy/optional deps (must precede app imports) ──────────────
_mock_psycopg2 = MagicMock()
_mock_psycopg2.InterfaceError = type("InterfaceError", (Exception,), {})
_mock_psycopg2.OperationalError = type("OperationalError", (Exception,), {})
sys.modules.setdefault("psycopg2", _mock_psycopg2)
sys.modules.setdefault("psycopg2.pool", MagicMock())

for _mod in ("crewai", "crewai.tools", "langchain_anthropic", "docker"):
    if _mod not in sys.modules:
        m = types.ModuleType(_mod)
        if _mod == "crewai.tools":
            m.tool = lambda name: (lambda fn: fn)
        sys.modules[_mod] = m


from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.epistemic.api import router  # noqa: E402


def _build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ============================================================================
# /epistemic/now
# ============================================================================

class TestEpistemicNow(unittest.TestCase):

    def test_now_without_task_id_returns_sentinel(self):
        client = _build_client()
        resp = client.get("/epistemic/now")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsNone(body["task_id"])
        self.assertIsNone(body["ledger"])
        self.assertEqual(body["load_bearing_count"], 0)
        self.assertEqual(body["unverified_load_bearing_count"], 0)

    def test_now_with_task_id_returns_ledger_shape(self):
        client = _build_client()
        from app.epistemic.ledger import (
            Claim, Ledger, Register, VerificationStatus,
        )
        # Mock at the API-module boundary: the unit under test is the
        # endpoint's composition, not the SQL plumbing (covered separately
        # in test_epistemic_span_writer.py).
        sample_claim = Claim.new(
            task_id="task_abc",
            agent_role="researcher",
            statement="x is not a symlink",
            status=VerificationStatus.INFERRED,
            register=Register.DECLARATIVE,
            load_bearing=True,
        )
        sample_ledger = Ledger.from_claims(
            task_id="task_abc", claims=[sample_claim],
        )
        with patch("app.epistemic.api.load_ledger_for_task",
                   return_value=sample_ledger), \
             patch("app.epistemic.api.list_bias_matches_for_task",
                   return_value=[]):
            resp = client.get("/epistemic/now?task_id=task_abc")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["task_id"], "task_abc")
        self.assertIsNotNone(body["ledger"])
        self.assertEqual(len(body["ledger"]), 1)
        self.assertEqual(body["ledger"][0]["claim_id"], sample_claim.claim_id)
        self.assertEqual(body["load_bearing_count"], 1)
        self.assertEqual(body["unverified_load_bearing_count"], 1)
        self.assertEqual(body["bias_match_count"], 0)


# ============================================================================
# /epistemic/feed
# ============================================================================

class TestEpistemicFeed(unittest.TestCase):

    def test_feed_default_window(self):
        client = _build_client()
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch("app.epistemic.span_writer.execute") as mock_exec:
            mock_exec.return_value = [
                {
                    "id": 1,
                    "task_id": "task_abc",
                    "claim_id": "clm_aa",
                    "bias_id": "inference_as_fact",
                    "severity": "high",
                    "matched_claim_ids": ["clm_aa"],
                    "detail": {"verifier_tool": "readlink"},
                    "detected_at": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
                },
            ]
            resp = client.get("/epistemic/feed")

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["window_minutes"], 60)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["matches"][0]["bias_id"], "inference_as_fact")
        self.assertEqual(body["matches"][0]["severity"], "high")

    def test_feed_window_validation(self):
        client = _build_client()
        # Window of 0 is below ge=1 — FastAPI returns 422.
        resp = client.get("/epistemic/feed?window_min=0")
        self.assertEqual(resp.status_code, 422)
        # Window of 5000 is above le=1440 — same.
        resp = client.get("/epistemic/feed?window_min=5000")
        self.assertEqual(resp.status_code, 422)


# ============================================================================
# /epistemic/biases
# ============================================================================

class TestEpistemicBiases(unittest.TestCase):

    def test_biases_returns_inference_as_fact(self):
        client = _build_client()
        resp = client.get("/epistemic/biases")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        ids = {b["id"] for b in body["biases"]}
        self.assertIn("inference_as_fact", ids)
        # Spot-check the entry
        entry = next(b for b in body["biases"] if b["id"] == "inference_as_fact")
        self.assertEqual(entry["severity"], "high")
        self.assertEqual(entry["phase"], "realtime")
        self.assertEqual(entry["corrective_action"], "hedge_or_verify")
        self.assertFalse(entry["blocking"])  # Phase 1 ships warn-mode


# ============================================================================
# /epistemic/verifiers
# ============================================================================

class TestEpistemicVerifiers(unittest.TestCase):

    def test_verifiers_returns_starter_set(self):
        from app.epistemic.verification import _reset_for_tests as _reset_registry
        _reset_registry()
        client = _build_client()
        resp = client.get("/epistemic/verifiers")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        ids = {v["id"] for v in body["verifiers"]}
        self.assertIn("filesystem.is_symlink", ids)
        self.assertIn("git.is_clean_tree", ids)
        # Each entry has the expected fields
        for v in body["verifiers"]:
            self.assertIn("tool", v)
            self.assertIn("expected_signal", v)
            self.assertIn("estimated_seconds", v)
            self.assertGreaterEqual(v["estimated_seconds"], 0)


# ============================================================================
# /epistemic/claim/{id}
# ============================================================================

class TestEpistemicClaim(unittest.TestCase):

    def test_existing_claim_returned(self):
        client = _build_client()
        row = {
            "claim_id": "clm_aa",
            "task_id": "task_abc",
            "span_id": None,
            "agent_role": "researcher",
            "statement": "x is not a symlink",
            "status": "inferred",
            "register": "declarative",
            "evidence": [],
            "verifying_action": None,
            "load_bearing": True,
            "tags": [],
            "superseded_by": None,
            "created_at": datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        }
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch("app.epistemic.span_writer.execute_one", return_value=row):
            resp = client.get("/epistemic/claim/clm_aa")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["claim_id"], "clm_aa")

    def test_missing_claim_404(self):
        client = _build_client()
        with patch.dict(os.environ, {"EPISTEMIC_ENABLED": "true"}), \
             patch("app.epistemic.span_writer.execute_one", return_value=None):
            resp = client.get("/epistemic/claim/clm_missing")
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
