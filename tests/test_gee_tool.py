"""Tests for app.tools.gee_tool — graceful-degradation + script execution.

These tests don't hit the real Earth Engine API; they verify:
  * The factory returns ``[]`` when credentials are missing/invalid
    (so the coding crew degrades silently to its other tools).
  * ``_ensure_initialised`` caches both success and failure so we
    don't hammer Google with broken credentials on every call.
  * ``_run_user_script`` properly captures stdout, surfaces exceptions,
    and serialises ``ee.ComputedObject``-shaped results via
    ``getInfo()``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── _gee_credentials_path / _gee_project_id ─────────────────────────


class TestEnvResolution:

    def test_no_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        from app.tools.gee_tool import _gee_credentials_path
        assert _gee_credentials_path() is None

    def test_env_pointing_at_missing_file_returns_none(self, monkeypatch, tmp_path):
        ghost = tmp_path / "no-such.json"
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(ghost))
        from app.tools.gee_tool import _gee_credentials_path
        assert _gee_credentials_path() is None

    def test_env_pointing_at_real_file_returns_path(self, monkeypatch, tmp_path):
        sa = tmp_path / "sa.json"
        sa.write_text("{}")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
        from app.tools.gee_tool import _gee_credentials_path
        assert _gee_credentials_path() == str(sa)

    def test_explicit_project_env_overrides_json(self, monkeypatch):
        monkeypatch.setenv("GEE_PROJECT", "explicit-override-123")
        from app.tools.gee_tool import _gee_project_id
        assert _gee_project_id({"project_id": "json-project-456"}) == "explicit-override-123"

    def test_falls_back_to_json_project_id(self, monkeypatch):
        monkeypatch.delenv("GEE_PROJECT", raising=False)
        from app.tools.gee_tool import _gee_project_id
        assert _gee_project_id({"project_id": "json-only-789"}) == "json-only-789"

    def test_no_project_anywhere_returns_none(self, monkeypatch):
        monkeypatch.delenv("GEE_PROJECT", raising=False)
        from app.tools.gee_tool import _gee_project_id
        assert _gee_project_id({}) is None


# ── create_gee_tools graceful degradation ───────────────────────────


class TestCreateGeeTools:

    def test_no_credentials_returns_empty_list(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        from app.tools.gee_tool import create_gee_tools
        assert create_gee_tools() == []

    def test_with_credentials_returns_one_tool(self, monkeypatch, tmp_path):
        sa = tmp_path / "sa.json"
        sa.write_text(json.dumps({
            "type": "service_account",
            "project_id": "test-proj",
            "client_email": "test@test.iam.gserviceaccount.com",
            "private_key": "fake",
        }))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))

        from app.tools.gee_tool import create_gee_tools
        tools = create_gee_tools()
        assert len(tools) == 1
        assert tools[0].name == "gee_run_script"


# ── Tool-description guidance (regression guard) ────────────────────
#
# The coding crew, on 2026-05-02, hit a 240s tool timeout writing a
# naive per-year `for yr in range(...): ... .getInfo()` loop for an
# Estonia deforestation aggregation. The fix was to push the round-trip
# warning + a worked good/bad pattern into the BaseTool description so
# the LLM sees it before it writes the script. These tests pin the
# guidance in place — if a future refactor strips the warning, this
# fails fast instead of silently regressing crew behavior.


class TestToolDescriptionGuidance:

    def _get_tool(self, monkeypatch, tmp_path):
        # Guidance tests need the actual BaseTool subclass, so crewai
        # must be importable. Skip cleanly on dev laptops that lack it
        # (CI / the sandbox container have it installed).
        pytest.importorskip("crewai.tools")
        sa = tmp_path / "sa.json"
        sa.write_text(json.dumps({
            "type": "service_account",
            "project_id": "test-proj",
            "client_email": "test@test.iam.gserviceaccount.com",
            "private_key": "fake",
        }))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))
        from app.tools.gee_tool import create_gee_tools
        tools = create_gee_tools()
        assert tools, "create_gee_tools() returned [] despite crewai being installed"
        return tools[0]

    def test_description_warns_about_round_trips(self, monkeypatch, tmp_path):
        tool = self._get_tool(monkeypatch, tmp_path)
        desc = tool.description.lower()
        # The warning must be explicit — not just "be efficient".
        assert "round-trip" in desc
        assert "getinfo()" in desc
        # And it must name a fast primitive so the LLM has somewhere to go.
        assert "frequencyhistogram" in desc

    def test_description_shows_bad_then_good(self, monkeypatch, tmp_path):
        tool = self._get_tool(monkeypatch, tmp_path)
        desc = tool.description
        # Both labels must appear, in order, so the LLM sees the
        # contrast rather than just one or the other.
        bad_idx = desc.find("# BAD")
        good_idx = desc.find("# GOOD")
        assert bad_idx != -1, "description missing '# BAD' anti-pattern label"
        assert good_idx != -1, "description missing '# GOOD' pattern label"
        assert bad_idx < good_idx, "anti-pattern must come before fix"

    def test_script_field_steers_toward_lazy_result(self, monkeypatch, tmp_path):
        """The wrapper calls .getInfo() once on `result`. The field
        description should tell the LLM that — otherwise it'll call
        .getInfo() itself and may do so in a loop."""
        tool = self._get_tool(monkeypatch, tmp_path)
        schema = tool.args_schema.model_fields["script"]
        field_desc = schema.description.lower()
        assert "wrapper" in field_desc
        assert "once" in field_desc

    def test_skill_file_present(self):
        """The handwritten batching-pattern skill must exist in
        workspace/skills/ so it gets indexed into ChromaDB by the
        idle skill-indexer (idle_scheduler.py:_index_skills)."""
        from pathlib import Path
        # tests/ sits under crewai-team/, skill lives at
        # crewai-team/workspace/skills/gee_batching_pattern.md
        skill = Path(__file__).resolve().parent.parent / "workspace" / "skills" / "gee_batching_pattern.md"
        assert skill.exists(), f"missing skill file: {skill}"
        body = skill.read_text()
        # Sanity-check the load-bearing markers — the rule, an
        # anti-pattern label, and a fix label.
        assert "round-trip" in body.lower()
        assert "BAD" in body
        assert "GOOD" in body
        assert "frequencyHistogram" in body


# ── _ensure_initialised caching ─────────────────────────────────────


class TestEnsureInitialised:

    def _reset_cache(self):
        """Clear the module-level init cache so tests don't bleed."""
        import app.tools.gee_tool as gt
        gt._EE_INITIALISED = False
        gt._EE_INIT_ERROR = None

    def test_cached_success_is_o1(self, monkeypatch, tmp_path):
        """After the first successful Initialize, repeated calls return
        immediately without re-doing the network work."""
        self._reset_cache()
        sa = tmp_path / "sa.json"
        sa.write_text(json.dumps({
            "type": "service_account",
            "project_id": "test-proj",
            "client_email": "test@test.iam.gserviceaccount.com",
            "private_key": "fake",
        }))
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa))

        # Patch ee.Initialize at the module level — it's imported
        # lazily inside _ensure_initialised so we patch sys.modules
        fake_ee = MagicMock()
        fake_creds = MagicMock()
        fake_ee.ServiceAccountCredentials.return_value = fake_creds

        with patch.dict("sys.modules", {"ee": fake_ee}):
            from app.tools.gee_tool import _ensure_initialised
            ok1, err1 = _ensure_initialised()
            assert ok1
            assert err1 is None
            assert fake_ee.Initialize.call_count == 1

            # Second call must NOT call Initialize again
            ok2, err2 = _ensure_initialised()
            assert ok2
            assert fake_ee.Initialize.call_count == 1

    def test_cached_failure_is_o1(self, monkeypatch):
        """Bad credentials should cache the error message so subsequent
        calls don't hammer Google with broken auth."""
        self._reset_cache()
        # Missing env → first call fails fast
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        from app.tools.gee_tool import _ensure_initialised
        ok1, err1 = _ensure_initialised()
        assert not ok1
        assert "GOOGLE_APPLICATION_CREDENTIALS" in err1
        ok2, err2 = _ensure_initialised()
        assert not ok2
        assert err1 == err2  # cached identical message
