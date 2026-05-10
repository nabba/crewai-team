"""Tests for the life-companion control panel — runtime overrides
that flip features on/off and edit tunables without a gateway
restart, surfaced via /api/cp/life_companion.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_runtime_settings(tmp_path, monkeypatch):
    """Redirect runtime_settings.json to a tmp file so each test
    starts clean and changes don't leak."""
    from app import runtime_settings as rs

    tmp_file = tmp_path / "runtime_settings.json"
    monkeypatch.setattr(rs, "_STATE_PATH", tmp_file)
    # Reset the in-memory cache so subsequent _ensure_initialized()
    # calls re-read from our tmp file.
    monkeypatch.setattr(rs, "_cache", None)
    yield tmp_file
    monkeypatch.setattr(rs, "_cache", None)


# ── Feature registry ───────────────────────────────────────────────


class TestFeatureRegistry:

    def test_act_now_digest_in_registry(self) -> None:
        from app.life_companion.feature_registry import get_feature
        f = get_feature("act_now_digest")
        assert f is not None
        assert f.feature_env_key == "LIFE_COMPANION_ACT_NOW_DIGEST_ENABLED"
        # Tunables wired through.
        keys = {t.env_key for t in f.tunables}
        assert "LIFE_COMPANION_ACT_NOW_TOP_K" in keys
        assert "LIFE_COMPANION_ACT_NOW_LOOKBACK_HOURS" in keys

    def test_email_in_registry(self) -> None:
        from app.life_companion.feature_registry import get_feature
        f = get_feature("email")
        assert f is not None
        keys = {t.env_key for t in f.tunables}
        assert "LIFE_COMPANION_EMAIL_URGENCY_THRESHOLD" in keys

    def test_unknown_key_returns_none(self) -> None:
        from app.life_companion.feature_registry import get_feature
        assert get_feature("nope") is None

    def test_find_tunable_reverse_lookup(self) -> None:
        from app.life_companion.feature_registry import find_tunable
        result = find_tunable("LIFE_COMPANION_ACT_NOW_TOP_K")
        assert result is not None
        feat, tun = result
        assert feat.key == "act_now_digest"
        assert tun.env_key == "LIFE_COMPANION_ACT_NOW_TOP_K"


# ── Runtime-settings override accessors ────────────────────────────


class TestOverrideAccessors:

    def test_get_overrides_starts_empty(self, tmp_runtime_settings) -> None:
        from app.runtime_settings import life_companion_get_overrides
        assert life_companion_get_overrides() == {}

    def test_set_then_get_enabled_override(self, tmp_runtime_settings) -> None:
        from app.runtime_settings import (
            life_companion_get_feature_enabled,
            life_companion_set_feature_override,
        )
        life_companion_set_feature_override("act_now_digest", enabled=False)
        assert life_companion_get_feature_enabled("act_now_digest") is False

    def test_set_then_get_tunable_override(self, tmp_runtime_settings) -> None:
        from app.runtime_settings import (
            life_companion_get_tunable,
            life_companion_set_feature_override,
        )
        life_companion_set_feature_override(
            "act_now_digest",
            tunables={"LIFE_COMPANION_ACT_NOW_TOP_K": "5"},
        )
        assert life_companion_get_tunable("LIFE_COMPANION_ACT_NOW_TOP_K") == "5"

    def test_clear_tunable_with_empty_string(self, tmp_runtime_settings) -> None:
        from app.runtime_settings import (
            life_companion_get_tunable,
            life_companion_set_feature_override,
        )
        life_companion_set_feature_override(
            "act_now_digest",
            tunables={"LIFE_COMPANION_ACT_NOW_TOP_K": "5"},
        )
        # Clear with empty string.
        life_companion_set_feature_override(
            "act_now_digest",
            tunables={"LIFE_COMPANION_ACT_NOW_TOP_K": ""},
        )
        assert life_companion_get_tunable("LIFE_COMPANION_ACT_NOW_TOP_K") is None

    def test_empty_entry_pruned(self, tmp_runtime_settings) -> None:
        """When an entry has neither enabled override nor tunables,
        the override store should drop it for cleanliness."""
        from app.runtime_settings import (
            life_companion_get_overrides,
            life_companion_set_feature_override,
        )
        # Set, then clear.
        life_companion_set_feature_override("act_now_digest", enabled=True)
        # ``enabled=None`` clears the toggle override.
        life_companion_set_feature_override("act_now_digest", enabled=None)
        assert "act_now_digest" not in life_companion_get_overrides()

    def test_unknown_feature_key_doesnt_block_storage(
        self, tmp_runtime_settings,
    ) -> None:
        """The runtime layer is permissive — only the API endpoint
        validates feature_key against the registry."""
        from app.runtime_settings import (
            life_companion_get_overrides,
            life_companion_set_feature_override,
        )
        life_companion_set_feature_override("brand_new_thing", enabled=True)
        assert "brand_new_thing" in life_companion_get_overrides()


# ── feature_enabled + get_tunable propagate overrides ─────────────


class TestCommonHelpersUseOverrides:

    def test_feature_enabled_override_wins_over_env(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.runtime_settings import life_companion_set_feature_override
        from app.life_companion._common import feature_enabled

        # Env says enabled.
        monkeypatch.setenv("LIFE_COMPANION_TEST_ENABLED", "true")
        # Override says disabled.
        life_companion_set_feature_override("test", enabled=False)
        assert feature_enabled("test") is False

    def test_feature_enabled_falls_back_to_env_when_no_override(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.life_companion._common import feature_enabled
        monkeypatch.setenv("LIFE_COMPANION_TEST_ENABLED", "false")
        # No override set; env wins.
        assert feature_enabled("test") is False

    def test_master_switch_off_overrides_everything(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.runtime_settings import life_companion_set_feature_override
        from app.life_companion._common import feature_enabled

        monkeypatch.setenv("LIFE_COMPANION_ENABLED", "false")
        life_companion_set_feature_override("test", enabled=True)
        # Master switch wins regardless of override.
        assert feature_enabled("test") is False

    def test_get_tunable_override_wins(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.runtime_settings import life_companion_set_feature_override
        from app.life_companion._common import get_tunable

        monkeypatch.setenv("LIFE_COMPANION_ACT_NOW_TOP_K", "7")
        life_companion_set_feature_override(
            "act_now_digest",
            tunables={"LIFE_COMPANION_ACT_NOW_TOP_K": "3"},
        )
        assert get_tunable("LIFE_COMPANION_ACT_NOW_TOP_K", "0") == "3"

    def test_get_tunable_falls_back_to_env(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.life_companion._common import get_tunable
        monkeypatch.setenv("LIFE_COMPANION_TEST_VAR", "from_env")
        assert get_tunable("LIFE_COMPANION_TEST_VAR", "from_default") == "from_env"

    def test_get_tunable_default_when_unset(
        self, tmp_runtime_settings, monkeypatch,
    ) -> None:
        from app.life_companion._common import get_tunable
        monkeypatch.delenv("LIFE_COMPANION_TEST_VAR_UNSET", raising=False)
        assert get_tunable("LIFE_COMPANION_TEST_VAR_UNSET", "fallback") == "fallback"


# ── Wiring contract: act_now_digest reads via get_tunable ──────────


class TestActNowDigestUsesTunableHelper:
    """The whole point of the override mechanism is that the digest
    job picks up changes WITHOUT a gateway restart.  Source-grep
    contract: act_now_digest must call get_tunable(), not
    os.getenv() directly, for its tunables."""

    def test_no_direct_os_getenv_for_tunable_reads(self) -> None:
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "life_companion" / "act_now_digest.py"
        ).read_text(encoding="utf-8")
        # The four tunable accessors must use get_tunable.
        for fn in ("_top_k", "_lookback_hours", "_max_candidates", "_body_chars"):
            idx = src.index(f"def {fn}()")
            # Slice the function body.
            body = src[idx:idx + 400]
            assert "get_tunable" in body, (
                f"{fn} must use get_tunable() so runtime overrides take "
                f"effect without restart"
            )

    def test_email_monitor_uses_tunable_helper(self) -> None:
        src = (
            Path(__file__).resolve().parent.parent
            / "app" / "life_companion" / "email_monitor.py"
        ).read_text(encoding="utf-8")
        for fn in ("_check_interval_s", "_urgency_threshold"):
            idx = src.index(f"def {fn}()")
            body = src[idx:idx + 300]
            assert "get_tunable" in body, (
                f"{fn} must use get_tunable()"
            )
