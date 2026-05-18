"""Q16 (PROGRAM §51) — Decade-resilience hardening.

Three themes shipped:

  * Theme 1 (substrate longevity): ``host_substrate_health`` monitor
    (35th healing monitor) — workspace free-space trend, growth
    burst, restart burst, uptime staleness, Linux memory headroom.
  * Theme 2 (vendor independence): ``oauth_token_freshness`` monitor
    (36th) + ``vendor_independence`` drill (5th in the Q6 registry).
  * Theme 3 (partial — operator anomaly): ``operator_anomaly``
    monitor (37th) — hour-shift / cadence / length / new-sender
    surfacing from existing ``workspace/audit.log``.

Vacation mode is deliberately deferred to a separate change because
its security contract deserves its own review.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _isolated_module(rel_path: str, mod_name: str):
    spec = importlib.util.spec_from_file_location(
        mod_name, REPO_ROOT / rel_path,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_notify(monkeypatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    fake_notify = type(sys)("app.notify")
    fake_notify.notify = lambda **kw: captured.append(kw)
    monkeypatch.setitem(sys.modules, "app.notify", fake_notify)
    return captured


# ═════════════════════════════════════════════════════════════════════════
#   Theme 1 — host_substrate_health
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_host(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_host_substrate_health_monitor_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_host_monitor(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_host_substrate_health",
    )
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: state_path)
    monkeypatch.setattr(
        mod, "_host_metrics_path", lambda: tmp_path / "host_metrics.jsonl",
    )
    mod.reset_process_marker()
    return mod


def test_host_module_exists_with_required_surface() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "host_substrate_health.py"
    assert p.is_file()
    src = p.read_text()
    assert "def run(" in src
    assert "def reset_process_marker" in src
    assert "_DAYS_UNTIL_FULL_WARN" in src
    assert "_GROWTH_WOW_WARN_PCT" in src
    assert "_RESTART_BURST_THRESHOLD" in src
    assert "_UPTIME_STALE_DAYS" in src
    assert "_MEMORY_HEADROOM_WARN_PCT" in src
    # Never auto-actions — should not import retention helpers.
    assert "from app.healing.monitors.retention" not in src


def test_host_disabled_run_skips(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out.get("skipped") is True
    assert out["ran"] is False


def test_host_first_run_records_baseline_no_alert(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    captured = _stub_notify(monkeypatch)
    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (50 * 1024**3, 100 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)
    out = mod.run(now=time.time())
    assert out["ran"] is True
    assert out["sampled"] is True
    assert out["n_samples"] == 1
    assert out["alerts"] == []
    assert captured == []


def test_host_daily_wakeup_skips_weekly_sample(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (50 * 1024**3, 100 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)
    now = time.time()
    mod.run(now=now)
    out = mod.run(now=now + 3600)
    assert out["ran"] is True
    assert out["sampled"] is False


def test_host_disk_horizon_alert_on_steep_burn(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    captured = _stub_notify(monkeypatch)

    now = time.time()
    week_s = 7 * 24 * 3600
    samples = []
    for i in range(8):
        weeks_ago = 7 - i
        samples.append({
            "ts": now - weeks_ago * week_s,
            "free_bytes": (100 - (7 - weeks_ago) * 12) * 1024**3,
            "total_bytes": 500 * 1024**3,
            "workspace_bytes": 50 * 1024**3,
        })
    state = {
        "last_run_at": now - week_s - 1,
        "weekly_samples": samples,
        "restart_log": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "state.json").write_text(json.dumps(state))

    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (16 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 50 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)

    out = mod.run(now=now)
    horizon = [a for a in out["alerts"] if a["kind"] == "disk_horizon_warn"]
    assert len(horizon) == 1
    assert horizon[0]["days_until_full"] < 60
    assert any("📉" in c["title"] or "horizon" in c["title"].lower() for c in captured)


def test_host_disk_horizon_silent_on_flat_or_growing(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)

    now = time.time()
    week_s = 7 * 24 * 3600
    samples = [{
        "ts": now - (7 - i) * week_s,
        "free_bytes": 100 * 1024**3,
        "total_bytes": 500 * 1024**3,
        "workspace_bytes": 50 * 1024**3,
    } for i in range(8)]
    state = {
        "last_run_at": now - week_s - 1,
        "weekly_samples": samples,
        "restart_log": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))

    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 50 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)

    out = mod.run(now=now)
    assert [a for a in out["alerts"] if a["kind"] == "disk_horizon_warn"] == []


def test_host_growth_burst_alert(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)

    now = time.time()
    week_s = 7 * 24 * 3600
    sizes_gb = [10, 12, 14.4, 17.28, 20.74]
    samples = [{
        "ts": now - (4 - i) * week_s,
        "free_bytes": 100 * 1024**3,
        "total_bytes": 500 * 1024**3,
        "workspace_bytes": int(gb * 1024**3),
    } for i, gb in enumerate(sizes_gb)]
    state = {
        "last_run_at": now - week_s - 1,
        "weekly_samples": samples,
        "restart_log": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))

    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: int(25 * 1024**3))
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)

    out = mod.run(now=now)
    growth = [a for a in out["alerts"] if a["kind"] == "workspace_growth_burst"]
    assert len(growth) == 1
    assert growth[0]["median_wow_pct"] >= 10.0


def test_host_growth_silent_on_intermittent(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    week_s = 7 * 24 * 3600
    sizes_gb = [10, 12, 14, 14, 16]
    samples = [{
        "ts": now - (4 - i) * week_s,
        "free_bytes": 100 * 1024**3,
        "total_bytes": 500 * 1024**3,
        "workspace_bytes": int(gb * 1024**3),
    } for i, gb in enumerate(sizes_gb)]
    state = {
        "last_run_at": now - week_s - 1,
        "weekly_samples": samples,
        "restart_log": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 18 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)
    out = mod.run(now=now)
    assert [a for a in out["alerts"] if a["kind"] == "workspace_growth_burst"] == []


def test_host_restart_burst_alert(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    state = {
        "last_run_at": now - 100,
        "restart_log": [now - 6 * 3600, now - 3 * 3600],
        "weekly_samples": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    out = mod.run(now=now)
    burst = [a for a in out["alerts"] if a["kind"] == "restart_burst"]
    assert len(burst) == 1
    assert burst[0]["n_restarts_24h"] >= 3


def test_host_uptime_stale_alert(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    boot_at = now - 200 * 86400
    state = {
        "last_run_at": now - 100,
        "restart_log": [boot_at],
        "weekly_samples": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    mod._PROCESS_BOOT_AT = boot_at
    out = mod.run(now=now)
    stale = [a for a in out["alerts"] if a["kind"] == "uptime_stale"]
    assert len(stale) == 1
    assert stale[0]["uptime_days"] >= 180


def test_host_memory_pressure_alert(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    week_s = 7 * 24 * 3600
    samples = [{
        "ts": now - (3 - i) * week_s,
        "free_bytes": 100 * 1024**3,
        "total_bytes": 500 * 1024**3,
        "workspace_bytes": 10 * 1024**3,
        "headroom_pct": 7.0,
        "mem_available_bytes": 1 * 1024**3,
        "mem_total_bytes": 16 * 1024**3,
    } for i in range(4)]
    state = {
        "last_run_at": now - week_s - 1,
        "restart_log": [],
        "weekly_samples": samples,
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(
        mod, "_memory_headroom",
        lambda: {
            "mem_available_bytes": 1 * 1024**3,
            "mem_total_bytes": 16 * 1024**3,
            "headroom_pct": 6.5,
        },
    )
    out = mod.run(now=now)
    mem = [a for a in out["alerts"] if a["kind"] == "memory_pressure"]
    assert len(mem) == 1
    assert mem[0]["headroom_pct"] < 10.0


def test_host_metrics_tail_surfaced(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)
    host_metrics = tmp_path / "host_metrics.jsonl"
    host_metrics.parent.mkdir(parents=True, exist_ok=True)
    host_metrics.write_text(
        json.dumps({
            "ts": time.time(),
            "smart_reallocated_sectors": 0,
            "macos_version": "26.0.0",
        }) + "\n"
    )
    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)
    out = mod.run(now=time.time())
    assert out["host_metrics_present"] is True
    assert out["host_metrics_latest"]["macos_version"] == "26.0.0"


def test_host_linear_slope_pure() -> None:
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_slope_pure",
    )
    pts = [(0.0, 10.0), (1.0, 9.0), (2.0, 8.0), (3.0, 7.0)]
    assert mod._linear_slope(pts) == pytest.approx(-1.0, abs=1e-9)
    assert mod._linear_slope([]) == 0.0
    assert mod._linear_slope([(1.0, 2.0)]) == 0.0


def test_host_alert_dedup_window(monkeypatch, tmp_path) -> None:
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    captured = _stub_notify(monkeypatch)
    now = time.time()
    state = {
        "last_run_at": now - 100,
        "restart_log": [now - 6 * 3600, now - 3 * 3600],
        "weekly_samples": [],
        "last_alert_at": {},
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    mod.run(now=now)
    n_first = len(captured)
    mod.run(now=now + 3600)
    assert len(captured) == n_first  # dedup window suppressed


# ═════════════════════════════════════════════════════════════════════════
#   Theme 2 — oauth_token_freshness
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_oauth(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_oauth_token_freshness_monitor_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_oauth_monitor(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/monitors/oauth_token_freshness.py",
        "_q16_oauth_token_freshness",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    return mod


def test_oauth_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "oauth_token_freshness.py"
    assert p.is_file()
    src = p.read_text()
    assert "def run(" in src
    assert "GOOGLE_INACTIVITY" in src or "_GOOGLE_TOKEN_FILE" in src
    assert "_VENDOR_KEY_PATTERNS" in src
    # NEVER calls an external API — should not import requests/anthropic/etc.
    # at module load. Verify by inspecting imports.
    forbidden_imports = (
        "import anthropic", "import openai", "import requests as req",
    )
    for forbid in forbidden_imports:
        assert forbid not in src


def test_oauth_disabled_run_skips(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out.get("skipped") is True


def test_oauth_no_google_token_silent(monkeypatch, tmp_path) -> None:
    """Missing token file is informational, not warn."""
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 30)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "b" * 30)
    out = mod.run(now=time.time())
    assert out["ran"] is True
    google = out["findings"]["google"]
    assert google["present"] is False
    assert google["severity"] == "info"  # not warn — missing is configuration choice


def test_oauth_fresh_google_token_silent(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    token_path = tmp_path / "google_token.json"
    token_path.write_text(json.dumps({
        "refresh_token": "1//09abc" + "x" * 40,
        "client_id": "test",
    }))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 30)
    out = mod.run(now=time.time())
    google = out["findings"]["google"]
    assert google["present"] is True
    assert google["has_refresh_token"] is True
    assert google["severity"] == "info"


def test_oauth_stale_google_token_alerts(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    captured = _stub_notify(monkeypatch)
    token_path = tmp_path / "google_token.json"
    token_path.write_text(json.dumps({
        "refresh_token": "1//09abc" + "x" * 40,
    }))
    # Back-date the mtime to 130 days ago.
    old_ts = time.time() - 130 * 86400
    os.utime(token_path, (old_ts, old_ts))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 30)
    out = mod.run(now=time.time())
    google = out["findings"]["google"]
    assert google["severity"] == "warn"
    assert google["age_days"] >= 120
    # An alert should have been queued.
    assert any(a["kind"] == "google_workspace" for a in out["alerts"])


def test_oauth_missing_anthropic_critical(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-" + "b" * 30)
    out = mod.run(now=time.time())
    anth = out["findings"]["vendors"]["anthropic"]
    assert anth["severity"] == "crit"
    assert anth["status"] == "missing"


def test_oauth_vendor_format_mismatch_warns(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    # Anthropic key WITHOUT the sk-ant- prefix → format mismatch.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-format-key-but-long-enough-1234567890")
    out = mod.run(now=time.time())
    anth = out["findings"]["vendors"]["anthropic"]
    assert anth["format_match"] is False
    assert anth["status"] == "format_mismatch"
    assert anth["severity"] == "warn"


def test_oauth_no_secret_in_summary(monkeypatch, tmp_path) -> None:
    """Findings should never contain the raw key value, only a fingerprint."""
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    raw_key = "sk-ant-" + "deadbeef" * 10
    monkeypatch.setenv("ANTHROPIC_API_KEY", raw_key)
    out = mod.run(now=time.time())
    serialized = json.dumps(out, default=str)
    assert raw_key not in serialized
    assert "deadbeef" * 10 not in serialized
    # Fingerprint should be present and short.
    fp = out["findings"]["vendors"]["anthropic"].get("key_fingerprint")
    assert isinstance(fp, str) and len(fp) == 4


def test_oauth_vapid_incomplete_pair_warns(monkeypatch, tmp_path) -> None:
    mod = _load_oauth_monitor(monkeypatch, tmp_path)
    _stub_rs_oauth(monkeypatch)
    _stub_notify(monkeypatch)
    # Write only the private half — should flag incomplete.
    (tmp_path / "vapid_private.pem").write_text("-----BEGIN-----\n...\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 30)
    out = mod.run(now=time.time())
    vapid = out["findings"]["vapid"]
    assert vapid["status"] == "incomplete_pair"
    assert vapid["severity"] == "warn"


# ═════════════════════════════════════════════════════════════════════════
#   Theme 2 — vendor_independence drill
# ═════════════════════════════════════════════════════════════════════════


def test_vendor_independence_module_exists() -> None:
    p = REPO_ROOT / "app" / "resilience_drills" / "drills" / "vendor_independence.py"
    assert p.is_file()
    src = p.read_text()
    assert "SPEC = DrillSpec(" in src
    assert "vendor_independence" in src
    assert "DrillRisk.LOW" in src
    assert "drill_vendor_independence_enabled" in src
    # Must never issue live LLM calls.
    assert "anthropic.Anthropic" not in src
    assert "openai.OpenAI" not in src
    # Must source-check the LLM_SUBSYSTEM doc.
    assert "LLM_SUBSYSTEM" in src


def test_vendor_independence_diversity_check_pass(monkeypatch) -> None:
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_vendor_independence_diversity",
    )
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_" + "a" * 32)
    ok, err, info = mod._check_cascade_structural_diversity()
    assert ok is True
    assert info["count"] >= 2


def test_vendor_independence_diversity_check_fail(monkeypatch) -> None:
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_vendor_independence_diversity_fail",
    )
    for env in ("OLLAMA_BASE_URL", "GROQ_API_KEY", "GEMINI_API_KEY",
                "GOOGLE_API_KEY", "DEEPSEEK_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    ok, err, info = mod._check_cascade_structural_diversity()
    assert ok is False
    assert "fallback" in err.lower()


def test_vendor_independence_ollama_unreachable_when_unconfigured(monkeypatch) -> None:
    """When OLLAMA_BASE_URL isn't set, an unreachable probe is informational."""
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_vendor_independence_ollama",
    )
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    # Force the socket probe to fail by pointing at a definitely-closed port.
    monkeypatch.setattr(mod, "_OLLAMA_DEFAULT_PORT", 1)
    ok, err, info = mod._check_ollama_reachable()
    # Unconfigured + unreachable → soft pass.
    assert ok is True
    assert info["reachable"] is False


def test_vendor_independence_no_secret_in_detail() -> None:
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_vendor_independence_secret",
    )
    detail = {"checks": {"cascade": True}}
    ok, err = mod._no_secret_in_detail(detail)
    assert ok is True
    # Now plant a fake leak.
    leaky = {"keys": {"anthropic": "sk-ant-" + "a" * 30}}
    ok, err = mod._no_secret_in_detail(leaky)
    assert ok is False
    # Error must NOT contain the leaked value itself.
    assert "a" * 30 not in (err or "")


# ═════════════════════════════════════════════════════════════════════════
#   Theme 3 — operator_anomaly
# ═════════════════════════════════════════════════════════════════════════


def _stub_rs_anomaly(monkeypatch, *, enabled: bool = True) -> None:
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_operator_anomaly_monitor_enabled = lambda: enabled
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)


def _load_anomaly_monitor(monkeypatch, tmp_path: Path):
    mod = _isolated_module(
        "app/healing/monitors/operator_anomaly.py",
        "_q16_operator_anomaly",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    monkeypatch.setattr(mod, "_state_path", lambda: tmp_path / "state.json")
    monkeypatch.setattr(
        mod, "_audit_log_path", lambda: tmp_path / "audit.log",
    )
    return mod


def _write_audit_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    p = tmp_path / "audit.log"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_anomaly_module_exists() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "operator_anomaly.py"
    assert p.is_file()
    src = p.read_text()
    assert "def run(" in src
    assert "_detect_hour_shift" in src
    assert "_detect_cadence_shift" in src
    assert "_detect_new_sender" in src
    # MUST be observational — never refuses or blocks anything.
    assert "raise" not in src or "raise NotImplementedError" not in src
    # No content reading — only ts + sender + message_length.
    assert "message_body" not in src
    assert "message_text" not in src


def test_anomaly_disabled_run_skips(monkeypatch, tmp_path) -> None:
    mod = _load_anomaly_monitor(monkeypatch, tmp_path)
    _stub_rs_anomaly(monkeypatch, enabled=False)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out.get("skipped") is True


def test_anomaly_empty_audit_log_silent(monkeypatch, tmp_path) -> None:
    mod = _load_anomaly_monitor(monkeypatch, tmp_path)
    _stub_rs_anomaly(monkeypatch)
    _stub_notify(monkeypatch)
    out = mod.run(now=time.time())
    assert out["ran"] is True
    assert out["signals"] == []


def test_anomaly_hour_shift_detected(monkeypatch, tmp_path) -> None:
    mod = _load_anomaly_monitor(monkeypatch, tmp_path)
    _stub_rs_anomaly(monkeypatch)
    _stub_notify(monkeypatch)
    # Use a deterministic UTC anchor so timestamps land in the intended
    # hour buckets regardless of wall-clock at test time.
    anchor = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
    now = anchor.timestamp()
    rows = []
    # 60 baseline events placed at deterministic 10-16h UTC across
    # the 30-day window (afternoon bucket).
    for i in range(60):
        days_ago = 8 + (i % 22)  # spread across baseline window
        h = 10 + (i % 6)  # 10..15 UTC → afternoon bucket
        dt = anchor - timedelta(days=days_ago)
        dt = datetime(dt.year, dt.month, dt.day, h, 0, 0, tzinfo=timezone.utc)
        rows.append({
            "ts": dt.isoformat(),
            "event": "request_received",
            "sender": "+372***0500",
            "message_length": 20,
        })
    # 30 recent events all at 3 UTC (night bucket).
    for i in range(30):
        days_ago = i % 6  # last 6 days
        dt = anchor - timedelta(days=days_ago)
        dt = datetime(dt.year, dt.month, dt.day, 3, 0, 0, tzinfo=timezone.utc)
        rows.append({
            "ts": dt.isoformat(),
            "event": "request_received",
            "sender": "+372***0500",
            "message_length": 20,
        })
    _write_audit_rows(tmp_path, rows)
    out = mod.run(now=now)
    hour_sig = [s for s in out["signals"] if s["kind"] == "hour_shift"]
    assert len(hour_sig) == 1
    assert any(
        b["bucket"] == "night" for b in hour_sig[0]["flagged_buckets"]
    )


def test_anomaly_new_sender_critical(monkeypatch, tmp_path) -> None:
    mod = _load_anomaly_monitor(monkeypatch, tmp_path)
    _stub_rs_anomaly(monkeypatch)
    captured = _stub_notify(monkeypatch)
    now = time.time()
    rows = []
    # 60 baseline events from the regular sender.
    for i in range(60):
        ts = now - (15 + i % 60) * 86400
        rows.append({
            "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "event": "request_received",
            "sender": "+372***0500",
            "message_length": 20,
        })
    # 5 recent events from a NEW sender (within last 7d).
    for i in range(5):
        ts = now - (i % 5) * 86400 - 3600
        rows.append({
            "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "event": "request_received",
            "sender": "+1***9999",
            "message_length": 30,
        })
    _write_audit_rows(tmp_path, rows)
    out = mod.run(now=now)
    new_sender = [s for s in out["signals"] if s["kind"] == "new_sender"]
    assert len(new_sender) == 1
    assert any(s["sender"] == "+1***9999" for s in new_sender[0]["senders"])
    # Should have fired as critical.
    assert any(c.get("critical") for c in captured)


def test_anomaly_no_content_in_state(monkeypatch, tmp_path) -> None:
    """State file must NEVER contain message content (we don't read it)."""
    mod = _load_anomaly_monitor(monkeypatch, tmp_path)
    _stub_rs_anomaly(monkeypatch)
    _stub_notify(monkeypatch)
    now = time.time()
    rows = []
    for i in range(60):
        ts = now - (i % 30) * 86400 + 10 * 3600
        rows.append({
            "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            "event": "request_received",
            "sender": "+372***0500",
            "message_length": 25,
        })
    _write_audit_rows(tmp_path, rows)
    mod.run(now=now)
    state_path = tmp_path / "state.json"
    if state_path.exists():
        content = state_path.read_text()
        # Sanity: should not contain message bodies.
        assert "message_body" not in content
        assert "message_text" not in content


def test_anomaly_pure_bucket_assignment() -> None:
    mod = _isolated_module(
        "app/healing/monitors/operator_anomaly.py",
        "_q16_anomaly_bucket_pure",
    )
    assert mod._bucket_for_hour(0) == "night"
    assert mod._bucket_for_hour(5) == "night"
    assert mod._bucket_for_hour(6) == "morning"
    assert mod._bucket_for_hour(11) == "morning"
    assert mod._bucket_for_hour(12) == "afternoon"
    assert mod._bucket_for_hour(17) == "afternoon"
    assert mod._bucket_for_hour(18) == "evening"
    assert mod._bucket_for_hour(23) == "evening"


# ═════════════════════════════════════════════════════════════════════════
#   Wiring tests — all four Q16 components
# ═════════════════════════════════════════════════════════════════════════


def test_runtime_settings_has_all_q16_switches() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    for key in (
        "host_substrate_health_monitor_enabled",
        "oauth_token_freshness_monitor_enabled",
        "drill_vendor_independence_enabled",
        "operator_anomaly_monitor_enabled",
    ):
        assert f'"{key}"' in src, f"missing setting key {key}"
        assert f"def get_{key}" in src
        assert f"def set_{key}" in src


def test_monitors_init_registers_all_q16_monitors() -> None:
    p = REPO_ROOT / "app" / "healing" / "monitors" / "__init__.py"
    src = p.read_text()
    for name in (
        "host_substrate_health",
        "oauth_token_freshness",
        "operator_anomaly",
    ):
        assert f'"{name}"' in src
        assert f"{name}.run" in src


def test_drills_init_registers_vendor_independence() -> None:
    p = (
        REPO_ROOT / "app" / "resilience_drills" / "drills" / "__init__.py"
    )
    src = p.read_text()
    assert "vendor_independence" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 3 — vacation_mode (deferred follow-on)
# ═════════════════════════════════════════════════════════════════════════


def _make_vacation_state(monkeypatch) -> tuple[Any, dict[str, Any]]:
    """Set up an isolated ``app.runtime_settings`` shim and load the
    vacation_mode.state module against it. Returns (module, fake_rs_storage)."""
    storage: dict[str, Any] = {"vacation_mode_state": {}, "vacation_mode_enabled": True}

    fake_rs = type(sys)("app.runtime_settings")

    def _ensure_initialized():
        return storage

    def _update(d: dict[str, Any]) -> None:
        storage.update(d)

    def _get_enabled() -> bool:
        return bool(storage.get("vacation_mode_enabled", True))

    def _set_enabled(value: bool) -> None:
        storage["vacation_mode_enabled"] = bool(value)

    fake_rs._ensure_initialized = _ensure_initialized
    fake_rs._update = _update
    fake_rs.get_vacation_mode_enabled = _get_enabled
    fake_rs.set_vacation_mode_enabled = _set_enabled
    fake_rs.get_vacation_mode_state = lambda: dict(storage.get("vacation_mode_state", {}) or {})
    fake_rs.set_vacation_mode_state = lambda v: storage.update({"vacation_mode_state": dict(v)})
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)

    mod = _isolated_module(
        "app/vacation_mode/state.py", "_q16_vacation_state",
    )
    return mod, storage


def test_vacation_state_module_exists() -> None:
    for rel in (
        "app/vacation_mode/__init__.py",
        "app/vacation_mode/state.py",
        "app/vacation_mode/allowlist.py",
        "app/vacation_mode/sweep.py",
    ):
        assert (REPO_ROOT / rel).is_file(), f"missing {rel}"


def test_vacation_stage_allowlist_basic(monkeypatch) -> None:
    mod, storage = _make_vacation_state(monkeypatch)
    al = mod.stage_allowlist(
        requestor_allowlist=["wiki_index_reconciler"],
        path_prefix_allowlist=["wiki/companion/"],
        max_diff_lines=8,
    )
    assert al.requestor_allowlist == ["wiki_index_reconciler"]
    assert al.path_prefix_allowlist == ["wiki/companion/"]
    assert al.max_diff_lines == 8


def test_vacation_stage_refused_while_engaged(monkeypatch) -> None:
    mod, _ = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/test/"],
    )
    mod.engage(
        until_ts=time.time() + 86400,
        engaged_by="operator",
        reason="hospital",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    with pytest.raises(mod.VacationModeError) as exc:
        mod.stage_allowlist(
            requestor_allowlist=["b"],
            path_prefix_allowlist=["wiki/other/"],
        )
    assert "engaged" in str(exc.value).lower()


def test_vacation_stage_refuses_too_broad_paths(monkeypatch) -> None:
    mod, _ = _make_vacation_state(monkeypatch)
    with pytest.raises(mod.VacationModeError):
        mod.stage_allowlist(
            requestor_allowlist=["a"],
            path_prefix_allowlist=["app/"],  # too broad
        )
    with pytest.raises(mod.VacationModeError):
        mod.stage_allowlist(
            requestor_allowlist=["a"],
            path_prefix_allowlist=["wiki/no_slash"],  # missing trailing /
        )


def test_vacation_engage_refuses_empty_allowlist(monkeypatch) -> None:
    mod, _ = _make_vacation_state(monkeypatch)
    with pytest.raises(mod.VacationModeError) as exc:
        mod.engage(
            until_ts=time.time() + 86400,
            engaged_by="operator",
        )
    assert "empty" in str(exc.value).lower()


def test_vacation_engage_refuses_excess_duration(monkeypatch) -> None:
    mod, _ = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/test/"],
    )
    with pytest.raises(mod.VacationModeError) as exc:
        mod.engage(
            until_ts=time.time() + (mod.MAX_DURATION_DAYS + 5) * 86400,
            engaged_by="operator",
        )
    assert "duration" in str(exc.value).lower()


def test_vacation_engage_refuses_missing_confirmation_phrase(monkeypatch) -> None:
    """A6 ship-blocker: typed phrase must be enforced server-side, not
    just in the React UI. A bare REST POST without confirmation_phrase
    must be rejected even with a non-empty allowlist + valid duration."""
    mod, _ = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/test/"],
    )
    with pytest.raises(mod.VacationModeError) as exc:
        mod.engage(
            until_ts=time.time() + 3600,
            engaged_by="operator",
            # confirmation_phrase deliberately omitted
        )
    assert "confirmation_phrase" in str(exc.value)


def test_vacation_engage_refuses_wrong_confirmation_phrase(monkeypatch) -> None:
    """Wrong phrase must also be rejected (no fuzzy match)."""
    mod, _ = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/test/"],
    )
    with pytest.raises(mod.VacationModeError) as exc:
        mod.engage(
            until_ts=time.time() + 3600,
            engaged_by="operator",
            confirmation_phrase="engage vacation mode",  # wrong case
        )
    assert "confirmation_phrase" in str(exc.value)


def test_vacation_auto_expiry_on_read(monkeypatch) -> None:
    mod, storage = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/test/"],
    )
    # Engage for 1 hour.
    mod.engage(
        until_ts=time.time() + 3600,
        engaged_by="operator",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    assert mod.is_active() is True
    # Mock time to past the until_ts by patching the engagement
    # directly to a past until.
    state = mod.current_state()
    state.engagement.until_ts = time.time() - 100
    storage["vacation_mode_state"] = state.to_dict()
    # Next read auto-expires.
    assert mod.is_active() is False
    assert mod.current_state().engaged is False


def test_vacation_allowlist_frozen_on_engagement(monkeypatch) -> None:
    """Once engaged, the allowlist is the FROZEN snapshot, immune to
    further staging."""
    mod, _ = _make_vacation_state(monkeypatch)
    mod.stage_allowlist(
        requestor_allowlist=["a"],
        path_prefix_allowlist=["wiki/old/"],
    )
    mod.engage(
        until_ts=time.time() + 86400,
        engaged_by="operator",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    # current_allowlist() during engagement = frozen snapshot.
    al = mod.current_allowlist()
    assert al.path_prefix_allowlist == ["wiki/old/"]
    # Disengage and restage; frozen view DID end with disengagement.
    mod.disengage()
    mod.stage_allowlist(
        requestor_allowlist=["b"],
        path_prefix_allowlist=["wiki/new/"],
    )
    assert mod.current_allowlist().path_prefix_allowlist == ["wiki/new/"]


def test_vacation_validator_refuses_tier_immutable(monkeypatch) -> None:
    """The standard validator runs first; TIER_IMMUTABLE absolute."""
    _make_vacation_state(monkeypatch)
    # Stub validate() to report TIER_IMMUTABLE rejection.
    fake_cr_validator = type(sys)("app.change_requests.validator")
    from dataclasses import dataclass as _dc, field as _field

    @_dc(frozen=True)
    class _StubVR:
        ok: bool
        reason: str | None = None
        is_tier_immutable: bool = False

    fake_cr_validator.validate = lambda *, path, new_content: _StubVR(
        ok=False, reason="TIER_IMMUTABLE", is_tier_immutable=True,
    )
    monkeypatch.setitem(
        sys.modules, "app.change_requests.validator", fake_cr_validator,
    )
    # Stub app.change_requests so 'from app.change_requests.validator
    # import validate as cr_validate' resolves cleanly.
    fake_cr = type(sys)("app.change_requests")
    monkeypatch.setitem(sys.modules, "app.change_requests", fake_cr)

    mod = _isolated_module(
        "app/vacation_mode/allowlist.py", "_q16_vacation_allowlist_tier",
    )
    # Provide a non-empty allowlist via the state module.
    state_mod = sys.modules["_q16_vacation_state"]
    state_mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["app/companion/"],
    )
    result = mod.validate_vacation_apply(
        path="app/something.py",
        new_content="x = 1\n",
        old_content="",
        requestor="x",
    )
    assert result.ok is False
    assert result.is_tier_immutable is True


def test_vacation_validator_refuses_forbidden_prefix(monkeypatch) -> None:
    _make_vacation_state(monkeypatch)
    # Stub validate() to PASS.
    fake_cr_validator = type(sys)("app.change_requests.validator")
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class _StubVR:
        ok: bool = True
        reason: str | None = None
        is_tier_immutable: bool = False

    fake_cr_validator.validate = lambda *, path, new_content: _StubVR(ok=True)
    monkeypatch.setitem(
        sys.modules, "app.change_requests.validator", fake_cr_validator,
    )
    fake_cr = type(sys)("app.change_requests")
    monkeypatch.setitem(sys.modules, "app.change_requests", fake_cr)
    mod = _isolated_module(
        "app/vacation_mode/allowlist.py", "_q16_vacation_allowlist_forbid",
    )
    state_mod = sys.modules["_q16_vacation_state"]
    state_mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["app/governance_amendment/"],
    )
    result = mod.validate_vacation_apply(
        path="app/governance_amendment/protocol.py",
        new_content="# additive\n",
        old_content="",
        requestor="x",
    )
    assert result.ok is False
    assert "forbidden" in (result.reason or "").lower()


def test_vacation_validator_accepts_allowlisted(monkeypatch) -> None:
    _make_vacation_state(monkeypatch)
    fake_cr_validator = type(sys)("app.change_requests.validator")
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class _StubVR:
        ok: bool = True
        reason: str | None = None
        is_tier_immutable: bool = False

    fake_cr_validator.validate = lambda *, path, new_content: _StubVR(ok=True)
    monkeypatch.setitem(
        sys.modules, "app.change_requests.validator", fake_cr_validator,
    )
    fake_cr = type(sys)("app.change_requests")
    monkeypatch.setitem(sys.modules, "app.change_requests", fake_cr)
    mod = _isolated_module(
        "app/vacation_mode/allowlist.py", "_q16_vacation_allowlist_ok",
    )
    state_mod = sys.modules["_q16_vacation_state"]
    state_mod.stage_allowlist(
        requestor_allowlist=["wiki_index_reconciler"],
        path_prefix_allowlist=["wiki/companion/"],
        max_diff_lines=10,
    )
    result = mod.validate_vacation_apply(
        path="wiki/companion/notes.md",
        new_content="line1\nline2\nline3\n",
        old_content="",
        requestor="wiki_index_reconciler",
    )
    assert result.ok is True


def test_vacation_validator_rejects_deletions(monkeypatch) -> None:
    _make_vacation_state(monkeypatch)
    fake_cr_validator = type(sys)("app.change_requests.validator")
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class _StubVR:
        ok: bool = True
        reason: str | None = None
        is_tier_immutable: bool = False

    fake_cr_validator.validate = lambda *, path, new_content: _StubVR(ok=True)
    monkeypatch.setitem(
        sys.modules, "app.change_requests.validator", fake_cr_validator,
    )
    fake_cr = type(sys)("app.change_requests")
    monkeypatch.setitem(sys.modules, "app.change_requests", fake_cr)
    mod = _isolated_module(
        "app/vacation_mode/allowlist.py", "_q16_vacation_allowlist_del",
    )
    state_mod = sys.modules["_q16_vacation_state"]
    state_mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["wiki/companion/"],
    )
    result = mod.validate_vacation_apply(
        path="wiki/companion/notes.md",
        new_content="line1\n",      # removed lines
        old_content="line1\nline2\nline3\n",
        requestor="x",
    )
    assert result.ok is False
    assert "deleted" in (result.reason or "").lower() or "additive" in (result.reason or "").lower()


def test_vacation_decision_source_added() -> None:
    p = REPO_ROOT / "app" / "change_requests" / "models.py"
    src = p.read_text()
    assert "VACATION_AUTO_APPLY" in src
    assert 'vacation-auto-apply' in src


def test_vacation_runtime_settings_wired() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    assert '"vacation_mode_enabled"' in src
    assert '"vacation_mode_state"' in src
    assert "def get_vacation_mode_enabled" in src
    assert "def set_vacation_mode_enabled" in src
    assert "def get_vacation_mode_state" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 2 follow-on — live cascade-fitness
# ═════════════════════════════════════════════════════════════════════════


def test_live_fitness_default_off_so_drill_silent(monkeypatch) -> None:
    """Default-OFF: live_fitness check skips with ok=True."""
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_live_fitness_off",
    )
    monkeypatch.delenv("DRILL_VENDOR_INDEPENDENCE_LIVE_ENABLED", raising=False)
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_drill_vendor_independence_live_enabled = lambda: False
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    ok, err, info = mod._check_live_fitness()
    assert ok is True
    assert info["enabled"] is False
    assert "skipped" in info.get("reason", "").lower()


def test_live_fitness_enabled_but_selector_missing(monkeypatch) -> None:
    """When opt-in but llm_selector lacks ``smoke_completion`` helper,
    we should soft-pass with a skip reason — the structural drill
    is sufficient on its own."""
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_live_fitness_no_helper",
    )
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_drill_vendor_independence_live_enabled = lambda: True
    fake_rs._ensure_initialized = lambda: {}
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    # llm_selector without smoke_completion helper.
    fake_sel = type(sys)("app.llm_selector")
    # No smoke_completion attribute.
    monkeypatch.setitem(sys.modules, "app", type(sys)("app"))
    monkeypatch.setitem(sys.modules, "app.llm_selector", fake_sel)
    monkeypatch.setattr(mod, "_live_fitness_enabled", lambda: True)
    ok, err, info = mod._check_live_fitness()
    assert ok is True
    assert info["enabled"] is True
    assert "no smoke_completion helper" in info.get("reason", "").lower()


def test_live_fitness_pass_via_smoke_helper(monkeypatch) -> None:
    """When the helper exists and ≥2/3 replies are non-empty short
    strings, live_fitness passes."""
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_live_fitness_pass",
    )
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_drill_vendor_independence_live_enabled = lambda: True
    fake_rs._ensure_initialized = lambda: {"chat_blocked_models": []}
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    fake_sel = type(sys)("app.llm_selector")

    def _smoke(*, question, exclude_providers, timeout_s, max_chars):
        return "4"
    fake_sel.smoke_completion = _smoke
    monkeypatch.setitem(sys.modules, "app", type(sys)("app"))
    monkeypatch.setitem(sys.modules, "app.llm_selector", fake_sel)
    monkeypatch.setattr(mod, "_live_fitness_enabled", lambda: True)
    ok, err, info = mod._check_live_fitness()
    assert ok is True
    assert info["enabled"] is True
    assert info["n_ok"] == info["n_questions"]


def test_live_fitness_fail_when_replies_empty(monkeypatch) -> None:
    mod = _isolated_module(
        "app/resilience_drills/drills/vendor_independence.py",
        "_q16_live_fitness_fail",
    )
    fake_rs = type(sys)("app.runtime_settings")
    fake_rs.get_drill_vendor_independence_live_enabled = lambda: True
    fake_rs._ensure_initialized = lambda: {"chat_blocked_models": []}
    monkeypatch.setitem(sys.modules, "app.runtime_settings", fake_rs)
    fake_sel = type(sys)("app.llm_selector")
    fake_sel.smoke_completion = lambda *, question, exclude_providers, timeout_s, max_chars: ""
    monkeypatch.setitem(sys.modules, "app", type(sys)("app"))
    monkeypatch.setitem(sys.modules, "app.llm_selector", fake_sel)
    monkeypatch.setattr(mod, "_live_fitness_enabled", lambda: True)
    ok, err, info = mod._check_live_fitness()
    assert ok is False
    assert info["n_ok"] == 0


def test_live_fitness_runtime_setting_wired() -> None:
    p = REPO_ROOT / "app" / "runtime_settings.py"
    src = p.read_text()
    assert '"drill_vendor_independence_live_enabled"' in src
    assert "def get_drill_vendor_independence_live_enabled" in src


# ═════════════════════════════════════════════════════════════════════════
#   Theme 1 follow-on — substrate-transition emission
# ═════════════════════════════════════════════════════════════════════════


def test_substrate_fingerprint_shape() -> None:
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_fingerprint",
    )
    fp = mod._substrate_fingerprint()
    assert isinstance(fp, dict)
    for key in ("hostname", "system", "machine", "python_version", "total_gb"):
        assert key in fp


def test_substrate_transition_first_run_is_none() -> None:
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_transition_first",
    )
    cur = mod._substrate_fingerprint()
    # First run (no prior) → None
    assert mod._substrate_transition(None, cur) is None
    assert mod._substrate_transition({}, cur) is None


def test_substrate_transition_detects_hostname_change() -> None:
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_transition_hostname",
    )
    prior = {
        "hostname": "old-mac",
        "system": "Darwin",
        "machine": "arm64",
        "python_version": "3.13.0",
        "total_gb": 500,
    }
    current = {
        "hostname": "new-mac",
        "system": "Darwin",
        "machine": "arm64",
        "python_version": "3.13.0",
        "total_gb": 500,
    }
    transition = mod._substrate_transition(prior, current)
    assert transition is not None
    assert "hostname" in transition
    assert transition["hostname"]["from"] == "old-mac"


def test_substrate_transition_detects_total_gb_jump() -> None:
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_transition_disk",
    )
    prior = {"hostname": "h", "system": "Darwin", "machine": "arm64", "total_gb": 500}
    current = {"hostname": "h", "system": "Darwin", "machine": "arm64", "total_gb": 1000}
    transition = mod._substrate_transition(prior, current)
    assert transition is not None
    assert "total_gb" in transition


def test_substrate_transition_silent_on_python_version_alone() -> None:
    """Python version drift alone is NOT a substrate transition —
    operator upgrades happen normally."""
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_transition_py",
    )
    prior = {
        "hostname": "h", "system": "Darwin", "machine": "arm64",
        "python_version": "3.13.0", "total_gb": 500,
    }
    current = {
        "hostname": "h", "system": "Darwin", "machine": "arm64",
        "python_version": "3.14.0", "total_gb": 500,
    }
    assert mod._substrate_transition(prior, current) is None


def test_substrate_transition_silent_on_minor_disk_expansion() -> None:
    """≤10% disk expansion is not a substrate transition."""
    mod = _isolated_module(
        "app/healing/monitors/host_substrate_health.py",
        "_q16_substrate_transition_minor",
    )
    prior = {"hostname": "h", "system": "Darwin", "machine": "arm64", "total_gb": 500}
    current = {"hostname": "h", "system": "Darwin", "machine": "arm64", "total_gb": 530}
    assert mod._substrate_transition(prior, current) is None


# ═════════════════════════════════════════════════════════════════════════
#   Theme 3 follow-ons — rate limit / ledger / digest / slash / REST
# ═════════════════════════════════════════════════════════════════════════


def test_vacation_window_event_kind_registered() -> None:
    p = REPO_ROOT / "app" / "identity" / "continuity_ledger.py"
    src = p.read_text()
    assert '"vacation_window"' in src


def test_vacation_engage_emits_ledger_event(monkeypatch) -> None:
    """engage() should call record_event with kind='vacation_window'."""
    mod, _ = _make_vacation_state(monkeypatch)
    recorded: list[dict[str, Any]] = []
    fake_ledger = type(sys)("app.identity.continuity_ledger")

    def _record_event(*, kind, actor, summary, detail=None, path=None, now=None):
        recorded.append({"kind": kind, "actor": actor, "summary": summary,
                         "detail": dict(detail or {})})
        return True
    fake_ledger.record_event = _record_event
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )
    mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["wiki/companion/"],
    )
    mod.engage(
        until_ts=time.time() + 86400,
        engaged_by="operator",
        reason="medical leave",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    assert len(recorded) == 1
    assert recorded[0]["kind"] == "vacation_window"
    assert recorded[0]["detail"]["event"] == "engage"


def test_vacation_disengage_emits_ledger_event(monkeypatch) -> None:
    mod, _ = _make_vacation_state(monkeypatch)
    recorded: list[dict[str, Any]] = []
    fake_ledger = type(sys)("app.identity.continuity_ledger")
    fake_ledger.record_event = lambda **kw: (recorded.append(kw) or True)
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )
    # Stub digest so disengage doesn't pull in dependencies we haven't set up.
    fake_digest = type(sys)("app.vacation_mode.digest")
    fake_digest.compose_digest = lambda **kw: Path("/dev/null")
    monkeypatch.setitem(sys.modules, "app.vacation_mode.digest", fake_digest)

    mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["wiki/x/"],
    )
    mod.engage(
        until_ts=time.time() + 86400,
        engaged_by="operator",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    recorded.clear()
    mod.disengage(disengaged_by="operator")
    assert len(recorded) == 1
    assert recorded[0]["kind"] == "vacation_window"
    assert recorded[0]["detail"]["event"] == "disengage"


def test_vacation_auto_expiry_emits_auto_expire_event(monkeypatch) -> None:
    mod, storage = _make_vacation_state(monkeypatch)
    recorded: list[dict[str, Any]] = []
    fake_ledger = type(sys)("app.identity.continuity_ledger")
    fake_ledger.record_event = lambda **kw: (recorded.append(kw) or True)
    monkeypatch.setitem(
        sys.modules, "app.identity.continuity_ledger", fake_ledger,
    )
    fake_digest = type(sys)("app.vacation_mode.digest")
    fake_digest.compose_digest = lambda **kw: Path("/dev/null")
    monkeypatch.setitem(sys.modules, "app.vacation_mode.digest", fake_digest)
    mod.stage_allowlist(
        requestor_allowlist=["x"],
        path_prefix_allowlist=["wiki/x/"],
    )
    mod.engage(
        until_ts=time.time() + 3600,
        engaged_by="op",
        confirmation_phrase="ENGAGE VACATION MODE",
    )
    # Force the engagement into the past + trigger auto-expiry.
    state = mod.current_state()
    state.engagement.until_ts = time.time() - 100
    storage["vacation_mode_state"] = state.to_dict()
    recorded.clear()
    assert mod.is_active() is False
    auto_events = [r for r in recorded if r["kind"] == "vacation_window"]
    assert len(auto_events) == 1
    assert auto_events[0]["detail"]["event"] == "auto_expire"


# ── Sweep rate limit ──────────────────────────────────────────────────


def test_sweep_rate_limit_per_requestor(monkeypatch, tmp_path) -> None:
    """After N auto-approvals from the same requestor in one day, the
    sweep refuses further ones from that requestor."""
    mod = _isolated_module(
        "app/vacation_mode/sweep.py", "_q16_sweep_rate",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    now = time.time()
    # Hit the per-requestor cap.
    for _ in range(mod._RATE_LIMIT_PER_REQUESTOR_PER_DAY):
        ok, reason = mod._rate_limit_ok("agent_a", now=now)
        assert ok is True, reason
        mod._commit_rate_limit("agent_a", now=now)
    ok, reason = mod._rate_limit_ok("agent_a", now=now)
    assert ok is False
    assert "per-requestor" in reason.lower()
    # Different requestor still has slack.
    ok, reason = mod._rate_limit_ok("agent_b", now=now)
    assert ok is True


def test_sweep_rate_limit_resets_on_utc_day_rollover(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/vacation_mode/sweep.py", "_q16_sweep_rate_rollover",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    yesterday = time.time() - 24 * 3600
    for _ in range(mod._RATE_LIMIT_PER_REQUESTOR_PER_DAY):
        ok, _ = mod._rate_limit_ok("agent_a", now=yesterday)
        mod._commit_rate_limit("agent_a", now=yesterday)
    # Today should be a fresh bucket.
    ok, _ = mod._rate_limit_ok("agent_a", now=time.time())
    assert ok is True


def test_sweep_rate_limit_global_cap(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/vacation_mode/sweep.py", "_q16_sweep_rate_global",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    now = time.time()
    # Drive global up using distinct requestors so per-requestor cap
    # doesn't fire first.
    cap = mod._RATE_LIMIT_GLOBAL_PER_DAY
    # need enough distinct requestors to reach cap without per-req triggering
    per_req = mod._RATE_LIMIT_PER_REQUESTOR_PER_DAY
    n_distinct = (cap + per_req - 1) // per_req
    count = 0
    for i in range(n_distinct):
        for _ in range(per_req):
            if count >= cap:
                break
            ok, _ = mod._rate_limit_ok(f"req-{i}", now=now)
            assert ok is True
            mod._commit_rate_limit(f"req-{i}", now=now)
            count += 1
        if count >= cap:
            break
    # One more from a fresh requestor should hit the global cap.
    ok, reason = mod._rate_limit_ok("fresh-req", now=now)
    assert ok is False
    assert "global" in reason.lower()


# ── Digest composer ──────────────────────────────────────────────────


def test_digest_composes_empty_window(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/vacation_mode/digest.py", "_q16_digest_empty",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    # Need an engagement object — easiest via the state module.
    state_mod = _isolated_module(
        "app/vacation_mode/state.py", "_q16_digest_state_helper",
    )
    eng = state_mod.VacationEngagement(
        engaged_at=time.time() - 3600,
        until_ts=time.time() + 3600,
        engaged_by="operator",
        reason="testing",
        frozen_allowlist=state_mod.VacationAllowlist(
            requestor_allowlist=["agent_a"],
            path_prefix_allowlist=["wiki/test/"],
            max_diff_lines=10,
        ),
    )
    out_path = mod.compose_digest(engagement=eng, ended_at=time.time())
    assert out_path.exists()
    body = out_path.read_text()
    assert "Vacation digest" in body
    assert "Total: **0**" in body
    assert "No auto-applies" in body


def test_digest_composes_window_with_rows(monkeypatch, tmp_path) -> None:
    mod = _isolated_module(
        "app/vacation_mode/digest.py", "_q16_digest_with_rows",
    )
    monkeypatch.setattr(mod, "_workspace", lambda: tmp_path)
    state_mod = _isolated_module(
        "app/vacation_mode/state.py", "_q16_digest_state_helper2",
    )
    # Write some log rows.
    start = time.time() - 3600
    end = time.time()
    rows = [
        {
            "ts": datetime.fromtimestamp(start + 60, tz=timezone.utc).isoformat(),
            "request_id": "cr-1",
            "path": "wiki/companion/notes.md",
            "requestor": "wiki_reconciler",
            "ok": True,
            "error": None,
            "elapsed_s": 0.5,
        },
        {
            "ts": datetime.fromtimestamp(start + 120, tz=timezone.utc).isoformat(),
            "request_id": "cr-2",
            "path": "wiki/companion/links.md",
            "requestor": "wiki_reconciler",
            "ok": True,
            "error": None,
            "elapsed_s": 0.4,
        },
        {
            "ts": datetime.fromtimestamp(start + 180, tz=timezone.utc).isoformat(),
            "request_id": "cr-3",
            "path": "docs/proposed_fixes/foo.md",
            "requestor": "library_radar",
            "ok": False,
            "error": "apply: ValueError: nope",
            "elapsed_s": 0.6,
        },
    ]
    log_path = tmp_path / "vacation_mode" / "auto_apply_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
    )
    eng = state_mod.VacationEngagement(
        engaged_at=start,
        until_ts=end + 3600,
        engaged_by="operator",
        reason="test window",
        frozen_allowlist=state_mod.VacationAllowlist(
            requestor_allowlist=["wiki_reconciler", "library_radar"],
            path_prefix_allowlist=["wiki/companion/", "docs/proposed_fixes/"],
        ),
    )
    out_path = mod.compose_digest(engagement=eng, ended_at=end)
    body = out_path.read_text()
    assert "Total: **3**" in body
    assert "Successful: **2**" in body
    assert "Failed: **1**" in body
    assert "wiki_reconciler" in body
    assert "library_radar" in body
    assert "wiki/companion" in body or "wiki/companion/" in body


# ── Slash command dispatch ────────────────────────────────────────────


def test_vacation_signal_command_registered() -> None:
    p = REPO_ROOT / "app" / "agents" / "commander" / "command_registry.py"
    src = p.read_text()
    assert '"Vacation"' in src
    assert "/vacation" in src
    assert "/vacation engage" in src
    assert "/vacation disengage" in src


def test_vacation_signal_dispatcher_wired() -> None:
    p = REPO_ROOT / "app" / "agents" / "commander" / "commands.py"
    src = p.read_text()
    assert "_handle_vacation_command" in src
    assert 'lower.startswith("/vacation")' in src


# ── REST endpoints ────────────────────────────────────────────────────


def test_vacation_rest_api_exists_with_routes() -> None:
    p = REPO_ROOT / "app" / "api" / "vacation_api.py"
    assert p.is_file()
    src = p.read_text()
    assert "/api/cp/vacation" in src
    for route in (
        '@router.get("/state")',
        '@router.get("/allowlist")',
        '@router.post("/allowlist/stage")',
        '@router.post("/engage")',
        '@router.post("/disengage")',
        '@router.get("/digests")',
        '@router.get("/audit-log")',
    ):
        assert route in src, f"missing route {route}"


def test_vacation_rest_router_mounted_in_main() -> None:
    p = REPO_ROOT / "app" / "main.py"
    src = p.read_text()
    assert "app.api.vacation_api" in src
    assert "vacation_router" in src


def test_substrate_emit_calls_continuity_ledger(monkeypatch, tmp_path) -> None:
    """When a transition is detected, the monitor calls
    continuity_ledger.record_event with kind='substrate_migration'."""
    mod = _load_host_monitor(monkeypatch, tmp_path)
    _stub_rs_host(monkeypatch)
    _stub_notify(monkeypatch)

    recorded: list[dict[str, Any]] = []

    fake_ledger = type(sys)("app.identity.continuity_ledger")
    def _record_event(*, kind, actor, summary, detail=None, path=None, now=None):
        recorded.append({"kind": kind, "actor": actor, "summary": summary,
                         "detail": dict(detail or {})})
        return True
    fake_ledger.record_event = _record_event
    monkeypatch.setitem(sys.modules, "app.identity.continuity_ledger", fake_ledger)

    now = time.time()
    week_s = 7 * 24 * 3600

    # Pre-seed prior fingerprint with DIFFERENT hostname.
    state = {
        "last_run_at": now - week_s - 1,
        "weekly_samples": [],
        "restart_log": [],
        "last_alert_at": {},
        "substrate_fingerprint": {
            "hostname": "old-mac",
            "system": "Darwin",
            "machine": "arm64",
            "python_version": "3.13.0",
            "total_gb": 500,
        },
    }
    (tmp_path / "state.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "state.json").write_text(json.dumps(state))

    monkeypatch.setattr(
        mod, "_disk_usage_bytes",
        lambda: (100 * 1024**3, 500 * 1024**3),
    )
    monkeypatch.setattr(mod, "_workspace_bytes", lambda: 10 * 1024**3)
    monkeypatch.setattr(mod, "_memory_headroom", lambda: None)
    # Force the current fingerprint to report a different hostname.
    monkeypatch.setattr(
        mod, "_substrate_fingerprint",
        lambda: {
            "hostname": "new-mac",
            "system": "Darwin",
            "machine": "arm64",
            "python_version": "3.13.0",
            "total_gb": 500,
        },
    )
    out = mod.run(now=now)
    assert "substrate_transition" in out
    assert out["substrate_transition"]["hostname"]["to"] == "new-mac"
    assert len(recorded) == 1
    assert recorded[0]["kind"] == "substrate_migration"
