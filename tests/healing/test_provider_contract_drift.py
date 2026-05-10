"""Tests for app.healing.monitors.provider_contract_drift (§2.7)."""

from __future__ import annotations

import json
from pathlib import Path

from app.healing.monitors import provider_contract_drift as pcd


_OPENAI_LIKE_BASELINE = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
    },
}

_ANTHROPIC_LIKE_BASELINE = {
    "id": "msg_abc",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "ok"}],
    "model": "claude-sonnet-4-7",
    "stop_reason": "end_turn",
    "usage": {
        "input_tokens": 10,
        "output_tokens": 1,
    },
}


# ── extract_signature ───────────────────────────────────────────────────


def test_extract_signature_flattens_nested_dicts() -> None:
    sig = pcd.extract_signature(_OPENAI_LIKE_BASELINE)
    assert sig["id"] == "str"
    assert sig["model"] == "str"
    assert sig["usage.total_tokens"] == "int"
    # choices[0] keys are flattened.
    assert "choices[0].finish_reason" in sig
    assert sig["choices[0].finish_reason"] == "str"
    assert sig["choices[0].message.role"] == "str"


def test_extract_signature_handles_list_of_dicts() -> None:
    sig = pcd.extract_signature(_ANTHROPIC_LIKE_BASELINE)
    assert sig["content[0].type"] == "str"
    assert sig["content[0].text"] == "str"


def test_extract_signature_records_null_type() -> None:
    sig = pcd.extract_signature({"a": None, "b": 1})
    assert sig["a"] == "null"
    assert sig["b"] == "int"


# ── diff_signatures ─────────────────────────────────────────────────────


def test_diff_detects_removed_key() -> None:
    a = {"x": "str", "y": "int"}
    b = {"x": "str"}
    d = pcd.diff_signatures(a, b)
    assert d["removed"] == ["y"]
    assert d["added"] == []
    assert d["changed_type"] == []


def test_diff_detects_added_key() -> None:
    a = {"x": "str"}
    b = {"x": "str", "y": "int"}
    d = pcd.diff_signatures(a, b)
    assert d["added"] == ["y"]
    assert d["removed"] == []


def test_diff_detects_type_change() -> None:
    a = {"finish_reason": "str"}
    b = {"finish_reason": "null"}
    d = pcd.diff_signatures(a, b)
    assert d["changed_type"] == [
        {"key": "finish_reason", "from": "str", "to": "null"},
    ]


# ── is_breaking_drift ───────────────────────────────────────────────────


def test_breaking_drift_for_removed_key() -> None:
    assert pcd.is_breaking_drift({"removed": ["x"], "added": [], "changed_type": []})


def test_breaking_drift_for_type_change() -> None:
    assert pcd.is_breaking_drift({
        "removed": [],
        "added": [],
        "changed_type": [{"key": "x", "from": "str", "to": "null"}],
    })


def test_addition_only_is_not_breaking() -> None:
    """Additions are additive-safe — parsers ignore unknown keys."""
    assert not pcd.is_breaking_drift({
        "removed": [],
        "added": ["new_field"],
        "changed_type": [],
    })


# ── run_one_pass ────────────────────────────────────────────────────────


def test_run_one_pass_seeds_baseline_on_first_run(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"

    out = pcd.run_one_pass(
        baseline_path=bp,
        history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE if p == "openrouter" else _ANTHROPIC_LIKE_BASELINE,
        providers=["openrouter", "anthropic"],
    )
    assert out["status"] == "ok"
    assert out["alerts"] == 0
    assert set(out["seeded"]) == {"openrouter", "anthropic"}
    assert bp.exists()
    baseline = json.loads(bp.read_text())
    assert "openrouter" in baseline
    assert "anthropic" in baseline


def test_run_one_pass_no_alert_when_signatures_match(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"

    pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )
    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )
    assert out["alerts"] == 0
    assert not hp.exists() or hp.read_text() == ""


def test_run_one_pass_alerts_on_removed_key(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"

    # Seed baseline.
    pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )

    # Mutate response — drop usage.total_tokens.
    drifted = json.loads(json.dumps(_OPENAI_LIKE_BASELINE))
    del drifted["usage"]["total_tokens"]

    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: drifted,
        providers=["openrouter"],
    )
    assert out["alerts"] == 1
    alert = out["alert_details"][0]
    assert alert["provider"] == "openrouter"
    assert "usage.total_tokens" in alert["diff"]["removed"]
    # History row written.
    rows = [json.loads(line) for line in hp.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["provider"] == "openrouter"


def test_run_one_pass_alerts_on_type_change(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"
    pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )

    drifted = json.loads(json.dumps(_OPENAI_LIKE_BASELINE))
    drifted["choices"][0]["finish_reason"] = None  # str → null
    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: drifted,
        providers=["openrouter"],
    )
    assert out["alerts"] == 1
    diff = out["alert_details"][0]["diff"]
    assert any(c["key"].endswith("finish_reason") for c in diff["changed_type"])


def test_addition_only_is_not_alerted(tmp_path: Path) -> None:
    """Provider adds a new optional field — not a breaking change."""
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"
    pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )

    drifted = json.loads(json.dumps(_OPENAI_LIKE_BASELINE))
    drifted["new_optional_field"] = "experimental"
    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: drifted,
        providers=["openrouter"],
    )
    assert out["alerts"] == 0


def test_skips_provider_when_probe_returns_none(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"
    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=lambda p: None,
        providers=["openrouter"],
    )
    assert out["alerts"] == 0
    assert any(s["provider"] == "openrouter" for s in out["skipped"])


def test_skips_provider_when_probe_raises(tmp_path: Path) -> None:
    bp = tmp_path / "baseline.json"
    hp = tmp_path / "history.jsonl"

    def boom(_p):
        raise RuntimeError("network down")

    out = pcd.run_one_pass(
        baseline_path=bp, history_path=hp,
        probe_fn=boom,
        providers=["openrouter"],
    )
    assert out["alerts"] == 0
    assert out["skipped"][0]["provider"] == "openrouter"


def test_disabled_short_circuits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROVIDER_CONTRACT_DRIFT_ENABLED", "false")
    out = pcd.run_one_pass(
        baseline_path=tmp_path / "b.json",
        history_path=tmp_path / "h.jsonl",
        probe_fn=lambda p: _OPENAI_LIKE_BASELINE,
        providers=["openrouter"],
    )
    assert out["status"] == "disabled"
    assert out["alerts"] == 0
