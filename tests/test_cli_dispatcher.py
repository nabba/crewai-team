"""Tests for the operator CLI dispatcher (``python -m app.cli``).

Covers:

* Global flag parsing (``--json`` / ``--quiet`` / ``--endpoint`` / ``--bearer``)
  works in BOTH positions — before AND after the subverb. Critical because
  argparse's ``parents=`` pattern silently overwrites parent values with
  leaf defaults unless ``default=argparse.SUPPRESS`` is used.
* Config resolution from flags / env / TOML / defaults.
* Transport error → exit code mapping (1 / 2 / 3).
* Local-only subcommands work without a gateway (notes save, healing run).
* Help renders for every subcommand.
"""
from __future__ import annotations

import io
import json
import os
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from app.cli import commands, transport
from app.cli.config import CLIConfig, resolve
from app.cli.main import build_parser, main


# --------------------------------------------------------------------------- #
# Global flag positioning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["--json", "ledger", "tail", "-n", "3"],
        ["ledger", "tail", "-n", "3", "--json"],
        ["ledger", "--json", "tail", "-n", "3"],
    ],
)
def test_json_flag_works_in_every_position(argv):
    """Regression: argparse parents= overwrites unless default=SUPPRESS."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    assert getattr(ns, "json", False) is True


def test_endpoint_flag_after_verb_is_preserved():
    parser = build_parser()
    ns = parser.parse_args(["status", "--endpoint", "http://x:1"])
    assert ns.endpoint == "http://x:1"


def test_bearer_flag_before_verb_is_preserved():
    parser = build_parser()
    ns = parser.parse_args(["--bearer", "tok", "status"])
    assert ns.bearer == "tok"


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #


def test_resolve_uses_explicit_flag_over_env(monkeypatch):
    monkeypatch.setenv("AAI_ENDPOINT", "http://env:1")
    cfg = resolve(endpoint="http://flag:1")
    assert cfg.endpoint == "http://flag:1"


def test_resolve_named_alias_local():
    cfg = resolve(endpoint="local")
    assert cfg.endpoint == "http://localhost:3100"


def test_resolve_named_alias_funnel_reads_env(monkeypatch):
    monkeypatch.setenv("DASHBOARD_PUBLIC_URL", "https://public.example/")
    cfg = resolve(endpoint="funnel")
    assert cfg.endpoint == "https://public.example"  # trailing / stripped


def test_resolve_bearer_falls_back_to_gateway_secret(monkeypatch):
    monkeypatch.delenv("AAI_BEARER", raising=False)
    monkeypatch.setenv("GATEWAY_SECRET", "g-secret")
    cfg = resolve(endpoint="local")
    assert cfg.bearer == "g-secret"
    assert cfg.auth_header() == {"Authorization": "Bearer g-secret"}


def test_resolve_no_bearer_yields_empty_header(monkeypatch):
    monkeypatch.delenv("AAI_BEARER", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET", raising=False)
    cfg = resolve(endpoint="local")
    assert cfg.auth_header() == {}


# --------------------------------------------------------------------------- #
# Transport error → exit code
# --------------------------------------------------------------------------- #


def test_status_returns_2_on_network_error(monkeypatch, capsys):
    monkeypatch.delenv("AAI_BEARER", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET", raising=False)
    # Endpoint that won't connect — definitive transport failure.
    code = main(["--endpoint", "http://127.0.0.1:1", "status"])
    captured = capsys.readouterr()
    assert code == 2
    assert "cannot reach" in captured.err.lower() or "cannot reach" in captured.out.lower()


def test_transport_error_class_exit_codes():
    assert transport.NetworkError("x").exit_code == 2
    assert transport.AuthError("x").exit_code == 2
    assert transport.GatewayError("x").exit_code == 3


# --------------------------------------------------------------------------- #
# Notes save — local-only
# --------------------------------------------------------------------------- #


def test_notes_save_writes_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    code = main(["notes", "save", "test-note", "--body", "hello world", "--overwrite"])
    assert code == 0
    target = tmp_path / "notes" / "test-note.md"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world"


def test_notes_save_refuses_overwrite_without_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "test-note.md").write_text("existing")
    code = main(["notes", "save", "test-note", "--body", "new"])
    assert code == 1
    assert "overwrite" in capsys.readouterr().err.lower()
    # Existing content preserved.
    assert (tmp_path / "notes" / "test-note.md").read_text() == "existing"


def test_notes_save_rejects_unsafe_title(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    code = main(["notes", "save", "../../etc/passwd", "--body", "x"])
    # Path traversal characters get stripped; the resulting safe filename
    # is "....etcpasswd.md" — still inside notes/. The point: never escapes.
    assert (tmp_path / "notes").exists()
    # No file created outside notes/ — strongest invariant.
    assert not (tmp_path / ".." / "etc" / "passwd").exists()


# --------------------------------------------------------------------------- #
# Logs tail — local-only
# --------------------------------------------------------------------------- #


def test_logs_tail_reads_workspace_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    errors = tmp_path / "errors.jsonl"
    errors.write_text(
        '{"ts":"2026-05-18T00:00:00Z","msg":"a"}\n'
        '{"ts":"2026-05-18T00:01:00Z","msg":"b"}\n'
        '{"ts":"2026-05-18T00:02:00Z","msg":"c"}\n',
        encoding="utf-8",
    )
    code = main(["logs", "tail", "-n", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "msg" in out
    # Only the last 2 lines.
    assert out.count("\n") == 2


def test_logs_tail_json_parses_each_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "errors.jsonl").write_text(
        '{"ts":"a","msg":"x"}\nnot-json\n', encoding="utf-8"
    )
    code = main(["--json", "logs", "tail", "-n", "5"])
    out = capsys.readouterr().out
    assert code == 0
    data = json.loads(out)
    assert data["lines"][0]["msg"] == "x"
    # Non-JSON line preserved verbatim under "raw".
    assert data["lines"][1]["raw"] == "not-json"


def test_logs_tail_returns_1_when_no_log_exists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "nope"))
    code = main(["logs", "tail"])
    assert code == 1
    assert "no log file" in capsys.readouterr().err.lower()


# --------------------------------------------------------------------------- #
# Healing run — unknown monitor fails with code 1
# --------------------------------------------------------------------------- #


def test_healing_run_unknown_monitor_returns_1(capsys):
    code = main(["healing", "run", "definitely_not_a_real_monitor_xyz"])
    err = capsys.readouterr().err
    assert code == 1
    assert "unknown monitor" in err.lower()


# --------------------------------------------------------------------------- #
# Help renders for every subverb
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["status", "--help"],
        ["healing", "run", "--help"],
        ["logs", "tail", "--help"],
        ["recall", "--help"],
        ["briefing", "--help"],
        ["ledger", "tail", "--help"],
        ["threads", "list", "--help"],
        ["threads", "show", "--help"],
        ["files", "list", "--help"],
        ["files", "send", "--help"],
        ["notes", "save", "--help"],
        ["skills", "list", "--help"],
        ["skills", "show", "--help"],
        ["cr", "list", "--help"],
        ["cr", "show", "--help"],
        ["amendments", "list", "--help"],
        ["bootstrap", "--help"],
        ["advisory", "goodhart", "--help"],
    ],
)
def test_every_subcommand_help_parses(argv, capsys):
    """Help text rendering is the cheapest possible smoke test for the parser."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0


# --------------------------------------------------------------------------- #
# TOML config loading
# --------------------------------------------------------------------------- #


def test_resolve_reads_toml_endpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("AAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AAI_BEARER", raising=False)
    monkeypatch.delenv("GATEWAY_SECRET", raising=False)
    # Point HOME at tmp_path so the config loader picks up our test file.
    config_dir = tmp_path / ".config" / "andrusai"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(textwrap.dedent("""
        [default]
        endpoint = "tailnet"

        [endpoints]
        tailnet = "http://example.tailnet:3100"

        [auth]
        bearer = "from-toml"
    """), encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    cfg = resolve()
    assert cfg.endpoint == "http://example.tailnet:3100"
    assert cfg.bearer == "from-toml"


# --------------------------------------------------------------------------- #
# Mode resolution
# --------------------------------------------------------------------------- #


def test_mode_quiet_takes_precedence_over_json(monkeypatch):
    parser = build_parser()
    ns = parser.parse_args(["--json", "--quiet", "status"])
    # Implementation choice: quiet wins (errors still go to stderr).
    assert commands._mode(ns) == "quiet"


def test_mode_defaults_to_text():
    parser = build_parser()
    ns = parser.parse_args(["status"])
    assert commands._mode(ns) == "text"
