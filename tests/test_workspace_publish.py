"""Tests for app.workspace_publish — GW publish helper for §3.G5 hooks.

Mocks `GlobalWorkspace.get_instance` so tests don't reach into the real
SubIA workspace (which talks to Postgres on first instantiation).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_workspace(monkeypatch: pytest.MonkeyPatch):
    """Replace GlobalWorkspace + WorkspaceCandidate with mocks; returns the
    mock workspace instance so tests can assert on what it received."""
    fake_gw = MagicMock(name="GlobalWorkspace_instance")
    fake_gw.compete_for_broadcast = MagicMock()

    class FakeCandidate:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    # Module to import the real symbols from — must match the path used in
    # the late import inside `publish_to_workspace`.
    import app.subia.scene.global_workspace as gw_mod
    monkeypatch.setattr(gw_mod, "GlobalWorkspace", MagicMock(get_instance=lambda: fake_gw))
    monkeypatch.setattr(gw_mod, "WorkspaceCandidate", FakeCandidate)

    return fake_gw


def test_publish_to_workspace_skips_below_noise_floor(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    result = publish_to_workspace(
        source="test",
        content="something",
        salience=0.01,           # below noise floor
        signal_type="disposition",
    )

    assert result is False
    mock_workspace.compete_for_broadcast.assert_not_called()


def test_publish_to_workspace_skips_empty_content(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    result = publish_to_workspace(
        source="test",
        content="",
        salience=0.5,
        signal_type="disposition",
    )

    assert result is False
    mock_workspace.compete_for_broadcast.assert_not_called()


def test_publish_to_workspace_clamps_salience(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    publish_to_workspace(
        source="test",
        content="x",
        salience=99.0,           # above 1.0
        signal_type="disposition",
    )

    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    assert candidate.salience == 1.0


def test_publish_to_workspace_truncates_content(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    long_content = "x" * 500
    publish_to_workspace(
        source="test",
        content=long_content,
        salience=0.5,
        signal_type="disposition",
    )

    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    assert len(candidate.content) <= 281  # 280 + 1 for the ellipsis
    assert candidate.content.endswith("…")


def test_publish_to_workspace_passes_through_fields(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    publish_to_workspace(
        source="test-src",
        content="hello",
        salience=0.6,
        signal_type="trend_reversal",
    )

    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    assert candidate.source_agent == "test-src"
    assert candidate.content == "hello"
    assert candidate.salience == 0.6
    assert candidate.signal_type == "trend_reversal"


def test_publish_to_workspace_swallows_errors(monkeypatch):
    """A workspace failure must NEVER propagate to the calling subsystem."""
    from app.workspace_publish import publish_to_workspace

    def _explode(*a, **kw):
        raise RuntimeError("postgres down")

    import app.subia.scene.global_workspace as gw_mod
    monkeypatch.setattr(gw_mod, "GlobalWorkspace",
                        MagicMock(get_instance=_explode))

    # Should not raise — returns False instead.
    result = publish_to_workspace(
        source="test",
        content="x",
        salience=0.5,
        signal_type="disposition",
    )

    assert result is False


def test_publish_to_workspace_handles_bad_salience(mock_workspace):
    from app.workspace_publish import publish_to_workspace

    result = publish_to_workspace(
        source="test",
        content="x",
        salience="not a number",   # type: ignore[arg-type]
        signal_type="disposition",
    )

    assert result is False
    mock_workspace.compete_for_broadcast.assert_not_called()


# ── publish_idle_outcome ─────────────────────────────────────────────────


def test_publish_idle_outcome_skips_zero_count(mock_workspace):
    from app.workspace_publish import publish_idle_outcome

    result = publish_idle_outcome(
        source="reconciler",
        signal_type="certainty_shift",
        counts={"synced": 0, "failed": 0},
        salience_key="synced",
        content_template="{synced} synced",
    )

    assert result is False
    mock_workspace.compete_for_broadcast.assert_not_called()


def test_publish_idle_outcome_skips_non_dict(mock_workspace):
    from app.workspace_publish import publish_idle_outcome

    result = publish_idle_outcome(
        source="reconciler",
        signal_type="certainty_shift",
        counts="not a dict",  # type: ignore[arg-type]
        salience_key="synced",
        content_template="{synced} synced",
    )

    assert result is False


def test_publish_idle_outcome_scales_salience_with_magnitude(mock_workspace):
    from app.workspace_publish import publish_idle_outcome

    publish_idle_outcome(
        source="reconciler",
        signal_type="certainty_shift",
        counts={"synced": 5, "failed": 0, "skipped": 0},
        salience_key="synced",
        content_template="{synced} synced",
        salience_floor=0.2,
        salience_per_unit=0.05,
        salience_ceiling=0.7,
    )

    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    # 0.2 + 5 * 0.05 = 0.45
    assert candidate.salience == pytest.approx(0.45)


def test_publish_idle_outcome_clamps_to_ceiling(mock_workspace):
    from app.workspace_publish import publish_idle_outcome

    publish_idle_outcome(
        source="reconciler",
        signal_type="certainty_shift",
        counts={"synced": 1000},
        salience_key="synced",
        content_template="{synced} synced",
        salience_ceiling=0.7,
    )

    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    assert candidate.salience == 0.7


def test_publish_idle_outcome_handles_missing_template_key(mock_workspace):
    """Malformed template falls back to a generic representation."""
    from app.workspace_publish import publish_idle_outcome

    result = publish_idle_outcome(
        source="reconciler",
        signal_type="certainty_shift",
        counts={"synced": 3},
        salience_key="synced",
        content_template="{nonexistent_key}",  # missing key
    )

    # Should still publish (with fallback content); not crash.
    assert result is True
    candidate = mock_workspace.compete_for_broadcast.call_args[0][0][0]
    assert "reconciler" in candidate.content
