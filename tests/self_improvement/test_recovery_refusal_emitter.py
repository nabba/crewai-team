"""Tests for emit_recovery_refusal + the forge_queue → gap_detector hook."""

from __future__ import annotations

from app.self_improvement import gap_detector
from app.self_improvement.types import GapSource


# ── direct emitter ───────────────────────────────────────────────────────


def test_emit_recovery_refusal_passes_through_to_emit_gap(monkeypatch) -> None:
    captured: list = []

    def fake_emit_gap(gap):
        captured.append(gap)
        return True

    monkeypatch.setattr(gap_detector, "emit_gap", fake_emit_gap)
    ok = gap_detector.emit_recovery_refusal(
        task="compute Estonian forest cover for 2024",
        category="data_unavailable",
        gap_key="forest_cover_2024__data_unavailable",
        attempts=3,
        queued_for_forge=True,
    )
    assert ok is True
    assert len(captured) == 1
    gap = captured[0]
    assert gap.source is GapSource.RECOVERY_REFUSAL
    assert "data_unavailable" in gap.description
    assert "Estonian forest cover" in gap.description
    assert gap.evidence["refusal_category"] == "data_unavailable"
    assert gap.evidence["attempts"] == 3
    assert gap.evidence["queued_for_forge"] is True
    assert 0.8 <= gap.signal_strength <= 1.0


def test_emit_recovery_refusal_short_circuits_on_empty_task(monkeypatch) -> None:
    called: list = []

    def fake_emit_gap(gap):
        called.append(gap)
        return True

    monkeypatch.setattr(gap_detector, "emit_gap", fake_emit_gap)
    ok = gap_detector.emit_recovery_refusal(task="   ")
    assert ok is False
    assert called == []


def test_emit_recovery_refusal_swallows_emit_failure(monkeypatch) -> None:
    def boom(_):
        raise RuntimeError("ChromaDB down")

    monkeypatch.setattr(gap_detector, "emit_gap", boom)
    # Must not raise; return False on failure.
    ok = gap_detector.emit_recovery_refusal(task="x")
    assert ok is False


def test_recovery_refusal_weight_is_in_source_weights() -> None:
    weight = gap_detector.SOURCE_WEIGHTS.get(GapSource.RECOVERY_REFUSAL)
    assert weight is not None
    assert 0.7 <= weight <= 0.95


# ── integration: forge_queue.execute calls the emitter ──────────────────


def test_forge_queue_execute_emits_recovery_refusal(monkeypatch, tmp_path) -> None:
    """When the recovery loop's last-resort forge_queue strategy runs,
    a LearningGap of source RECOVERY_REFUSAL is emitted with the task +
    category + frequency-counter visible in the evidence."""
    from app.recovery.librarian import Alternative
    from app.recovery.strategies import forge_queue

    # Isolate the frequency file so the test doesn't touch shared state.
    monkeypatch.setattr(
        forge_queue, "_FREQUENCY_PATH", tmp_path / "frequency.json",
    )
    monkeypatch.setattr(
        forge_queue, "_LEARNING_QUEUE", tmp_path / "learning_queue.md",
    )

    captured: list = []

    def fake_emit(**kw):
        captured.append(kw)
        return True

    monkeypatch.setattr(
        gap_detector, "emit_recovery_refusal", fake_emit,
    )
    # The hook in forge_queue does a local import — patch that path too.
    import app.self_improvement.gap_detector as gd_module
    monkeypatch.setattr(gd_module, "emit_recovery_refusal", fake_emit)

    alt = Alternative(
        strategy="forge_queue",
        rationale="last-resort diagnostic",
        est_cost_usd=0.0,
        est_latency_s=0.5,
        sync=True,
    )
    ctx = {"refusal_category": "data_unavailable"}
    result = forge_queue.execute(
        task="compute the Estonian forest cover for 2024",
        alt=alt,
        ctx=ctx,
    )
    # The forge_queue strategy always claims success (returns diagnostic).
    assert result.success is True
    # The LearningGap emitter was called.
    assert len(captured) == 1
    payload = captured[0]
    assert "Estonian forest cover" in payload["task"]
    assert payload["category"] == "data_unavailable"
    assert payload["attempts"] >= 1
