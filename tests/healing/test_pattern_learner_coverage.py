"""Regression: pattern_learner._registered_signatures() must actually
return the set of covered signatures — not silently return an empty set
because an internal symbol was renamed.

The pre-fix shape (commit 72f3f6e9 fixed it on disk, but the bug was
running in production for a while because the gateway hadn't been
redeployed):

  pattern_learner.py imported `_LOCK` from runbooks.py, but the symbol
  was named `_registry_lock`. The ImportError was caught by the broad
  ``except Exception: return set()`` fallback, so the function silently
  returned an empty set every call. This meant pattern_learner thought
  EVERY registered signature was uncovered, including
  numeric_overflow_widen_cr (sig c38013f929816242, 45 ×/week) which
  has been correctly registered the whole time.

This test exercises the function against a real ``_REGISTERED_RUNBOOKS``
dict so a future symbol rename can't slip through unnoticed.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def _runbook_registry():
    """Provide direct access to the live registry for inspection."""
    pytest.importorskip("app.healing.runbooks")
    from app.healing import runbooks
    return runbooks


class TestRegisteredSignaturesContract:
    """The function must successfully read the registry — not silently
    swallow an ImportError."""

    def test_function_uses_real_lock_symbol(self) -> None:
        """The internal lock import must reference the actual symbol
        name in runbooks.py. If the import fails, we'd silently return
        an empty set and the proposer would re-flag covered patterns."""
        import inspect
        from app.healing.pattern_learner import _registered_signatures

        src = inspect.getsource(_registered_signatures)
        # The actual symbol in runbooks.py is _registry_lock (not _LOCK).
        # If a future refactor renames either side, this fails LOUDLY
        # rather than producing the silent-failure mode we just fixed.
        assert (
            "from app.healing.runbooks import" in src
        ), "function must import from runbooks module"
        assert "_registry_lock" in src, (
            "must import _registry_lock — _LOCK was the old broken name "
            "(see commit 72f3f6e9 'Phase E')"
        )

    def test_lock_symbol_actually_exists_in_runbooks(self) -> None:
        """Belt-and-suspenders: confirm the symbol pattern_learner imports
        actually exists in runbooks. Catches a future rename of the
        runbooks-side symbol."""
        from app.healing import runbooks
        assert hasattr(runbooks, "_registry_lock"), (
            "runbooks._registry_lock must exist for pattern_learner's import"
        )
        assert hasattr(runbooks, "_REGISTERED_RUNBOOKS"), (
            "runbooks._REGISTERED_RUNBOOKS must exist"
        )


class TestRegisteredSignaturesReturnsRealCoverage:

    def test_hash_pattern_handler_is_seen_as_covered(
        self, _runbook_registry,
    ) -> None:
        """When a handler is registered with a hex-hash pattern, the
        proposer must recognize that signature as covered."""
        from app.healing.pattern_learner import _registered_signatures
        from app.healing.runbooks import register_runbook, unregister_runbook

        # Use a unique fake signature so we don't collide with any
        # real registration in this process.
        fake_sig = "deadbeef00112233"
        try:
            register_runbook(
                "test_coverage_fake_handler",
                fake_sig,
                lambda anomaly: None,
            )
            sigs = _registered_signatures()
            assert fake_sig in sigs, (
                f"signature {fake_sig} should be covered after registration; "
                f"got {len(sigs)} sigs: {sorted(sigs)}"
            )
        finally:
            unregister_runbook("test_coverage_fake_handler")

    def test_catch_all_pattern_is_not_claimed_as_covered(
        self, _runbook_registry,
    ) -> None:
        """A handler registered with `.*` (the log_only fallback or a
        multi-router) shouldn't claim every signature as covered —
        otherwise the proposer would never re-flag any pattern."""
        from app.healing.pattern_learner import _registered_signatures
        from app.healing.runbooks import register_runbook, unregister_runbook

        try:
            register_runbook(
                "test_coverage_catch_all",
                r".*",
                lambda anomaly: None,
            )
            sigs = _registered_signatures()
            # `.*` should not contribute to the covered set.
            assert ".*" not in sigs

            # And specifically — a fictional signature must NOT be
            # claimed as covered just because the catch-all exists.
            assert "feedfacedeadbeef" not in sigs
        finally:
            unregister_runbook("test_coverage_catch_all")

    def test_returns_non_empty_when_handlers_registered(
        self, _runbook_registry,
    ) -> None:
        """Smoke test for the original silent-failure mode. With at
        least one hash-pattern handler in the registry, the function
        MUST NOT return an empty set. (Empty set would indicate the
        ``_LOCK`` import bug has been re-introduced.)"""
        from app.healing.pattern_learner import _registered_signatures
        from app.healing.runbooks import register_runbook, unregister_runbook

        try:
            register_runbook(
                "test_non_empty_smoke",
                "abcdef0123456789",
                lambda anomaly: None,
            )
            sigs = _registered_signatures()
            assert len(sigs) > 0, (
                "_registered_signatures returned 0 — likely a silent "
                "ImportError fallback (see commit 72f3f6e9)"
            )
        finally:
            unregister_runbook("test_non_empty_smoke")
