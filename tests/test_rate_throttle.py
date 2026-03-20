"""Tests for app/rate_throttle.py — RequestCostTracker and request tracking."""
import threading
import unittest

from app.rate_throttle import (
    RequestCostTracker,
    start_request_tracking,
    stop_request_tracking,
    get_active_tracker,
    set_active_tracker,
)


class TestRequestCostTracker(unittest.TestCase):
    """Test the RequestCostTracker accumulator."""

    def test_initial_state(self):
        t = RequestCostTracker("req-1")
        assert t.request_id == "req-1"
        assert t.total_prompt_tokens == 0
        assert t.total_completion_tokens == 0
        assert t.total_cost_usd == 0.0
        assert t.call_count == 0
        assert t.total_tokens == 0
        assert len(t.models_used) == 0

    def test_record_accumulates(self):
        t = RequestCostTracker("req-2")
        t.record("model-a", 100, 50, 0.001)
        t.record("model-b", 200, 100, 0.005)
        assert t.total_prompt_tokens == 300
        assert t.total_completion_tokens == 150
        assert t.total_tokens == 450
        assert abs(t.total_cost_usd - 0.006) < 1e-9
        assert t.call_count == 2
        assert t.models_used == {"model-a", "model-b"}

    def test_same_model_counted_once_in_set(self):
        t = RequestCostTracker()
        t.record("model-a", 10, 5, 0.0)
        t.record("model-a", 20, 10, 0.0)
        assert len(t.models_used) == 1
        assert t.call_count == 2

    def test_summary_format(self):
        t = RequestCostTracker()
        t.record("claude", 1000, 500, 0.0123)
        summary = t.summary()
        assert "1 LLM calls" in summary
        assert "1,500 tokens" in summary
        assert "$0.0123" in summary
        assert "claude" in summary

    def test_thread_safety(self):
        t = RequestCostTracker("thread-test")
        threads = []
        for _ in range(100):
            th = threading.Thread(target=t.record, args=("m", 10, 5, 0.001))
            threads.append(th)
            th.start()
        for th in threads:
            th.join()
        assert t.call_count == 100
        assert t.total_prompt_tokens == 1000
        assert t.total_completion_tokens == 500
        assert abs(t.total_cost_usd - 0.1) < 1e-9


class TestRequestTrackingContextVar(unittest.TestCase):
    """Test start/stop/get/set tracking via contextvars."""

    def test_start_and_stop(self):
        tracker = start_request_tracking("req-A")
        assert get_active_tracker() is tracker
        tracker.record("m", 50, 25, 0.001)
        stopped = stop_request_tracking()
        assert stopped is tracker
        assert stopped.total_tokens == 75
        assert get_active_tracker() is None

    def test_stop_without_start_returns_none(self):
        # Ensure clean state
        set_active_tracker(None)
        result = stop_request_tracking()
        assert result is None

    def test_set_active_tracker(self):
        t = RequestCostTracker("manual")
        set_active_tracker(t)
        assert get_active_tracker() is t
        set_active_tracker(None)
        assert get_active_tracker() is None

    def test_nested_tracking_overwrites(self):
        """Starting a new tracking session replaces the previous one."""
        t1 = start_request_tracking("req-1")
        t2 = start_request_tracking("req-2")
        assert get_active_tracker() is t2
        stop_request_tracking()
        assert get_active_tracker() is None


if __name__ == "__main__":
    unittest.main()
