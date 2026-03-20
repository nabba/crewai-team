"""Tests for app/conversation_store.py — dynamic ETA system."""
import sqlite3
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from app.conversation_store import (
    estimate_eta, get_crew_avg_duration, _DEFAULT_ETA,
)


class TestDefaultETA(unittest.TestCase):
    """Test that default ETAs are sensible when no history exists."""

    def test_known_crews_have_defaults(self):
        for crew in ("commander", "research", "coding", "writing", "self_improvement", "retrospective"):
            assert crew in _DEFAULT_ETA, f"Missing default ETA for {crew}"
            assert _DEFAULT_ETA[crew] > 0

    def test_unknown_crew_fallback(self):
        """Unknown crew should fall back to 120s default."""
        duration = get_crew_avg_duration("nonexistent_crew_xyz")
        assert duration == 120.0

    def test_estimate_eta_returns_int(self):
        result = estimate_eta("research")
        assert isinstance(result, int)


class TestGetCrewAvgDuration(unittest.TestCase):
    """Test the dynamic ETA calculation with mocked database."""

    @patch("app.conversation_store._get_conn")
    def test_returns_average_when_enough_data(self, mock_conn):
        """With >= 3 historical data points, return the average."""
        conn = MagicMock()
        mock_conn.return_value = conn
        # Simulate AVG=95.5, COUNT=5
        conn.execute.return_value.fetchone.return_value = (95.5, 5)
        result = get_crew_avg_duration("research")
        assert result == 95.5

    @patch("app.conversation_store._get_conn")
    def test_falls_back_when_insufficient_data(self, mock_conn):
        """With < 3 data points, return the default."""
        conn = MagicMock()
        mock_conn.return_value = conn
        # Simulate AVG=90.0, COUNT=2 (below threshold of 3)
        conn.execute.return_value.fetchone.return_value = (90.0, 2)
        result = get_crew_avg_duration("research")
        assert result == float(_DEFAULT_ETA["research"])

    @patch("app.conversation_store._get_conn")
    def test_falls_back_when_no_data(self, mock_conn):
        """With no historical data, return the default."""
        conn = MagicMock()
        mock_conn.return_value = conn
        conn.execute.return_value.fetchone.return_value = (None, 0)
        result = get_crew_avg_duration("coding")
        assert result == float(_DEFAULT_ETA["coding"])

    @patch("app.conversation_store._get_conn")
    def test_falls_back_on_db_error(self, mock_conn):
        """Database errors should fall back to default gracefully."""
        mock_conn.side_effect = Exception("DB connection failed")
        result = get_crew_avg_duration("writing")
        assert result == float(_DEFAULT_ETA["writing"])


class TestEstimateEta(unittest.TestCase):
    """Test the estimate_eta convenience function."""

    @patch("app.conversation_store.get_crew_avg_duration")
    def test_returns_int_from_float(self, mock_avg):
        mock_avg.return_value = 123.7
        assert estimate_eta("research") == 123

    @patch("app.conversation_store.get_crew_avg_duration")
    def test_passes_crew_name(self, mock_avg):
        mock_avg.return_value = 60.0
        estimate_eta("coding")
        mock_avg.assert_called_with("coding")


if __name__ == "__main__":
    unittest.main()
