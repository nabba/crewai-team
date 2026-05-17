"""Pinning + sanity tests for app.substrate.cloud_cost.

The estimator is the operator-visible safety gate before any
``terraform apply``. Tests pin:
  * Tier monotonicity (prod >= cheapest, every region).
  * Itemization invariants (always has compute + storage line items).
  * Region multipliers behave (europe-north1 > us-central1).
  * Rough numeric range — within these bands, the estimate is "right
    enough". The bands are wide because list prices drift; the test
    catches a silent 10× rate-card change, not a 5% drift.

Refresh policy: when GCP/AWS publishes new list prices, update the
constants in ``cloud_cost.py`` AND widen/narrow these bands as needed.
The bands are the operator's source of truth for "what should I expect".
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.substrate.cloud_cost import (
    CostBreakdown,
    estimate_monthly_cost,
    format_breakdown,
)


class TestPublicAPI:
    def test_basic_call_returns_breakdown(self):
        b = estimate_monthly_cost("gcp", "cheapest")
        assert isinstance(b, CostBreakdown)
        assert b.target == "gcp"
        assert b.tier == "cheapest"
        # Default Helsinki-closest region
        assert b.region == "europe-north1"

    def test_aws_default_region(self):
        b = estimate_monthly_cost("aws", "cheapest")
        assert b.region == "eu-north-1"

    def test_explicit_region_honored(self):
        b = estimate_monthly_cost("gcp", "cheapest", region="us-central1")
        assert b.region == "us-central1"

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="unknown cloud target"):
            estimate_monthly_cost("azure", "cheapest")  # type: ignore[arg-type]

    def test_unknown_tier_raises(self):
        with pytest.raises(ValueError, match="unknown tier"):
            estimate_monthly_cost("gcp", "huge")  # type: ignore[arg-type]


class TestItemization:
    def test_always_has_compute_and_storage(self):
        for target in ("gcp", "aws"):
            for tier in ("cheapest", "prod"):
                b = estimate_monthly_cost(target, tier)  # type: ignore[arg-type]
                cats = {li.category for li in b.line_items}
                assert "compute" in cats, f"{target}/{tier} missing compute"
                assert "storage" in cats, f"{target}/{tier} missing storage"
                assert "network" in cats, f"{target}/{tier} missing network"

    def test_domain_adds_load_balancer_line(self):
        b_no = estimate_monthly_cost("gcp", "cheapest", has_domain=False)
        b_yes = estimate_monthly_cost("gcp", "cheapest", has_domain=True)
        assert b_yes.total_monthly_usd > b_no.total_monthly_usd
        # has_domain adds exactly one line item (the LB)
        assert len(b_yes.line_items) == len(b_no.line_items) + 1

    def test_monitoring_off_drops_a_line(self):
        b_on = estimate_monthly_cost("gcp", "cheapest", enable_monitoring=True)
        b_off = estimate_monthly_cost("gcp", "cheapest", enable_monitoring=False)
        assert b_on.total_monthly_usd > b_off.total_monthly_usd


class TestMonotonicity:
    def test_prod_costs_more_than_cheapest_everywhere(self):
        """For both clouds, prod tier must exceed cheapest tier."""
        for target in ("gcp", "aws"):
            cheap = estimate_monthly_cost(target, "cheapest")  # type: ignore[arg-type]
            prod = estimate_monthly_cost(target, "prod")  # type: ignore[arg-type]
            assert prod.total_monthly_usd > cheap.total_monthly_usd, (
                f"{target}: prod must cost more than cheapest"
            )

    def test_europe_costs_more_than_us(self):
        """Region multiplier sanity — europe-north1 > us-central1."""
        eu = estimate_monthly_cost("gcp", "cheapest", region="europe-north1")
        us = estimate_monthly_cost("gcp", "cheapest", region="us-central1")
        assert eu.total_monthly_usd > us.total_monthly_usd


class TestNumericBands:
    """Pin the rough range — wide bands catch silent 10× drift, allow
    normal 10-20% rate-card movement without churn.
    """

    @pytest.mark.parametrize("target,tier,low,high", [
        # GCP at europe-north1 with monitoring on, no domain
        ("gcp", "cheapest",  100, 250),
        ("gcp", "prod",      450, 900),
        # AWS at eu-north-1, same shape
        ("aws", "cheapest",  150, 350),
        ("aws", "prod",      400, 700),
    ])
    def test_default_install_within_band(self, target, tier, low, high):
        b = estimate_monthly_cost(target, tier)  # type: ignore[arg-type]
        assert low <= b.total_monthly_usd <= high, (
            f"{target}/{tier}/{b.region}: ${b.total_monthly_usd} not in [${low}, ${high}]. "
            f"If list prices changed, update the band AND the relevant "
            f"constants in app/substrate/cloud_cost.py."
        )


class TestByCategory:
    def test_by_category_sums_to_total(self):
        b = estimate_monthly_cost("gcp", "cheapest")
        cat_total = sum(b.by_category().values())
        # Floating-point tolerance
        assert abs(cat_total - b.total_monthly_usd) < 0.05


class TestFormatBreakdown:
    def test_human_format_includes_target_tier_total(self):
        b = estimate_monthly_cost("gcp", "cheapest")
        out = format_breakdown(b)
        assert "gcp" in out
        assert "cheapest" in out
        assert "europe-north1" in out
        assert "TOTAL" in out
        assert f"{b.total_monthly_usd:.2f}" in out


class TestDictSerialization:
    def test_to_dict_roundtrip_shape(self):
        b = estimate_monthly_cost("gcp", "cheapest")
        d = b.to_dict()
        assert d["target"] == "gcp"
        assert d["tier"] == "cheapest"
        assert d["region"] == "europe-north1"
        assert "line_items" in d
        assert "by_category" in d
        assert "total_monthly_usd" in d
        assert "total_annual_usd" in d
        # Line items themselves are dicts
        for li in d["line_items"]:
            assert "category" in li
            assert "resource" in li
            assert "monthly_usd" in li
