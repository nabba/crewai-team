"""
test_measurement_tools.py — Unit tests for measurement_tools.

Run: pytest tests/test_measurement_tools.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The pure conversion helpers do not depend on crewai; the tool-surface
# classes do. We import the helpers eagerly and gate the tool tests on
# crewai availability via a per-class skip marker.
from app.tools.measurement_tools import (
    UnitError,
    _convert,
    _resolve,
    create_measurement_tools,
)

_crewai_missing = False
try:
    import crewai  # noqa: F401
except ImportError:
    _crewai_missing = True

requires_crewai = pytest.mark.skipif(
    _crewai_missing, reason="crewai not installed (runs in Docker)"
)


def _approx(a: float, b: float, *, rel: float = 1e-9, abs_: float = 1e-9) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ── Factory ──────────────────────────────────────────────────────────


@requires_crewai
class TestFactory:
    def test_returns_three_tools(self):
        tools = create_measurement_tools("test")
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"convert_unit", "measurement_calculator", "list_units"}

    def test_each_tool_has_args_schema(self):
        tools = create_measurement_tools("test")
        for t in tools:
            assert t.args_schema is not None


# ── Resolution / aliases ─────────────────────────────────────────────


class TestResolve:
    def test_primary_keys_resolve(self):
        assert _resolve("kg") == ("mass", "kg")
        assert _resolve("ft") == ("length", "ft")
        assert _resolve("C")  == ("temperature", "C")

    def test_aliases_resolve_case_insensitively(self):
        assert _resolve("Kilograms") == ("mass", "kg")
        assert _resolve("FEET") == ("length", "ft")
        assert _resolve("celsius") == ("temperature", "C")
        assert _resolve("US gallon") == ("volume", "gal_us")
        assert _resolve("imperial gallon") == ("volume", "gal_uk")

    def test_unknown_unit_raises(self):
        with pytest.raises(UnitError):
            _resolve("snorgleflorps")
        with pytest.raises(UnitError):
            _resolve("")


# ── Conversion correctness ───────────────────────────────────────────


class TestLengthConversions:
    def test_inch_to_cm_exact(self):
        # 1 in = 2.54 cm exactly
        v, *_ = _convert(1, "in", "cm")
        assert _approx(v, 2.54)

    def test_mile_to_km(self):
        # 1 mi = 1.609344 km exactly
        v, *_ = _convert(1, "mi", "km")
        assert _approx(v, 1.609344)

    def test_round_trip_meters_feet(self):
        v1, *_ = _convert(100, "m", "ft")
        v2, *_ = _convert(v1, "ft", "m")
        assert _approx(v2, 100, abs_=1e-9)

    def test_nautical_mile_is_1852_m(self):
        v, *_ = _convert(1, "nmi", "m")
        assert _approx(v, 1852.0)


class TestMassConversions:
    def test_pound_to_kg_exact_definition(self):
        v, *_ = _convert(1, "lb", "kg")
        assert _approx(v, 0.45359237)

    def test_us_short_ton_vs_uk_long_ton(self):
        # short ton = 2000 lb ; long ton = 2240 lb. Long is 12% heavier.
        short_kg, *_ = _convert(1, "ton_us", "kg")
        long_kg, *_ = _convert(1, "ton_uk", "kg")
        assert long_kg > short_kg
        assert _approx(long_kg / short_kg, 2240.0 / 2000.0, rel=1e-6)

    def test_stone_is_14_lb(self):
        v, *_ = _convert(1, "stone", "lb")
        assert _approx(v, 14.0, rel=1e-9)


class TestVolumeConversionsUSvsImperial:
    """The whole reason for this tool: keep US and Imperial gallons separate."""

    def test_us_gallon_vs_imperial_gallon_differ(self):
        us_l, *_ = _convert(1, "gal_us", "L")
        uk_l, *_ = _convert(1, "gal_uk", "L")
        assert _approx(us_l, 3.785411784, abs_=1e-9)
        assert _approx(uk_l, 4.54609,     abs_=1e-9)
        assert uk_l > us_l  # imperial gallon is larger

    def test_us_floz_vs_uk_floz(self):
        # US fl oz ≈ 29.57 mL ; UK fl oz ≈ 28.41 mL — US is *larger*
        # (despite the imperial gallon being larger overall, because
        # 1 gal_us = 128 floz_us but 1 gal_uk = 160 floz_uk).
        us_ml, *_ = _convert(1, "floz_us", "mL")
        uk_ml, *_ = _convert(1, "floz_uk", "mL")
        assert _approx(us_ml, 29.5735295625, abs_=1e-9)
        assert _approx(uk_ml, 28.4130625,    abs_=1e-9)
        assert us_ml > uk_ml

    def test_us_subdivisions_consistent(self):
        gal, *_ = _convert(1, "gal_us", "L")
        qt, *_ = _convert(4, "qt_us", "L")
        pt, *_ = _convert(8, "pt_us", "L")
        cup, *_ = _convert(16, "cup_us", "L")
        floz, *_ = _convert(128, "floz_us", "L")
        for v in (qt, pt, cup, floz):
            assert _approx(v, gal, rel=1e-6)


class TestTemperatureConversions:
    def test_freezing_point(self):
        v, *_ = _convert(0, "C", "F")
        assert _approx(v, 32.0)
        v, *_ = _convert(0, "C", "K")
        assert _approx(v, 273.15)

    def test_boiling_point(self):
        v, *_ = _convert(100, "C", "F")
        assert _approx(v, 212.0)
        v, *_ = _convert(100, "C", "K")
        assert _approx(v, 373.15)

    def test_absolute_zero(self):
        v, *_ = _convert(0, "K", "C")
        assert _approx(v, -273.15)
        v, *_ = _convert(0, "K", "F")
        assert _approx(v, -459.67, abs_=1e-9)

    def test_round_trip_f_to_c(self):
        for f in (-40, -32, 0, 32, 75, 100, 212, 451):
            c, *_ = _convert(f, "F", "C")
            f2, *_ = _convert(c, "C", "F")
            assert _approx(f, f2, abs_=1e-9)

    def test_minus_40_is_meeting_point(self):
        # The famous F == C crossover.
        v, *_ = _convert(-40, "F", "C")
        assert _approx(v, -40.0)

    def test_rankine(self):
        v, *_ = _convert(491.67, "R", "F")
        assert _approx(v, 32.0, abs_=1e-6)


class TestSpeedConversions:
    def test_kmh_to_mph(self):
        v, *_ = _convert(100, "km/h", "mph")
        assert _approx(v, 100 / 1.609344, rel=1e-9)

    def test_knot_definition(self):
        # 1 knot = 1 nmi/h = 1852 m/3600 s
        v, *_ = _convert(1, "knot", "m/s")
        assert _approx(v, 1852.0 / 3600.0)


class TestCrossCategoryRejected:
    def test_kg_to_meters_errors(self):
        with pytest.raises(UnitError):
            _convert(1, "kg", "m")


# ── Tool surface ─────────────────────────────────────────────────────


@requires_crewai
class TestConvertUnitTool:
    def setup_method(self):
        tools = create_measurement_tools("t")
        self.tool = next(t for t in tools if t.name == "convert_unit")

    def test_simple_conversion(self):
        out = self.tool._run(value=100, from_unit="km", to_unit="mi")
        assert "km" in out and "mi" in out
        # 100 km ≈ 62.1371 mi
        assert "62.13" in out

    def test_temperature_conversion(self):
        out = self.tool._run(value=75, from_unit="F", to_unit="C")
        # 75 F = 23.888... C → format as 23.8889
        assert "C" in out
        assert "23.8" in out

    def test_unknown_unit_returns_error_string(self):
        out = self.tool._run(value=1, from_unit="bogus", to_unit="m")
        assert out.startswith("Error:")

    def test_cross_category_returns_error_string(self):
        out = self.tool._run(value=1, from_unit="kg", to_unit="L")
        assert out.startswith("Error:")


@requires_crewai
class TestMeasurementCalculatorTool:
    def setup_method(self):
        tools = create_measurement_tools("t")
        self.tool = next(t for t in tools if t.name == "measurement_calculator")

    def test_add_same_unit(self):
        out = self.tool._run(
            operation="add", a_value=3, a_unit="ft", b_value=2, b_unit="ft"
        )
        # 3 ft + 2 ft = 5 ft
        assert "5 ft" in out

    def test_add_mixed_units(self):
        # 3 ft + 12 in = 4 ft
        out = self.tool._run(
            operation="add", a_value=3, a_unit="ft", b_value=12, b_unit="in"
        )
        assert "4 ft" in out

    def test_add_with_result_unit(self):
        # 5 kg + 200 g in lb ≈ 11.4641 lb
        out = self.tool._run(
            operation="add", a_value=5, a_unit="kg",
            b_value=200, b_unit="g", result_unit="lb",
        )
        assert "lb" in out
        assert "11.4" in out

    def test_subtract(self):
        out = self.tool._run(
            operation="subtract", a_value=10, a_unit="kg",
            b_value=2, b_unit="kg",
        )
        assert "8 kg" in out

    def test_temperature_addition_treats_b_as_interval(self):
        # 20 C + 5 C should give 25 C, not 313.3 K.
        out = self.tool._run(
            operation="add", a_value=20, a_unit="C", b_value=5, b_unit="C"
        )
        assert "25 C" in out

    def test_temperature_addition_mixed_scales(self):
        # 20 C + 9 F (where 9 F-degrees = 5 C-degrees as an interval)
        # → 25 C
        out = self.tool._run(
            operation="add", a_value=20, a_unit="C", b_value=9, b_unit="F"
        )
        # Allow some formatting flex
        assert " C " in out + " "
        assert "25" in out

    def test_multiply_by_scalar(self):
        out = self.tool._run(
            operation="multiply", a_value=30, a_unit="mph",
            b_value=2, b_unit=None,
        )
        assert "60 mph" in out

    def test_divide_by_scalar(self):
        out = self.tool._run(
            operation="divide", a_value=100, a_unit="km",
            b_value=4, b_unit=None,
        )
        assert "25 km" in out

    def test_divide_by_zero_returns_error(self):
        out = self.tool._run(
            operation="divide", a_value=10, a_unit="m",
            b_value=0, b_unit=None,
        )
        assert "Error" in out and "zero" in out

    def test_multiply_with_unit_b_returns_error(self):
        out = self.tool._run(
            operation="multiply", a_value=2, a_unit="m",
            b_value=3, b_unit="m",
        )
        assert out.startswith("Error:")

    def test_add_cross_category_returns_error(self):
        out = self.tool._run(
            operation="add", a_value=1, a_unit="kg",
            b_value=1, b_unit="m",
        )
        assert out.startswith("Error:")

    def test_unknown_operation_returns_error(self):
        out = self.tool._run(
            operation="modulate", a_value=1, a_unit="m",
            b_value=1, b_unit="m",
        )
        assert out.startswith("Error:")


@requires_crewai
class TestListUnitsTool:
    def setup_method(self):
        tools = create_measurement_tools("t")
        self.tool = next(t for t in tools if t.name == "list_units")

    def test_list_all(self):
        out = self.tool._run(category=None)
        # Spot-check several categories present
        for cat in ("length", "mass", "volume", "temperature", "speed"):
            assert cat in out
        # And US/Imperial split is visible
        assert "gal_us" in out
        assert "gal_uk" in out

    def test_list_filtered(self):
        out = self.tool._run(category="temperature")
        assert "temperature" in out
        assert "K" in out and "C" in out and "F" in out
        # Other categories should not appear
        assert "length" not in out

    def test_unknown_category(self):
        out = self.tool._run(category="vibes")
        assert out.startswith("Error:")
