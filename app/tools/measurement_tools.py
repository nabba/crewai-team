"""
measurement_tools.py — Unit conversion and measurement arithmetic.

Why this exists
---------------
Agents frequently need to convert between metric, imperial, and US units
(recipes, weather, geography, engineering, household conversions). Asking
the LLM to do these in-head is unreliable — especially around the well-known
US-vs-Imperial gallon/fluid-ounce divergence and around affine temperature
conversions. This module provides a deterministic, dependency-free toolset.

Three tools are exposed:

  * ``convert_unit`` — convert a single value between units in the same
    physical category (length, mass, volume, temperature, area, speed,
    pressure, energy, power, time).

  * ``measurement_calculator`` — add or subtract two measurements (with
    automatic unit reconciliation and a configurable result unit), or
    multiply / divide a measurement by a scalar.

  * ``list_units`` — discover supported categories and unit codes (so the
    agent doesn't have to guess).

US vs Imperial
--------------
Volume and mass units that differ between the two systems are kept
*explicitly* distinct:

  * ``gal_us`` (3.785411784 L) vs ``gal_uk`` (4.54609 L)
  * ``floz_us`` (≈29.5735 mL) vs ``floz_uk`` (≈28.4131 mL)
  * ``pt_us``, ``qt_us``, ``cup_us`` (US customary) vs ``pt_uk``, ``qt_uk``
  * ``ton_us`` (short ton, 907.18474 kg) vs ``ton_uk`` (long ton,
    1016.0469088 kg) vs ``tonne`` (metric, 1000 kg)

Conversion factors
------------------
All factors use the legally exact NIST / SI / ISO definitions where one
exists (inch=0.0254 m, pound=0.45359237 kg, gal_us=3.785411784 L, etc.).
The ``calorie`` is the thermochemical calorie (4.184 J). The BTU is the
International Table BTU (1055.05585262 J). The horsepower is the
mechanical horsepower (≈745.6999 W).

Usage
-----
    from app.tools.measurement_tools import create_measurement_tools
    tools = create_measurement_tools("agent_id")
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Unit registry ────────────────────────────────────────────────────
#
# Each unit maps to a tuple ``(factor, offset, aliases)`` where the unit's
# value in the category's SI base is ``base = value * factor + offset``.
# For all non-temperature categories ``offset`` is 0. Aliases include
# common spellings, plurals, and full names — all matched case-insensitively
# after stripping whitespace.

_LENGTH_BASE = "m"
_LENGTH_UNITS: dict[str, tuple[float, float, list[str]]] = {
    # SI / metric
    "m":   (1.0,        0.0, ["meter", "meters", "metre", "metres"]),
    "km":  (1000.0,     0.0, ["kilometer", "kilometers", "kilometre", "kilometres"]),
    "cm":  (0.01,       0.0, ["centimeter", "centimeters", "centimetre", "centimetres"]),
    "mm":  (0.001,      0.0, ["millimeter", "millimeters", "millimetre", "millimetres"]),
    "um":  (1e-6,       0.0, ["micrometer", "micrometers", "micron", "microns"]),
    "nm":  (1e-9,       0.0, ["nanometer", "nanometers"]),
    # Imperial / US — definitions are exact
    "in":  (0.0254,     0.0, ["inch", "inches", '"']),
    "ft":  (0.3048,     0.0, ["foot", "feet", "'"]),
    "yd":  (0.9144,     0.0, ["yard", "yards"]),
    "mi":  (1609.344,   0.0, ["mile", "miles"]),
    "nmi": (1852.0,     0.0, ["nautical mile", "nautical miles", "nm_naut"]),
}

_MASS_BASE = "kg"
_MASS_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "kg":     (1.0,            0.0, ["kilogram", "kilograms"]),
    "g":      (0.001,          0.0, ["gram", "grams"]),
    "mg":     (1e-6,           0.0, ["milligram", "milligrams"]),
    "ug":     (1e-9,           0.0, ["microgram", "micrograms"]),
    "tonne":  (1000.0,         0.0, ["metric ton", "metric tons", "t"]),
    # Avoirdupois pound is the legal definition: exactly 0.45359237 kg.
    "lb":     (0.45359237,     0.0, ["pound", "pounds", "lbs"]),
    "oz":     (0.028349523125, 0.0, ["ounce", "ounces"]),     # 1/16 lb
    "stone":  (6.35029318,     0.0, ["st", "stones"]),         # 14 lb
    "ton_us": (907.18474,      0.0, ["short ton", "short tons", "us ton"]),     # 2000 lb
    "ton_uk": (1016.0469088,   0.0, ["long ton", "long tons", "uk ton", "imperial ton"]),  # 2240 lb
}

_VOLUME_BASE = "L"
_VOLUME_UNITS: dict[str, tuple[float, float, list[str]]] = {
    # SI
    "L":   (1.0,    0.0, ["liter", "liters", "litre", "litres", "l"]),
    "mL":  (0.001,  0.0, ["milliliter", "milliliters", "millilitre", "millilitres", "ml", "cc"]),
    "m3":  (1000.0, 0.0, ["cubic meter", "cubic meters", "m^3"]),
    "cm3": (0.001,  0.0, ["cubic centimeter", "cubic centimeters", "cm^3"]),
    # US customary (1 gal_us = 3.785411784 L exactly; subdivisions exact too)
    "gal_us":  (3.785411784,    0.0, ["us gallon", "us gallons"]),
    "qt_us":   (0.946352946,    0.0, ["us quart", "us quarts"]),       # gal_us / 4
    "pt_us":   (0.473176473,    0.0, ["us pint", "us pints"]),         # gal_us / 8
    "cup_us":  (0.2365882365,   0.0, ["us cup", "us cups", "cup"]),    # gal_us / 16
    "floz_us": (0.0295735295625, 0.0, ["us fl oz", "us fluid ounce", "us fluid ounces"]),  # gal_us / 128
    "tbsp_us": (0.01478676478125, 0.0, ["us tablespoon", "us tablespoons", "tbsp"]),  # floz_us / 2
    "tsp_us":  (0.00492892159375, 0.0, ["us teaspoon", "us teaspoons", "tsp"]),       # floz_us / 6
    # Imperial / UK (1 gal_uk = 4.54609 L exactly)
    "gal_uk":  (4.54609,        0.0, ["imperial gallon", "imperial gallons", "uk gallon", "uk gallons"]),
    "qt_uk":   (1.1365225,      0.0, ["imperial quart", "imperial quarts", "uk quart"]),
    "pt_uk":   (0.56826125,     0.0, ["imperial pint", "imperial pints", "uk pint"]),
    "floz_uk": (0.0284130625,   0.0, ["imperial fl oz", "imperial fluid ounce", "uk fl oz"]),
}

_TEMPERATURE_BASE = "K"
# K = value * factor + offset, where K is the base.
#   C: K = C + 273.15                 → factor=1, offset=273.15
#   F: K = (F - 32) * 5/9 + 273.15    → factor=5/9, offset=273.15 - 32*5/9
#   R: K = R * 5/9                    → factor=5/9, offset=0
_TEMPERATURE_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "K": (1.0,     0.0,                       ["kelvin", "kelvins"]),
    "C": (1.0,     273.15,                    ["celsius", "centigrade", "degc", "°c"]),
    "F": (5.0 / 9, 273.15 - 32.0 * 5.0 / 9,   ["fahrenheit", "degf", "°f"]),
    "R": (5.0 / 9, 0.0,                       ["rankine", "degr", "°r"]),
}

_AREA_BASE = "m2"
_AREA_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "m2":  (1.0,             0.0, ["square meter", "square meters", "sqm", "m^2"]),
    "km2": (1_000_000.0,     0.0, ["square kilometer", "square kilometers", "sqkm", "km^2"]),
    "cm2": (0.0001,          0.0, ["square centimeter", "square centimeters", "cm^2"]),
    "mm2": (1e-6,            0.0, ["square millimeter", "square millimeters", "mm^2"]),
    "ha":  (10000.0,         0.0, ["hectare", "hectares"]),
    "in2": (0.00064516,      0.0, ["square inch", "square inches", "sqin", "in^2"]),
    "ft2": (0.09290304,      0.0, ["square foot", "square feet", "sqft", "ft^2"]),
    "yd2": (0.83612736,      0.0, ["square yard", "square yards", "sqyd", "yd^2"]),
    "mi2": (2_589_988.110336, 0.0, ["square mile", "square miles", "sqmi", "mi^2"]),
    "ac":  (4046.8564224,    0.0, ["acre", "acres"]),  # exactly 43560 ft²
}

_SPEED_BASE = "m/s"
_SPEED_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "m/s":  (1.0,          0.0, ["meters per second", "metres per second", "mps"]),
    "km/h": (1000.0 / 3600, 0.0, ["kph", "kmh", "kilometers per hour", "kilometres per hour"]),
    "mph":  (1609.344 / 3600, 0.0, ["miles per hour"]),
    "knot": (1852.0 / 3600,  0.0, ["knots", "kt", "kn"]),
    "ft/s": (0.3048,         0.0, ["feet per second", "fps"]),
}

_PRESSURE_BASE = "Pa"
_PRESSURE_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "Pa":   (1.0,                  0.0, ["pascal", "pascals"]),
    "kPa":  (1000.0,               0.0, ["kilopascal", "kilopascals"]),
    "MPa":  (1_000_000.0,          0.0, ["megapascal", "megapascals"]),
    "hPa":  (100.0,                0.0, ["hectopascal", "hectopascals"]),
    "bar":  (100_000.0,            0.0, ["bars"]),
    "mbar": (100.0,                0.0, ["millibar", "millibars"]),
    "atm":  (101325.0,             0.0, ["atmosphere", "atmospheres"]),
    "psi":  (6894.757293168361,    0.0, ["pounds per square inch", "lb/in2"]),
    "mmHg": (133.322387415,        0.0, ["torr", "millimeters of mercury"]),
    "inHg": (3386.389,             0.0, ["inches of mercury"]),
}

_ENERGY_BASE = "J"
_ENERGY_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "J":    (1.0,                  0.0, ["joule", "joules"]),
    "kJ":   (1000.0,               0.0, ["kilojoule", "kilojoules"]),
    "MJ":   (1_000_000.0,          0.0, ["megajoule", "megajoules"]),
    "cal":  (4.184,                0.0, ["calorie", "calories"]),         # thermochemical
    "kcal": (4184.0,               0.0, ["kilocalorie", "kilocalories", "Calorie", "food calorie"]),
    "Wh":   (3600.0,               0.0, ["watt-hour", "watt hours"]),
    "kWh":  (3_600_000.0,          0.0, ["kilowatt-hour", "kilowatt hours"]),
    "BTU":  (1055.05585262,        0.0, ["btu", "british thermal unit"]),  # IT BTU
    "ftlb": (1.3558179483314004,   0.0, ["foot-pound", "ft-lb", "foot pound"]),
}

_POWER_BASE = "W"
_POWER_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "W":     (1.0,             0.0, ["watt", "watts"]),
    "kW":    (1000.0,          0.0, ["kilowatt", "kilowatts"]),
    "MW":    (1_000_000.0,     0.0, ["megawatt", "megawatts"]),
    "hp":    (745.6998715822702, 0.0, ["horsepower", "mechanical horsepower"]),
    "BTU/h": (0.29307107017,   0.0, ["btu/h", "btu per hour"]),
}

_TIME_BASE = "s"
_TIME_UNITS: dict[str, tuple[float, float, list[str]]] = {
    "s":   (1.0,        0.0, ["second", "seconds", "sec"]),
    "ms":  (0.001,      0.0, ["millisecond", "milliseconds"]),
    "us":  (1e-6,       0.0, ["microsecond", "microseconds"]),
    "min": (60.0,       0.0, ["minute", "minutes"]),
    "h":   (3600.0,     0.0, ["hour", "hours", "hr"]),
    "d":   (86400.0,    0.0, ["day", "days"]),
    "wk":  (604800.0,   0.0, ["week", "weeks"]),
    "yr":  (31_557_600.0, 0.0, ["year", "years", "Julian year"]),  # 365.25 d
}


_CATEGORIES: dict[str, tuple[str, dict[str, tuple[float, float, list[str]]]]] = {
    "length":      (_LENGTH_BASE,      _LENGTH_UNITS),
    "mass":        (_MASS_BASE,        _MASS_UNITS),
    "volume":      (_VOLUME_BASE,      _VOLUME_UNITS),
    "temperature": (_TEMPERATURE_BASE, _TEMPERATURE_UNITS),
    "area":        (_AREA_BASE,        _AREA_UNITS),
    "speed":       (_SPEED_BASE,       _SPEED_UNITS),
    "pressure":    (_PRESSURE_BASE,    _PRESSURE_UNITS),
    "energy":      (_ENERGY_BASE,      _ENERGY_UNITS),
    "power":       (_POWER_BASE,       _POWER_UNITS),
    "time":        (_TIME_BASE,        _TIME_UNITS),
}


# ── Lookup helpers ───────────────────────────────────────────────────


def _normalize(token: str) -> str:
    """Canonicalize a unit token for case-insensitive alias matching."""
    return token.strip().lower()


def _build_alias_index() -> dict[str, tuple[str, str]]:
    """Map every alias (and primary key, lowercased) to ``(category, primary_key)``."""
    index: dict[str, tuple[str, str]] = {}
    for cat_name, (_, units) in _CATEGORIES.items():
        for primary, (_, _, aliases) in units.items():
            for token in (primary, *aliases):
                key = _normalize(token)
                # First mapping wins; collisions across categories are by design rare.
                # (e.g. "t" is set by tonne, but lowercase "t" is unambiguous within volume/mass.)
                if key and key not in index:
                    index[key] = (cat_name, primary)
    return index


_ALIAS_INDEX = _build_alias_index()


class UnitError(ValueError):
    """Raised when a unit token cannot be resolved or when categories disagree."""


def _resolve(token: str) -> tuple[str, str]:
    """Resolve a user-supplied token to ``(category, primary_unit_key)``."""
    key = _normalize(token)
    if not key:
        raise UnitError("empty unit")
    if key in _ALIAS_INDEX:
        return _ALIAS_INDEX[key]
    raise UnitError(f"unknown unit: {token!r}")


def _convert(value: float, from_unit: str, to_unit: str) -> tuple[float, str, str, str]:
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    Returns ``(converted_value, category, from_primary, to_primary)``. Raises
    :class:`UnitError` if either token is unknown or if the categories differ.
    """
    cat_from, prim_from = _resolve(from_unit)
    cat_to, prim_to = _resolve(to_unit)
    if cat_from != cat_to:
        raise UnitError(
            f"cannot convert {prim_from} ({cat_from}) to {prim_to} ({cat_to}) — "
            f"different physical categories"
        )
    units = _CATEGORIES[cat_from][1]
    f_factor, f_offset, _ = units[prim_from]
    t_factor, t_offset, _ = units[prim_to]
    base = value * f_factor + f_offset
    converted = (base - t_offset) / t_factor
    return converted, cat_from, prim_from, prim_to


def _format(value: float) -> str:
    """Format a number for human display: scientific for extreme magnitudes,
    fixed-point with up to 6 significant figures otherwise."""
    if value == 0:
        return "0"
    abs_v = abs(value)
    if abs_v >= 1e12 or abs_v < 1e-4:
        return f"{value:.6g}"
    return f"{value:.6g}"


# ── Public factory ───────────────────────────────────────────────────


def create_measurement_tools(agent_id: str) -> list:
    """Construct the measurement tools for a CrewAI agent.

    Returns ``[]`` if the CrewAI / Pydantic dependencies are not importable
    (matches the convention used by sibling tools — graceful degradation in
    minimal environments).
    """
    try:
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field
        from typing import Type
    except ImportError:
        logger.debug("measurement_tools: crewai or pydantic not installed")
        return []

    # ── convert_unit ─────────────────────────────────────────────────

    class _ConvertInput(BaseModel):
        value: float = Field(description="The numeric value to convert.")
        from_unit: str = Field(
            description=(
                "Source unit. Accepts primary keys (e.g. 'kg', 'ft', 'C', 'gal_us', "
                "'gal_uk') and common spellings ('pound', 'celsius', 'imperial gallon'). "
                "Call list_units to discover supported codes."
            )
        )
        to_unit: str = Field(description="Target unit, same category as from_unit.")

    class ConvertUnitTool(BaseTool):
        name: str = "convert_unit"
        description: str = (
            "Convert a value between units within the same physical category "
            "(length, mass, volume, temperature, area, speed, pressure, energy, "
            "power, time). Distinguishes US and Imperial units explicitly: use "
            "'gal_us' / 'gal_uk', 'floz_us' / 'floz_uk', 'ton_us' / 'ton_uk', etc."
        )
        args_schema: Type[BaseModel] = _ConvertInput

        def _run(self, value: float, from_unit: str, to_unit: str) -> str:
            try:
                converted, category, prim_from, prim_to = _convert(value, from_unit, to_unit)
            except UnitError as e:
                return f"Error: {e}"
            return (
                f"{_format(value)} {prim_from} = {_format(converted)} {prim_to} "
                f"({category})"
            )

    # ── measurement_calculator ───────────────────────────────────────

    class _CalcInput(BaseModel):
        operation: str = Field(
            description="One of 'add', 'subtract', 'multiply', 'divide'."
        )
        a_value: float = Field(description="First operand: the numeric value.")
        a_unit: str = Field(description="First operand: the unit (e.g. 'ft', 'kg', 'C').")
        b_value: float = Field(
            description=(
                "Second operand value. For add/subtract this is another measurement; "
                "for multiply/divide this is treated as a dimensionless scalar."
            )
        )
        b_unit: Optional[str] = Field(
            default=None,
            description=(
                "Second operand unit. Required for add/subtract (must share a "
                "category with a_unit). Leave empty/None for multiply/divide "
                "(scalar)."
            ),
        )
        result_unit: Optional[str] = Field(
            default=None,
            description=(
                "Optional unit for the returned value. Defaults to a_unit. Must "
                "share the same category."
            ),
        )

    class MeasurementCalculatorTool(BaseTool):
        name: str = "measurement_calculator"
        description: str = (
            "Perform arithmetic on physical measurements. Add/subtract two "
            "quantities with units (auto-converts b to a's category before "
            "combining); multiply/divide a quantity by a dimensionless scalar. "
            "For temperature, add/subtract operate on the underlying Kelvin "
            "values, so '20 C + 5 C' = 25 C (not a meaningless 313.3 K sum)."
        )
        args_schema: Type[BaseModel] = _CalcInput

        def _run(
            self,
            operation: str,
            a_value: float,
            a_unit: str,
            b_value: float,
            b_unit: Optional[str] = None,
            result_unit: Optional[str] = None,
        ) -> str:
            op = (operation or "").strip().lower()
            try:
                cat_a, prim_a = _resolve(a_unit)
            except UnitError as e:
                return f"Error: {e}"

            target = result_unit or prim_a
            try:
                cat_target, prim_target = _resolve(target)
            except UnitError as e:
                return f"Error: result_unit invalid — {e}"
            if cat_target != cat_a:
                return (
                    f"Error: result_unit '{prim_target}' ({cat_target}) does not "
                    f"match a_unit '{prim_a}' ({cat_a})"
                )

            if op in ("add", "subtract", "+", "-"):
                if not b_unit:
                    return "Error: add/subtract requires b_unit (a measurement, not a scalar)"
                try:
                    cat_b, prim_b = _resolve(b_unit)
                except UnitError as e:
                    return f"Error: {e}"
                if cat_b != cat_a:
                    return (
                        f"Error: cannot {op} {prim_a} ({cat_a}) and {prim_b} "
                        f"({cat_b}) — different categories"
                    )
                if cat_a == "temperature":
                    # Operate on intervals (Kelvin differences), then re-anchor
                    # the result on a_value's reading. Adding 5 C to 20 C should
                    # give 25 C, not 313.15 K + 278.15 K = nonsense.
                    units = _CATEGORIES[cat_a][1]
                    b_factor, _, _ = units[prim_b]
                    a_factor, _, _ = units[prim_a]
                    # Convert b's magnitude (interval) to a_unit's interval scale,
                    # then to target_unit's interval scale.
                    b_in_a_interval = b_value * b_factor / a_factor
                    combined_in_a = (
                        a_value + b_in_a_interval if op in ("add", "+")
                        else a_value - b_in_a_interval
                    )
                    # Now convert the absolute reading from a_unit to target_unit.
                    combined, _, _, _ = _convert(combined_in_a, prim_a, prim_target)
                    op_sign = "+" if op in ("add", "+") else "-"
                    return (
                        f"{_format(a_value)} {prim_a} {op_sign} "
                        f"{_format(b_value)} {prim_b} = "
                        f"{_format(combined)} {prim_target} ({cat_a})"
                    )
                # Non-temperature: convert both to a common unit (target), sum.
                a_in_target, _, _, _ = _convert(a_value, prim_a, prim_target)
                b_in_target, _, _, _ = _convert(b_value, prim_b, prim_target)
                result = (
                    a_in_target + b_in_target if op in ("add", "+")
                    else a_in_target - b_in_target
                )
                op_sign = "+" if op in ("add", "+") else "-"
                return (
                    f"{_format(a_value)} {prim_a} {op_sign} "
                    f"{_format(b_value)} {prim_b} = "
                    f"{_format(result)} {prim_target} ({cat_a})"
                )

            if op in ("multiply", "*", "x", "times"):
                if b_unit:
                    return (
                        "Error: multiply expects a scalar (b_unit must be empty). "
                        "Multiplying two measurements changes the dimension and is "
                        "not supported by this tool."
                    )
                a_in_target, _, _, _ = _convert(a_value, prim_a, prim_target)
                result = a_in_target * b_value
                return (
                    f"{_format(a_value)} {prim_a} × {_format(b_value)} = "
                    f"{_format(result)} {prim_target} ({cat_a})"
                )

            if op in ("divide", "/"):
                if b_unit:
                    return (
                        "Error: divide expects a scalar (b_unit must be empty). "
                        "Dividing two measurements yields a ratio or new dimension "
                        "and is not supported by this tool."
                    )
                if b_value == 0:
                    return "Error: division by zero"
                a_in_target, _, _, _ = _convert(a_value, prim_a, prim_target)
                result = a_in_target / b_value
                return (
                    f"{_format(a_value)} {prim_a} ÷ {_format(b_value)} = "
                    f"{_format(result)} {prim_target} ({cat_a})"
                )

            return (
                f"Error: unknown operation {operation!r}. Use add, subtract, "
                f"multiply, or divide."
            )

    # ── list_units ───────────────────────────────────────────────────

    class _ListInput(BaseModel):
        category: Optional[str] = Field(
            default=None,
            description=(
                "Optional category filter: length, mass, volume, temperature, "
                "area, speed, pressure, energy, power, time. Omit to list all."
            ),
        )

    class ListUnitsTool(BaseTool):
        name: str = "list_units"
        description: str = (
            "List supported unit categories and the canonical unit codes within "
            "each. Use this to discover the exact code to pass to convert_unit "
            "or measurement_calculator (especially the explicit US/Imperial "
            "splits like gal_us / gal_uk)."
        )
        args_schema: Type[BaseModel] = _ListInput

        def _run(self, category: Optional[str] = None) -> str:
            cats = _CATEGORIES.items()
            if category:
                key = category.strip().lower()
                if key not in _CATEGORIES:
                    return (
                        f"Error: unknown category {category!r}. Known: "
                        f"{', '.join(sorted(_CATEGORIES))}"
                    )
                cats = [(key, _CATEGORIES[key])]
            lines: list[str] = []
            for cat_name, (base, units) in cats:
                lines.append(f"== {cat_name} (base: {base}) ==")
                for primary, (_, _, aliases) in units.items():
                    alias_str = f" — {', '.join(aliases[:3])}" if aliases else ""
                    lines.append(f"  {primary}{alias_str}")
                lines.append("")
            return "\n".join(lines).rstrip()

    return [
        ConvertUnitTool(),
        MeasurementCalculatorTool(),
        ListUnitsTool(),
    ]
