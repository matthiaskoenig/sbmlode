"""The string of a unit definition, `mmol/min`.

The string is the one sbmlutils writes for a unit (`sbmlutils.report.units`), without
pint: a unit is `(multiplier * 10^scale * kind)^exponent`, a term `multiplier *
10^scale * kind` is written with the closest SI prefix, `ms`, and a magnitude which
is not 1 in front of it, `160 s`; a factor which names a unit of its own is written
by that name, `min`, `hr`, `day`, `cm`. A term of a dimensionless kind (dimensionless,
item, avogadro) gets no prefix. A product is `*`, a quotient `/`, a term with a
magnitude is grouped in parentheses when it is a factor of more, `mmol/(160 s)`.
"""

import math

import libsbml

__all__ = ["udef_to_string", "unit_term"]

#: the symbol of every unit kind of SBML, as pint abbreviates it
SYMBOLS: dict[str, str] = {
    "ampere": "A",
    "avogadro": "avogadro",
    "becquerel": "Bq",
    "candela": "cd",
    "coulomb": "C",
    "dimensionless": "",
    "farad": "F",
    "gram": "g",
    "gray": "Gy",
    "henry": "H",
    "hertz": "Hz",
    "item": "item",
    "joule": "J",
    "katal": "kat",
    "kelvin": "K",
    "kilogram": "g",
    "litre": "l",
    "liter": "l",
    "lumen": "lm",
    "lux": "lx",
    "metre": "m",
    "meter": "m",
    "mole": "mol",
    "newton": "N",
    "ohm": "Ω",
    "pascal": "Pa",
    "radian": "rad",
    "second": "s",
    "siemens": "S",
    "sievert": "Sv",
    "steradian": "sr",
    "tesla": "T",
    "volt": "V",
    "watt": "W",
    "weber": "Wb",
    "Celsius": "°C",
}

#: the kinds which are dimensionless, they get no prefix
_DIMENSIONLESS = frozenset({"dimensionless", "item", "avogadro"})

#: the SI prefixes by their power of ten
_PREFIXES: dict[int, str] = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "μ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

#: units which are named for a multiple of an SBML base unit, as
#: (kind, factor, name); a factor is compared to the unit, never the text
_NAMED_UNITS: list[tuple[str, float, str]] = [
    ("second", 60.0, "min"),
    ("second", 3600.0, "hr"),
    ("second", 86400.0, "day"),
    ("metre", 0.01, "cm"),
]

#: the short names of the base units, for a unit given by its kind
_SHORT_NAMES: dict[str, str] = {
    "metre": "m",
    "meter": "m",
    "liter": "l",
    "litre": "l",
    "dimensionless": "-",
    "second": "s",
}


def _isclose(a: float, b: float) -> bool:
    """`numpy.isclose` with its default tolerances."""
    return abs(a - b) <= 1e-8 + 1e-5 * abs(b)


def unit_term(factor: float, kind: str) -> str:
    """The term `factor * kind` of a unit without its exponent.

    Args:
        factor: the multiplier times ten to the power of the scale of the unit
        kind: the SBML unit kind, as `libsbml.UnitKind_toString` names it

    Returns:
        the term, e.g. `ms`, `160 s` or `min`; a dimensionless term is its
        magnitude, the empty string if that is 1
    """
    for named_kind, named_factor, name in _NAMED_UNITS:
        if kind == named_kind and _isclose(factor, named_factor):
            return name
    symbol = SYMBOLS.get(kind, kind)
    if kind == "kilogram":
        # the prefix of the kilogram is the one of the gram
        factor *= 1000.0
    magnitude = factor
    if kind not in _DIMENSIONLESS and factor > 0:
        power = 3 * math.floor(math.log10(factor) / 3)
        power = max(-24, min(24, power))
        magnitude = factor / 10.0**power
        symbol = _PREFIXES[power] + symbol
    if _isclose(magnitude, 1.0):
        return symbol
    number = f"{magnitude:g}"
    return f"{number} {symbol}" if symbol else number


def _group(term: str) -> str:
    """Enclose a term which consists of a magnitude and a unit in parentheses."""
    if " " in term and not term.startswith("("):
        return f"({term})"
    return term


def udef_to_string(
    udef: libsbml.UnitDefinition | str | None, model: libsbml.Model | None = None
) -> str | None:
    """The string of a unit definition.

    Args:
        udef: the unit definition, or its id, or the name of a unit kind
        model: the model which resolves an id

    Returns:
        the string, `-` for a unit definition without units, `None` without a unit

    Raises:
        ValueError: for an id without a model
    """
    if udef is None:
        return None
    ud: libsbml.UnitDefinition | None
    if isinstance(udef, str):
        if libsbml.UnitKind_forName(udef) != libsbml.UNIT_KIND_INVALID:
            return _SHORT_NAMES.get(udef, udef)
        if model is None:
            raise ValueError(
                f"A model is required to resolve the unit definition '{udef}'."
            )
        ud = model.getUnitDefinition(udef)
    else:
        ud = udef
    if not ud:
        return "-"
    noms: list[str] = []
    denoms: list[str] = []
    for u in ud.getListOfUnits():
        e = u.getExponentAsDouble()
        k = libsbml.UnitKind_toString(u.getKind())
        term = unit_term(float(u.getMultiplier()) * 10 ** u.getScale(), k)
        if not term or e == 0.0:
            continue
        if abs(e) != 1.0:
            exponent = f"{abs(e):g}"
            # the exponent applies to the magnitude as well: (2.1 g)^2
            term = f"({term})^{exponent}" if " " in term else f"{term}^{exponent}"
        (noms if e > 0.0 else denoms).append(term)
    if len(noms) > 1 or denoms:
        noms = [_group(t) for t in noms]
        denoms = [_group(t) for t in denoms]
    nom = "*".join(noms)
    denom = "/".join(denoms)
    if nom and denom:
        return f"{nom}/{denom}"
    if nom:
        return nom
    if denom:
        return f"1/{denom}"
    return "-"
