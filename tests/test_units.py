"""The string of a unit definition, as sbmlutils writes it (`sbmlutils.report.units`)."""

import json
from pathlib import Path

import libsbml
import pytest

from sbmlode.units import udef_to_string, unit_term

#: the terms `sbmlutils.report.units._unit_term_to_string` (pint) wrote for every
#: kind and a range of factors, `"<kind>|<factor>": term`
UNIT_TERMS: dict[str, str] = json.loads(
    (Path(__file__).parent / "data" / "unit_terms.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(("key", "expected"), sorted(UNIT_TERMS.items()))
def test_unit_term(key: str, expected: str) -> None:
    """A term is the one pint wrote for sbmlutils, with the closest SI prefix."""
    kind, factor = key.split("|")
    assert unit_term(float(factor), kind) == expected


#: the documents of the unit definitions of the tests, kept alive
_DOCS: list[libsbml.SBMLDocument] = []


def _model(units: list[tuple[int, float, int, float]]) -> libsbml.UnitDefinition:
    """A unit definition `u` of `(kind, multiplier, scale, exponent)` units."""
    doc = libsbml.SBMLDocument(3, 2)
    model: libsbml.Model = doc.createModel()
    unit_def: libsbml.UnitDefinition = model.createUnitDefinition()
    unit_def.setId("u")
    for kind, multiplier, scale, exponent in units:
        unit: libsbml.Unit = unit_def.createUnit()
        unit.setKind(kind)
        unit.setMultiplier(multiplier)
        unit.setScale(scale)
        unit.setExponent(exponent)
    # the document must outlive the unit definition, libsbml frees it with the model
    _DOCS.append(doc)
    return unit_def


S, G, M, L, MOL, DIM, ITEM = (
    libsbml.UNIT_KIND_SECOND,
    libsbml.UNIT_KIND_GRAM,
    libsbml.UNIT_KIND_METRE,
    libsbml.UNIT_KIND_LITRE,
    libsbml.UNIT_KIND_MOLE,
    libsbml.UNIT_KIND_DIMENSIONLESS,
    libsbml.UNIT_KIND_ITEM,
)

testdata_units = [
    ([(MOL, 1, -12, 1)], "pmol"),
    ([(S, 3600, 0, 1)], "hr"),
    ([(L, 1, -3, 1), (L, 1, 0, -1)], "ml/l"),
    ([(MOL, 1, -3, 1), (S, 60, 0, -1)], "mmol/min"),
    ([(M, 1, 0, 3)], "m^3"),
    ([(M, 1, 0, 3), (S, 1, 0, -1)], "m^3/s"),
    ([(MOL, 1, -3, 1), (L, 1, 0, -1)], "mmol/l"),
    ([(L, 1, -3, 1), (S, 1, 0, -1), (G, 1, 3, -1)], "ml/s/kg"),
    ([(DIM, 1, 0, 1)], "-"),
    ([(ITEM, 1, 0, 1)], "item"),
    ([(S, 160, 0, 1)], "160 s"),
    ([(G, 2.1, 0, 1)], "2.1 g"),
    ([(S, 11, 0, 1)], "11 s"),
    ([(S, 0.5, 0, 1)], "500 ms"),
    ([(S, 60, 0, 1)], "min"),
    ([(S, 60, 0, -1)], "1/min"),
    ([(S, 60, 0, 2)], "min^2"),
    ([(S, 3600, 0, -2)], "1/hr^2"),
    ([(S, 86400, 0, 1)], "day"),
    ([(M, 1, -2, 1)], "cm"),
    ([(M, 10, -3, 1)], "cm"),
    ([(M, 1, -2, 3)], "cm^3"),
    ([(L, 1, -3, 1)], "ml"),
    ([(G, 1, 3, -1), (S, 1, 0, 1)], "s/kg"),
    ([(G, 2.1, 0, 2)], "(2.1 g)^2"),
    ([(S, 1, 0, 0.5)], "s^0.5"),
    ([(S, 1, 0, -1.5)], "1/s^1.5"),
    ([(MOL, 1, -3, 1), (S, 160, 0, -1)], "mmol/(160 s)"),
    ([(S, 160, 0, -1)], "1/(160 s)"),
    ([(G, 2.1, 0, 1), (MOL, 1, 0, 1)], "(2.1 g)*mol"),
    ([(MOL, 1, 0, 1), (G, 2.1, 0, -2)], "mol/(2.1 g)^2"),
]


@pytest.mark.parametrize(("units", "expected"), testdata_units)
def test_unit_definition(
    units: list[tuple[int, float, int, float]], expected: str
) -> None:
    """A multiplier is a number, a named unit only for its exact factor."""
    unit_def = _model(units)
    assert udef_to_string(unit_def, unit_def.getModel()) == expected


def test_unit_definition_by_id() -> None:
    """An id is resolved in the model, a base unit is its short name."""
    unit_def = _model([(MOL, 1, -3, 1)])
    model = unit_def.getModel()
    assert udef_to_string("u", model) == "mmol"
    assert udef_to_string("second", model) == "s"
    assert udef_to_string("mole", model) == "mole"
    assert udef_to_string(None, model) is None
