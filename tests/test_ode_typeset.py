"""The typed target: `OdeSystem.typeset` and the `wrap` of its symbols."""

import libsbml
import pytest
from ode_helpers import edit_sbml, model_sbml
from resources import REPRESSILATOR_SBML

from sbmlode import OdeSystem
from sbmlode.documents import TypesetEquation, TypesetSystem
from sbmlode.system import Symbol


@pytest.fixture(scope="module")
def repressilator() -> OdeSystem:
    """The ODE system of the repressilator."""
    return OdeSystem.from_sbml(REPRESSILATOR_SBML)


def _link(symbol: Symbol, typeset: str) -> str:
    """A symbol wrapped with the last part of its element."""
    return rf"\htmlData{{id={symbol.source[-1]}}}{{{typeset}}}"


def test_typeset_is_typed(repressilator: OdeSystem) -> None:
    """The system has its sections as dataclasses with the variables as symbols."""
    typeset = repressilator.typeset()
    assert isinstance(typeset, TypesetSystem)
    assert (len(typeset.odes), len(typeset.reactions), len(typeset.assignments)) == (
        6,
        12,
        9,
    )
    ode = typeset.odes[0]
    assert isinstance(ode, TypesetEquation)
    assert ode.variable is not None
    assert ode.variable.sid == "PX"
    assert ode.lhs == r"\frac{\mathrm{d} \,\mathrm{PX}}{\mathrm{d} t}"
    assert ode.lines == (r"v_{\mathrm{Reaction4}} - v_{\mathrm{Reaction7}}",)
    assert ode.origin == "reactions"
    assert typeset.reactions[0].variable.sid == "Reaction1"
    assert typeset.assignments[0].variable is not None
    assert typeset.assignments[0].origin == "assignment_rule"


def test_wrap_sees_every_symbol(repressilator: OdeSystem) -> None:
    """Every symbol of the math and of a left hand side is wrapped once."""
    typeset = repressilator.typeset("latex", "id", _link)
    ode = typeset.odes[0]
    assert ode.lhs == r"\frac{\mathrm{d} \,\htmlData{id=PX}{\mathrm{PX}}}{\mathrm{d} t}"
    assert ode.lines == (
        r"\htmlData{id=Reaction4}{v_{\mathrm{Reaction4}}} - "
        r"\htmlData{id=Reaction7}{v_{\mathrm{Reaction7}}}",
    )
    reaction = typeset.reactions[9]
    assert reaction.symbol == r"\htmlData{id=Reaction10}{v_{\mathrm{Reaction10}}}"
    assert r"\htmlData{id=PZ}{\mathrm{PZ}}" in reaction.lines[0]
    assert r"\htmlData{id=KM}" in reaction.lines[0]


def test_without_wrap_the_documents_are_unchanged(repressilator: OdeSystem) -> None:
    """`typeset` without `wrap` is what the documents are rendered from."""
    typeset = repressilator.typeset("latex")
    tex = repressilator.render("latex")
    assert all(ode.lhs in tex for ode in typeset.odes)
    assert all(line in tex for r in typeset.reactions for line in r.lines)


def test_typst_dialect(repressilator: OdeSystem) -> None:
    """The typst dialect writes typst math."""
    typeset = repressilator.typeset("typst")
    assert typeset.odes[0].lhs == '(dif upright("PX"))/(dif t)'


def test_unknown_dialect(repressilator: OdeSystem) -> None:
    """A dialect which is neither latex nor typst raises."""
    with pytest.raises(ValueError, match="dialect"):
        repressilator.typeset("markdown")  # ty: ignore[invalid-argument-type]


def _local_parameter(model: libsbml.Model) -> None:
    """A local parameter `k` of the reaction J0."""
    law: libsbml.KineticLaw = model.getReaction("J0").getKineticLaw()
    parameter: libsbml.LocalParameter = law.createLocalParameter()
    parameter.setId("k")
    parameter.setValue(0.2)


def test_local_parameter_element() -> None:
    """A renamed local parameter names its reaction and its own id."""
    sbml = edit_sbml(
        model_sbml("""
            compartment c = 1; species S in c = 10
            J0: S -> ; k*S; k = 0.1; J0_k = 5
        """),
        _local_parameter,
    )
    system = OdeSystem.from_sbml(sbml)
    local = system.symbol("J0_k_1")
    assert local.element == ("J0", "k")
    assert local.source == ("J0", "k")
    assert system.symbol("J0_k").source == ("J0_k",)
    typeset = system.typeset("latex", "id", _link)
    assert r"\htmlData{id=k}" in typeset.reactions[0].lines[0]


def test_amount_element() -> None:
    """The amount of a species names the species."""
    system = OdeSystem.from_sbml(
        model_sbml("""
            compartment c = 2; c' = 0.1; species S in c = 10
            J0: S -> ; k*S; k = 0.1
        """)
    )
    amount = system.quantity("n_S")
    assert amount.symbol.source == ("S",)
    typeset = system.typeset("latex", "id", _link)
    lhs = {ode.variable.sid: ode.lhs for ode in typeset.odes if ode.variable}
    assert r"\htmlData{id=S}{n_{S}}" in lhs["n_S"]


def test_event_and_unsupported_are_typed() -> None:
    """An event has its assignments, an unsupported construct its element."""
    sbml = model_sbml("""
        compartment c = 1; species S in c = 10; x = 1
        E0: at time > 1: S = 1
        0 = x - 2*S
    """)
    typeset = OdeSystem.from_sbml(sbml).typeset("latex", "id", _link)
    event = typeset.events[0]
    assert event.symbol.sid == "E0"
    assert event.assignments[0].variable.sid == "S"
    assert event.assignments[0].lhs == r"\htmlData{id=S}{S}"
    assert typeset.unsupported[0].construct == "algebraic rule"
