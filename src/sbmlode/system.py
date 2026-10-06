"""Analysis of an SBML model into its system of ordinary differential equations.

`OdeSystem.from_sbml` resolves the semantics of SBML core once, as roadrunner
implements them, so that a format only prints what the system holds:

- **States** are the species which are neither constant nor boundary nor assigned
  and take part in a reaction, and every compartment, species, parameter and species
  reference with a rate rule. Every other quantity is constant or assigned; a
  quantity an event changes is constant between the events (`Quantity.constant` is
  the flag of SBML, `Quantity.role` the role in the system).
- **Species** are states in the quantity the model declares, in amount if
  `hasOnlySubstanceUnits`, else in concentration, as roadrunner holds them. The ODE of
  a species in concentration is the one of SBML L3V2 section 3.4.6 (rateOf),
  `d[S]/dt = (1/V) dS/dt - ([S]/V) dV/dt`: the reaction terms divided by its
  compartment and, if the size of the compartment changes continuously, the
  dilution `-(S/V) dV/dt` (`Ode.size_rate`). A rate rule applies to the species as
  written.
- **The rate of a size** `dV/dt` of a compartment with a rate rule or an assignment
  rule is the assignment of origin `size_rate` of a symbol of kind `rate`
  (`OdeSystem.size_rates`, `d<V>_dt` made unique against the ids of the model, its
  source the compartment): the right hand side of the rate rule, or the chain rule of
  the assignment rule, `dV/dt = df/dt + sum df/dy dy/dt` with the rates of the
  states inlined (`astutil.derivative`); a rule which cannot be differentiated (e.g.
  `delay`) is unsupported, `rate of an assigned size`.
- **A species in concentration in a compartment whose size changes continuously**
  which is no state otherwise (a boundary, constant or not reacting species, whose
  amount roadrunner keeps) is a state with the dilution alone, origin `dilution`.
- **An event which changes a size** keeps the amount of every species in
  concentration of the compartment (without a rule of its own): it is rescaled,
  `S = S * V / V_new`, with the size after the assignments of the event, of a
  compartment the event assigns or of a compartment whose assignment rule depends on
  a variable the event assigns; an event which assigns the concentration of such a
  species assigns `S = S_new * V / V_new` (`EventAssignment` keeps the value of SBML
  and this conversion apart, they are evaluated at different times).
- **Conversion factors**: the conversion factor of a species, else of the model,
  multiplies the reaction terms of the species.
- **Stoichiometry** is a number, or the id of the species reference if a rule, an
  initial assignment or an event sets it; a species reference with an id is a
  quantity of the system.
- **Local parameters** are renamed to `<reaction id>_<id>`, made unique, and are
  constant parameters of the system.
- **`rateOf(x)`** is replaced by the right hand side of `x` for a state, by `0` for a
  constant, by the rate of the size of a compartment with an assignment rule; the
  rate of another assigned variable is unsupported.
- **Assignments** (the rates of the sizes, assignment rules and reaction rates) are ordered
  by their dependencies, ties in the order of the document; a cycle is an error.
- **Initial values** at t=0 (`OdeSystem.initial`) are every state, every constant set
  by an initial assignment or converted between amount and concentration, every
  assigned variable and the reaction rates these depend on, in one order of their
  dependencies, so that an initial assignment of a rule and a rule of an initial
  assignment are both right.
- **Events** keep their trigger as written and get its continuous root function
  (`events.trigger_root`, of the trigger with its function definitions expanded);
  an event without an id is `event<index>`, one without a trigger never fires.
- **Defaults** as roadrunner holds them: a compartment without a size has the size
  1 (its `Quantity.value`), a stoichiometry which is not set is 1, a reaction
  without a kinetic law has the rate 0; a rule, initial assignment, event
  assignment or function definition without math is ignored (L3V2).
- **Constants in `OdeSystem.initial`** are only those set by an initial assignment
  and those whose value is a conversion with another quantity (a species in
  concentration with an initial amount, divided by its compartment, and the
  reverse); every other constant has its value in `Quantity.value`, so that a
  value passed for it is kept.
- **Unsupported** constructs are collected as `(construct, element id)`: algebraic
  rules, `delay`, fast reactions, distrib functions, an event assignment to a
  constant, a trigger without a continuous root function, the rate of an assigned
  variable, the rate of an assigned size which cannot be differentiated. A comp model is flattened first, an L1 or L2 model read as L3V2.

Every math of the system is a deep copy owned by the system, so the document can be
freed; a sum is written with the signs of its terms (`astutil.signed_sum`). The
analysis itself is `sbmlode.analysis`. The dataclasses which hold
math compare by identity (`eq=False`), a libsbml math has no value equality.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import libsbml

if TYPE_CHECKING:
    from sbmlode.documents import TypesetSystem

__all__ = [
    "Assignment",
    "Event",
    "EventAssignment",
    "FunctionDefinition",
    "ModelInfo",
    "Ode",
    "OdeSystem",
    "Participant",
    "Quantity",
    "Reaction",
    "SizeRate",
    "Symbol",
]


Kind = Literal[
    "compartment",
    "species",
    "parameter",
    "reaction",
    "species_reference",
    "function",
    "event",
    "rate",
]
Role = Literal["constant", "state", "assigned"]
Origin = Literal[
    "assignment_rule",
    "initial_assignment",
    "reaction",
    "initial_value",
    "size_rate",
]


@dataclass(frozen=True)
class Symbol:
    """An element with an id: its name, unit, SBO term and kind.

    Attributes:
        sid: the id in the system, which the analysis makes up or changes for a
            renamed local parameter (`<reaction>_<id>`) and the rate of the size of a
            compartment (`d<compartment>_dt`)
        name: the name
        unit: the unit
        sbo: the SBO term
        kind: the kind of the element
        element: the element of the model the symbol stands for, if it is not the
            element of `sid`: `(reaction, id)` for a local parameter, `(compartment,)`
            for the rate of its size; empty otherwise, see `source`
    """

    sid: str
    name: str | None
    unit: str | None
    sbo: str | None
    kind: Kind
    element: tuple[str, ...] = ()

    @property
    def source(self) -> tuple[str, ...]:
        """The element of the model the symbol stands for.

        `(sid,)` for an element of the model, `(reaction, id)` for a local parameter
        of a reaction, `(compartment,)` for the rate of its size; an application which
        links a symbol to its element resolves this, e.g. SBML4Humans.
        """
        return self.element or (self.sid,)


@dataclass(frozen=True)
class Quantity:
    """A compartment, species, parameter or species reference.

    Attributes:
        symbol: the symbol
        value: the value of the document in the representation of the quantity, the
            default 1 of a compartment without a size; `None` if it is not set or
            needs a conversion, which `OdeSystem.initial` then holds
        constant: the constant flag of SBML
        role: the role in the system
        compartment: the compartment of a species, `None` for a
            species in amount without a compartment, which L3 requires and
            libsbml reads
        amount: a species in amount (`hasOnlySubstanceUnits`)
        boundary: the boundary condition of a species
        conversion_factor: the conversion factor of a species, else of the model
    """

    symbol: Symbol
    value: float | None
    constant: bool
    role: Role
    compartment: str | None = None
    amount: bool | None = None
    boundary: bool | None = None
    conversion_factor: str | None = None


@dataclass(frozen=True, eq=False)
class SizeRate:
    """The rate of change `dV/dt` of the size of a compartment, see the module.

    Attributes:
        symbol: the symbol of the rate, of kind `rate`, its source the compartment
        compartment: the id of the compartment
        math: the rate, the right hand side of the rate rule of the compartment or
            the derivative of its assignment rule; `None` if the rule cannot be
            differentiated
    """

    symbol: Symbol
    compartment: str
    math: libsbml.ASTNode | None


@dataclass(frozen=True, eq=False)
class FunctionDefinition:
    """A function definition, called by the math."""

    symbol: Symbol
    arguments: tuple[str, ...]
    body: libsbml.ASTNode


@dataclass(frozen=True, eq=False)
class Assignment:
    """The value of a variable from its math: a rule, a rate, an initial value."""

    variable: str
    math: libsbml.ASTNode
    origin: Origin


@dataclass(frozen=True)
class Participant:
    """A reactant or product with its stoichiometry, a number or a species reference."""

    species: str
    stoichiometry: float | str


@dataclass(frozen=True, eq=False)
class Reaction:
    """A reaction; the rate refers to the renamed local parameters."""

    symbol: Symbol
    reactants: tuple[Participant, ...]
    products: tuple[Participant, ...]
    modifiers: tuple[str, ...]
    reversible: bool
    rate: libsbml.ASTNode
    local_parameters: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Ode:
    """The ordinary differential equation of a state.

    Attributes:
        variable: the state
        rhs: the complete right hand side
        origin: the reactions, the rate rule of the state or the dilution alone of
            a species in concentration in a compartment whose size changes
        reaction_terms: the sum of stoichiometry, conversion factor and rate of each
            reaction of a species, before the division by the volume
        volume: the compartment the reaction terms are divided by
        size_rate: the id of the rate of the size of the compartment of a species in
            concentration, whose dilution `-(S/V) dV/dt` the right hand side ends with
    """

    variable: str
    rhs: libsbml.ASTNode
    origin: Literal["reactions", "rate_rule", "dilution"]
    reaction_terms: libsbml.ASTNode | None = None
    volume: str | None = None
    size_rate: str | None = None


@dataclass(frozen=True, eq=False)
class EventAssignment:
    """The new value of a variable when an event is executed.

    The new value is `value * scale / new(divisor)`, each part optional:

    - `value` is `math`, evaluated as SBML says: at the trigger time if the event
      uses the values from the trigger time, else at the execution; `1` if `math` is
      `None`;
    - `scale` is evaluated at the execution of the event, with the values before any
      assignment of the event is applied, i.e. after the events executed before it;
    - `new(divisor)` is the size of the compartment `divisor` after the assignments
      of the event without divisor are applied: the size the event assigns, or the
      size of an assignment rule evaluated with the values after the event.

    The assignments without divisor come first in `Event.assignments`.

    Attributes:
        variable: the variable
        math: the value SBML assigns, `None` for a species whose amount stays
        scale: `V` for a rescaled concentration, `S * V` for a concentration whose
            amount stays
        divisor: the compartment whose new size divides a rescaled concentration,
            `S = S_value * V / V_new`
    """

    variable: str
    math: libsbml.ASTNode | None
    scale: libsbml.ASTNode | None = None
    divisor: str | None = None


@dataclass(frozen=True, eq=False)
class Event:
    """An event; `root` is the root function of the trigger, `None` if it has none."""

    symbol: Symbol
    trigger: libsbml.ASTNode
    root: libsbml.ASTNode | None
    initial_value: bool
    persistent: bool
    delay: libsbml.ASTNode | None
    priority: libsbml.ASTNode | None
    use_values_from_trigger_time: bool
    assignments: tuple[EventAssignment, ...]


@dataclass(frozen=True)
class ModelInfo:
    """The model: id, name, SBML level and version, notes as plain text, units."""

    sid: str | None
    name: str | None
    level: int
    version: int
    notes: str | None
    units: Mapping[str, str | None]
    source: str | None


@dataclass(frozen=True, eq=False)
class OdeSystem:
    """The ODE system of an SBML model, see the module for the semantics."""

    info: ModelInfo
    compartments: tuple[Quantity, ...]
    species: tuple[Quantity, ...]
    parameters: tuple[Quantity, ...]
    species_references: tuple[Quantity, ...]
    functions: tuple[FunctionDefinition, ...]
    size_rates: tuple[SizeRate, ...]
    assignments: tuple[Assignment, ...]
    initial: tuple[Assignment, ...]
    reactions: tuple[Reaction, ...]
    odes: tuple[Ode, ...]
    events: tuple[Event, ...]
    unsupported: tuple[tuple[str, str], ...]

    @classmethod
    def from_sbml(cls, source: Path | str | libsbml.SBMLDocument) -> OdeSystem:
        """Analyse an SBML model.

        Args:
            source: path, SBML string or document, which is not changed

        Returns:
            the ODE system

        Raises:
            ValueError: if the source cannot be read or the model is not well
                defined, e.g. its assignments depend on each other in a cycle
        """
        # the analysis builds the dataclasses of this module
        from sbmlode.analysis import analyse

        return analyse(source)

    def render(self, fmt: str, **options: object) -> str:
        """Render the system in a format, see `formats.render`.

        Args:
            fmt: the name of the format, a key of `formats.FORMATS`
            **options: the options of the format

        Returns:
            the code or the document
        """
        # the formats render the dataclasses of this module
        from sbmlode import formats

        return formats.render(self, fmt, **options)

    def typeset(
        self,
        dialect: Literal["latex", "typst"] = "latex",
        symbols: Literal["id", "name"] = "id",
        wrap: Callable[[Symbol, str], str] | None = None,
    ) -> TypesetSystem:
        """The system typeset for an application, as typed data instead of a document.

        The sections are those of the documents (`latex`, `typst`, `markdown`), which
        are rendered from the same data: the ODEs, the reaction rates, the
        assignments, the function definitions, the initial values, the events and
        the unsupported constructs, every text escaped and every math typeset in
        the dialect.

        Args:
            dialect: the dialect of the math and the text, `latex` (the math of
                KaTeX and MathJax as well) or `typst`
            symbols: `"id"` or `"name"`, what the math symbols are made of
            wrap: the function every math symbol is transformed with, from its
                `Symbol` and its typeset symbol, e.g. into a link to the element
                `Symbol.source` names; called for the symbols the documents make up
                as well, the rate `v` of a reaction and the rate `dV/dt` of a size

        Returns:
            the typeset system

        Raises:
            ValueError: for a dialect which is neither `latex` nor `typst` or
                symbols which are neither `id` nor `name`
        """
        if dialect not in ("latex", "typst"):
            raise ValueError(
                f"The dialect of typeset is 'latex' or 'typst', not {dialect!r}."
            )
        from sbmlode.documents import DocumentContext

        return DocumentContext(self, dialect, symbols, wrap).typeset()

    def write(
        self, path: Path | str, fmt: str | None = None, **options: object
    ) -> Path:
        """Write the system to a file in the format of its suffix, see `formats.write`.

        Args:
            path: the path of the file
            fmt: the name of the format, by default the format of the suffix
            **options: the options of the format

        Returns:
            the path
        """
        from sbmlode import formats

        return formats.write(self, path, fmt, **options)

    def render_template(
        self, template: Path | str, fmt: str = "python", **options: object
    ) -> str:
        """Render the system with a template of its own, see `formats.render_template`.

        Args:
            template: the path of the jinja2 template
            fmt: the name of the format whose context the template gets
            **options: the options of the format

        Returns:
            the rendered template
        """
        from sbmlode import formats

        return formats.render_template(self, template, fmt, **options)

    @property
    def states(self) -> tuple[str, ...]:
        """The ids of the states, in the order of the odes."""
        return tuple(ode.variable for ode in self.odes)

    @property
    def constants(self) -> tuple[str, ...]:
        """The ids of the constants, in the order of `quantities`."""
        return tuple(q.symbol.sid for q in self.quantities if q.role == "constant")

    @property
    def assigned(self) -> tuple[str, ...]:
        """The assigned variables and reactions, in the order of `assignments`."""
        return tuple(a.variable for a in self.assignments)

    def quantity(self, sid: str) -> Quantity:
        """The quantity of an id, `KeyError` if it is none."""
        return self._quantities[sid]

    def symbol(self, sid: str) -> Symbol:
        """The symbol of an id, `KeyError` if it is none."""
        return self._symbols[sid]

    @property
    def quantities(self) -> tuple[Quantity, ...]:
        """The compartments, species, parameters and species references."""
        return (
            *self.compartments,
            *self.species,
            *self.parameters,
            *self.species_references,
        )

    @cached_property
    def _quantities(self) -> dict[str, Quantity]:
        """The quantities by their id."""
        return {q.symbol.sid: q for q in self.quantities}

    @cached_property
    def _symbols(self) -> dict[str, Symbol]:
        """The symbols by their id."""
        symbols = {sid: q.symbol for sid, q in self._quantities.items()}
        for element in (
            *self.reactions,
            *self.functions,
            *self.events,
            *self.size_rates,
        ):
            symbols[element.symbol.sid] = element.symbol
        return symbols
