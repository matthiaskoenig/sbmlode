# API reference

The export of a model as its system of ordinary differential equations in python, julia, R, typst, LaTeX and markdown. The guide to it is the [Guide](formats.md), the typed data of an application is described in [Typed target](typeset.md).

::: sbmlode

## The ODE system

The parts of an `OdeSystem`, the result of the analysis.

::: sbmlode.system
    options:
      filters: ["!^_", "!^OdeSystem$"]

## The typeset system

The typed data `OdeSystem.typeset` returns.

::: sbmlode.documents
    options:
      members: ["TypesetSystem", "TypesetEquation", "TypesetReaction", "TypesetFunction", "TypesetEvent", "TypesetEventAssignment", "TypesetUnsupported", "TypesetModel", "TypesetUnit", "TypesetRow", "TypesetSpeciesRow", "TypesetAmount", "Wrap"]

## Reading and units

::: sbmlode.io

::: sbmlode.units
