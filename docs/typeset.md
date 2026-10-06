# Typed target

The documents of sbmlode (typst, LaTeX, markdown) lay out the equations of a model as a page. An application which lays out the equations itself, in its own user interface, needs them as data instead: `OdeSystem.typeset` returns the system typeset for a dialect as frozen dataclasses, the same data the documents are rendered from.

```python
from sbmlode import OdeSystem

system = OdeSystem.from_sbml("model.xml")
typeset = system.typeset("latex")

for ode in typeset.odes:
    print(ode.lhs, "=", " ".join(ode.lines))
```

```text
\frac{\mathrm{d} \,\mathrm{PX}}{\mathrm{d} t} = v_{\mathrm{Reaction4}} - v_{\mathrm{Reaction7}}
...
```

## The sections

`typeset(dialect="latex", symbols="id", wrap=None)` returns a `TypesetSystem` with the sections of a document:

| section | content |
| --- | --- |
| `odes` | the ODE of every state, `TypesetEquation` with `lhs` the derivative `dX/dt` and `lines` the right hand side written with the rates of the reactions and divided by the volume of a species in concentration, or the rate rule; `origin` is `reactions` or `rate_rule` |
| `reactions` | the rate of every reaction, `TypesetReaction` with `symbol` (`v` with the id as subscript), `lines` (the kinetic law), the reaction equation, the modifiers and the local parameters |
| `assignments` | the assignment rules in the order of their dependencies and the concentrations of the species held as amount, `TypesetEquation` with `origin` `assignment_rule` or `concentration` |
| `functions` | the function definitions, `TypesetFunction` with `lhs` `f(x, y)` and `rhs` |
| `initial` | the initial assignments and the initial values which are a conversion between amount and concentration, `TypesetEquation` with `origin` `initial_assignment` or `initial_value` |
| `events` | the events, `TypesetEvent` with the trigger, delay, priority, flags and the assignments with their effective value |
| `unsupported` | the constructs the system does not hold, `TypesetUnsupported` with the construct (`algebraic rule`, `delay`, `fast reaction`) and the id of its element |
| `model`, `units`, `compartments`, `species`, `parameters`, `amounts` | the title and metadata and the tables of a document |

The math is LaTeX (which KaTeX and MathJax typeset) or typst, a long sum is split into `lines` of at most four terms, a line after the first begins with its sign. Text, such as a name, is escaped for the dialect. Every equation carries the `Symbol` of its left hand side as `variable`, so that an application knows which element an equation belongs to.

## Linking the symbols

`wrap(symbol, typeset)` is called with the `Symbol` and the typeset symbol of every element the math writes, and its result is written instead. It is called for the symbols the documents make up as well: the rate `v_{J0}` is called with the symbol of the reaction `J0`, the amount `n_{S}` of a species held as amount with the symbol of its amount, the derivative `dS/dt` wraps the symbol of `S` inside it.

`Symbol.source` names the element of the model a symbol stands for: `(sid,)` for an element of the model, `(reaction, id)` for a local parameter, which the analysis renames to `<reaction>_<id>`, and `(species,)` for the amount of a species. An application resolves it to its own reference of the element. [SBML4Humans](https://sbml4humans.de) wraps every symbol as a link of KaTeX, so that a click on a symbol of an equation opens its element:

```python
def wrap(symbol: Symbol, typeset: str) -> str:
    pk = resolve(symbol.source)  # the element in the report, None if there is none
    return typeset if pk is None else rf"\htmlData{{pk={pk}}}{{{typeset}}}"


typeset = system.typeset("latex", "id", wrap)
```

The symbols are typeset from the ids (`symbols="id"`), which are SIds and safe inside such a wrapper; `symbols="name"` typesets the name of an element where it is a valid symbol.
