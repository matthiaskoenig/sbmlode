# Native quantities: design

Status: approved in conversation on 2026-10-06, for sbmlode 0.2.0.

Part of [sbml4humans#114](https://github.com/matthiaskoenig/sbml4humans/issues/114): the differential equations are written in the native quantity of every SBML state variable, amount or concentration, and a change of a compartment size is handled as the SBML specification describes it.

## Goal

A species is a state in the quantity the model declares: in amount if `hasOnlySubstanceUnits`, else in concentration, also in a compartment whose size changes. The amount `n_S`, which sbmlode 0.1.0 introduces for a species in concentration in a variable compartment (`OdeSystem.amounts`, the assignment `S = n_S / V` of origin `concentration`), is removed. The ODE system reads like the model: the reader finds the species they know, not a variable sbmlode made up.

## Decisions

| Question | Decision |
|---|---|
| Form of a concentration ODE | SBML L3V2 Release 2, section 3.4.6 (rateOf): `d[x]/dt = (1/V) dx/dt - ([x]/V) dV/dt`, with `dx/dt` the stoichiometric sum of the reaction rates with their conversion factors. |
| `dV/dt` of a constant size, a size changed by events only | 0, the term is left out, the ODE is the one of 0.1.0. |
| `dV/dt` of a size with a rate rule | The symbol of the rate of `V`, which stands for the right hand side of the ODE of `V`; it is not inlined. |
| `dV/dt` of a size with an assignment rule `V = f` | The assignment `dV/dt = df/dt` by the chain rule, origin `size_rate`. This goes beyond the specification, which excludes the case from rateOf. |
| Differentiation | A symbolic differentiator of sbmlode (`astutil.derivative`), not `libsbml.ASTNode.derivative`, which returns `None` for a power with a symbolic exponent, `piecewise` and calls of function definitions, and does not simplify. |
| An event which changes a size | The amount of every species in concentration in the compartment is conserved: `[x] := [x] * V_before / V_after`. |
| Compatibility | A breaking change of the system: `OdeSystem.amounts`, `Quantity.amount_of`, `Ode.amount_of` and the origin `concentration` are removed, the origin `size_rate` is added. sbmlode is below 1.0, the version is 0.2.0. |

## The analysis

### States

Every species which is a state in 0.1.0 is a state, the species itself, never its amount. In addition, a species in concentration whose amount roadrunner keeps when the size of its compartment changes (a boundary species, a constant species, which `_read_species` today holds as a constant amount) is a state with the ODE of its dilution alone, `d[x]/dt = -([x]/V) dV/dt`, if the size changes continuously (a rate rule or an assignment rule); with a size changed by events only it stays a constant between the events and is rescaled by the event. The SBML test suite decides the cases; a case which passes with 0.1.0 passes with 0.2.0.

### The rate of a size

`_size_rate(cid)` returns the math of `dV/dt`:

- constant, or changed by events only: `None` (no term);
- a rate rule: the name of the rate of `V`, a symbol of the system (`Symbol` of kind `rate`, typeset as `\frac{\mathrm{d}V}{\mathrm{d}t}`, a name `dV_dt` in code, made unique against the ids of the model) whose value is the right hand side of the ODE of `V`; the formats evaluate the right hand sides of the size rates before the ODEs of the species;
- an assignment rule `V = f`: the symbol of the rate of `V` as above, assigned `dV/dt = df/dt` (origin `size_rate`), ordered with the assignments by its dependencies.

`df/dt = ∂f/∂t + Σ_j ∂f/∂y_j · dy_j/dt` over the ids `y_j` of `f`, with `t` the csymbol time: `dy_j/dt` is the right hand side of a state, the rate of another assigned variable (recursively, by its own rule; a cycle is an error like a cycle of assignments), `0` for a constant. Function definitions are expanded before the differentiation. A math which cannot be differentiated (`delay`, `rateOf`, a csymbol of a package, a distrib function) adds the unsupported construct `rate of an assigned size`; the species of the compartment get no ODE.

`rateOf(x)` of a species in concentration is the right hand side of `x`; `rateOf(V)` of a size with an assignment rule is its size rate. The unsupported construct `rateOf of an assigned variable` remains for every other assigned variable.

### The differentiator

`astutil.derivative(ast, variable) -> ASTNode | None` on the libsbml AST: `+`, `-` (unary and binary), `*` (product rule over n arguments), `/` (quotient rule), `^` and `pow` (the general rule `d(u^v) = u^v (v' ln u + v u'/u)`, the power rule if `v` is a number), `exp`, `ln`, `log` (base 10 and with a base), `root`, `sqrt`, `abs` (`sign(u) u'`, written as `piecewise`), `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `arcsin`, `arccos`, `arctan`, the hyperbolic functions, `piecewise` (each piece differentiated, the conditions kept), numbers and constants (`0`), names (`1` for the variable, else `0`), time (`1` if the variable is time). Every other node returns `None`. The result is simplified with the helpers of `astutil` (`drop_zero_terms`, `product`): no `0 *`, `1 *`, `+ 0`, `^1`.

### Events

`_event_assignments` rescales every species in concentration of a compartment whose size the event changes, `[x] := [x] * V_before / V_after`, with the mechanism of `EventAssignment` (`scale`, `divisor_position`) which 0.1.0 uses for a species in concentration with a rate rule. The size changes if the event assigns it, or if its size has an assignment rule which depends (transitively) on a quantity the event assigns; `V_after` is then the rule evaluated with the values after the assignments of the event. An event which assigns a species in concentration assigns it as written, no conversion to an amount.

### Initial values

Unchanged in meaning: a species in concentration with an initial amount starts at `n0 / V(0)`, a species in amount with an initial concentration at `c0 * V(0)` (`OdeSystem.initial`, the conversion of 0.1.0).

## The formats

The templates lose the amounts (`amount_of`, the comment `amount of ...`) and gain the size rates: in code an assignment `dV_dt = ...` before the rates of change of the species which use it, in the documents a row `\frac{dV}{dt} = ...` in the assignments. The typeset target (`OdeSystem.typeset`) carries the size rate as an equation of origin `size_rate`; its symbol is the rate of `V` with the source of `V`, so a consumer links it to the compartment.

## Verification

- Every semantic case of the SBML test suite 3.4.0 which passes with 0.1.0 passes, in python, julia and R; the variable compartment cases (among them 01506, 01779 and the cases with a compartment of an assignment rule) are named in a test of their own.
- `astutil.derivative`: a unit test per node type, a test which compares the derivative with a central finite difference on random points for a set of expressions, and a test that the unsupported nodes return `None`.
- The goldens are regenerated, the documentation (`docs/formats.md`, the docstring of `sbmlode.system`) describes the native quantities.
