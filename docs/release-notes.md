# Release notes

What changed in every version of sbmlode, the newest first. The notes of a version are the text of its [release on GitHub](https://github.com/matthiaskoenig/sbmlode/releases), where the source of that version is archived.

## 0.3.0

sbmlode 0.3.0 writes the ODE system of an SBML model as python code of [JAX](https://docs.jax.dev), which [diffrax](https://docs.kidger.site/diffrax/) simulates: `jax.jit` compiles the simulation, `jax.vmap` batches it over constants and initial states, and `jax.grad` and `jax.jacfwd` differentiate it, a model with events included. It passes every case of the SBML test suite which it supports and libroadrunner simulates. sbmlode requires python 3.12 or newer, as jax does.

### New features
- the format `diffrax` ([#8](https://github.com/matthiaskoenig/sbmlode/issues/8)), written with `system.write("model.py", fmt="diffrax")` (`.py` stays the suffix of `python`): `initial_values(p)`, `f_dxdt(t, x, p)` (a vector field of diffrax), `f_y(t, x, p)`, the functions of the events, and with `simulator=True` `simulate(ts, p, x0, ...)`, which returns a `Simulation` of the output times, states, assigned values and constants, and `to_frame`, its pandas `DataFrame`; the file switches JAX to float64 on import
- the events of SBML in JAX: delays, priorities, persistence, the initial values of the triggers, the values from the trigger time and cascades, on arrays of fixed shapes (`max_segments`, `max_pending`), in loops of the kind the `adjoint` of diffrax differentiates (`RecursiveCheckpointAdjoint` for `jax.grad`, `ForwardMode` for `jax.jacfwd`, `DirectAdjoint` for both); the derivative includes the shift of an event time with the constants
- `default_solver()`, `diffrax.Kvaerno5` with `Chord`, a chord method which ends at an increment below `kappa` ([patrick-kidger/diffrax#782](https://github.com/patrick-kidger/diffrax/issues/782)); the steps of a model without events end at the output times
- `JaxPrinter`, the dialect `jax` of the math, which JAX traces: `jnp.where` for `piecewise`, the logic of `jnp`, float literals and `jnp.power`
- the extra `diffrax` (jax, diffrax, equinox, optimistix, pandas), which the generated code runs with
- `scripts/ode_report.py --format diffrax`, the pass rate of the diffrax code over the SBML test suite: all 1481 cases which are supported and which libroadrunner simulates

### Documentation
- a section of the guide on the diffrax code, with `jax.jit`, `jax.vmap`, `jax.grad` and `jax.jacfwd` and a solve of one's own, the conventions of the formats in a compact table, the diffrax code of the repressilator and its row in the verification table
- the design of the diffrax export, `docs/design/2026-10-06-diffrax-export-design.md`

### Breaking changes
- python 3.11 is no longer supported, sbmlode requires python 3.12 or newer

## 0.2.0

sbmlode 0.2.0 writes the ODE system in the native quantities of the SBML state variables: a species is a state in amount or in concentration as the model declares it, also in a compartment whose size changes. This changes the system and its typed target, see the breaking changes.

### New features
- a species in concentration in a compartment whose size changes stays in concentration: its ODE is the one of SBML Level 3 Version 2, section 3.4.6, the reaction terms divided by the size minus the dilution `S/V dV/dt` (`Ode.size_rate`); a boundary, constant or not reacting species in such a compartment keeps its amount and is diluted alone (origin `dilution`) ([matthiaskoenig/sbml4humans#114](https://github.com/matthiaskoenig/sbml4humans/issues/114))
- the rate of the size of a compartment `dV/dt` is an assignment of its own (origin `size_rate`, `OdeSystem.size_rates`, a symbol of kind `rate` whose source is the compartment, typeset as the derivative of the compartment): the right hand side of its rate rule, or the derivative of its assignment rule by the chain rule; a size with an assignment rule of constants only is constant, a rule which cannot be differentiated is the unsupported construct `rate of an assigned size`
- `astutil.derivative(ast, variable)`, the symbolic derivative of a math, simplified
- an event which changes a size rescales every species in concentration of the compartment with the size after the event, also when it changes a parameter of the assignment rule of the size; the code writes the function `event_sizes_<id>` of these sizes
- `rateOf` of a species in concentration in a variable compartment and of a size with an assignment rule

### Documentation
- the logo of sbmlode in the README, the documentation, the release notes and as favicon

### Breaking changes
- `OdeSystem.amounts`, `Quantity.amount_of`, `Ode.amount_of`, the origin `concentration` and the section `amounts` of the typed target and of the documents are removed, a species is never replaced by its amount `n_S`
- `TypesetEventAssignment` has no field `species`, its `conversion` is `None` or `resized`
- the code context of an event assignment has `divisor_index` into the new list `sizes` of the event instead of `divisor_position`

## 0.1.0

We are pleased to release the first version of sbmlode, the ODE export of sbmlutils as a package of its own. sbmlode writes the system of ordinary differential equations of an SBML model as python, julia and R code which simulates it and as typst, LaTeX and markdown documents which describe it. It depends on python-libsbml and jinja2 only, so that a tool which only needs the equations of a model, such as [SBML4Humans](https://sbml4humans.de), does not install the model building of sbmlutils.

### New features
- the ODE export of sbmlutils 0.14.0 (`sbmlutils.converters.ode`, [matthiaskoenig/sbmlutils#492](https://github.com/matthiaskoenig/sbmlutils/pull/492)) as the package `sbmlode`: `OdeSystem.from_sbml(source)`, `render(fmt, **options)`, `write(path)` and `render_template(template)` in the formats python, julia, R, typst, LaTeX and markdown, with the same output apart from the line which names the version that wrote it
- `OdeSystem.typeset(dialect, symbols, wrap)` returns the typeset system as typed data instead of a document: the ODEs, the reaction rates, the assignment rules, the function definitions, the initial values and the events with their left hand sides and right hand sides in LaTeX or typst, which an application lays out itself; the documents of typst, LaTeX and markdown are rendered from it
- `wrap(symbol, typeset)` is called for every symbol the typeset math writes, including the rate `v` of a reaction and the amount `n` of a species, so that an application can turn a symbol into a link to its element; `Symbol.element` names the element of the model a symbol stands for, the reaction and the id of a local parameter which the analysis renames

### Packaging and development
- the package reads and flattens a document with libsbml alone (`sbmlode.io`) and writes the units of a model without pint (`sbmlode.units`)
- the tests download the semantic cases of the SBML test suite 3.4.0 once into the cache directory, or read the directory `SBMLODE_TESTSUITE` names; `SBMLODE_JULIA` and `SBMLODE_RSCRIPT` set the commands of julia and R
- continuous integration runs the suite on Linux, Windows and macOS, the generated R code with deSolve and the documents with tectonic and typst for every pull request, and the julia code for a release and on demand; the documentation is published at <https://matthiaskoenig.github.io/sbmlode>
