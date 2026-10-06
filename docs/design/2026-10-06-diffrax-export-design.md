# diffrax export: design

Status: approved in conversation on 2026-10-06, for sbmlode 0.3.0.

Closes [#8](https://github.com/matthiaskoenig/sbmlode/issues/8): the ODE system of an SBML model is written as python code which simulates it with [diffrax](https://docs.kidger.site/diffrax/) on [JAX](https://docs.jax.dev).

## Goal

A new code format `diffrax` writes a self contained python file whose `simulate` is a JAX function: it is compiled with `jit`, batched with `vmap` over parameters and initial values, and differentiated with `grad` and `jacfwd` with respect to them, events included. This is what the numpy export cannot do: parameter estimation with gradients, sensitivities, and the simulation of many parameter sets on an accelerator. The correctness contract is the one of the other code formats: the file reproduces roadrunner over the SBML semantic test suite, every case the python export passes, the diffrax export passes.

## Decisions

| Question | Decision |
|---|---|
| Purpose | A JAX native model: `jit`, `vmap` and `grad` of `simulate` for every model. |
| Events | All SBML event semantics (trigger initial values, delays, priorities, persistence, values from the trigger time, cascades) inside JAX, so that the transformations work for a model with events as well. |
| Shape | A new format `diffrax` with its own template and math dialect; the file carries its whole simulator as the python, julia and R files do. No runtime module in sbmlode, which keeps its dependencies `python-libsbml` and `jinja2`, and no `backend` option of the python format, whose simulator shares nothing with this one. |
| Interface | `simulate(ts, p=None, x0=None, ...)` with the output times `ts` and flat arrays `p` and `x0` in the order of `PIDS` and `XIDS`, returning a `NamedTuple` of arrays; `to_frame` converts it to pandas outside of `jit`. |
| Precision | The file enables float64 (`jax_enable_x64`) on import: without it the tolerances which reproduce roadrunner cannot be met. |
| Python | sbmlode requires python >= 3.12, the version jax requires; python 3.11 is dropped for the whole package. |

## Python 3.12

jax 0.11 requires python >= 3.12. sbmlode drops python 3.11: `requires-python = ">=3.12"` and the classifiers in `pyproject.toml`, `envlist = ty, lowest, py3.{12,13,14,15}` and the `lowest` environment on python 3.12 in `tox.ini`, the matrix of `ci-cd.yml`, `docs/development.md` and `CLAUDE.md`. ruff takes its target from `requires-python`; what pyupgrade flags for the new target is fixed in the same change. The design records in `docs/design/` describe their time and are not changed.

## Format and printer

`FORMATS["diffrax"]`: kind `code`, template `diffrax.py.jinja`, printer `jax`, options `{"simulator": True}`, 0-indexed, no suffixes: `.py` is the suffix of the format `python`, the diffrax file is written with `system.write("model.py", fmt="diffrax")` or `system.render("diffrax")`. `simulator=False` writes the pure JAX functions only (initial values, right hand side, `f_y`, the event functions), to be composed into a `diffeqsolve` of one's own.

`symbols.code_names` knows the language `jax`: the reserved names of python plus the names the file uses (`jax`, `jnp`, `lax`, `diffrax`, `eqx`, `optx`, `t`, `x`, `p` and the names of its functions).

`JaxPrinter(PythonPrinter)` in `printers/jax.py` overrides what JAX cannot trace or what differs in `jax.numpy`:

| Construct | python | jax |
|---|---|---|
| functions, constants | `np.exp`, `np.inf`, ... | `jnp.exp`, `jnp.inf`, ... |
| `max`, `min` | `max(a, b, c)` | `jnp.maximum(a, jnp.maximum(b, c))` |
| power | `a ** b` | `jnp.power(a, b)`, no complex result for literal operands |
| `piecewise` | `x if c else y` (lazy) | `jnp.where(c, x, jnp.where(c2, y, z))`, `jnp.nan` without an otherwise |
| `and`, `or`, `xor`, `not` | `and`, `or`, `bool(a) ^ bool(b)`, `not` | `jnp.logical_and`, `jnp.logical_or`, `jnp.logical_xor`, `jnp.logical_not` (not `&` and `~`: `~True` is `-2`) |
| condition as a number | `float(c)` | `jnp.where(c, 1.0, 0.0)` |
| `factorial` | `math.gamma(x + 1)` | `jax.scipy.special.gamma(x + 1)` |
| `rem`, `quotient`, `log` | `np.fmod`, `np.trunc(a / b)`, `np.log10` | `jnp.fmod`, `jnp.trunc(a / b)`, `jnp.log10` |

The relations of the base printer are binary, a relation of more than two operands is a conjunction of its pairs (`logic_and`), so python's chained comparison never occurs.

`jnp.where` evaluates every branch. The value is exact, the branch which does not apply is discarded even if it is `inf` or `NaN`; its gradient is not: a branch which does not apply and is singular at the point, `piecewise(0, x == 0, 1/x)` at `x = 0`, makes the gradient `NaN`. This is the known behavior of `where` under differentiation, which no general rewrite avoids; the documentation states it.

## The generated file

The layout follows the python format, a reader of one reads the other:

1. A docstring with the model, its source, the version of sbmlode, the units, and the use of `jit`, `vmap` and `grad`.
2. The imports: `jax`, `jax.numpy as jnp`, `diffrax`, `equinox as eqx`, and with events `jax.lax as lax` and `optimistix as optx`. `jax.config.update("jax_enable_x64", True)` follows `import jax`, with a comment: it switches float64 on for the process which imports the file.
3. The tables `XIDS`, `PIDS`, `YIDS`, `NAMES`, `UNITS` and `P0 = jnp.array([...])`.
4. The model as pure JAX functions: the function definitions, `initial_values(p=None) -> (x0, p)`, `f_dxdt(t, x, p)` in the order `(t, y, args)` of a diffrax vector field, so that `diffrax.ODETerm(f_dxdt)` takes it as it is, `f_y(t, x, p)`, and the event functions. An array is changed with `x.at[i].set(v)`, never in place.
5. With `simulator=True`, the simulator:

```python
class Simulation(NamedTuple):
    t: jax.Array  # (n,), the output times
    x: jax.Array  # (n, len(XIDS)), the states
    y: jax.Array  # (n, len(YIDS)), the assigned values and reaction rates
    p: jax.Array  # (n, len(PIDS)), the constants, which events change

@eqx.filter_jit
def simulate(ts, p=None, x0=None, *, rtol=1e-8, atol=1e-10, solver=None,
             max_step=None, max_steps=100_000, ...) -> Simulation: ...

def to_frame(simulation: Simulation) -> pd.DataFrame: ...

if __name__ == "__main__":
    print(to_frame(simulate(jnp.linspace(0.0, 10.0, 101))).head())
```

The semantics are the ones of the python `simulate`. The integration starts at t = 0, where the initial values and the events at t = 0 apply; `ts` are the output times, a non-decreasing array with `ts[0] >= 0` whose length is the static shape of the result. `p` defaults to `P0`, `initial_values(p)` computes the constants set by initial assignments, `x0`, if given, replaces the initial states. `y` is `f_y` mapped over the saved times, states and constants. `to_frame` writes the columns of the python `DataFrame`: `time`, the states, the assigned values and the constants which events change; it imports pandas itself.

The solver is `diffrax.Kvaerno5()` (implicit, L-stable, order 5, SBML models are often stiff) with `diffrax.PIDController(rtol, atol, dtmax=max_step)`; `rtol=1e-8` and `atol=1e-10` are the defaults of the python export, `max_step` defaults to `ts[-1] / (len(ts) - 1)`, the spacing of the output times of the python export (`t_end / (points - 1)`) for `ts = linspace(0, t_end, points)`, and to no limit for a single output time, `max_steps` limits the steps of every `diffeqsolve`, which raises beyond it (`throw=True`). `solver` takes any diffrax solver, `diffrax.Tsit5()` for a model which is not stiff.

`eqx.filter_jit` traces the arrays `ts`, `p` and `x0` and holds every other argument static: a new tolerance or solver compiles anew, new parameters do not. `vmap` and `grad` over `p` and `x0` are applied from outside.

A model without events is integrated by a single `diffeqsolve` with `SaveAt(ts=ts)`; the file has no event engine then, it stays short and compiles fast.

## The event engine

The engine is the algorithm of the python and julia exports (`execute_events`, `first_change`), so that the three formats agree with roadrunner in the same way; the lists and loops of python become arrays of fixed shape and traced loops.

### Event functions

Written with `simulator=True` and `False`, pure JAX:

- `event_triggers(t, x, p)`: the continuous root function of every trigger (`events.trigger_root`), positive where the trigger holds;
- `event_conditions(t, x, p)`: the exact truth value of every trigger;
- per event, readable as in python: `event_delay_<id>`, `event_priority_<id>`, `event_values_<id>` and `event_assign_<id>(t, x, p, values) -> (x, p)`, with the size conversions of `scale` and `divisor`;
- the tables of fixed shape the engine indexes: `INITIAL_VALUE`, `PERSISTENT` and `USE_TRIGGER_VALUES` (boolean arrays), `event_delays(t, x, p)` and `event_priorities(t, x, p)` (shape `(n_events,)`, 0 for an event without delay or priority, as in python), `event_values(t, x, p)` (shape `(n_events, A)`, `A` the most assignments of an event, padded with 0) and `event_assign(k, t, x, p, values)`, a `lax.switch` over the assignments of the events.

### State

A `NamedTuple` carried through the loops: the time `t`, the states `x`, the constants `p`, `holds` (whether each trigger held), the queue of the scheduled executions with the capacity `max_pending` (the index of the event, -1 for a free slot; the time, `inf` for a free slot; the values from the trigger time and whether they are used; an insertion counter which reproduces the order of the python list for ties), the buffers of the output and the counters of the segments and executions.

### Algorithm

1. At t = 0: `initial_values`, `holds = INITIAL_VALUE`, `execute_events`. A trigger which holds at t = 0 and has the initial value false fires at t = 0.
2. The loop of segments, while `t < ts[-1]`:
   - `t_stop = min(ts[-1], the earliest scheduled time)`: a delayed execution ends a segment exactly, without a root finding.
   - `diffeqsolve` from `t` to `t_stop` with a `diffrax.Event` of one condition per trigger, `where(holds_k, -root_k, root_k)`, `direction=True`, and the root finder `optx.Newton`: only a change away from `holds` stops the integration, so that a trigger which starts at its root, `time >= 1` at t = 1 after it fired, is not found again.
   - At a stop at a root, the time is refined to the first time at which the exact `event_conditions` differ from `holds`, as `first_change` of julia does it: bisection on the linear extrapolation of the states within 1e-13 (relative) of the root. `time >= 1` changes at 1, `time > 1` right after it.
   - The output times inside `(t, t_stop)` take the states saved by the segment (`SaveAt(ts=clip(ts, t, t_stop))`, masked), an output time at `t_stop` (to 1e-14 relative) the states after its events, as in python.
   - `execute_events` at `t_stop`.
3. `execute_events`, the cascade of python on arrays: a trigger which turned true schedules an execution at `t + delay` (with its values if it uses the values from the trigger time), a trigger of a non-persistent event which turned false drops its executions; of the executions due, the one of the highest priority is executed, ties in the order of the events, then of their scheduling; the triggers are evaluated anew until no execution is due.

### Differentiability

The event times carry their gradients. The time of a root is differentiated by diffrax through the root finder (implicit function theorem); the refinement is an offset of at most 1e-13 under `lax.stop_gradient`; a delayed time is `t + delay(t, x, p)`, which is traced. The discrete decisions (which event, which priority) are piecewise constant. Forward mode (`jvp`, `jacfwd`) passes any loop; reverse mode (`grad`) needs loops which JAX can reverse: `lax.scan` over a static bound with `lax.cond` skipping the iterations after the end, or the checkpointed while loop of equinox (`equinox.internal.while_loop`, which diffrax uses itself, an internal interface). The spike decides, for the loop of segments and the cascade.

### Limits and errors

The limits of the python export plus the two which fixed shapes need, each raising inside `jit` with the message of python (`eqx.error_if`, `diffeqsolve(throw=True)`): `max_steps` per segment (100000), `MAX_CASCADE` executions at one time (10000), and the static arguments `max_segments` and `max_pending`, whose defaults the spike sets. Under `vmap`, an error of one member of the batch raises for the batch.

## Spike

The first step of the work, its code is thrown away. One model with a trigger crossing which depends on a parameter, a delayed event and a non-persistent event, written by hand in the shape of the generated file:

1. `jax.grad` of a loss of `simulate` agrees with central finite differences (relative 1e-5), the shift of the event time included;
2. `jit` and `vmap` over 100 parameter sets;
3. `time >= 1` and `time > 1` stop where the python export stops;
4. the time of compilation and of a run, for both loop constructions.

The loop construction is chosen by correctness first, then by the time of compilation; the result, the loop construction and the defaults of `max_segments` and `max_pending` are recorded in this document.

## Dependencies

The package keeps `python-libsbml` and `jinja2`. A new extra `diffrax` lists what the generated file imports: `jax`, `diffrax`, `equinox`, `optimistix`, and `pandas` for `to_frame`; the `lowest` environment of tox verifies the lower bounds. The extra `simulate` stays numpy, pandas and scipy; `test` becomes `sbmlode[simulate,diffrax]` with what it has now. `uv lock` after the change.

## Testing

- `test_ode_printers.py`: the dialect `jax` against golden strings for every type of AST node, and its evaluation against roadrunner over the formulas which cover every construct of the math.
- `test_ode_diffrax.py`: `initial_values`, `f_dxdt`, `f_y` and `simulate` against roadrunner for the test models, with the tolerances of the python export (`rtol=1e-6`, `atol=1e-9`, relaxed to `1e-4` and `1e-6` with events); the contract of JAX: `jit`, `vmap` over `p` equal to a loop of single runs, `grad` and `jacfwd` equal to central finite differences for a model without events and for a model whose event time depends on the parameter; a file with `simulator=False` composed into a `diffeqsolve`; `max_pending`, `max_segments` and `MAX_CASCADE` raise.
- `test_ode_safety.py` runs over all formats and so covers the injection and SId tests of `diffrax`; `test_ode_docs.py` its output in the documentation.
- `test_ode_testsuite.py`: every case the python export passes, the diffrax export passes; a known failure is a strict xfail with its reason. The full sweep runs behind the `sbml_testsuite` marker, `scripts/ode_report.py --format diffrax` reports the pass rate. If the curated cases cost more than about 5 minutes of compilation on CI, the default run takes `CURATED_DIFFRAX`, a subset with every feature tag (events, delays, priorities and persistence included), and the others stay in the sweep; the times of the spike decide.

## Continuous integration

No job of its own: jax installs from wheels on linux, macos and windows, the tests of `diffrax` run in the matrix of `tests` (python 3.12 to 3.14, macos and windows with 3.14). On python 3.15, without libroadrunner, the comparisons with roadrunner skip as today, the tests of the JAX contract run.

## Documentation

- `docs/formats.md`: a section `diffrax` with the writing (`fmt="diffrax"`), the interface (`simulate(ts, p, x0)`, `Simulation`, `to_frame`), examples of `jit`, `vmap` and `grad`, float64 as a switch of the process, the gradient of `where`, the choice of the solver, `max_segments` and `max_pending`.
- The repressilator as diffrax file in `docs/images/ode/`, written by `scripts/docs_images`.
- The README, `docs/index.md` and `CLAUDE.md` name the format, the extra and python 3.12.
- The release notes of 0.3.0 are written with the release.

## Order of the work

1. Python 3.12: the package, tox, CI and documentation.
2. The spike; its result recorded here.
3. `JaxPrinter` with its tests.
4. The format, the template without events, `test_ode_diffrax.py` for the models without events.
5. The event engine in the template, its tests, the curated cases of the test suite.
6. The sweep over the test suite, `scripts/ode_report.py`, the known failures.
7. Documentation.
