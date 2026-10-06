"""Test the diffrax code of the ODE export: against roadrunner and as JAX code."""

import ast
import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import libsbml
import numpy as np
import pandas as pd
import pytest
from ode_helpers import (
    EVENT_MODELS,
    FORMULAS,
    TWO_EVENTS,
    assert_diffrax_as_roadrunner,
    assert_diffrax_trajectory_as_roadrunner,
    assert_table_as_roadrunner,
    diffrax_frame,
    diffrax_module,
    edit_sbml,
    model_sbml,
    sbml_with_rate,
)
from resources import (
    COMP_DEX_LIVER,
    COMP_SPT_LIVER,
    DEMO_SBML,
    GALACTOSE_SINGLECELL_SBML,
    MODELS_DIR,
    REPRESSILATOR_SBML,
    VARIABLE_COMPARTMENT,
    VDP_SBML,
)

import sbmlode
from sbmlode import FORMATS, OdeSystem
from sbmlode.formats import context
from sbmlode.symbols import RESERVED, code_names

jax = pytest.importorskip("jax")
diffrax = pytest.importorskip("diffrax")
jnp = jax.numpy

INTERPOLATION_LINEAR_SBML = MODELS_DIR / "interpolation" / "data1_linear.xml"

# function definitions, assignment rules, an initial assignment of a constant and a
# species in a compartment whose size has a rate rule
RULES = """
    function mm(S, km)
        S / (km + S)
    end
    function hill(S, k, n)
        S^n / (k^n + S^n)
    end
    compartment c = 2; c' = 0.1
    species A in c = 3; species B in c = 1
    vmax = 2; km = 0.5; scale = 1.5
    total := A + B
    ratio := A / total
    keff = 2 * vmax
    J1: A -> B; keff * mm(A, km) * c
    J2: B -> ; vmax * hill(B, 1, 2)
"""

# a decay whose rate and initial value are constants, for the derivatives
DECAY = """
    species S = 2; species P = 0
    k = 0.3; k2 = 0.1
    J1: S -> P; k * S
    J2: P -> ; k2 * P^2
"""


def _system(antimony: str) -> OdeSystem:
    """The ODE system of a model written in antimony."""
    return OdeSystem.from_sbml(model_sbml(antimony))


def _module(antimony: str, tmp_path: Path, **options: object) -> ModuleType:
    """The diffrax module of a model written in antimony."""
    return diffrax_module(_system(antimony), tmp_path / "model.py", **options)


# --- correctness against roadrunner ---------------------------------------------------


@pytest.mark.parametrize("formula", FORMULAS)
def test_diffrax_formula(formula: str, tmp_path: Path) -> None:
    """The generated JAX evaluates every construct of the math as roadrunner."""
    assert_diffrax_as_roadrunner(sbml_with_rate(formula), tmp_path)


@pytest.mark.parametrize(
    "sbml_path",
    [
        DEMO_SBML,
        REPRESSILATOR_SBML,
        VDP_SBML,
        COMP_DEX_LIVER,
        COMP_SPT_LIVER,
        INTERPOLATION_LINEAR_SBML,
        GALACTOSE_SINGLECELL_SBML,
    ],
    ids=lambda p: p.stem,
)
def test_diffrax_model(sbml_path: Path, tmp_path: Path) -> None:
    """The generated JAX of a model computes the values of roadrunner."""
    assert_diffrax_as_roadrunner(sbml_path, tmp_path)


def test_diffrax_rules_and_functions(tmp_path: Path) -> None:
    """Function definitions, rules, an initial assignment and a variable size."""
    module = assert_diffrax_as_roadrunner(model_sbml(RULES), tmp_path)
    assert module.XIDS == ["c", "A", "B"]
    x0, p = module.initial_values()
    # the initial assignment of the constant is set in p
    assert p[module.PIDS.index("keff")] == 4.0
    assert x0[module.XIDS.index("A")] == pytest.approx(3.0)


def test_diffrax_initial_values_take_constants(tmp_path: Path) -> None:
    """`initial_values` evaluates the initial values with the constants passed."""
    module = _module(RULES, tmp_path)
    p = module.P0.at[module.PIDS.index("vmax")].set(10.0)
    x0, p_new = module.initial_values(p)
    assert p_new[module.PIDS.index("keff")] == 20.0
    # the constants passed are not changed, an array of JAX is immutable
    assert p[module.PIDS.index("keff")] != 20.0
    assert np.array_equal(x0, module.initial_values()[0])


@pytest.mark.parametrize(
    "sbml_path",
    [REPRESSILATOR_SBML, VDP_SBML, COMP_DEX_LIVER, GALACTOSE_SINGLECELL_SBML],
    ids=lambda p: p.stem,
)
def test_diffrax_simulate_matches_roadrunner(sbml_path: Path, tmp_path: Path) -> None:
    """`simulate` integrates a model as roadrunner."""
    system = OdeSystem.from_sbml(sbml_path)
    module = diffrax_module(system, tmp_path / "model.py")
    df = diffrax_frame(module)
    assert list(df.columns) == ["time", *system.states, *system.assigned]
    assert_diffrax_trajectory_as_roadrunner(sbml_path, module)


def test_diffrax_simulate_rules(tmp_path: Path) -> None:
    """Rules, function definitions and a variable size integrate as roadrunner."""
    sbml = model_sbml(RULES)
    module = diffrax_module(OdeSystem.from_sbml(sbml), tmp_path / "model.py")
    assert_diffrax_trajectory_as_roadrunner(sbml, module)


def test_diffrax_model_without_states(tmp_path: Path) -> None:
    """A model of assignment rules only has no states and simulates."""
    sbml = model_sbml("""
        k = 2
        y := k * time
        z := y^2
    """)
    assert assert_diffrax_as_roadrunner(sbml, tmp_path).XIDS == []
    module = diffrax_module(OdeSystem.from_sbml(sbml), tmp_path / "s.py")
    simulation = module.simulate(jnp.linspace(0.0, 4.0, 5))
    assert simulation.x.shape == (5, 0)
    assert np.allclose(simulation.y[:, module.YIDS.index("z")], [0, 4, 16, 36, 64])
    assert_table_as_roadrunner(sbml, diffrax_frame(module, 4.0, 5), t_end=4.0, points=5)


def test_diffrax_simulate_from_x0_and_p(tmp_path: Path) -> None:
    """`simulate` starts from the initial states and the constants passed."""
    module = _module(DECAY, tmp_path)
    ts = jnp.linspace(0.0, 5.0, 6)
    p = module.P0.at[module.PIDS.index("k")].set(0.0)
    simulation = module.simulate(ts, p, jnp.array([4.0, 0.0]))
    # without the reaction S keeps the initial state passed
    assert np.allclose(simulation.x[:, module.XIDS.index("S")], 4.0)
    assert np.allclose(simulation.p, p)


def test_diffrax_output_times(tmp_path: Path) -> None:
    """The output times are any times from 0 on, a single one included."""
    module = _module(DECAY, tmp_path)
    ts = jnp.array([0.5, 1.0, 1.0, 7.25])
    simulation = module.simulate(ts)
    reference = module.simulate(jnp.linspace(0.0, 7.25, 30))
    assert np.array_equal(simulation.t, ts)
    assert simulation.x[1] == pytest.approx(simulation.x[2])
    assert simulation.x[3] == pytest.approx(reference.x[-1], rel=1e-7)
    single = module.simulate(jnp.array([0.0]))
    assert np.allclose(single.x, module.initial_values()[0])


@pytest.mark.parametrize("ts", [[1.0, 0.5], [-1.0, 1.0]])
def test_diffrax_output_times_raise(ts: list[float], tmp_path: Path) -> None:
    """Output times which decrease or are negative raise."""
    module = _module(DECAY, tmp_path)
    with pytest.raises(RuntimeError, match="output times"):
        module.simulate(jnp.array(ts))


def test_diffrax_max_steps(tmp_path: Path) -> None:
    """An integration which takes more than `max_steps` steps raises."""
    module = _module(DECAY, tmp_path)
    with pytest.raises(RuntimeError, match="steps"):
        module.simulate(jnp.linspace(0.0, 10.0, 3), max_steps=5, max_step=0.01)


def test_diffrax_solver(tmp_path: Path) -> None:
    """Another solver of diffrax integrates the model as the default one."""
    module = _module(DECAY, tmp_path)
    ts = jnp.linspace(0.0, 10.0, 11)
    default = module.simulate(ts)
    explicit = module.simulate(ts, solver=diffrax.Tsit5())
    assert np.allclose(explicit.x, default.x, rtol=1e-6, atol=1e-9)


# --- the contract of JAX --------------------------------------------------------------


def _loss(
    module: ModuleType, ts: jax.Array, **options: object
) -> Callable[[jax.Array], jax.Array]:
    """A smooth function of the simulation of the constants."""

    def loss(p: jax.Array) -> jax.Array:
        return jnp.sum(module.simulate(ts, p, **options).x ** 2)

    return loss


def _finite_differences(
    function: Callable[[jax.Array], jax.Array], p: np.ndarray, step: float = 1e-6
) -> np.ndarray:
    """The gradient of a function by central differences of the relative step."""
    gradient = []
    for k in range(len(p)):
        h = step * max(1.0, abs(float(p[k])))
        e = np.zeros(len(p))
        e[k] = h
        gradient.append((function(p + e) - function(p - e)) / (2 * h))
    return np.array(gradient)


def test_diffrax_vmap(tmp_path: Path) -> None:
    """`jax.vmap` over the constants is a simulation per row."""
    module = _module(DECAY, tmp_path)
    ts = jnp.linspace(0.0, 5.0, 11)
    ps = module.P0[None, :] * jnp.linspace(0.5, 1.5, 7)[:, None]
    xs = jax.vmap(lambda p: module.simulate(ts, p).x)(ps)
    for k in (0, 3, 6):
        assert np.allclose(xs[k], module.simulate(ts, ps[k]).x, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    ("adjoint", "transform"),
    [
        ("RecursiveCheckpointAdjoint", "grad"),
        ("ForwardMode", "jacfwd"),
        ("DirectAdjoint", "grad"),
        ("DirectAdjoint", "jacfwd"),
    ],
)
def test_diffrax_gradient(adjoint: str, transform: str, tmp_path: Path) -> None:
    """The derivative of a simulation is its derivative by finite differences."""
    module = _module(DECAY, tmp_path)
    ts = jnp.linspace(0.0, 5.0, 11)
    loss = _loss(module, ts, adjoint=getattr(diffrax, adjoint)())
    gradient = getattr(jax, transform)(loss)(module.P0)
    expected = _finite_differences(loss, np.asarray(module.P0))
    assert np.allclose(gradient, expected, rtol=1e-5, atol=1e-8)


def test_diffrax_derivative_of_initial_states(tmp_path: Path) -> None:
    """The simulation is differentiated with respect to the initial states."""
    module = _module(DECAY, tmp_path)
    ts = jnp.linspace(0.0, 2.0, 3)

    def final_s(x0: jax.Array) -> jax.Array:
        return module.simulate(ts, None, x0).x[-1, 0]

    k = float(module.P0[module.PIDS.index("k")])
    # S(t) = S0 exp(-k t)
    assert jax.grad(final_s)(jnp.array([2.0, 0.0]))[0] == pytest.approx(
        np.exp(-k * 2.0), rel=1e-6
    )


def test_diffrax_functions_without_simulator(tmp_path: Path) -> None:
    """The functions of `simulator=False` are integrated with diffrax as they are."""
    module = _module(DECAY, tmp_path, simulator=False)
    assert not hasattr(module, "simulate")
    x0, p = module.initial_values()
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(module.f_dxdt),
        diffrax.Tsit5(),
        t0=0.0,
        t1=3.0,
        dt0=None,
        y0=x0,
        args=p,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
    )
    k = float(p[module.PIDS.index("k")])
    assert solution.ys[-1, 0] == pytest.approx(2.0 * np.exp(-k * 3.0), rel=1e-8)


# --- events ---------------------------------------------------------------------------

EVENT_RTOL = 1e-4
EVENT_ATOL = 1e-6

# a dose whenever A falls below thr, a delayed event of a strict trigger, a
# non-persistent event which its trigger drops and an event which changes a constant:
# the event times depend on the constants
DOSES = """
    A = 3; B = 1
    k = 0.5; thr = 1; dose = 2; d = 0.55; kb = 0.2; thr2 = 2.8
    A' = -k * A
    B' = -kb * B
    E0: at A < thr: A = A + dose
    E1: at d after time > 1, fromTrigger=false: B = B + 1
    E2: at 0.3 after A > thr2, persistent=false, fromTrigger=true: B = B + 10
    E3: at time >= 2: kb = 2 * kb
"""


def _assert_events_as_roadrunner(antimony: str, tmp_path: Path) -> pd.DataFrame:
    """Assert that a model with events simulates as roadrunner.

    Returns:
        the simulation of the generated diffrax code
    """
    sbml = model_sbml(antimony)
    system = OdeSystem.from_sbml(sbml)
    assert system.events
    module = diffrax_module(system, tmp_path / "events.py")
    df = diffrax_frame(module)
    assert_table_as_roadrunner(sbml, df, rtol=EVENT_RTOL, atol=EVENT_ATOL)
    return df


@pytest.mark.parametrize(
    "name", [name for name in EVENT_MODELS if name != "infinite_cascade"]
)
def test_diffrax_events_as_roadrunner(name: str, tmp_path: Path) -> None:
    """Every model with events of the numerical formats simulates as roadrunner."""
    _assert_events_as_roadrunner(EVENT_MODELS[name], tmp_path)


def test_diffrax_doses_as_roadrunner(tmp_path: Path) -> None:
    """Repeated doses, delays and a dropped execution simulate as roadrunner."""
    df = _assert_events_as_roadrunner(DOSES, tmp_path)
    assert list(df.columns) == ["time", "A", "B", "kb"]
    assert df["kb"].iloc[-1] == 0.4
    # the delayed execution of the strict trigger is after the output time 1.55
    row = df[np.isclose(df["time"], 1.6)].iloc[0]
    assert row["B"] == pytest.approx(np.exp(-0.2 * 1.6) + np.exp(-0.2 * 0.05))


def test_diffrax_two_events(tmp_path: Path) -> None:
    """Events with a delay and priorities, a constant which an event changes."""
    df = _assert_events_as_roadrunner(TWO_EVENTS, tmp_path)
    assert list(df.columns) == ["time", "S", "R1", "k", "total"]
    assert df["k"].iloc[-1] == 1.0
    assert df["total"].iloc[-1] >= 2


@pytest.mark.parametrize("relation", [">=", ">"])
def test_diffrax_event_at_t0(relation: str, tmp_path: Path) -> None:
    """A trigger which holds at t = 0 fires there if its initial value is false."""
    df = _assert_events_as_roadrunner(EVENT_MODELS[f"at_t0[{relation}]"], tmp_path)
    fired = 1.0 if relation == ">=" else 0.0
    assert df["B"].iloc[0] == fired
    assert df["B"].iloc[-1] == 1.0
    assert df["C"].iloc[-1] == 1.0 - fired
    assert df["D"].iloc[-1] == 3.0


@pytest.mark.parametrize(("relation", "before"), [(">=", False), (">", True)])
def test_diffrax_event_at_a_time_point(
    relation: str, before: bool, tmp_path: Path
) -> None:
    """At an output time the values are those after its events.

    `time >= 2` fires at t = 2, `time > 2` right after it, so that the output time
    t = 2 has the values before the event.
    """
    df = _assert_events_as_roadrunner(
        EVENT_MODELS[f"at_time_point[{relation}]"], tmp_path
    )
    row = df[np.isclose(df["time"], 2.0)].iloc[0]
    assert row["A"] == pytest.approx(2.0 if before else 10.0)


def test_diffrax_event_at_a_time_point_to_the_rounding(tmp_path: Path) -> None:
    """An execution at an output time to the rounding of the integration is at it."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["rounding"], tmp_path)
    assert df[np.isclose(df["time"], 3.4)]["B"].iloc[0] == 1.0


def test_diffrax_event_priority(tmp_path: Path) -> None:
    """Events at the same time execute in the order of their priorities."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["priority"], tmp_path)
    assert df[np.isclose(df["time"], 2.0)]["B"].iloc[0] == 4.0
    assert df["B"].iloc[-1] == 17.0


def test_diffrax_event_delay(tmp_path: Path) -> None:
    """A delayed event executes after its delay, each time its trigger turns true."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["delay"], tmp_path)
    assert df["n"].iloc[-1] == 4.0
    assert df["d"].iloc[-1] == 0.5
    assert df["S"].iloc[-1] == pytest.approx(4.831581402482879, rel=1e-6)


def test_diffrax_event_use_values_from_trigger_time(tmp_path: Path) -> None:
    """The values of a delayed event are from the trigger time or the execution."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["trigger_values"], tmp_path)
    assert df["B"].iloc[-1] == pytest.approx(1.0)
    assert df["C"].iloc[-1] == pytest.approx(3.0)


def test_diffrax_event_persistent(tmp_path: Path) -> None:
    """An event which is not persistent is dropped if its trigger turns false."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["persistent"], tmp_path)
    assert df["B"].iloc[-1] == 0.0
    assert df["C"].iloc[-1] == 1.0
    assert df["F"].iloc[-1] == 1.0


def test_diffrax_event_changes_compartment(tmp_path: Path) -> None:
    """An event which changes a size keeps the amounts of its species."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["compartment"], tmp_path)
    assert df["V"].iloc[-1] == 0.5
    row = df[np.isclose(df["time"], 2.2)].iloc[0]
    assert row["S"] == pytest.approx(0.8025187974297124, rel=1e-6)
    assert row["T"] == pytest.approx(0.62, rel=1e-6)
    assert df["S"].iloc[-1] == pytest.approx(7.278367917708202, rel=1e-6)
    assert df["T"].iloc[-1] == pytest.approx(4.1, rel=1e-6)


def test_diffrax_event_cascade(tmp_path: Path) -> None:
    """An execution which makes another trigger true fires it at the same time."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["cascade"], tmp_path)
    assert df["B"].iloc[-1] == 1.0
    assert df[np.isclose(df["time"], 1.2)]["B"].iloc[0] == 1.0


def test_diffrax_event_assigns_its_threshold(tmp_path: Path) -> None:
    """An event which sets its trigger to the root keeps integrating."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["threshold"], tmp_path)
    assert df["A"].between(1.0, 2.0).all()
    assert df["A"].iloc[-1] == pytest.approx(1.5)


def test_diffrax_event_model_without_states(tmp_path: Path) -> None:
    """A model of events and rules only integrates nothing but its events."""
    df = _assert_events_as_roadrunner(EVENT_MODELS["without_states"], tmp_path)
    assert list(df.columns) == ["time", "y", "B"]
    assert df["B"].iloc[-1] == 5.0


def _swelling(model: libsbml.Model) -> None:
    """An event at t = 1 which doubles the parameter `Va0` of the rule of `Va`."""
    event: libsbml.Event = model.createEvent()
    event.setId("swell")
    event.setUseValuesFromTriggerTime(True)
    trigger: libsbml.Trigger = event.createTrigger()
    trigger.setInitialValue(True)
    trigger.setPersistent(True)
    trigger.setMath(libsbml.parseL3Formula("time > 1"))
    assignment: libsbml.EventAssignment = event.createEventAssignment()
    assignment.setVariable("Va0")
    assignment.setMath(libsbml.parseL3Formula("2 * Va0"))


@pytest.mark.parametrize("swelling", [False, True])
def test_diffrax_variable_compartments(swelling: bool, tmp_path: Path) -> None:
    """Sizes of a rate rule, an assignment rule and events simulate as roadrunner.

    The event `swell` changes a parameter of the rule of the size `Va`, the amount of
    `S2` in it is kept.
    """
    sbml = VARIABLE_COMPARTMENT.read_text()
    if swelling:
        sbml = edit_sbml(sbml, _swelling)
    assert_diffrax_as_roadrunner(sbml, tmp_path)
    module = diffrax_module(OdeSystem.from_sbml(sbml), tmp_path / "v.py")
    df = diffrax_frame(module)
    assert_table_as_roadrunner(sbml, df, rtol=EVENT_RTOL, atol=EVENT_ATOL)


def test_diffrax_event_infinite_cascade_raises(tmp_path: Path) -> None:
    """Events which trigger each other at one time without end raise."""
    module = _module(EVENT_MODELS["infinite_cascade"], tmp_path)
    assert module.MAX_CASCADE == 10000
    with pytest.raises(RuntimeError, match="infinite cascade"):
        module.simulate(jnp.linspace(0.0, 2.0, 11))


@pytest.mark.parametrize("event", ["", "; E1: at time > 5: x = 2"])
def test_diffrax_raises_for_a_state_without_bound(event: str, tmp_path: Path) -> None:
    """A state which grows without bound raises, it does not integrate for ever.

    `x' = x^2` with `x(0) = 1` is `1 / (1 - t)`, which has a pole at t = 1.
    """
    module = _module(f"x = 1; x' = x^2{event}", tmp_path)
    with pytest.raises(RuntimeError):
        module.simulate(jnp.linspace(0.0, 2.0, 11))


def test_diffrax_max_pending(tmp_path: Path) -> None:
    """More scheduled executions than `max_pending` raise.

    The trigger turns true every 2 pi / 10, its executions are 5 later: 8 of them are
    scheduled at a time.
    """
    module = _module("B = 0; E1: at 5 after sin(10 * time) > 0.5: B = B + 1", tmp_path)
    ts = jnp.linspace(0.0, 10.0, 101)
    assert module.simulate(ts).x.shape == (101, 0)
    with pytest.raises(RuntimeError, match="max_pending"):
        module.simulate(ts, max_pending=4)


def test_diffrax_max_segments(tmp_path: Path) -> None:
    """More segments of the integration than `max_segments` raise."""
    module = _module(EVENT_MODELS["delay"], tmp_path)
    with pytest.raises(RuntimeError, match="3 segments"):
        module.simulate(jnp.linspace(0.0, 10.0, 11), max_segments=3)


def test_diffrax_trigger_within_a_step(tmp_path: Path) -> None:
    """A trigger is found at the end of a step, which is at most `max_step`.

    The trigger of `window` holds from t = 1 to 1.5, which the steps of the output
    times, 0.2, find. A trigger from t = 1.01 to 1.06 is stepped over by them and
    found by steps of at most 0.01.
    """
    ts = jnp.linspace(0.0, 4.0, 21)
    module = _module(EVENT_MODELS["window"], tmp_path)
    assert module.simulate(ts).p[-1, module.PIDS.index("B")] == 1.0
    short = diffrax_module(
        _system("B = 0; E1: at time > 1.01 && time < 1.06: B = B + 1"),
        tmp_path / "short.py",
    )
    index = short.PIDS.index("B")
    assert short.simulate(ts).p[-1, index] == 0.0
    assert short.simulate(ts, max_step=0.01).p[-1, index] == 1.0


@pytest.mark.parametrize("antimony", [EVENT_MODELS["rounding"], "S = 1; S' = -S"])
def test_diffrax_simulate_without_time(antimony: str, tmp_path: Path) -> None:
    """Output times at t = 0 only are the initial states, after the events at 0."""
    module = _module(antimony, tmp_path)
    x0, _ = module.initial_values()
    simulation = module.simulate(jnp.zeros(5))
    assert np.array_equal(simulation.x, np.broadcast_to(x0, (5, len(x0))))


def test_diffrax_events_without_simulator(tmp_path: Path) -> None:
    """`simulator=False` writes the functions of the events for a solver of one's own."""
    module = _module(TWO_EVENTS, tmp_path, simulator=False)
    assert not hasattr(module, "simulate")
    assert module.EVENT_IDS == ["E1", "E2"]
    x, p = module.initial_values()
    assert module.event_triggers(0.0, x, p).tolist() == [-5.0, -3.0]
    assert module.event_conditions(0.0, x, p).tolist() == [False, False]
    assert module.event_delay(0, 0.0, x, p) == 1.0
    assert module.event_delay(1, 0.0, x, p) == 0.0
    assert module.event_priorities(0.0, x, p).tolist() == [2.0, 1.0]
    values = module.event_values(0, 0.0, x, p)
    x_new, p_new = module.event_assign(0, 0.0, x, p, values)
    assert x_new[module.XIDS.index("S")] == 15.0
    assert p_new[module.PIDS.index("total")] == 1.0
    assert x[module.XIDS.index("S")] == 10.0


def test_model_without_events_has_no_events(tmp_path: Path) -> None:
    """The code of a model without events has no functions of events."""
    module = _module(DECAY, tmp_path)
    assert not hasattr(module, "EVENT_IDS")
    assert "event_triggers" not in (tmp_path / "model.py").read_text()


def test_diffrax_events_vmap(tmp_path: Path) -> None:
    """`jax.vmap` over the constants of a model with events, whose times differ."""
    module = _module(DOSES, tmp_path)
    ts = jnp.linspace(0.0, 10.0, 51)
    ps = module.P0[None, :] * jnp.linspace(0.8, 1.2, 5)[:, None]
    xs = jax.vmap(lambda p: module.simulate(ts, p).x)(ps)
    for k in (0, 2, 4):
        assert np.allclose(xs[k], module.simulate(ts, ps[k]).x, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("adjoint", "transform"),
    [
        ("RecursiveCheckpointAdjoint", "grad"),
        ("ForwardMode", "jacfwd"),
        ("DirectAdjoint", "grad"),
    ],
)
def test_diffrax_events_gradient(adjoint: str, transform: str, tmp_path: Path) -> None:
    """The derivative of a simulation with events is its derivative by differences.

    The times of the doses depend on `k`, `thr` and `dose`, the time of the delayed
    execution on `d`: the derivative includes the shift of the event times. The
    derivative is the one of the integration with its steps, the differences change
    the steps as well, which a small step of the differences amplifies: at the
    tolerances of the reference and a step of 1e-4 both are the derivative of the
    solution.
    """
    module = _module(DOSES, tmp_path)
    # output times which no event time is near, the loss is smooth in the constants
    ts = jnp.linspace(0.0, 10.0, 21)
    loss = _loss(
        module, ts, adjoint=getattr(diffrax, adjoint)(), rtol=1e-10, atol=1e-12
    )
    gradient = np.asarray(getattr(jax, transform)(loss)(module.P0))
    expected = _finite_differences(loss, np.asarray(module.P0), step=1e-4)
    # `thr2` changes the loss only by the integration, its event is always dropped
    for k, pid in enumerate(module.PIDS):
        if pid != "thr2":
            assert gradient[k] == pytest.approx(expected[k], rel=1e-5), pid


# --- the code -------------------------------------------------------------------------


def test_diffrax_script_prints_the_simulation(tmp_path: Path) -> None:
    """The file run as a script prints the head of a simulation."""
    path = tmp_path / "decay.py"
    _system(DECAY).write(path, "diffrax")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    assert "time" in result.stdout
    assert "S" in result.stdout


def test_diffrax_needs_its_format(tmp_path: Path) -> None:
    """`.py` is written as python, diffrax is named with `fmt`."""
    system = _system(DECAY)
    assert FORMATS["diffrax"].suffixes == ()
    assert "import numpy as np" in system.write(tmp_path / "a.py").read_text()
    code = system.write(tmp_path / "b.py", "diffrax").read_text()
    assert "import diffrax" in code
    assert code == system.render("diffrax")


def test_diffrax_layout() -> None:
    """The code reads as the model: named locals with their name and unit."""
    code = OdeSystem.from_sbml(REPRESSILATOR_SBML).render("diffrax")
    lines = code.splitlines()
    assert lines[0].startswith('"""')
    assert f"sbmlode {sbmlode.__version__}" in code
    assert any(line.strip().startswith("PX = x[0]  # ") for line in lines)
    assert 'jax.config.update("jax_enable_x64", True)' in code
    statements = [
        n.lineno for n in ast.walk(ast.parse(code)) if isinstance(n, ast.stmt)
    ]
    assert len(statements) == len(set(statements))
    assert not [line for line in lines if line != line.rstrip() or "\t" in line]
    assert code.endswith("\n")
    assert not code.endswith("\n\n")


@pytest.mark.parametrize(
    "antimony",
    [RULES, DECAY, TWO_EVENTS, DOSES, EVENT_MODELS["without_states"]],
    ids=["rules", "decay", "events", "doses", "stateless"],
)
def test_diffrax_names_are_reserved(antimony: str) -> None:
    """Every name the code writes, apart from the ids, is reserved for JAX."""
    system = _system(antimony)
    ids = [q.symbol.sid for q in system.quantities]
    ids += [r.symbol.sid for r in system.reactions]
    ids += [f.symbol.sid for f in system.functions]
    ids += [e.symbol.sid for e in system.events]
    ids += [r.symbol.sid for r in system.size_rates]
    model_names = set(code_names(ids, "jax").values())
    events = context(system, FORMATS["diffrax"], {"simulator": True})["events"]
    assert isinstance(events, list)
    model_names |= {f for e in events for f in e["functions"].values() if f}
    for simulator in (True, False):
        tree = ast.parse(system.render("diffrax", simulator=simulator))
        written: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                written.add(node.id)
            elif isinstance(node, ast.arg):
                written.add(node.arg)
            elif isinstance(node, ast.FunctionDef | ast.ClassDef):
                written.add(node.name)
            elif isinstance(node, ast.alias):
                written.add((node.asname or node.name).split(".")[0])
        arguments = {"S", "km", "k", "n"}
        assert written - model_names - arguments <= RESERVED["jax"]


@pytest.mark.skipif(
    importlib.util.find_spec("ruff") is None, reason="ruff is not installed"
)
@pytest.mark.parametrize("simulator", [True, False])
@pytest.mark.parametrize(
    "antimony",
    [RULES, DECAY, "k = 2; y := k * time", TWO_EVENTS, EVENT_MODELS["without_states"]],
    ids=["rules", "decay", "rule", "events", "stateless"],
)
def test_diffrax_code_passes_ruff(antimony: str, simulator: bool) -> None:
    """The generated diffrax code passes `ruff check` with the default rules."""
    code = _system(antimony).render("diffrax", simulator=simulator)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--no-cache",
            "--stdin-filename",
            "model.py",
            "-",
        ],
        input=code,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
