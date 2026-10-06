"""The ODE system of the model BIOMD0000000012: Elowitz2000 - Repressilator.

Written by sbmlode 0.2.0 from BIOMD0000000012_urn.xml, SBML L2V3.

Units of the model:

    time       min
    substance  item
    extent     item
    volume     fl
    area       m^2
    length     m

States x:

       id  name          unit
    0  PX  LacI protein  item
    1  PY  TetR protein  item
    2  PZ  cI protein    item
    3  X   LacI mRNA     item
    4  Y   TetR mRNA     item
    5  Z   cI mRNA       item

`initial_values(p)` returns the initial states x0 and the constants p at t = 0,
`f_dxdt(t, x, p)` the rates of change of the states and `f_y(t, x, p)` the assigned
values y, the rules and the reaction rates. They are functions of JAX: `jax.jit`
compiles them, `jax.vmap` maps them over arrays and `jax.grad` differentiates them.
`simulate(ts)` integrates the model with diffrax and returns the states, the assigned
values and the constants at the output times ts, `to_frame` a table of them, the file
run as a script prints the head of a simulation:

    ts = jnp.linspace(0.0, 10.0, 101)
    simulation = simulate(ts)
    # a simulation for each row of the constants ps
    xs = jax.vmap(lambda p: simulate(ts, p).x)(ps)
    # the gradient of a function of the simulation with respect to the constants
    gradient = jax.grad(lambda p: jnp.sum(simulate(ts, p).x ** 2))(P0)
"""

from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# float64, the precision of the SBML simulators, which the tolerances of the
# integration need, before the first array; a switch of the process which imports
# this file
jax.config.update("jax_enable_x64", True)

# the ids of the states x, the constants p and the assigned values y
XIDS = [
    "PX",  # LacI protein [item]
    "PY",  # TetR protein [item]
    "PZ",  # cI protein [item]
    "X",  # LacI mRNA [item]
    "Y",  # TetR mRNA [item]
    "Z",  # cI mRNA [item]
]
PIDS = [
    "cell",  # [fl]
    "eff",  # translation efficiency
    "n",
    "KM",
    "tau_mRNA",  # mRNA half life
    "tau_prot",  # protein half life
    "ps_a",  # tps_active
    "ps_0",  # tps_repr
]
YIDS = [
    "t_ave",  # average mRNA life time
    "beta",
    "k_tl",
    "a_tr",
    "a0_tr",
    "kd_prot",
    "kd_mRNA",
    "alpha",
    "alpha0",
    "Reaction1",  # degradation of LacI transcripts [item/min]
    "Reaction2",  # degradation of TetR transcripts [item/min]
    "Reaction3",  # degradation of CI transcripts [item/min]
    "Reaction4",  # translation of LacI [item/min]
    "Reaction5",  # translation of TetR [item/min]
    "Reaction6",  # translation of CI [item/min]
    "Reaction7",  # degradation of LacI [item/min]
    "Reaction8",  # degradation of TetR [item/min]
    "Reaction9",  # degradation of CI [item/min]
    "Reaction10",  # transcription of LacI [item/min]
    "Reaction11",  # transcription of TetR [item/min]
    "Reaction12",  # transcription of CI [item/min]
]

# the names and the units of the ids
NAMES = {
    "PX": "LacI protein",
    "PY": "TetR protein",
    "PZ": "cI protein",
    "X": "LacI mRNA",
    "Y": "TetR mRNA",
    "Z": "cI mRNA",
    "cell": None,
    "eff": "translation efficiency",
    "n": "n",
    "KM": "KM",
    "tau_mRNA": "mRNA half life",
    "tau_prot": "protein half life",
    "ps_a": "tps_active",
    "ps_0": "tps_repr",
    "t_ave": "average mRNA life time",
    "beta": "beta",
    "k_tl": "k_tl",
    "a_tr": "a_tr",
    "a0_tr": "a0_tr",
    "kd_prot": "kd_prot",
    "kd_mRNA": "kd_mRNA",
    "alpha": "alpha",
    "alpha0": "alpha0",
    "Reaction1": "degradation of LacI transcripts",
    "Reaction2": "degradation of TetR transcripts",
    "Reaction3": "degradation of CI transcripts",
    "Reaction4": "translation of LacI",
    "Reaction5": "translation of TetR",
    "Reaction6": "translation of CI",
    "Reaction7": "degradation of LacI",
    "Reaction8": "degradation of TetR",
    "Reaction9": "degradation of CI",
    "Reaction10": "transcription of LacI",
    "Reaction11": "transcription of TetR",
    "Reaction12": "transcription of CI",
}
UNITS = {
    "PX": "item",
    "PY": "item",
    "PZ": "item",
    "X": "item",
    "Y": "item",
    "Z": "item",
    "cell": "fl",
    "eff": None,
    "n": None,
    "KM": None,
    "tau_mRNA": None,
    "tau_prot": None,
    "ps_a": None,
    "ps_0": None,
    "t_ave": None,
    "beta": None,
    "k_tl": None,
    "a_tr": None,
    "a0_tr": None,
    "kd_prot": None,
    "kd_mRNA": None,
    "alpha": None,
    "alpha0": None,
    "Reaction1": "item/min",
    "Reaction2": "item/min",
    "Reaction3": "item/min",
    "Reaction4": "item/min",
    "Reaction5": "item/min",
    "Reaction6": "item/min",
    "Reaction7": "item/min",
    "Reaction8": "item/min",
    "Reaction9": "item/min",
    "Reaction10": "item/min",
    "Reaction11": "item/min",
    "Reaction12": "item/min",
}

# the default values of the constants, `jnp.nan` for one without a value, e.g. one
# which an initial assignment sets and `initial_values` computes
P0 = jnp.array([
    1.0,  # cell
    20.0,  # eff
    2.0,  # n
    40.0,  # KM
    2.0,  # tau_mRNA
    10.0,  # tau_prot
    0.5,  # ps_a
    0.0005,  # ps_0
], dtype=float)


def initial_values(p: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
    """The initial states x0 and the constants p at t = 0.

    The initial values, initial assignments and the rules they need are evaluated
    in the order of their dependencies.

    Every constant keeps the value passed.

    Args:
        p: the constants, `P0` if not given

    Returns:
        the initial states x0 and the constants p, a new array
    """
    p = P0 if p is None else jnp.asarray(p, dtype=float)
    # initial values
    PX = 0.0  # LacI protein [item]
    PY = 0.0  # TetR protein [item]
    PZ = 0.0  # cI protein [item]
    X = 0.0  # LacI mRNA [item]
    Y = 20.0  # TetR mRNA [item]
    Z = 0.0  # cI mRNA [item]
    x0 = jnp.array([PX, PY, PZ, X, Y, Z], dtype=float)
    return x0, p


def f_dxdt(t: jax.Array, x: jax.Array, p: jax.Array) -> jax.Array:
    """The rates of change dx/dt of the states x at the time t.

    The arguments are in the order of a vector field of diffrax, `diffrax.ODETerm`
    takes the function with the constants p as its `args`.
    """
    # states
    PX = x[0]  # LacI protein [item]
    PY = x[1]  # TetR protein [item]
    PZ = x[2]  # cI protein [item]
    X = x[3]  # LacI mRNA [item]
    Y = x[4]  # TetR mRNA [item]
    Z = x[5]  # cI mRNA [item]
    # constants
    eff = p[1]  # translation efficiency
    n = p[2]
    KM = p[3]
    tau_mRNA = p[4]  # mRNA half life
    tau_prot = p[5]  # protein half life
    ps_a = p[6]  # tps_active
    ps_0 = p[7]  # tps_repr
    # assigned values and reaction rates
    t_ave = tau_mRNA / jnp.log(2.0)  # average mRNA life time
    k_tl = eff / t_ave
    a_tr = (ps_a - ps_0) * 60.0
    a0_tr = ps_0 * 60.0
    kd_prot = jnp.log(2.0) / tau_prot
    kd_mRNA = jnp.log(2.0) / tau_mRNA
    Reaction1 = kd_mRNA * X  # degradation of LacI transcripts [item/min]
    Reaction2 = kd_mRNA * Y  # degradation of TetR transcripts [item/min]
    Reaction3 = kd_mRNA * Z  # degradation of CI transcripts [item/min]
    Reaction4 = k_tl * X  # translation of LacI [item/min]
    Reaction5 = k_tl * Y  # translation of TetR [item/min]
    Reaction6 = k_tl * Z  # translation of CI [item/min]
    Reaction7 = kd_prot * PX  # degradation of LacI [item/min]
    Reaction8 = kd_prot * PY  # degradation of TetR [item/min]
    Reaction9 = kd_prot * PZ  # degradation of CI [item/min]
    Reaction10 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PZ, n))  # transcription of LacI [item/min]
    Reaction11 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PX, n))  # transcription of TetR [item/min]
    Reaction12 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PY, n))  # transcription of CI [item/min]
    # rates of change
    return jnp.array([
        Reaction4 - Reaction7,  # dPX/dt
        Reaction5 - Reaction8,  # dPY/dt
        Reaction6 - Reaction9,  # dPZ/dt
        -Reaction1 + Reaction10,  # dX/dt
        -Reaction2 + Reaction11,  # dY/dt
        -Reaction3 + Reaction12,  # dZ/dt
    ], dtype=float)


def f_y(t: jax.Array, x: jax.Array, p: jax.Array) -> jax.Array:
    """The assigned values y at the time t, the rules and the reaction rates."""
    # states
    PX = x[0]  # LacI protein [item]
    PY = x[1]  # TetR protein [item]
    PZ = x[2]  # cI protein [item]
    X = x[3]  # LacI mRNA [item]
    Y = x[4]  # TetR mRNA [item]
    Z = x[5]  # cI mRNA [item]
    # constants
    eff = p[1]  # translation efficiency
    n = p[2]
    KM = p[3]
    tau_mRNA = p[4]  # mRNA half life
    tau_prot = p[5]  # protein half life
    ps_a = p[6]  # tps_active
    ps_0 = p[7]  # tps_repr
    # assigned values and reaction rates
    t_ave = tau_mRNA / jnp.log(2.0)  # average mRNA life time
    beta = tau_mRNA / tau_prot
    k_tl = eff / t_ave
    a_tr = (ps_a - ps_0) * 60.0
    a0_tr = ps_0 * 60.0
    kd_prot = jnp.log(2.0) / tau_prot
    kd_mRNA = jnp.log(2.0) / tau_mRNA
    alpha = a_tr * eff * tau_prot / (jnp.log(2.0) * KM)
    alpha0 = a0_tr * eff * tau_prot / (jnp.log(2.0) * KM)
    Reaction1 = kd_mRNA * X  # degradation of LacI transcripts [item/min]
    Reaction2 = kd_mRNA * Y  # degradation of TetR transcripts [item/min]
    Reaction3 = kd_mRNA * Z  # degradation of CI transcripts [item/min]
    Reaction4 = k_tl * X  # translation of LacI [item/min]
    Reaction5 = k_tl * Y  # translation of TetR [item/min]
    Reaction6 = k_tl * Z  # translation of CI [item/min]
    Reaction7 = kd_prot * PX  # degradation of LacI [item/min]
    Reaction8 = kd_prot * PY  # degradation of TetR [item/min]
    Reaction9 = kd_prot * PZ  # degradation of CI [item/min]
    Reaction10 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PZ, n))  # transcription of LacI [item/min]
    Reaction11 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PX, n))  # transcription of TetR [item/min]
    Reaction12 = a0_tr + a_tr * jnp.power(KM, n) / (jnp.power(KM, n) + jnp.power(PY, n))  # transcription of CI [item/min]
    return jnp.array([
        t_ave, beta, k_tl, a_tr, a0_tr, kd_prot, kd_mRNA, alpha, alpha0, Reaction1,
        Reaction2, Reaction3, Reaction4, Reaction5, Reaction6, Reaction7, Reaction8,
        Reaction9, Reaction10, Reaction11, Reaction12,
    ], dtype=float)


class Simulation(NamedTuple):
    """A simulation at its output times: the states, assigned values and constants."""

    t: jax.Array  # the output times
    x: jax.Array  # the states, a row per output time, a column per id of XIDS
    y: jax.Array  # the assigned values, a column per id of YIDS
    p: jax.Array  # the constants, a column per id of PIDS


# the limit of the steps of an integration beyond those which the largest step forces
MAX_STEPS = 100000


@eqx.filter_jit
def simulate(
    ts: jax.Array,
    p: jax.Array | None = None,
    x0: jax.Array | None = None,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_step: float | None = None,
    max_steps: int | None = None,
) -> Simulation:
    """Simulate the model from t = 0 to the last output time.

    `eqx.filter_jit` compiles the simulation for the number of output times and the
    options, the output times, the constants and the initial states are traced: a
    simulation with other values runs without a compilation.

    Args:
        ts: the output times, from 0 on and not decreasing
        p: the constants, `P0` if not given
        x0: the initial states, those of `initial_values` if not given
        rtol: the relative tolerance of the integration
        atol: the absolute tolerance of the integration
        solver: the solver of diffrax, `diffrax.Kvaerno5()` if not given, an implicit
            solver for stiff models; `diffrax.Tsit5()` for a model which is not stiff
        adjoint: the adjoint of diffrax, which decides how the simulation is
            differentiated: `diffrax.RecursiveCheckpointAdjoint()` if not given, for
            reverse mode (`jax.grad`), `diffrax.ForwardMode()` for forward mode
            (`jax.jacfwd`), `diffrax.DirectAdjoint()` for both
        max_step: the largest step of the integration, by default the distance of
            the output times, `ts[-1] / (len(ts) - 1)`
        max_steps: the largest number of steps of the integration, by default
            `MAX_STEPS` plus twice the number of output times

    Returns:
        the time, the states, the assigned values and the constants at the output
        times

    Raises:
        RuntimeError: for output times which are negative or decrease, if the
            integration fails or takes more than `max_steps` steps, e.g. when a
            state grows without bound
    """
    ts = jnp.asarray(ts, dtype=float)
    ts = eqx.error_if(
        ts,
        (ts[0] < 0.0) | jnp.any(ts[1:] < ts[:-1]),
        "The output times ts must not be negative and must not decrease.",
    )
    x_initial, p = initial_values(p)
    x = x_initial if x0 is None else jnp.asarray(x0, dtype=float)
    points = ts.shape[0]
    t_end = ts[-1]
    if max_step is None:
        max_step = t_end / (points - 1) if points > 1 else jnp.inf
        max_step = jnp.where(max_step > 0.0, max_step, jnp.inf)
    if max_steps is None:
        max_steps = MAX_STEPS + 2 * points
    ps = jnp.broadcast_to(p, (points, p.shape[0]))
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(f_dxdt),
        diffrax.Kvaerno5() if solver is None else solver,
        t0=0.0,
        t1=t_end,
        dt0=None,
        y0=x,
        args=p,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=max_step),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    xs = solution.ys
    return Simulation(ts, xs, jax.vmap(f_y)(ts, xs, ps), ps)


def to_frame(simulation: Simulation) -> pd.DataFrame:
    """The simulation as a table: the time, the states and the assigned values, a row
    per output time."""
    columns = ["time", *XIDS, *YIDS]
    data = np.column_stack([simulation.t, simulation.x, simulation.y])
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    print(to_frame(simulate(jnp.linspace(0.0, 10.0, 101))).head())
