"""Test the diffrax code of the ODE export: against roadrunner and as JAX code."""

import ast
import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ode_helpers import (
    FORMULAS,
    assert_diffrax_as_roadrunner,
    assert_diffrax_trajectory_as_roadrunner,
    assert_table_as_roadrunner,
    diffrax_frame,
    diffrax_module,
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
    function: Callable[[jax.Array], jax.Array], p: np.ndarray
) -> np.ndarray:
    """The gradient of a function by central differences."""
    gradient = []
    for k in range(len(p)):
        h = 1e-6 * max(1.0, abs(float(p[k])))
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


@pytest.mark.parametrize("antimony", [RULES, DECAY], ids=["rules", "decay"])
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
@pytest.mark.parametrize("antimony", [RULES, DECAY, "k = 2; y := k * time"])
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
