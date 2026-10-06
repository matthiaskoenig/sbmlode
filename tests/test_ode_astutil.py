"""Test the math helpers of the ODE export."""

import math

import libsbml
import pytest

from sbmlode.astutil import (
    TIME,
    derivative,
    drop_zero_terms,
    is_number,
    name,
    negated,
    number,
    signed_sum,
    walk,
)


def parse(text: str) -> libsbml.ASTNode:
    """Math of an infix formula."""
    ast = libsbml.parseL3Formula(text)
    assert ast is not None, libsbml.getLastParseL3Error()
    return ast


def formula(ast: libsbml.ASTNode) -> str:
    """Infix formula of the math, as libsbml writes it."""
    return str(libsbml.formulaToL3String(ast))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2", "-2"),
        ("-x", "x"),
        ("b - 1", "1 - b"),
        ("2 * J", "-2 * J"),
        ("f * J", "-f * J"),
        ("J / c", "-J / c"),
        ("min(a, b)", "-min(a, b)"),
        ("x", "-x"),
    ],
)
def test_negated(text: str, expected: str) -> None:
    """A negation carries the sign on a number, a difference or a first factor."""
    assert formula(negated(parse(text))) == expected


def test_signed_sum() -> None:
    """The terms are written with their signs, the zero terms are dropped."""
    terms = [
        (-1, name("J0")),
        (1, parse("2 * J1")),
        (1, number(0.0)),
        (1, name("J2")),
        (-1, parse("f * J3")),
        (-1, name("J4")),
    ]
    assert formula(signed_sum(terms)) == "-J0 + 2 * J1 + J2 - f * J3 - J4"
    assert formula(signed_sum([(1, number(0.0))])) == "0"
    assert formula(signed_sum([(-1, parse("2 * J"))])) == "-2 * J"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a + 0", "a"),
        ("0 + a - 0", "a"),
        ("0 - a", "-a"),
        ("a + -b", "a - b"),
        ("k * (a + 0)", "k * a"),
        ("0 + 0", "0"),
        ("a * 0", "a * 0"),
    ],
)
def test_drop_zero_terms(text: str, expected: str) -> None:
    """The zero terms of every sum are dropped, a product keeps its factors."""
    assert formula(drop_zero_terms(parse(text))) == expected


def test_is_number_and_walk() -> None:
    """A number is recognized by its value, the walk visits every node."""
    assert is_number(parse("0"), 0.0)
    assert is_number(parse("2.5"))
    assert not is_number(parse("x"))
    assert [n.getName() for n in walk(parse("a + b * c")) if n.isName()] == [
        "a",
        "b",
        "c",
    ]


# --- derivative -----------------------------------------------------------------------


def _evaluate(ast: libsbml.ASTNode, values: dict[str, float]) -> float:
    """The value of a math at the given values, evaluated by python."""

    def piecewise(*args: float) -> float:
        for k in range(0, len(args) - 1, 2):
            if args[k + 1]:
                return args[k]
        return args[-1] if len(args) % 2 else math.nan

    functions = {
        "exp": math.exp,
        "ln": math.log,
        "log10": math.log10,
        "log": lambda base, x: math.log(x, base),
        "root": lambda n, x: x ** (1 / n),
        "sqrt": math.sqrt,
        "abs": abs,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sec": lambda x: 1 / math.cos(x),
        "csc": lambda x: 1 / math.sin(x),
        "cot": lambda x: 1 / math.tan(x),
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "piecewise": piecewise,
        "exponentiale": math.e,
        "pi": math.pi,
        "time": values.get("\x00time", 0.0),
    }
    text = formula(ast).replace("^", "**")
    return float(eval(text, functions, dict(values)))  # noqa: S307


DIFFERENTIABLE = [
    "x",
    "2 * x",
    "x * y",
    "x * y * x",
    "x / y",
    "y / x",
    "x^3",
    "x^y",
    "y^x",
    "x^-2",
    "exp(k * x)",
    "ln(x)",
    "log10(x)",
    "log(2, x)",
    "root(3, x)",
    "sqrt(x)",
    "sin(x) * cos(x)",
    "tan(x) + sec(x) + csc(x) + cot(x)",
    "arcsin(x / 3) + arccos(x / 3) + arctan(x)",
    "sinh(x) + cosh(x) + tanh(x)",
    "V0 * (1 + k * x)",
    "x^2 / (K + x^2)",
    "-x",
    "y - x",
    "abs(x - 1.2)",
    "piecewise(2 * x, x > 1.2, x^2)",
    "exponentiale^x + pi * x",
]


@pytest.mark.parametrize("text", DIFFERENTIABLE)
def test_derivative_is_the_finite_difference(text: str) -> None:
    """The derivative equals the central finite difference at a few points."""
    ast = parse(text)
    rate = derivative(ast, "x")
    assert rate is not None
    for point in (0.6, 0.9, 1.7, 2.3):
        values = {"x": point, "y": 1.3, "k": 0.7, "K": 2.0, "V0": 1.5}
        h = 1e-6
        up = _evaluate(ast, {**values, "x": point + h})
        down = _evaluate(ast, {**values, "x": point - h})
        expected = (up - down) / (2 * h)
        assert _evaluate(rate, values) == pytest.approx(expected, rel=1e-5, abs=1e-7)


def test_derivative_of_time() -> None:
    """The csymbol time is differentiated by `TIME`."""
    ast = parse("V0 * exp(k * time)")
    rate = derivative(ast, TIME)
    assert rate is not None
    assert formula(rate) == "V0 * k * exp(k * time)"
    by_k = derivative(ast, "k")
    assert by_k is not None
    assert formula(by_k) == "V0 * time * exp(k * time)"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("y + 2", "0"),
        ("k * x", "k"),
        ("x + y", "1"),
        ("3 * x^2", "6 * x"),
        ("x / y", "1 / y"),
        ("V0 * (1 + k * x)", "V0 * k"),
        ("piecewise(2 * x, x > 1, x^2)", "piecewise(2, x > 1, 2 * x)"),
    ],
)
def test_derivative_is_simplified(text: str, expected: str) -> None:
    """The derivative has no factor 1 and no term 0, its numbers multiplied."""
    rate = derivative(parse(text), "x")
    assert rate is not None
    assert formula(rate) == expected


@pytest.mark.parametrize("text", ["delay(x, 1)", "f(x)", "rateOf(x)", "normal(x, 1)"])
def test_derivative_of_unsupported_math(text: str) -> None:
    """The derivative of a math with an unknown function is unknown."""
    assert derivative(parse(text), "x") is None


def test_derivative_of_unsupported_math_without_the_variable() -> None:
    """The derivative of a call of a function definition is unknown, even without x."""
    assert derivative(parse("f(y) + x"), "x") is None
