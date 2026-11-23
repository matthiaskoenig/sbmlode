"""Test the math helpers of the ODE export."""

import math

import libsbml
import pytest

from sbmlode.astutil import (
    MAX_DEPTH,
    TIME,
    UnsupportedMathError,
    check_depth,
    depth,
    derivative,
    drop_zero_terms,
    is_number,
    name,
    negated,
    number,
    signed_sum,
    walk,
)
from sbmlode.events import trigger_root


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


# --- wide and deep math ---------------------------------------------------------------

#: the operands of the wide sums and products of the tests, past the depth a
#: recursive walk reads; libsbml reads the flat n-ary sum of a MathML document as a
#: chain of binary nodes, so this is the depth of their math as well
WIDE = 2000


def chain(operator: str, operands: int) -> libsbml.ASTNode:
    """Math of a flat n-ary operator of MathML, which libsbml reads as a chain."""
    ast = libsbml.readMathMLFromString(
        f'<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><{operator}/>'
        f"{'<ci>a</ci>' * operands}</apply></math>"
    )
    assert ast is not None
    assert ast.getNumChildren() == 2, "libsbml no longer reads a chain"
    assert depth(ast) == operands
    return ast


def test_depth() -> None:
    """The depth of math is the longest path from it down to a leaf."""
    assert depth(parse("x")) == 1
    assert depth(parse("x + 1")) == 2
    assert depth(parse("sin(x + 1)")) == 3
    assert depth(parse("x + sin(sin(x))")) == 4


def test_walk_of_a_wide_sum() -> None:
    """Every node of a sum of many terms is visited, however deep it is."""
    assert sum(1 for _ in walk(chain("plus", WIDE))) == 2 * WIDE - 1


def test_negated_of_a_long_product() -> None:
    """A long product is negated on its first factor, without recursion."""
    negation = negated(chain("times", WIDE))
    assert formula(negation) == " * ".join(["-a", *["a"] * (WIDE - 1)])


def test_drop_zero_terms_of_a_wide_sum() -> None:
    """The zero terms of a wide sum are dropped, however deep the sum is."""
    ast = libsbml.readMathMLFromString(
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/>'
        f"{'<ci>a</ci><cn> 0 </cn>' * (WIDE // 2)}</apply></math>"
    )
    assert ast is not None
    assert depth(ast) == WIDE
    assert formula(drop_zero_terms(ast)) == " + ".join(["a"] * (WIDE // 2))


def test_check_depth_names_the_element() -> None:
    """The check of the depth names the element of the math and its depth."""
    levels = MAX_DEPTH + 1
    check_depth(parse("x + sin(y)"), "V")
    with pytest.raises(
        UnsupportedMathError, match=f"The math of 'V' is nested {levels} levels"
    ):
        check_depth(parse("-" * (levels - 1) + "x"), "V")


def test_derivative_of_the_deepest_math() -> None:
    """Math of as many levels as the walks read is differentiated."""
    rate = derivative(parse("-" * (MAX_DEPTH - 1) + "x"), "x")
    assert rate is not None
    assert formula(rate) in {"1", "-1"}


def test_derivative_of_math_which_is_too_deep() -> None:
    """Math of more levels is refused with its depth, never a `RecursionError`."""
    levels = MAX_DEPTH + 1
    ast = parse("-" * (levels - 1) + "x")
    assert depth(ast) == levels
    with pytest.raises(
        UnsupportedMathError, match=f"nested {levels} levels deep.*{MAX_DEPTH} levels"
    ):
        derivative(ast, "x")


def test_trigger_root_of_math_which_is_too_deep() -> None:
    """A trigger of too many levels is refused as the derivative of a math is."""
    levels = MAX_DEPTH + 1
    trigger = parse("x > " + "-" * (levels - 2) + "y")
    assert depth(trigger) == levels
    with pytest.raises(UnsupportedMathError, match="levels deep"):
        trigger_root(trigger)
