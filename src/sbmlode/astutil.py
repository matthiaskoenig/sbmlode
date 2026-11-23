"""Building and reading the libsbml math of the ODE export.

The analysis builds the math of the system from nodes it owns. A node which is added
to another one is owned by it (libsbml disowns the python object), so a node is added
to one parent only and a node of another math is deep copied first.

The math is built to be read: a sum is written with the signs of its terms, `a - b`
rather than `a + -b`, its first term without a sign if it is positive, a negative
product carries the sign on its first factor, `-2 * J` and `-f * J`, and terms which
are the number 0 are dropped (`signed_sum`, `drop_zero_terms`).

`derivative` differentiates a math symbolically, for the rate of a size given by an
assignment rule (the chain rule of the analysis); its result is simplified the same
way, without the factors 1 and the terms 0 of the rules of differentiation.

libsbml reads the flat n-ary operators of a MathML document as chains of binary
nodes: `<apply><plus/>` with 1,000 `<ci>` children is math 1,000 levels deep whose
root has two children. A walk of the math is therefore written with a stack of its
own (`walk`, `negated`, `drop_zero_terms`), not with the stack of the interpreter,
and `check_depth` refuses math which is nested deeper than `MAX_DEPTH` before the
walks which stay recursive (`derivative`) read it.
"""

import math
from collections.abc import Callable, Iterator, Sequence
from typing import NamedTuple

import libsbml

__all__ = [
    "MAX_DEPTH",
    "TIME",
    "UnsupportedMathError",
    "check_depth",
    "depth",
    "derivative",
    "drop_zero_terms",
    "is_number",
    "multiply",
    "name",
    "negated",
    "node",
    "number",
    "product",
    "signed_sum",
    "walk",
]

_NUMBERS = frozenset(
    {libsbml.AST_INTEGER, libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL}
)


class UnsupportedMathError(NotImplementedError):
    """Math which this library cannot read or write.

    Raised by the engine of the printers with a message which names the math and the
    printer, for a construct a dialect cannot express and for an identifier without a
    symbol, and by `check_depth` for math which is nested deeper than the recursive
    walks of the library read.
    """


#: the levels of nesting the recursive walks of the library read, see `check_depth`
#:
#: A walk which recurses spends a few frames of the interpreter per level of the
#: math, three in the printers (`_print`, `_dispatch` and the printing of an
#: operand) and one in `derivative`, and CPython allows a thread 1,000 frames
#: (`sys.getrecursionlimit()`) of which the caller has already spent some. Measured
#: on CPython 3.15 from an empty stack, the printers read 330 levels of nested
#: functions and 493 of nested piecewise, `derivative` 990 levels; the limit is a
#: third of the smallest of them, so that math which passes it is written within
#: a third of the frames of the interpreter, whatever the caller has spent and
#: however the walks are written. The nesting of real math is far below it: the
#: chains libsbml reads a flat n-ary sum or product of a document as are walked
#: with a stack and cost one level (`printers.base`), and the other n-ary
#: operators, `and`, `or`, `piecewise` and the relations, it keeps flat.
MAX_DEPTH = 100


def depth(ast: libsbml.ASTNode) -> int:
    """The levels of nesting of math, measured without recursion.

    Args:
        ast: the math

    Returns:
        the number of the nodes of the longest path from the math down to a leaf,
        `1` for a number or a name
    """
    levels = 0
    level = [ast]
    while level:
        levels += 1
        below: list[libsbml.ASTNode] = []
        for current in level:
            below.extend(current.getChild(k) for k in range(current.getNumChildren()))
        level = below
    return levels


def check_depth(ast: libsbml.ASTNode, element: str = "") -> None:
    """Check that math is not nested deeper than the recursive walks read.

    In front of a walk of the math which recurses per level, so that math which is
    nested too deeply is refused with a message of its own rather than read until
    the stack of the interpreter is exhausted.

    Args:
        ast: the math
        element: the id of the element the math belongs to, named in the message

    Raises:
        UnsupportedMathError: if the math is nested deeper than `MAX_DEPTH`
    """
    levels = depth(ast)
    if levels <= MAX_DEPTH:
        return
    subject = f"The math of '{element}'" if element else "The math"
    raise UnsupportedMathError(
        f"{subject} is nested {levels} levels deep, more than the {MAX_DEPTH} levels "
        f"sbmlode reads. Math nested this deeply is written by a tool rather than by "
        f"hand: write the inner expressions as assignment rules of their own and "
        f"refer to them by their id."
    )


def node(ast_type: int, *children: libsbml.ASTNode) -> libsbml.ASTNode:
    """A node of the math with the given children, which it takes ownership of."""
    ast = libsbml.ASTNode(ast_type)
    for child in children:
        ast.addChild(child)
    return ast


def name(sid: str) -> libsbml.ASTNode:
    """The name of an id."""
    ast = libsbml.ASTNode(libsbml.AST_NAME)
    ast.setName(sid)
    return ast


def number(value: float | None) -> libsbml.ASTNode:
    """A real number, `NaN` for a value which is not set."""
    ast = libsbml.ASTNode(libsbml.AST_REAL)
    ast.setValue(math.nan if value is None else float(value))
    return ast


def is_number(ast: libsbml.ASTNode, value: float | None = None) -> bool:
    """Check that the math is a number, the given one if a value is given."""
    if ast.getType() not in _NUMBERS:
        return False
    return value is None or ast.getValue() == value


def walk(ast: libsbml.ASTNode) -> Iterator[libsbml.ASTNode]:
    """Every node of the math in prefix order, a node before its children."""
    stack = [ast]
    while stack:
        current = stack.pop()
        yield current
        count = current.getNumChildren()
        stack.extend(current.getChild(k) for k in reversed(range(count)))


def _children(ast: libsbml.ASTNode) -> list[libsbml.ASTNode]:
    """Copies of the children of a node."""
    return [ast.getChild(k).deepCopy() for k in range(ast.getNumChildren())]


def negated(ast: libsbml.ASTNode) -> libsbml.ASTNode:
    """The negation of the math, which it takes ownership of.

    A number changes its sign, `-x` is `x`, `a - b` is `b - a`, a product or quotient
    negates its first factor, `-2 * J`, `-f * J`, anything else is `-x`. The chain of
    the products and quotients which carry the sign on their first factor is walked
    with a stack, so that a long product is negated without recursion.
    """
    # the products and quotients whose first factor carries the sign, and the copies
    # of their factors after the first; the sign is put on the first factor of the
    # innermost one, which is the only one of them the negation keeps
    chain: list[tuple[int, list[libsbml.ASTNode]]] = []
    current = ast
    while current.getType() in {libsbml.AST_TIMES, libsbml.AST_DIVIDE}:
        count = current.getNumChildren()
        if count <= 1:
            break
        others = [current.getChild(k).deepCopy() for k in range(1, count)]
        chain.append((current.getType(), others))
        current = current.getChild(0)
    if chain:
        # the innermost first factor is part of the math, the sign goes on a copy
        current = current.deepCopy()
    negation = _negated_term(current)
    for ast_type, others in reversed(chain):
        negation = node(ast_type, negation, *others)
    return negation


def _negated_term(ast: libsbml.ASTNode) -> libsbml.ASTNode:
    """The negation of math which is no product or quotient, see `negated`."""
    ast_type = ast.getType()
    if ast_type in _NUMBERS:
        return number(-ast.getValue())
    children = _children(ast)
    if ast_type == libsbml.AST_MINUS and len(children) == 1:
        return children[0]
    if ast_type == libsbml.AST_MINUS and len(children) == 2:
        return node(libsbml.AST_MINUS, children[1], children[0])
    return node(libsbml.AST_MINUS, ast)


def product(factors: Sequence[libsbml.ASTNode]) -> libsbml.ASTNode:
    """The product of the factors, the factor itself if it is the only one."""
    return factors[0] if len(factors) == 1 else node(libsbml.AST_TIMES, *factors)


def signed_sum(terms: Sequence[tuple[int, libsbml.ASTNode]]) -> libsbml.ASTNode:
    """The sum of signed terms, written with their signs.

    A positive term is added, a negative one subtracted, the first one is negated if
    it is negative; a term which is the number 0 is dropped, the sum of no term is 0.
    The positive terms which follow each other are one n-ary sum, so the depth of the
    math grows with the changes of sign only.

    Args:
        terms: the sign (`1` or `-1`) and the magnitude of each term, which the sum
            takes ownership of

    Returns:
        the sum
    """
    kept = [(sign, term) for sign, term in terms if not is_number(term, 0.0)]
    if not kept:
        return number(0.0)
    sign, first = kept[0]
    total = first if sign > 0 else negated(first)
    open_sum = False
    for sign, term in kept[1:]:
        if sign > 0 and open_sum:
            total.addChild(term)
        elif sign > 0:
            total, open_sum = node(libsbml.AST_PLUS, total, term), True
        else:
            total, open_sum = node(libsbml.AST_MINUS, total, term), False
    return total


class _Rewriting(NamedTuple):
    """A node whose children are being rewritten by `drop_zero_terms`.

    Attributes:
        ast: the node, which the rewriting owns
        left: the copies of the children which are left to rewrite, in reverse order,
            since the stack reads them from its end
        rewritten: the children which are rewritten, in their order
    """

    ast: libsbml.ASTNode
    left: list[libsbml.ASTNode]
    rewritten: list[libsbml.ASTNode]


def drop_zero_terms(ast: libsbml.ASTNode) -> libsbml.ASTNode:
    """The math with the terms of its sums which are the number 0 dropped.

    Every sum and difference is rewritten by `signed_sum`, e.g. after a `rateOf` of
    a constant was replaced by 0. The math is walked with a stack of its own, a node
    after its children, so that a long sum is rewritten without recursion.

    Args:
        ast: the math, which it takes ownership of

    Returns:
        the math without zero terms
    """
    # the nodes whose children are being rewritten, the deepest last
    stack = [_Rewriting(ast, list(reversed(_children(ast))), [])]
    while True:
        current = stack[-1]
        if current.left:
            child = current.left.pop()
            stack.append(_Rewriting(child, list(reversed(_children(child))), []))
            continue
        stack.pop()
        rewritten = _without_zero_terms(current.ast, current.rewritten)
        if not stack:
            return rewritten
        stack[-1].rewritten.append(rewritten)


def _without_zero_terms(
    ast: libsbml.ASTNode, children: list[libsbml.ASTNode]
) -> libsbml.ASTNode:
    """The node of a math whose children are rewritten, see `drop_zero_terms`.

    Args:
        ast: the node, which it takes ownership of
        children: the rewritten children, which it takes ownership of

    Returns:
        the node without the zero terms of its sum
    """
    ast_type = ast.getType()
    signs: list[int] = []
    if ast_type == libsbml.AST_PLUS and children:
        signs = [1] * len(children)
    elif ast_type == libsbml.AST_MINUS and len(children) in {1, 2}:
        signs = [1, -1] if len(children) == 2 else [-1]
    if signs:
        terms = []
        for sign, child in zip(signs, children, strict=True):
            # a negation is a negative term, `a + -b` is `a - b`
            negation = child.getType() == libsbml.AST_MINUS
            if negation and child.getNumChildren() == 1:
                sign, child = -sign, child.getChild(0).deepCopy()
            terms.append((sign, child))
        return signed_sum(terms)
    for k, child in enumerate(children):
        ast.replaceChild(k, child, True)
    return ast


# --- the derivative -------------------------------------------------------------------

#: the variable of `derivative` which stands for the csymbol time, no SId
TIME = "\x00time"


def _copy(ast: libsbml.ASTNode, k: int) -> libsbml.ASTNode:
    """A copy of a child of a node."""
    return ast.getChild(k).deepCopy()


def multiply(*factors: libsbml.ASTNode) -> libsbml.ASTNode:
    """The simplified product of the factors, which it takes ownership of.

    For the derivative of a math and the chain rule of the analysis.

    The factors of a product among them are flattened into the product, the numbers
    are multiplied into one coefficient in front, `0` if a factor is 0; the factor 1
    is dropped and a coefficient -1 negates the product.
    """
    flat: list[libsbml.ASTNode] = []
    coefficient = 1.0
    stack = list(reversed(factors))
    while stack:
        factor = stack.pop()
        if factor.getType() == libsbml.AST_TIMES:
            stack.extend(reversed(_children(factor)))
        elif is_number(factor):
            coefficient *= factor.getValue()
        else:
            flat.append(factor)
    if coefficient == 0.0:
        return number(0.0)
    if not flat:
        return number(coefficient)
    if coefficient == 1.0:
        return product(flat)
    if coefficient == -1.0:
        return negated(product(flat))
    return product([number(coefficient), *flat])


def _div(numerator: libsbml.ASTNode, denominator: libsbml.ASTNode) -> libsbml.ASTNode:
    """The simplified quotient, `0` for a numerator 0, the numerator for a denominator 1."""
    if is_number(numerator, 0.0) or is_number(denominator, 1.0):
        return numerator
    return node(libsbml.AST_DIVIDE, numerator, denominator)


def _pow(base: libsbml.ASTNode, exponent: libsbml.ASTNode) -> libsbml.ASTNode:
    """The simplified power, the base for the exponent 1, `1` for the exponent 0."""
    if is_number(exponent, 1.0):
        return base
    if is_number(exponent, 0.0):
        return number(1.0)
    return node(libsbml.AST_POWER, base, exponent)


def _call(ast_type: int, *arguments: libsbml.ASTNode) -> libsbml.ASTNode:
    """A call of a function of MathML, e.g. `cos(u)`."""
    return node(ast_type, *arguments)


def _sqrt(ast: libsbml.ASTNode) -> libsbml.ASTNode:
    """The square root of a math, a root of degree 2 as libsbml reads `sqrt`."""
    return node(libsbml.AST_FUNCTION_ROOT, number(2.0), ast)


def _square(ast: libsbml.ASTNode) -> libsbml.ASTNode:
    """The square of a math."""
    return _pow(ast, number(2.0))


def _sum(*terms: tuple[int, libsbml.ASTNode]) -> libsbml.ASTNode:
    """The signed sum, see `signed_sum`, its terms 0 dropped."""
    return signed_sum(list(terms))


# d f(u) / du for the functions of one argument, from the argument `u` (a copy each
# call); the chain rule multiplies it with du/dx
_OUTER: dict[int, Callable[[Callable[[], libsbml.ASTNode]], libsbml.ASTNode]] = {
    libsbml.AST_FUNCTION_EXP: lambda u: _call(libsbml.AST_FUNCTION_EXP, u()),
    libsbml.AST_FUNCTION_LN: lambda u: _div(number(1.0), u()),
    libsbml.AST_FUNCTION_SIN: lambda u: _call(libsbml.AST_FUNCTION_COS, u()),
    libsbml.AST_FUNCTION_COS: lambda u: negated(_call(libsbml.AST_FUNCTION_SIN, u())),
    libsbml.AST_FUNCTION_TAN: lambda u: _square(_call(libsbml.AST_FUNCTION_SEC, u())),
    libsbml.AST_FUNCTION_SEC: lambda u: multiply(
        _call(libsbml.AST_FUNCTION_SEC, u()), _call(libsbml.AST_FUNCTION_TAN, u())
    ),
    libsbml.AST_FUNCTION_CSC: lambda u: negated(
        multiply(
            _call(libsbml.AST_FUNCTION_CSC, u()), _call(libsbml.AST_FUNCTION_COT, u())
        )
    ),
    libsbml.AST_FUNCTION_COT: lambda u: negated(
        _square(_call(libsbml.AST_FUNCTION_CSC, u()))
    ),
    libsbml.AST_FUNCTION_ARCSIN: lambda u: _div(
        number(1.0),
        _sqrt(_sum((1, number(1.0)), (-1, _square(u())))),
    ),
    libsbml.AST_FUNCTION_ARCCOS: lambda u: negated(
        _div(
            number(1.0),
            _sqrt(_sum((1, number(1.0)), (-1, _square(u())))),
        )
    ),
    libsbml.AST_FUNCTION_ARCTAN: lambda u: _div(
        number(1.0), _sum((1, number(1.0)), (1, _square(u())))
    ),
    libsbml.AST_FUNCTION_SINH: lambda u: _call(libsbml.AST_FUNCTION_COSH, u()),
    libsbml.AST_FUNCTION_COSH: lambda u: _call(libsbml.AST_FUNCTION_SINH, u()),
    libsbml.AST_FUNCTION_TANH: lambda u: _div(
        number(1.0), _square(_call(libsbml.AST_FUNCTION_COSH, u()))
    ),
    libsbml.AST_FUNCTION_ARCSINH: lambda u: _div(
        number(1.0),
        _sqrt(_sum((1, _square(u())), (1, number(1.0)))),
    ),
    libsbml.AST_FUNCTION_ARCCOSH: lambda u: _div(
        number(1.0),
        _sqrt(_sum((1, _square(u())), (-1, number(1.0)))),
    ),
    libsbml.AST_FUNCTION_ARCTANH: lambda u: _div(
        number(1.0), _sum((1, number(1.0)), (-1, _square(u())))
    ),
}

# the math whose derivative is 0: numbers, constants, functions which are constant
# almost everywhere, and conditions, which are values only as the argument of a
# piecewise
_CONSTANT = (
    _NUMBERS
    | {
        libsbml.AST_CONSTANT_E,
        libsbml.AST_CONSTANT_PI,
        libsbml.AST_CONSTANT_TRUE,
        libsbml.AST_CONSTANT_FALSE,
        libsbml.AST_NAME_AVOGADRO,
        libsbml.AST_FUNCTION_CEILING,
        libsbml.AST_FUNCTION_FLOOR,
    }
    | set(range(libsbml.AST_LOGICAL_AND, libsbml.AST_LOGICAL_XOR + 1))
    | set(range(libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ + 1))
)


def derivative(ast: libsbml.ASTNode, variable: str) -> libsbml.ASTNode | None:
    """The derivative of a math by a variable, simplified, `None` if it is unknown.

    The rules of differentiation of sums, products, quotients, powers (with a
    symbolic exponent, `d u^v = u^v (v' ln(u) + v u'/u)`), roots, logarithms, the
    exponential, the absolute value, the trigonometric and hyperbolic functions and
    their inverses; a `piecewise` is differentiated piece by piece with its
    conditions, `ceiling` and `floor` have the derivative 0. The derivative of a
    call of a function definition, `delay`, `rateOf`, a distrib function, a csymbol
    of a package and any other function is unknown, whether the variable is in its
    arguments or not, and so is the derivative of a math which holds one.

    The rules of differentiation are read recursively, a level of the math per call,
    so the math is refused above `MAX_DEPTH` levels (`check_depth`).

    Args:
        ast: the math, which is not changed
        variable: an id, or `TIME` for the csymbol time

    Returns:
        a new math, `None` if the derivative is unknown

    Raises:
        UnsupportedMathError: if the math is nested deeper than `MAX_DEPTH`
    """
    check_depth(ast)
    return _derivative(ast, variable)


def _derivative(ast: libsbml.ASTNode, variable: str) -> libsbml.ASTNode | None:
    """The derivative of math whose depth is checked, see `derivative`."""
    ast_type = ast.getType()
    if ast_type == libsbml.AST_NAME:
        return number(1.0 if ast.getName() == variable else 0.0)
    if ast_type == libsbml.AST_NAME_TIME:
        return number(1.0 if variable == TIME else 0.0)
    if ast_type in _CONSTANT:
        return number(0.0)
    count = ast.getNumChildren()
    if ast_type == libsbml.AST_FUNCTION_PIECEWISE:
        return _piecewise(ast, variable)
    rates = [_derivative(ast.getChild(k), variable) for k in range(count)]
    if any(rate is None for rate in rates):
        return None
    d = [rate for rate in rates if rate is not None]
    if ast_type == libsbml.AST_PLUS:
        return signed_sum([(1, rate) for rate in d])
    if ast_type == libsbml.AST_MINUS and count == 1:
        return signed_sum([(-1, d[0])])
    if ast_type == libsbml.AST_MINUS and count == 2:
        return signed_sum([(1, d[0]), (-1, d[1])])
    if ast_type == libsbml.AST_TIMES:
        # the product rule, the derivative of a factor in the place of the factor
        terms = []
        for k, rate in enumerate(d):
            if not is_number(rate, 0.0):
                others = [_copy(ast, j) for j in range(count)]
                others[k] = rate
                terms.append((1, multiply(*others)))
        return signed_sum(terms)
    if ast_type == libsbml.AST_DIVIDE and count == 2:
        return _quotient(ast, d[0], d[1])
    if ast_type in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER) and count == 2:
        return _power(_copy(ast, 0), _copy(ast, 1), d[0], d[1])
    if ast_type == libsbml.AST_FUNCTION_ROOT and count in (1, 2):
        # a root is the power of the inverse of its degree, sqrt the degree 2
        degree = _copy(ast, 0) if count == 2 else number(2.0)
        degree_rate = d[0] if count == 2 else number(0.0)
        inverse = (
            number(1.0 / degree.getValue())
            if is_number(degree)
            else _div(number(1.0), degree.deepCopy())
        )
        inverse_rate = _quotient_of(number(1.0), degree, number(0.0), degree_rate)
        return _power(_copy(ast, count - 1), inverse, d[-1], inverse_rate)
    if ast_type == libsbml.AST_FUNCTION_LOG and count in (1, 2):
        base = _copy(ast, 0) if count == 2 else number(10.0)
        if count == 2 and not is_number(d[0], 0.0):
            return None
        argument = _copy(ast, count - 1)
        logarithm = _call(libsbml.AST_FUNCTION_LN, base)
        return _div(d[-1], multiply(argument, logarithm))
    if ast_type == libsbml.AST_FUNCTION_ABS and count == 1:
        positive = node(libsbml.AST_RELATIONAL_GT, _copy(ast, 0), number(0.0))
        if is_number(d[0], 0.0):
            return d[0]
        return node(
            libsbml.AST_FUNCTION_PIECEWISE, d[0].deepCopy(), positive, negated(d[0])
        )
    outer = _OUTER.get(ast_type)
    if outer is not None and count == 1:
        if is_number(d[0], 0.0):
            return d[0]
        rate = outer(lambda: _copy(ast, 0))
        if rate.getType() == libsbml.AST_DIVIDE and is_number(rate.getChild(0)):
            # c u' / f(u) rather than u' * (c / f(u))
            return _div(multiply(_copy(rate, 0), d[0]), _copy(rate, 1))
        return multiply(d[0], rate)
    return None


def _piecewise(ast: libsbml.ASTNode, variable: str) -> libsbml.ASTNode | None:
    """The derivative of a piecewise: of each piece, under its condition.

    The depth of the math is checked by `derivative`, see `_derivative`.
    """
    count = ast.getNumChildren()
    children = []
    for k in range(count):
        # the values are the children 0, 2, ... and the last of an odd count
        if k % 2 == 0:
            rate = _derivative(ast.getChild(k), variable)
            if rate is None:
                return None
            children.append(rate)
        else:
            children.append(_copy(ast, k))
    values = [children[k] for k in range(0, count, 2)]
    if all(is_number(value, 0.0) for value in values):
        return number(0.0)
    return node(libsbml.AST_FUNCTION_PIECEWISE, *children)


def _quotient(
    ast: libsbml.ASTNode,
    numerator_rate: libsbml.ASTNode,
    denominator_rate: libsbml.ASTNode,
) -> libsbml.ASTNode:
    """The derivative of a quotient `u/v`."""
    return _quotient_of(_copy(ast, 0), _copy(ast, 1), numerator_rate, denominator_rate)


def _quotient_of(
    u: libsbml.ASTNode,
    v: libsbml.ASTNode,
    u_rate: libsbml.ASTNode,
    v_rate: libsbml.ASTNode,
) -> libsbml.ASTNode:
    """The derivative `(u' v - u v') / v^2` of `u/v`, `u'/v` for a constant `v`."""
    if is_number(v_rate, 0.0):
        return _div(u_rate, v)
    numerator = _sum((1, multiply(u_rate, v.deepCopy())), (-1, multiply(u, v_rate)))
    return _div(numerator, _square(v))


def _power(
    base: libsbml.ASTNode,
    exponent: libsbml.ASTNode,
    base_rate: libsbml.ASTNode,
    exponent_rate: libsbml.ASTNode,
) -> libsbml.ASTNode:
    """The derivative of a power `u^v`.

    `v u^(v-1) u'` for a constant exponent, else `u^v (v' ln(u) + v u'/u)`.
    """
    if is_number(exponent_rate, 0.0):
        if is_number(base_rate, 0.0):
            return number(0.0)
        if is_number(exponent):
            lowered = number(exponent.getValue() - 1.0)
        else:
            lowered = _sum((1, exponent.deepCopy()), (-1, number(1.0)))
        return multiply(exponent, _pow(base, lowered), base_rate)
    power = _pow(base.deepCopy(), exponent.deepCopy())
    logarithm = _call(libsbml.AST_FUNCTION_LN, base.deepCopy())
    inner = _sum(
        (1, multiply(exponent_rate, logarithm)),
        (1, multiply(exponent, _div(base_rate, base))),
    )
    return multiply(power, inner)
