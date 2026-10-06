"""The JAX dialect of the math printer.

The expressions use `jax.numpy` as `jnp` and `jax.scipy.special`, the time is `t`.
JAX traces them (`jax.jit`, `jax.vmap`, `jax.grad`), so that no expression decides
on a value with python: a `piecewise` is `jnp.where`, which evaluates every piece
and selects the value of the piece which applies; the value is exact, a piece which
does not apply and is `inf` or `NaN` at the point is discarded, but its gradient is
not (`piecewise(0, x == 0, 1/x)` at `x = 0` has a `NaN` gradient). A condition is a
boolean array, combined with `jnp.logical_and` and its kin (`~True` of python is
`-2`), which is `jnp.where(c, 1.0, 0.0)` where it is used as a number. Every number
is a float literal, `2.0`: a power of integers in `jnp` raises for a negative
exponent and overflows, and every power is `jnp.power`, which is `NaN` for a negative
base and a fractional exponent where python's `**` of two literals is complex.
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import libsbml

from sbmlode.printers.base import Precedence, Printed
from sbmlode.printers.python import PythonPrinter


def _jnp(table: Mapping[int, str]) -> dict[int, str]:
    """The entries of a table of the python dialect which are numpy, as `jnp`."""
    return {key: f"j{code}" for key, code in table.items() if code.startswith("np.")}


class JaxPrinter(PythonPrinter):
    """Printer of SBML math as an expression of `jax.numpy`, which JAX traces."""

    name: ClassVar[str] = "jax"
    MODULES: ClassVar[frozenset[str]] = frozenset({"jnp", "jax"})
    # `max` and `min` of python decide on the values, see `minmax`
    FUNCTIONS: ClassVar[Mapping[int, str]] = _jnp(PythonPrinter.FUNCTIONS)
    CONSTANTS: ClassVar[Mapping[int, str]] = {
        **PythonPrinter.CONSTANTS,
        **_jnp(PythonPrinter.CONSTANTS),
    }
    RECIPROCALS: ClassVar[Mapping[int, str]] = _jnp(PythonPrinter.RECIPROCALS)
    OF_RECIPROCALS: ClassVar[Mapping[int, str]] = _jnp(PythonPrinter.OF_RECIPROCALS)

    def integer(self, value: int) -> str:
        """Float literal of an integer, `2.0`."""
        return self.number(float(value))

    def number(self, value: float) -> str:
        """Python literal of a float, `jnp.inf` and `jnp.nan` included."""
        if math.isnan(value):
            return "jnp.nan"
        if math.isinf(value):
            return "jnp.inf" if value > 0 else "-jnp.inf"
        return repr(float(value))

    def power(self, base: Printed, exponent: Printed) -> Printed:
        """`jnp.power(a, b)`."""
        return self.call("jnp.power", [base, exponent])

    def piecewise(
        self, pieces: Sequence[tuple[Printed, Printed]], otherwise: Printed | None
    ) -> Printed:
        """`jnp.where(c, x, y)`, nested in `y`, `jnp.nan` without an otherwise."""
        code = (
            otherwise if otherwise is not None else self.literal(self.number(math.nan))
        )
        for value, condition in reversed(pieces):
            code = self.call("jnp.where", [condition, value, code])
        return code

    def rem(self, dividend: Printed, divisor: Printed) -> Printed:
        """`jnp.fmod`, which has the sign of the dividend."""
        return self.call("jnp.fmod", [dividend, divisor])

    def quotient(self, dividend: Printed, divisor: Printed) -> Printed:
        """`jnp.trunc(a / b)`."""
        return self.call("jnp.trunc", [self.divide(dividend, divisor)])

    def log(self, base: Printed | None, value: Printed) -> Printed:
        """`jnp.log10(x)` or `jnp.log(x) / jnp.log(b)`."""
        if base is None:
            return self.call("jnp.log10", [value])
        return self.divide(self.call("jnp.log", [value]), self.call("jnp.log", [base]))

    def root(self, degree: Printed | None, value: Printed) -> Printed:
        """`jnp.sqrt(x)` or `jnp.power(x, 1.0 / n)`."""
        if degree is None:
            return self.call("jnp.sqrt", [value])
        return self.power(value, self.divide(self._one(), degree))

    def factorial(self, value: Printed) -> Printed:
        """`jax.scipy.special.gamma(x + 1.0)`, the factorial extended to the reals."""
        return self.call(
            "jax.scipy.special.gamma",
            [self.infix([value, self._integer(1)], self.PLUS, Precedence.SUM)],
        )

    def minmax(self, function: int, arguments: Sequence[Printed]) -> Printed:
        """`jnp.maximum(a, jnp.maximum(b, c))`, `jnp.minimum` alike."""
        name = "jnp.maximum" if function == libsbml.AST_FUNCTION_MAX else "jnp.minimum"
        return self._fold(name, arguments)

    def logic_and(self, operands: Sequence[Printed]) -> Printed:
        """`jnp.logical_and(a, jnp.logical_and(b, c))`."""
        return self._fold("jnp.logical_and", operands)

    def logic_or(self, operands: Sequence[Printed]) -> Printed:
        """`jnp.logical_or(a, jnp.logical_or(b, c))`."""
        return self._fold("jnp.logical_or", operands)

    def logic_xor(self, operands: Sequence[Printed]) -> Printed:
        """`jnp.logical_xor(a, jnp.logical_xor(b, c))`, true for an odd number."""
        return self._fold("jnp.logical_xor", operands)

    def logic_not(self, operand: Printed) -> Printed:
        """`jnp.logical_not(a)`."""
        return self.call("jnp.logical_not", [operand])

    def bool_to_number(self, condition: Printed) -> Printed:
        """`jnp.where(c, 1.0, 0.0)`, so that a condition is a number in arithmetic."""
        return self.call("jnp.where", [condition, self._one(), self._integer(0)])

    def number_to_bool(self, value: Printed, constant: float | None) -> Printed:
        """`x != 0.0`, a condition is a boolean array, `NaN` holds as in C.

        A literal is the constant `True` or `False`.
        """
        if constant is not None:
            truth = (
                libsbml.AST_CONSTANT_FALSE
                if constant == 0
                else libsbml.AST_CONSTANT_TRUE
            )
            return Printed(self.CONSTANTS[truth], Precedence.ATOM)
        return self.relation(
            libsbml.AST_RELATIONAL_NEQ, value, self.literal(self.number(0.0))
        )

    def _fold(self, function: str, operands: Sequence[Printed]) -> Printed:
        """The binary function folded over the operands from the right.

        A single operand is the operand itself.
        """
        code = operands[-1]
        for operand in reversed(operands[:-1]):
            code = self.call(function, [operand, code])
        return code
