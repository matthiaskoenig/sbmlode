"""Printers of SBML math, one per dialect, on the engine of `base.MathPrinter`."""

from sbmlode.printers.base import (
    MathPrinter,
    Precedence,
    Printed,
    SymbolMap,
    Term,
    UnsupportedMathError,
)
from sbmlode.printers.document import DocumentPrinter
from sbmlode.printers.julia import JuliaPrinter
from sbmlode.printers.latex import LatexPrinter
from sbmlode.printers.python import PythonPrinter
from sbmlode.printers.r import RPrinter
from sbmlode.printers.typst import TypstPrinter

PRINTERS: dict[str, type[MathPrinter]] = {
    PythonPrinter.name: PythonPrinter,
    JuliaPrinter.name: JuliaPrinter,
    RPrinter.name: RPrinter,
    LatexPrinter.name: LatexPrinter,
    TypstPrinter.name: TypstPrinter,
}
"""The printer of each dialect, by its name."""

__all__ = [
    "PRINTERS",
    "DocumentPrinter",
    "JuliaPrinter",
    "LatexPrinter",
    "MathPrinter",
    "Precedence",
    "Printed",
    "PythonPrinter",
    "RPrinter",
    "SymbolMap",
    "Term",
    "TypstPrinter",
    "UnsupportedMathError",
]
