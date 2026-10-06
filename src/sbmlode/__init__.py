"""Export of an SBML model as its system of ordinary differential equations.

`OdeSystem.from_sbml` analyses a model into its ODE system, which is rendered in the
formats of `FORMATS`:

```python
from sbmlode import OdeSystem

system = OdeSystem.from_sbml("model.xml")
code = system.render("python", simulator=True)
system.write("model.py")
```

The methods `render`, `write` and `render_template` of `OdeSystem` are the functions of
the same names of this package, which take the system as their first argument.

The math of the model is written by the printers of `sbmlode.printers`,
one per dialect, the text of the model through the helpers of
`sbmlode.text`, the formats by `sbmlode.formats`:
python, julia and R code, and typst, LaTeX and markdown documents
(`sbmlode.documents`), e.g. `system.write("model.typ")`.
"""

# the version comes first, the modules below read it when they render
__version__ = "0.1.0"

from sbmlode.formats import (
    FORMATS,
    Format,
    render,
    render_template,
    write,
)
from sbmlode.system import OdeSystem

__all__: list[str] = [
    "FORMATS",
    "Format",
    "OdeSystem",
    "__version__",
    "render",
    "render_template",
    "write",
]
