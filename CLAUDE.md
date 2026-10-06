# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project

`sbmlode` writes the system of ordinary differential equations of an SBML model as python, julia and R code which simulates it, as typst, LaTeX and markdown documents which describe it, and as typed data (`OdeSystem.typeset`) for an application which lays out the equations itself (SBML4Humans). Pure library, no CLI. Requires python >= 3.11, packaged with hatchling (version read from `src/sbmlode/__init__.py`). Runtime dependencies are `python-libsbml` and `jinja2` only, keep it so: no pint, numpy or sbmlutils in the package. The `simulate` extra (numpy, pandas, scipy) is what the generated python code runs with, the `test` extra what the tests need (antimony, libroadrunner as the reference, typst, markdown-it-py), `dev` everything. It is the ODE export of sbmlutils 0.14 (`sbmlutils.converters.ode`), which sbmlutils re-exports from 0.15 on.

## Commands

```bash
uv sync --extra dev
uv run pre-commit install
uv lock                                       # after every change of a dependency in pyproject.toml

uv run pytest -m "not sbml_testsuite"         # the suite, what continuous integration runs
uv run pytest -m sbml_testsuite               # the sweep over the SBML test suite
tox r -e py3.14 -- tests/test_ode_system.py   # one module in a tox env (py3.11-3.15, lowest, ty)
tox r -e julia | tox r -e R | tox r -e latex  # the toolchains, see docs/development.md
uv run python scripts/ode_report.py           # pass rates over the SBML test suite, `--format julia --format r`
SBMLODE_UPDATE_GOLDEN=1 uv run pytest tests/test_ode_presentation.py -k golden   # rewrite tests/golden/

uv run ruff check && uv run ruff format --check
uv run ty check                               # warnings are errors

uv run python -m scripts.docs_images docs/images/ode   # the output the guide shows, before zensical serve/build
uv run python -m scripts.releasenotes [--check]        # docs/release-notes.md from release-notes/
uv run zensical build --clean --strict
```

The tests download the semantic cases of the SBML test suite 3.4.0 into `$XDG_CACHE_HOME/sbmlode` (`~/.cache/sbmlode`), or read `SBMLODE_TESTSUITE`; offline they skip. Julia and R run through the command prefixes `SBMLODE_JULIA` and `SBMLODE_RSCRIPT` (docker works, see `docs/development.md`); their tests skip without the toolchain and fail with `SBMLODE_REQUIRE_TOOLCHAINS=1`, which the tox environments `julia`, `R` and `latex` set.

`develop` is the default branch and takes every change through a pull request; the rulesets in `.github/rulesets/` (applied with `apply.sh`) require the checks `tests`, `R`, `latex`, `ruff`, `ty` and `docs`, squash or rebase merges only. `main` only tracks the latest release and is fast-forwarded by `sync-main` of `ci-cd.yml`. Releases: write `release-notes/<version>.md`, `uv run bump-my-version bump [major|minor|patch]` (updates `__init__.py` and `CITATION.cff`, regenerates the release notes page, commits without tag), pull request, then the tag on `develop` triggers the PyPI release.

## Architecture

Three layers, described in `docs/design/2026-10-05-ode-export-design.md`:

1. **Analysis** (`analysis.py`, `system.py`, `dependencies.py`, `events.py`): `OdeSystem.from_sbml` reads the document (`io.py`: libsbml only, comp flattened, L1/L2 converted to L3V2) into frozen dataclasses: `Symbol`, `Quantity`, `Assignment`, `Reaction`, `Ode`, `Event`, `OdeSystem`. Math stays a libsbml `ASTNode`. Algebraic rules, `delay()` and fast reactions are recorded in `unsupported`, never dropped in silence. Local parameters are renamed `<reaction>_<id>`, `Symbol.element` keeps what a symbol stands for.
2. **Printers** (`printers/`): one dialect per language on the AST (`MathPrinter` with precedence, `DocumentPrinter` for LaTeX and typst), each takes a `symbols: Mapping[sid, str]`. `symbols.py` makes code identifiers and typeset symbols (`tau_mRNA` as `\tau_{\mathrm{mRNA}}`), `text.py` escapes text and checks every id is an SId.
3. **Formats** (`formats.py`, `documents.py`, `templates/*.jinja`): a jinja2 template per format. `DocumentContext` typesets the system for a document dialect into the dataclasses of `TypesetSystem`, which `OdeSystem.typeset(dialect, symbols, wrap)` returns to an application; `wrap(symbol, typeset)` transforms every symbol, e.g. into a link. The templates read the same dataclasses (jinja falls back from item to attribute access).

`units.py` writes a unit like sbmlutils did with pint, pinned by `tests/data/unit_terms.json`. The tests import `tests/resources.py` (models in `tests/data/models/`), `tests/ode_helpers.py` and `tests/testsuite.py` (the cases of the SBML test suite and the isolated run of a case, the worker of `scripts/ode_report.py`).

## Conventions

- ruff with the rule set of `.ruff.toml` (google docstrings, isort, pyupgrade, bugbear, bandit, lazy logging), every module, class and function annotated and documented; ty with `error-on-warning`, suppress with a rule specific `# ty: ignore[rule]`. libsbml has no stubs: annotate libsbml objects and use the getters.
- `docs/images/ode/` is output of the export, committed and kept current by `tests/test_ode_docs.py`; ruff and the whitespace hooks leave it alone. The typst SVG pages are compiled, not committed.
- Markdown has no hard line wraps; never use the em dash.
