# Development

Contributions are welcome. The repository is [matthiaskoenig/sbmlode](https://github.com/matthiaskoenig/sbmlode); development happens against the `develop` branch via pull requests.

## Branch model

Two branches are permanent:

- **`develop`** is the default branch and the branch everything is integrated into. The documentation on [matthiaskoenig.github.io/sbmlode](https://matthiaskoenig.github.io/sbmlode) is published from it.
- **`main`** tracks the latest published release. It is fast-forwarded to the released commit by the `sync-main` job of the `CI-CD` workflow after the package went to PyPI, so `main` and the newest version on PyPI always agree. Nothing is developed on `main` and nothing is merged into it by hand.

Work happens on short lived branches off `develop`, which GitHub deletes after the merge. Releases are tagged on `develop`, see [Release](#release).

## Pull requests

Neither branch accepts a direct push, every change goes through a pull request against `develop`. This includes the maintainer, there is no bypass. A pull request can only be merged once the required checks are green:

| check | workflow | content |
| --- | --- | --- |
| `tests` | `ci-cd.yml` | the test matrix, linux with python 3.12 to 3.15 and the lowest versions of the dependencies, macos and windows with 3.14 |
| `R` | `ci-cd.yml` | the generated R code, run with Rscript and deSolve |
| `latex` | `ci-cd.yml` | the typst and LaTeX documents, compiled with typst and tectonic |
| `ruff` | `ruff.yml` | `ruff check` and `ruff format --check` |
| `ty` | `ty.yml` | `tox r -e ty` |
| `docs` | `docs.yml` | the release notes check and the strict zensical build including the API reference |

`tests` aggregates the test matrix into a single job, so the name of the required check stays the same when the matrix changes. The job `julia` runs the generated julia code; its packages take long to install and precompile, so it runs only for a release tag and on demand (`workflow_dispatch`): it is no required check of a pull request, but the release of a tag waits for it.

Further rules: conversations have to be resolved before the merge, an approval is dismissed when new commits are pushed, the history stays linear (squash or rebase, no merge commits), and the maintainer is the code owner (`.github/CODEOWNERS`).

### Repository policies { #repository-policies }

The protection is implemented with [repository rulesets](https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets) in `.github/rulesets/`, so a change to a policy is reviewed like any other change:

| ruleset | applies to | rules |
| --- | --- | --- |
| `develop.json` | `develop` | pull request required, the checks above, resolved conversations, linear history, no force push, no deletion, no bypass |
| `main.json` | `main` | linear history, no force push, no deletion, no bypass; the fast-forward of the release workflow needs none |
| `tags.json` | all tags | a tag cannot be deleted or moved |
| `tag-creation.json` | all tags | only a repository admin can create a tag, every tag is a release to PyPI |

Changing a policy means changing the json and applying it with `.github/rulesets/apply.sh`, which is idempotent and also sets the merge settings of the repository (auto-merge, delete branch on merge, squash and rebase only). It needs the [github cli](https://cli.github.com) authenticated as an admin of the repository.

## Setup

Development needs [uv](https://docs.astral.sh/uv/) and a checkout of the repository:

```bash
git clone https://github.com/matthiaskoenig/sbmlode.git
cd sbmlode
uv sync --extra dev
uv run pre-commit install
```

The environment is resolved from the committed `uv.lock`; after a change of a dependency in `pyproject.toml` run `uv lock`, the `documentation` workflow syncs with `uv sync --locked`. The tox environments resolve from `pyproject.toml`, whose lower bounds the `lowest` environment verifies. The `dev` extra holds everything used below.

## Testing

```bash
uv run pytest -m "not sbml_testsuite"         # the suite without the sweep over the SBML test suite
uv run pytest -m sbml_testsuite               # the sweep, every case of the SBML test suite
tox r -e py3.14                               # one python version, as continuous integration runs it
tox r -e py3.14 -- tests/test_ode_system.py   # one module in it
tox r -e lowest                               # the lowest versions of the dependencies on python 3.12
```

The tests download the semantic cases of the [SBML test suite](https://github.com/sbmlteam/sbml-test-suite) 3.4.0 once into the cache directory (`$XDG_CACHE_HOME/sbmlode`, else `~/.cache/sbmlode`); `SBMLODE_TESTSUITE` names a directory which holds `semantic/` instead. Offline and without it, the tests which read a case skip. The golden documents of `tests/golden/` are written again with `SBMLODE_UPDATE_GOLDEN=1`.

On Linux the tests run with the standalone interpreters of uv (`UV_PYTHON_PREFERENCE=only-managed`); the extension of libroadrunner links against `libpython3.X.so.1.0`, which these interpreters ship in their `lib` directory, so the workflow puts that directory on `LD_LIBRARY_PATH`. libroadrunner, the reference of the simulations, has no wheels for python 3.15 yet; the tests which need it skip there.

### Toolchains

The julia and R code and the typst and LaTeX documents are run and compiled by the tests. Typst comes with the `typst` python package of the `test` extra. Julia, R and tectonic are no python packages: their tests skip when the toolchain is missing, and three tox environments, pinned to python 3.14 and not part of `envlist`, run them with `SBMLODE_REQUIRE_TOOLCHAINS=1`, which turns a missing toolchain into a failure:

```bash
tox r -e julia    # the generated julia code, julia with the packages of tests/julia/Project.toml
tox r -e R        # the generated R code, Rscript with the package deSolve
tox r -e latex    # the typst and LaTeX documents, tectonic on the path
```

Julia and R are called through the command prefixes in `SBMLODE_JULIA` (default `julia`) and `SBMLODE_RSCRIPT` (default `Rscript`), which can be a docker run instead of a local installation:

```bash
# julia: the packages, once, into the depot ~/.julia-docker
mkdir -p ~/.julia-docker/environments/sbmlode
cp tests/julia/Project.toml ~/.julia-docker/environments/sbmlode/
docker run --rm -v $HOME/.julia-docker:/root/.julia julia:1.11 \
    julia --project=@sbmlode -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
export SBMLODE_JULIA="docker run --rm -v /tmp:/tmp -v $HOME/.julia-docker:/root/.julia -e JULIA_LOAD_PATH=@:@stdlib julia:1.11 julia --project=@sbmlode"

# R: the image with deSolve, once
docker build -t sbmlode-r -f tests/docker/r.Dockerfile tests/docker
export SBMLODE_RSCRIPT="docker run --rm -v /tmp:/tmp sbmlode-r Rscript"
```

`scripts/ode_report.py` runs the export over the whole SBML test suite, every case in a process of its own, and prints the pass rates of the [Verification](formats.md#verification):

```bash
uv run python scripts/ode_report.py
uv run python scripts/ode_report.py --format julia --format r
```

## Linting, formatting and type checking

```bash
uv run ruff check
uv run ruff format
uv run ty check
```

Warnings of ty are errors. Suppress an unavoidable diagnostic with a rule specific `# ty: ignore[rule-name]`. libsbml has no type stubs and creates its objects through SWIG: annotate the libsbml objects and use the explicit getters (`getId()`).

## Documentation

The documentation is built with [Zensical](https://zensical.org/) from `docs/`, configured in `zensical.toml`, and published from `develop` by the `documentation` workflow.

```bash
uv run python -m scripts.docs_images docs/images/ode   # the output of the export the guide shows
uv run python -m scripts.releasenotes                  # docs/release-notes.md from release-notes/
uv run zensical serve                                  # the preview
uv run zensical build --clean --strict                 # the build, a broken link fails
```

`docs/images/ode/` holds the output of the export of the repressilator, which the guide includes; the text files are committed and `tests/test_ode_docs.py` fails when they are no longer the output of the export. The SVG pages of the typst document are compiled by the workflow and are not committed. Math is typeset by MathJax, which is vendored in `docs/javascripts/mathjax/`. The API reference is rendered from the docstrings by mkdocstrings.

## Release

A release is made from `develop`:

1. branch off `develop`: `git switch -c release/x.y.z develop`
2. write the release notes in `release-notes/x.y.z.md`, the title and the logo as in the notes of the earlier releases; they are the body of the GitHub release and a section of the release notes page
3. bump the version: `uv run bump-my-version bump [major|minor|patch]`, which updates `src/sbmlode/__init__.py` and `CITATION.cff`, regenerates `docs/release-notes.md` and commits, without a tag
4. push the branch, open the pull request against `develop` and merge it once the checks are green
5. tag the merged commit on `develop` and push the tag:

    ```bash
    git switch develop && git pull
    git tag x.y.z
    git push origin x.y.z
    ```

    This starts the `CI-CD` workflow, which runs the tests, builds the distributions (`build`), publishes them to [PyPI](https://pypi.org/project/sbmlode/) by trusted publishing (`publish`), creates the GitHub release (`github-release`) and fast-forwards `main` (`sync-main`). A tag cannot be moved or deleted afterwards.

6. once Zenodo has archived the release, cite it through a pull request: the version DOI (listed by `https://zenodo.org/api/records?q=conceptrecid:23179199&all_versions=true&sort=mostrecent`) and `date-released` in `CITATION.cff`, and the version, the month and the DOI of the citation in `README.md` and `docs/index.md`
