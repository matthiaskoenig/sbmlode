"""Test that the documentation shows the current output of the ODE export.

The page `docs/formats.md` includes the files of `docs/images/ode`, which
`scripts/docs_images.py` writes (`python -m scripts.docs_images
docs/images/ode`). A change of the export which changes them fails here until they
are written again, so the documentation never shows an output the export no longer
writes. The version of sbmlode in the files is not compared, it changes with every
release. The SVG pages of the typst document are not committed, the docs workflow
compiles them; the page list of `docs/formats.md` is checked against a compilation.
"""

import re
from pathlib import Path

import pytest
from resources import REPRESSILATOR_SBML

from scripts.docs_images import ENDINGS, compile_typst, export

DOCS_DIR: Path = Path(__file__).parents[3] / "docs" / "images" / "ode"
"""The files of the ODE export in the documentation."""

pytestmark = pytest.mark.skipif(
    not DOCS_DIR.is_dir(), reason="the documentation is only part of the repository"
)

VERSION = re.compile(r"sbmlode \d+\.\d+\.\d+\S*")
"""The version of sbmlode which wrote a file."""


def _text(path: Path) -> str:
    """The text of a file without the version of sbmlode which wrote it."""
    return VERSION.sub("sbmlode VERSION", path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """The files of the repressilator, written by the example."""
    return export(REPRESSILATOR_SBML, tmp_path_factory.mktemp("ode"), "repressilator")


def test_every_format_is_shown(exported: list[Path]) -> None:
    """The example writes one file per format and the markdown fragment."""
    assert [p.name for p in exported] == [
        *(f"repressilator{ending}" for ending in ENDINGS.values()),
        "repressilator_fragment.md",
    ]


@pytest.mark.parametrize(
    "name",
    [
        *(f"repressilator{ending}" for ending in ENDINGS.values()),
        "repressilator_fragment.md",
    ],
)
def test_docs_are_current(name: str, exported: list[Path]) -> None:
    """The file in the documentation is the one the export writes now."""
    (path,) = (p for p in exported if p.name == name)
    assert _text(DOCS_DIR / name) == _text(path), (
        f"docs/images/ode/{name} is outdated, run "
        "`python -m scripts.docs_images docs/images/ode`"
    )


def test_typst_pages_are_shown(exported: list[Path]) -> None:
    """The documentation shows every page of the compiled typst document.

    The pages are compiled by the docs workflow, not committed, so the page list of
    `docs/formats.md` is checked against a compilation: a link and an image per page.
    """
    pytest.importorskip("typst")
    (typ,) = (p for p in exported if p.suffix == ".typ")
    pages = [p.name for p in compile_typst(typ)]
    assert pages
    page = (DOCS_DIR.parents[1] / "formats.md").read_text(encoding="utf-8")
    shown = re.findall(r"images/ode/(repressilator-\d+\.svg)", page)
    assert shown == [name for name in pages for _ in range(2)]
