"""The release notes page of the documentation.

`release-notes/<version>.md` in the repository root is the only place where a
release is described: the file is the body of the GitHub release of that
version. This module writes `docs/release-notes.md` from these files, the
versions newest first, so that the documentation site tells the same story.

Run `uv run python -m scripts.releasenotes` after a change of the release notes
and commit the page; `bump-my-version` runs it as a hook of the version bump.
`uv run python -m scripts.releasenotes --check` fails
when the committed page is stale, when the version of the package has no
release notes or when a file of `release-notes/` is not named after a version.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

NOTES_DIR = Path("release-notes")
PAGE_PATH = Path("docs/release-notes.md")
VERSION_PATH = Path("src/sbmlode/__init__.py")

RELEASES_URL = "https://github.com/matthiaskoenig/sbmlode/releases"

# the notes of a release end with the greeting of the GitHub release, which the
# page of all releases does not repeat under every version
SIGN_OFF = "Your sbmlode team"

_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_HEADING = re.compile(r"^(#+) ")
_FENCE = re.compile(r"^\s*(```|~~~)")


class ReleaseNotesError(Exception):
    """The release notes are incomplete or the generated page is not current."""


def _version_key(version: str) -> tuple[int, int, int]:
    """The numbers of a version, by which the releases are ordered.

    Args:
        version: a version `major.minor.patch`.

    Returns:
        The three numbers of the version.

    Raises:
        ReleaseNotesError: when the text is not such a version.
    """
    match = _VERSION.fullmatch(version)
    if match is None:
        raise ReleaseNotesError(
            f"{NOTES_DIR / version}.md is not named after a version major.minor.patch"
        )
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def versions(root: Path) -> list[str]:
    """The versions which have release notes, the newest first.

    Args:
        root: the repository root.

    Returns:
        The versions, one per file of the release notes directory.
    """
    names = [path.stem for path in (root / NOTES_DIR).glob("*.md")]
    return sorted(names, key=_version_key, reverse=True)


def package_version(root: Path) -> str:
    """The version of the package, read without importing it.

    Args:
        root: the repository root.

    Returns:
        The version `__init__.py` of the package states.

    Raises:
        ReleaseNotesError: when the module states no version.
    """
    text = (root / VERSION_PATH).read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', text, flags=re.MULTILINE)
    if match is None:
        raise ReleaseNotesError(f"{VERSION_PATH} states no __version__")
    return match.group(1)


def section(version: str, notes: str) -> str:
    """The notes of one release as a section of the page.

    The title of the file gives way to the heading of the version, every other
    heading moves one level down below it, and the greeting at the end of the
    notes is left out. A `#` inside a fenced code block is not a heading.

    Args:
        version: the version of the release.
        notes: the content of its release notes file.

    Returns:
        The markdown of the section, without a trailing newline.
    """
    lines: list[str] = []
    fenced = False
    for line in notes.strip().splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        if not fenced and line.strip() == SIGN_OFF:
            continue
        heading = None if fenced else _HEADING.match(line)
        if heading is None:
            lines.append(line)
        elif len(heading.group(1)) > 1:
            lines.append(f"#{line}")
        # the only heading of the first level is the title of the file
    body = "\n".join(lines).strip()
    return f"## {version}\n\n{body}"


def page(root: Path) -> str:
    """The release notes page.

    Args:
        root: the repository root.

    Returns:
        The markdown of the page.
    """
    sections = [
        section(version, (root / NOTES_DIR / f"{version}.md").read_text("utf-8"))
        for version in versions(root)
    ]
    intro = (
        "What changed in every version of sbmlode, the newest first. The notes "
        f"of a version are the text of its [release on GitHub]({RELEASES_URL}), "
        "where the source of that version is archived."
    )
    return "\n\n".join(["# Release notes", intro, *sections]) + "\n"


def check(root: Path) -> None:
    """Check that the page is current and that the version has release notes.

    Args:
        root: the repository root.

    Raises:
        ReleaseNotesError: with everything which is wrong.
    """
    problems: list[str] = []
    version = package_version(root)
    if version not in versions(root):
        problems.append(
            f"the version {version} of the package has no release notes, "
            f"write {NOTES_DIR / version}.md"
        )
    path = root / PAGE_PATH
    if not path.is_file() or path.read_text(encoding="utf-8") != page(root):
        problems.append(
            f"{PAGE_PATH} is not current, run `uv run python -m scripts.releasenotes`"
        )
    if problems:
        raise ReleaseNotesError("\n".join(problems))


def main(argv: list[str]) -> int:
    """Write the release notes page, or check that it is current.

    Args:
        argv: the command line, `--check` checks instead of writing.

    Returns:
        The exit code, 1 when the page is stale or the notes are incomplete.
    """
    root = REPO_ROOT
    try:
        if "--check" in argv[1:]:
            check(root)
            print("the release notes page is current")
            return 0
        (root / PAGE_PATH).write_text(page(root), encoding="utf-8")
    except ReleaseNotesError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"written {PAGE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
