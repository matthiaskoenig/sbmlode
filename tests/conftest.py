"""The distribution of the tests over the workers of pytest-xdist."""

import re

import pytest

#: the module scoped fixtures which run the julia or R of all tests of a module in
#: one process per language on the first request; every worker of pytest-xdist which
#: ran one of their tests would start that process again
BATCHED: frozenset[str] = frozenset({"evaluate", "julia", "r", "jobs", "suite"})

#: the language of a test by its name, e.g. `test_r_curated` or `test_evaluation[julia-x]`
LANGUAGE = re.compile(r"(?:^test_|[\[_-])(julia|r)(?:[\]_-]|$)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep the tests of a batched fixture and a language on one worker.

    `--dist loadgroup` of tox.ini runs the tests of an `xdist_group` on one worker,
    all other tests, also those of a batched fixture in python or JAX, are
    distributed as they come.
    """
    for item in items:
        batched = BATCHED.intersection(getattr(item, "fixturenames", ()))
        language = LANGUAGE.search(item.name)
        if batched and language:
            module = item.nodeid.split("::")[0]
            group = f"{module}::{min(batched)}::{language.group(1)}"
            item.add_marker(pytest.mark.xdist_group(group))
