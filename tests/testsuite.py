"""The semantic cases of the SBML test suite and the isolated run of a case.

The cases are the release 3.4.0 of the SBML test suite
(https://github.com/sbmlteam/sbml-test-suite), which is downloaded once into the
cache directory (`$XDG_CACHE_HOME/sbmlode`, `~/.cache/sbmlode`), or read from the
directory `SBMLODE_TESTSUITE` names (the directory which holds `semantic/`). Without
either, offline, the tests which read a case skip, see `requires_testsuite`.

A case of the sweep runs in a python process of its own, see `run_case_isolated`:
libroadrunner, which simulates the reference, and the integrators are native code,
and a crash in native code ends one case instead of the whole session.
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

#: the release of the SBML test suite and its semantic cases
TESTSUITE_URL: str = (
    "https://github.com/sbmlteam/sbml-test-suite/releases/download/3.4.0/"
    "semantic_tests.v3.4.0.zip"
)


def _cache_dir() -> Path:
    """The cache directory of sbmlode."""
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "sbmlode"


def _semantic_dir() -> Path:
    """The directory of the semantic cases, downloaded once into the cache.

    The workers of pytest-xdist import this module at the same time: each one
    downloads into a directory of its own, the first rename wins and the others
    discard theirs.

    Returns:
        the directory, which does not exist if the cases could not be downloaded
    """
    given = os.environ.get("SBMLODE_TESTSUITE")
    if given:
        return Path(given) / "semantic"
    root = _cache_dir() / "sbml-test-suite-3.4.0"
    semantic = root / "semantic"
    if semantic.is_dir():
        return semantic
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            archive = Path(tmp) / "semantic.zip"
            urllib.request.urlretrieve(TESTSUITE_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
            try:
                (Path(tmp) / "semantic").rename(semantic)
            except OSError:
                # another process renamed its download first
                if not semantic.is_dir():
                    raise
    except OSError as err:
        logger.warning("The SBML test suite could not be downloaded: %s", err)
    return semantic


#: the semantic cases of the SBML test suite, see the module
SEMANTIC_DIR: Path = _semantic_dir()

requires_testsuite = pytest.mark.skipif(
    not SEMANTIC_DIR.is_dir(),
    reason=f"requires the SBML test suite, which could not be downloaded: {SEMANTIC_DIR}",
)

#: uniform timecourse of the comparison with roadrunner
T_END: float = 10.0
T_STEPS: int = 51


def suite_case(case: str, level_version: str = "l3v2") -> Path:
    """Resolve a semantic test suite case to its SBML path.

    Args:
        case: the five digit case id, e.g. `"00001"`
        level_version: the SBML level and version flavour, e.g. `"l3v2"`

    Returns:
        the path of the SBML file of the case
    """
    return SEMANTIC_DIR / case / f"{case}-sbml-{level_version}.xml"


SWEEP_CASES: list[Path] = sorted(SEMANTIC_DIR.glob("*/*-sbml-l3v2.xml"))


def sbml_case_idfn(sbml_path: Path) -> str:
    """Inject the case name into the test name."""
    return sbml_path.name


#: cases whose model does not simulate the same twice, so a comparison of
#: trajectories says nothing. Found by simulating every case with two or more
#: events 100 times: in each, events with the same or no priority trigger at once,
#: and SBML leaves their order to chance.
NONDETERMINISTIC: dict[str, str] = dict.fromkeys(
    [
        "00952", "00953", "00962", "00964", "00965", "00966", "01466",
        "01588", "01590", "01591", "01599", "01605", "01626", "01627",
    ],
    "events with the same or no priority trigger at once, their order is random",
)  # fmt: skip


class Outcome(StrEnum):
    """How a case which ran in a process of its own ended."""

    #: the code simulates like roadrunner
    PASSED = "passed"
    #: roadrunner simulates the case, the code raises or simulates differently
    FAILED = "failed"
    #: the process crashed in native code
    CRASHED = "crashed in native code"
    #: the process was killed after the timeout
    TIMED_OUT = "timed out"
    #: the model does not simulate the same twice, see `NONDETERMINISTIC`; the
    #: sweep sets it without running the case
    NOT_DETERMINISTIC = "original is not deterministic"
    #: the worker exited without recording an outcome, a bug of the harness
    WORKER_ERROR = "worker error"
    #: the model has a construct the code does not support
    UNSUPPORTED = "unsupported construct"
    #: roadrunner does not simulate the case, so there is no reference
    NO_REFERENCE = "no roadrunner reference"


@dataclass
class CaseResult:
    """The result of a case which ran in a process of its own.

    Attributes:
        outcome: how the case ended
        stage: the stage the case was in when it ended
        detail: the error of a failure, the signal of a crash
    """

    outcome: Outcome
    stage: str
    detail: str


#: the file the worker records its stage and its outcome in
OUTCOME_FILE: str = "outcome.json"


def record(case_dir: Path, stage: str, outcome: Outcome | None, detail: str) -> None:
    """Record the stage and the outcome of a case, in the worker.

    The stage is recorded before it starts, so that the stage a process was killed
    in is known.

    Args:
        case_dir: the directory of the case
        stage: the stage the case is in
        outcome: the outcome, `None` while the case is still running
        detail: the error of a failure
    """
    (case_dir / OUTCOME_FILE).write_text(
        json.dumps({"stage": stage, "outcome": outcome, "detail": detail})
    )


def _crash(returncode: int) -> str | None:
    """Name the crash a process ended with.

    A process which crashed in native code is killed by a signal on POSIX, which
    `subprocess` reports as a negative return code. On Windows it exits with the
    NTSTATUS of the exception, e.g. `0xC0000005` for an access violation, which is
    an error status from `0xC0000000` on.

    Args:
        returncode: the return code of the process

    Returns:
        the signal or the status of the crash, `None` if it did not crash
    """
    if returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return f"signal {-returncode}"
    if sys.platform == "win32" and returncode >= 0xC0000000:
        return f"0x{returncode:08X}"
    return None


def run_case_isolated(
    sbml_path: Path,
    case_dir: Path,
    timeout: float,
    worker: Path,
    arguments: Sequence[str] = (),
) -> CaseResult:
    """Check a case in a python process of its own.

    A crash in native code ends the process of the case, and it is reported as
    `Outcome.CRASHED` with the stage it happened in. A fresh interpreter is started
    rather than a fork, since importing roadrunner starts native threads, which a
    fork does not survive safely.

    Args:
        sbml_path: path of the SBML file of the case
        case_dir: directory of the case, which the outcome is written to
        timeout: seconds after which the process is killed
        worker: the script run as `worker <sbml_path> <case_dir> [arguments]`
        arguments: further arguments of the worker, e.g. the format

    Returns:
        the result of the case
    """
    try:
        process = subprocess.run(
            [sys.executable, str(worker), str(sbml_path), str(case_dir), *arguments],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        process = None

    outcome_path = case_dir / OUTCOME_FILE
    recorded: dict[str, Any] = (
        json.loads(outcome_path.read_text())
        if outcome_path.exists()
        else {"stage": "start the worker", "outcome": None, "detail": ""}
    )
    stage: str = recorded["stage"]
    if process is None:
        return CaseResult(Outcome.TIMED_OUT, stage, f"after {timeout:.0f} s")
    crash = _crash(process.returncode)
    if crash is not None:
        return CaseResult(Outcome.CRASHED, stage, crash)
    if recorded["outcome"] is not None:
        return CaseResult(Outcome(recorded["outcome"]), stage, recorded["detail"])

    stderr = process.stderr.strip().splitlines()
    return CaseResult(
        Outcome.WORKER_ERROR,
        stage,
        f"exit code {process.returncode}: {stderr[-1] if stderr else ''}",
    )


def run_case(
    sbml_path: Path,
    timeout: float,
    tmp_dir: Path,
    worker: Path,
    arguments: Sequence[str] = (),
) -> tuple[str, CaseResult]:
    """Run a single case in a process of its own.

    Args:
        sbml_path: path of the SBML file of the case
        timeout: seconds after which the case is killed
        tmp_dir: the scratch directory, which the directory of the case is made in
        worker: the worker of `run_case_isolated`
        arguments: further arguments of the worker

    Returns:
        the case id and its result
    """
    case = sbml_path.name[:5]
    case_dir = tmp_dir / case
    shutil.rmtree(case_dir, ignore_errors=True)
    if case in NONDETERMINISTIC:
        return case, CaseResult(Outcome.NOT_DETERMINISTIC, "", NONDETERMINISTIC[case])

    case_dir.mkdir(parents=True)
    return case, run_case_isolated(sbml_path, case_dir, timeout, worker, arguments)


def reason_of(detail: str) -> str:
    """The reason of an error, which summarizes many cases.

    The reason is the error without the formula or id it quotes and the C++
    function it names.

    Args:
        detail: the error

    Returns:
        the reason
    """
    return re.sub(r"'[^']*'", "'...'", detail.split(", at ")[0])
