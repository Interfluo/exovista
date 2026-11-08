"""Every Python code block in README.md runs, so the documentation cannot drift.

The blocks run in order in one namespace and one scratch directory, like a
reader pasting them into a single session. A block right after an HTML comment
``<!-- readme-test: skip (...) -->`` is not run (used only for signatures).
Warnings raised by a block are errors, so the README never shows code that
warns.

Blocks that call ``mesh.save("x.exo")`` need an installed exovista (its
``pyvista.writers`` entry points) and are skipped in a source tree that is not
installed.
"""

import glob
import importlib.util
import os
import re
import shutil
import warnings
from importlib.metadata import entry_points

import pytest

pv = pytest.importorskip("pyvista")

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
README = os.path.join(ROOT, "README.md")
DATA = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

_FENCE = re.compile(r"^```(\w*)\s*$")
_SKIP = re.compile(r"<!--\s*readme-test:\s*skip\b")
_SAVE_EXODUS = re.compile(r"\.save\(\s*['\"][^'\"]+\.(?:exo|e|ex2|exii|gen)['\"]")


def _python_blocks(path):
    """``(first line number, source, skipped)`` of every ```python block."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i])
        if not match or not match.group(1):
            i += 1
            continue
        lang, start = match.group(1), i + 1
        j = start
        while j < len(lines) and not lines[j].startswith("```"):
            j += 1
        if lang == "python":
            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            skipped = k >= 0 and bool(_SKIP.search(lines[k]))
            blocks.append((start + 1, "\n".join(lines[start:j]) + "\n", skipped))
        i = j + 1
    return blocks


BLOCKS = _python_blocks(README)


def _has_save_plugin():
    if not hasattr(pv, "registered_writers"):  # PyVista < 0.48 (unsupported)
        return False
    names = {ep.name for ep in entry_points(group="pyvista.writers")}
    return ".exo" in names


def _has_legacy():
    return importlib.util.find_spec("exovista.legacy") is not None


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Shared namespace and working directory of the README session."""
    workdir = tmp_path_factory.mktemp("readme")
    # The parallel-read example reads "run.exo.4.*".
    for part in sorted(glob.glob(os.path.join(DATA, "edges.exo.4.*"))):
        suffix = os.path.basename(part)[len("edges"):]
        shutil.copy(part, workdir / f"run{suffix}")
    # Import-time warnings of the dependencies (e.g. numpy ABI notices) are
    # not the README's business: import them before warnings become errors.
    import exovista  # noqa: F401
    import netCDF4  # noqa: F401

    state = {"namespace": {"__name__": "__readme__"}, "failed": None}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        yield state
    finally:
        os.chdir(cwd)


def test_readme_has_python_blocks():
    runnable = [b for b in BLOCKS if not b[2]]
    assert len(runnable) >= 10


@pytest.mark.parametrize(
    "lineno, source, skipped", BLOCKS, ids=[f"line{b[0]}" for b in BLOCKS]
)
def test_readme_block(session, lineno, source, skipped):
    if skipped:
        pytest.skip("marked readme-test: skip")
    if session["failed"] is not None:
        pytest.skip(f"an earlier README block failed (line {session['failed']})")
    if _SAVE_EXODUS.search(source) and not _has_save_plugin():
        pytest.skip("mesh.save('x.exo') needs exovista installed (pip install -e .)")
    if "exovista.legacy" in source and not _has_legacy():
        pytest.skip("exovista.legacy is not available")

    session["failed"] = lineno
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # third-party deprecations are not the README's business
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        code = compile(source, f"README.md:{lineno}", "exec")
        exec(code, session["namespace"])
    session["failed"] = None
