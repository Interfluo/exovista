"""Every Python code block in README.md runs, so the documentation cannot drift.

The blocks run in order in one namespace and one scratch directory, like a
reader pasting them into a single session. A block right after an HTML comment
``<!-- readme-test: skip (...) -->`` is not run (used only for signatures).
Warnings raised by a block are errors, so the README never shows code that
warns.

The documented output of every ``print(...)`` is checked: the comment at the
end of the print statement (or on the line right after it) must contain what
was printed, whitespace normalised, so a comment may add prose around the
value (``# (27, 3): components recombined``). A comment that holds no output
is marked ``(readme-test: prose)``.

Blocks that call ``mesh.save("x.exo")`` need an installed exovista (its
``pyvista.writers`` entry points) and are skipped in a source tree that is not
installed. The README is written for current PyVista (``extract_surface(
algorithm=None)`` needs 0.47, the documented reader output 0.49): on older
PyVista the blocks are skipped (exovista itself supports PyVista 0.42).
"""

import ast
import builtins
import glob
import importlib.util
import io
import os
import re
import shutil
import sys
import tokenize
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

_PROSE = "(readme-test: prose)"


def _documented_outputs(source):
    """``[(first line, last line, comment)]`` of every ``print(...)`` call.

    The comment is the one at the end of the call's last line, or else a
    line holding only a comment right after it ("" when there is none).
    """
    comments = {}
    code_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            comments[tok.start[0]] = tok.string.lstrip("#").strip()
        elif tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.ENDMARKER):
            code_lines.add(tok.start[0])
    only_comment = set(comments) - code_lines
    calls = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            last = node.end_lineno
            comment = comments.get(last, "")
            if not comment and last + 1 in only_comment:
                comment = comments[last + 1]
            calls.append((node.lineno, last, comment))
    return calls


def _normalise(text):
    return " ".join(str(text).split())


def _check_outputs(source, printed):
    """Every printed text appears in its documented comment."""
    calls = _documented_outputs(source)
    outputs = {}
    for line, text in printed:
        for first, last, comment in calls:
            if first <= line <= last:
                outputs.setdefault((first, comment), []).append(text)
                break
    problems = []
    for (first, comment), texts in outputs.items():
        if _PROSE in comment:
            continue
        text = _normalise(" ".join(texts))
        pattern = r"(?<![\w.])" + re.escape(text) + r"(?!\w)"
        if not comment or not re.search(pattern, _normalise(comment)):
            problems.append(
                f"block line {first}: printed {text!r}, documented {comment!r}"
            )
    assert not problems, "README output differs:\n" + "\n".join(problems)


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


def test_documented_outputs_are_found():
    calls = _documented_outputs(
        "print(1)  # 1\nx = [2]\nprint(x,\n      3)\n# [2] 3\nprint(4)\n"
    )
    assert calls == [(1, 1, "1"), (3, 4, "[2] 3"), (6, 6, "")]
    _check_outputs("print(5)  # five: 5 items\n", [(1, "5")])
    with pytest.raises(AssertionError, match="printed '6'"):
        _check_outputs("print(6)  # 66\n", [(1, "6")])


@pytest.mark.requires_pyvista("0.49", "the README snippets (written for current "
                              "PyVista) and their documented outputs")
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
    filename = f"README.md:{lineno}"
    printed = []

    def recording_print(*args, **kwargs):
        frame = sys._getframe(1)
        if frame.f_code.co_filename == filename:
            buffer = io.StringIO()
            builtins.print(*args, **{**kwargs, "file": buffer})
            printed.append((frame.f_lineno, buffer.getvalue()))
        builtins.print(*args, **kwargs)

    namespace = session["namespace"]
    namespace["print"] = recording_print
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # third-party deprecations are not the README's business
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        code = compile(source, filename, "exec")
        exec(code, namespace)
    _check_outputs(source, printed)
    session["failed"] = None
