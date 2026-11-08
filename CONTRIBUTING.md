# Contributing to exovista

Thanks for helping! Bug reports, small reproducers, documentation fixes and
pull requests are all welcome. Please be kind and constructive in issues and
reviews.

## Reporting a bug

Open an issue with the **Bug report** form and include:

1. **Versions** — the output of

   ```shell
   python -c "import pyvista; print(pyvista.Report(additional=['exovista', 'netCDF4', 'scipy'], gpu=False))"
   ```

   (or at least the exovista, pyvista, vtk, netCDF4, numpy and Python
   versions and your operating system).
2. **A minimal reproducer** — a short script that builds the mesh with
   PyVista (e.g. `pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()`)
   and calls `write_exo` / `read_exo`, with the full traceback or the wrong
   result next to what you expected.
3. **A minimal `.exo` file** when reading a file goes wrong: the smallest
   file that still shows the problem (zipped, attached to the issue). If you
   cannot share the file, attach its header instead: `ncdump -h model.exo`
   (netCDF tools) or `exovista-read model.exo`.
4. **What VTK says**: does `pyvista.read("model.exo")` (VTK's Exodus reader)
   read the file correctly? That tells us whether the file or exovista is at
   fault.

Security issues: please do not open a public issue; use GitHub's private
vulnerability reporting on the repository instead.

## Development setup

```shell
git clone https://github.com/bijective-systems-org/exovista.git
cd exovista
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The editable install matters: it registers the `exovista-read` command and the
`pyvista.writers` entry points that `mesh.save("x.exo")` needs (tests of those
are skipped in a checkout that is not installed).

## Running the tests and the linter

```shell
python -m pytest test/                          # the whole suite
python -m pytest test/test_write_sidesets.py    # one file
flake8 exovista test examples docs              # the linter, as in CI
```

No display is needed for the tests. Regenerating the README image does
render: `python docs/images/make_hero.py` (on a headless Linux machine,
`xvfb-run -a python docs/images/make_hero.py`).

The documentation is tested too: `test/test_readme.py` runs every Python
block of `README.md` in order, and `test/test_examples.py` runs every script
in `examples/`. If you change behaviour, update the README and examples in the
same pull request.

## How exovista is tested: oracles, not round trips

A writer checked only by its own reader can be consistently wrong. So every
writer feature is checked against an **independent reader**:

- **VTK's Exodus reader** (`pv.ExodusIIReader`, the reader inside ParaView) is
  the main oracle: element types and node order (cell volumes, point ids),
  block ids and names, side-set faces (their geometry, not just their
  numbers), node sets, variables at every time step.
- **netCDF4** inspection of the raw file for what VTK does not expose
  (dimensions, truth tables, name padding, `num_dim`, the element id map).
- **meshio** as a second reader where it applies.
- `read_exo` is then tested against the files in `test/data` (written by
  Sandia tools, serial and parallel) and against files written by VTK and
  meshio, as well as on exovista's own output.

The oracles need a newer PyVista than exovista itself (which supports
PyVista >= 0.42): `pv.ExodusIIReader` (0.45), `validate_mesh` (0.48), the
quadratic cells of `pyvista.examples.cells` (0.45), ... Reach them through
the wrappers in `test/_pvcompat.py` (`exodus_reader`, `validate_mesh`,
`example_cell`, `extract_surface`, `requires_pyvista`): on an older PyVista
the test is skipped with a reason that names the missing feature, while tests
of exovista's own behaviour keep running. The CI job "Oldest supported
dependencies" runs the suite with the lower bounds of `pyproject.toml`.

When you add a feature or fix a bug, add a test that would fail without the
change, and when it concerns the file contents, check them with VTK's reader
or netCDF4 rather than only with `read_exo`.

## Code layout

| Module | Role |
|---|---|
| `exovista/write_pyvista.py` | `write_exo`: PyVista objects -> data model (blocks, side-set matching, variables) |
| `exovista/read_pyvista.py` | `read_exo` and the `read_*` helpers: file -> PyVista / NumPy |
| `exovista/_celltypes.py` | single source of truth for element types, node permutations and side tables |
| `exovista/_model.py` | plain dataclasses (`ExodusModel`, `Block`, `SideSet`, `NodeSet`) |
| `exovista/_backend.py` | the only module that touches the vendored ExodusII core |
| `exovista/_pyvista_plugin.py` | `mesh.save("x.exo")` |
| `exovista/legacy.py` and `file.py`, `parallel_file.py`, ... | the vendored Sandia `exodusii` core and its low-level API |

Keep changes to the vendored core small and reach it only through
`_backend.py`, so that it can later be replaced by the upstream `exodusii`
package in one place.

## Pull requests

- Branch from `main`, keep the change focused, and describe what it fixes.
- `python -m pytest test/` and `flake8 exovista test examples docs` pass.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md` for anything a user
  could notice.
- Public behaviour is documented in the docstrings of `write_exo` /
  `read_exo` and in `README.md`; keep them in agreement.

By contributing you agree that your contribution is licensed under the
project's [BSD-2-Clause-Views license](LICENSE).
