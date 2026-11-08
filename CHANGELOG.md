# Changelog

All notable changes to exovista are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [semantic versioning](https://semver.org/spec/v2.0.0.html). Released
versions are on [PyPI](https://pypi.org/project/exovista/#history).

## [Unreleased]

## [2.0.0] - 2026-11-08

2.0 rebuilds `write_exo` and `read_exo` on a new internal model with one
table of element types, so that what is written is exactly what is read back,
by exovista and by VTK's Exodus reader (which the test suite uses as an
independent oracle). It is a breaking release: read
[Migrating from 1.x](#migrating-from-1x) first.

### Migrating from 1.x

Every change below can alter the files you write or the objects you read.

**Writing (`write_exo`)**

| | 1.x | 2.0 |
|---|---|---|
| Arguments | every argument could be passed by position | only `filename, volume, surface` by position; everything else keyword-only (`write_exo(f, vol, surf, region_key="r")`) |
| Block ids | `1..N` in internal order | the region value when all regions are integers >= 1 with one element type each; otherwise `1..N` by sorted region, then element type |
| Block order | by VTK cell type, then region | by sorted region value, then element-type name |
| Default block names | `block_0`, `block_1`, ... (0-based, regardless of id) | `block_<id>`; the string for string regions; the leaf key for `MultiBlock` input |
| `block_names=[...]` | applied in internal block order, so names could land on the wrong blocks | ordered like the sorted unique region values; a dict `{region: name}` or a single string also work; a region with several element types gets `<name>_<type>` blocks |
| Side-set ids and default names | `1..N`, `side_0`, `side_1`, ... | integer surface regions >= 1 are the ids; `side_set_<id>`; the string for string regions |
| Floating point | float32 | float64 (`precision="single"` for float32); the `EXODUSII_FLOAT_WORD_SIZE` environment variable affects only `exovista.legacy` |
| Static cell arrays | dropped, or written as node variables when `n_cells == n_points` | element variables (`save_cell_arrays=True`) |
| Multi-component arrays | crashed the writer | split into `name_X, name_Y, name_Z` (tensors `_XX ...`, other sizes `_0 ...`) |
| Helper arrays | an `orig_pts` node variable leaked into every file | bookkeeping arrays are never written |
| Planar meshes | always `num_dim = 3` | `num_dim = 2` when there are no 3D cells and z is constant; z is not stored, so a flat mesh at z != 0 reads back at z = 0 (with a warning when written; `num_dim=3` keeps 3D); `read_exo` output of a 3D file stays 3D |
| 2D cells in 3D files | `QUAD4` / `TRI3` | `SHELL4`, `SHELL8`, `TRI3`/`TRI6` (triangular shells), `QUAD9` |
| Element-type names | `TET4`, `HEX8`, `WEDGE6`, `PYRAMID5`, `QUAD4`, `TRI3` | `TETRA`, `HEX8`, `WEDGE`, `PYRAMID`, `QUAD4`, `TRI3`: only the linear tetrahedron, wedge and pyramid are spelled differently (as meshio reads them) |
| Unsupported cells | logged, then an inconsistent file was written | `ValueError` before anything is written |
| Side-set matching | nearest face centre, no distance check | topological (nodes within `side_set_tolerance`); mismatches raise `ValueError`; interface faces go to the element the surface normal points out of |
| Time | no time value when only static arrays were written (read as 9.97e36) | a single step at t = 0.0 whenever variables are written |
| `time_step` dimension | unlimited | fixed length (one compressed chunk per step): steps cannot be appended to a file afterwards |
| Default title | `pyvista_mesh` | `exovista`; titles longer than 80 bytes are cut (with a warning) |
| `region_key` | a name missing from the mesh gave one block | the default `"region"` may be missing; any other name must be a cell array of the volume or surface (`KeyError`) |
| Names | truncated at 32 characters | up to 256 bytes (UTF-8), NUL padded |
| Logging | `logging.info` on the root logger | module logger only (`exovista.write_pyvista`), no output by default; problems are warnings or exceptions |

**Reading (`read_exo` and helpers)**

| | 1.x | 2.0 |
|---|---|---|
| Cell order | file order (blocks concatenated) | the order of the mesh that was written, when the element id map allows it (`original_order=False` for file order) |
| Element columns of `read_element_fields` | file order | same rule (`original_order=`) |
| Vector and tensor variables | separate scalars (`vel_X`, ...) | recombined into one array (`combine_components=False` for scalars) |
| New cell data | — | `exo_elem_id` (element id map) |
| Global variables | not read | `grid.field_data[name]` at `time_step` (components recombined into `(1, n)` arrays); `read_global_variables=` |
| Field data | `times`, `num_dim` | also `time` (the selected time value) |
| Side-set keys | the set name; a later set with the same (or no) name replaced the earlier one, so only one of them was returned | unique keys; unnamed or repeated names become `side_set_<id>` |
| Side-set cell data | `orig_elem_id`, `face_id` | `exo_elem_id`, `exo_side`, `cell_id` (the old names remain as deprecated aliases) |
| Unsupported element blocks | the whole read failed | the block is skipped with a warning |

**Package**

- The low-level API (`File`, `exo_file`, `exodusii_file`, `ExodusIIFile`,
  `write_globals`, `parallel_exodusii_file`, `MFExodusIIFile`, `allclose`,
  `similar`, `lineout`, `find_element_data_in_region`,
  `find_node_data_in_region`, `exoread` and the `compute_*` helpers) moved to
  `exovista.legacy`. `exovista.<name>` still works in 2.x with a
  `DeprecationWarning`, but `from exovista import *` no longer brings these
  names (use `from exovista.legacy import ...`).
- The helpers of the `exovista.write_pyvista` module (`vtk2exo`,
  `vtk2exo_faceorder`, `vtk2exo_permute`, `process_meshes`) were removed;
  `write_exo` is its only public function.
- The console script `exoread.py` is now `exovista-read`.
- Python >= 3.10 is required (declared since 1.2.0; 2.0 supports 3.10–3.14).
- netCDF4 is required (the pure-Python netCDF fallback is gone); the minimum
  versions are numpy 1.21.3, scipy 1.7.2, netCDF4 1.5.8 and pyvista 0.42
  (0.42 and 0.43 with VTK < 9.4). `mesh.save("x.exo")` needs PyVista >= 0.48;
  the README and examples use `extract_surface(algorithm=None)` (0.47) and
  write back `pv.ExodusIIReader` output with side and node sets (0.49).

### Added

- `write_exo` keywords `save_cell_arrays`, `node_sets`, `global_fields`,
  `num_dim`, `precision`, `compression_level`, `side_set_tolerance` and
  `title`.
- `write_exo` accepts any PyVista dataset (cast to an `UnstructuredGrid`) or a
  `pyvista.MultiBlock` (one region per leaf, named after the leaf; points
  merged; arrays present on only some leaves kept, NaN elsewhere). Inputs are
  never modified.
- `write_exo` recognises the output of VTK's Exodus reader (`pv.read`,
  `pv.ExodusIIReader(...).read()`): block ids and names are kept, side sets
  and node sets are converted, VTK's glued vectors are written under their
  original component names. Nodes are joined by `GlobalNodeId` when present;
  joining by coordinates warns when it visibly joins distinct nodes (differing
  point values, nodes of one block), as does writing a flat model as 2D
  without `num_dim` (VTK's output does not record the file's dimension) and
  leaving out its global variables.
- Writing back `read_exo` output keeps its global variables (field data) and
  its dimension (a 3D file, e.g. a planar shell model, stays 3D).
- Quadratic and other element types on write and read: `TETRA10`, `HEX20`,
  `HEX27`, `WEDGE15`, `PYRAMID13`, `TRI6`, `QUAD8`, `QUAD9`, shells
  (`TRI3`/`TRI6` triangular shells and `SHELL4`/`SHELL8` in 3D files; any
  `TRISHELL*`/`SHELL*` on read), `BAR2`/`BAR3` and `SPHERE`, each node ordering
  checked against VTK's reader. Element-type names are recognised as by the
  Exodus library (first three letters plus node count: `HEXA8`, `HEX_8`,
  `TRIA3`, `EDGE2`, `TRUSS2`, `BEAM3`, `CIRCLE`, ...).
- Side sets on the edges of planar 2D meshes.
- Node sets (`node_sets={name: indices}` or boolean masks) on write;
  `read_node_sets` on read.
- Global variables (`global_fields`) on write; `read_global_fields` (with
  `combine_components`) and `read_exo(read_global_variables=...)` on read,
  components recombined like node and element variables.
- The element id map records each element's original cell index, which
  `read_exo` uses to restore the cell order.
- Element variables that are NaN on a whole block are left out of that block
  in the truth table (undefined), and read back as NaN.
- Atomic writes: the file is written to a temporary file and moved into place;
  on error (including errors when the file is closed and HDF5 flushes it) no
  file is created and an existing file is untouched; a target that is not
  writable raises `PermissionError` (as in 1.x), although a rename could
  replace it.
- `read_exo(..., combine_components=True, original_order=True,
  read_global_variables=True)`; `read_side_sets(..., original_order=True)`;
  `read_node_fields` / `read_element_fields` gained `combine_components`
  (and `original_order` for elements).
- Side-set `PolyData` cell data `exo_elem_id`, `exo_side` and `cell_id`, and
  field data `exo_side_set_id`.
- Reading files that store coordinates in one `coord` variable (as meshio
  writes them), "normal model" files that store all nodal variables in one
  `vals_nod_var` variable, and files written by exovista 1.x (read as they
  were meant). A nodal variable without stored values reads as NaN with a
  warning; a netCDF file that is not an Exodus file raises a clear
  `ValueError`.
- `NSIDED`/`NFACED` and unknown blocks are skipped with a warning while
  element numbering and side sets stay aligned.
- PyVista writer plugin: `mesh.save("model.exo", **write_exo_kwargs)` for
  `.exo`, `.e`, `.ex2`, `.exii` and `.gen`, through PyVista's
  `pyvista.writers` entry point (PyVista >= 0.48; `pv.read` does not open
  `.gen` files, use `read_exo` or `pv.read(path, force_ext=".exo")`).
- `exovista.legacy` module; `exovista.__version__`; `py.typed` marker.
- `exovista-read --version` reports the exovista version; `python -m exovista`
  runs the same command. Errors (missing or non-Exodus files, unknown
  variables or steps) are reported in one line without a traceback, and a
  closed output pipe (`exovista-read model.exo | head`) ends quietly.
- Documentation: README rewrite (guides, verified comparison with VTK and
  meshio, API reference), this changelog, `CITATION.cff`, `CONTRIBUTING.md`,
  issue templates, examples `example_pyvista_save.py` and
  `example_read_back.py`; tests run every README snippet and every example.
- Internal modules `_celltypes` (element types, permutations, side tables),
  `_model` (plain data model) and `_backend` (the only bridge to the vendored
  Sandia core).

### Changed

- See [Migrating from 1.x](#migrating-from-1x) for every behaviour change of
  `write_exo` and `read_exo`.
- The vendored ExodusII core writes float64 by default (`EXODUSII_FLOAT_WORD_SIZE=4`
  restores float32 for the legacy API), uses the Exodus string lengths
  (names of 256 bytes, strings of 33, lines of 81) and records
  `maximum_name_length`.
- Biquadratic quads are written as `QUAD9` even in 3D files, because VTK
  (<= 9.7) reads `SHELL9` blocks with an 8-node stride.
- Single time-step reads slice the netCDF variable instead of loading the
  whole history, and every time-dependent variable is compressed one time
  step per chunk, so that reading one step (also in VTK and ParaView)
  decompresses only that step; parallel joins use array maps (1M hexahedra:
  0.45 s instead of 6.8 s).
- `api_version` and `version` are written as `NC_FLOAT`, like the Exodus
  library does. An invalid `EXODUSII_COMPRESSION_LEVEL` (or
  `EXODUSII_FLOAT_WORD_SIZE`) is ignored with a warning instead of being
  clamped silently.
- Package metadata: PEP 639 license expression `BSD-2-Clause-Views`,
  classifiers for Python 3.10–3.14, project URLs on
  `bijective-systems-org/exovista`, `test` and `dev` extras.
- Continuous integration: flake8, tests on Linux with Python 3.10–3.14 and on
  macOS and Windows, a job with the oldest supported dependencies (PyVista
  0.42 with VTK < 9.4; tests whose VTK oracles or helpers need a newer
  PyVista skip there), a weekly run against the latest releases, and a test
  of the built wheel. Releases
  are published to PyPI from GitHub releases (trusted publishing) after the
  test suite has passed against the built wheel.

### Deprecated

- Top-level access to the legacy API (`exovista.File`,
  `exovista.exodusii_file`, `exovista.allclose`, ...); import from
  `exovista.legacy`. Removed in 3.0.
- Side-set cell data `orig_elem_id` and `face_id`; use `exo_elem_id` and
  `exo_side`.

### Removed

- `clean.py`, `exovista/_put_solution.py` and `test/test.sh`.
- The `exovista.write_pyvista` helpers `vtk2exo`, `vtk2exo_faceorder`,
  `vtk2exo_permute` and `process_meshes`.
- The legacy names from `from exovista import *` (they are still reachable
  as attributes, with a `DeprecationWarning`, and from `exovista.legacy`).
- The pure-Python netCDF fallback (`exovista/netcdf.py`) and the
  `EXODUSII_USE_NETCDF4` environment variable.
- The process-wide `warnings.filterwarnings("ignore")` that importing the
  package installed.

### Fixed

- `read_exo` failed on every parallel decomposition (`AttributeError`).
- Vector point arrays crashed `write_exo`.
- An internal `orig_pts` array was written to every file, and cell arrays
  were written as node variables when a mesh had as many cells as points.
- Files without explicit times had the netCDF fill value (9.97e36) as their
  time.
- `block_names` could be applied to the wrong blocks.
- 2D side sets crashed or picked edges of the wrong boundary loop; a face
  far from the volume was matched to the nearest element anyway.
- Quadratic cells produced a file VTK could not read, without an error.
- `read_exo` failed on `SHELL`, `BAR` and `SPHERE` blocks, read `HEX20` and
  `TETRA10` as invalid linear cells, failed on files with a combined `coord`
  variable, and returned only one of the side sets that shared a name.
- Names (blocks, sets, variables) are read up to their first NUL byte, as
  the Exodus library does: a name changed in place by the Exodus library
  keeps the old name's tail after the NUL.
- Names longer than 32 characters were truncated or crashed the writer;
  non-ASCII names crashed it.
- File handles were not always closed after reading.
- Parallel reads: negative time steps other than -1, side sets on the joined
  element numbering, and incomplete decompositions (now a `ValueError`).
- Upstream `put_map` fix ported; the `noh.exo` and `edges.base.exo` reference
  files were restored.

## [1.3.0] - 2026-06-29

### Added

- `read_node_fields` and `read_element_fields`: whole time histories as
  `{name: (n_steps, n)}` without building the mesh.
- `read_exo(read_node_variables=..., read_element_variables=...)` accept a
  name or a list of names.

### Fixed

- `time_step=0` read the last step, and every negative `time_step` read the
  last step. Steps are now 1-based, negative values count from the end and
  `0` raises `IndexError`.

## [1.2.0] - 2026-06-27

### Added

- `read_exo`: an Exodus file (or parallel decomposition) as a PyVista
  `UnstructuredGrid` with `exo_block_id`/`exo_block_name` cell data, node and
  element variables at a chosen time step, and optionally the side sets as
  `PolyData` (`orig_elem_id`, `face_id`).
- Continuous integration (flake8 and pytest on Python 3.10–3.12) and
  packaging metadata (`requires-python >= 3.10`, classifiers, URLs, `dev`
  extra).

### Fixed

- The parallel join crashed on node sets without distribution factors.
- The `exoread.py` console script pointed at the wrong package.
- numpy 2 compatibility (`np.row_stack`) and `Quad4.center`.
- `similar()` compared a file with itself for block ids.
- Invalid block ids raised `TypeError` (`raise None`); a `NameError` in
  `copy_variable_histories`.
- `write_exo` added helper arrays to the caller's meshes.

## [1.1.1] and [1.1.0] - 2026-06-26 and 2026-06-25

### Added

- 1.1.0: time-varying fields: `write_exo(times=..., node_fields=...,
  element_fields=...)`; static point arrays are repeated at every step.

### Changed

- 1.1.1: netCDF-4 output is zlib-compressed (level from
  `EXODUSII_COMPRESSION_LEVEL`, default 1; 0 disables it), e.g. 32.7 MB ->
  1.3 MB for a 64k-node transient mesh.

## [1.0.0] to [1.0.12] - 2025-11-15 to 2026-03-26

The first releases of exovista, a fork of Sandia's
[exodusii](https://github.com/sandialabs/exodusii) renamed and extended with
`write_exo`: element blocks by cell type and region (`region_key`), side sets
from a surface mesh, optional surface, hybrid meshes (1.0.x, November 2025);
point data as node variables and `block_names`/`side_set_names` (1.0.10 and
1.0.11, December 2025); 2D quadrilateral and triangle blocks with edge side
sets, `VOXEL`/`PIXEL` cells, and a 40x faster side-set lookup (1.0.12, March
2026).
