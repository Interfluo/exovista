# exovista

**Write PyVista meshes to ExodusII — element blocks, side sets, node sets and
time-varying fields — and read them back.**

[![PyPI version](https://img.shields.io/pypi/v/exovista.svg)](https://pypi.org/project/exovista/)
[![Python versions](https://img.shields.io/pypi/pyversions/exovista.svg)](https://pypi.org/project/exovista/)
[![CI](https://github.com/bijective-systems-org/exovista/actions/workflows/ci.yml/badge.svg)](https://github.com/bijective-systems-org/exovista/actions/workflows/ci.yml)
[![License: BSD-2-Clause-Views](https://img.shields.io/badge/license-BSD--2--Clause--Views-blue.svg)](https://github.com/bijective-systems-org/exovista/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/exovista.svg)](https://pypistats.org/packages/exovista)

> Not to be confused with NASA's ExoVista exoplanet-imaging package: this
> `exovista` is about finite-element meshes in the Exodus II format.

![A pipe segment written with exovista and read back: two element blocks, three side sets and a node variable at one time step](https://raw.githubusercontent.com/bijective-systems-org/exovista/main/docs/images/hero.png)

```python
import pyvista as pv
import exovista

mesh = pv.ImageData(dimensions=(5, 3, 3)).cast_to_unstructured_grid()
exovista.write_exo("box.exo", mesh, mesh.extract_surface(algorithm=None))
grid, side_sets = exovista.read_exo("box.exo", read_side_sets=True)
```

## Contents

- [Why exovista](#why-exovista)
- [Features](#features)
- [How it compares](#how-it-compares)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Guides](#guides): [element blocks](#element-blocks-and-regions) ·
  [side sets](#side-sets) · [node sets](#node-sets) ·
  [variables](#variables-and-time-steps) · [2D meshes](#2d-meshes) ·
  [element types](#element-types) · [reading](#reading) ·
  [writing back what `pv.read` returns](#writing-back-what-pvread-returns)
- [API reference](#api-reference)
- [Limitations](#limitations)
- [Migrating from 1.x](#migrating-from-1x)
- [Contributing](#contributing) · [Citation](#citation) ·
  [License and attribution](#license-and-attribution)

## Why exovista

[Exodus II](https://github.com/sandialabs/seacas) is the mesh and results
format of the Sandia SEACAS tools, Cubit, MOOSE, Sierra, ALEGRA, ParaView and
many other finite-element codes. A solver input deck needs more than cells and
points: it needs **element blocks** (materials), **side sets** (boundary
faces for loads and boundary conditions) and **node sets**, all with ids and
names.

PyVista already *reads* Exodus files well: `pv.read("model.exo")` and
`pv.ExodusIIReader` use VTK's `vtkExodusIIReader`, which covers almost all of
the format. What PyVista does not have is a *writer* — without exovista,
`mesh.save("model.exo")` raises `ValueError` — and the alternatives fall
short: VTK's `vtkExodusIIWriter` has no direct way to create side sets or
node sets, and meshio writes no side sets, block names or time history (see
[How it compares](#how-it-compares)).

exovista fills that gap. `write_exo` turns any PyVista mesh into an Exodus
file with element blocks taken from a region array, side sets matched from a
boundary surface, node sets, static and time-varying node / element / global
variables, and zlib compression. Its counterpart `read_exo` is a convenience
reader that returns *one* `UnstructuredGrid` with float64 points in file node
order, the cells in the order they were written, and the side sets, node sets
and whole time histories as plain PyVista and NumPy objects.

## Features

- **Element blocks** from any integer or string cell array (`region` by
  default) or from the leaves of a `pv.MultiBlock`; integer region values
  become block ids, blocks are named (`"block_<id>"` by default).
- **Side sets** from a boundary surface (`mesh.extract_surface()`, or any
  subset of it), matched topologically to element sides, with a documented
  rule for faces on internal interfaces.
- **Node sets** from index arrays or boolean masks.
- **Variables**: point and cell arrays, time histories
  (`node_fields`, `element_fields`), global variables; vectors are split into
  `_X/_Y/_Z` components (which ParaView regroups), tensors into
  `_XX ... _XZ`.
- **Element types**: linear and quadratic tetrahedra, hexahedra, wedges and
  pyramids, triangles and quadrilaterals (planar or shells), bars and
  spheres — every node ordering checked against VTK's own Exodus reader.
- **2D meshes** are written as true `num_dim = 2` files automatically.
- **Precision and size**: float64 by default (`precision="single"` for
  float32), zlib-compressed netCDF-4 output, atomic writes (no half-written
  file on error).
- **Reading back**: `read_exo` restores the written mesh exactly (points,
  cell order, blocks, variables), recombines vector and tensor components, and
  joins parallel decompositions (`model.exo.4.0` ... `model.exo.4.3`) into one
  mesh with global numbering. Side sets, node sets and full time histories
  have their own functions.
- **PyVista integration**: installing exovista registers a `pyvista.writers`
  plugin, so `mesh.save("model.exo")` works.

## How it compares

Checked on 2026-11-08 with pyvista 0.49.2, VTK 9.7.2, meshio 5.3.5 and
netCDF4 1.7.5, by writing and reading a small two-block hexahedral mesh, a
unit-cube `HEX20` and the parallel test files of this repository, and
inspecting the files with netCDF4 and VTK's reader. Other versions may
behave differently.

**Writing a PyVista mesh.** PyVista itself has no Exodus writer
(`UnstructuredGrid.save` accepts `.vtu`, `.vtk`, `.case`, `.vtkhdf`).

| | `exovista.write_exo` | VTK `vtkExodusIIWriter` | `meshio.write(..., "exodus")` |
|---|---|---|---|
| Element blocks | from a region array; integer regions are the block ids | from a cell array (`SetBlockIdArrayName`); otherwise one block per cell type, with the VTK cell-type number as id (10 for tetrahedra, 12 for hexahedra) | one block per meshio cell block, ids from 0 |
| Block names | yes | only from `MultiBlock` input, which duplicates the nodes the blocks share (45 nodes for a 36-node mesh) | no |
| Side sets | yes, from a boundary surface | no direct API (a hand-filled `vtkModelMetadata` was not tried); passed through from a `vtkExodusIIReader` pipeline, but written empty unless the reader generates global id arrays | no |
| Node sets | yes | like side sets: passed through from a `vtkExodusIIReader` pipeline only | yes (`point_sets`) |
| Time steps | yes (`times` + fields) | needs a time-aware VTK pipeline and `WriteAllTimeSteps`; a plain mesh gives one step at t = 0 | no (one step at t = 0) |
| Point and cell arrays | yes (vectors as `v_X, v_Y, v_Z`) | yes (vectors as `vX, vY, vZ`) | point arrays only; cell arrays are dropped |
| Global variables | yes (`global_fields`) | yes (from field data) | no |
| Quadratic node order (`HEX20` unit cube read back by VTK) | correct (volume 1.0) | wrong (volume 0.083) | wrong (volume 0.083) |
| Planar `num_dim = 2` files | automatic | no (always 3) | only from 2D point arrays (`pv.to_meshio` gives 3) |
| Compression | zlib, netCDF-4 classic | none (netCDF-3 64-bit offset) | none (netCDF-4) |

**Reading.** VTK's reader is the most complete; exovista's reader is a
convenience for getting back exactly what was written.

| | `exovista.read_exo` | `pv.read` / `pv.ExodusIIReader` (VTK) | `meshio.read` |
|---|---|---|---|
| Result | one `UnstructuredGrid` (+ one `PolyData` per side set) | `MultiBlock` with one grid per element block | `meshio.Mesh` |
| Block ids and names per cell | yes (`exo_block_id`, `exo_block_name`) | yes (`ObjectId` array, block names as keys) | no (one cell block per element block) |
| Side sets | yes, with element and side numbers | yes | no |
| Node sets | yes (index arrays) | yes (`pv.read` does not load them by default) | yes (`point_sets`) |
| Choose the time step | yes (`time_step=`) | yes (`set_active_time_point`) | no (first step only) |
| Global variables | yes | `read_global()` after enabling the arrays; `enable_all_global_arrays()` enables only some of them (5 of the 74 in `noh.exo`), `reader.reader.SetAllArrayStatus(...)` all | no |
| Whole time history in one call | yes (`read_node_fields`, ...) | global variables only | no |
| Point coordinates | float64 as stored | float32 (`10000000.001` read as `10000000.0`) | float64 |
| Quadratic elements | yes | yes | node order not converted (`HEX20` volume 0.083) |
| Parallel decompositions (`*.exo.N.p`) | joined into one mesh with global numbering (738 nodes for `edges.exo.4.*`) | `pv.PExodusIIReader`: per-block grids, nodes on part boundaries repeated (756 points) | one part at a time |
| Edge and face blocks | no (ignored) | yes | no |

VTK's reader also moves the points by displacement variables (`DISPL*`)
unless `reader.disable_displacements()` is called; exovista never does.

## Installation

```shell
pip install exovista
```

exovista needs Python >= 3.10 (tested on 3.10 to 3.14) and installs
NumPy >= 1.21.3, SciPy >= 1.7.2, netCDF4 >= 1.5.8 and PyVista >= 0.49 (which
brings VTK). It is not on conda-forge yet; inside a conda environment,
install the dependencies with conda and exovista with pip.

For development:

```shell
git clone https://github.com/bijective-systems-org/exovista.git
cd exovista
pip install -e ".[dev]"
```

## Quick start

Write a mesh with two materials and two boundary side sets:

```python
import numpy as np
import pyvista as pv

import exovista

# 4 x 2 x 2 hexahedra, two materials split at x = 2
mesh = pv.ImageData(dimensions=(5, 3, 3)).cast_to_unstructured_grid()
mesh.cell_data["region"] = np.where(mesh.cell_centers().points[:, 0] < 2, 1, 2)
mesh.point_data["temperature"] = 20.0 + 10.0 * mesh.points[:, 0]

# the boundary faces at x = 0 and x = 4 become two side sets
surface = mesh.extract_surface(algorithm=None)
x = surface.cell_centers().points[:, 0]
ends = np.isclose(x, 0.0) | np.isclose(x, 4.0)
surface = surface.extract_cells(np.flatnonzero(ends))
surface.cell_data["region"] = np.where(np.isclose(x[ends], 0.0), "inlet", "outlet")

exovista.write_exo(
    "model.exo", mesh, surface,
    block_names={1: "copper", 2: "steel"},
    node_sets={"corner": [0]},
)
```

The file has element blocks 1 (`copper`) and 2 (`steel`), side sets `inlet`
and `outlet`, node set `corner` and a node variable `temperature` (one time
step at t = 0). It opens in ParaView.

With exovista installed, PyVista's `save` does the same (the keyword
arguments go to `write_exo`):

```python
mesh.save("saved.exo", surface=surface, block_names={1: "copper", 2: "steel"})
```

The plugin handles `.exo`, `.e`, `.ex2`, `.exii` and `.gen`. `mesh.save` keeps
its own `binary`, `texture` and `compression` arguments; exovista's
compression setting is `compression_level`.

Read it back:

```python
grid = exovista.read_exo("model.exo")
print(grid.n_cells, grid.points.dtype)                # 16 float64
print(np.unique(grid.cell_data["exo_block_name"]))   # ['copper' 'steel']
print(grid.point_data["temperature"][:3])             # [20. 30. 40.]

grid, side_sets = exovista.read_exo("model.exo", read_side_sets=True)
print(side_sets.keys())                               # ['inlet', 'outlet']
print(exovista.read_node_sets("model.exo"))           # {'corner': array([0])}
```

`read_exo` returns the mesh as it was written: same points (float64, same
order), same cells in the same order, plus `exo_block_id`, `exo_block_name`
and `exo_elem_id` cell arrays.

Use **`pv.read("model.exo")`** (VTK's reader) instead when you want
ParaView's view of a file: one grid per block, edge and face blocks, element
sets, displacements applied, or the widest coverage of Exodus files written
by other codes. Use **`exovista.read_exo`** when you want one grid that
matches what you wrote, float64 coordinates, side sets with their element and
side numbers, node sets as index arrays, whole time histories, or a parallel
decomposition joined into one mesh.

```python
blocks = pv.read("model.exo")["Element Blocks"]       # VTK: one grid per block
print(blocks.keys())                                  # ['copper', 'steel']
```

More in [`examples/`](https://github.com/bijective-systems-org/exovista/tree/main/examples)
(each script writes its files to the current directory):

| Script | Shows |
|---|---|
| `example_multi_block.py` | element blocks from a region array, a `MultiBlock` and string regions |
| `example_mixed_elements.py` | hexahedra and tetrahedra in one model |
| `example_side_sets.py` | side sets from part of the boundary, node sets |
| `example_2d_quads.py` | a planar (`num_dim = 2`) mesh |
| `example_time_fields.py` | time-varying node, element and global variables |
| `example_read_exo.py` | the round trip `write_exo` -> `read_exo`, side sets included |
| `example_read_field_history.py` | whole time histories, variable selection, `time_step` |
| `example_pyvista_save.py` | `mesh.save("model.exo", ...)` |
| `example_read_back.py` | edit a file through `pv.read` or `read_exo` and write it back |
| `save_exo.py` | larger meshes (`--all` downloads PyVista's example data) |

## Guides

### Element blocks and regions

An Exodus element block holds elements of one type. `write_exo` groups cells
by **(region, element type)**, where the region is the value of the cell array
named by `region_key` (`"region"` by default; integer or string values).
Without that array the whole mesh is one region.

- **Ids.** If every region value is an integer >= 1 and each region has a
  single element type, the block id is the region value. Otherwise blocks are
  numbered `1..N` in block order (sorted region value, then element-type name).
- **Names.** `block_names` takes a list (ordered like the sorted unique region
  values), a dict `{region_value: name}`, or a single string for a single
  region. Defaults: the region value for string regions, the leaf key for a
  `MultiBlock`, otherwise `"block_<id>"`. Names must be unique and at most 256
  bytes.
- **Several element types in one region** give one block per type; a given
  (or string-region) name gets the type as suffix, e.g. `steel_hex8`,
  `steel_tetra4`.
- **`pv.MultiBlock` input**: each non-empty leaf is a region (values `1..N` in
  leaf order) named after its key; points with identical coordinates are
  merged.

```python
centers = mesh.cell_centers().points

# integer regions >= 1, one element type each: the values are the block ids
mesh.cell_data["region"] = np.where(centers[:, 0] < 2, 10, 20)
exovista.write_exo("blocks.exo", mesh)  # blocks 10 "block_10" and 20 "block_20"

# string regions name the blocks (ids 1..N in sorted order)
mesh.cell_data["region"] = np.where(centers[:, 0] < 2, "copper", "steel")
exovista.write_exo("blocks.exo", mesh)  # blocks 1 "copper" and 2 "steel"

# a MultiBlock: one block per leaf, named after its key
cube = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
parts = pv.MultiBlock({"left": cube, "right": cube.translate((2.0, 0.0, 0.0))})
exovista.write_exo("blocks.exo", parts)  # blocks 1 "left" and 2 "right"

# hexahedra and tetrahedra in one region: one block per element type
mixed = cube.merge(cube.translate((2.0, 0.0, 0.0)).triangulate())
exovista.write_exo("blocks.exo", mixed, block_names="solid")
print(np.unique(exovista.read_exo("blocks.exo")["exo_block_name"]))
# ['solid_hex8' 'solid_tetra4']
```

The element id map of the file stores each element's original cell index + 1,
which is how `read_exo` restores the input cell order.

### Side sets

Pass the faces to write as `surface` (any PyVista dataset whose cells are
faces of the volume cells, or edges for a 2D mesh). One side set is written
per value of `surface.cell_data[region_key]` (a single set without it); ids
and names follow the block rules, with defaults `"side_set_<id>"`. Note that
`extract_surface()` copies the volume's cell arrays, `region` included, onto
the faces: overwrite or delete that array to choose the sets.

```python
volume = pv.ImageData(dimensions=(4, 4, 4)).cast_to_unstructured_grid().triangulate()
surface = volume.extract_surface(algorithm=None)
z = surface.cell_centers().points[:, 2]
surface.cell_data["region"] = np.select([z > 2.99, z < 0.01], ["top", "bottom"], "walls")
exovista.write_exo("sides.exo", volume, surface)
print(exovista.read_side_sets("sides.exo").keys())  # ['bottom', 'top', 'walls']
```

How faces are matched:

- Each surface point is matched to its nearest volume node within
  `side_set_tolerance` (default `1e-6` times the bounding-box diagonal); then
  each surface cell's corner nodes are looked up among the element sides. A
  surface point or cell that matches nothing raises `ValueError` (with counts
  and the largest distance), so a wrong surface never writes a wrong file.
- **Internal interfaces.** A face shared by two elements goes to the element
  that the surface normal points *out of*. So the faces of
  `region_mesh.extract_surface(algorithm=None)` belong to that region's
  elements. (Prefer `algorithm=None`: with PyVista's current default,
  `"dataset_surface"`, VTK orients the faces of linear wedges inward, so their
  interface faces would go to the neighbour.) For 2D edges the in-plane normal
  `(dy, -dx)` is used (outward for counter-clockwise boundaries). Ties go to
  the lower block id, then the lower element index.
- **Quadratic cells** match by their corner nodes. By default PyVista splits
  quadratic faces into linear pieces that match no element side; extract
  them unsplit with `nonlinear_subdivision=0`.
- Repeated faces within a set are written once, with a warning. Sides keep the
  surface cell order.
- Shells, bars and spheres are not side-set targets.

```python
# the boundary of region 10, assigned to region-10 elements even on the
# interface with region 20
mesh.cell_data["region"] = np.where(centers[:, 0] < 2, 10, 20)
left = mesh.extract_cells(np.flatnonzero(mesh.cell_data["region"] == 10))
exovista.write_exo("iface.exo", mesh, left.extract_surface(algorithm=None),
                   side_set_names="left_boundary")
grid = exovista.read_exo("iface.exo")
faces = exovista.read_side_sets("iface.exo")["left_boundary"]
print(np.unique(grid.cell_data["exo_block_id"][faces.cell_data["cell_id"]]))  # [10]

# quadratic elements: keep the faces unsplit
from pyvista import examples

hex20 = examples.cells.QuadraticHexahedron()
exovista.write_exo("hex20.exo", hex20,
                   hex20.extract_surface(algorithm=None, nonlinear_subdivision=0))

# a surface that is slightly off the volume nodes: widen the tolerance
shifted = volume.extract_surface(algorithm=None).translate((1e-4, 0.0, 0.0))
exovista.write_exo("sides.exo", volume, shifted, side_set_tolerance=1e-3)
```

### Node sets

`node_sets` is a dict `{name: indices}` or a list of index arrays (named
`"node_set_<id>"`). Indices are 0-based point indices of the volume (unique,
in range); a boolean mask over the points also works. Integer dict keys >= 1
are used as node-set ids.

```python
bottom = np.flatnonzero(np.isclose(volume.points[:, 2], 0.0))
exovista.write_exo("sets.exo", volume, node_sets={
    "bottom": bottom,
    "top": volume.points[:, 2] > 2.99,  # a boolean mask
})
print({name: len(ids) for name, ids in exovista.read_node_sets("sets.exo").items()})
# {'bottom': 16, 'top': 16}
```

### Variables and time steps

- **Static arrays**: `volume.point_data` becomes node variables and
  `volume.cell_data` element variables (`save_node_arrays` /
  `save_cell_arrays`, both `True`), constant over all time steps. The
  `region_key` array and bookkeeping arrays (`vtkOriginalPointIds`,
  `vtkGhostType`, `exo_block_id`, VTK reader ids such as `ObjectId`, ...) are
  not written. Booleans and integers are written as floats; complex and
  non-numeric arrays are skipped with a warning.
- **Time histories**: `times` plus `node_fields={name: (n_steps, n_points)}`,
  `element_fields={name: (n_steps, n_cells)}` (columns in the original cell
  order) and `global_fields={name: (n_steps,)}`. A trailing component axis
  is allowed. For a single step, `(n_points,)` / `(n_cells,)` arrays are
  accepted. Without `times`, variables are written at one step, t = 0.0. A
  time-varying field replaces a static array of the same name.
- **Components**: multi-component arrays are split into scalar variables with
  these suffixes, which `read_exo` recombines (`combine_components=True`).
  VTK's reader regroups the `_X/_Y/_Z` vectors (as `name_`) but not the
  tensors (a 9-component tensor shows up there as three vectors
  `name_X`, `name_Y`, `name_Z`).

  | Components | Suffixes |
  |---|---|
  | 2 | `_X, _Y` |
  | 3 | `_X, _Y, _Z` |
  | 6 (symmetric tensor, VTK order) | `_XX, _YY, _ZZ, _XY, _YZ, _XZ` |
  | 9 (full tensor) | `_XX, _XY, _XZ, _YX, _YY, _YZ, _ZX, _ZY, _ZZ` |
  | other n | `_0, _1, ..., _{n-1}` |

- **Undefined values**: an element variable that is NaN on every element of a
  block is not stored on that block (Exodus truth table); `read_exo` returns
  NaN there.
- **Precision**: `precision="double"` (default) or `"single"` for coordinates,
  times and variables. Integer arrays beyond 2**53 (2**24 in single precision)
  lose exactness and are reported with a warning.

```python
grid = pv.ImageData(dimensions=(3, 3, 3)).cast_to_unstructured_grid()
grid.point_data["velocity"] = np.ones((grid.n_points, 3))  # velocity_X/_Y/_Z
grid.cell_data["stress"] = np.zeros((grid.n_cells, 6))     # stress_XX ... _XZ

times = np.linspace(0.0, 1.0, 5)
exovista.write_exo(
    "fields.exo", grid, times=times,
    node_fields={"temperature": np.outer(times, grid.points[:, 0])},
    element_fields={"damage": np.outer(times, np.arange(grid.n_cells))},
    global_fields={"energy": times**2},
)
```

`write_exo` writes the whole history at once, and the `time_step` dimension
of the file has a fixed length, so more steps cannot be appended to the file
later (see [Limitations](#limitations)).

### 2D meshes

With `num_dim=None` (the default) a mesh without 3D cells whose points all
share one z coordinate is written as a planar `num_dim = 2` file: `TRI*` and
`QUAD*` elements, x and y coordinates only, and side sets made of element
edges (pass line cells as `surface`). Pass `num_dim=3` to keep z; 2D cells are
then written as shells.

```python
plate = pv.Plane(i_resolution=4, j_resolution=2).cast_to_unstructured_grid()
edges = plate.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                    manifold_edges=False, non_manifold_edges=False)
edges.cell_data["region"] = np.where(edges.cell_centers().points[:, 0] < -0.49,
                                     "left", "rest")
exovista.write_exo("plate.exo", plate, edges)        # num_dim 2, QUAD4, line sides
exovista.write_exo("shell.exo", plate, num_dim=3)    # SHELL4 elements in 3D
print(exovista.read_exo("plate.exo").field_data["num_dim"])  # [2]
```

### Element types

Every VTK -> Exodus node permutation below is checked by writing the element
with exovista and reading it with VTK's own Exodus reader (and with
`read_exo`).

| VTK cell | Written as (`num_dim = 3`) | Written as (`num_dim = 2`) |
|---|---|---|
| `TETRA`, `QUADRATIC_TETRA` | `TETRA` (4 nodes), `TETRA10` | — |
| `HEXAHEDRON`, `VOXEL`, `QUADRATIC_HEXAHEDRON`, `TRIQUADRATIC_HEXAHEDRON` | `HEX8`, `HEX8`, `HEX20`, `HEX27` | — |
| `WEDGE`, `QUADRATIC_WEDGE` | `WEDGE` (6 nodes), `WEDGE15` | — |
| `PYRAMID`, `QUADRATIC_PYRAMID` | `PYRAMID` (5 nodes), `PYRAMID13` | — |
| `TRIANGLE`, `QUADRATIC_TRIANGLE` | `TRI3`, `TRI6` (triangular shells) | `TRI3`, `TRI6` |
| `QUAD`, `PIXEL`, `QUADRATIC_QUAD` | `SHELL4`, `SHELL4`, `SHELL8` | `QUAD4`, `QUAD4`, `QUAD8` |
| `BIQUADRATIC_QUAD` | `QUAD9` | `QUAD9` |
| `LINE`, `QUADRATIC_EDGE` | `BAR2`, `BAR3` | `BAR2`, `BAR3` |
| `VERTEX` | `SPHERE` | `SPHERE` |

Notes:

- Linear tetrahedra, wedges and pyramids are written as `TETRA`, `WEDGE` and
  `PYRAMID`, and triangles in 3D files as `TRI3`/`TRI6`: every Exodus reader
  identifies an element type by its first three letters and its node count,
  and these spellings are also readable by meshio (<= 5.3). In a 3D file a
  `TRI` element is a triangular shell, as for the Exodus library and VTK.
- Biquadratic quads are written as `QUAD9` even in 3D, because VTK (<= 9.7)
  reads `SHELL9` blocks with the wrong stride.
- On read, any spelling the Exodus library accepts works (`HEX`, `HEXA8`,
  `HEX_8`, `TRIA3`, `TRUSS2`, `BEAM3`, `EDGE2`, `CIRCLE`, `SHELL9`, ...).
- Not supported: `POLYGON`, `POLYHEDRON`, `POLY_LINE`, `POLY_VERTEX`,
  `TRIANGLE_STRIP`, Lagrange and Bezier cells and other higher-order cells.
  `write_exo` raises `ValueError` naming the types and counts (convert with
  e.g. `mesh.triangulate()` or drop them with `extract_cells_by_type`).
  `read_exo` skips `NSIDED`/`NFACED` and unknown blocks with a warning.

### Reading

```python
mesh = exovista.read_exo("fields.exo", time_step=2)  # 1-based; -1 (default) = last
print(mesh.field_data["time"])                       # [0.25] (field_data["times"]: all 5)
print(mesh.point_data["velocity"].shape)              # (27, 3): components recombined
print(mesh.field_data["energy"])                      # global variable at step 2: [0.0625]

raw = exovista.read_exo("fields.exo", combine_components=False)
print(sorted(raw.point_data.keys())[:3])              # ['temperature', 'velocity_X', 'velocity_Y']

mesh = exovista.read_exo("fields.exo", read_node_variables="temperature",
                         read_element_variables=False)

# whole histories without building the mesh
temperature = exovista.read_node_fields("fields.exo", names="temperature")["temperature"]
stress = exovista.read_element_fields("fields.exo", combine_components=True)["stress"]
energy = exovista.read_global_fields("fields.exo")["energy"]
print(temperature.shape, stress.shape, energy.shape)  # (5, 27) (5, 8, 6) (5,)
```

- **`time_step`** is 1-based; negative values count from the end; `0` and
  out-of-range steps raise `IndexError`. `field_data["times"]` holds all time
  values and `field_data["time"]` the selected one.
- **Cell order**: with `original_order=True` (default) cells and element
  variables come back in the order of the mesh that was written, whenever the
  element id map is a permutation of `1..num_elem` (as written by
  `write_exo`); otherwise, and with `original_order=False`, in file order
  (blocks concatenated).
- **Cell data**: `exo_block_id`, `exo_block_name` (`""` if unnamed),
  `exo_elem_id` (from the element id map), and the element variables (NaN on
  blocks where a variable is not defined).
- **Side sets** (`read_side_sets` or `read_exo(..., read_side_sets=True)`):
  one `PolyData` per set, keyed by name, sharing the grid's points. Cell data
  `exo_elem_id` (1-based element index in file order), `exo_side` (1-based
  Exodus side) and `cell_id` (0-based index of the cell in the grid returned
  with the same `original_order`). `orig_elem_id` and `face_id` are kept as
  deprecated aliases. Field data `exo_side_set_id` holds the set id.
- **Node sets** (`read_node_sets`): `{name: 0-based point indices}`.
- Unnamed sets and repeated names get the keys `"side_set_<id>"` /
  `"node_set_<id>"`, so every set is returned.
- **Parallel decompositions**: pass all parts, or a glob pattern; they are
  joined into one mesh with global node and element numbering.

```python
joined = exovista.read_exo("run.exo.4.*")  # or read_exo("run.exo.4.0", "run.exo.4.1", ...)
print(joined.n_points, joined.n_cells)     # 738 320
```

Files written by exovista 1.x are recognised and read as they were meant
(their triangles keep edge sides; their unwritten time value reads as 0.0).

### Writing back what `pv.read` returns

`write_exo` recognises the `MultiBlock` that VTK's reader returns (`pv.read`,
`pv.ExodusIIReader(...).read()` or its `"Element Blocks"`): element blocks keep
their ids and names, its side sets and node sets are written (unless
`surface` / `node_sets` are given), VTK's glued vectors (`vel_`) are written
under their original component names, and its other entities are ignored
with a warning.

```python
reader = pv.ExodusIIReader("model.exo")
reader.node_sets.enable_all()          # pv.read loads side sets, not node sets
data = reader.read()
for block in data["Element Blocks"]:
    block.point_data["temperature"] += 1.0
exovista.write_exo("model_copy.exo", data)
copy = exovista.read_exo("model_copy.exo", read_side_sets=True)[1]
print(copy.keys(), exovista.read_node_sets("model_copy.exo"))
# ['inlet', 'outlet'] {'corner': array([0])}
```

Notes: VTK's reader moves the points by displacement variables (`DISPL*`)
unless you call `reader.disable_displacements()`; its nodes are merged by
`GlobalNodeId` when present (`reader.reader.GenerateGlobalNodeIdArrayOn()`),
otherwise by coordinates; only the reader's active time step is in the
`MultiBlock`, so pass `times=[reader.active_time_value]` to keep its time
value.

A grid from `read_exo` can be written back too: use its block ids as
regions. Side sets and node sets are not part of the grid; pass them back as
`surface` and `node_sets` (see `examples/example_read_back.py`).

```python
grid = exovista.read_exo("model.exo")
names = dict(zip(grid.cell_data["exo_block_id"], grid.cell_data["exo_block_name"]))
exovista.write_exo("model_copy.exo", grid, region_key="exo_block_id", block_names=names)
```

## API reference

All functions take `str` or `os.PathLike` file names. The full docstrings are
the reference (`help(exovista.write_exo)`); this is a summary.

### `write_exo`

<!-- readme-test: skip (signature, not a call) -->

```python
exovista.write_exo(
    filename, volume, surface=None, *,
    region_key="region", block_names=None, side_set_names=None, node_sets=None,
    save_node_arrays=True, save_cell_arrays=True,
    times=None, node_fields=None, element_fields=None, global_fields=None,
    num_dim=None, precision="double", compression_level=None,
    side_set_tolerance=None, title=None,
) -> None
```

| Parameter | Type | Description |
|---|---|---|
| `filename` | path | Output file. Written to a temporary file next to it and moved into place when complete: on error no file is created and an existing file is left untouched. A symbolic link's target is replaced; an existing file keeps its permissions. |
| `volume` | `pyvista.DataSet` or `pyvista.MultiBlock` | The mesh (cast to `UnstructuredGrid` as needed). A `MultiBlock` gives one region per non-empty leaf; VTK Exodus reader output is recognised. Never modified. |
| `surface` | `pyvista.DataSet`, optional | Faces (3D) or edges (`num_dim = 2`) to write as side sets, one set per region value. |
| `region_key` | `str`, default `"region"` | Cell array that assigns cells (and surface cells) to regions; integer or string values. |
| `block_names` | list, dict or str, optional | Block names: list in sorted region order, `{region: name}`, or one name. Default region string / leaf key / `"block_<id>"`. |
| `side_set_names` | list, dict or str, optional | Side-set names, like `block_names`; default `"side_set_<id>"`. |
| `node_sets` | dict or list, optional | `{name: indices}` or a list of index arrays (`"node_set_<id>"`); 0-based point indices or a boolean mask. |
| `save_node_arrays` | `bool`, default `True` | Write `volume.point_data` as node variables (constant in time). |
| `save_cell_arrays` | `bool`, default `True` | Write `volume.cell_data` as element variables (constant in time); `region_key` is not written. |
| `times` | array-like, optional | Time value of each step. Default: one step at t = 0.0 when any variable is written. |
| `node_fields` | dict, optional | `{name: (n_steps, n_points[, n_components])}`; for one step also `(n_points[, n_components])`. Replaces a point array of the same name. |
| `element_fields` | dict, optional | Like `node_fields` with one column per cell, in the original `volume` cell order. |
| `global_fields` | dict, optional | `{name: (n_steps[, n_components])}` global variables. |
| `num_dim` | `None`, 2 or 3 | `None`: 2 when there are no 3D cells and all points share one z, else 3. |
| `precision` | `"double"` or `"single"` | Storage of coordinates, times and variables (float64 / float32). |
| `compression_level` | `int` 0-9, optional | zlib level (0 disables compression); default from `EXODUSII_COMPRESSION_LEVEL` (1). |
| `side_set_tolerance` | `float`, optional | Largest surface-point-to-node distance; default `1e-6` times the bounding-box diagonal. |
| `title` | `str`, optional | Database title (default `"exovista"`). |

Raises `ValueError` for an empty mesh, unsupported cell types, a surface that
does not match the volume, or inconsistent names and shapes; `TypeError` for
inputs of the wrong type.

### `read_exo`

<!-- readme-test: skip (signature, not a call) -->

```python
exovista.read_exo(
    filename, *files, time_step=-1,
    read_node_variables=True, read_element_variables=True,
    read_global_variables=True, read_side_sets=False,
    combine_components=True, original_order=True,
) -> pyvista.UnstructuredGrid | tuple[pyvista.UnstructuredGrid, pyvista.MultiBlock]
```

| Parameter | Type | Description |
|---|---|---|
| `filename` | path | The file, or the first part of a parallel decomposition; glob patterns (`"run.exo.4.*"`) are expanded. |
| `*files` | paths | The other parts of a parallel decomposition (joined with global numbering). |
| `time_step` | `int`, default `-1` | 1-based step for node, element and global variables; negative counts from the end; `0` and out-of-range steps raise `IndexError`. Ignored (except `0`) when the file has no steps. |
| `read_node_variables` | bool, str or list of str | Node variables to load into `point_data`: all, none, or the given names (combined or stored names). Unknown names raise `KeyError`. |
| `read_element_variables` | bool, str or list of str | Same for element variables (`cell_data`). |
| `read_global_variables` | bool, str or list of str | Same for global variables (`field_data`, one-element arrays at `time_step`). |
| `read_side_sets` | `bool`, default `False` | Also return the side sets (`read_side_sets` result). |
| `combine_components` | `bool`, default `True` | Recombine `_X/_Y/_Z`, tensor and `_0.._n` component variables into one array when all components are present and the combined name is not itself a variable. |
| `original_order` | `bool`, default `True` | Sort cells by element id when the element id map is a permutation of `1..num_elem` (restores `write_exo`'s input order); otherwise file order. |

Returns the grid (float64 points in file node order; cell data
`exo_block_id`, `exo_block_name`, `exo_elem_id` and element variables; point
data the node variables; field data `times`, `time`, `num_dim` and the global
variables), and with `read_side_sets=True` also the side-set `MultiBlock`.
Warns (`UserWarning`) about skipped unsupported blocks, unresolvable side-set
entries, and variables whose names clash with the reserved arrays.

### Sets and histories

| Function | Returns |
|---|---|
| `read_side_sets(filename, *files, original_order=True)` | `pyvista.MultiBlock`: one `PolyData` per side set (cell data `exo_elem_id`, `exo_side`, `cell_id`; deprecated `orig_elem_id`, `face_id`; field data `exo_side_set_id`). `original_order` must match the grid's. |
| `read_node_sets(filename, *files)` | `dict[str, numpy.ndarray]`: 0-based int64 point indices into the `read_exo` points. |
| `read_node_fields(filename, *files, names=None, combine_components=False)` | `{name: (n_steps, n_nodes[, n_components])}` float64 histories. |
| `read_element_fields(filename, *files, names=None, combine_components=False, original_order=True)` | `{name: (n_steps, n_cells[, n_components])}`; NaN on blocks where a variable is not defined; columns in the `read_exo` cell order. |
| `read_global_fields(filename, *files, names=None)` | `{name: (n_steps,)}` float64 histories. |

`names` selects variables (a name or a list; unknown names raise `KeyError`).

### Other entry points

- **`mesh.save("model.exo", **kwargs)`**: the `pyvista.writers` plugin for
  `.exo`, `.e`, `.ex2`, `.exii` and `.gen`; equivalent to
  `write_exo("model.exo", mesh, **kwargs)`.
- **`exovista-read`**: command-line summary of a file, or values of chosen
  variables (`exovista-read model.exo`, `exovista-read -n temperature -i -1 model.exo`;
  see `exovista-read --help`; `python -m exovista` is the same). It was
  called `exoread.py` in 1.x.
- **`exovista.legacy`**: the low-level Sandia API this project grew from
  (`exodusii_file`, `parallel_exodusii_file`, `allclose`, `similar`,
  `lineout`, `find_element_data_in_region`, ...). Importing these names from
  the top-level `exovista` package still works in 2.x but emits a
  `DeprecationWarning`; they will be removed from it in 3.0.

```python
from exovista.legacy import exodusii_file

with exodusii_file("model.exo") as f:
    print(f.num_dimensions(), f.get_element_block_ids())  # 3 [1 2]
```

## Limitations

- **Polygons and polyhedra** (`NSIDED`/`NFACED`, VTK `POLYGON`/`POLYHEDRON`)
  are not supported, nor are other higher-order cells than those in the
  [element types](#element-types) table.
- **Shells, bars and spheres are not side-set targets**: side sets are faces
  of 3D elements or edges of planar 2D elements.
- **Fixed-length time dimension**: files are written in one go, with a
  fixed-length `time_step` dimension (much faster to read with netCDF-4); time
  steps cannot be appended to an existing file, and the whole history must fit
  in memory.
- **netCDF-4 output**: files are `NETCDF4_CLASSIC` (HDF5-based, also with
  `compression_level=0`). VTK/ParaView, meshio and netCDF4-python read them
  (tested), as should any Exodus reader built on a netCDF library with
  netCDF-4 support; tools linked against a netCDF-3-only library cannot.
- **Not written by `write_exo`**: edge and face blocks, element sets,
  element attributes, QA and info records, Nemesis (parallel)
  decompositions. `read_exo` ignores edge and face blocks too (use `pv.read`
  for those).
- **Variables are floating point**: integer arrays are stored as floats.
- **meshio (<= 5.3)** cannot read a file in which an element variable is left
  out of some blocks (NaN on a whole block, see
  [variables](#variables-and-time-steps)); fill such arrays with a value if
  the file is meant for meshio.
- `read_exo` reads coordinates, never applying displacement variables.

## Migrating from 1.x

2.0 rewrites `write_exo` and `read_exo` on a new model. The main changes
(details in the [changelog](https://github.com/bijective-systems-org/exovista/blob/main/CHANGELOG.md#migrating-from-1x)):

- Default names are `block_<id>` and `side_set_<id>` (were `block_0`,
  `side_0`, ...); integer region values >= 1 are now the block ids.
- Coordinates and variables are float64 by default (`precision="single"` for
  float32).
- Arguments after `surface` are keyword-only.
- Cell arrays are written as element variables, vectors are split into
  `_X/_Y/_Z` components, and 2D meshes are written with `num_dim = 2`.
- Unsupported cells and surfaces that do not match the volume raise
  `ValueError` instead of producing a wrong file.
- `read_exo` returns cells in the original order (`original_order=False` for
  file order), recombines components (`combine_components=False` for scalars),
  adds `exo_elem_id`, and reads global variables; side sets carry
  `exo_elem_id`, `exo_side` and `cell_id` (`orig_elem_id` and `face_id` are
  deprecated aliases) and are keyed uniquely.
- The low-level API moved to `exovista.legacy`; the command-line tool is
  `exovista-read`; Python >= 3.10 is required.
- Files have a fixed-length time dimension.

## Contributing

Bug reports with a small `.exo` file (or the script that writes it) are very
welcome. See [CONTRIBUTING.md](https://github.com/bijective-systems-org/exovista/blob/main/CONTRIBUTING.md)
for the development setup, tests and the oracle-testing approach (VTK's reader
checks what exovista writes).

## Citation

If exovista helps your research, please cite it (GitHub's "Cite this
repository" button reads
[CITATION.cff](https://github.com/bijective-systems-org/exovista/blob/main/CITATION.cff)):

```bibtex
@software{exovista,
  author  = {{Interfluo} and {Bijective Systems}},
  title   = {exovista: write PyVista meshes to ExodusII and read them back},
  version = {2.0.0},
  year    = {2026},
  url     = {https://github.com/bijective-systems-org/exovista},
}
```

## License and attribution

BSD-2-Clause-Views; see [LICENSE](https://github.com/bijective-systems-org/exovista/blob/main/LICENSE).

exovista began as a fork of Sandia's
[exodusii](https://github.com/sandialabs/exodusii) Python package, whose
ExodusII core it still vendors, and inherits its copyright notice:

```
Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

SCR# 2748
```
