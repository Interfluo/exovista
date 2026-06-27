# ExoVista

An extension of the Sandia exodusii library for exporting PyVista meshes to Exodus format with user-defined element blocks and side sets. This project focuses on a simpler interface for PyVista users (e.g., meshes created via meshio or PyVista's geometric utilities).

> **Note**: This code was initially a fork of: https://github.com/sandialabs/exodusii

## Features

- **Element Blocks**: Automatically split meshes by cell type and user-defined regions
- **Side Sets**: Define boundary conditions via surface meshes with region arrays
- **2D & 3D Support**: Works with Quads, Triangles, Hexahedra, Tetrahedra, Wedges, Pyramids
- **VTK Compatibility**: Handles VOXEL and PIXEL cell types (auto-permuted to Exodus ordering)
- **Node Arrays**: Save point data arrays to the Exodus file
- **Time-Varying Fields**: Write a time history of node and element results (transient data)
- **Compressed Output**: Array data is zlib/deflate compressed by default (netCDF4), dramatically shrinking file size
- **Named Blocks/Sets**: Optionally provide custom names for element blocks and side sets
- **Read Back to PyVista**: Load an Exodus file straight into a PyVista `UnstructuredGrid` with `read_exo` (inverse of `write_exo`)

## Installation

### From PyPI
```shell
pip install exovista
```

### Editable Install (Development)
```shell
git clone https://github.com/Interfluo/exovista.git
cd exovista
pip install -e .
```

## Quick Start

```python
import exovista
import numpy as np
import pyvista as pv

# Load a tetrahedral mesh
volume = pv.examples.download_letter_a()
volume.points -= volume.center

# Assign regions for element blocks
volume["region"] = 1 * (volume.cell_centers().points[:, 0] > 0)

# Extract surface and assign regions for side sets
surface = volume.extract_surface()
surface["region"] = 1 * (surface.cell_centers().points[:, 2] > 0)

# Write to Exodus file
exovista.write_exo("output.exo", volume, surface, region_key="region")
```

## API Reference

### `exovista.write_exo`

```python
write_exo(
    filename: str,
    volume: pv.UnstructuredGrid,
    surface: pv.PolyData = None,
    region_key: str = "region",
    block_names: list[str] = None,
    side_set_names: list[str] = None,
    save_node_arrays: bool = True,
    times: np.ndarray | list = None,
    node_fields: dict = None,
    element_fields: dict = None,
)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `str` | Output Exodus file path |
| `volume` | `pv.UnstructuredGrid` | Volume mesh containing cells |
| `surface` | `pv.PolyData` | Surface mesh for side sets (optional) |
| `region_key` | `str` | Cell data array name used to split blocks/sets |
| `block_names` | `list[str]` | Custom names for element blocks |
| `side_set_names` | `list[str]` | Custom names for side sets |
| `save_node_arrays` | `bool` | Whether to save point data arrays |
| `times` | `array_like` | Time values, one per time step (enables a time history) |
| `node_fields` | `dict` | `name -> array (n_steps, n_nodes)` time-varying node results |
| `element_fields` | `dict` | `name -> array (n_steps, n_cells)` time-varying element results |

### Time-Varying Fields

Supply `times` together with `node_fields` and/or `element_fields` to write a
transient result history. Each field array carries one row per time value;
element fields are given in the original `volume` cell order and are
automatically distributed to the correct element blocks. Static node arrays
(from `save_node_arrays`) are broadcast across every time step.

```python
import numpy as np

times = np.linspace(0.0, 1.0, 11)                 # 11 time steps

# (n_steps, n_nodes) node field and (n_steps, n_cells) element field
temperature = np.array([t * volume.points[:, 0] for t in times])
energy = np.outer(times, np.arange(volume.n_cells))

exovista.write_exo(
    "transient.exo",
    volume,
    times=times,
    node_fields={"temperature": temperature},
    element_fields={"energy": energy},
)
```

For a single time step a 1D array (length `n_nodes` or `n_cells`) is also
accepted. The resulting file animates over time in ParaView.

### `exovista.read_exo`

The inverse of `write_exo`: load an ExodusII database into a PyVista
`UnstructuredGrid`.

```python
read_exo(
    filename: str,
    *files: str,
    time_step: int = -1,
    read_node_variables: bool = True,
    read_element_variables: bool = True,
    read_side_sets: bool = False,
) -> pv.UnstructuredGrid | tuple[pv.UnstructuredGrid, pv.MultiBlock]
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | `str` | Input Exodus file path |
| `*files` | `str` | Additional file parts of a parallel decomposition |
| `time_step` | `int` | 1-based time step to sample variables at; negative indexes from the end (`-1` = last step) |
| `read_node_variables` | `bool` | Load node variables into `point_data` |
| `read_element_variables` | `bool` | Load element variables into `cell_data` |
| `read_side_sets` | `bool` | Also reconstruct side sets and return them alongside the grid |

The returned grid reconstructs node coordinates (2D meshes are embedded in 3D
with `z = 0`) and all element blocks, concatenated in block order. Each cell
records its originating Exodus block via `cell_data["exo_block_id"]` and
`cell_data["exo_block_name"]`. Node and element variables are sampled at the
requested `time_step`, and the full time history is available in
`grid.field_data["times"]` (`grid.field_data["num_dim"]` holds the original
spatial dimension).

When `read_side_sets=True`, a second value is returned: a `pv.MultiBlock` with
one `pv.PolyData` per side set (named after the side set), whose faces are
rebuilt over the shared volume nodes. Each face carries `orig_elem_id` (1-based
Exodus element id) and `face_id` (1-based local side id) cell data. 3D element
faces become polygons; 2D element edges become line cells.

```python
import exovista

mesh = exovista.read_exo("output.exo")          # last time step
mesh_t0 = exovista.read_exo("output.exo", time_step=1)  # first time step

# Also reconstruct side sets
mesh, side_sets = exovista.read_exo("output.exo", read_side_sets=True)
top_faces = side_sets["side_0"]

# Parallel decomposition (pass each part, or use exovista.File for globbing)
joined = exovista.read_exo("run.exo.4.0", "run.exo.4.1",
                           "run.exo.4.2", "run.exo.4.3")
```

### Output Compression

When writing netCDF4 files (the default), array variables are stored with
zlib/deflate compression, which typically shrinks output files by an order of
magnitude with negligible effect on read performance. The behavior is
controlled by an environment variable:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXODUSII_COMPRESSION_LEVEL` | `1` | Deflate level. `1` is fast and captures most of the savings; `9` is maximum compression; `0` disables it (legacy uncompressed output). |

Compression only applies to the netCDF4 backend; it is skipped automatically
for the netCDF3 fallback. Files remain standard ExodusII and are readable by
ParaView, Sierra, and other Exodus tools without any changes.

## Examples

Example scripts are located in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `example_2d_quads.py` | 2D Quad mesh with regions |
| `example_multi_block.py` | Multi-block Hex mesh split by regions |
| `example_side_sets.py` | 3D Tet mesh with side sets |
| `example_mixed_elements.py` | Mixed Hex and Tet elements |
| `example_time_fields.py` | Time-varying node and element fields (transient data) |
| `example_read_exo.py` | Round-trip: write a mesh then read it back with `read_exo` |
| `save_exo.py` | Comprehensive example with multiple mesh types |

Run an example:
```shell
python examples/example_2d_quads.py
```

## Development

### Running Tests
The test suite uses `pytest`:
```shell
pip install -e . pytest
pytest test/
```

Tests that depend on upstream ExodusII reference files not redistributed with
this fork (e.g. `noh.exo`, `edges.base.exo`) are skipped automatically when the
files are absent. Continuous integration runs `flake8` and the full test suite
across Python 3.10–3.12 (see `.github/workflows/ci.yml`).

### Cleaning Generated Files
```shell
python clean.py
```

This removes generated `.exo` files, `__pycache__` directories, and temporary test outputs.

## Copyright

This repo is a fork of https://github.com/sandialabs/exodusii and therefore inherits the below copyright: 

```
Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

SCR# 2748
```
