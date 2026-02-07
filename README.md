# ExoVista

An extension of the Sandia exodusii library for exporting PyVista meshes to Exodus format with user-defined element blocks and side sets. This project focuses on a simpler interface for PyVista users (e.g., meshes created via meshio or PyVista's geometric utilities).

> **Note**: This code was initially a fork of: https://github.com/sandialabs/exodusii

## Features

- **Element Blocks**: Automatically split meshes by cell type and user-defined regions
- **Side Sets**: Define boundary conditions via surface meshes with region arrays
- **2D & 3D Support**: Works with Quads, Triangles, Hexahedra, Tetrahedra, Wedges, Pyramids
- **VTK Compatibility**: Handles VOXEL and PIXEL cell types (auto-permuted to Exodus ordering)
- **Node Arrays**: Save point data arrays to the Exodus file
- **Named Blocks/Sets**: Optionally provide custom names for element blocks and side sets

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

## Examples

Example scripts are located in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `example_2d_quads.py` | 2D Quad mesh with regions |
| `example_multi_block.py` | Multi-block Hex mesh split by regions |
| `example_side_sets.py` | 3D Tet mesh with side sets |
| `example_mixed_elements.py` | Mixed Hex and Tet elements |
| `save_exo.py` | Comprehensive example with multiple mesh types |

Run an example:
```shell
python examples/example_2d_quads.py
```

## Development

### Running Tests
```shell
PYTHONPATH=. python test/test_2d_save.py
PYTHONPATH=. python test/test_write_exo.py
PYTHONPATH=. python test/test_connectivity.py
```

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
