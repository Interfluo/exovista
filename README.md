# ExoVista

An extension of the Sandia exodusii library for exporting PyVista meshes to Exodus format with user-defined element blocks and side sets. This project focuses on a simpler interface for PyVista users (e.g., meshes created via meshio or PyVista’s geometric utilities).

# Why ExoVista?

While PyVista can read and write Exodus files via meshio, this support is limited. For example, in the SIERRA toolset, element blocks are essential for defining material zones, and side sets are commonly used to assign boundary conditions. Currently, meshio cannot save side sets or user-defined element blocks.

ExoVista makes it easy to define side sets and element blocks by assigning a "region" cell array to the surface and volume meshes. Internally, the library automatically splits element blocks by cell type before exporting to an Exodus file.

## Usage
basic imports
```python
import exovista
import numpy as np
import pyvista as pv
```

Load the letter_a PyVista example (a tetrahedral volume mesh).
After centering the mesh, extract the surface and assign cell-wise region arrays that define element blocks and side sets in the exported Exodus file.
```python
def export_tetra_mesh():
    volume = pv.examples.download_letter_a()
    volume.points -= volume.center
    volume["tag"] = 1 * (volume.cell_centers().points[:, 0] > 0)
    volume.plot(show_edges=True, categories=True, parallel_projection=True, text="element blocks")

    surface = volume.extract_surface()
    surface["tag"] = 1 * (surface.cell_centers().points[:, 2] > 0)
    surface.plot(show_edges=True, categories=True, parallel_projection=True, text="side sets")

    exovista.write_exo("tetra.exo", volume, surface, "tag")
    return None
```
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/38b84cb6-7b45-43fa-82a4-1ae72bd81b14" width="48%">
  <img src="https://github.com/user-attachments/assets/1140ffbf-c8b3-4e88-8db6-47c421d3fccc" width="48%">
</div>

This example constructs a structured cylindrical volume made of hexahedral cells.
As before, extract the surface and assign region arrays before exporting.
```python
def export_hex_mesh():
    volume = pv.CylinderStructured(radius=np.linspace(1, 2, 5)).cast_to_unstructured_grid()
    surface = volume.extract_surface()

    volume["region"] = 1*(volume.cell_centers().points[:, 1] > 0) + 2*(volume.cell_centers().points[:, 2] > 0)
    volume.plot(show_edges=True, categories=True, parallel_projection=True, text="element blocks")


    surface["region"] = 1*(surface.cell_centers().points[:, 0] > 0)
    surface.plot(show_edges=True, categories=True, parallel_projection=True, text="side sets")

    exovista.write_exo("hex.exo", volume, surface)
    exovista.write_exo("hex_no_sides.exo", volume, None)
    return None
```
<div style="display: flex; gap: 10px;">
  <img src="https://github.com/user-attachments/assets/2cdf9a4a-fdc6-491b-9e3b-724d2d817c92" width="48%">
  <img src="https://github.com/user-attachments/assets/7400879b-49a3-45d6-8d4f-3f4297fbf166" width="48%">
</div>

## Install
install from pypi
```shell
pip install exovista
```
install editable version
```shell
git clone https://github.com/Interfluo/exovista.git
cd exovista
pip install -e .
```

## Copyright
This repo is a fork of https://github.com/sandialabs/exodusii and therefore inherits the below copyright: 

```
Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC
(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

SCR# 2748
```
