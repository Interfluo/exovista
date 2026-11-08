"""Read an ExodusII file, change it, and write it back.

Two ways:

1. VTK's reader (``pv.read`` / ``pv.ExodusIIReader``) returns a MultiBlock
   that ``write_exo`` recognises: element blocks keep their ids and names,
   side sets and node sets are written back.
2. ``exovista.read_exo`` returns one UnstructuredGrid whose ``exo_block_id``
   cell array gives the element blocks back; side sets and node sets are
   passed back explicitly (``read_side_sets`` / ``read_node_sets``).
"""

import numpy as np
import pyvista as pv

import exovista


def make_input(filename):
    """A small model, as a mesher or solver would hand it over."""
    grid = pv.ImageData(dimensions=(5, 3, 3)).cast_to_unstructured_grid()
    grid.cell_data["region"] = np.where(grid.cell_centers().points[:, 0] < 2, 1, 2)
    grid.point_data["temperature"] = 20.0 + grid.points[:, 0]
    surface = grid.extract_surface(algorithm=None)
    top = surface.cell_centers().points[:, 2] > 1.99
    surface = surface.extract_cells(np.flatnonzero(top))
    del surface.cell_data["region"]  # copied from the volume; one side set
    exovista.write_exo(
        filename, grid, surface,
        block_names={1: "copper", 2: "steel"},
        side_set_names="top",
        node_sets={"origin": [0]},
    )


def describe(label, filename):
    mesh = exovista.read_exo(filename)
    print(f"{label}: blocks {np.unique(mesh.cell_data['exo_block_name'])}, "
          f"side sets {exovista.read_side_sets(filename).keys()}, "
          f"node sets {list(exovista.read_node_sets(filename))}, "
          f"x range {mesh.bounds[:2]}, point arrays {list(mesh.point_data)}")


def main():
    make_input("example_read_back_input.exo")
    describe("input        ", "example_read_back_input.exo")

    # 1. VTK's reader: scale the model from m to mm.
    reader = pv.ExodusIIReader("example_read_back_input.exo")
    reader.node_sets.enable_all()  # pv.read loads side sets, not node sets
    data = reader.read()
    for name in ("Element Blocks", "Side Sets", "Node Sets"):
        for block in data[name]:
            block.points *= 1000.0
    exovista.write_exo("example_read_back.exo", data)
    describe("via pv.read  ", "example_read_back.exo")

    # 2. exovista's reader: add a node variable. The grid carries the block
    #    ids; the sets are passed back explicitly.
    grid, side_sets = exovista.read_exo("example_read_back_input.exo",
                                        read_side_sets=True)
    grid.point_data["pressure"] = 1.0e5 - 1.0e3 * grid.points[:, 2]
    grid.cell_data["region"] = grid.cell_data["exo_block_id"]
    block_names = dict(zip(grid.cell_data["exo_block_id"],
                           grid.cell_data["exo_block_name"]))
    faces, side_set_names = [], {}
    for name in side_sets.keys():
        sset = side_sets[name].copy()
        set_id = int(sset.field_data["exo_side_set_id"][0])
        sset.cell_data["region"] = np.full(sset.n_cells, set_id)
        faces.append(sset)
        side_set_names[set_id] = name
    exovista.write_exo(
        "example_read_back.exo", grid, pv.MultiBlock(faces).combine(),
        block_names=block_names,
        side_set_names=side_set_names,
        node_sets=exovista.read_node_sets("example_read_back_input.exo"),
    )
    describe("via read_exo ", "example_read_back.exo")


if __name__ == "__main__":
    main()
