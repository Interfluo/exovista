"""Render docs/images/hero.png (the README banner).

The mesh is written with exovista and read back with exovista before it is
drawn, so the picture shows what is in the file: two element blocks, three
side sets and a time-varying node field.

Run from the repository root (it needs an OpenGL context; on a headless
Linux machine use ``xvfb-run -a python docs/images/make_hero.py``)::

    python docs/images/make_hero.py
"""

import os
import tempfile

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import exovista

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "hero.png")

WIDTH, HEIGHT = 1200, 360  # the 3D panels
BAND, FOOT = 44, 82  # title band above, legend band below
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#c9c8c3"
# The first three slots of a colour-blind-checked categorical palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CAMERA = [(5.6, -6.7, 6.3), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
SHADING = dict(show_edges=True, edge_color="#ffffff", line_width=1.0,
               ambient=0.35, diffuse=0.75, specular=0.0)
FONTS = (
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
)


def font(size):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def build_mesh():
    """A thick pipe segment of hexahedra with a quarter cut away."""
    pipe = pv.CylinderStructured(
        radius=np.linspace(1.0, 2.0, 7),
        height=2.5,
        theta_resolution=49,
        z_resolution=11,
        direction=(0.0, 0.0, 1.0),
    ).cast_to_unstructured_grid()
    centers = pipe.cell_centers().points
    keep = ~((centers[:, 0] > 0) & (centers[:, 1] < 0))
    pipe = pipe.extract_cells(np.flatnonzero(keep)).clean()
    pipe.points[:, 2] -= pipe.bounds[4]
    radius = np.hypot(*pipe.cell_centers().points[:, :2].T)
    pipe.cell_data["region"] = np.where(radius < 1.5, "liner", "jacket")
    return pipe


def side_sets_of(pipe):
    """Top, inner wall and the two cut planes (outer wall and bottom left out)."""
    surface = pipe.extract_surface(algorithm=None)
    c = surface.cell_centers().points
    r = np.hypot(c[:, 0], c[:, 1])
    region = np.full(surface.n_cells, "", dtype=object)
    region[np.isclose(c[:, 2], pipe.bounds[5], atol=1e-6)] = "top"
    bottom = np.isclose(c[:, 2], pipe.bounds[4], atol=1e-6)
    side = (region == "") & ~bottom
    region[side & (r < 1.05)] = "inner_wall"
    region[side & (r > 1.05) & (r < 1.95)] = "cut"
    keep = np.flatnonzero(region != "")
    faces = surface.extract_cells(keep)
    faces.cell_data["region"] = region[keep].astype(str)
    return faces


def render(grid, side_sets):
    """The three 3D panels, without any text."""
    p = pv.Plotter(shape=(1, 3), window_size=(WIDTH, HEIGHT), off_screen=True,
                   border=False)
    p.set_background(SURFACE)

    # 1. element blocks
    p.subplot(0, 0)
    names = np.asarray(grid.cell_data["exo_block_name"])
    for name, color in (("liner", BLUE), ("jacket", ORANGE)):
        p.add_mesh(grid.extract_cells(np.flatnonzero(names == name)), color=color,
                   **SHADING)
    p.camera_position = CAMERA

    # 2. side sets over the rest of the boundary (no coplanar overlap)
    p.subplot(0, 1)
    boundary = grid.extract_surface(algorithm=None)
    covered = np.vstack([side_sets[k].extract_surface(algorithm=None)
                         .cell_centers().points for k in side_sets.keys()])
    dist, _ = cKDTree(covered).query(boundary.cell_centers().points)
    p.add_mesh(boundary.extract_cells(np.flatnonzero(dist > 1e-9)), color=MUTED,
               **SHADING)
    for key, color in (("top", BLUE), ("inner_wall", ORANGE), ("cut", AQUA)):
        p.add_mesh(side_sets[key].extract_surface(algorithm=None), color=color,
                   **SHADING)
    p.camera_position = CAMERA

    # 3. a time-varying node field
    p.subplot(0, 2)
    p.add_mesh(grid, scalars="temperature", cmap="Oranges", clim=(20, 100),
               show_scalar_bar=False, **SHADING)
    p.camera_position = CAMERA

    image = p.screenshot(None, transparent_background=False, scale=2,
                         return_img=True)
    p.close()
    return Image.fromarray(image).resize((WIDTH, HEIGHT), Image.LANCZOS)


def annotate(panels, n_steps):
    canvas = Image.new("RGB", (WIDTH, BAND + HEIGHT + FOOT), SURFACE)
    canvas.paste(panels, (0, BAND))
    draw = ImageDraw.Draw(canvas)
    third = WIDTH // 3
    titles = ("Element blocks", "Side sets", f"Node variable, step 4 of {n_steps}")
    for i, title in enumerate(titles):
        draw.text((i * third + 20, 14), title, fill=TEXT, font=font(20))

    def legend(x0, entries):
        y = BAND + HEIGHT + 6
        for label, color in entries:
            draw.rounded_rectangle((x0, y + 3, x0 + 14, y + 17), radius=3, fill=color)
            draw.text((x0 + 22, y), label, fill=TEXT_2, font=font(15))
            y += 24

    legend(20, [("block 'liner'", BLUE), ("block 'jacket'", ORANGE)])
    legend(third + 20, [("side set 'top'", BLUE), ("side set 'inner_wall'", ORANGE),
                        ("side set 'cut'", AQUA)])

    # stepped colour bar of the node variable
    from matplotlib import colormaps

    cmap = colormaps["Oranges"]
    x0, y0, step, n = 2 * third + 20, BAND + HEIGHT + 32, 24, 8
    for i in range(n):
        rgb = tuple(int(255 * v) for v in cmap((i + 0.5) / n)[:3])
        draw.rectangle((x0 + i * step, y0, x0 + (i + 1) * step - 2, y0 + 12), fill=rgb)
    small = font(13)
    draw.text((x0, y0 - 26), "temperature", fill=TEXT_2, font=font(15))
    draw.text((x0, y0 + 16), "20", fill=TEXT_2, font=small)
    draw.text((x0 + n * step - 2, y0 + 16), "100", fill=TEXT_2, font=small, anchor="ra")
    return canvas


def main():
    pipe = build_mesh()
    faces = side_sets_of(pipe)
    times = np.linspace(0.0, 1.0, 6)
    z = pipe.points[:, 2]
    r = np.hypot(pipe.points[:, 0], pipe.points[:, 1])
    temperature = np.array(
        [20.0 + 80.0 * np.exp(-2.0 * (z - 2.5 * t) ** 2) * (2.0 - r) for t in times]
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pipe.exo")
        exovista.write_exo(path, pipe, faces, times=times,
                           node_fields={"temperature": temperature})
        grid, side_sets = exovista.read_exo(path, time_step=4, read_side_sets=True)

    image = annotate(render(grid, side_sets), len(times))
    # A 256-colour palette makes the file a third of the size.
    image.quantize(colors=256, method=Image.Quantize.FASTOCTREE,
                   dither=Image.Dither.FLOYDSTEINBERG).save(OUT, optimize=True)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB, "
          f"{image.width}x{image.height})")


if __name__ == "__main__":
    main()
