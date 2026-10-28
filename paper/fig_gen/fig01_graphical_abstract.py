"""Fig. 1 (fig:graphical_abstract): from an STL part to a smooth optimized part, on a fine mesh.

ThreeHoleBracketThick at about 200k DOF (run_round2_designs.py, _cache/round2_threehole.npz; the tutorial problem:
supports on the two left holes, load on the right hole, volume fraction <= 0.25, MMA with AD gradients):
  (a) the design-domain STL with the supports (blue) and the loaded surface (red)
  (b) the FE result on the solid domain: von Mises stress on the hexahedral mesh
  (c) the optimized design (density > 0.5), elements touching the supports in blue
  (d) the smooth STL recovered from it, as the GUI does it (CNN smoothing, iso-surface, Boolean); supports and
      load faces as in (a)

Run from paper/fig_gen:  python fig01_graphical_abstract.py  ->  ../figures/generated/fig01_graphical_abstract.pdf
"""
import json
import os

import numpy as np

from _render import add_design, add_surface, bc_faces, camera, hex_grid, panel_grid, plotter, recovered_part, transfer_roles
from _style import CACHE, INK, LOAD_ARROW, REPO, TEXT_WIDTH, save

STL = os.path.join(REPO, "Models", "ThreeHoleBracket", "ThreeHoleBracketThick.STL")
VIEW = dict(direction=(0.42, -0.62, 1.0), up=(0, 1, 0), distance=2.2)
VM_CMAP = "viridis"


def _cam(p, xyz):
    camera(p, xyz.min(0), xyz.max(0), **VIEW)
    lo, hi = xyz.min(0), xyz.max(0)
    p.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))      # fit the domain, not this panel's part


def shot(p):
    # uncropped: all panels share one camera and are cropped with one box in main(), so they keep one scale
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return img


def renders(d):
    import pyvista as pv
    xyz, elems = d["node_xyz"], d["elems"]
    tol = 1.0 * float(np.max(d["elem_size"]))
    grid = hex_grid(xyz, elems)
    out = {}

    p = plotter()
    ref = bc_faces(pv.read(STL), xyz, d["fixed"], d["force"], tol, split=False, skip_axis=2)   # the hole walls
    add_surface(p, ref, color="#D4F2D4", role=ref.cell_data["role"])
    f = d["force"].reshape(-1, 3)
    loaded = np.nonzero(np.abs(f).sum(1))[0]
    direction = f[loaded].sum(0) / np.linalg.norm(f[loaded].sum(0))
    start = xyz[loaded].mean(0)
    start[2] = xyz[:, 2].max() + 0.002                     # from the hole centre on the front face
    p.add_mesh(pv.Arrow(start=start, direction=direction, scale=0.03, tip_length=0.3, tip_radius=0.11,
                        shaft_radius=0.04), color=LOAD_ARROW, ambient=0.4)
    _cam(p, xyz)
    out["stl"] = shot(p)

    p = plotter()
    g = grid.copy()
    vm = d["vm_solid"] / 1e6
    g.cell_data["vm"] = vm
    clim = (0.0, float(np.percentile(vm, 99.5)))
    p.add_mesh(g, scalars="vm", cmap=VM_CMAP, clim=clim, show_scalar_bar=False, show_edges=True, edge_color="#404040",
               line_width=0.3, ambient=0.25, diffuse=0.8)
    _cam(p, xyz)
    out["fea"] = shot(p)

    p = plotter()
    add_design(p, grid, d["x"], elems, fixed=d["fixed"], edges=True)
    _cam(p, xyz)
    out["design"] = shot(p)

    p = plotter()
    part = recovered_part(d, STL, os.path.join(CACHE, "round2_threehole_part.vtp"))
    add_surface(p, part, role=transfer_roles(part, ref, 1e-3 * tol))
    _cam(p, xyz)
    out["part"] = shot(p)
    return out, clim


def main():
    import matplotlib as mpl
    d = dict(np.load(os.path.join(CACHE, "round2_threehole.npz")))
    stats = json.loads(str(d["stats"]))
    imgs, clim = renders(d)
    order = ["stl", "fea", "design", "part"]
    # one horizontal range for all panels (one scale), one vertical range per row (no empty band under the design)
    _, xs = np.nonzero(np.any([imgs[k][..., 3] > 0 for k in order], axis=0))
    tiles = []
    for row in (order[:2], order[2:]):
        ys, _ = np.nonzero(np.any([imgs[k][..., 3] > 0 for k in row], axis=0))
        tiles += [imgs[k][ys.min() - 8:ys.max() + 8, xs.min() - 8:xs.max() + 8] for k in row]
    right = 0.62                                                     # margin for the colorbar of (b), in
    fig = panel_grid(tiles, ["(a)", "(b)", "(c)", "(d)"], ncols=2, width=TEXT_WIDTH - right, gap=0.3)
    s = (TEXT_WIDTH - right) / TEXT_WIDTH
    fig.set_size_inches(TEXT_WIDTH, fig.get_size_inches()[1])
    for ax in fig.axes:
        x0, y0, ww, hh = ax.get_position().bounds
        ax.set_position([x0 * s, y0, ww * s, hh])
    for t in fig.texts:
        t.set_position((t.get_position()[0] * s, t.get_position()[1]))
    x0, y0, ww, hh = fig.axes[1].get_position().bounds              # colorbar to the right of (b)
    cax = fig.add_axes([x0 + ww + 0.01, y0 + 0.12 * hh, 0.014, 0.76 * hh])
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(*clim), VM_CMAP), cax=cax, extend="max")
    cb.set_label("von Mises stress (MPa)", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig01_graphical_abstract")
    print("caption:", stats, f"von Mises colour range 0-{clim[1]:.1f} MPa (99.5th percentile), max "
          f"{d['vm_solid'].max() / 1e6:.1f} MPa, max |u| {d['umag_solid'].max() * 1e6:.2f} um")


if __name__ == "__main__":
    main()
