"""Fig. 17 (fig:casestudy_stl): body-fitted re-analysis of recovered parts (Results/BodyFitted, body_fitted_study.py).

Top: ThreeHoleBracketThick, (a) voxel design (density > 0.5), (b) recovered STL, (c) its 10-node tetrahedral mesh
(corner nodes drawn), same view as Fig. 1. Bottom: the stress-designed L-bracket, von Mises stress on one colour
scale, (d) voxel design (element centres), (e) recovered part (largest of the 4 Gauss-point values per tetrahedron).
The scale stops at 4 MPa (the voxel design peaks at 3.1 MPa); the colour bar's arrow marks higher values, which the
recovered part reaches at a sharp edge (28.8 MPa).

Run from paper/fig_gen:  python fig17_casestudy.py  ->  ../figures/generated/fig17_casestudy.pdf
"""
import os

import numpy as np

from _render import EDGE, LIGHT, add_design, camera, hex_grid, plotter, shot
from _style import INK, MATERIAL, REPO, TEXT_WIDTH, save

BF = os.path.join(REPO, "Results", "BodyFitted")
VMAX = 4.0                                     # MPa
CMAP = "viridis"
VIEW_3H, UP_3H = (0.42, -0.62, 1.0), (0, 1, 0)
VIEW_LB, UP_LB = (-0.85, 0.55, 1.0), (0, 1, 0)


def threehole():
    import pyvista as pv
    d = np.load(os.path.join(BF, "ThreeHoleBracket", "design.npz"))
    xyz, elems = d["node_xyz"].astype(float), d["elems"]
    lo, hi = xyz.min(0), xyz.max(0)
    out = []
    p = plotter()
    add_design(p, hex_grid(xyz, elems), d["final"], elems)
    camera(p, lo, hi, VIEW_3H, UP_3H)
    out.append(shot(p))
    p = plotter()
    p.add_mesh(pv.read(os.path.join(BF, "ThreeHoleBracket", "recovered.stl")), color=MATERIAL, smooth_shading=True,
               split_sharp_edges=True, feature_angle=30, **LIGHT)
    camera(p, lo, hi, VIEW_3H, UP_3H)
    out.append(shot(p))
    p = plotter()
    p.add_mesh(pv.read(os.path.join(BF, "ThreeHoleBracket", "tet_mesh.vtu")).extract_surface(), color=MATERIAL,
               show_edges=True, edge_color=EDGE, line_width=0.7, **LIGHT)
    camera(p, lo, hi, VIEW_3H, UP_3H)
    out.append(shot(p))
    return out


def lbracket():
    case = os.path.join(BF, "LBracket_stress")
    d = np.load(os.path.join(case, "design.npz"))
    xyz, elems = d["node_xyz"].astype(float), d["elems"]
    lo, hi = xyz.min(0), xyz.max(0)
    bar = dict(cmap=CMAP, clim=(0, VMAX), show_scalar_bar=False, **LIGHT)
    out = []
    p = plotter()
    g = hex_grid(xyz, elems)
    g.cell_data["vm"] = np.load(os.path.join(case, "hex_von_mises.npy")) / 1e6
    p.add_mesh(g.extract_cells(np.nonzero(d["final"] > 0.5)[0]), scalars="vm", **bar)
    camera(p, lo, hi, VIEW_LB, UP_LB)
    out.append(shot(p))
    import pyvista as pv
    t = pv.read(os.path.join(case, "tet_mesh.vtu"))
    t.cell_data["vm"] = np.load(os.path.join(case, "tet_von_mises_elem.npy")) / 1e6
    p = plotter()
    p.add_mesh(t.extract_surface(), scalars="vm", **bar)
    camera(p, lo, hi, VIEW_LB, UP_LB)
    out.append(shot(p))
    return out


def main():
    import matplotlib.pyplot as plt
    top, bottom = threehole(), lbracket()
    W = TEXT_WIDTH
    gap, lab = 0.1, 0.2
    w_top = (W - 2 * gap) / 3
    h_top = max(i.shape[0] / i.shape[1] for i in top) * w_top
    w_bot = (W - 0.75 - gap) / 2                       # room for the colour bar on the right
    h_bot = max(i.shape[0] / i.shape[1] for i in bottom) * w_bot
    H = h_top + h_bot + 2 * lab + 0.12
    fig = plt.figure(figsize=(W, H))

    def place(img, x, y, w, h, label):
        ax = fig.add_axes([x / W, y / H, w / W, h / H])
        ax.imshow(img, interpolation="antialiased")
        ax.set_anchor("S")
        ax.axis("off")
        fig.text((x + w / 2) / W, (y - 0.03) / H, label, ha="center", va="top", fontsize=8, color=INK)

    y_top = H - h_top
    for k, (img, label) in enumerate(zip(top, ["(a)", "(b)", "(c)"])):
        place(img, k * (w_top + gap), y_top, w_top, h_top, label)
    y_bot = lab
    for k, (img, label) in enumerate(zip(bottom, ["(d)", "(e)"])):
        place(img, k * (w_bot + gap), y_bot, w_bot, h_bot, label)
    cax = fig.add_axes([(2 * w_bot + gap + 0.12) / W, (y_bot + 0.12 * h_bot) / H, 0.08 / W, 0.76 * h_bot / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, VMAX)), cax=cax, extend="max",
                      extendfrac=0.06)
    cb.set_label("von Mises stress (MPa)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    save(fig, "fig17_casestudy")


if __name__ == "__main__":
    main()
