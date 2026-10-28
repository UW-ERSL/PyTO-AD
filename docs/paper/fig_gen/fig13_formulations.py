"""Fig. 13 (fig:formulations): one L-bracket, four objective strings (MMA, AD), delivered designs.

From the showcase runs of pyto.examples_benchmarks.ad_showcase (Results/AD_Showcase/<case>/history.npz: final density, mesh, node labels, load); same
camera for all four, supports (top face) in blue, the load in red, elements with density above 0.5 drawn.

Run from docs/paper/fig_gen:  python fig13_formulations.py  ->  ../figures/generated/fig13_formulations.pdf
"""
import os

import numpy as np

from _render import README_VIEW, add_design, add_load_arrows, camera, hex_grid, panel_grid, plotter, shot
from _style import REPO, TEXT_WIDTH, save

SHOWCASE = os.path.join(REPO, "Results", "AD_Showcase")
CASES = [("LBracket_compliance", "(a)"), ("LBracket_stress", "(b)"), ("LBracket_displacement", "(c)"),
         ("LBracket_mass", "(d)")]           # formulas: in the caption (FIGURE_GUIDE.md)
VIEW = README_VIEW                            # the README GIFs' view of the L-bracket


def render(case):
    d = np.load(os.path.join(SHOWCASE, case, "history.npz"))
    xyz, elems = d["node_xyz"].astype(float), d["elems"]
    p = plotter((1800, 1700))
    add_design(p, hex_grid(xyz, elems), d["final"], elems, fixed=np.nonzero(d["labels"] == 1)[0])
    add_load_arrows(p, xyz, d["force"], n_max=4)
    camera(p, xyz.min(0), xyz.max(0), VIEW, (0, 1, 0))
    return shot(p, pad=10)


def main():
    imgs = [render(c) for c, _ in CASES]
    h = max(i.shape[0] for i in imgs)
    w = max(i.shape[1] for i in imgs)
    imgs = [np.pad(i, ((h - i.shape[0], 0), ((w - i.shape[1]) // 2, w - i.shape[1] - (w - i.shape[1]) // 2),
                       (0, 0))) for i in imgs]
    fig = panel_grid(imgs, [lab for _, lab in CASES], ncols=4, width=TEXT_WIDTH, gap=0.08)
    save(fig, "fig13_formulations")


if __name__ == "__main__":
    main()
