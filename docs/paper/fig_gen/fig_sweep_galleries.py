"""Figs. 14 and 15 (fig:gallery_nhg; the thermo-structural figure; fig:intro_gegrabcad moved to fig_intro_gegrabcad.py): designs from the benchmark sweep of 2026-10-15
(fig:gallery_nhg (d): the converged stress design of the review runs, 2026-10-24).

The sweep saved images, not densities, and rerunning it would not reproduce the tabulated values to the last digit
(multithreaded round-off), so its own renders are used, in PyTO's own colours, cropped to the part (no title, no
axis triad). Elements with density above 0.5 are drawn, as in the sweep.

Run from docs/paper/fig_gen:  python fig_sweep_galleries.py  ->  ../figures/generated/fig14_*, fig15_thermostructural
"""
import os

import numpy as np

from _render import crop_sweep_png, panel_grid, sweep_part_box
from _style import REPO, TEXT_WIDTH, save

SWEEP = os.path.join(REPO, "Results", "Results_2026-10-15")


def img(group, name):
    return crop_sweep_png(os.path.join(SWEEP, group, "DENSITYMMA", f"{name}.png"))


# (d): the stress-constrained L-bracket (minimum volume, max von Mises <= 400 MPa), converged in 229 iterations with
# MaxIterations 400 (review item C5; pyto.examples_benchmarks.review_designs, tag 2026-10-24): PyTO's own render of the
# 0/1 design, the same renderer and image size as the sweep's
C5 = os.path.join(REPO, "Results", "ReviewDesigns_2026-10-24", "LBracketTopLoad_Vol_Stress_400it", "topology.png")
fig = panel_grid([img("Structural-NonCompliance", "Inverter"), img("Structural-BodyForce", "CentrifugalPlate"),
                  img("Structural-BodyForce", "GravityPlate"), crop_sweep_png(C5)],
                 ["(a)", "(b)", "(c)", "(d)"], ncols=4, width=TEXT_WIDTH,
                 gap=0.15)
save(fig, "fig14_gallery_no_handwritten_gradient")

# Thermo-structural (replaces the former BiClamp and heated-MBB figures): (a), (b) BiClamp at a temperature rise of
# 0 and 1 K; (c), (d) the half MBB beam (Ooms et al. set-up) without and with the hot lower edge. Each pair is cropped
# with one box (the union of its two parts), so the two renders of a pair keep the sweep's common camera and scale.
# Compliances (Table biclamp; Sect. thermo-structural) are written on the tiles.
PAIRS = [(("BiClampDT0", "0.1089 J"), ("BiClamp", "0.6761 J")),
         (("MBBBeamNoHeat", "38.02 J"), ("MBBBeam", "276.7 J"))]


def pair(names):
    paths = [os.path.join(SWEEP, "ThermoStructural", "DENSITYMMA", f"{n}.png") for n in names]
    boxes = np.array([sweep_part_box(p) for p in paths])
    box = (boxes[:, 0].min(), boxes[:, 1].max(), boxes[:, 2].min(), boxes[:, 3].max())
    imgs = [crop_sweep_png(p, box=box) for p in paths]
    head = int(0.075 * imgs[0].shape[1])            # white headroom for the value (8 pt at print), so it never covers the part
    return [np.pad(im, ((head, 0), (0, 0), (0, 0)), constant_values=255) for im in imgs]


imgs = [im for pr in PAIRS for im in pair([n for n, _ in pr])]
fig = panel_grid(imgs, ["(a)", "(b)", "(c)", "(d)"], ncols=2, width=TEXT_WIDTH, gap=0.25)
for ax, (_name, value) in zip(fig.axes, [v for pr in PAIRS for v in pr]):
    im = ax.get_images()[0].get_array()                                    # centred in the white headroom
    ax.text(im.shape[1] / 2, int(0.075 * im.shape[1]) / 2, f"$J$ = {value}", ha="center", va="center", fontsize=8)
save(fig, "fig15_thermostructural")

# Tutorial problems at a glance (Sect. 8.1): eight compliance designs of the sweep, PyTO's own renders (elements with
# density above 0.5), each cropped to its part. Row 1: 3D problems; row 2: EdgeCantilever (3D) and three 2.5D STL
# problems. GEGrabCAD, ThreeHoleBracketThick and the thermal problems are shown elsewhere or left out.
GALLERY = [("Structural-Compliance3D", "Table"), ("Structural-Compliance3D", "LBracketThickMidLoad"),
           ("Structural-Compliance3D", "LBracketThickTopLoad"), ("Structural-Compliance3D", "Multiload"),
           ("Structural-Compliance3D", "EdgeCantilever"), ("Structural-Compliance2.5D_2", "ThreeHoleBracket"),
           ("Structural-Compliance2.5D_2", "TorquePlate"), ("Structural-Compliance2.5D_1", "Mitchell")]
fig = panel_grid([img(g, n) for g, n in GALLERY], [f"({chr(97 + i)})" for i in range(len(GALLERY))], ncols=4,
                 width=TEXT_WIDTH, gap=0.15)
save(fig, "fig_sweep_gallery_compliance")
