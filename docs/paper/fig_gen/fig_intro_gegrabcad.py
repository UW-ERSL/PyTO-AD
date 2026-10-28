"""Introduction figure (fig:intro_gegrabcad): the GE jet-engine bracket on a fine mesh.

GEGrabCAD at about 300k requested DOF (run_round2_designs.py, _cache/round2_ge.npz; its tutorial problem, MMA with AD):
  (a) the input STL with the supports (blue) and the loaded surface (red)
  (b) the optimized part: the material iso-surface of the 0/1 design after the recovery's CNN smoothing. The GE STL
      is not a closed volume (2 non-manifold edges), so the Boolean step of the STL recovery is not applied.

Run from docs/paper/fig_gen:  python fig_intro_gegrabcad.py  ->  ../figures/generated/fig_intro_gegrabcad.pdf
"""
import json
import os

import numpy as np

from _render import add_surface, camera, panel_grid, plotter, smoothed_material_surface
from _style import CACHE, LOAD_ARROW, REPO, TEXT_WIDTH, save

STL = os.path.join(REPO, "Models", "GEGrabCad", "GEGrabCAD.STL")
VIEW = dict(direction=(0.35, -1.0, 0.9), up=(0, 0, 1), distance=2.2)
FIXED_TRI = np.arange(3245, 3533)                       # pyto.examples_benchmarks.hex_structural_examples
FORCE_TRI = np.r_[5071:5287, 6351:6567]
SIGMA = float(os.environ.get("GE_SIGMA", 1.0))   # Gaussian width of the recovery's smoothing, grid spacings


def view(p, xyz):
    camera(p, xyz.min(0), xyz.max(0), **VIEW)
    lo, hi = xyz.min(0), xyz.max(0)
    p.reset_camera(bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))      # same fit in both panels


def shot(p):
    img = p.screenshot(transparent_background=True, return_img=True)
    p.close()
    return img


def main():
    import pyvista as pv
    d = dict(np.load(os.path.join(CACHE, "round2_ge.npz")))
    xyz = d["node_xyz"]
    f = d["force"].reshape(-1, 3)
    loaded = np.nonzero(np.abs(f).sum(1))[0]

    p = plotter()
    stl = pv.read(STL)
    role = np.zeros(stl.n_cells)                           # the problem's own STL triangle lists (createGEGrabCADProblem)
    role[FIXED_TRI] = 1
    role[FORCE_TRI] = 2
    add_surface(p, stl, color="#D4F2D4", role=role)
    direction = f[loaded].sum(0) / np.linalg.norm(f[loaded].sum(0))
    span = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    start = xyz[loaded].mean(0)                            # from the pin axis, out through the lugs
    p.add_mesh(pv.Arrow(start=start, direction=direction, scale=0.25 * span, tip_length=0.3,
                        tip_radius=0.11, shaft_radius=0.04), color=LOAD_ARROW, ambient=0.4)
    view(p, xyz)
    a = shot(p)

    p = plotter()
    add_surface(p, smoothed_material_surface(d, os.path.join(CACHE, f"round2_ge_surface_s{SIGMA:g}.vtp"), sigma=SIGMA))
    view(p, xyz)
    b = shot(p)

    ys, xs = np.nonzero((a[..., 3] > 0) | (b[..., 3] > 0))
    tiles = [im[ys.min() - 8:ys.max() + 8, xs.min() - 8:xs.max() + 8] for im in (a, b)]
    save(panel_grid(tiles, ["(a)", "(b)"], ncols=2, width=TEXT_WIDTH, gap=0.25), os.environ.get("GE_OUT", "fig_intro_gegrabcad"))
    print("caption:", json.loads(str(d["stats"])), "load direction", direction.round(3), "loaded nodes", len(loaded))


if __name__ == "__main__":
    main()
