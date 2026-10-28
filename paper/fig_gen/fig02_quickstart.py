"""Fig. 2 (fig:quickstart): the three-hole bracket of Listings 1 and 2, delivered designs.

Runs the two tested scripts (docs/examples/quickstart.py and quickstart_displacement.py, in fig_gen/_cache) with only
nDOFDesired raised from 20000 to N_DOF, so the thin members are resolved; keeps their final densities, and renders both the same way: one 3D view of the smooth STL
recovered from each design (as the GUI does it), supports in blue, the load in red. Also records each run's time and, for the caption, the compliance of both designs and the
share of elements that differ. Cached in _cache/quickstart_*_<N_DOF>.npz.

Run from paper/fig_gen:  python fig02_quickstart.py  ->  ../figures/generated/fig02_quickstart.pdf
"""
import contextlib
import io
import os
import time

import numpy as np

from _render import add_hole_arrow, add_surface, bc_faces, camera, panel_grid, plotter, recovered_part, shot, transfer_roles
from _style import CACHE, REPO, TEXT_WIDTH, save

STL = os.path.join(REPO, "Models", "ThreeHoleBracket", "ThreeHoleBracketThick.STL")
N_DOF = 80000                                     # the listings use 20000; finer for the figure only
SCRIPTS = {"compliance": "quickstart.py", "displacement": "quickstart_displacement.py"}


def run(key):
    out = os.path.join(CACHE, f"quickstart_{key}_{N_DOF}.npz")
    if not os.path.exists(out):
        os.makedirs(CACHE, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(CACHE)
        try:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()) as log:
                path = os.path.join(REPO, "docs", "examples", SCRIPTS[key])
                src = open(path).read()
                assert src.count("nDOFDesired=20000") == 1, path
                g = {"__name__": "__main__", "__file__": path}
                exec(compile(src.replace("nDOFDesired=20000", f"nDOFDesired={N_DOF}"), path, "exec"), g)
            wall = time.perf_counter() - t0
        finally:
            os.chdir(cwd)
        import torch
        from pyto.autodiff.material_model import MaterialModel
        mesh, bc, fe = g["mesh"], g["bc"], g["fe"]
        x = np.asarray(mesh.elemPseudoDensity, float)
        with torch.no_grad():                                 # compliance of the delivered design (no filter)
            compliance = float(np.asarray(bc.force) @ fe.solve(torch.tensor(x), MaterialModel.SIMP).numpy())
        np.savez_compressed(out, x=np.asarray(mesh.elemPseudoDensity, float), node_xyz=np.asarray(mesh.node_xyz),
                            elems=np.asarray(mesh.elemArray)[:, :8].astype(np.int64),
                            fixed=np.unique(np.asarray(bc.fixed_dofs) // 3), force=np.asarray(bc.force, float),
                            n_fea=int(g["n_fea"]), log=log.getvalue(), wall_s=wall, compliance=compliance)
        print(key, log.getvalue().strip().splitlines()[-2:], flush=True)
    return dict(np.load(out))


def render(key, d):
    xyz = d["node_xyz"]
    h = float(np.min([np.diff(np.unique(np.round(xyz[:, k], 9))).min() for k in range(3)]))
    part = recovered_part(d, STL, os.path.join(CACHE, f"quickstart_{key}_{N_DOF}_part.vtp"))
    import pyvista as pv
    ref = bc_faces(pv.read(STL), xyz, d["fixed"], d["force"], 1.0 * h, split=False, skip_axis=2)   # the hole walls
    p = plotter((2400, 1500))
    add_surface(p, part, role=transfer_roles(part, ref, 1e-3 * h))
    add_hole_arrow(p, xyz, d["force"], 0.02, z_front=xyz[:, 2].max() + 0.002)
    camera(p, xyz.min(0), xyz.max(0), (0.55, 0.45, 1.0), (0, 1, 0))
    return shot(p, pad=12)


def main():
    data = [run(k) for k in SCRIPTS]
    imgs = [render(k, d) for k, d in zip(SCRIPTS, data)]
    a, b = (d["x"] > 0.5 for d in data)
    print("caption:", {k: (round(float(d["compliance"]), 4), int(d["n_fea"]), round(float(d["wall_s"]), 1),
                           round(float(d["x"].mean()), 3)) for k, d in zip(SCRIPTS, data)},
          f"elements differing {100 * np.mean(a != b):.1f} %, elements {a.size}, DOF {3 * len(data[0]['node_xyz'])}", flush=True)
    h = max(i.shape[0] for i in imgs)
    imgs = [np.pad(i, ((h - i.shape[0], 0), (0, 0), (0, 0))) for i in imgs]
    fig = panel_grid(imgs, ["(a)", "(b)"], ncols=2, width=TEXT_WIDTH, gap=0.25)
    from matplotlib.lines import Line2D
    from _style import LOAD_ARROW, SUPPORT
    keys = [Line2D([], [], ls="", marker="s", ms=5, color=SUPPORT, label="supports"),
            Line2D([], [], ls="", marker="v", ms=5, color=LOAD_ARROW, label="load")]
    fig.legend(handles=keys, loc="lower center", ncol=2, fontsize=6.8, frameon=False, handletextpad=0.3,
               columnspacing=1.5, bbox_to_anchor=(0.5, -0.02))
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + 0.16)
    for ax in fig.axes:
        x0, y0, ww, hh = ax.get_position().bounds
        ax.set_position([x0, (y0 * h + 0.16) / (h + 0.16), ww, hh * h / (h + 0.16)])
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x, (y * h + 0.16) / (h + 0.16)))
    save(fig, "fig02_quickstart")


if __name__ == "__main__":
    main()
