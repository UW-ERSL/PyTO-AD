"""Fig. 6 (fig:teaching): the MBB beam in a 3 x 3 grid, filter radius (rows) x SIMP exponent (columns).

Each tile is one MMA run (AD gradients, volume fraction <= 0.5, the benchmark's MBB beam, 2.5D) with only those two
settings changed, delivered as the CONTINUOUS final density, drawn on one scale (white 0, PyTO green 1), so grey
regions show. Each tile is labelled with
  grey   fraction of elements with 0.05 < density < 0.95
  c01    compliance of the volume-preserving 0/1 threshold of the design, evaluated with p = 3; the runs' own
         compliances use different stiffness laws (p = 1, 3, 5) and are not comparable across columns.

Runs are cached in fig_gen/_cache/teaching_*.npz (delete to rerun). Run from paper/fig_gen:
    python fig06_teaching_grid.py          ->  ../figures/generated/fig06_teaching_grid.pdf
"""
import contextlib
import io
import os
import sys

import numpy as np

from _style import CACHE, DENSITY_CMAP_COLORS, INK, REPO, TEXT_WIDTH, save

sys.path.insert(0, os.path.join(REPO, "src"))
RADII = (1.1, 1.5, 3.0)              # filter radius, in elements
PENALTIES = (1.0, 3.0, 5.0)          # SIMP exponent p
NDOF = 20000
VOLUME = 0.5


def run(radius, p):
    out = os.path.join(CACHE, f"teaching_r{radius}_p{p}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    import torch
    import pyto.autodiff.material_model as mm
    from pyto.autodiff.sparse_solve import Solvers
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.drivers.mma import topopt_mma
    from pyto.topopt.spec import (ConstraintSpec, ManufacturingSpec, ObjectiveSpec, OptimizationSpec,
                                  RegularizationSpec, compile_spec)
    with contextlib.redirect_stdout(io.StringIO()):
        mesh, mat, bc, ebf, to_params = getStructuralTOProblem(StructuralTOExamples.MBBBeam, nDOFDesired=NDOF)
    fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=Solvers.PARDISO, elem_body_force=ebf)
    spec = OptimizationSpec(objective=ObjectiveSpec("Compliance()"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", VOLUME)],
                            regularization=RegularizationSpec(filter_radius=radius),
                            manufacturing=ManufacturingSpec(extrude="Z"))
    to_params = compile_spec(spec, base=to_params)
    mm.initialize_SIMP_STRUCTURAL_PENALTY(p)
    with contextlib.redirect_stdout(io.StringIO()):
        _u, history, ok, msg, n_fe = topopt_mma(fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
                                                binarize_topology=False, print_progress=False)
    x = np.asarray(mesh.elemPseudoDensity, float)
    # 0/1 design at the volume-preserving threshold, compliance with p = 3 (comparable across tiles)
    thr = np.sort(x)[int((1 - x.mean()) * len(x))]
    x01 = (x >= thr).astype(float)
    mm.initialize_SIMP_STRUCTURAL_PENALTY(3.0)
    with torch.no_grad():
        u01 = fe.solve(torch.tensor(x01), mm.MaterialModel.SIMP).numpy()
    res = dict(x=x, centers=np.asarray(mesh.elem_centers, float), elem_size=np.asarray(mesh.elem_size, float),
               c01=float(np.asarray(bc.force) @ u01), grey=float(((x > 0.05) & (x < 0.95)).mean()),
               fe_solves=int(n_fe), radius=radius, p=p, volume=float(x.mean()))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, **res)
    print(f"r = {radius}, p = {p}: c01 {res['c01']:.4g} J, grey {res['grey']:.3f}, {n_fe} FE solves, "
          f"volume {res['volume']:.3f}", flush=True)
    return res


def image(res):
    """Element densities of the extruded design as a 2D array (rows = y, top first)."""
    c, h = res["centers"], res["elem_size"]
    z0 = np.unique(np.round(c[:, 2] / h[2]))[0]
    sel = np.round(c[:, 2] / h[2]) == z0
    i = np.round((c[sel, 0] - c[sel, 0].min()) / h[0]).astype(int)
    j = np.round((c[sel, 1] - c[sel, 1].min()) / h[1]).astype(int)
    img = np.full((j.max() + 1, i.max() + 1), np.nan)
    img[j, i] = res["x"][sel]
    return img[::-1]


def main():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("pyto_density", DENSITY_CMAP_COLORS)
    results = {(r, p): run(r, p) for r in RADII for p in PENALTIES}
    img0 = image(results[(RADII[0], PENALTIES[0])])
    aspect = img0.shape[0] / img0.shape[1]
    left, right, top, bottom, gap = 0.5, 0.06, 0.4, 0.42, 0.06
    tile_w = (TEXT_WIDTH - left - right - 2 * gap) / 3
    lab_h = 0.2
    tile_h = tile_w * aspect
    H = top + bottom + 3 * (tile_h + lab_h) + 2 * gap
    fig = plt.figure(figsize=(TEXT_WIDTH, H))
    for a, r in enumerate(RADII):
        for b, p in enumerate(PENALTIES):
            res = results[(r, p)]
            x0 = left + b * (tile_w + gap)
            y0 = H - top - (a + 1) * (tile_h + lab_h) - a * gap + lab_h
            ax = fig.add_axes([x0 / TEXT_WIDTH, y0 / H, tile_w / TEXT_WIDTH, tile_h / H])
            ax.imshow(image(res), cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
                s.set_color("0.55")
            ax.text(0.5, -0.04, f"$c_{{0/1}}$ = {res['c01']:.3g} J,  grey {100 * res['grey']:.0f} %",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)
            if a == 0:
                ax.set_title(f"$p$ = {p:g}", fontsize=8.5, pad=3)
            if b == 0:
                ax.text(-0.06, 0.5, f"$r$ = {r:g}", transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=8.5)
    fig.text(0.1 / TEXT_WIDTH, 1 - (top + 1.5 * (tile_h + lab_h) + gap) / H, "filter radius (elements)",
             rotation=90, ha="center", va="center", fontsize=8)
    fig.text((left + 1.5 * tile_w + gap) / TEXT_WIDTH, 1 - 0.04 / H, "SIMP exponent", ha="center", va="top",
             fontsize=8)
    # grey scale bar
    cax = fig.add_axes([(left + tile_w + gap) / TEXT_WIDTH, 0.17 / H, tile_w / TEXT_WIDTH, 0.06 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax,
                      orientation="horizontal")
    cb.set_ticks([0, 0.5, 1])
    cb.ax.tick_params(labelsize=7.5, length=2, pad=1)
    cb.outline.set_linewidth(0.5)
    cax.text(-0.04, 0.5, "density", transform=cax.transAxes, ha="right", va="center", fontsize=7.5)
    save(fig, "fig06_teaching_grid")


if __name__ == "__main__":
    main()
