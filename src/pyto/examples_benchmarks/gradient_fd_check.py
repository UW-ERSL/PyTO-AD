"""Gradients from automatic differentiation against central finite differences, as a function of the step size.

For quantities that have no hand-written gradient (and compliance, as a reference), the directional derivative
dJ/dx . d from autograd is compared with the central difference (J(x + h d) - J(x - h d)) / (2h) for steps h from
1e-10 to 1e-1. The relative error falls as h^2 (truncation) until round-off takes over (about eps |J| / h): a
V-shaped curve whose bottom reaches ~1e-8 or below. A gradient that was only approximately right would instead
level off at its own error for small h. The quantities are evaluated with the same expression compiler the
optimizer uses, on small meshes, at a fixed non-uniform density field and a fixed random direction.

  compliance             Compliance()                                    L-bracket (3D thick, small mesh)
  displacement p-norm    Displacement(FootEnd, magnitude, pnorm)         L-bracket: nodes on the foot's end face,
                                                                         away from the loaded nodes
  stress p-norm          StressPNorm()                                   L-bracket (smooth stand-in for max stress)
  inverter output        Displacement(Output, x, mean)                   compliant inverter (small mesh)

Run from the repository root:
    PYTHONPATH=src MPLBACKEND=Agg python -m pyto.examples_benchmarks.gradient_fd_check
Output: Results/GradientFD/fd_vs_step.csv (one row per quantity and step); the figure fig_gradient_vs_fd_step is
drawn by make_paper_figures.
"""
import contextlib
import csv
import io
import os

import numpy as np
import torch

OUT = os.path.join("Results", "GradientFD")
STEPS = np.logspace(-10, -1, 19)
NDOF = 6000


def _problem(name):
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.solve.solvers import Solvers
    with contextlib.redirect_stdout(io.StringIO()):
        mesh, mat, bc, ebf, _ = getStructuralTOProblem(getattr(StructuralTOExamples, name), nDOFDesired=NDOF)
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=Solvers.PARDISO, rtol=1e-12,
                              elem_body_force=ebf)
    return fe


def _selections(fe, which):
    xyz, h = fe.mesh.node_xyz, fe.mesh.elem_size[0] / 2
    labels = np.asarray(fe.mesh.node_indices[:, 3])
    if which == "LBracket":                       # foot end face (x = xMax) without the loaded nodes
        nodes = np.where((np.abs(xyz[:, 0] - xyz[:, 0].max()) < h) & (labels != 2))[0]
        return {"FootEnd": {"nodes": nodes}}
    y_mid = (xyz[:, 1].max() + xyz[:, 1].min()) / 2   # inverter output nodes, as the benchmark defines them
    nodes = np.where((np.abs(xyz[:, 0] - xyz[:, 0].max()) < h) & (np.abs(xyz[:, 1] - y_mid) < h))[0]
    return {"Output": {"nodes": nodes}}


CASES = [("compliance", "LBracketThickTopLoad", "LBracket", "Compliance()"),
         ("displacement p-norm", "LBracketThickTopLoad", "LBracket", "Displacement(FootEnd, magnitude, pnorm)"),
         ("stress p-norm", "LBracketThickTopLoad", "LBracket", "StressPNorm()"),
         ("inverter output", "Inverter", "Inverter", "Displacement(Output, x, mean)")]


def main():
    from pyto.autodiff.material_model import MaterialModel
    from pyto.topopt.expressions import compile_expression
    os.makedirs(OUT, exist_ok=True)
    rows, cache = [], {}
    for label, problem, which, expression in CASES:
        if problem not in cache:
            cache[problem] = _problem(problem)
        fe = cache[problem]
        fn = compile_expression(expression, _selections(fe, which)).fn
        rng = np.random.default_rng(7)
        x0 = torch.tensor(rng.uniform(0.2, 0.9, fe.mesh.num_elems), dtype=torch.float64)
        d = torch.tensor(rng.uniform(-1.0, 1.0, fe.mesh.num_elems), dtype=torch.float64)

        def J(x):
            return fn(fe.solve(x, MaterialModel.SIMP), x, fe)

        x = x0.clone().requires_grad_(True)
        j0 = J(x)
        (g,) = torch.autograd.grad(j0, x)
        ad = float(g @ d)
        with torch.no_grad():
            for h in STEPS:
                fd = (float(J(x0 + h * d)) - float(J(x0 - h * d))) / (2 * h)
                rows.append({"quantity": label, "expression": expression, "problem": problem,
                             "elements": fe.mesh.num_elems, "J": float(j0), "step": h, "ad": ad, "fd": fd,
                             "rel_error": abs(fd - ad) / max(abs(ad), 1e-300)})
        best = min(r["rel_error"] for r in rows if r["quantity"] == label)
        print(f"{label:22s} {expression:42s} J = {float(j0):.4g}  dJ.d = {ad:.6g}  best rel. error {best:.1e}",
              flush=True)
    with open(os.path.join(OUT, "fd_vs_step.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}/fd_vs_step.csv")


if __name__ == "__main__":
    main()
