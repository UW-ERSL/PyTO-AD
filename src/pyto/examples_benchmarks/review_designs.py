"""0/1 designs for the review: thermo-elastic BiClamp at dT = 4 and 10 K, and a feasible stress-constrained L-bracket.

Each case is a benchmark problem (built as the sweep builds it) with a few TOParams overrides, optimized with MMA.
The continuous design is kept (no binarization in the driver); this script then thresholds it to 0/1 (volume
preserving, as the drivers do, plus hanging-element removal where the problem asks for it) and evaluates both:
objective, grey fraction, volume fraction and, for stress problems, the peak von Mises stress against the limit.

    PYTHONPATH=src python -m pyto.examples_benchmarks.review_designs [--cases name ...] [--tag TAG]

Writes Results/ReviewDesigns_<tag>/<case>/{result.json,history.csv,topology.png} and summary.csv.
"""
import argparse
import csv
import datetime
import json
import os
import time

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import compute_objective_and_gradient
from pyto.autodiff.qoi.stress import compute_pnorm_stress_autograd
from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples
from pyto.examples_benchmarks.topopt_thermostructural_benchmarks import ThermoStructuralTOExamples
from pyto.topopt.common import TO_QOI
from pyto.topopt.drivers._shared import setup_driver_state
from pyto.topopt.drivers.mma import topopt_mma

PROJ = dict(HeavisideProjection=True, MaxIterations=350)     # beta 1 -> 32, doubling every 50 iterations
CASES = {
    # C4: BiClamp (Rodrigues & Fernandes Example A) as in the sweep, then with projection
    "BiClampDT4": (ThermoStructuralTOExamples.BiClampDT4, {}),
    "BiClampDT4_proj": (ThermoStructuralTOExamples.BiClampDT4, PROJ),
    "BiClampDT10": (ThermoStructuralTOExamples.BiClampDT10, {}),
    "BiClampDT10_proj": (ThermoStructuralTOExamples.BiClampDT10, PROJ),
    # C5: minimum volume under a 400 MPa stress limit; more iterations, then projection
    "LBracketTopLoad_Vol_Stress_400it": (StructuralTOExamples.LBracketTopLoad_Vol_Stress, dict(MaxIterations=400)),
    "LBracketTopLoad_Vol_Stress_proj": (StructuralTOExamples.LBracketTopLoad_Vol_Stress, dict(PROJ, MaxIterations=400)),
    "LBracketMidLoad_Vol_Stress_proj": (StructuralTOExamples.LBracketMidLoad_Vol_Stress, dict(PROJ, MaxIterations=400)),
}


def zero_one(x, fe, remove_hanging):
    """Volume-preserving threshold (as the drivers binarize), then keep the largest connected component."""
    t = np.sort(x)[int((1 - np.mean(x)) * len(x))]
    x01 = np.where(x < t, 0.0, 1.0)
    if remove_hanging:
        fe.mesh.setPseudoDensity(torch.tensor(x01))
        comps = fe.mesh.find_connected_components()
        if len(comps) > 1:
            x01[:] = 0.0
            x01[list(max(comps, key=len))] = 1.0
    return x01


def evaluate(x, fe, to_params, KE):
    """Objective (and peak von Mises stress for stress problems) of a physical density x, without filtering."""
    xt = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        sol = fe.solve(xt, MaterialModel.SIMP)
        out = {"objective": float(compute_objective_and_gradient(to_params, sol, xt, fe, KE, MaterialModel.SIMP)),
               "volfrac": float(np.mean(x))}
        if any(c[0] is TO_QOI.MAX_VONMISES_STRESS for c in to_params.Constraints):   # relaxed stress, as optimized
            _p, vm = compute_pnorm_stress_autograd(sol, xt, fe)
            limit = min(c[2] for c in to_params.Constraints if c[0] is TO_QOI.MAX_VONMISES_STRESS)
            out.update(max_vm=float(vm), stress_limit=limit, stress_violation_pct=100 * (float(vm) / limit - 1))
    return out


def run_case(name, root):
    problem, overrides = CASES[name]
    fe, to_params, _mode = build_benchmark(problem)
    for k, v in overrides.items():
        setattr(to_params, k, v)
    remove_hanging = bool(to_params.Eliminate_Hanging_Elements)
    to_params.Eliminate_Hanging_Elements = False                # keep the continuous design; thresholded below
    out_dir = os.path.join(root, name)
    os.makedirs(out_dir, exist_ok=True)
    iters = []
    t0 = time.perf_counter()
    _u, history, success, msg, n_fea = topopt_mma(
        fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations, binarize_topology=False,
        print_progress=False, iteration_callback=lambda h: iters.append(float(np.ravel(h["objective"][-1])[0])))
    wall = time.perf_counter() - t0
    x = np.asarray(fe.mesh.elemPseudoDensity, float).ravel()
    if not np.isfinite(x).all():                                # diverged: record it and go on with the next case
        finite = [v for v in iters if np.isfinite(v)]
        res = dict(case=name, problem=problem.name, overrides=overrides, status="diverged", iterations=int(n_fea),
                   max_iterations=to_params.MaxIterations, message=msg, wall_s=wall,
                   last_finite_iteration=len(finite), last_finite_objective=finite[-1] if finite else None)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        print(f"{name}: diverged (density not finite after {int(n_fea)} iterations)", flush=True)
        return res
    KE =torch.tensor(setup_driver_state(fe, to_params)["KE"])
    filt = to_params.APPLY_FILTER_TO_DENSITY
    to_params.APPLY_FILTER_TO_DENSITY = False
    try:
        cont, x01 = evaluate(x, fe, to_params, KE), zero_one(x, fe, remove_hanging)
        bin_ = evaluate(x01, fe, to_params, KE)
    finally:
        to_params.APPLY_FILTER_TO_DENSITY = filt
    tail = iters[-11:]
    res = dict(case=name, problem=problem.name, overrides=overrides, status="ok", iterations=int(n_fea),  # one FE solve per MMA it.
               max_iterations=to_params.MaxIterations, success=bool(success), message=msg,
               stopped_before_max=int(n_fea) < to_params.MaxIterations,
               last10_rel_change=float(abs(tail[-1] - tail[0]) / abs(tail[0])) if len(tail) > 1 else None,
               grey_fraction=float(np.mean((x > 0.1) & (x < 0.9))), wall_s=wall,
               continuous=cont, zero_one=bin_, zero_one_vs_continuous=bin_["objective"] / cont["objective"])
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, "history.csv"), "w") as f:
        f.write("k,objective\n" + "".join(f"{k + 1},{v}\n" for k, v in enumerate(iters)))
    np.save(os.path.join(out_dir, "density_continuous.npy"), x)
    np.save(os.path.join(out_dir, "density_01.npy"), x01)
    fe.mesh.setPseudoDensity(torch.tensor(x01))                 # image: the 0/1 design, PyTO's own render
    try:
        fe.plot_mesh(save_path=os.path.join(out_dir, "topology.png"), plot_bc=None, title=None,
                     camera_position="xy" if to_params.ExtrudeZ else "iso")
    except Exception as e:
        print(f"{name}: image not written: {e}", flush=True)
    print(f"{name}: {int(n_fea)} it, J {cont['objective']:.5g} (0/1 {bin_['objective']:.5g}), grey "
          f"{res['grey_fraction']:.3f}" + (f", stress {cont['stress_violation_pct']:+.1f}% (0/1 "
          f"{bin_['stress_violation_pct']:+.1f}%)" if "max_vm" in cont else ""), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES))
    ap.add_argument("--tag", default=datetime.date.today().isoformat())
    args = ap.parse_args()
    root = os.path.join("Results", f"ReviewDesigns_{args.tag}")
    rows = []
    for name in args.cases:
        r = run_case(name, root)
        rows.append({k: v for k, v in r.items() if not isinstance(v, dict)} |
                    {f"cont_{k}": v for k, v in r.get("continuous", {}).items()} |
                    {f"01_{k}": v for k, v in r.get("zero_one", {}).items()})
        with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r_ in rows for k in r_)))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
