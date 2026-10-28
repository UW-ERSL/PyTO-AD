"""Binarization tie check for ThreeHoleBracketThick (MMA, AD): why AD repeat 2 of GradientComparison_2026-10-18b ended
at 0.299473 while the other five runs ended at 0.2995160, although all six followed the same continuous path
(objectives equal to 1e-13 at every one of the 63 iterations).

The optimizer is rerun exactly as gradient_comparison runs it (build_benchmark, topopt_mma, binarize). The continuous
physical design x is recorded just before thresholding, and the threshold rule of topopt_mma is examined:
    k = int((1 - mean(x)) * n),  threshold = sort(x)[k],  design = (x >= threshold)
  * how close (1 - mean(x)) * n is to an integer: if it is within round-off, k (and the threshold) can differ by one
    element between runs whose x differ only by round-off;
  * the density gaps around the threshold;
  * how many elements flip when x is perturbed at round-off level (relative 1e-14 .. 1e-12), 200 trials each;
  * the compliance of the designs thresholded at k - 1, k, k + 1 (does one element change it by ~1.5e-4?).

Supports the threshold-tie footnote of the paper (ThreeHoleBracketThick, 0/1 volume fractions 0.250000 and 0.250043).

Run from the repository root:  PYTHONPATH=src MPLBACKEND=Agg python docs/paper/analysis/tie_check.py   (a few minutes)
Output: Results/BinarizationTie/tie_check.json (and x_continuous.npy).
"""
import contextlib
import io
import json
import os

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))       # repository root
HERE = os.path.join(ROOT, "Results", "BinarizationTie")                             # output folder


def main():
    from pyto.autodiff.material_model import MaterialModel
    from pyto.examples_benchmarks.topopt_run_benchmarks import StructuralTOExamples, build_benchmark
    from pyto.topopt.common import gradient_method
    from pyto.topopt.expressions import compile_expression
    import pyto.topopt.drivers.mma as mma

    with contextlib.redirect_stdout(io.StringIO()):
        fe, to_params, _ = build_benchmark(StructuralTOExamples.ThreeHoleBracketThick)
    to_params.Gradient = gradient_method("AUTODIFF")
    mesh = fe.mesh

    # record every density set on the mesh after the MMA loop: first the continuous x, then the 0/1 design
    after_loop, recorded = {"on": False}, []
    set_orig = mesh.setPseudoDensity

    def set_w(rho):
        if after_loop["on"]:
            recorded.append(np.asarray(rho.detach().cpu().numpy() if torch.is_tensor(rho) else rho, float).copy())
        return set_orig(rho)

    mesh.setPseudoDensity = set_w
    run_orig = mma.runMMA

    def run_w(*a, **k):
        out = run_orig(*a, **k)
        after_loop["on"] = True
        return out

    mma.runMMA = run_w
    _u, history, _ok, msg, n_fe = mma.topopt_mma(fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
                                                 binarize_topology=not to_params.HeavisideProjection,
                                                 print_progress=False)
    x, design = recorded[0], recorded[-1]
    n = len(x)
    os.makedirs(HERE, exist_ok=True)
    np.save(os.path.join(HERE, "x_continuous.npy"), x)

    t = (1 - x.mean()) * n
    k = int(t)
    xs = np.sort(x)
    thr = xs[k]
    result = {"fe_solves": int(n_fe), "objective_final": float(np.asarray(history["objective"][-1]).reshape(-1)[0]),
              "elements": n, "mean_density": float(x.mean()), "t=(1-mean)*n": float(t), "k=int(t)": k,
              "fractional_part_of_t": float(t - k), "distance_of_t_to_nearest_integer": float(abs(t - round(t))),
              "threshold": float(thr), "density_below": float(xs[k - 1]), "density_above": float(xs[k + 1]),
              "gap_below": float(thr - xs[k - 1]), "gap_above": float(xs[k + 1] - thr),
              "elements_within_1e-12_of_threshold": int((np.abs(x - thr) < 1e-12).sum()),
              "elements_within_1e-10_of_threshold": int((np.abs(x - thr) < 1e-10).sum()),
              "elements_within_1e-8_of_threshold": int((np.abs(x - thr) < 1e-8).sum()),
              "solid_elements_in_final_design": int(design.sum())}

    def binarize(v):
        s = np.sort(v)
        return (v >= s[int((1 - v.mean()) * len(v))]).astype(float)

    base = binarize(x)
    result["recorded_design_equals_rule"] = bool(np.array_equal(base, design))
    rng = np.random.default_rng(0)
    for rel in (1e-14, 1e-13, 1e-12):
        flips = [int((binarize(x * (1 + rel * rng.standard_normal(n))) != base).sum()) for _ in range(200)]
        result[f"flips_with_relative_noise_{rel:g}"] = {"trials_with_any_flip": int(np.count_nonzero(flips)),
                                                        "max_flipped": int(max(flips)), "of_trials": 200}

    comp = compile_expression("Compliance()").fn
    with torch.no_grad():
        def compliance(d):
            dt = torch.tensor(d, dtype=torch.float64)
            return float(comp(fe.solve(dt, MaterialModel.SIMP), dt, fe))
        c = {j: compliance((x >= xs[k + j]).astype(float)) for j in (-1, 0, 1)}
    result["compliance_threshold_k-1"] = c[-1]
    result["compliance_threshold_k"] = c[0]
    result["compliance_threshold_k+1"] = c[1]
    result["rel_change_one_element_down"] = abs(c[-1] - c[0]) / c[0]
    result["rel_change_one_element_up"] = abs(c[1] - c[0]) / c[0]
    result["comparison_runs"] = {"AD r1, r3, AN r1-3": 0.2995159985452456, "AD r2": 0.29947250331317854,
                                 "rel_difference": abs(0.29947250331317854 - 0.2995159985452456) / 0.2995159985452456}
    with open(os.path.join(HERE, "tie_check.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
