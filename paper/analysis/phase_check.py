"""C2: where the per-phase time differences between AD and analytical runs come from.

Times every step of an iteration separately (wrapping the functions the drivers call): FE solve, gradient
(torch.autograd.grad or manual_gradients), density-filter products (forward H x and the transposed H^T g of the
analytical chain rule), objective and constraint evaluation, MMA subproblem (mmasub) and KKT check, OC's postprocess,
and Python's garbage collector. The remainder of the iteration is everything else (mostly freeing memory, numpy
conversions, bookkeeping). Medians over iterations 2..N.

Supports the per-phase timing paragraph of the paper ("Where the time goes": time outside the FE solve and the
gradient, the filter chain rule, the MMA subproblem). The paper's numbers were measured on a frozen copy of the code
(the state used for the timing sessions); point PYTHONPATH at such a copy to repeat them, or at src for the current code.

Run from the repository root (about 10 min, idle machine):  PYTHONPATH=src python paper/analysis/phase_check.py
Arguments (optional): cases as METHOD:Problem, e.g. MMA:EdgeCantilever OC:Bridge; PHASE_TAG=<suffix> names the output.
Output: Results/C2_phase_check/phase_check<PHASE_TAG>.json
"""
import collections
import contextlib
import gc
import io
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))       # repository root
HERE = os.path.join(ROOT, "Results", "C2_phase_check")                              # output folder
CASES = [("MMA", "EdgeCantilever"), ("OC", "EdgeCantilever"), ("OC", "Bridge")]
if len(sys.argv) > 1:                          # e.g. MMA:EdgeCantilever OC:Bridge
    CASES = [tuple(a.split(":")) for a in sys.argv[1:]]
N_ITER = 20


def run(method, problem, gradient):
    import pyto.topopt.drivers.mma as mma
    import pyto.topopt.drivers.mmaWrapper as mw
    import pyto.topopt.drivers.oc as oc
    from pyto.examples_benchmarks.topopt_run_benchmarks import StructuralTOExamples, build_benchmark
    from pyto.solve.solvers import Solvers
    from pyto.topopt.common import GRADIENT_METHOD

    with contextlib.redirect_stdout(io.StringIO()):
        fe, tp, _ = build_benchmark(getattr(StructuralTOExamples, problem), solver=Solvers.PARDISO)
    tp.Gradient = getattr(GRADIENT_METHOD, gradient)
    acc = collections.defaultdict(float)
    saved = []

    def timed(key, fn):
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                acc[key] += time.perf_counter() - t
        return w

    def sparse_mm(A, B):                       # the analytical chain rule passes H.t(): an uncoalesced COO matrix
        key = "filter_forward" if A.is_coalesced() else "filter_transpose"
        return timed(key, orig_mm)(A, B)

    gc_t = [0.0]
    def gc_cb(phase, info):
        if phase == "start":
            gc_t[0] = time.perf_counter()
        else:
            acc["gc"] += time.perf_counter() - gc_t[0]

    orig_mm = torch.sparse.mm
    solve_orig = fe.solve

    def release_then_solve(*a, **k):             # the retention fix's clearing, timed on its own before the solve
        t = time.perf_counter()
        fe.sol = fe.deformation = fe.stiff_mtrx = fe.total_force = fe.elem_thermal_strain = None
        acc["release_prev"] += time.perf_counter() - t
        return solve_orig(*a, **k)
    fe.solve = release_then_solve
    patches = [(fe, "solve", "fe_solve"), (torch.autograd, "grad", "gradient"),
               (mma, "manual_gradients", "gradient"), (oc, "manual_gradients", "gradient"),
               (mma, "compute_objective_and_gradient", "objective"), (oc, "compute_objective_and_gradient", "objective"),
               (mma, "compute_constraint_and_gradient", "constraints"), (mw, "mmasub", "mma_subproblem"),
               (mw, "kktcheck", "kkt_check"), (fe, "postprocess", "postprocess")]
    originals = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    originals.append((fe, "solve", solve_orig))
    for obj, name, key in patches:
        setattr(obj, name, timed(key, getattr(obj, name)))
    torch.sparse.mm = sparse_mm
    gc.callbacks.append(gc_cb)
    t_last = [None]

    def callback(history):
        now = time.perf_counter()
        rec = dict(acc, iter_s=now - t_last[0])
        rec["remainder"] = rec["iter_s"] - sum(v for k, v in acc.items())
        saved.append(rec)
        acc.clear()
        t_last[0] = time.perf_counter()

    try:
        t_last[0] = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            if method == "MMA":
                mma.topopt_mma(fe, to_params=tp, maxMMAIterations=N_ITER, binarize_topology=False,
                               print_progress=False, iteration_callback=callback)
            else:
                tp.MaxIterations = N_ITER
                oc.topopt_optimality_criteria(fe, to_params=tp, maxIterations=N_ITER, binarize_topology=False,
                                              print_progress=False, iteration_callback=callback)
    finally:
        for obj, name, fn in originals:
            setattr(obj, name, fn)
        torch.sparse.mm = orig_mm
        gc.callbacks.remove(gc_cb)
    keys = sorted({k for r in saved for k in r})
    return {k: float(np.median([r.get(k, 0.0) for r in saved[1:]])) for k in keys}, len(saved)


def main():
    out = {}
    for method, problem in CASES:
        for gradient in ("AUTODIFF", "ANALYTICAL"):
            med, n = run(method, problem, gradient)
            out[f"{method} {problem} {gradient}"] = dict(med, iterations=n)
            print(f"{method} {problem} {gradient} ({n} it): " + ", ".join(f"{k} {v:.4f}" for k, v in med.items()),
                  flush=True)
    tag = os.environ.get("PHASE_TAG", "")
    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, f"phase_check{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
