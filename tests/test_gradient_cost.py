"""Computational cost of the autograd gradient: forward, forward + backward, and backward = the difference.

What is timed is the chain the MMA driver differentiates every iteration:
    x -> density filter -> FE solve(s) -> objective      (forward, with the autograd graph built)
    torch.autograd.grad(objective, x)                     (backward: adjoint solve(s) + chain rule)

Method (per case):
  * `warmup` forward+backward runs are discarded (first calls pay imports, solver set-up and allocator growth).
  * `repeat` iterations then alternate one forward-only run and one forward+backward run, so slow drift
    (thermal throttling, other load) hits both measurements equally.
  * backward = mean(forward+backward) - mean(forward); its spread combines both spreads (sqrt(s_f^2 + s_fb^2)).
    The directly timed backward (the grad call alone inside each forward+backward run) is reported as a check.

Under pytest this runs small problems as a smoke test. For real numbers run it as a script, e.g.
    PYTHONPATH=src python tests/test_gradient_cost.py --ndof 25000 --repeat 10 --solver PARDISO
"""
import argparse
import gc
import time

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel


def _structural(problem, ndof, solver):
    from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    mesh, mat, bc, ebf, p = getStructuralTOProblem(getattr(StructuralTOExamples, problem), nDOFDesired=ndof)
    return HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver, elem_body_force=ebf), p


def _thermal(problem, ndof, solver):
    from pyto.examples_benchmarks.topopt_thermal_benchmarks import ThermalTOExamples, getThermalTOProblem
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    mesh, mat, bc, ebf, p = getThermalTOProblem(getattr(ThermalTOExamples, problem), nDOFDesired=ndof)
    return HexThermalFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver, elem_body_force=ebf), p


def _thermostructural(problem, ndof, solver):
    from pyto.examples_benchmarks.topopt_thermostructural_benchmarks import (ThermoStructuralTOExamples,
                                                                               getThermoStructuralTOProblem)
    from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA
    mesh, mat, sbc, tbc, ebf, p = getThermoStructuralTOProblem(getattr(ThermoStructuralTOExamples, problem),
                                                               nDOFDesired=ndof)
    fe = ThermoStructuralFEA(mesh=mesh, mat_prop=mat, structural_bc=sbc, thermal_bc=tbc, solver=solver,
                             elem_body_force=ebf, thermoElasticReferenceTemperature=p.ThermalReferenceTemperature,
                             conductivity_penalty=p.ConductivityPenalty,
                             conductivity_void_ratio=p.ConductivityVoidRatio)
    return fe, p


# name -> (builder, benchmark problem). The objective is whatever that benchmark optimizes.
CASES = {
    "structural compliance": (_structural, "MBBBeam"),
    "structural p-norm stress": (_structural, "LBracketTopLoad_Stress_Vol"),
    "thermal compliance": (_thermal, "HeatPlate"),
    "thermo-structural compliance": (_thermostructural, "BiClamp"),
}


def build_chain(case, ndof, solver):
    """Return (objective(x_raw), x0, ndof actually built) for one case, wired exactly like the MMA driver."""
    from pyto.autodiff.qoi import compute_objective_and_gradient
    from pyto.topopt.common import createFilters
    from pyto.topopt.drivers._shared import compute_element_stiffness, torch_sparse_filter

    builder, problem = CASES[case]
    fe, p = builder(problem, ndof, solver)
    H, Hs = torch_sparse_filter(*createFilters(fe, p))
    KE = torch.tensor(compute_element_stiffness(fe))

    def objective(x_raw):
        x = torch.sparse.mm(H, x_raw.unsqueeze(1)).squeeze(1) / Hs
        sol = fe.solve(x, MaterialModel.SIMP)
        return compute_objective_and_gradient(p, sol, x, fe, KE, MaterialModel.SIMP)

    vf = next((c[2] for c in p.Constraints if c[0].name == "VOLUME_FRACTION"), 0.5)
    return objective, torch.full((fe.mesh.num_elems,), float(vf), dtype=torch.float64), fe.bc.num_dofs


def measure_gradient_cost(objective, x0, warmup=2, repeat=10):
    """Mean/std wall-clock seconds of forward, forward+backward, derived and direct backward."""
    def run(backward):
        x = x0.clone().requires_grad_(True)
        t0 = time.perf_counter()
        y = objective(x)
        t1 = time.perf_counter()
        if backward:
            torch.autograd.grad(y, x)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t0, t2 - t1

    for _ in range(warmup):
        run(True)
    fwd, fwd_bwd, bwd_direct = [], [], []
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()  # a collection landing inside one sample would add noise to that sample only
    try:
        for _ in range(repeat):
            fwd.append(run(False)[0])
            _, fb, b = run(True)
            fwd_bwd.append(fb)
            bwd_direct.append(b)
    finally:
        if gc_was_enabled:
            gc.enable()
    fwd, fwd_bwd, bwd_direct = map(np.array, (fwd, fwd_bwd, bwd_direct))
    std = lambda a: a.std(ddof=1) if a.size > 1 else 0.0
    return {
        "forward": (fwd.mean(), std(fwd)),
        "forward+backward": (fwd_bwd.mean(), std(fwd_bwd)),
        "backward": (fwd_bwd.mean() - fwd.mean(), float(np.hypot(std(fwd), std(fwd_bwd)))),
        "backward (direct)": (bwd_direct.mean(), std(bwd_direct)),
        "backward/forward": (fwd_bwd.mean() - fwd.mean()) / fwd.mean(),
        "repeat": repeat,
    }


def format_table(rows):
    head = f"{'case':30s} {'ndof':>8s} {'forward [s]':>17s} {'fwd+bwd [s]':>17s} {'backward [s]':>17s} " \
           f"{'bwd direct [s]':>17s} {'bwd/fwd':>8s}"
    lines = [head, "-" * len(head)]
    for case, ndof, r in rows:
        cell = lambda k: f"{r[k][0]:8.4f} ± {r[k][1]:6.4f}"
        lines.append(f"{case:30s} {ndof:8d} {cell('forward'):>17s} {cell('forward+backward'):>17s} "
                     f"{cell('backward'):>17s} {cell('backward (direct)'):>17s} {r['backward/forward']:8.2f}")
    return "\n".join(lines)


def run_cases(cases, ndof, solver, warmup, repeat):
    rows = []
    for case in cases:
        objective, x0, built_ndof = build_chain(case, ndof, solver)
        rows.append((case, built_ndof, measure_gradient_cost(objective, x0, warmup, repeat)))
    return rows


def test_gradient_cost_small_problems():
    # Smoke test only: timings are machine-dependent, so the assertions check consistency, not speed.
    import pyto.solve.numpy_backend as lin
    rows = run_cases(list(CASES), ndof=2000,  # LBracket needs >= 2000 to find its load nodes
                      solver=lin.Solvers.SPSOLVE, warmup=1, repeat=3)
    print("\n" + format_table(rows))
    for case, _, r in rows:
        assert r["forward"][0] > 0 and r["forward+backward"][0] > r["forward"][0] * 0.5, case
        # derived and directly timed backward describe the same work; allow for timing noise
        assert abs(r["backward"][0] - r["backward (direct)"][0]) <= 0.5 * r["forward+backward"][0] + 0.05, case


if __name__ == "__main__":
    import pyto.solve.numpy_backend as lin
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ndof", type=int, default=25000, help="desired number of DOFs per problem")
    ap.add_argument("--repeat", type=int, default=10, help="timed repetitions (each: 1 forward + 1 fwd+bwd)")
    ap.add_argument("--warmup", type=int, default=2, help="discarded forward+backward runs before timing")
    ap.add_argument("--solver", default="PARDISO", choices=[s.name for s in lin.Solvers])
    ap.add_argument("--cases", nargs="*", default=list(CASES), choices=list(CASES), metavar="CASE")
    a = ap.parse_args()
    print(f"solver {a.solver}, torch threads {torch.get_num_threads()}, warmup {a.warmup}, repeat {a.repeat}")
    print(format_table(run_cases(a.cases, a.ndof, getattr(lin.Solvers, a.solver), a.warmup, a.repeat)))
