"""Manual gradients must equal the autograd gradients of the same objective/constraints."""
import contextlib
import io
import os
import sys

import numpy as np
import pytest
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import compute_constraint_and_gradient, compute_objective_and_gradient
from pyto.topopt.common import TOParams, TO_QOI
from pyto.topopt.manual_sensitivities import manual_gradients, manual_support

MM = MaterialModel.SIMP


def _autograd(to_params, fe, x0):
    KE = torch.as_tensor(fe.elem_stiff[0])
    x = x0.clone().requires_grad_(True)
    u = fe.solve(x, MM)
    obj = compute_objective_and_gradient(to_params, u, x, fe, KE, MM)
    cons, _ = compute_constraint_and_gradient(to_params, u, x, fe, KE, MM)
    grads = [torch.autograd.grad(v, x, retain_graph=True)[0].numpy() for v in [obj] + list(cons.reshape(-1))]
    return u, grads[0], np.array(grads[1:]).reshape(len(grads) - 1, -1)


def _compare(to_params, fe, seed=0, tol=1e-8):
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(seed).random(fe.mesh.num_elems), dtype=torch.float64)
    u, g_obj, g_cons = _autograd(to_params, fe, x0)
    fe.solve(x0, MM)                                               # the manual path uses the last solve
    m_obj, m_cons = manual_gradients(to_params, fe.sol, x0, fe, MM)
    rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))
    assert rel(m_obj, g_obj) < tol, rel(m_obj, g_obj)
    for i in range(len(g_cons)):
        assert rel(m_cons[i], g_cons[i]) < tol, (i, rel(m_cons[i], g_cons[i]))


def _params(objective, *constraints):
    p = TOParams()
    p.Objective = objective
    p.Constraints = list(constraints)
    return p


def test_structural(structural_fe_solver):
    fe = structural_fe_solver
    p = _params((TO_QOI.COMPLIANCE, None), (TO_QOI.VOLUME_FRACTION, None, 0.5), (TO_QOI.VOLUME_FRACTION_MIN, None, 0.1),
                (TO_QOI.COMPLIANCE, None, 1.0), (TO_QOI.PNORM_STRESS, None, 1e8))
    assert manual_support(p, fe) == (True, "")
    _compare(p, fe)
    _compare(_params((TO_QOI.VOLUME_FRACTION, None), (TO_QOI.COMPLIANCE, None, 1.0)), fe)


def test_pnorm_stress(structural_fe_solver):
    """p-norm stress objective and constraint: value and adjoint gradient equal the autograd ones."""
    _compare(_params((TO_QOI.PNORM_STRESS, None), (TO_QOI.VOLUME_FRACTION, None, 0.5),
                     (TO_QOI.PNORM_STRESS, None, 1e8)), structural_fe_solver, tol=1e-7)


def test_thermal(thermal_fe_solver):
    _compare(_params((TO_QOI.COMPLIANCE, None), (TO_QOI.VOLUME_FRACTION, None, 0.5)), thermal_fe_solver)


def test_thermal_prescribed_temperature_only():
    """FourCornersThermal: no heat source, only prescribed temperatures (compliance_sign = -1 problems)."""
    import pyto.solve.numpy_backend as lin
    from pyto.examples_benchmarks.topopt_thermal_benchmarks import ThermalTOExamples, getThermalTOProblem
    from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
    with contextlib.redirect_stdout(io.StringIO()):
        mesh, mat, bc, ebf, _ = getThermalTOProblem(ThermalTOExamples.FourCornersThermal, nDOFDesired=2000)
    fe = HexThermalFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=lin.Solvers.SPSOLVE)
    p = _params((TO_QOI.COMPLIANCE, None), (TO_QOI.VOLUME_FRACTION, None, 0.5))
    ok, _ = manual_support(p, fe)
    if ok:
        _compare(p, fe)
    else:
        pytest.skip("manual gradient declared unsupported for prescribed-temperature problems")


def test_thermo_structural():
    sys.path.insert(0, os.path.dirname(__file__))
    from test_thermoelastic import _coupled_problem
    fe = _coupled_problem()
    _compare(_params((TO_QOI.COMPLIANCE, None), (TO_QOI.VOLUME_FRACTION, None, 0.5)), fe, tol=1e-6)


def test_unsupported_formulations_say_why(structural_fe_solver, thermal_fe_solver):
    fe = structural_fe_solver
    ok, reason = manual_support(_params((TO_QOI.GFUNCTION, lambda s, x, f: s.sum())), fe)
    assert not ok and "automatic differentiation" in reason
    for q in (TO_QOI.MASS, TO_QOI.MAX_VONMISES_STRESS, TO_QOI.STRESS_FAILURE_FACTOR):
        ok, reason = manual_support(_params((TO_QOI.COMPLIANCE, None), (q, None, 1.0)), fe)
        assert not ok and q.name in reason
    ok, reason = manual_support(_params((TO_QOI.PNORM_STRESS, None)), thermal_fe_solver)
    assert not ok and "structural" in reason
    fe.elem_body_force = np.ones(3 * fe.mesh.num_elems)
    try:
        ok, reason = manual_support(_params((TO_QOI.COMPLIANCE, None)), fe)
        assert not ok and "body force" in reason
    finally:
        fe.elem_body_force = None

def _history(driver, fe_solver, to_params, gradient, **kw):
    import copy
    p = copy.deepcopy(to_params)
    p.Gradient = gradient
    with contextlib.redirect_stdout(io.StringIO()):
        h = driver(fe_solver=fe_solver, to_params=p, print_progress=False, plot_progress=False, **kw)[1]
    return np.array([float(np.ravel(v)[0]) for v in h["objective"]])


@pytest.mark.parametrize("heaviside", [False, True])
def test_mma_manual_matches_autodiff(structural_problem, heaviside):
    """Whole MMA path, including the hand-written filter/projection chain rule: same iterates either way."""
    from pyto.topopt.drivers.mma import topopt_mma
    to_params = structural_problem[4]
    to_params.HeavisideProjection, to_params.HeavisideBetaInterval = heaviside, 2   # beta changes inside the run
    runs = [_history(topopt_mma, _new_structural(structural_problem), to_params, g, maxMMAIterations=5)
            for g in ("autodiff", "manual")]
    to_params.HeavisideProjection, to_params.HeavisideBetaInterval = False, 50
    np.testing.assert_allclose(runs[1], runs[0], rtol=1e-8)


def test_mma_manual_pnorm_matches_autodiff(structural_problem):
    from pyto.topopt.drivers.mma import topopt_mma
    import copy
    to_params = copy.deepcopy(structural_problem[4])
    to_params.Objective = (TO_QOI.PNORM_STRESS, None)
    runs = [_history(topopt_mma, _new_structural(structural_problem), to_params, g, maxMMAIterations=4)
            for g in ("autodiff", "manual")]
    np.testing.assert_allclose(runs[1], runs[0], rtol=1e-6)


def _new_structural(problem):
    import pyto.autodiff.sparse_solve as torch_spsolve
    import pyto.physics.structural.hex_structural_fea as hsf
    import pyto.solve.deflation as deflation
    mesh, mat, bc, ebf, _ = problem
    return hsf.HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=torch_spsolve.Solvers.SPSOLVE,
                                dsolver=deflation.DeflationSolver(use_gpu=False), elem_body_force=ebf)


def test_gradient_method_enum_and_old_names():
    from pyto.topopt.common import GRADIENT_METHOD, TOParams, gradient_method
    assert TOParams().Gradient is GRADIENT_METHOD.AUTODIFF
    for value, expected in [(GRADIENT_METHOD.ANALYTICAL, GRADIENT_METHOD.ANALYTICAL), ("analytical", GRADIENT_METHOD.ANALYTICAL),
                            ("manual", GRADIENT_METHOD.ANALYTICAL), ("AUTODIFF", GRADIENT_METHOD.AUTODIFF),
                            ("autodiff", GRADIENT_METHOD.AUTODIFF), (None, GRADIENT_METHOD.AUTODIFF)]:
        assert gradient_method(value) is expected
    with pytest.raises(ValueError):
        gradient_method("finite-difference")


def test_gradient_comparison_benchmark_runs(tmp_path, monkeypatch):
    """AD and analytical runs (MMA, each its own process), measured per iteration: both follow the same
    optimization path, the FE solve is timed in both, only AD saves an autograd graph, and the report is written."""
    from pyto.examples_benchmarks.gradient_comparison import main
    monkeypatch.chdir(tmp_path)                                   # writes ./Results
    with contextlib.redirect_stdout(io.StringIO()):
        pairs = main(["--problems", "HeatPlate", "--max-iters", "3", "--tag", "t", "--max-busy", "5",
                      "--wait-quiet", "0"])        # no waiting: the test checks the run, not its timings
    assert [p["method"] for p in pairs] == ["MMA"]
    for p in pairs:
        assert p["same_path"] and p["fe_solves_ad"] == p["fe_solves_an"]
        assert p["objective_ad"] == pytest.approx(p["objective_an"], rel=1e-9)   # same gradient, same iterates
        assert p["s_per_iter_ad"] > 0 and p["s_per_iter_an"] > 0 and p["fe_solve_an"] > 0
        assert p["ad_graph_gb"] > 0
    root = tmp_path / "Results" / "GradientComparison_t"
    for g in ("AUTODIFF", "ANALYTICAL"):
        assert (root / "runs" / "MMA" / "HeatPlate" / f"{g}_r1" / "iterations.csv").exists()
    text = (root / "GRADIENT_COMPARISON_t.md").read_text()
    assert "## MMA" in text and "## OC" not in text
