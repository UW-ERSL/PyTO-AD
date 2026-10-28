"""User-defined objective/constraint (TO_QOI.GFUNCTION): gradients, error messages, driver support."""
import numpy as np
import pytest
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import compute_constraint_and_gradient, compute_objective_and_gradient
from pyto.topopt.common import TOParams, TO_QOI


def _tip_dof(fe):
    return int(np.argmax(np.abs(fe.bc.force)))


def _check_gradient(fe, scalar_of_x, n_check=4, eps=1e-6):
    rng = np.random.default_rng(0)
    x0 = torch.tensor(0.3 + 0.6 * rng.random(fe.mesh.num_elems), dtype=torch.float64)
    x = x0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(scalar_of_x(x), x)
    for e in rng.choice(fe.mesh.num_elems, size=n_check, replace=False):
        up, dn = x0.clone(), x0.clone()
        up[e] += eps
        dn[e] -= eps
        fd = (scalar_of_x(up) - scalar_of_x(dn)).item() / (2 * eps)
        assert abs(fd - g[e].item()) <= 1e-4 * abs(fd) + 1e-12, (e, fd, g[e].item())


def test_user_objective_gradient_matches_finite_difference(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    tip = _tip_dof(fe)

    def tip_deflection_squared_plus_volume(sol, x, fe_solver):  # a typical user objective, torch only
        return 1e12 * sol[tip] ** 2 + 0.1 * x.mean()

    params = TOParams()
    params.Objective = (TO_QOI.GFUNCTION, tip_deflection_squared_plus_volume)
    _check_gradient(fe, lambda x: compute_objective_and_gradient(
        params, fe.solve(x, MaterialModel.SIMP), x, fe, structural_KE, MaterialModel.SIMP))


def test_user_constraint_value_and_gradient(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    tip = _tip_dof(fe)
    user_fn = lambda sol, x, fe_solver: sol[tip].abs()
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64)
    sol = fe.solve(x, MaterialModel.SIMP)
    limit = 2.0 * sol[tip].abs().item()

    params = TOParams()
    params.Constraints = [(TO_QOI.GFUNCTION, user_fn, limit), (TO_QOI.GFUNCTION, user_fn, None)]
    c, _ = compute_constraint_and_gradient(params, sol, x, fe, structural_KE, MaterialModel.SIMP)
    assert c[0, 0].item() == pytest.approx(-0.5)                 # |u_tip| / (2 |u_tip|) - 1
    assert c[1, 0].item() == pytest.approx(sol[tip].abs().item())  # limit None: the value itself (g <= 0 form)

    params.Constraints = params.Constraints[:1]
    _check_gradient(fe, lambda x: compute_constraint_and_gradient(
        params, fe.solve(x, MaterialModel.SIMP), x, fe, structural_KE, MaterialModel.SIMP)[0][0, 0])


@pytest.mark.parametrize("bad_fn, error, message", [
    (lambda sol, x, fe: torch.tensor(float(sol.detach().numpy()[0])), RuntimeError, "no gradient path"),
    (lambda sol, x, fe: float(sol.detach()[0]), TypeError, "must return a torch"),
    (lambda sol, x, fe: sol[:3], ValueError, "single value"),
    ("not a function", TypeError, "expected a function"),
])
def test_user_objective_mistakes_give_clear_errors(structural_fe_solver, structural_KE, bad_fn, error, message):
    fe = structural_fe_solver
    params = TOParams()
    params.Objective = (TO_QOI.GFUNCTION, bad_fn)
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64, requires_grad=True)
    with pytest.raises(error, match=message):
        compute_objective_and_gradient(params, fe.solve(x, MaterialModel.SIMP), x, fe, structural_KE)


def test_user_constraint_rejects_non_positive_limit(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    params = TOParams()
    params.Constraints = [(TO_QOI.GFUNCTION, lambda sol, x, f: sol.abs().sum(), 0.0)]
    x = torch.full((fe.mesh.num_elems,), 0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="limit must be > 0"):
        compute_constraint_and_gradient(params, fe.solve(x, MaterialModel.SIMP), x, fe, structural_KE)


def test_mma_runs_user_objective(structural_fe_solver):
    from pyto.topopt.drivers.mma import topopt_mma
    fe = structural_fe_solver
    tip = _tip_dof(fe)
    params = TOParams()
    params.Objective = (TO_QOI.GFUNCTION, lambda sol, x, f: sol[tip].abs())
    params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    J = np.ravel(np.array(topopt_mma(fe, to_params=params, maxMMAIterations=10, print_progress=False,
                                     binarize_topology=False)[1]["objective"], dtype=float))
    assert np.all(np.isfinite(J)) and J[-1] < J[0]
