"""Finite-difference checks for the torch-native QOI functions used by the
autodiff optimization pipeline (topopt_mma.py), on small real structural
and thermal meshes (not just the isolated linear-solve primitive).

Every QOI registered in pyto.autodiff.qoi is covered here (compliance,
volume fraction, p-norm stress, mass, gvector, max von Mises), for both
physics where applicable -- the Phase 4 gate. Two of these (compliance,
max von Mises) are additionally cross-checked against
pyto.autodiff.reference_adjoint, the independently-derived legacy NumPy
adjoint oracle, not just against finite differences of the torch code
itself -- so a bug shared between the torch formula and its own FD check
can't hide.
"""
import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import (
    compute_compliance_torch,
    compute_volume_constraint_torch,
    compute_pnorm_stress_autograd,
    compute_mass_torch,
    compute_gvector_torch,
    compute_constraint_and_gradient,
)
from pyto.autodiff.reference_adjoint import (
    compute_compliance,
    compute_pnorm_stress_and_sensitivity,
)
from pyto.topopt.common import TOParams, TO_QOI

EPS = 1e-6
REL_TOL = 1e-4


def _central_diff_grad(f_of_x, x0, eps=EPS):
    grad = torch.zeros_like(x0)
    for e in range(x0.numel()):
        xp = x0.clone(); xp[e] += eps
        xm = x0.clone(); xm[e] -= eps
        grad[e] = (f_of_x(xp) - f_of_x(xm)) / (2 * eps)
    return grad


def _sample_elements(n, k=6):
    step = max(1, n // k)
    return list(range(0, n, step))[:k]


def test_compliance_gradient_matches_finite_difference(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64)

    def compliance(x):
        x = x.clone().requires_grad_(True)
        sol = fe.solve(x, MaterialModel.SIMP)
        return compute_compliance_torch(sol, x, fe, structural_KE, MaterialModel.SIMP)

    x0.requires_grad_(True)
    c = compliance(x0)
    (analytic,) = torch.autograd.grad(c, x0)

    for e in _sample_elements(n):
        def f(x):
            with torch.no_grad():
                sol = fe.solve(x, MaterialModel.SIMP)
            return compute_compliance_torch(sol, x, fe, structural_KE, MaterialModel.SIMP).item()
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_volume_fraction_gradient_matches_finite_difference():
    n = 20
    x0 = torch.rand(n, dtype=torch.float64, requires_grad=True)
    volfrac_upper = 0.4

    c = compute_volume_constraint_torch(x0, volfrac_upper)
    (analytic,) = torch.autograd.grad(c, x0)

    def f(x):
        return compute_volume_constraint_torch(x, volfrac_upper).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL


def test_pnorm_stress_gradient_matches_finite_difference(structural_fe_solver):
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    pnorm, _max_vm = compute_pnorm_stress_autograd(sol, x0, fe)
    (analytic,) = torch.autograd.grad(pnorm, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        pnorm, _ = compute_pnorm_stress_autograd(sol, x, fe)
        return pnorm.item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_max_vonmises_stress_gradient_matches_finite_difference(structural_fe_solver):
    # New in Phase 4: MAX_VONMISES_STRESS reuses compute_pnorm_stress_autograd's
    # second return value rather than the old NumPy-only sensitivity path.
    #
    # max() is only piecewise-smooth: its gradient has a kink wherever two
    # elements' stress values tie. Verified directly: at the *uniform*
    # x0=0.5 this fixture's mesh has an exact symmetry tie between elements
    # 23 and 167 (von Mises stresses agree to >7 significant digits), so
    # the argmax flips between them for a perturbation as small as 1e-7 in
    # either direction -- not a floating-point-precision artifact, a real
    # kink in the objective sitting exactly at this x0. A non-uniform,
    # randomized x0 (as any real optimization iterate would have) gives a
    # unique argmax and a well-defined, cleanly-verifiable gradient
    # (confirmed separately: relative error ~1e-9 once the tie is broken).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    torch.manual_seed(42)
    x0 = (0.3 + 0.4 * torch.rand(n, dtype=torch.float64)).requires_grad_(True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    _pnorm, max_vm = compute_pnorm_stress_autograd(sol, x0, fe)
    (analytic,) = torch.autograd.grad(max_vm, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        _, max_vm = compute_pnorm_stress_autograd(sol, x, fe)
        return max_vm.item()

    direction = analytic / analytic.norm()
    step = 1e-5
    xp = (x0.detach() + step * direction).clone()
    xm = (x0.detach() - step * direction).clone()
    fd_directional = (f(xp) - f(xm)) / (2 * step)
    analytic_directional = (analytic * direction).sum().item()
    rel_err = abs(fd_directional - analytic_directional) / max(abs(fd_directional), 1e-12)
    assert rel_err < REL_TOL, f"analytic={analytic_directional:.6e} fd={fd_directional:.6e}"


def test_mass_gradient_matches_finite_difference(structural_fe_solver):
    # New torch-native QOI in Phase 4 (was NumPy-only, crashed on grad-tracked x).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = torch.rand(n, dtype=torch.float64, requires_grad=True)

    mass = compute_mass_torch(x0, fe)
    (analytic,) = torch.autograd.grad(mass, x0)

    def f(x):
        return compute_mass_torch(x, fe).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL


def test_gvector_gradient_matches_finite_difference(structural_fe_solver):
    # New torch-native QOI in Phase 4 (was NumPy-only, crashed on grad-tracked sol).
    fe = structural_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)
    rng = np.random.default_rng(0)
    g = rng.standard_normal(fe.bc.num_dofs)

    sol = fe.solve(x0, MaterialModel.SIMP)
    obj = compute_gvector_torch(sol, g)
    (analytic,) = torch.autograd.grad(obj, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        return compute_gvector_torch(sol, g).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += EPS
        xm = x0.detach().clone(); xm[e] -= EPS
        fd = (f(xp) - f(xm)) / (2 * EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_compliance_gradient_matches_finite_difference_thermal(thermal_fe_solver, thermal_KE):
    # Both-physics coverage for the Phase 4 gate: thermal compliance,
    # exercising HexThermalFEA's new torch-native solve().
    #
    # Thermal compliance/gradient are ~4 orders of magnitude smaller here
    # than structural's (O(0.1) and O(1e-5) vs O(10) and O(1) respectively),
    # so the structural test's EPS=1e-6 hits SPSOLVE's floating-point noise
    # floor for this problem (verified: relative error swings from ~2e-5 at
    # eps=1e-3/1e-4 to ~1% at eps=1e-6). A larger, thermal-appropriate step
    # avoids that -- this is a finite-difference step-size choice, not a
    # looser correctness bar.
    THERMAL_EPS = 1e-4
    fe = thermal_fe_solver
    n = fe.mesh.num_elems
    x0 = 0.5 * torch.ones(n, dtype=torch.float64, requires_grad=True)

    sol = fe.solve(x0, MaterialModel.SIMP)
    c = compute_compliance_torch(sol, x0, fe, thermal_KE, MaterialModel.SIMP)
    (analytic,) = torch.autograd.grad(c, x0)

    def f(x):
        with torch.no_grad():
            sol = fe.solve(x, MaterialModel.SIMP)
        return compute_compliance_torch(sol, x, fe, thermal_KE, MaterialModel.SIMP).item()

    for e in _sample_elements(n):
        xp = x0.detach().clone(); xp[e] += THERMAL_EPS
        xm = x0.detach().clone(); xm[e] -= THERMAL_EPS
        fd = (f(xp) - f(xm)) / (2 * THERMAL_EPS)
        rel_err = abs(fd - analytic[e].item()) / max(abs(fd), 1e-12)
        assert rel_err < REL_TOL, f"elem {e}: analytic={analytic[e].item():.6e} fd={fd:.6e}"


def test_compliance_torch_matches_reference_adjoint_value(structural_fe_solver, structural_KE):
    # Cross-check against the independently-derived legacy NumPy oracle
    # (pyto.autodiff.reference_adjoint), not just finite differences of the
    # torch code itself -- the Phase 4 gate's actual wording.
    fe = structural_fe_solver
    x0 = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64, requires_grad=True)
    sol = fe.solve(x0, MaterialModel.SIMP)
    torch_value = compute_compliance_torch(sol, x0, fe, structural_KE, MaterialModel.SIMP).item()

    sol_np = sol.detach().cpu().numpy()
    x_np = x0.detach().cpu().numpy()
    KE_np = structural_KE.detach().cpu().numpy()
    reference_value = compute_compliance(sol_np, x_np, fe, KE_np, MaterialModel.SIMP)

    assert abs(torch_value - reference_value) / abs(reference_value) < 1e-8


def test_max_vonmises_torch_matches_reference_adjoint_value(structural_fe_solver, structural_KE):
    fe = structural_fe_solver
    x0 = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64, requires_grad=True)
    sol = fe.solve(x0, MaterialModel.SIMP)
    _pnorm, max_vm_torch = compute_pnorm_stress_autograd(sol, x0, fe)

    fe.postprocess()  # populates fe.vonMisesStress/stressComponents/pNormStress
    sol_np = sol.detach().cpu().numpy()
    x_np = x0.detach().cpu().numpy()
    KE_np = structural_KE.detach().cpu().numpy()

    # compute_pnorm_stress_and_sensitivity (reference_adjoint, unmodified
    # legacy code) solves a second system via linear_solvers.solve(
    # fe.stiff_mtrx, ...), which expects a SciPy sparse matrix -- but
    # fe.stiff_mtrx is now a torch sparse tensor since solve() became
    # torch-native (Phase 3/4). Convert at this test's boundary rather than
    # touching the oracle, which is meant to stay exactly as it was.
    import scipy.sparse as sp
    stiff_coo = fe.stiff_mtrx.coalesce()
    idx = stiff_coo.indices().detach().numpy()
    vals = stiff_coo.values().detach().numpy()
    fe.stiff_mtrx = sp.coo_matrix((vals, (idx[0], idx[1])), shape=tuple(stiff_coo.shape)).tocsc()

    _vm_pnorm, _sensitivity, max_vm_reference = compute_pnorm_stress_and_sensitivity(
        sol_np, x_np, fe, KE_np, MaterialModel.SIMP
    )

    assert abs(max_vm_torch.item() - max_vm_reference) / abs(max_vm_reference) < 1e-8


def test_stress_scaling_is_isolated_per_to_params():
    # Regression test for the Phase 4 fix: stress_scaling used to be a
    # mutable attribute on the compute_constraint_and_gradient function
    # object (module-global for the whole process), so two separate
    # optimization runs sharing that attribute would silently leak state
    # into each other. It's now a plain field on TOParams.
    a = TOParams()
    b = TOParams()
    assert a.stress_scaling == 1.0
    assert b.stress_scaling == 1.0
    a.stress_scaling = 0.42
    assert b.stress_scaling == 1.0, "stress_scaling leaked across separate TOParams instances"


def test_stress_failure_factor_constraint_direction(structural_fe_solver):
    # Regression test for a sign/reciprocal bug fixed after being flagged
    # (not fixed) in Phase 4: STRESS_FAILURE_FACTOR's constraint used
    # `1.0 / constraintLimit` where it should use `constraintLimit`. Per
    # topopt_common.py's TO_QOI enum comment and
    # topopt_structural_benchmarks.py's own comment ("failure limit is the
    # opposite of safety factor"), failure_factor = stress/yieldStrength,
    # and the constraint failure_factor <= constraintLimit means
    # stress <= yieldStrength * constraintLimit. The old formula instead
    # allowed stress up to yieldStrength / constraintLimit -- with a
    # constraintLimit of 0.5 (the benchmark's own value), that permitted
    # stress up to *2x* yield strength instead of limiting it to half.
    #
    # This test doesn't need an independent ground truth for the actual
    # stress value: it finds the constraintLimit at which the constraint
    # is exactly at the boundary (c == 0) by construction, then checks c's
    # sign moves the right direction on either side of it -- self-
    # consistent regardless of the mesh/material specifics.
    fe = structural_fe_solver
    x0 = 0.5 * torch.ones(fe.mesh.num_elems, dtype=torch.float64, requires_grad=True)
    sol = fe.solve(x0, MaterialModel.SIMP)

    pnorm_stress, _max_vm = compute_pnorm_stress_autograd(sol, x0, fe)
    yield_strength = fe.mat_prop.yield_strength
    # constraintLimit at which allowed_stress == pnorm_stress exactly (c == 0)
    boundary_limit = (pnorm_stress / yield_strength).item()

    def constraint_value(constraint_limit):
        to_params = TOParams()
        to_params.Constraints = [(TO_QOI.STRESS_FAILURE_FACTOR, None, constraint_limit)]
        c, _dc = compute_constraint_and_gradient(to_params, sol, x0, fe, None, MaterialModel.SIMP)
        return c[0, 0].item()

    c_at_boundary = constraint_value(boundary_limit)
    c_looser = constraint_value(boundary_limit * 2.0)   # more allowed stress -> should satisfy (c <= 0)
    c_tighter = constraint_value(boundary_limit * 0.5)  # less allowed stress -> should violate (c > 0)

    assert abs(c_at_boundary) < 1e-8, f"expected c==0 at the boundary limit, got {c_at_boundary}"
    assert c_looser < 0, f"looser constraintLimit should satisfy the constraint, got c={c_looser}"
    assert c_tighter > 0, f"tighter constraintLimit should violate the constraint, got c={c_tighter}"


def test_stress_scaling_update_rule(structural_fe_solver):
    # Reference-era rule: every call sets s <- 0.25*max_vm/pnorm + 0.75*s, whether or not the limit is
    # exceeded (a later refactor updated only when max_vm > limit and stopped stress problems converging).
    from pyto.autodiff.qoi import compute_constraint_and_gradient
    from pyto.autodiff.qoi.stress import compute_pnorm_stress_autograd
    fe = structural_fe_solver
    torch.manual_seed(1)
    x = 0.3 + 0.4 * torch.rand(fe.mesh.num_elems, dtype=torch.float64)
    sol = fe.solve(x, MaterialModel.SIMP)
    pnorm, max_vm = compute_pnorm_stress_autograd(sol, x, fe)
    for limit in (0.5 * float(max_vm), 2.0 * float(max_vm)):  # limit below and above the actual max
        tp = TOParams()
        tp.Constraints = [(TO_QOI.MAX_VONMISES_STRESS, None, limit)]
        compute_constraint_and_gradient(tp, sol, x, fe, None, MaterialModel.SIMP)
        assert abs(tp.stress_scaling - (0.25 * float(max_vm / pnorm) + 0.75)) < 1e-12


def test_compliance_sign_only_flips_for_prescribed_value_problems():
    # FourCornersThermal / BridgeThermal have no load, only prescribed temperatures: minimizing u^T K u
    # would delete the design, so the optimizer minimizes -J there (the Reference-era behaviour).
    from types import SimpleNamespace
    from pyto.autodiff.qoi.compliance import compliance_sign
    from pyto.core.bc import BC
    zero, hot = np.zeros(4), np.array([1.0, 0.0])
    driven = SimpleNamespace(bc=BC(force=zero, fixed_dofs=np.array([0, 1]), dirichlet_values=hot), elem_body_force=None)
    loaded = SimpleNamespace(bc=BC(force=np.ones(4), fixed_dofs=np.array([0, 1]), dirichlet_values=hot), elem_body_force=None)
    fixed_only = SimpleNamespace(bc=BC(force=np.ones(4), fixed_dofs=np.array([0]), dirichlet_values=np.zeros(1)), elem_body_force=None)
    assert compliance_sign(driven) == -1.0
    assert compliance_sign(loaded) == 1.0
    assert compliance_sign(fixed_only) == 1.0


def test_mma_wrapper_handles_a_negative_objective():
    # Regression: runMMA divided f0 and its gradient by the SIGNED first objective value, so a negative
    # objective (e.g. -J/J0) was silently turned around and minimized the wrong way.
    from pyto.topopt.drivers.mmaWrapper import runMMA
    n = 6
    def fn(x):  # minimize -sum(x) s.t. mean(x) <= 0.5  -> optimum sits on the constraint at mean 0.5
        x = np.asarray(x).reshape(-1, 1)
        return np.array([[-float(x.sum())]]), -np.ones((n, 1)), np.array([[float(x.mean()) / 0.5 - 1.0]]), np.ones((1, n)) / n / 0.5
    # start well below the limit: correct behaviour adds material up to 0.5; the flipped objective removes it
    out = runMMA(n, 1, fn, 0.2 * np.ones((n, 1)), np.zeros((n, 1)), np.ones((n, 1)), maxIterations=40)
    assert np.mean(out[0]) > 0.4, f"negative objective was minimized the wrong way (final mean x = {np.mean(out[0]):.2f})"


def test_resolve_constraint_matrix_folds_single_dof_rows_and_rejects_sliding():
    import pytest
    import scipy.sparse as sp
    from pyto.core.bc import BC, resolve_constraint_matrix
    single = BC(force=np.zeros(6), fixed_dofs=[], dirichlet_values=[],  # empty lists, as the problem builders pass
                constraint_matrix=sp.csr_matrix(np.array([[1.0, 0, 0, 0, 0, 0], [0, 0, 2.0, 0, 0, 0]])),
                constraint_rhs=np.array([0.0, 4.0]))
    resolve_constraint_matrix(single)
    assert single.constraint_matrix is None
    assert dict(zip(single.fixed_dofs.tolist(), single.dirichlet_values.tolist())) == {0: 0.0, 2: 2.0}
    sliding = BC(force=np.zeros(6), fixed_dofs=np.array([5]), dirichlet_values=np.zeros(1),
                 constraint_matrix=sp.csr_matrix(np.array([[1.0, -1.0, 0, 0, 0, 0]])), constraint_rhs=np.zeros(1))
    with pytest.raises(NotImplementedError, match="couple several DOFs"):
        resolve_constraint_matrix(sliding)


def test_volume_fraction_min_is_a_lower_bound():
    # VOLUME_FRACTION_MIN: c = 1 - mean(x)/limit <= 0, i.e. mean(x) >= limit. Lets self-weight problems
    # (whose unconstrained optimum is "almost no material") keep a minimum amount of material.
    from pyto.autodiff.qoi import compute_constraint_and_gradient
    tp = TOParams()
    tp.Constraints = [(TO_QOI.VOLUME_FRACTION_MIN, None, 0.5)]
    x = torch.full((10,), 0.25, dtype=torch.float64, requires_grad=True)
    c, _ = compute_constraint_and_gradient(tp, None, x, None, None, MaterialModel.SIMP)
    assert abs(float(c[0, 0]) - 0.5) < 1e-12           # mean 0.25 < 0.5: violated
    (g,) = torch.autograd.grad(c[0, 0], x)
    assert torch.all(g < 0)                             # adding material reduces the violation
    x2 = torch.full((10,), 0.75, dtype=torch.float64)
    c2, _ = compute_constraint_and_gradient(tp, None, x2, None, None, MaterialModel.SIMP)
    assert float(c2[0, 0]) < 0                          # mean 0.75 >= 0.5: satisfied
