"""Gradcheck the core autodiff primitive: SparseLinearSolve.

This is deliberately isolated from any FEA/mesh machinery -- it verifies
the hand-written adjoint in torch_spsolve.SparseLinearSolve.backward()
(solve A^T y = grad_x; grad_b = y; grad_A[i,j] = -y_i * x_j) against
torch.autograd.gradcheck's numerical Jacobian on a small random sparse
SPD system, independent of any physics.
"""
import numpy as np
import torch

from pyto.autodiff.sparse_solve import SparseLinearSolve, Solvers


def _random_spd_sparse(n, density=0.4, seed=0):
    rng = np.random.default_rng(seed)
    dense = rng.uniform(-1.0, 1.0, size=(n, n))
    dense[rng.uniform(size=(n, n)) > density] = 0.0
    spd = dense @ dense.T + n * np.eye(n)  # guarantee SPD and well-conditioned
    idx = np.nonzero(spd)
    indices = torch.tensor(np.array(idx), dtype=torch.long)
    values = torch.tensor(spd[idx], dtype=torch.float64, requires_grad=True)
    return indices, values, spd.shape


def test_sparse_linear_solve_gradcheck():
    torch.manual_seed(0)
    n = 8
    indices, values, shape = _random_spd_sparse(n)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def f(values, b):
        mtrx = torch.sparse_coo_tensor(indices, values, shape).coalesce()
        return SparseLinearSolve.apply(mtrx, b, "spsolve", {})

    assert torch.autograd.gradcheck(f, (values, b), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_unimplemented_solver_warns_before_falling_back():
    # PYAMG has no torch-path implementation; it must say so, not silently substitute a direct solve.
    import pytest
    from pyto.autodiff.sparse_solve import solve
    indices, values, shape = _random_spd_sparse(6)
    A = torch.sparse_coo_tensor(indices, values.detach(), shape)
    with pytest.warns(UserWarning, match="not implemented"):
        solve(A, torch.ones(6, dtype=torch.float64), solver=Solvers.PYAMG)


def test_iterative_solve_gradcheck():
    """PCG/DPCG in the differentiable solve: forward and adjoint (transposed) CG solves pass gradcheck."""
    indices, values, shape = _random_spd_sparse(8)
    b = torch.randn(8, dtype=torch.float64, requires_grad=True)

    def f(values, b):
        A = torch.sparse_coo_tensor(indices, values, shape)
        return SparseLinearSolve.apply(A, b, "dpcg", {"rtol": 1e-14})     # no dsolver: Jacobi PCG

    assert torch.autograd.gradcheck(f, (values, b), eps=1e-6, atol=1e-5, rtol=1e-4)


def _dpcg_solver(mesh):
    from pyto.solve.deflation import DeflationSolver
    d = DeflationSolver()
    d.create_deflation_groups(mesh, min(d.maxGroups, max(d.minGroups, round(3 * mesh.num_nodes / d.dofPerGroup))))
    d.create_deflation_matrix(mesh)
    return d


def test_dpcg_matches_direct_solve_and_its_gradients(structural_problem):
    """DPCG (deflation + Jacobi) on the Dirichlet-modified stiffness: same displacement and autograd gradient as
    SPSOLVE, and the analytical p-norm stress gradient (adjoint via adjoint_solve) equals autograd."""
    import numpy as np
    from pyto.autodiff.material_model import MaterialModel
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    from pyto.topopt.common import TOParams, TO_QOI
    from pyto.topopt.manual_sensitivities import manual_gradients
    from pyto.autodiff.qoi import compute_objective_and_gradient
    mesh, mat, bc, ebf, _ = structural_problem
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(1).random(mesh.num_elems), dtype=torch.float64)
    p = TOParams()
    p.Objective, p.Constraints = (TO_QOI.PNORM_STRESS, None), [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    out = {}
    for name, solver, dsolver in (("direct", Solvers.SPSOLVE, None), ("dpcg", Solvers.DPCG, _dpcg_solver(mesh))):
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver, dsolver=dsolver, rtol=1e-12)
        x = x0.clone().requires_grad_(True)
        u = fe.solve(x, MaterialModel.SIMP)
        J = compute_objective_and_gradient(p, u, x, fe, torch.tensor(fe.elem_stiff[0]), MaterialModel.SIMP)
        g_ad = torch.autograd.grad(J, x)[0].numpy()
        g_an = manual_gradients(p, u, x0, fe, MaterialModel.SIMP)[0]
        out[name] = (u.detach().numpy(), g_ad, g_an)
        assert type(fe._last_solve["factorization"]).__name__ == ("_IterativeSolve" if name == "dpcg" else "_LUFactors")
    rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
    assert rel(out["dpcg"][0], out["direct"][0]) < 1e-8
    assert rel(out["dpcg"][1], out["direct"][1]) < 1e-6
    for name in out:
        assert rel(out[name][2], out[name][1]) < 1e-6, name             # analytical == autograd, same solver


def test_adjoint_solve_reuses_the_forward_factorization(structural_problem):
    """The analytical adjoint uses the last solve's solver object: no new factorization (PARDISO and SPSOLVE)."""
    import numpy as np
    import pyto.autodiff.sparse_solve as S
    from pyto.autodiff.material_model import MaterialModel
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    mesh, mat, bc, ebf, _ = structural_problem
    solvers = [Solvers.SPSOLVE] + ([Solvers.PARDISO] if S.pypardiso is not None else [])
    for solver in solvers:
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver)
        with torch.no_grad():
            u = fe.solve(torch.full((mesh.num_elems,), 0.5, dtype=torch.float64), MaterialModel.SIMP)
        made = []
        original = S._factorize
        S._factorize = lambda *a: made.append(1) or original(*a)
        try:
            lam = fe.adjoint_solve(fe.bc.force)                   # K lam = f (fixed dofs zero) must give u
        finally:
            S._factorize = original
        assert not made, solver
        free = np.setdiff1d(np.arange(bc.num_dofs), bc.fixed_dofs)
        assert np.abs(lam[free] - u.numpy()[free]).max() < 1e-9 * np.abs(u.numpy()).max(), solver


# ---------------------------------------------------------------- plan S8: PARDISO factor reuse in the adjoint
def test_pardiso_adjoint_reuses_factors_and_matches_spsolve(structural_problem):
    import gc
    import weakref
    import numpy as np
    import torch
    pytest = __import__("pytest")
    sp = pytest.importorskip("pypardiso")
    import pyto.autodiff.sparse_solve as S
    import pyto.solve.numpy_backend as lin
    from pyto.autodiff.material_model import MaterialModel
    from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
    mesh, mat, bc, ebf, _ = structural_problem
    x0 = torch.tensor(0.3 + 0.6 * np.random.default_rng(0).random(mesh.num_elems), dtype=torch.float64)

    def grads(solver):
        fe = HexStructuralFEA(mesh=mesh, mat_prop=mat, bc=bc, solver=solver)
        x = x0.clone().requires_grad_(True)
        u = fe.solve(x, MaterialModel.SIMP)
        f1, f2 = (u ** 2).sum(), u.abs().max() + u[::7].sum()
        g1 = torch.autograd.grad(f1, x, retain_graph=True)[0]      # two backward passes on one graph,
        g2 = torch.autograd.grad(f2, x)[0]                          # as MMA does for objective + constraints
        return g1, g2

    created = []
    original_init = S._PardisoFactors.__init__

    def spy(self, A, symmetric=None):
        original_init(self, A, symmetric)
        created.append(weakref.ref(self))

    S._PardisoFactors.__init__ = spy
    try:
        p1, p2 = grads(lin.Solvers.PARDISO)
    finally:
        S._PardisoFactors.__init__ = original_init
    s1, s2 = grads(lin.Solvers.SPSOLVE)
    assert torch.allclose(p1, s1, rtol=1e-8, atol=1e-12 * s1.abs().max())
    assert torch.allclose(p2, s2, rtol=1e-8, atol=1e-12 * s2.abs().max())
    gc.collect()
    assert created and all(r() is None for r in created)            # factors freed with the graph
