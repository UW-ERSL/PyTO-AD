import torch
import numpy as np
import scipy.sparse as spy_sprs
from scipy.sparse.linalg import spsolve
from typing import Dict
from scipy.sparse.linalg import splu

# Try to use PARDISO if available
try:
    import pypardiso
except ImportError:
    pypardiso = None

# Try to use PETSc if available
try:
    import petsc4py.PETSc as PETSc
except ImportError:
    PETSc = None


from pyto.solve.solvers import Solvers

def _petsc_solve(A_csr: spy_sprs.csr_matrix, rhs, solver_options: dict = None):
    """
    Solve A x = rhs using PETSc via petsc4py, using options
    from the provided dictionary.
    
    This function expects 'solver_options' to be the dictionary that
    might contain a 'petsc_solver' sub-dictionary, matching the JAX logic.
    """
    if PETSc is None:
        raise ImportError("petsc4py is not available; cannot use PETSC solver.")

    if solver_options is None:
        solver_options = {}

    # This logic correctly parses the nested dictionary, just like the JAX version.
    petsc_opts = solver_options.get("petsc_solver", {})
    # --- CHANGED LINES ---
    # Set the DEFAULT ksp_type to "preonly"
    ksp_type = petsc_opts.get("ksp_type", "preonly")
    # Set the DEFAULT pc_type to "cholesky"
    pc_type = petsc_opts.get("pc_type", "cholesky")
    # --- END CHANGED LINES ---

    A_petsc = PETSc.Mat().createAIJ(
        size=A_csr.shape,
        csr=(
            A_csr.indptr.astype(PETSc.IntType, copy=False),
            A_csr.indices.astype(PETSc.IntType, copy=False),
            A_csr.data,
        ),
    )

    rhs_vec = PETSc.Vec().createSeq(len(rhs))
    rhs_vec.setValues(range(len(rhs)), np.asarray(rhs))

    ksp = PETSc.KSP().create()
    ksp.setOperators(A_petsc)
    ksp.setFromOptions()  # Allow command-line overrides

    # Set types from options dict (or defaults)
    ksp.setType(ksp_type)
    ksp.getPC().setType(pc_type)

    # Add the special case from JAX version
    if ksp_type == "tfqmr":
        ksp.getPC().setFactorSolverType("mumps")

    x_vec = PETSc.Vec().createSeq(len(rhs))
    ksp.solve(rhs_vec, x_vec)

    return x_vec.getArray()


def _direct_solve(
    A: spy_sprs.spmatrix, rhs, solver_kind: str, solver_options: dict = None
):
    """Solve A x = rhs using the requested backend."""
    A_csr = A.tocsr()

    if solver_kind == "petsc" and PETSc is not None:
        # Pass the entire options dictionary
        x = _petsc_solve(A_csr, rhs, solver_options)
    elif solver_kind == "pardiso" and pypardiso is not None:
        x = pypardiso.spsolve(A_csr, rhs)
        pypardiso.ps.free_memory()
    else:
        x = spsolve(A_csr, rhs)
    return x


_PARDISO_POOL = []   # idle PyPardisoSolver objects: constructing one costs ~0.5 s (MKL set-up), far more than a solve
# Idle solvers that keep PARDISO's symbolic analysis (fill-reducing ordering, phase 11) of one sparsity pattern, by
# pattern. An optimization factorizes matrices of one fixed pattern every iteration; the analysis depends only on the
# pattern (about 6 % of a factorization at 300k dof), so it is done once and each later factorization is numeric only
# (phase 22). Releasing a factorization frees only its factors (phase 0) and keeps the analysis.
_PARDISO_ANALYZED: Dict[tuple, list] = {}
_PARDISO_IDLE_MAX = 8


def _pattern_key(A_csr):
    import zlib
    return (A_csr.shape, A_csr.nnz, zlib.crc32(np.ascontiguousarray(A_csr.indptr)),
            zlib.crc32(np.ascontiguousarray(A_csr.indices)))


def _pardiso_factorize(A_csr):
    """A PyPardisoSolver holding the factors of A_csr, and A's pattern key: numeric factorization only (phase 22)
    when an idle solver already holds the analysis of this pattern, analysis + factorization (phase 12) otherwise."""
    key = _pattern_key(A_csr)
    analyzed = _PARDISO_ANALYZED.get(key)
    if analyzed:
        solver, phase = analyzed.pop(), 22
    else:
        solver, phase = (_PARDISO_POOL.pop() if _PARDISO_POOL else pypardiso.PyPardisoSolver()), 12
    solver._check_A(A_csr)                          # as PyPardisoSolver.factorize, with the phase chosen here
    solver.factorized_A = (solver._hash_csr_matrix(A_csr) if A_csr.nnz > solver.size_limit_storage
                           else A_csr.copy())
    solver.set_phase(phase)
    solver._call_pardiso(A_csr, np.zeros((A_csr.shape[0], 1)))
    return solver, key


def _pardiso_release(solver, key):
    """Free the factors but keep the pattern's analysis for the next factorization of that pattern."""
    idle = sum(len(v) for v in _PARDISO_ANALYZED.values())
    if idle < _PARDISO_IDLE_MAX:
        solver.free_memory(False)                   # phase 0: L and U only
        _PARDISO_ANALYZED.setdefault(key, []).append(solver)
    else:
        solver.free_memory(True)                    # phase -1: everything
        if len(_PARDISO_POOL) < 8:
            _PARDISO_POOL.append(solver)


class _PardisoFactors:
    """Keeps a PARDISO factorization alive from the forward solve to the adjoint solve(s).

    The adjoint of a symmetric system (stiffness, conductivity; Dirichlet rows/columns are replaced symmetrically)
    is the same matrix, so the backward pass solves with these factors instead of factorizing again (plan S8). At
    38k dof: forward (factorize + solve) ~208 ms, adjoint with stored factors ~33 ms instead of ~200 ms.

    PyPardisoSolver.solve() only reuses factors made by an explicit factorize() call, hence the two-step forward.
    The backward may run several times on one graph (objective + each constraint), so the factors are released only
    when autograd frees the graph: weakref.finalize on this holder, which lives on the autograd context. The solver
    object then goes back to a small pool (PyPardisoSolver has no __del__ that would free the factors itself).
    """

    def __init__(self, A_csr, symmetric=None):
        import weakref
        self.A = A_csr
        self.solver, key = _pardiso_factorize(A_csr)
        # the FE solvers say their matrices are symmetric (stiffness, conductivity); check only when not told
        self.symmetric = (bool(symmetric) if symmetric is not None
                          else abs(A_csr - A_csr.T).max() <= 1e-12 * abs(A_csr).max())
        weakref.finalize(self, _pardiso_release, self.solver, key)

    def solve(self, b):
        return self.solver.solve(self.A, b)     # phase 33: solve with the stored factors


class _LUFactors:
    """SciPy SuperLU factors (SPSOLVE): forward and transposed solves reuse them."""

    def __init__(self, A_csc):
        self.lu = splu(A_csc)

    def solve(self, b):
        return self.lu.solve(b)

    def solve_transpose(self, b):
        return self.lu.solve(b, "T")


class _PardisoSolve:
    """PARDISO factors kept for transposed (adjoint) solves; unsymmetric matrices factorize A^T when asked."""

    def __init__(self, A_csr, symmetric=None):
        self.factors = _PardisoFactors(A_csr, symmetric)

    def solve(self, b):
        return self.factors.solve(b)

    def solve_transpose(self, b):
        if self.factors.symmetric:
            return self.factors.solve(b)
        return _direct_solve(self.factors.A.T.tocsc(), b, "pardiso")


class _DirectEachTime:
    """Solvers without reusable factors here (PETSc, or PARDISO missing): each solve is a fresh direct solve."""

    def __init__(self, A_csc, kind, options):
        self.A, self.kind, self.options = A_csc, kind, options

    def solve(self, b):
        return _direct_solve(self.A, b, self.kind, self.options)

    def solve_transpose(self, b):
        return _direct_solve(self.A.T.tocsc(), b, self.kind, self.options)


class _IterativeSolve:
    """Deflated (DPCG) or plain Jacobi-preconditioned CG on the Dirichlet-modified matrix, for large problems.

    The matrix is symmetric positive definite (fixed rows and columns are replaced by the identity), so the adjoint
    solve is another CG solve with the same matrix and preconditioner. DPCG needs `dsolver` (a DeflationSolver
    with its deflation matrix built for the mesh, as build_benchmark does above DIRECT_SOLVER_DOF_CUTOFF); without
    it, or when its deflation space does not match the system (thermal: one dof per node), plain PCG is used.
    """

    def __init__(self, A_csr, kind, options):
        self.A = A_csr
        self.rtol = float(options.get("rtol", 1e-8))
        diag = A_csr.diagonal()
        diag[np.abs(diag) < 1e-12] = 1.0
        self.M = spy_sprs.diags(1.0 / diag)
        dsolver = options.get("dsolver")
        W = getattr(dsolver, "W", None)
        self.dsolver = dsolver if kind == "dpcg" and W is not None and W.shape[0] == A_csr.shape[0] else None
        self.W = W if self.dsolver is not None else None

    def solve(self, b):
        if self.dsolver is not None:
            return self.dsolver.deflatedPCG(self.A, np.asarray(b, dtype=float), W=self.W, M=self.M, rtol=self.rtol)
        from scipy.sparse.linalg import cg
        x, info = cg(self.A, b, M=self.M, rtol=self.rtol, maxiter=10 * self.A.shape[0])
        if info != 0:
            import warnings
            warnings.warn(f"PCG did not converge (info={info}); the result may be inaccurate.", stacklevel=3)
        return x

    def solve_transpose(self, b):
        return self.solve(b)          # symmetric


def _factorize(A_csr, solver_kind, solver_params):
    """The solver object for A (CSR): solve(b) and solve_transpose(b), reusing whatever the solver can reuse."""
    if solver_kind == "spsolve":
        return _LUFactors(A_csr.tocsc())              # SuperLU wants CSC
    if solver_kind == "pardiso" and pypardiso is not None:
        return _PardisoSolve(A_csr, solver_params.get("symmetric"))
    if solver_kind in ("dpcg", "pcg"):
        return _IterativeSolve(A_csr, solver_kind, solver_params)
    return _DirectEachTime(A_csr, solver_kind, solver_params)


def _solve_forward(ctx, indices, values, shape, b, solver_kind, solver_params):
    """Factorize the coalesced matrix (indices, values), solve for b, and keep what the adjoint needs on ctx."""
    idx = indices.detach().cpu().numpy()
    vals = values.detach().cpu().numpy()
    b_np = b.detach().cpu().numpy()

    # A coalesced tensor lists its entries row by row, columns ascending: that is CSR order already, so the
    # matrix is built without sorting (a COO -> CSC -> CSR round trip sorted all entries twice per solve).
    indptr = np.zeros(shape[0] + 1, dtype=np.int64)
    np.cumsum(np.bincount(idx[0], minlength=shape[0]), out=indptr[1:])
    A_csr = spy_sprs.csr_matrix((vals, idx[1], indptr), shape=shape)
    A_csr.has_sorted_indices = True

    ctx.mtrx_shape = shape
    ctx.b_dtype = b.dtype
    ctx.b_device = b.device

    # One solver object per forward solve: the adjoint (backward) and any analytical-sensitivity solves
    # use the same matrix and the same factors/preconditioner (see FE solvers' adjoint_solve).
    factorization = _factorize(A_csr, solver_kind, solver_params)
    x_np = factorization.solve(b_np)
    ctx.factorization = factorization
    keep = solver_params.get("keep")
    if keep is not None:
        keep["factorization"] = factorization

    # Save minimal tensors for backward
    ctx.save_for_backward(indices, torch.from_numpy(np.asarray(x_np)))

    return torch.tensor(x_np, dtype=b.dtype, device=b.device)


def _solve_backward(ctx, grad_x, need_vals, need_b):
    """Adjoint A^T y = grad_x with the forward solve's factors / preconditioner. Returns the gradients w.r.t. the
    stored values (-y_i x_j at each entry (i, j)) and w.r.t. b (y), None where not needed."""
    indices, x_torch = ctx.saved_tensors
    y_np = ctx.factorization.solve_transpose(grad_x.cpu().numpy())
    y_t = torch.tensor(y_np, dtype=ctx.b_dtype, device=ctx.b_device)
    grad_vals = None
    if need_vals:
        grad_vals = -(y_t.index_select(0, indices[0]) * x_torch.index_select(0, indices[1]))
    return grad_vals, (y_t if need_b else None)


class SparseLinearSolve(torch.autograd.Function):
    """x = A^-1 b for a torch sparse A. Its gradient w.r.t. A is a sparse tensor, which autograd maps back onto
    whatever built A with sparse_mask (sorting the gradient) on every solve; the FE solvers use SparseValuesSolve."""

    @staticmethod
    def forward(ctx, mtrx, b, solver_kind, solver_params):
        mtrx = mtrx.coalesce()
        return _solve_forward(ctx, mtrx.indices(), mtrx.values(), mtrx.shape, b, solver_kind, solver_params)

    @staticmethod
    def backward(ctx, grad_x):
        grad_vals, grad_b = _solve_backward(ctx, grad_x, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        grad_A_sparse = None
        if grad_vals is not None:
            grad_A_sparse = torch.sparse_coo_tensor(ctx.saved_tensors[0], grad_vals, ctx.mtrx_shape)
        return grad_A_sparse, grad_b, None, None


class SparseValuesSolve(torch.autograd.Function):
    """SparseLinearSolve for A given by the values of a fixed, coalesced pattern (indices). The gradient w.r.t. the
    values is a dense vector, so no sparse tensor is in the autograd graph: at 318k dofs the sparse_mask of
    SparseLinearSolve's gradient took 1.8 s per solve, longer than a PARDISO factorization at 59k dofs."""

    @staticmethod
    def forward(ctx, values, b, indices, shape, solver_kind, solver_params):
        return _solve_forward(ctx, indices, values, shape, b, solver_kind, solver_params)

    @staticmethod
    def backward(ctx, grad_x):
        grad_vals, grad_b = _solve_backward(ctx, grad_x, ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return grad_vals, grad_b, None, None, None, None


def solve(
    mtrx: torch.Tensor, b: torch.Tensor, solver=None, solver_options: dict = None, keep: dict = None
) -> torch.Tensor:
    """
    Wrapper for SparseLinearSolve.apply with a linear-solver choice.

    Args:
        mtrx: Sparse torch tensor (ndof, ndof).
        b: Dense torch tensor (ndof,).
        solver: Optional solver enum (e.g., Solvers.PETSC /
                PARDISO / SPSOLVE).
        solver_options: Optional dictionary of solver-specific settings,
                        mirroring the JAX implementation's `params`.
                        E.g.: {"petsc_solver": {"ksp_type": "gmres", "pc_type": "lu"}};
                        for DPCG/PCG: {"dsolver": DeflationSolver, "rtol": 1e-8}.
        keep: Optional dict; receives keep["factorization"], the solver object of this solve (solve /
              solve_transpose with the same factors), for analytical adjoint solves outside autograd.

    Returns:
        x: Solution tensor (ndof,).
    """
    solver_kind, solver_options = _solver_setup(solver, solver_options, keep)
    return SparseLinearSolve.apply(mtrx, b, solver_kind, solver_options)


def solve_values(
    indices: torch.Tensor, values: torch.Tensor, shape, b: torch.Tensor, solver=None, solver_options: dict = None,
    keep: dict = None
) -> torch.Tensor:
    """solve() for a matrix given as the values of a coalesced pattern (indices: (2, nnz), row-major and unique),
    differentiable w.r.t. values and b without sparse tensors in the graph (see SparseValuesSolve)."""
    solver_kind, solver_options = _solver_setup(solver, solver_options, keep)
    return SparseValuesSolve.apply(values, b, indices, tuple(shape), solver_kind, solver_options)


def _solver_setup(solver, solver_options, keep):
    """(solver_kind, solver_options) for the solve Functions, from a Solvers enum and the caller's options."""
    # We don't import linear_solvers here; we just look at .name if present.
    solver_name = getattr(solver, "name", None)
    kinds = {"PETSC": "petsc", "PARDISO": "pardiso", "SPSOLVE": "spsolve", "SPLU": "spsolve", None: "spsolve",
             "DPCG": "dpcg", "PCG": "pcg"}
    if solver_name in kinds:
        solver_kind = kinds[solver_name]
    else:
        # PYAMG is only implemented for pyto.solve.numpy_backend (free-dof elimination); here it falls back.
        import warnings
        warnings.warn(
            f"Solvers.{solver_name} is not implemented for the torch-autodiff sparse solve path "
            "(pyto.autodiff.sparse_solve) -- falling back to a direct SPSOLVE factorization.",
            stacklevel=3,
        )
        solver_kind = "spsolve"

    # Pass an empty dict if None, to be handled gracefully downstream
    solver_options = dict(solver_options or {})
    if keep is not None:
        keep["factorization"] = None          # release the previous solve's factors before factorizing again
        solver_options["keep"] = keep

    return solver_kind, solver_options
