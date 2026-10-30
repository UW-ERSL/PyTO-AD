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

    def __init__(self, A_csr):
        import weakref
        self.A = A_csr
        self.solver = _PARDISO_POOL.pop() if _PARDISO_POOL else pypardiso.PyPardisoSolver()
        self.solver.factorize(A_csr)
        self.symmetric = abs(A_csr - A_csr.T).max() <= 1e-12 * abs(A_csr).max()
        weakref.finalize(self, _PardisoFactors._release, self.solver)

    @staticmethod
    def _release(solver):
        solver.free_memory(True)
        if len(_PARDISO_POOL) < 8:
            _PARDISO_POOL.append(solver)

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

    def __init__(self, A_csr):
        self.factors = _PardisoFactors(A_csr)

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
        return _PardisoSolve(A_csr)
    if solver_kind in ("dpcg", "pcg"):
        return _IterativeSolve(A_csr, solver_kind, solver_params)
    return _DirectEachTime(A_csr, solver_kind, solver_params)


class SparseLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mtrx, b, solver_kind, solver_params):
        mtrx = mtrx.coalesce()
        idx = mtrx.indices().detach().cpu().numpy()
        vals = mtrx.values().detach().cpu().numpy()
        shape = mtrx.shape
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
        ctx.save_for_backward(mtrx.indices(), mtrx.values(), torch.from_numpy(np.asarray(x_np)))

        return torch.tensor(x_np, dtype=b.dtype, device=b.device)

    @staticmethod
    def backward(ctx, grad_x):
        mtrx_indices, mtrx_values, x_torch = ctx.saved_tensors
        shape = ctx.mtrx_shape

        grad_A_sparse = None
        grad_b = None

        # adjoint: A^T y = grad_x, with the forward solve's factors / preconditioner
        y_np = ctx.factorization.solve_transpose(grad_x.cpu().numpy())

        y_t = torch.tensor(y_np, dtype=ctx.b_dtype, device=ctx.b_device)

        if ctx.needs_input_grad[1]:
            grad_b = y_t

        if ctx.needs_input_grad[0]:
            rows, cols = mtrx_indices[0], mtrx_indices[1]
            y_rows = y_t.index_select(0, rows)
            x_cols = x_torch.index_select(0, cols)
            grad_vals = -(y_rows * x_cols)
            grad_A_sparse = torch.sparse_coo_tensor(mtrx_indices, grad_vals, shape)

        return grad_A_sparse, grad_b, None, None


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
            stacklevel=2,
        )
        solver_kind = "spsolve"

    # Pass an empty dict if None, to be handled gracefully downstream
    solver_options = dict(solver_options or {})
    if keep is not None:
        keep["factorization"] = None          # release the previous solve's factors before factorizing again
        solver_options["keep"] = keep

    return SparseLinearSolve.apply(mtrx, b, solver_kind, solver_options)
