"""Boundary conditions module."""

import dataclasses
import numpy as np
import scipy.sparse as spy_sprs
import torch


@dataclasses.dataclass
class BC:
  """Class for handling boundary conditions.

  Attributes:
    force: Array of size (num_dofs,) that contain the imposed load on each dof.
    fixed_dofs: Array of size (num_fixed_dofs,) that contain all the dof numbers
      that are fixed.
    dirichlet_values: Array of size (num_fixed_dofs,) that contain the values of
      the fixed dirichlet BC.
  """
  force: np.ndarray
  fixed_dofs: np.ndarray
  dirichlet_values: np.ndarray
  constraint_matrix: np.ndarray = None # for example, sliding bc
  constraint_rhs: np.ndarray = None # for example, zeros, for sliding bc

  @property
  def num_dofs(self)->int:
    return self.force.shape[0]


  @property
  def free_dofs(self)-> np.ndarray:
    # cached for the fixed dofs it was built from; read-only so a caller cannot change the cached copy
    key = (self.num_dofs, np.asarray(self.fixed_dofs).tobytes())
    cache = getattr(self, "_free_dofs_cache", None)
    if cache is None or cache[0] != key:
      free = np.setdiff1d(np.arange(self.num_dofs), self.fixed_dofs)
      free.flags.writeable = False
      cache = self._free_dofs_cache = (key, free)
    return cache[1]

  def set_force(self, force: np.ndarray):
    """Set the force array.
    Useful for transient problems.
    """
    self.force = force


def resolve_constraint_matrix(bc: "BC") -> None:
  """Make `bc` usable by the torch solve, which only knows fixed_dofs/dirichlet_values.

  A constraint row with a single non-zero (coef * u_j = rhs) is just a Dirichlet condition u_j = rhs / coef,
  so those rows are folded into fixed_dofs/dirichlet_values and constraint_matrix is cleared. A row that
  couples several DOFs (e.g. a sliding support) cannot be expressed that way and is not supported by the
  torch solve: raise instead of silently ignoring it. Also normalizes fixed_dofs/dirichlet_values to arrays
  (problems may pass empty lists). Safe to call repeatedly.
  """
  fixed = np.asarray(bc.fixed_dofs, dtype=np.int64).reshape(-1)
  values = np.asarray(bc.dirichlet_values, dtype=float).reshape(-1)
  if bc.constraint_matrix is not None:
    C = spy_sprs.csr_matrix(bc.constraint_matrix)
    rhs = np.zeros(C.shape[0]) if bc.constraint_rhs is None else np.asarray(bc.constraint_rhs, dtype=float).reshape(-1)
    if np.any(np.diff(C.indptr) != 1):
      raise NotImplementedError(
          "constraint_matrix rows that couple several DOFs (e.g. sliding supports) are not supported by the "
          "torch solve; only single-DOF rows (plain Dirichlet conditions) are.")
    merged = dict(zip(fixed.tolist(), values.tolist()))
    merged.update(zip(C.indices.tolist(), (rhs / C.data).tolist()))
    fixed, values = np.array(list(merged), dtype=np.int64), np.array(list(merged.values()), dtype=float)
    bc.constraint_matrix, bc.constraint_rhs = None, None
  bc.fixed_dofs, bc.dirichlet_values = fixed, values


def _rigid_body_mode_matrix(node_xyz: np.ndarray, node_ids: np.ndarray,
                             comps: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
  """Evaluate the 6 classical small-rigid-body-displacement fields
  (translate x/y/z, rotate x/y/z about `reference_point`) at a set of
  (node, component) dofs. Row i, column j is mode j's displacement value
  at dof i. See check_rigid_body_constraint for how this is used.
  """
  rel = node_xyz[node_ids] - reference_point
  x, y, z = rel[:, 0], rel[:, 1], rel[:, 2]
  M = np.zeros((len(node_ids), 6))
  M[comps == 0, 0] = 1.0
  M[comps == 1, 1] = 1.0
  M[comps == 2, 2] = 1.0
  M[comps == 0, 4] = z[comps == 0]   # rotate_y: u_x = z
  M[comps == 0, 5] = -y[comps == 0]  # rotate_z: u_x = -y
  M[comps == 1, 3] = -z[comps == 1]  # rotate_x: u_y = -z
  M[comps == 1, 5] = x[comps == 1]   # rotate_z: u_y = x
  M[comps == 2, 3] = y[comps == 2]   # rotate_x: u_z = y
  M[comps == 2, 4] = -x[comps == 2]  # rotate_y: u_z = -x
  return M


def check_rigid_body_constraint(mesh, bc: "BC", tol: float = 1e-6) -> tuple[bool, float]:
  """Check whether bc's fixed_dofs + force together make a well-posed 3D
  structural equilibrium problem -- not just whether every rigid-body
  mode (3 translations + 3 rotations) is individually constrained.

  A direct solver doesn't reliably signal near-singularity the way an
  iterative solver's failure-to-converge does -- if the fixed-DOF node
  set leaves any rigid-body mode free (e.g. because the fixed nodes are
  colinear, so rotation about that line costs zero strain energy), a
  load with a nonzero component along that free mode produces a
  numerically meaningless displacement -- and hence compliance -- while
  still solving "successfully". This was the root cause of a real,
  previously-silent bug: LBracketThickMidLoad's fixed nodes are colinear
  (all at the same y, z), so rotation about that line is unconstrained,
  and its off-axis load directly excites that mode, producing a
  compliance of roughly -1e11 (compliance can never be negative)
  reported as success.

  Critically, an unconstrained rigid-body mode is only an actual problem
  if the applied load has a nonzero generalized-force component along
  it -- otherwise the mode is simply never excited and the solve remains
  perfectly well-defined (e.g. LBracketThickTopLoad has the exact same
  colinear support as LBracketThickMidLoad and is therefore also
  "missing" a mode by the naive test, but its load happens to lie in the
  plane containing the free axis and so never excites it; a 2.5D
  extruded model like Mitchell is even more extreme -- translation along
  the extrusion axis is completely free, which is normal and harmless as
  long as nothing ever pushes in that direction). So this checks force
  compatibility with the null space, not just the null space itself --
  verified directly to have zero false positives across every benchmark
  problem in topopt_structural_benchmarks.py.

  Args:
    mesh: Mesh object (uses node_xyz, num_nodes).
    bc: BC object (uses fixed_dofs, force). Always reports well-posed
      when bc uses a constraint_matrix instead of fixed_dofs (e.g.
      sliding BCs), since that's a different, not-yet-covered constraint
      mechanism.
    tol: Singular-value threshold, relative to the largest singular
      value, below which a rigid-body mode is considered unconstrained.

  Returns:
    (well_posed, worst_relative_component): well_posed is False only when
    an unconstrained mode is actually excited by bc.force above a tiny
    relative tolerance; worst_relative_component is the largest such
    (dimensionless) excitation found, for use in an error message (None
    if every mode was already fully constrained, i.e. nothing to check).
  """
  if bc.constraint_matrix is not None:
    return True, None
  fixed_dofs = np.asarray(bc.fixed_dofs)
  reference_point = mesh.node_xyz.mean(axis=0)
  if len(fixed_dofs) == 0:
    return (not np.any(bc.force)), None

  fixed_nodes = fixed_dofs // 3
  fixed_comps = fixed_dofs % 3
  M_fixed = _rigid_body_mode_matrix(mesh.node_xyz, fixed_nodes, fixed_comps, reference_point)
  _U, S, Vt = np.linalg.svd(M_fixed, full_matrices=True)
  S_full = np.concatenate([S, np.zeros(6 - len(S))]) if len(S) < 6 else S
  threshold = tol * max(S_full[0], 1e-12)
  free_mask = S_full < threshold
  if not free_mask.any():
    return True, None
  free_directions = Vt[free_mask]  # (k, 6): combined rigid-body modes left unconstrained

  if not np.any(bc.force):
    return True, None

  all_node_ids = np.repeat(np.arange(mesh.num_nodes), 3)
  all_comps = np.tile(np.arange(3), mesh.num_nodes)
  M_all = _rigid_body_mode_matrix(mesh.node_xyz, all_node_ids, all_comps, reference_point)
  generalized_force = M_all.T @ bc.force  # (6,): work-conjugate force/moment resultants

  char_force = np.abs(bc.force).sum()
  char_length = max(np.ptp(mesh.node_xyz, axis=0).max(), 1e-12)
  scale = char_force * char_length

  worst = max(abs(generalized_force @ v) for v in free_directions) / scale
  return worst < 1e-6, worst


def impose_dirichlet_bc(A: spy_sprs.csr_matrix,
                        b: np.ndarray,
                        bc: BC,
                        ) -> tuple[spy_sprs.csr_matrix, np.ndarray]:
  """Imposes Dirichlet boundary conditions on the system of equations.

  Modifies the right hand side to account for the Dirichlet boundary
  conditions.

  Args:
    A: The stiffness matrix of the system of shape (n,n).
    b: The right hand side of the system of shape (n,).
    bc: The boundary conditions.

  Returns: The modified stiffness matrix and right hand side.
  """
  if (bc.constraint_matrix is  None):
    free = bc.free_dofs
    # Modify the right hand side (nothing to move over for homogeneous Dirichlet values).
    b_modified = b - A[:, bc.fixed_dofs] @ bc.dirichlet_values if np.any(bc.dirichlet_values) else b

    # Remove rows and columns corresponding to the Dirichlet boundary conditions; the slices copy, A and b stay as given.
    return A[free, :][:, free], b_modified[free]
  else:
    raise NotImplementedError("impose_dirichlet_bc not implemented for constraint_matrix BCs; see linear_solvers.solver function")
  
def _dirichlet_pattern(bc: BC, idx, ndof: int, fixed):
  """(zero, diag) boolean masks over the stored entries idx of a coalesced matrix: entries in a fixed dof's row or
  column, and the fixed dofs' diagonal entries. Cached on bc for the pattern and fixed dofs it was built for; None
  when some fixed dof has no stored diagonal entry (the caller then adds the diagonal the general way)."""
  fixed_np = np.asarray(bc.fixed_dofs)
  nnz = int(idx.shape[1])
  sample = idx[:, ::max(1, nnz // 64)]                       # guards against a new pattern at a reused address
  key = (idx.data_ptr(), nnz, int(ndof), hash(fixed_np.tobytes()), hash(sample.cpu().numpy().tobytes()))
  cache = getattr(bc, "_dirichlet_cache", None)
  if cache is not None and cache[0] == key:
    return cache[1]
  rows, cols = idx[0], idx[1]
  zero = torch.isin(rows, fixed) | torch.isin(cols, fixed)
  diag = zero & (rows == cols)
  pattern = (zero, diag) if int(diag.sum()) == len(np.unique(fixed_np)) else None
  bc._dirichlet_cache = (key, pattern)
  return pattern


def _dirichlet_rhs(idx, val, f, bc: BC):
  """f - K u_pres with the fixed dofs' entries set to their prescribed values, for K given by its stored entries
  (idx, val). K @ u_pres is written as an explicit scatter over K's stored nonzeros rather than torch.sparse.mm:
  the backward of sparse.mm w.r.t. K (SparseAddmmBackward) materializes a dense (ndof x ndof) gradient -- ~48 GB at
  77k dof -- and OOM-killed large problems (LBracketThick, ThreeHoleBracket, GEGrabCAD) inside torch.autograd.grad."""
  device, dtype = val.device, val.dtype
  fixed = torch.from_numpy(bc.fixed_dofs).to(device=device, dtype=torch.long)
  dir_vals = torch.from_numpy(bc.dirichlet_values).to(device=device, dtype=dtype)
  u_pres = torch.zeros(f.shape[0], device=device, dtype=dtype)
  u_pres[fixed] = dir_vals
  reaction = torch.zeros(f.shape[0], device=device, dtype=dtype).index_add(0, idx[0], val * u_pres[idx[1]])
  f_mod = f - reaction
  f_mod[fixed] = dir_vals
  return f_mod


def _dirichlet_values(idx, val, bc: BC):
  """The stored values of the Dirichlet-modified matrix (fixed rows and columns zeroed, their diagonal 1) for a
  coalesced matrix with entries (idx, val), or None when some fixed dof has no stored diagonal entry.

  The sparsity pattern and the fixed dofs do not change between solves of one problem: which stored entries to
  zero and which are the fixed dofs' diagonal entries is found once and cached on bc. Each solve then only
  replaces values, and the pattern keeps its (coalesced) order, so nothing is sorted again."""
  fixed = torch.from_numpy(bc.fixed_dofs).to(device=val.device, dtype=torch.long)
  pattern = _dirichlet_pattern(bc, idx, bc.num_dofs, fixed)
  if pattern is None:
    return None
  zero, diag = pattern
  val = torch.where(zero, torch.zeros((), device=val.device, dtype=val.dtype), val)
  return torch.where(diag, torch.ones((), device=val.device, dtype=val.dtype), val)


def apply_dirichlet_bc_values(idx, val, f, bc: BC):
  """(val_mod, f_mod) of apply_dirichlet_bc_torch on the stored values of a coalesced matrix (indices idx), so the
  solve can take values and no sparse tensor is in the autograd graph; None when some fixed dof has no stored
  diagonal entry (use apply_dirichlet_bc_torch then)."""
  val_mod = _dirichlet_values(idx, val, bc)
  return None if val_mod is None else (val_mod, _dirichlet_rhs(idx, val, f, bc))


def apply_dirichlet_bc_torch(K, f, bc: BC):
  """Imposes Dirichlet boundary conditions on a torch-based linear system.

  Modifies the right-hand side to account for prescribed displacements and
  enforces them in the stiffness matrix using a penalty-free replacement
  (rows/cols zeroed, diagonal set to 1).

  Args:
    K: Stiffness matrix of the system, as a torch sparse COO tensor of shape (n, n).
    f: Right hand side vector of the system, as a dense torch tensor of shape (n,).
    bc: Boundary conditions object of type BC, with fields:
        - fixed_dofs: numpy array of fixed dof indices.
        - dirichlet_values: numpy array of prescribed displacement values.

  Returns:
    A tuple (K_mod, f_mod) where:
      K_mod: Modified stiffness matrix as a torch sparse COO tensor of shape (n, n).
      f_mod: Modified right hand side as a dense torch tensor of shape (n,).
  """

  device, dtype = K.device, K.dtype

  # coalesce once (a no-op when the assembly flags K as coalesced, as the hex FE solvers do)
  Kc = K.coalesce()
  idx = Kc.indices()
  val = Kc.values()

  f_mod = _dirichlet_rhs(idx, val, f, bc)
  val_mod = _dirichlet_values(idx, val, bc)
  if val_mod is not None:
    return torch.sparse_coo_tensor(idx, val_mod, Kc.shape, device=device, dtype=dtype, is_coalesced=True), f_mod

  fixed = torch.from_numpy(bc.fixed_dofs).to(device=device, dtype=torch.long)
  # zero rows and cols corresponding to fixed dofs
  rows, cols = idx[0], idx[1]
  mask = torch.isin(rows, fixed) | torch.isin(cols, fixed)
  val = torch.where(mask, torch.tensor(0.0, device=device, dtype=dtype), val)

  K_mod = torch.sparse_coo_tensor(idx, val, Kc.shape, device=device, dtype=dtype)

  # add identity on fixed dofs (K_ii = 1)
  diag_idx = torch.stack([fixed, fixed], dim=0)
  diag_val = torch.ones_like(fixed, dtype=dtype)
  K_diag = torch.sparse_coo_tensor(diag_idx, diag_val, Kc.shape, device=device, dtype=dtype)

  return (K_mod + K_diag).coalesce(), f_mod


def elem_body_force_from_acceleration(mesh, mass_density: float, acceleration) -> np.ndarray | None:
  """Element body force (3*num_elems,) for a uniform acceleration field (gravity, linear acceleration).

  Same convention as the benchmark problems (e.g. createGravityPlateProblem): each element carries
  mass_density * acceleration * element volume, and HexStructuralFEA.solve scales it by the element density x.
  Returns None when the acceleration is zero, so callers can pass it straight to HexStructuralFEA.
  """
  a = np.asarray(acceleration, dtype=float).reshape(3)
  if not np.any(a):
    return None
  force_per_elem = mass_density * np.prod(mesh.elem_size) * a
  return np.tile(force_per_elem, mesh.num_elems)
