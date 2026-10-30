"""Thermal Finite Element Analysis."""

import time
import numpy as np
import pyto.solve.numpy_backend as lin_sol
import pyto.physics.hex_element_stiffness as elem_stiff
import pyto.core.mat_lib as mat_lib
import pyto.core.bc as bound_cond
from pyto.core.bc import apply_dirichlet_bc_torch, apply_dirichlet_bc_values
import os
import pyvista as pv
import scipy.sparse as sp
import pyto.solve.deflation as deflation
import pyto.autodiff.sparse_solve as torch_spsolve
from pyto.autodiff.sparse_solve import solve as sparse_spsolve, solve_values
from pyto.autodiff.material_model import *
from pyto.examples_benchmarks.hex_thermal_examples import HexThermalExamples
from pyto.gui.hex_plotter import HexFEAPlotter

script_dir = os.path.dirname(os.path.abspath(__file__))

class HexThermalFEA:
  """Linear Thermal Finite Element Analysis using Hex8 elements."""

  def __init__(self,
							 mesh,
							 mat_prop: mat_lib.Material,
							 bc: bound_cond.BC,
							 solver: torch_spsolve.Solvers,
               dsolver: deflation.DeflationSolver = None,
                elem_body_force: np.ndarray = None,
                thermoElasticReferenceTemperature = 23.0,
                conductivity_penalty: float = None,
                conductivity_void_ratio: float = None,
							 **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.thermoElasticReferenceTemperature = thermoElasticReferenceTemperature  # Reference temperature for thermal expansion
    # Per-problem conductivity interpolation (None = the module defaults in autodiff/material_model.py)
    self.conductivity_penalty, self.conductivity_void_ratio = conductivity_penalty, conductivity_void_ratio
    self.solver, self.kwargs = solver, kwargs
    self.dsolver = dsolver
    self.sol = None
    # Handle single material or list of materials
    if isinstance(mat_prop, list):
      # Create element stiffness matrix for each material
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_thermal(mp.thermal_conductivity, mesh.elem_size)
                         for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)  # shape: (num_materials, 8, 8)
    else:
      self.elem_stiff = np.expand_dims(
        elem_stiff.hex8_stiffness_matrix_thermal(mat_prop.thermal_conductivity, mesh.elem_size), axis=0)
    self.elem_stiff_torch = torch.tensor(self.elem_stiff, dtype=torch.float64)

    # mesh.edofMat is the generic name the shared torch QOI code
    # (pyto.autodiff.qoi.*) reads regardless of physics -- populate it here
    # from the thermal-specific edofMatThermal rather than requiring every
    # caller to remember `mesh.edofMat = mesh.edofMatThermal` themselves
    # (previously only done by tests/conftest.py and PyTOGUI.py; see the
    # matching fix in HexStructuralFEA.__init__).
    if not hasattr(mesh, "edofMatThermal"):
      mesh.createEdofMatThermal()
    mesh.edofMat = mesh.edofMatThermal

    self.node_idx = np.stack((
                      np.kron(self.mesh.edofMatThermal, np.ones((8, 1))).flatten(),
                      np.kron(self.mesh.edofMatThermal, np.ones((1, 8))).flatten())
                      ).T.astype(int)

    # Precomputed sparse-assembly indices (constant across solves), same
    # dedup-and-scatter pattern as HexStructuralFEA.__init__.
    ndof = bc.num_dofs
    node_idx_np = self.node_idx.astype(np.int64)
    rows_np = node_idx_np[:, 0]
    cols_np = node_idx_np[:, 1]
    keys = rows_np * ndof + cols_np

    uniq_keys, inv = np.unique(keys, return_inverse=True)
    uniq_rows = (uniq_keys // ndof).astype(np.int64)
    uniq_cols = (uniq_keys %  ndof).astype(np.int64)

    self._unique_indices  = torch.from_numpy(np.stack([uniq_rows, uniq_cols], axis=0))
    self._scatter_inverse = torch.from_numpy(inv.astype(np.int64))
    self._n_unique        = self._unique_indices.shape[1]

    self.elem_body_force = elem_body_force
    self._bc_force_torch = torch.tensor(self.bc.force, dtype=torch.float64)

    self.plotter = HexFEAPlotter(mesh)


##################################################################
  def set_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
    """
    Set or update the thermal material(s) and recompute element stiffness matrices.
    Args:
      mat_prop: Single Material or list of Materials.
    """
    self.mat_prop = mat_prop
    if isinstance(mat_prop, list):
      elem_stiff_list = [elem_stiff.hex8_stiffness_matrix_thermal(mp.thermal_conductivity, self.mesh.elem_size)
                         for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)

    else:
      self.elem_stiff = np.expand_dims(
        elem_stiff.hex8_stiffness_matrix_thermal(mat_prop.thermal_conductivity, self.mesh.elem_size), axis=0)   
#################################################################
  def _solver_options(self):
    """Options for the differentiable solve: the deflation solver and tolerance for DPCG/PCG."""
    return {"dsolver": getattr(self, "dsolver", None), "rtol": self.kwargs.get("rtol", 1e-8),
            "symmetric": True}       # K is symmetric (Dirichlet rows and columns are replaced symmetrically)

  def adjoint_solve(self, rhs):
    """Solve K(x) lam = rhs (numpy) with the matrix and solver of the LAST solve: the same factors
    (PARDISO, SPSOLVE) or the same preconditioner and deflation space (DPCG/PCG), no new factorization.
    Fixed dofs are Dirichlet (identity rows/columns), so lam is zero there. For analytical sensitivities."""
    factorization = getattr(self, "_last_solve", {}).get("factorization")
    if factorization is None:
      raise RuntimeError("adjoint_solve needs a previous solve() of this solver.")
    rhs = np.array(rhs, dtype=float, copy=True)
    rhs[np.asarray(self.bc.fixed_dofs, dtype=int)] = 0.0
    return np.asarray(factorization.solve_transpose(rhs))

  def solve(self, x: torch.Tensor = None, material_model: MaterialModel = None, elem_mat_id: np.ndarray = None) -> torch.Tensor:
    """Solve the thermal finite element problem.

    Torch-differentiable, mirroring HexStructuralFEA.solve() -- x must be a
    torch tensor (grad-tracked when used inside an autodiff optimization
    chain). Multi-material problems use only the first material's element
    stiffness (elem_stiff_torch broadcasts against elem_material_scaling),
    same simplification already made by the structural solve and by the
    optimization drivers (which use KE_list[0] for multi-material meshes).

    Args:
       x: Torch tensor of (num_elems,) of the material scaling.
      This is used in SIMP topology optimization

    Returns: Torch tensor of (num_dofs,) of the solution to the finite element problem.
    """
    # Drop the previous solve's results first: on the AD path they hold its graph and factorization (see
    # HexStructuralFEA.solve).
    self.sol = self.stiff_mtrx = None
    self._last_solve = {}

    if x is None:
      x = torch.ones(self.mesh.num_elems, dtype=torch.float64)

    self.x = x
    device, dtype = x.device, x.dtype
    ndof = self.bc.num_dofs

    elem_material_scaling = get_thermal_material_model_scaling_torch(
        x, material_model, self.conductivity_penalty, self.conductivity_void_ratio)

    elem_stiff_mtrx = torch.einsum("mij,m->mij", self.elem_stiff_torch, elem_material_scaling)
    vals = elem_stiff_mtrx.reshape(-1)

    agg_vals = torch.zeros(self._n_unique, dtype=dtype, device=device)
    agg_vals.index_add_(0, self._scatter_inverse, vals)

    self.stiff_mtrx = torch.sparse_coo_tensor(       # unique, sorted indices (np.unique): already coalesced
        self._unique_indices, agg_vals, (ndof, ndof), device=device, dtype=dtype, is_coalesced=True
    )

    f = self._bc_force_torch

    # Dirichlet BCs and solve on the stored values when the pattern allows (see HexStructuralFEA.solve)
    self._last_solve = {}
    bc_values = apply_dirichlet_bc_values(self._unique_indices, agg_vals, f, self.bc)
    if bc_values is not None:
      K_vals, f_bc = bc_values
      sol = solve_values(self._unique_indices, K_vals, (ndof, ndof), f_bc, solver=self.solver,
                         solver_options=self._solver_options(), keep=self._last_solve)
    else:
      K_bc, f_bc = apply_dirichlet_bc_torch(self.stiff_mtrx, f, self.bc)
      sol = sparse_spsolve(K_bc, f_bc, solver=self.solver, solver_options=self._solver_options(),
                           keep=self._last_solve)

    self.sol = sol
    return sol
  #################################################################
  def getHMatrix(self, dx, dy, dz, nu):
      """
      Compute H matrix for thermo-elastic coupling with constitutive matrix D included.
      
      This matrix relates nodal temperatures to thermal forces:
          f_thermal =  H @ (T_nodes - T_ref)
      
      
      Parameters:
      -----------
      
      dx, dy, dz : float
          Element dimensions (m)
      nu : float
          Poisson's ratio
      
      Returns:
      --------
      H : ndarray (24, 8)
          Thermo-elastic coupling matrix; rows are node-major (3*j + component), like edofMat.
          
      Notes:
      ------
      This function was converted from MATLAB symbolic output.
      Generated by Symbolic Math Toolbox on 27-Nov-2025 14:09:59
      """
      # Intermediate calculations
      t2 = nu * 2.0
      t3 = t2 - 1.0
      t4 = 1.0 / t3
      t5 = ( dx * dy * t4) / 18.0
      t6 = ( dx * dy * t4) / 36.0
      t7 = ( dx * dz * t4) / 18.0
      t8 = ( dx * dz * t4) / 36.0
      t9 = ( dy * dz * t4) / 18.0
      t10 = ( dy * dz * t4) / 36.0
      t13 = (dx * dy * t4) / 72.0
      t16 = (dx * dz * t4) / 72.0
      t19 = (dy * dz * t4) / 72.0
      
      # Negative terms
      t11 = -t5
      t12 = -t6
      t14 = -t7
      t15 = -t8
      t17 = -t9
      t18 = -t10
      t20 = -t13
      t21 = -t16
      t22 = -t19
      
      # Assemble H matrix (24x8) using column-major order from MATLAB
      H_flat = np.array([
          t9, t17, t18, t10, t10, t18, t22, t19,
          t7, t8, t15, t14, t8, t16, t21, t15,
          t5, t6, t13, t6, t11, t12, t20, t12,
          t9, t17, t18, t10, t10, t18, t22, t19,
          t8, t7, t14, t15, t16, t8, t15, t21,
          t6, t5, t6, t13, t12, t11, t12, t20,
          t10, t18, t17, t9, t19, t22, t18, t10,
          t8, t7, t14, t15, t16, t8, t15, t21,
          t13, t6, t5, t6, t20, t12, t11, t12,
          t10, t18, t17, t9, t19, t22, t18, t10,
          t7, t8, t15, t14, t8, t16, t21, t15,
          t6, t13, t6, t5, t12, t20, t12, t11,
          t10, t18, t22, t19, t9, t17, t18, t10,
          t8, t16, t21, t15, t7, t8, t15, t14,
          t5, t6, t13, t6, t11, t12, t20, t12,
          t10, t18, t22, t19, t9, t17, t18, t10,
          t16, t8, t15, t21, t8, t7, t14, t15,
          t6, t5, t6, t13, t12, t11, t12, t20,
          t19, t22, t18, t10, t10, t18, t17, t9,
          t16, t8, t15, t21, t8, t7, t14, t15,
          t13, t6, t5, t6, t20, t12, t11, t12,
          t19, t22, t18, t10, t10, t18, t17, t9,
          t8, t16, t21, t15, t7, t8, t15, t14,
          t6, t13, t6, t5, t12, t20, t12, t11
      ])
      
      # Reshape to 24x8 using Fortran (column-major) order to match MATLAB
      H = H_flat.reshape((24, 8), order='F')

      # The symbolic output orders the 24 rows COMPONENT-major (all x-forces of nodes 1..8, then all y, then
      # all z). Every other element quantity in PyTO (edofMat, the element stiffness, the displacement vector
      # sol[edof]) is NODE-major (3*j + c), so reorder the rows. Without this the thermal force was scrambled
      # (a freely expanding body did not expand freely; see tests/test_thermoelastic.py).
      H = H.reshape(3, 8, 8).transpose(1, 0, 2).reshape(24, 8)

      return H
  #################################################################
  def get_thermoelastic_force_torch(self, T: torch.Tensor, x: torch.Tensor,
                                    material_model: MaterialModel = None) -> torch.Tensor:
    """Differentiable nodal thermal force F_th(x, T), shape (3 * num_nodes,).

    Torch twin of get_thermoelastic_force (same numbers): per element
        f_e = s_beta(x_e) * E * alpha * H (T_e - T_ref),
    where s_beta is the structural stiffness scaling (the thermal stress coefficient beta = E*alpha is
    penalized like E, Rodrigues & Fernandes 1995; Ooms et al.), and H = getHMatrix(nu) [24 x 8].
    getHMatrix is proportional to 1/(2*nu - 1), so a single reference matrix (nu = 0) serves every
    element and material: H(nu) = H(0) / (1 - 2*nu).

    Autograd flows through both the temperatures T (so through the thermal solve) and the design x.
    """
    dx, dy, dz = self.mesh.elem_size
    elem_nodes = torch.as_tensor(self.mesh.elemArray[:, :8], dtype=torch.long)          # (E, 8)
    dofs = (3 * elem_nodes.unsqueeze(-1) + torch.arange(3)).reshape(elem_nodes.shape[0], 24)  # node-major, like H
    H0 = torch.as_tensor(self.getHMatrix(dx, dy, dz, 0.0), dtype=T.dtype)                 # (24, 8)

    n_elem = elem_nodes.shape[0]
    if isinstance(self.mat_prop, list):
      E = torch.tensor([mp.youngs_modulus for mp in self.mat_prop], dtype=T.dtype)
      alpha = torch.tensor([mp.thermal_expansion_coefficient for mp in self.mat_prop], dtype=T.dtype)
      nu = torch.tensor([mp.poissons_ratio for mp in self.mat_prop], dtype=T.dtype)
    else:
      E, alpha, nu = (torch.full((n_elem,), v, dtype=T.dtype) for v in (
          self.mat_prop.youngs_modulus, self.mat_prop.thermal_expansion_coefficient, self.mat_prop.poissons_ratio))
    scale = get_structural_material_model_scaling_torch(x, material_model) * E * alpha / (1.0 - 2.0 * nu)  # (E,)

    dT = T[elem_nodes] - self.thermoElasticReferenceTemperature                            # (E, 8)
    f_elem = scale.unsqueeze(1) * (dT @ H0.T)                                              # (E, 24)
    return torch.zeros(3 * self.mesh.num_nodes, dtype=T.dtype).index_add(0, dofs.reshape(-1), f_elem.reshape(-1))

  def get_element_thermal_strain(self, T: torch.Tensor) -> np.ndarray:
    """Free thermal strain of each element, (E, 6) Voigt: alpha * (T_mean_e - T_ref) * [1, 1, 1, 0, 0, 0].

    Used to remove the thermal part from the total strain when computing stresses."""
    elem_nodes = self.mesh.elemArray[:, :8]
    T_np = T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T)
    dT = T_np[elem_nodes].mean(axis=1) - self.thermoElasticReferenceTemperature
    if isinstance(self.mat_prop, list):
      alpha = np.array([mp.thermal_expansion_coefficient for mp in self.mat_prop])
    else:
      alpha = self.mat_prop.thermal_expansion_coefficient
    strain = np.zeros((elem_nodes.shape[0], 6))
    strain[:, :3] = (alpha * dT)[:, None]
    return strain

  def get_thermoelastic_force(self, x: np.ndarray = None, material_model: MaterialModel = None) -> np.ndarray:
    """
    Add thermal forces to the finite element system.
    Supports multi-material: uses per-element E, alpha, nu.
    """
    if self.sol is None:
        raise ValueError("Solution not computed yet. Call solve() before get_thermal_force().")

    if x is None:
        x = np.ones((self.mesh.num_elems,))

    self.x = x  # store for postprocessing
    elem_material_scaling = get_structural_material_model_scaling(x, material_model)
    dx, dy, dz = self.mesh.elem_size

    # self.sol is a torch tensor since solve() became torch-native; this
    # method's own math (elem_material_scaling, HMatrix, f_thermoelastic)
    # is deliberately plain NumPy, so convert once at the boundary.
    sol_np = self.sol.detach().cpu().numpy() if hasattr(self.sol, "detach") else self.sol

    f_thermoelastic = np.zeros(3 * self.mesh.num_nodes)  # Elastic force vector

    # Multi-material support
    if isinstance(self.mat_prop, list):
        for elem in range(self.mesh.num_elems):
            mp = self.mat_prop[elem]
            E = mp.youngs_modulus
            alpha = mp.thermal_expansion_coefficient
            nu = mp.poissons_ratio
            HMatrix = self.getHMatrix(dx, dy, dz, nu)
            elem_nodes = self.mesh.elemArray[elem]
            node_temp = sol_np[elem_nodes]
            f_thermal_elem = elem_material_scaling[elem] * E * alpha * HMatrix @ (node_temp - self.thermoElasticReferenceTemperature)
            for j in range(8):
                f_thermoelastic[3 * elem_nodes[j]] += f_thermal_elem[3 * j]
                f_thermoelastic[3 * elem_nodes[j] + 1] += f_thermal_elem[3 * j + 1]
                f_thermoelastic[3 * elem_nodes[j] + 2] += f_thermal_elem[3 * j + 2]
    else:
        E = self.mat_prop.youngs_modulus
        alpha = self.mat_prop.thermal_expansion_coefficient
        nu = self.mat_prop.poissons_ratio
        HMatrix = self.getHMatrix(dx, dy, dz, nu)
        for elem in range(self.mesh.num_elems):
            elem_nodes = self.mesh.elemArray[elem]
            node_temp = sol_np[elem_nodes]
            f_thermal_elem = elem_material_scaling[elem] * E * alpha * HMatrix @ (node_temp - self.thermoElasticReferenceTemperature)
            for j in range(8):
                f_thermoelastic[3 * elem_nodes[j]] += f_thermal_elem[3 * j]
                f_thermoelastic[3 * elem_nodes[j] + 1] += f_thermal_elem[3 * j + 1]
                f_thermoelastic[3 * elem_nodes[j] + 2] += f_thermal_elem[3 * j + 2]
    return f_thermoelastic


  #################################################################
  def postprocess(self):
      """Computes the stresses at the center of each element.

      Args:
          u: Displacement field (num_dofs,).

      Returns:
          Array of (num_elems, 6) of the stresses at the center of each element.
          The order of the stress components is:
          sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy
      """
   
      gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
      ])

      # self.sol is a torch tensor since solve() became torch-native.
      sol_np = self.sol.detach().cpu().numpy() if hasattr(self.sol, "detach") else self.sol

      # Get element degrees of freedom
      edof = self.mesh.edofMatThermal

      # Compute displacement gradients
      self.strain = gradN @ sol_np[edof].T

#################################################################
  def plot_elem_field(self,
            elem_field,
            mask_low_pseudodensity = True,
            title = '',
            save_path=None,
            colormap = 'jet',
            auto_close = True,
            fontsize=10,
            cross_section=None,
            show_geometry=False,
            plotter = None,
            annotate_max_min = False,
            colors=None):  # New parameter for custom colors
  

    # Forward the caller's arguments (this wrapper used to pass fixed defaults, so plotter/save_path/title were
    # silently ignored and a GUI call opened a separate window).
    return self.plotter.plot_elem_field(elem_field,
            mask_low_pseudodensity = mask_low_pseudodensity,
            title = title,
            save_path=save_path,
            colormap = colormap,
            auto_close = auto_close,
            fontsize=fontsize,
            cross_section=cross_section,
            show_geometry=show_geometry,
            plotter = plotter,
            annotate_max_min = annotate_max_min)
    
  #################################################################
  def plot_temperature(self,auto_close = True,plotter=None, save_path=None, annotate_max_min = False):
    # Return if no solution exists yet (__init__ sets sol = None, so hasattr alone is always true)
    if getattr(self, 'sol', None) is None:
      return None
    # sol is a torch tensor (grad-tracked after an optimization); pyvista needs a plain NumPy array.
    sol_np = self.sol.detach().cpu().numpy() if hasattr(self.sol, "detach") else np.asarray(self.sol)
    return self.plotter.plot_temperature(sol_np, auto_close = auto_close, plotter=plotter,
                                       save_path=save_path, annotate_max_min = annotate_max_min)

#################################################################
  def plot_mesh(self, title = None,plot_bc = True,auto_close = True, save_path=None, camera_position = None, plotter=None):
    
    return self.plotter.plot_mesh_thermal(self.bc, title = title,plot_bc = plot_bc,
                auto_close = auto_close, save_path=save_path, camera_position = camera_position, plotter=plotter)

#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            plotter = None,
            fontsize=10):

    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, plotter = plotter, fontsize=fontsize)
    
  def plot_pseudo_density_realtime(self, title='Pseudo density', iteration=0, external_plotter=None):
    """
    Real-time visualization with proper Qt event loop handling.
    
    Args:
        title: Title for the visualization
        iteration: Current iteration number
        external_plotter: Optional external plotter (e.g., from GUI)
    
    Notes:
        This method delegates to the HexFEAPlotter, which handles the actual visualization.
        When external_plotter is provided (GUI mode), it will use that plotter instead of
        creating a new BackgroundPlotter window.
    """
    return self.plotter.plot_pseudo_density_realtime(
        title=title, 
        iteration=iteration,
        external_plotter=external_plotter
    )


#################################################################
if __name__ == "__main__":
    import pyto.physics.thermal.hex_thermal_fea as fea
    import pyto.solve.numpy_backend as lin_solv
    import time	
    from pyto.examples_benchmarks.hex_thermal_examples import *

    problem = HexThermalExamples.LBracket
    solver = lin_solv.Solvers.PARDISO
    mesh, mat_prop, bc, elem_body_force = getThermalProblem(problem)
    
    fe_solver = HexThermalFEA(mesh=mesh,
                  mat_prop=mat_prop,
                  bc=bc,
                  solver=solver)
    
    startTime = time.time()
    T = fe_solver.solve()
    TMax = np.max(np.abs(T ))


    print(f'DOF: {fe_solver.mesh.num_nodes}, Max u: {TMax:.3e}')
    
    nDOF = fe_solver.mesh.num_nodes
    print('-----------------------------')
    print("nDof: ", nDOF)
    print('Solver: ', fe_solver.solver.name)
    print("FEA time: ", time.time() - startTime)
    print('Max Temp: ', TMax)
    print('-----------------------------')
	
    fe_solver.plot_temperature()
 
    
   
