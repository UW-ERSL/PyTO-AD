"""Structural Finite Element Analysis."""

from pyto.autodiff.material_model import *
from pyto.autodiff.material_model import _EVOID_RELATIVE, _PNORM_EXPONENT
import numpy as np
import os
import time
import pyvista as pv
import pyto.core.mat_lib as mat_lib
import pyto.core.bc as bound_cond
from pyto.core.bc import apply_dirichlet_bc_torch
import pyto.physics.hex_element_stiffness as hex_element_stiffness
import pyto.solve.deflation as deflation
import pyto.autodiff.sparse_solve as torch_spsolve
import pyto.solve.numpy_backend as linear_solvers
from pyto.autodiff.sparse_solve import solve as sparse_spsolve
from pyto.gui.hex_plotter import HexFEAPlotter


script_dir = os.path.dirname(os.path.abspath(__file__))

class HexStructuralFEA:
  """Linear Structural Finite Element Analysis."""

  def __init__(self,
         mesh,
         mat_prop: mat_lib.Material | list[mat_lib.Material],
         bc: bound_cond.BC,
         solver: torch_spsolve.Solvers,
         dsolver: deflation.DeflationSolver = None,
         elem_body_force: np.ndarray = None,
         thermo_elastic_force = None,
         **kwargs):

    self.mesh, self.mat_prop, self.bc = mesh, mat_prop, bc
    self.thermo_elastic_force = thermo_elastic_force
    self.elem_thermal_strain = None  # (E, 6) free thermal strain of the last solve, for stress post-processing
    self.solver, self.kwargs = solver, kwargs
    self.dsolver = dsolver

    bound_cond.resolve_constraint_matrix(bc)  # single-DOF constraint rows become Dirichlet; sliding rows raise
    well_posed, worst_relative_component = bound_cond.check_rigid_body_constraint(mesh, bc)
    if not well_posed:
      raise ValueError(
          f"BC is under-constrained: at least one rigid-body mode (translation "
          f"or rotation) is left free by fixed_dofs, and the applied force has a "
          f"significant component along it (relative magnitude "
          f"{worst_relative_component:.3g}, vs. a well-posed problem's ~1e-6 or "
          f"less). The solve would still 'succeed' but produce a numerically "
          f"meaningless displacement/compliance (e.g. negative compliance). Fix "
          f"the boundary condition's fixed_dofs so every rigid-body mode is "
          f"constrained, or make sure the load has no component along any mode "
          f"that is intentionally left free.")

    # Handle single material or list of materials
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(mp.youngs_modulus, mp.poissons_ratio, mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
          hex_element_stiffness.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus,mat_prop.poissons_ratio, mesh.elem_size), axis=0)
    self.elem_stiff_torch = torch.tensor(self.elem_stiff, dtype=torch.float64)

    # mesh.edofMat is the generic name the shared torch QOI code
    # (pyto.autodiff.qoi.*) reads regardless of physics -- populate it here
    # from the structural-specific edofMatStructural rather than requiring
    # every caller to remember `mesh.edofMat = mesh.edofMatStructural`
    # themselves (previously only done by tests/conftest.py and
    # PyTOGUI.py; every other caller, e.g. topopt_run_benchmarks.py, hit
    # an AttributeError here).
    if not hasattr(mesh, "edofMatStructural"):
      mesh.createEdofMatStructural()
    mesh.edofMat = mesh.edofMatStructural

    self.node_idx = np.stack((
            np.kron(self.mesh.edofMat, np.ones((24, 1))).flatten(),
            np.kron(self.mesh.edofMat, np.ones((1, 24))).flatten())
            ).T.astype(int)
    
    ndof = bc.num_dofs
    node_idx_np = self.node_idx.astype(np.int64)
    rows_np = node_idx_np[:, 0]
    cols_np = node_idx_np[:, 1]
    keys = rows_np * ndof + cols_np                         # linearize (row,col)

    # unique keys + inverse mapping back to each duplicate
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    uniq_rows = (uniq_keys // ndof).astype(np.int64)
    uniq_cols = (uniq_keys %  ndof).astype(np.int64)

    self._unique_indices   = torch.from_numpy(np.stack([uniq_rows, uniq_cols], axis=0))
    self._scatter_inverse  = torch.from_numpy(inv.astype(np.int64))
    self._n_unique             = self._unique_indices.shape[1]
    self.elem_body_force = elem_body_force

    self._bc_force_torch = torch.tensor(self.bc.force, dtype=torch.float64)

    if self.elem_body_force is not None:
        self._elem_body_force_torch = torch.tensor(self.elem_body_force, dtype=torch.float64)
        m = self.mesh.elem_to_node_field_mapping.tocoo()  # scipy (num_nodes, num_elems); dense would be huge
        self._elem_to_node_map_torch = torch.sparse_coo_tensor(
            np.vstack([m.row, m.col]), m.data, m.shape, dtype=torch.float64).coalesce()
    else:
        self._elem_body_force_torch = None
        self._elem_to_node_map_torch = None
    self._spring_K = None
    self._spring_pos = None
    if getattr(self.mesh, "externalSprings", None):
        ks, dofs = zip(*self.mesh.externalSprings)
        # a spring adds k to its dof's diagonal entry, which the pattern already stores: find those positions once,
        # so each solve adds the values in place and K keeps its (coalesced) pattern
        d = np.asarray(dofs, dtype=np.int64)
        pos = np.searchsorted(uniq_keys, d * ndof + d)
        if np.all(pos < len(uniq_keys)) and np.array_equal(uniq_keys[np.minimum(pos, len(uniq_keys) - 1)], d * ndof + d):
            self._spring_pos = (torch.from_numpy(pos), torch.tensor(ks, dtype=torch.float64))
        else:                                                   # not on the pattern's diagonal: add them as a matrix
            spring_idx  = torch.tensor([dofs, dofs], dtype=torch.long)
            spring_vals = torch.tensor(ks, dtype=torch.float64)
            self._spring_K = (spring_idx, spring_vals)  # store raw parts

    #default camera position
    view_distance = 2.5 * self.mesh.bbox.diag_length
    offset = 0.2 * view_distance  # Offset for object position
    self.camera_position =  [
                    (view_distance*0.5, -view_distance*0.3, view_distance),
                    (offset, offset, 0),   # Focus point - right and bottom
                    (0, 0.8, 0.4)]         # Up vector - Y axis up
    self.create_pyvista_plotter()
    self.plotter = HexFEAPlotter(mesh)

#################################################################
  def _solver_options(self):
    """Options for the differentiable solve: the deflation solver and tolerance for DPCG/PCG."""
    return {"dsolver": getattr(self, "dsolver", None), "rtol": self.kwargs.get("rtol", 1e-8)}

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

  def create_pyvista_plotter(self):
    self.pyVistaPlotter = pv.Plotter(window_size=(500, 400))
    self.pyVistaPlotter.camera_position =self.camera_position
    # Enable anti-aliasing for better quality
    self.pyVistaPlotter.enable_anti_aliasing()
    
#################################################################
  def set_structural_material(self, mat_prop: mat_lib.Material | list[mat_lib.Material]):
    
    if isinstance(mat_prop, list):
    # Create element stiffness matrix for each material
      elem_stiff_list = [hex_element_stiffness.hex8_stiffness_matrix_structural(mp.youngs_modulus,mp.poissons_ratio, self.mesh.elem_size) 
                for mp in mat_prop]
      self.elem_stiff = np.stack(elem_stiff_list)
    else:
      self.elem_stiff = np.expand_dims(
          hex_element_stiffness.hex8_stiffness_matrix_structural(mat_prop.youngs_modulus,mat_prop.poissons_ratio, self.mesh.elem_size), axis=0)

  #################################################################

  def solve(self, x=None, material_model: MaterialModel = None, extra_force=None, elem_thermal_strain=None):
      """Solve K(x) u = f.

      Args:
        x: (E,) design densities (torch). None means fully solid (a plain analysis), as in HexThermalFEA.solve.
        material_model: SIMP/RAMP/... (see pyto.autodiff.material_model).
        extra_force: optional (ndof,) torch tensor added to the load. Differentiable: this is how the
          design-dependent thermal load F_th(x, T) of a coupled thermo-mechanical problem enters
          (autograd flows through it).
        elem_thermal_strain: optional (E, 6) free thermal strain; only used by postprocess() so the
          stresses are computed from the elastic strain (total - thermal).
      """

      # Drop the previous solve's results first. On the AD path they reference its graph, whose solve node holds
      # that solve's factorization, so keeping them until the new solve finished held two factorizations at the
      # memory peak.
      self.sol = self.deformation = self.stiff_mtrx = self.total_force = self.elem_thermal_strain = None
      self._last_solve = {}

      if x is None:
          x = torch.ones(self.mesh.num_elems, dtype=torch.float64)
      self.x = x
      device, dtype = x.device, x.dtype
      ndof = self.bc.num_dofs

      # 0) material scaling
      elem_material_scaling = get_structural_material_model_scaling_torch(x, material_model)

      # 1) element stiffness scaling (einsum) -> vals
      elem_stiff_mtrx = torch.einsum("mij,m->mij", self.elem_stiff_torch, elem_material_scaling)
      vals = elem_stiff_mtrx.reshape(-1)

      # 2) build global COO (indices) + coalesce

      agg_vals   = torch.zeros(self._n_unique, dtype=dtype, device=device)
      agg_vals.index_add_(0, self._scatter_inverse, vals)         # sum duplicates once
      if self._spring_pos is not None:                            # external springs, on stored diagonal entries
          spring_pos, spring_k = self._spring_pos
          agg_vals = agg_vals.index_add(0, spring_pos.to(device), spring_k.to(device=device, dtype=dtype))

      # the index list is np.unique of row*ndof+col: unique and in coalesced order, so later coalesce() calls
      # (Dirichlet BC, solve) need not sort the ~25 entries per DOF again
      self.stiff_mtrx = torch.sparse_coo_tensor(
          self._unique_indices, agg_vals, (ndof, ndof), device=device, dtype=dtype, is_coalesced=True
      )
      # 3) external springs off the pattern's diagonal (none in the benchmarks), precomputed once in __init__
      if self._spring_K is not None:
          spring_idx, spring_vals = self._spring_K
          spring_K = torch.sparse_coo_tensor(
              spring_idx.to(device), spring_vals.to(device=device, dtype=dtype),
              (ndof, ndof), device=device, dtype=dtype
          ).coalesce()
          self.stiff_mtrx = (self.stiff_mtrx + spring_K).coalesce()


      # 4) assemble force vector (incl. body forces)
      f = self._bc_force_torch

      if self._elem_body_force_torch is not None:
          elem_force = self._elem_body_force_torch # (3E,)
          # body force is proportional to density x (not to the SIMP stiffness factor)
          elem_force = (elem_force.view(-1, 3) * x.view(-1, 1)).reshape(-1)
          map_vec = self._elem_to_node_map_torch
          fx, fy, fz = (torch.sparse.mm(map_vec, elem_force[i::3].unsqueeze(1)).squeeze(1) for i in range(3))
          node_forces = torch.stack([fx, fy, fz], dim=1).reshape(-1)
          f = f + node_forces
      if self.thermo_elastic_force is not None:  # static (non-differentiable) thermal load given at construction
          f = f + torch.as_tensor(self.thermo_elastic_force, dtype=f.dtype, device=f.device)
      if extra_force is not None:
          f = f + extra_force
      self.elem_thermal_strain = elem_thermal_strain
      self.total_force = f


      # 5) apply Dirichlet BCs
      K_bc, f_bc = apply_dirichlet_bc_torch(self.stiff_mtrx, f, self.bc)


      # 6) linear solve
      self._last_solve = {}
      u = sparse_spsolve(K_bc, f_bc, solver=self.solver, solver_options=self._solver_options(),
                         keep=self._last_solve)  # (ndof,)

      # 7) post (deformation magnitudes)
      self.sol = u
      self.deformation = torch.sqrt(u[0::3]**2 + u[1::3]**2 + u[2::3]**2)
      self.max_deformation = self.deformation.detach().max().item()


      return u



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
      sol_np = self.sol.detach().cpu().numpy()
      x_np = self.x.detach().cpu().numpy()
      gradN = (1 / 8) * np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1], 
        [-1, -1, -1, -1, 1, 1, 1, 1]
      ])
      for i in range(3):
        gradN[i, :] = 2*gradN[i,:] / self.mesh.elem_size[i]
      # Get element degrees of freedom
      edof = self.mesh.edofMat
      # Compute displacement gradients
      uGrad = gradN @ sol_np[edof[:, ::3]].T
      vGrad = gradN @ sol_np[edof[:, 1::3]].T
      wGrad = gradN @ sol_np[edof[:, 2::3]].T
      
      # Compute Engineering strains
      strain = np.stack([
        uGrad[0], vGrad[1], wGrad[2],
        uGrad[1] + vGrad[0],
        uGrad[2] + wGrad[0],
        vGrad[2] + wGrad[1]
      ], axis=1)  # Shape: (num_elems, 6)
      self.strainComponents = strain.copy() # store the total strain (elastic + thermal)
    
      if self.elem_thermal_strain is not None:  # stresses come from the ELASTIC strain only
        strain = strain - self.elem_thermal_strain

      # STRESS_RELAXATION method;
      q = 0.5  # SIMP like penalization for stress
      if isinstance(self.mat_prop, list):
        # Create D matrix for each material
        D_list = []
      
        for mp in self.mat_prop:
          E = mp.youngs_modulus
          nu = mp.poissons_ratio
          D = hex_element_stiffness.isotropic_constitutive_matrix ( E, nu)
          D_list.append(D)
        D_stack = np.stack(D_list)
        self.stressComponents =  np.einsum('eij, ej -> ei', D_stack, strain)
      else:
        E = self.mat_prop.youngs_modulus
        nu = self.mat_prop.poissons_ratio
        D = hex_element_stiffness.isotropic_constitutive_matrix ( E, nu)
        self.stressComponents = np.einsum('ij,ej->ei', D, strain)


      correction = (_EVOID_RELATIVE + (1-_EVOID_RELATIVE) * (x_np**q)).reshape((-1,1))
      eStress = correction * self.stressComponents
      self.vonMisesStress = np.sqrt(0.5*((eStress[:,0]-eStress[:,1])**2 +
                (eStress[:,1]-eStress[:,2])**2 +
                (eStress[:,2]-eStress[:,0])**2) +
                3*(eStress[:,3]**2 + eStress[:,4]**2 +
                   eStress[:,5]**2))
      
      self.pNormStress = (np.sum(self.vonMisesStress**_PNORM_EXPONENT))**(1/_PNORM_EXPONENT)  
     

      self.elemStrainEnergy = 0.5 * np.sum(strain * eStress, axis=1)  # Element-wise strain energy
      #print(f"Maximum von Mises stress: {np.max(self.vonMisesStress):.4e}")
      return 
#################################################################
  def plot_mesh(self, title = None,plot_bc = True,rel_arrow_scale = 0.5,
                auto_close = True, save_path=None,offsetArrow = False,transparency = 1.0, plotter=None,
                camera_position=None):
    
    self.pyVistaPlotter.clear()
    if (title is None):
      title = f'DOF: {3*self.mesh.num_nodes}'

    vertices = self.mesh.node_xyz
    # We only plot the boundary faces to save on memory
    faceIndex = np.array([[0,4,7,3],
                          [0,1,5,4],
                          [0,3,2,1],
                          [1,2,6,5],
                          [2,3,7,6],
                          [4,5,6,7]], dtype=np.uint32)
    nFacesPerHex = 6
    faces = []
    face_densities = []
    
    for e in range(self.mesh.num_elems):
      if self.mesh.elemPseudoDensity[e] < 0.5:
        continue
      elif (self.mesh.elemPseudoDensity[e] > 0.5 and 
            np.all(self.mesh.elemNeighborsArray[e] > 0) and 
            np.all(self.mesh.elemPseudoDensity[[int(elem) for elem in 
                                      self.mesh.elemNeighborsArray[e]]] > 0.5)):
        continue

      # Add all faces for this element
      for j in range(nFacesPerHex):
        faces.append(self.mesh.elemArray[e,faceIndex[j,:]])
        face_densities.append(self.mesh.elemPseudoDensity[e])

    # Convert to numpy arrays
    faces = np.array(faces)
    face_densities = np.array(face_densities)
    
    if len(faces) == 0:
      print("No faces to plot after filtering")
      return None

    # Create cells array for PyVista
    n_faces = len(faces)
    cells = np.hstack(( np.full((n_faces, 1), 4),  # 4 vertices per face
                      faces))

    pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities
    externalPlotter = False # assume that a plotter is provided
    if (plotter is None):
      externalPlotter = True
      # Create plotter
      if save_path is None:
        plotter = self.pyVistaPlotter
        if plotter.iren is None:
          self.create_pyvista_plotter()
          plotter = self.pyVistaPlotter 
          plotter.show(interactive_update=True, auto_close=False)
      else:
        plotter = pv.Plotter(off_screen=True) # for saving images
        plotter.enable_anti_aliasing()
      # Add coordinate axes widget
      plotter.add_axes(
                    xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                    line_width=2,
                    labels_off=False,  # Show axis labels
                    color='black'
                    )
    
      plotter.add_title(title, font_size=8)
  
    plotter.add_mesh(
            pv_mesh,
            color='lightgreen',
            show_edges=True,
            edge_color='black',
            line_width=1,
            opacity=transparency  # Add transparency (0.0 is fully transparent, 1.0 is fully opaque)
            )

    if (plot_bc):
      # Add dots and force arrows for labeled nodes
      point_size = 10.0  # Size of dots in pixels

      # Add black dots for label 1 (fixed nodes)
      label1_nodes = np.where(self.mesh.node_indices[:, 3] == 1)[0]
  
      if len(label1_nodes) > 0 and self.bc is not None:
        points1 = vertices[label1_nodes]
        pts = pv.PointSet(points1)
        plotter.add_mesh(pts, color='black')

      # Add force arrows for label 2 (without red dots)
      label2_nodes = np.where(self.mesh.node_indices[:, 3] == 2)[0]
      if len(label2_nodes) > 0  and self.bc is not None: #structural
        # Add force arrows
        
        force_norm_avg = np.linalg.norm(self.bc.force)
        for node in label2_nodes:
          # Get force components for this node
          fx = self.bc.force[3*node]
          fy = self.bc.force[3*node + 1]
          fz = self.bc.force[3*node + 2]
          force_vec = np.array([fx, fy, fz])
          
          force_nrm = np.linalg.norm(force_vec)
          # Only add arrow if force is non-zero
          if force_nrm > 0:
            arrow_scale = rel_arrow_scale * self.mesh.bbox.diag_length*force_nrm/force_norm_avg
            # Normalize and scale force vector
            force_vec_dir = force_vec / force_nrm
            
            # Create arrow
            start_point = vertices[node]
            if (offsetArrow):
              start_point = start_point -  force_vec_dir*arrow_scale  # Offset start point by arrow length so that it is visible
            # Add arrow to plot
            arrow = pv.Arrow(start = start_point,
                            direction = force_vec_dir,
                            scale = arrow_scale)
            plotter.add_mesh(arrow, color='red')
    
    # Save image if path is provided

    if save_path:
      # Set the camera position after all geometry is in the scene, not
      # before (as this used to do at plotter-creation time) -- a string
      # preset like 'xy'/'iso' fits the view to the renderer's current
      # data bounds, which are empty/default before add_mesh() runs. That
      # produced wildly inconsistent framing across problems (some
      # cropped/zoomed-in, some tiny/zoomed-out in the middle of a blank
      # frame) depending on each mesh's actual size relative to pyvista's
      # camera default, not the mesh itself.
      plotter.camera_position = camera_position if camera_position is not None else self.camera_position
      plotter.screenshot(save_path)
      plotter.close()
    else:
      if (externalPlotter):
        # Show plot interactively if not saving
        plotter.show(interactive_update=not auto_close, auto_close=auto_close)
      else:
        plotter.show()
    self.camera_position = plotter.camera_position # For all future displays
    return
################################################################# 
  def plot_deformation(self,show_geometry=False, auto_close = True, save_path=None, plotter=None):
    """Plot the deformed mesh with the deformation scaled by a factor."""
    # Return if no solution exists yet
    if not hasattr(self, 'sol'):
      return None

    # Create vertices array
    vertices = self.mesh.node_xyz.copy()
  
    sol = self.sol.detach().cpu().numpy().copy()
    sol = sol.reshape((-1, 3))
    delta = self.deformation.detach().cpu().numpy()
    # Scale and colour range from the nodes of the solid design (density > 0.5). In an optimized design the
    # near-void region moves far more than the structure; using all nodes made the structure look undeformed.
    density = np.asarray(self.mesh.elemPseudoDensity)
    solid = density > 0.5
    solid_nodes = np.unique(self.mesh.elemArray[solid][:, :8]) if solid.any() else np.arange(self.mesh.num_nodes)
    deltaMax = float(delta[solid_nodes].max())
    delta_clim = [float(delta[solid_nodes].min()), deltaMax]
    if deltaMax < 1e-16:
      deltaMax = 1e-16
    scale = float(0.1*self.mesh.bbox.diag_length/deltaMax)
    vertices += scale*sol


    # Match plotMeshOld exactly
    faceIndex = np.array([[0,4,7,3],
                          [0,1,5,4],
                          [0,3,2,1],
                          [1,2,6,5],
                          [2,3,7,6],
                          [4,5,6,7]], dtype=np.uint32)
    nFacesPerHex = 6
    faces = []
    face_densities = []
    
    for e in range(self.mesh.num_elems):
      if self.mesh.elemPseudoDensity[e] < 0.5:
        continue
      elif (self.mesh.elemPseudoDensity[e] > 0.5 and 
            np.all(self.mesh.elemNeighborsArray[e] > 0) and 
            np.all(self.mesh.elemPseudoDensity[[int(elem) for elem in 
                                      self.mesh.elemNeighborsArray[e]]] > 0.5)):
        continue

      # Add all faces for this element
      for j in range(nFacesPerHex):
        faces.append(self.mesh.elemArray[e,faceIndex[j,:]])
        face_densities.append(self.mesh.elemPseudoDensity[e])

    # Convert to numpy arrays
    faces = np.array(faces)
    face_densities = np.array(face_densities)
    
    if len(faces) == 0:
      print("No faces to plot after filtering")
      return None

    # Create cells array for PyVista
    n_faces = len(faces)
    cells = np.hstack((
                      np.full((n_faces, 1), 4),  # 4 vertices per face
                      faces
                      ))

    pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD

    # Add scalar values
    pv_mesh.point_data['values'] = delta
    
    # Add density values to cells
    pv_mesh.cell_data['density'] = face_densities

    externalPlotter = False # assume that a plotter is provided
    if plotter is None: 
      externalPlotter = True # create a new plotter
      # Create plotter
      if save_path is  None:
        plotter = self.pyVistaPlotter 
        if plotter.iren is None:
          self.create_pyvista_plotter()
          plotter = self.pyVistaPlotter 
          plotter.show(interactive_update=True, auto_close=False)
      else:
        plotter = pv.Plotter(off_screen=True)
      # Add coordinate axes widget
      plotter.add_axes(
                    xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                    line_width=2,
                    labels_off=False,  # Show axis labels
                    color='black'
                    )
   
    # Add mesh to plotter
    nDOF = 3*self.mesh.num_nodes
    plotter.add_mesh(
                    pv_mesh,
                    scalars='values' if delta is not None else 'density',
                    clim=delta_clim,
                    show_edges=True,
                    cmap='jet',
                    edge_color='black',
                    line_width=1,
                    scalar_bar_args={
                            'title': 'Deformation',
                            'vertical': True,
                            'position_x': 0.8,
                            'position_y': 0.3,
                            'width': 0.06
                            }
                  )
    if (show_geometry):
      vertices = self.mesh.stlGeom.mesh.vectors.reshape(-1, 3)
      faces = np.arange(len(vertices)).reshape(-1, 3)
      faces = np.column_stack((np.full(len(faces), 3), faces))
      geomMesh = pv.PolyData(vertices, faces)
      plotter.add_mesh(geomMesh, color='white', show_edges=True)

    # Save image if path is provided
    if save_path:
      #plotter.show(screenshot = save_path)
      plotter.screenshot(save_path)
      plotter.close()
    else:
      if (externalPlotter):
        plotter.show(interactive_update=not auto_close, auto_close=auto_close)
      else:
        plotter.show()
    self.camera_position = plotter.camera_position # For all future displays
    return 

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
            plotter = None,
            colors=None):  # New parameter for custom colors
    """
    Plot element field on the mesh, with optional discrete custom colors.

    Args:
        elem_field: Field values for each element (e.g., material indices)
        mask_low_pseudodensity: Whether to filter out elements with low density
        title: Plot title
        save_path: Path to save the image
        colormap: Color map for the field visualization (ignored if colors is provided)
        auto_close: Whether to auto-close the plotter
        fontsize: Font size for the title
        cross_section: Dict with keys 'axis' ('x', 'y', or 'z') and 'position' (float)
                      to create a cross-section view, or None for full view
        colors: List or array of hex codes or RGB tuples for each element (discrete coloring)
    """
    # Filter elements based on pseudo_density
    if mask_low_pseudodensity:
        mask = self.mesh.elemPseudoDensity > 0.5
        filtered_elems = self.mesh.elemArray[mask]
        filtered_field = np.array(elem_field)[mask]
        filtered_colors = None
        if colors is not None:
            filtered_colors = np.array(colors)[mask]
    else:
        filtered_elems = self.mesh.elemArray
        filtered_field = np.array(elem_field)
        filtered_colors = np.array(colors) if colors is not None else None

    if len(filtered_elems) == 0:
        print("No elements to plot after filtering")
        return None

    # Create vertices array
    vertices = self.mesh.node_xyz

    # Create cells array for PyVista
    cells = np.hstack((
        np.full((len(filtered_elems), 1), 8),  # 8 vertices per hexahedron
        filtered_elems
    ))

    # Create PyVista mesh
    pv_mesh = pv.UnstructuredGrid({12: cells[:, 1:]}, vertices)  # 12 is VTK_HEXAHEDRON

    # Add field data to cell data
    pv_mesh.cell_data['field'] = filtered_field

    externalPlotter = False
    if plotter is None:
        externalPlotter = True
        if save_path is None:
            plotter = self.pyVistaPlotter
            if plotter.iren is None:
                self.create_pyvista_plotter()
                plotter = self.pyVistaPlotter
        else:
            plotter = pv.Plotter(off_screen=True)
        plotter.add_axes(
            xlabel='X',
            ylabel='Y',
            zlabel='Z',
            line_width=2,
            labels_off=False,
            color='black'
        )

    # If cross-section is specified, create a clipped mesh
    if cross_section is not None:
        axis = cross_section.get('axis', 'x').lower()
        rel_position = cross_section.get('position', 0.0)
        bbox_min = pv_mesh.bounds[::2]
        bbox_max = pv_mesh.bounds[1::2]
        if axis == 'x':
            normal = (1, 0, 0)
            position = bbox_min[0] + (bbox_max[0] - bbox_min[0]) * (rel_position + 1) / 2
            origin = (position, 0, 0)
        elif axis == 'y':
            normal = (0, 1, 0)
            position = bbox_min[1] + (bbox_max[1] - bbox_min[1]) * (rel_position + 1) / 2
            origin = (0, position, 0)
        elif axis == 'z':
            normal = (0, 0, 1)
            position = bbox_min[2] + (bbox_max[2] - bbox_min[2]) * (rel_position + 1) / 2
            origin = (0, 0, position)
        else:
            print(f"Invalid axis '{axis}'. Using 'x' instead.")
            normal = (1, 0, 0)
            position = 0
            origin = (position, 0, 0)
        pv_mesh = pv_mesh.clip(normal=normal, origin=origin)
        if title:
            title += f" (Cross-section {axis}={position})"
        else:
            title = f"Cross-section {axis}={position}"

    # Add mesh to plotter
    if filtered_colors is not None:
        # Use custom discrete colors for each cell
        plotter.add_mesh(
            pv_mesh,
            scalars=filtered_colors,
            rgb=True,
            show_edges=True,
            edge_color='black',
            line_width=1,
            opacity=1.0,
            scalar_bar_args={
                'title': title,
                'vertical': True,
                'position_x': 0.85,   # Move to the right (0.0 = left, 1.0 = right)
                'position_y': 0.05,   # Move to the bottom (0.0 = bottom, 1.0 = top)
                'width': 0.08,        # Make it narrower
                'height': 0.9,        # Make it taller
                'title_font_size': 12,
                'label_font_size': 12
            }
        )
    else:
        plotter.add_mesh(
            pv_mesh,
            scalars='field',
            cmap=colormap,
            show_edges=True,
            edge_color='black',
            line_width=1,
            scalar_bar_args={
          'title': title,
          'vertical': True,
          'position_x': 0.85,   # Move to the right (0.0 = left, 1.0 = right)
          'position_y': 0.05,   # Move to the bottom (0.0 = bottom, 1.0 = top)
          'width': 0.08,        # Make it narrower
          'height': 0.9,        # Make it taller
          'title_font_size': 12,
          'label_font_size': 12
            }
        )

    # Save image if path is provided
    if save_path:
        # Get mesh bounds
        bounds = pv_mesh.bounds
        center = [(bounds[0] + bounds[1]) / 2,
                  (bounds[2] + bounds[3]) / 2,
                  (bounds[4] + bounds[5]) / 2]

        # Set camera above the mesh, looking down the z-axis
        distance = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 2
        camera_location = [center[0], center[1], center[2] + distance]
        focal_point = center
        view_up = [0, 1, 0]

        plotter.camera_position = [camera_location, focal_point, view_up]
        plotter.camera.zoom(1.3)
        plotter.screenshot(save_path)
        plotter.close()
    else:
        if externalPlotter:
            plotter.show(interactive_update=not auto_close, auto_close=auto_close)
        else:
            plotter.show()
    self.camera_position = plotter.camera_position
    return
#################################################################
  def plot_vonMisesStress(self,
            save_path=None,
            fontsize=8,plotter = None):
    
    self.plot_elem_field(self.vonMisesStress, title = f'vonMises stress ',
                          save_path=save_path, fontsize=fontsize,plotter = plotter)

#################################################################    
  def plot_strain_component(self,strainComponent = 0,
            save_path=None,
            fontsize=8):
    
    self.plot_elem_field(self.strainComponents[:,strainComponent], title = f'Strain: {strainComponent} ',
                          save_path=save_path,fontsize=fontsize)
#################################################################
  def plot_stress_component(self,stressComponent = 0,
            save_path=None,
            fontsize=10):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.stressComponents[:,stressComponent], title = f'Stress: {stressComponent} ',
                          save_path=save_path, fontsize=fontsize)
#################################################################
  def plot_pseudo_density(self,
            save_path=None,
            auto_close = True,
            title = 'Pseudo density',
            fontsize=10,
            plotter = None):
    self.pyVistaPlotter.clear()
    self.plot_elem_field(self.mesh.elemPseudoDensity, colormap='gray_r', auto_close = auto_close,
                         mask_low_pseudodensity=False, title= title,
                save_path=save_path, fontsize=fontsize,plotter = plotter)

  def plot_pseudo_density_realtime(self, title='Pseudo density', iteration=0, external_plotter=None):
    """
    Real-time visualization with proper Qt event loop handling.

    Args:
        title: Title for the visualization
        iteration: Current iteration number
        external_plotter: Optional external plotter (e.g., from GUI)

    Notes:
        Mirrors HexThermalFEA.plot_pseudo_density_realtime -- this method
        was previously missing here, so any driver's plot_progress=True
        path (the drivers call fe_solver.plot_pseudo_density_realtime(...)
        generically for both physics) raised an AttributeError for structural problems.
        Delegates to the HexFEAPlotter, which handles the actual
        visualization. When external_plotter is provided (GUI mode), it
        will use that plotter instead of creating a new BackgroundPlotter
        window.
    """
    return self.plotter.plot_pseudo_density_realtime(
        title=title,
        iteration=iteration,
        external_plotter=external_plotter
    )

#################################################################
if __name__ == "__main__":
  from pyto.examples_benchmarks.hex_structural_examples import StructuralExamples,getStructuralProblem

  problem = StructuralExamples.LBracketThick
  nDOFDesired = 10000
  mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,nDOFDesired = nDOFDesired)
  solver = linear_solvers.Solvers.DPCG # typically DPCG or PARDISO
  
  dsolver = deflation.DeflationSolver()

  if (solver == linear_solvers.Solvers.DPCG):
    nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
    dsolver.create_deflation_groups(mesh, nGroups)
    #dsolver.plot_deflation_groups(mesh)
    dsolver.create_deflation_matrix(mesh)
    dsolver.W = dsolver.W[bc.free_dofs, :]
  
  fe_solver = HexStructuralFEA(mesh = mesh,
        mat_prop = mat_prop,
        bc = bc,
        solver = solver,
        dsolver = dsolver,
        rtol = 1e-8,
        elem_body_force = elem_body_force)

  
  fe_solver.plot_mesh(plot_bc = True,offsetArrow = True)
  startTime = time.time()

  fe_solver.solve()
  print(f"Time to solve: {time.time() - startTime:.2f} seconds")
  fe_solver.postprocess()
  print(f"Maximum deformation: {fe_solver.max_deformation:.4e}")
  print(f"Maximum von Mises stress: {np.max(fe_solver.vonMisesStress):.4e}")
  print(f"Maximum p-norm stress (PNORM = {_PNORM_EXPONENT}): {fe_solver.pNormStress:.4e}")

  fe_solver.plot_deformation(show_geometry=True)
  fe_solver.plot_vonMisesStress()
  fe_solver.plot_stress_component(0)
  
