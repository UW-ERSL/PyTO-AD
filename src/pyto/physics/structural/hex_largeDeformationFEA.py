import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import roots_legendre
import numpy as np
import pyto.solve.deflation as deflation
import pyto.solve.numpy_backend as linear_solvers
from pyto.autodiff.sparse_solve import _PardisoFactors
import pyvista as pv
from numba import njit
import time
import enum

dsolver = deflation.DeflationSolver()

class ControlType(enum.Enum):
	ForceControl = enum.auto()
	DisplacementControl = enum.auto()

class largeDeformationFEA:
    def __init__(self,mesh, mat_prop, bc, shape='Linear',solver= linear_solvers.Solvers.PARDISO):
        self.shape = shape
        self.material_model = 'GeneralizedNeoHookean1'
        self.mesh = mesh
        self.material_properties = mat_prop
        self.bc= bc
        E = self.material_properties.youngs_modulus  
        nu = self.material_properties.poissons_ratio
        self.bulkModulus =  E/(3*(1-2*nu)) 
        self.shearModulus = E/(2*(1+nu))
        self.neuman_force = np.zeros((self.mesh.num_nodes * 3, 1))
        self.nodal_dirichlet = np.zeros((self.mesh.num_nodes * 3, 1))
        self.F = np.zeros((self.mesh.num_nodes * 3, 1))

        nQuadPts1D = 2
        self.xi, self.wt = self.gauss_quad_3d(nQuadPts1D)    
        self.n, self.grad_n = self.get_shapefunction_gradients_at_quadpts()

        self.solver = solver
        if (self.solver == linear_solvers.Solvers.DPCG):
            print('Using DPCG Solver')
            nGroups =  min(dsolver.maxGroups,max(dsolver.minGroups,round(3*mesh.num_nodes/dsolver.dofPerGroup)))
            dsolver.create_deflation_groups(mesh, nGroups)
            dsolver.create_deflation_matrix(mesh)
            dsolver.W = dsolver.W[bc.free_dofs, :]
        elif (self.solver == linear_solvers.Solvers.PARDISO):
            print('Using Pardiso Solver')
        else:
            print('Solver not implemented')


    def solve_nonlinear_fem_force_control(self,verbose = True, n_steps = 5,max_iter = 30,tol = 1e-8):
        """
        Solve nonlinear finite element problem using Newton-Raphson method.
        """
        print('Force Control')
        success = False
        n_dof = 3*self.mesh.num_nodes
        self.sol = np.zeros(n_dof)
        areaof_forceappliedface = self.mesh.bbox.ly*self.mesh.bbox.lz
        self.neuman_force=self.bc.force
        neuman_force_on_face = self.neuman_force*areaof_forceappliedface
        fixed_dofs = self.bc.fixed_dofs
        free_dofs = self.free_dof = self.bc.free_dofs
        nodal_dirichlet_fixed = np.ravel(self.nodal_dirichlet)[fixed_dofs]
        for step in range(1, n_steps + 1):
            if verbose:
                print(f'Load Step {step}/{n_steps}')
            load_factor = step/n_steps
            iter = 1
            err = 1
            while (iter < max_iter) and (err > tol):
                self.assemble_k()  # Assembles K, M matrices and F vector
                b = self.F - (load_factor * neuman_force_on_face).flatten()
                u_fixed = -load_factor * nodal_dirichlet_fixed + self.sol[fixed_dofs]

                K_free_rows = self.K[free_dofs, :]
                K_ff = K_free_rows[:, free_dofs]
                K_fd = K_free_rows[:, fixed_dofs]

                b_free = b[free_dofs] - K_fd.dot(u_fixed)
                if (self.solver == linear_solvers.Solvers.DPCG):
                    M = linear_solvers._jacobi_preconditioner(K_ff)
                    d_u_free = -dsolver.deflatedPCG(K_ff,
                                b_free,
                                W = dsolver.W,
                                M = M,
                                rtol = 1e-8)
                elif (self.solver == linear_solvers.Solvers.PARDISO):
                    # same pattern every Newton step: PARDISO's symbolic analysis is reused, only the numeric factorization reruns
                    d_u_free = -_PardisoFactors(K_ff, symmetric=False).solve(b_free)
                else:
                    print('Solver not implemented')

                d_u = np.zeros(n_dof, dtype=b.dtype)
                d_u[free_dofs] = d_u_free

                self.delta_sol = d_u
                self.sol += self.delta_sol
             
                self.u = self.sol[::3]
                self.v = self.sol[1::3]
                self.w = self.sol[2::3]
                
                err = np.linalg.norm(self.delta_sol)/np.linalg.norm(self.sol)
                
                if verbose:
                    print(f'\tIteration: {iter}\t {err:E}')
                iter += 1
        
        success = True if err < tol else False
        self.deformation = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.max_deformation = np.max(self.deformation)
        
        return  success


    def solve_nonlinear_fem_displacement_control(self,verbose = True, n_steps = 5,max_iter = 30,tol = 1e-9):
        """
        Solve nonlinear finite element problem using Newton-Raphson method.
        """
        success = False
        n_dof = 3*self.mesh.num_nodes
        self.sol = np.zeros(n_dof)
        #areaof_forceappliedface = self.mesh.bbox.ly*self.mesh.bbox.lz
        self.neuman_force=self.bc.force
        self.nodal_dirichlet = self.bc.dirichlet_values
        #neuman_force_on_face = self.neuman_force*areaof_forceappliedface
        self.delta_sol = np.zeros(n_dof)
        fixed_dofs = self.bc.fixed_dofs
        free_dofs = self.free_dof = self.bc.free_dofs
        load_factor = 1/n_steps

        for step in range(1, n_steps + 1):
            if verbose:
                print(f'Load Step {step}/{n_steps}')
            
            self.delta_sol = load_factor * self.nodal_dirichlet
            iter = 0
            err = 1
            while (iter < max_iter) and (err > tol):
                if verbose:
                    print(f'Load Step {step}/{n_steps} Iteration {iter}')
                iter += 1
                self.assemble_k()  # Assembles K, M matrices and F vector
                
                b = self.F

                # u_fixed = np.empty_like(fixed_dofs, dtype=b.dtype)
                # for idx, i in enumerate(fixed_dofs):
                #     u_fixed[idx] = self.delta_sol[i].item()


                K_free_rows = self.K[free_dofs, :]
                K_ff = K_free_rows[:, free_dofs]

                if iter == 1:
                    b[free_dofs] = b[free_dofs] + K_free_rows[:, fixed_dofs].dot(self.delta_sol[fixed_dofs])

                if (self.solver == linear_solvers.Solvers.DPCG):
                    M = linear_solvers._jacobi_preconditioner(K_ff)
                    d_u_free = -dsolver.deflatedPCG(K_ff,
                                b[free_dofs],
                                W = dsolver.W,
                                M = M,
                                rtol = 1e-8)
                elif (self.solver == linear_solvers.Solvers.PARDISO):
                    d_u_free = -_PardisoFactors(K_ff, symmetric=False).solve(b[free_dofs])
                else:
                    print('Solver not implemented')

                #d_u = np.zeros(n_dof, dtype=b.dtype)
                self.delta_sol[free_dofs] = d_u_free

                #self.delta_sol = d_u

                if iter == 1:
                    self.sol += self.delta_sol
                else:
                    self.sol[free_dofs] += self.delta_sol[free_dofs]
             
                err = np.linalg.norm(self.delta_sol[free_dofs])/np.linalg.norm(self.sol[free_dofs])
                errb = np.linalg.norm(b)/n_dof
                
                if verbose:
                    print(f'{iter}\t {err:E}\t {errb:E}')
            #plots.plotMesh(self.mesh, bc=self.boundary_conditions, u=self.sol, title = 'Large Deformation Beam')
        success = True if err < tol else False        
        self.u = self.sol[::3]
        self.v = self.sol[1::3]
        self.w = self.sol[2::3]                
        self.deformation = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        self.max_deformation = np.max(self.deformation)
        
        return  success

    def assemble_k(self):
        """
        Assemble global stiffness matrix and force vector
        """
        self.dof_per_node = 3
        n_dof = self.dof_per_node * self.mesh.num_nodes
        
        self.nodes_per_element = 8
        self.dof_per_elem = self.dof_per_node * self.nodes_per_element
        n_elements = self.mesh.num_elems
        nzmax = self.dof_per_elem**2 * n_elements # Maximum number of non-zero entries
        
        row_triplets = np.zeros(nzmax)
        col_triplets = np.zeros(nzmax)
        entry_triplets = np.zeros(nzmax)
        f = np.zeros(n_dof)
        
        grad_n_cell = self.grad_n
        dof_per_elem = self.dof_per_elem
        wt = self.wt
        nodes_per_element = self.nodes_per_element
        material_model = self.material_model
        shearModulus = self.shearModulus
        bulkModulus = self.bulkModulus

        elemArray = self.mesh.elemArray
        node_xyz = self.mesh.node_xyz
        sol = self.sol

        compute_K_global(n_elements, elemArray, node_xyz, sol, grad_n_cell, dof_per_elem, wt, nodes_per_element, material_model, shearModulus, bulkModulus,
                                row_triplets, col_triplets, entry_triplets, f)
            
        self.K = csr_matrix((entry_triplets, (row_triplets, col_triplets)), 
                            shape=(n_dof, n_dof))
        self.F = f
    

    def compute_strain_energy(self):
        """
        Compute strain energy
        """
        self.strain_energy = 0.5 * self.sol @ (self.K[:3*self.mesh.num_nodes, :3*self.mesh.num_nodes] @ self.sol)
        return self
    
    def gauss_quad_2d_quad(self, num_gq=4):
        """
        Gauss quadrature points for 2D quadrilateral elements using tensor product.
        Args:
            num_gq (int): Number of quadrature points per direction.
        Returns:
            tuple: (xi_GQ, wt_GQ) 2D Parametric coordinates and weights.
        """
        N = int(np.sqrt(num_gq))
        x, w = roots_legendre(N)
        
        # Reverse the nodes and weights to get positives first
        x = x[::-1]
        w = w[::-1]
        
        xi_GQ = np.zeros((2, N * N))
        xi_GQ[0, :] = np.repeat(x, N)
        xi_GQ[1, :] = np.tile(x, N)
        
        wt_GQ = np.outer(w, w).flatten()
        
        return xi_GQ, wt_GQ
    
    def gauss_quad_3d(self, num_gq=1):
        """
        Gauss quadrature points for 3D elements using tensor product
        Args:
            num_gq (int): Number of quadrature points per direction
        Returns:
            tuple: (xi_GQ, wt_GQ) 3D Parametric coordinates and weights
        """
        # Get 1D Gauss points and weights
        x, w = self.lgwt(num_gq, -1, 1)
        # Reverse the nodes and weights to get positives first
        x = x[::-1]
        w = w[::-1]
        # X, Y, Z are each of shape (N, N, N)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        # Flatten the grids and stack them into a (3, N^3) array.
        xi_gq = np.vstack((X.flatten(order='F'),
                    Y.flatten(order='F'),
                    Z.flatten(order='F')))
        # Form the 3D quadrature weights as the tensor product of the 1D weights.
        wt_gq = np.kron(np.kron(w, w), w)
        return xi_gq, wt_gq

  
    def lgwt(self,N, a, b):
        """
        Compute Legendre-Gauss nodes and weights for interval [a,b]
        
        Args:
            N (int): Number of points
            a (float): Left endpoint
            b (float): Right endpoint
        
        Returns:
            tuple: (x,w) nodes and weights
        """
        x, w = roots_legendre(N)
        x = 0.5 * (b - a) * x + 0.5 * (a + b)
        w = 0.5 * (b - a) * w
        return x, w
    

    def jacobian(position_nodes, shape, xi):
        """
        Compute Jacobian matrix at parametric coordinates for an element
        Args:
            elem: Element number
            xi: Parametric coordinates
        Returns:
            ndarray: Jacobian matrix
        """
    
        _, grad_N = shape_function(shape, xi)
        return grad_N @ position_nodes

    def get_shapefunction_gradients_at_quadpts(self):
        """
        Get all shape function gradients at quadrature points
        """
        # Precompute shape functions at quadrature points
        n_cell = []
        grad_n_cell = []
        for i in range(len(self.xi[0])):
            N, grad_N = shape_function(self.shape, self.xi[:,i])
            n_cell.append(N)
            grad_n_cell.append(grad_N)
        return n_cell, grad_n_cell

    def print_sparse_matrix(self):
        """
        Print the non-zero values of the sparse matrix K in (i, j) value format.
        """
        coo = self.K.tocoo()  # Convert to COO format
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if v != 0:
                print(f"({i}, {j}) {v}")
    #################################################################
    def plot_mesh(self, title = None,plot_bc = True, save_path=None):
        
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
        cells = np.hstack((
                        np.full((n_faces, 1), 4),  # 4 vertices per face
                        faces
                        ))

        pv_mesh = pv.UnstructuredGrid(cells, np.full(len(cells), pv.CellType.QUAD), vertices) # 9 is VTK_QUAD
        
        # Add density values to cells
        pv_mesh.cell_data['density'] = face_densities

        # Create plotter
        save_path = None
        if save_path is  None:
            plotter = pv.Plotter(window_size=(500, 400))
        else:
            plotter = pv.Plotter(off_screen=True)
        
        plotter.add_title(title, font_size=8)
    
        plotter.add_mesh(
                        pv_mesh,
                        color='lightgreen',
                        show_edges=True,
                        edge_color='black',
                        line_width=1
                    )


        # Add coordinate axes widget
        plotter.add_axes(
                        xlabel='X',
                        ylabel='Y',
                        zlabel='Z',
                        line_width=2,
                        labels_off=False,  # Show axis labels
                        color='black'
                        )
        
        if (plot_bc):
            # Add dots and force arrows for labeled nodes
            point_size = 10  # Size of dots in pixels

        # Add black dots for label 1 (fixed nodes)
        label1_nodes = np.where(self.mesh.node_indices[:, 3] == 1)[0]
        if len(label1_nodes) > 0 and self.bc is not None:
            points1 = vertices[label1_nodes]
            dots1 = pv.PolyData(points1)
            plotter.add_points(dots1,
                            color='black',
                            point_size=point_size,
                            render_points_as_spheres=True)

        # Add force arrows for label 2 (without red dots)
        label2_nodes = np.where(self.mesh.node_indices[:, 3] == 2)[0]
        if len(label2_nodes) > 0  and self.bc is not None: #structural
            # Add force arrows
            arrow_scale = 0.1 * self.mesh.bbox.diag_length
            for node in label2_nodes:
                # Get force components for this node
                fx = self.bc.force[3*node]
                fy = self.bc.force[3*node + 1]
                fz = self.bc.force[3*node + 2]
                force_vec = np.array([fx, fy, fz])
                
                # Only add arrow if force is non-zero
                if np.linalg.norm(force_vec) > 0:
                    # Normalize and scale force vector
                    force_vec = force_vec / np.linalg.norm(force_vec) * arrow_scale
                    
                    # Create arrow
                    start_point = vertices[node]
            
                    # Add arrow to plot
                    arrow = pv.Arrow(start = start_point,
                                    direction = force_vec,
                                    scale = arrow_scale)
                    plotter.add_mesh(arrow, color='red')
        
        # Set camera position for left-bottom-forward view
        view_distance = 2.5 * self.mesh.bbox.diag_length
        offset = 0.2 * view_distance  # Offset for object position
        plotter.camera_position = [
                        (view_distance*0.5, -view_distance*0.3, view_distance),
                        (offset, offset, 0),   # Focus point - right and bottom
                        (0, 0.8, 0.4)]         # Up vector - Y axis up

        # Reset camera and zoom out slightly
        plotter.camera.zoom(0.8)
        
        # Enable anti-aliasing for better quality
        plotter.enable_anti_aliasing()
        
        # Save image if path is provided
        if save_path:
        #plotter.show(screenshot = save_path)
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show() 
        return
    ################################################################# 
    def plot_deformation(self):
        # Return if no solution exists yet
        if not hasattr(self, 'sol'):
            return None

        # Create vertices array
        vertices = self.mesh.node_xyz
    
        sol = self.sol.copy()
        sol = sol.reshape((-1, 3))
        delta = self.deformation
        deltaMax = self.max_deformation
        scale = float(0.1*self.mesh.bbox.diag_length/deltaMax)
        scale = 1.0 # for large deformation
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

        # Create plotter
        save_path = None
        if save_path is  None:
            plotter = pv.Plotter(window_size=(500, 400))
        else:
            plotter = pv.Plotter(off_screen=True)
        
        plotter.add_title(f'Deformation scale: {scale:.2g}', font_size=8)
        # Add mesh to plotter
        nDOF = 3*self.mesh.num_nodes
        plotter.add_mesh(
                        pv_mesh,
                        scalars='values' if delta is not None else 'density',
                        show_edges=True,
                        cmap='jet',
                        edge_color='black',
                        line_width=1,
                        scalar_bar_args={
                                'title': '',
                                'vertical': True,
                                'position_x': 0.8,
                                'position_y': 0.3,
                                'width': 0.06
                                }
                    )

        # Add coordinate axes widget
        plotter.add_axes(
                        xlabel='X',
                        ylabel='Y',
                        zlabel='Z',
                        line_width=2,
                        labels_off=False,  # Show axis labels
                        color='black'
                        )

        # Set camera position for left-bottom-forward view
        view_distance = 2.5 * self.mesh.bbox.diag_length
        offset = 0.2 * view_distance  # Offset for object position
        plotter.camera_position = [
                        (view_distance*0.5, -view_distance*0.3, view_distance),
                        (offset, offset, 0),   # Focus point - right and bottom
                        (0, 0.8, 0.4)]         # Up vector - Y axis up

        # Reset camera and zoom out slightly
        plotter.camera.zoom(0.8)
        
        # Enable anti-aliasing for better quality
        plotter.enable_anti_aliasing()
        
        # Save image if path is provided
        if save_path:
            #plotter.show(screenshot = save_path)
            plotter.screenshot(save_path)
            plotter.close()
        else:
            plotter.show() 
        
        return 


@njit(cache=True)
def compute_K_global(n_elements, elemArray, node_xyz, sol, grad_n, dof_per_elem, wt, nodes_per_element, material_model, shearModulus, bulkModulus,
                    row_triplets, col_triplets, entry_triplets, f):
    index = 0
    for elem in range(n_elements):
        elem_nodes = elemArray[elem]
        position_nodes = node_xyz[elem_nodes, :]
        sol_elem = np.vstack((sol[3*elem_nodes], sol[3*elem_nodes+1], sol[3*elem_nodes+2]))

        k_elem, f_elem = compute_element_stiffness_finite_strain_spatial_conf(grad_n, dof_per_elem, wt, nodes_per_element, sol_elem, position_nodes, material_model, shearModulus, bulkModulus)
        
        if (k_elem is None):
            print(f'Element {elem}: det(F) < 0')
            continue
        dof = np.vstack((3*elem_nodes, 3*elem_nodes + 1, 3*elem_nodes + 2))
        #dof = dof.reshape(-1, order='F')
        dof = dof.T.flatten()

        # Create temp matrix by replicating dof myDOFPerElem times
        temp = np.empty((dof_per_elem, dof.size), dtype=dof.dtype)
        for i in range(dof_per_elem):
            temp[i, :] = dof
        # Transpose temp to prepare for Fortran-like flattening
        temp_T = temp.T

        # Flatten the transposed array
        row_index = temp_T.flatten()

        # Since temp_T is the transpose, its flattening gives the desired column-major order
        col_index = temp.flatten()

        # For k_elem, transpose before flattening to achieve Fortran-like order
        entries = k_elem.T.flatten()

        row_triplets[index:index+dof_per_elem**2] = row_index
        col_triplets[index:index+dof_per_elem**2] = col_index
        entry_triplets[index:index+dof_per_elem**2] = entries
        
        index += dof_per_elem**2
        f[dof] += f_elem
        
@njit(cache=True)
def compute_element_stiffness_finite_strain_spatial_conf(grad_n, dof_per_elem, wt, nodes_per_element, sol_elem, position_nodes, material_model, shearModulus, bulkModulus):
    """
    Computes elemental stiffness matrix and force vector for given element
    Args:
        elem: Element number
    Returns:
        tuple: (K_elem, f_elem) Elemental stiffness matrix and force vector
    """
    grad_n_cell = grad_n  # Shape function gradient values at quadrature points
    k_material = np.zeros((dof_per_elem, dof_per_elem))
    k_geometric = np.zeros((dof_per_elem, dof_per_elem))
    f_elem = np.zeros(dof_per_elem)
    #xi_gq = xi  # Quadrature points
    wt_gq = wt  # Quadrature weights
    num_gq = len(wt_gq)
    
    nodes = nodes_per_element
 
    k_elem = compute_k_elem(num_gq, grad_n_cell, wt_gq, position_nodes, sol_elem, nodes, k_material, k_geometric, f_elem, material_model, shearModulus, bulkModulus)
    if (k_elem is None):
        return None, None
    return k_elem, f_elem


@njit(cache=True)
def compute_k_elem(num_gq, grad_n_cell, wt_gq, position_nodes, sol, nodes, k_material, k_geometric, f_elem, material_model, shearModulus, bulkModulus):
    for g in range(num_gq):
        grad_n_all = grad_n_cell[g]
        j_total = grad_n_all @ position_nodes
        grad_ndx = np.linalg.solve(j_total, grad_n_all)
        F = np.eye(3) + sol@ grad_ndx.T
        b = F @ F.T  # Left green deformation tensor
        F_inv = np.linalg.inv(F)
        
        grad_ndxs = np.zeros_like(grad_ndx)
        for k in range(nodes):
            for i in range(3):
                grad_ndxs[i,k] = np.sum(grad_ndx[:,k] * F_inv[:,i])

        J_F = np.linalg.det(F)
        if J_F < 0:
            #print('Determinant of elem F negative')
            return None
            
        stress = kirchhoff_stress(material_model, shearModulus, bulkModulus, b, J_F)

        C = compute_elasticity_tensor_generalized_neo_hookean(material_model, shearModulus, bulkModulus, b, J_F)
        dJ = abs(np.linalg.det(j_total))

        compute_K_material_geometric(nodes, g, dJ, grad_ndxs, wt_gq, stress, C, k_material, k_geometric, f_elem)

    k_elem = k_material + k_geometric
    return k_elem


@njit(cache=True)
def compute_K_material_geometric(nodes, g, dJ, grad_ndxs, wt_gq, stress, C, k_material, k_geometric, f_elem):
    """
    Compute material and geometric stiffness matrices for an element
    Args:
        elem (int): Element number
        xi_gq (ndarray): Quadrature points
        wt_gq (ndarray): Quadrature weights
    Returns:
        tuple: (K_material, K_geometric) Material and geometric stiffness matrices
    """
    for A in range(nodes):
            for i in range(3):
                for B in range(nodes):
                    for k in range(3):
                        for j in range(3):
                            for l in range(3):
                                k_material[3*A+i,3*B+k] += wt_gq[g]*dJ*grad_ndxs[j,A]*C[i,j,k,l]*grad_ndxs[l,B]
                            k_geometric[3*A+i,3*B+k] -= wt_gq[g]*dJ*grad_ndxs[k,A]*grad_ndxs[j,B]*stress[i,j]

                for J in range(3):
                    f_elem[3*A + i] += wt_gq[g]*dJ*stress[i,J]*grad_ndxs[J,A]


#@njit(cache=True)
def shape_function(shape, xi):
    """
    Compute shape functions and gradients at parametric coordinates
    Args:
        xi: Parametric coordinates (1D, 2D or 3D)
    Returns:
        tuple: (N, grad_N) Shape functions and gradients
    """
    if np.size(xi) == 1:
        if shape == 'Linear':
            N = np.array([(1-xi)/2, (1+xi)/2])
            grad_N = np.array([-1/2, 1/2])
        elif shape == 'Quadratic':
            N = np.array([xi*(xi-1)/2, (1-xi)*(1+xi), xi*(xi+1)/2])
            grad_N = np.array([(2*xi-1)/2, -2*xi, (2*xi+1)/2])
    elif np.size(xi) == 2:
        N = 0.25 * np.array([
            (1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]),
            (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])
        ])
        grad_N = 0.25 * np.array([
            [xi[1]-1, 1-xi[1], xi[1]+1, -xi[1]-1],
            [xi[0]-1, -xi[0]-1, xi[0]+1, 1-xi[0]]
        ])
    elif np.size(xi) == 3:
        N = 0.125 * np.array([
            (1-xi[0])*(1-xi[1])*(1-xi[2]), (1+xi[0])*(1-xi[1])*(1-xi[2]),
            (1+xi[0])*(1+xi[1])*(1-xi[2]), (1-xi[0])*(1+xi[1])*(1-xi[2]),
            (1-xi[0])*(1-xi[1])*(1+xi[2]), (1+xi[0])*(1-xi[1])*(1+xi[2]),
            (1+xi[0])*(1+xi[1])*(1+xi[2]), (1-xi[0])*(1+xi[1])*(1+xi[2])
        ])
        grad_N = 0.125 * np.array([
            [(xi[1]-1)*(1-xi[2]), (1-xi[1])*(1-xi[2]), (xi[1]+1)*(1-xi[2]), (-xi[1]-1)*(1-xi[2]),
                (xi[1]-1)*(1+xi[2]), (1-xi[1])*(1+xi[2]), (xi[1]+1)*(1+xi[2]), (-xi[1]-1)*(1+xi[2])],
            [(xi[0]-1)*(1-xi[2]), (-xi[0]-1)*(1-xi[2]), (xi[0]+1)*(1-xi[2]), (1-xi[0])*(1-xi[2]),
                (xi[0]-1)*(1+xi[2]), (-xi[0]-1)*(1+xi[2]), (xi[0]+1)*(1+xi[2]), (1-xi[0])*(1+xi[2])],
            [-(1-xi[0])*(1-xi[1]), -(1+xi[0])*(1-xi[1]), -(1+xi[0])*(1+xi[1]), -(1-xi[0])*(1+xi[1]),
                (1-xi[0])*(1-xi[1]), (1+xi[0])*(1-xi[1]), (1+xi[0])*(1+xi[1]), (1-xi[0])*(1+xi[1])]
        ])
    return N, grad_N

@njit(cache=True)
def kirchhoff_stress(material_model, shearModulus, bulkModulus, B, J):
    """     Calculate Kirchhoff stress tensor for hyperelastic materials.
    This function implements two variations of the Generalized Neo-Hookean material model:
    GeneralizedNeoHookean1:
        Uses a volumetric energy term k*J*(J-1)^2/2
        Suitable for moderate compressibility
        More commonly used in literature
    GeneralizedNeoHookean2:
        Uses a volumetric energy term k*(J^2-1-2*ln(J))/4 
        Better suited for highly compressible materials
        More numerically stable at large deformations
    Parameters
    ----------
    B : ndarray, shape (3,3)
        Left Cauchy-Green deformation tensor
    J : float
        Determinant of the deformation gradient
    Returns
    -------
    stress : ndarray, shape (3,3)
        Kirchhoff stress tensor """
    stress = np.zeros((3, 3))
    delta = np.eye(3)
    Bkk = np.trace(B)
    
    if material_model == 'GeneralizedNeoHookean1':
        kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress)
                
    elif material_model == 'GeneralizedNeoHookean2':
        kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress)
                
    return stress

@njit(cache=True)
def kirchoff_stress_generalized_neo_hookean1(B, J, Bkk, delta, shearModulus, bulkModulus, stress):
    for i in range(3):
            for j in range(3):
                stress[i,j] = shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + bulkModulus * J * (J-1) * delta[i,j]

@njit(cache=True)
def kirchoff_stress_generalized_neo_hookean2(B, J, Bkk, delta, shearModulus, bulkModulus, stress):
    for i in range(3):
            for j in range(3):
                stress[i,j] = shearModulus * (B[i,j] - Bkk * delta[i,j] / 3.0) / J**(2/3) + 0.5 * bulkModulus * J * (J-1/J) * delta[i,j]
                
                
@njit(cache=True)
def compute_elasticity_tensor_generalized_neo_hookean(material_model, shearModulus, bulkModulus, B, J):
    """
    3D Elasticity tensor for hyperelastic material modeled by Generalized neohookean model
    """
    delta = np.eye(3)
    Bqq = np.trace(B)
    
    C = np.zeros((3, 3, 3, 3))
    
    if material_model == 'GeneralizedNeoHookean1':
        elasticity_tensor_generalised_neo_hookean_1(shearModulus, bulkModulus, Bqq, delta, B, J, C)
                        
    elif material_model == 'GeneralizedNeoHookean2':
        elasticity_tensor_generalised_neo_hookean_2(shearModulus, bulkModulus, Bqq, delta, B, J, C)

    
    return C

@njit(cache=True)
def elasticity_tensor_generalised_neo_hookean_1(shearModulus, bulkModulus, Bqq, delta, B, J, C):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k] 
                                    - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                    + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                + bulkModulus * (2*J - 1) * J * delta[i,j] * delta[k,l]
               
@njit(cache=True)
def elasticity_tensor_generalised_neo_hookean_2(shearModulus, bulkModulus, Bqq, delta, B, J, C):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = shearModulus * (delta[i,k] * B[j,l] + B[i,l] * delta[j,k]
                                    - (2/3) * (B[i,j] * delta[k,l] + delta[i,j] * B[k,l])
                                    + (2/3) * Bqq * delta[i,j] * delta[k,l] / 3) / J**(2/3) \
                                + bulkModulus * J * J * delta[i,j] * delta[k,l]
    
def get_corner_nodes_at_midplane_x(mesh, tol=1e-6) -> np.ndarray:
    """Get the node IDs that are closest to the z midplane and have min/max x, y coordinates.

    Args:
        tol (float): Tolerance for comparing coordinates.

    Returns:
        np.ndarray: Array of node IDs at the midplane of the z-axis that are also corner nodes.
    """
    nelx, nely, nelz = mesh.grid  # Number of elements in each direction
    ix_mid = nelx // 2  # Integer division to find the middle element index

    # Find nodes closest to the midplane
    midplane_nodes = np.where(mesh.node_indices[:, 0] == ix_mid-1)[0]
   
    # Get node coordinates
    node_coords = mesh.node_xyz[midplane_nodes]

    # Find the minimum z-coordinate
    min_z = np.min(node_coords[:, 2])
    # Find all nodes with the minimum z-coordinate
    min_z_nodes = midplane_nodes[np.isclose(node_coords[:, 2], min_z, atol=tol)]
    # Get coordinates of nodes with the minimum z-coordinate
    min_z_coords = mesh.node_xyz[min_z_nodes]
    # Find the nodes with the minimum and maximum y-coordinates among the min_z_nodes
    min_y = np.min(min_z_coords[:, 1])
    max_y = np.max(min_z_coords[:, 1])
    min_y_nodes_min_z = min_z_nodes[np.isclose(min_z_coords[:, 1], min_y, atol=tol)]
    max_y_nodes_min_z = min_z_nodes[np.isclose(min_z_coords[:, 1], max_y, atol=tol)]

    # Find the maximum z-coordinate
    max_z = np.max(node_coords[:, 2])
    # Find all nodes with the maximum z-coordinate
    max_z_nodes = midplane_nodes[np.isclose(node_coords[:, 2], max_z, atol=tol)]
    # Get coordinates of nodes with the maximum z-coordinate
    max_z_coords = mesh.node_xyz[max_z_nodes]
    # Find the nodes with the minimum and maximum y-coordinates among the max_z_nodes
    min_y = np.min(max_z_coords[:, 1])
    max_y = np.max(max_z_coords[:, 1])
    min_y_nodes_max_z = max_z_nodes[np.isclose(max_z_coords[:, 1], min_y, atol=tol)]
    max_y_nodes_max_z = max_z_nodes[np.isclose(max_z_coords[:, 1], max_y, atol=tol)]

    # Combine all corner nodes
    corner_nodes = np.concatenate((min_y_nodes_min_z, max_y_nodes_min_z, min_y_nodes_max_z, max_y_nodes_max_z))

    return corner_nodes

 
def get_corner_nodes_at_midplane_z(mesh, tol=1e-9) -> np.ndarray:
    """Get the node IDs that are closest to the z midplane and have min/max x, y coordinates.

    Args:
        tol (float): Tolerance for comparing coordinates.

    Returns:
        np.ndarray: Array of node IDs at the midplane of the z-axis that are also corner nodes.
    """
    nelx, nely, nelz = mesh.grid  # Number of elements in each direction
    iz_mid = nelz // 2  # Integer division to find the middle element index

    # Find nodes closest to the midplane
    midplane_nodes = np.where(mesh.node_indices[:, 2] == iz_mid-(iz_mid//10)+2)[0]
   
    # Get node coordinates
    node_coords = mesh.node_xyz[midplane_nodes]

    # Find the minimum x-coordinate
    min_x = np.min(node_coords[:, 0])
    # Find all nodes with the minimum x-coordinate
    min_x_nodes = midplane_nodes[np.isclose(node_coords[:, 0], min_x, atol=tol)]
    # Get coordinates of nodes with the minimum x-coordinate
    min_x_coords = mesh.node_xyz[min_x_nodes]
    # Find the nodes with the minimum and maximum y-coordinates among the min_x_nodes
    min_y = np.min(min_x_coords[:, 1])
    max_y = np.max(min_x_coords[:, 1])
    min_y_nodes_min_x = min_x_nodes[np.isclose(min_x_coords[:, 1], min_y, atol=tol)]
    max_y_nodes_min_x = min_x_nodes[np.isclose(min_x_coords[:, 1], max_y, atol=tol)]

    # Find the maximum x-coordinate
    max_x = np.max(node_coords[:, 0])
    # Find all nodes with the maximum x-coordinate
    max_x_nodes = midplane_nodes[np.isclose(node_coords[:, 0], max_x, atol=tol)]
    # Get coordinates of nodes with the maximum x-coordinate
    max_x_coords = mesh.node_xyz[max_x_nodes]
    # Find the nodes with the minimum and maximum y-coordinates among the max_x_nodes
    min_y = np.min(max_x_coords[:, 1])
    max_y = np.max(max_x_coords[:, 1])
    min_y_nodes_max_x = max_x_nodes[np.isclose(max_x_coords[:, 1], min_y, atol=tol)]
    max_y_nodes_max_x = max_x_nodes[np.isclose(max_x_coords[:, 1], max_y, atol=tol)]

    # Combine all corner nodes
    corner_nodes = np.concatenate((min_y_nodes_min_x, max_y_nodes_min_x, min_y_nodes_max_x, max_y_nodes_max_x))

    return corner_nodes


def get_cnodes_for_symmetryBC() -> np.ndarray:
    """Get the node IDs that are closest to the z midplane and have min/max x, y coordinates.

    Args:
        tol (float): Tolerance for comparing coordinates.

    Returns:
        np.ndarray: Array of node IDs at the midplane of the z-axis that are also corner nodes.
    """
    nelx, nely, nelz = mesh.grid  # Number of elements in each direction
    ix_mid = nelx // 2  # Integer division to find the middle element index
    iy_mid = nely // 2  # Integer division to find the middle element index

    # Find nodes closest to the midplane
    midplane_xnodes = np.where(mesh.node_indices[:, 0] == ix_mid)[0]
    midplane_ynodes = np.where(mesh.node_indices[:, 1] == iy_mid)[0]
   
    return midplane_xnodes, midplane_ynodes


if __name__ == "__main__":
 
    from pyto.examples_benchmarks.hex_structural_examples import StructuralExamples,getStructuralProblem

    problem = StructuralExamples.BeamSurfaceLoad
    control = ControlType.ForceControl 


    solver = linear_solvers.Solvers.PARDISO
    if (problem == StructuralExamples.BeamSurfaceLoad): # beam
        nDOFDesired= 2000
        totalLoad = 10000
        nForceSteps = max(1,int(totalLoad/20000))

    elif (problem == StructuralExamples.TorsionBar): 
        nDOFDesired= 2000
        totalLoad = 45000
        nForceSteps = 2
    elif (problem == StructuralExamples.FilletedBeam): 
        nDOFDesired= 500
        totalLoad = 7000# 7000 leads to negative det(F)
        nForceSteps = 2
    elif (problem == StructuralExamples.ArrowHead):
        nDOFDesired= 100000
        totalLoad = 1000000
    elif (problem == StructuralExamples.Brain):
        nDOFDesired= 100000 # not relevant
        totalLoad = 100
        nForceSteps = 2
        solver = linear_solvers.Solvers.DPCG

    
    mesh, mat_prop, bc,elem_body_force = getStructuralProblem(problem,totalLoad = totalLoad,nDOFDesired = nDOFDesired)
    if control == ControlType.ForceControl:
        pass
    else:
        print(f'Displacement Control')
        if (problem == 0):    
            fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
            dirichlet_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
            nodal_displacemnt = -0.03
            dirichlet_dofs = 3 * dirichlet_nodes + 2  # z direction
            fixed_dofs = np.concatenate((bc.fixed_dofs, dirichlet_dofs))
            mesh.node_indices[dirichlet_nodes, 3] = 2
            dirichlet_values = np.zeros(3*mesh.num_nodes)
            dirichlet_values[fixed_dofs] = 0
            dirichlet_values[dirichlet_dofs] = nodal_displacemnt
            
            bc.force = np.zeros(3*mesh.num_nodes)
            bc.fixed_dofs = fixed_dofs
            bc.dirichlet_values = np.zeros(3*mesh.num_nodes)
            bc.dirichlet_values = dirichlet_values
        elif problem == 1:
            #below two lines are required for applying dirichilet boundary conditions (nodal dirichilet)
            fixed_nodes = mesh.getNodesOnBoundingBoxPlane(2,True)  # z = 0 plane
            dirichlet_nodes = mesh.getNodesOnBoundingBoxPlane(2,False)  # z = zMax plane

            displacemnt_percent = -0.10 # 0.2 for 20% of the length of the beam in the z direction
            nodal_displacemnt = mesh.bbox.lz * displacemnt_percent 
            #Dimensions of the arrow head lattice are mesh.bbox.lz = np.float64(0.029999999213032424) #mesh.bbox.ly = np.float64(0.017000000807456672)
            fixed_dirch_nodesmerged = np.concatenate((fixed_nodes, dirichlet_nodes))
        
            fixed_dofs = np.array([3 * fixed_dirch_nodesmerged,
                    3 * fixed_dirch_nodesmerged + 1,
                    3 * fixed_dirch_nodesmerged + 2]).flatten().astype(int)
            midplane_xnodes, midplane_ynodes = get_cnodes_for_symmetryBC()
            fixed_xnodes_dofs = np.array([3 * midplane_xnodes]).flatten().astype(int) 
            fixed_ynodes_dofs = np.array([3 * midplane_ynodes + 1]).flatten().astype(int) 
            
            fixed_dofs = np.concatenate((fixed_dofs, fixed_xnodes_dofs, fixed_ynodes_dofs))
            ###For symmetry #####
            # Fixed boundary conditions    
            dirichlet_dofs = 3 * dirichlet_nodes + 2  # z direction
            mesh.node_indices[dirichlet_nodes, 3] = 2
            mesh.node_indices[midplane_xnodes, 3] = 1
            mesh.node_indices[midplane_ynodes, 3] = 1
            dirichlet_values = np.zeros(3*mesh.num_nodes)
            dirichlet_values[fixed_dofs] = 0
            dirichlet_values[dirichlet_dofs] = nodal_displacemnt
            
            bc.force = np.zeros(3*mesh.num_nodes)
            bc.fixed_dofs = fixed_dofs
            bc.dirichlet_values = np.zeros(3*mesh.num_nodes)
            bc.dirichlet_values = dirichlet_values
        

    ldFEA = largeDeformationFEA(mesh, mat_prop, bc, solver =  solver)
    print(f'Number of DOF: {ldFEA.mesh.num_nodes*3}')
    ldFEA.plot_mesh(plot_bc = True, title = f'DOF: {ldFEA.mesh.num_nodes*3}', save_path = None)

    start_time = time.time()
    if control == ControlType.ForceControl:
        ldFEA.solve_nonlinear_fem_force_control(n_steps = nForceSteps, verbose = True)
    else:
        ldFEA.solve_nonlinear_fem_displacement_control(verbose = True, n_steps = 100, tol = 1e-5)
    
    end_time = time.time()
    
    ldFEA.compute_strain_energy()
    print(f'Strain Energy: {ldFEA.strain_energy}')
    
    # Log the time taken
    time_taken = end_time - start_time
    print(f"Time taken: {time_taken} seconds")

    ldFEA.plot_deformation()
   
