import numpy as np
import pyto.core.mat_lib as mat_lib
import pyto.core.bc as bound_cond
import pyto.core.hex_mesher as hex_mesher
import os
import enum
import scipy.sparse as spy_sprs
from pyto.io.stl_reader import STLGeom
from scipy.sparse import lil_matrix
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # points at src/, matching pre-Phase-6 location (this file moved from src/ to src/pyto/examples_benchmarks/)


class StructuralExamples(enum.Enum):
	TensileBar = enum.auto()
	TorsionBar = enum.auto()
	BeamBending = enum.auto()
	ShearBlock = enum.auto()
	Mitchell = enum.auto()
	EdgeCantilever = enum.auto()
	EdgeCantileverConstraintMatrix = enum.auto()
	ShortCantileverTipLoad = enum.auto()
	ShortCantileverMidLoad = enum.auto()
	CantileverTipLoad = enum.auto()
	CantileverMidLoad = enum.auto()
	TensilePlate = enum.auto()
	TwoBar = enum.auto()
	Inverter = enum.auto()
	ThreeHoleBracket = enum.auto()
	ThreeHoleBracketThick = enum.auto()
	MBBBeam = enum.auto()
	Bridge = enum.auto()
	DistributedLoad = enum.auto()
	Multiload = enum.auto()
	GravityBar = enum.auto() 
	GravityPlate = enum.auto()
	LBracket = enum.auto()
	ArrowHead = enum.auto()
	CompliantMechanism = enum.auto()
	FilletedBeam = enum.auto()
	BeamSurfaceLoad = enum.auto()
	CentrifugalPlate = enum.auto()
	TorquePlate = enum.auto()
	KnuckleAssembly =  enum.auto()
	Table =  enum.auto()
	LBracketThick = enum.auto()
	GEGrabCAD = enum.auto()
	BliskSection =  enum.auto()
	BliskQuarter= enum.auto()
	BliskPressureLoading = enum.auto()

def getStructuralProblem(problem: StructuralExamples, **kwargs):
  """Returns a structural problem based on the given problem name.

  Parameters:
  ----------
  problem : StructuralExamples
    The name of the problem to return.
  **kwargs : dict
    Additional keyword arguments to pass to the problem creation function.

  Returns:
  -------
  tuple
    A tuple containing the mesh, material properties, and boundary conditions for the problem.
  """
  if problem == StructuralExamples.TensileBar:
    return createTensileBarProblem(**kwargs)
  elif problem == StructuralExamples.TorsionBar:
    return createTorsionBarProblem(**kwargs)
  elif problem == StructuralExamples.BeamBending:
    return createBeamBendingProblem(**kwargs)
  elif problem == StructuralExamples.ShearBlock:
    return createShearBlockProblem(**kwargs)
  elif problem == StructuralExamples.Mitchell:
    return createMitchellProblem(**kwargs)
  elif problem == StructuralExamples.EdgeCantilever:
    return createEdgeCantileverProblem(**kwargs)
  elif problem == StructuralExamples.EdgeCantileverConstraintMatrix:
    return createEdgeCantileverConstraintMatrixProblem(**kwargs)
  elif problem == StructuralExamples.ShortCantileverTipLoad:
    return createShortCantileverTipLoadProblem(**kwargs)
  elif problem == StructuralExamples.ShortCantileverMidLoad:
    return createShortCantileverMidLoadProblem(**kwargs)
  elif problem == StructuralExamples.CantileverTipLoad:
    return createCantileverTipLoadProblem(**kwargs)
  elif problem == StructuralExamples.CantileverMidLoad:
    return createCantileverMidLoadProblem(**kwargs)
  elif problem == StructuralExamples.MBBBeam:
    return createMBBBeamProblem(**kwargs)
  elif problem == StructuralExamples.Bridge:
    return createBridgeProblem(**kwargs)
  elif problem == StructuralExamples.LBracket:
    return createLBracketProblem(**kwargs)
  elif problem == StructuralExamples.TensilePlate:
    return createTensilePlateProblem(**kwargs)
  elif problem == StructuralExamples.TwoBar:
    return createTwoBarProblem(**kwargs)
  elif problem == StructuralExamples.DistributedLoad:
    return createDistributedLoadProblem(**kwargs)
  elif problem == StructuralExamples.Inverter:
    return createInverterProblem(**kwargs)
  elif problem == StructuralExamples.ThreeHoleBracket:
    return createThreeHoleBracketProblem(**kwargs)
  elif problem == StructuralExamples.ThreeHoleBracketThick:
    return createThreeHoleBracketThickProblem(**kwargs)
  elif problem == StructuralExamples.Multiload:
    return createMultiloadProblem(**kwargs)
  elif problem == StructuralExamples.GravityBar:
    return createGravityBarProblem(**kwargs)
  elif problem == StructuralExamples.GravityPlate:
    return createGravityPlateProblem(**kwargs)
  elif problem == StructuralExamples.CompliantMechanism:
    return createCompliantMechanismProblem(**kwargs)
  elif problem == StructuralExamples.FilletedBeam:
    return createFilletedBeamProblem(**kwargs)
  elif problem == StructuralExamples.BeamSurfaceLoad:
    return createBeamSurfaceLoadProblem(**kwargs)
  elif problem == StructuralExamples.CentrifugalPlate:
    return createCentrifugalPlateProblem(**kwargs)
  elif problem == StructuralExamples.TorquePlate:
    return createTorquePlateProblem(**kwargs)
  elif problem == StructuralExamples.LBracketThick:
    return createLBracketThickProblem(**kwargs)
  elif problem == StructuralExamples.KnuckleAssembly:
    return createKnuckleAssemblyProblem(**kwargs)
  elif problem == StructuralExamples.Table:
    return createTableProblem(**kwargs)
  elif problem == StructuralExamples.ArrowHead:
    return createArrowHeadProblem(**kwargs)
  elif problem == StructuralExamples.GEGrabCAD:
    return createGEGrabCADProblem(**kwargs)
  elif problem == StructuralExamples.BliskSection:
    return createBliskSectionProblem(**kwargs)
  elif problem == StructuralExamples.BliskQuarter:
    return createBliskQuarterProblem(**kwargs)
  elif problem == StructuralExamples.BliskPressureLoading:
    return createBliskPressureLoadingProblem(**kwargs)
  else:
    raise ValueError("Invalid structural example name.")
  

def createTensileBarProblem(nDOFDesired: int = 10000,tensileForce = 10000,allow_yz_displacement: bool = False):
  """Creates a tensile problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
   
  stl_file = os.path.join(script_dir, '../Models/Beam/beam.STL')
  nElemsDesired = nDOFDesired /3	# estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
  print("Structural num nodes:", mesh.num_nodes)

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  if (allow_yz_displacement):
    fixed_dofs = np.array([3 * fixed_nodes + 0]).flatten().astype(int) # fixed only in x direction
  else:
    fixed_dofs = np.array([3 * fixed_nodes,
                3 * fixed_nodes + 1,
                3 * fixed_nodes + 2]).flatten().astype(int)
    
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  force = np.zeros(3*mesh.num_nodes)
  # Add forces on x=xMax plane with different magnitudes for edges and corners
  # that is consistent with numerical integration over the surface
  if (abs(tensileForce) > 0):
    load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
    # Find edge and corner nodes
    edge_nodes = []
    corner_nodes = []
    face_nodes = [] 

    for node in load_nodes:
      coords = mesh.node_xyz[node]
      num_extremes = 0
      if abs(coords[1] - min(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2 or abs(coords[1] - max(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2:
        num_extremes += 1
      if abs(coords[2] - min(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2 or abs(coords[2] - max(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2:
        num_extremes += 1
      
      if num_extremes == 2:
        corner_nodes.append(node)
      elif num_extremes == 1:
        edge_nodes.append(node) 
      else:
        face_nodes.append(node)
    
    # Split into lists for clarity
    corner_nodes = np.array(corner_nodes)
    edge_nodes = np.array(edge_nodes)
    face_nodes = np.array(face_nodes)
    # Apply forces according to node type
    
    force[3*face_nodes] = 4.0  
    force[3*edge_nodes] = 2.0  
    force[3*corner_nodes] = 1.0 
    
    # Normalize forces to achieve desired total load
    total_load = np.sum(force[3*load_nodes])
    force[3*load_nodes] *= tensileForce/total_load
    load_nodes = np.union1d(np.union1d(face_nodes, edge_nodes), corner_nodes)
    mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")  
  elem_body_force = None
  if (abs(tensileForce) > 0):
    print('-----------------------------')
    print("Theoretical max displacement: {:.2g}".format(tensileForce*0.5/(mat_prop.youngs_modulus*0.05*0.05)))
    print("Theoretical max stress: {:.2g}".format(tensileForce/(0.05*0.05)))
    print('-----------------------------')
  return mesh, mat_prop, bc, elem_body_force

def createTorsionBarProblem(nDOFDesired: int = 10000, L: float = [1, 0.2, 0.2], totalLoad = 1000):
    """Creates a torsion bar problem with approximate desired DOFs.

    Parameters:
    ----------
    nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
    L : list of float  
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
    youngs_modulus : float
    Young's modulus of material (default 2e11)
    poissons_ratio : float
    Poisson's ratio of material (default 0.3)
    torque : float
    Applied torque magnitude (default 1000)

    Returns:
    -------
    tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and torque on right face
    """
    nVoxelsDesired = nDOFDesired/3    
    # Let the number of voxels be proportional to the length in each direction
    alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
    nelx = round(alpha*L[0])
    nely = round(alpha*L[1])
    nelz = round(alpha*L[2])
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems = (nelx, nely, nelz),
            elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
    mesh.createEdofMatStructural()

    # Fix all DOFs at x=0 plane
    fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True)
    fixed_dofs = np.array([3 * fixed_nodes,
          3 * fixed_nodes + 1,
          3 * fixed_nodes + 2]).flatten().astype(int)
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
    mesh.node_indices[fixed_nodes, 3] = 1

    # Get nodes on x=xMax plane for applying torque
    load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
    load_dofs = []
    force_values = []

    # Calculate center of right face
    right_face_nodes = mesh.node_xyz[load_nodes]
    face_center = np.mean(right_face_nodes, axis=0)
    # Apply torque forces on each node
    for node in load_nodes:
    # Get node position relative to face center
      node_pos = mesh.node_xyz[node] - face_center
      # Calculate radius in yz plane
      r = np.sqrt(node_pos[1]**2 + node_pos[2]**2)
     
      # Unit vector in tangential direction
      tangent_dir = np.array([0, -node_pos[2], node_pos[1]])/r
      # Force magnitude proportional to radius
      force_mag = totalLoad / (len(load_nodes) * r)
      # Add force components
      load_dofs.extend([3*node + i for i in range(3)])
      force_values.extend(force_mag * tangent_dir)

    mesh.node_indices[load_nodes, 3] = 2

    force = np.zeros(3*mesh.num_nodes)
    force[load_dofs] = force_values

    bc = bound_cond.BC(force = force,
        fixed_dofs = fixed_dofs,
        dirichlet_values = dirichlet_values) 

    mat_prop = mat_lib.get_material("Steel") 
    elem_body_force = None

    return mesh, mat_prop, bc, elem_body_force
  # ----------------------------------------

def createBeamBendingProblem(nDOFDesired: int = 10000, L: float = [0.5, 0.05, 0.05],appliedLoad = 1000):
  """Creates a beam bending problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
 
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  # Fix all DOFs at x=0 plane (cantilever)
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,3 * fixed_nodes+1,
                         3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  # Add downward forces at x=xMax plane
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane
  # Find edge and corner nodes
  edge_nodes = []
  corner_nodes = []
  face_nodes = [] 

  for node in load_nodes:
    coords = mesh.node_xyz[node]
    num_extremes = 0
    if abs(coords[1] - min(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2 or abs(coords[1] - max(mesh.node_xyz[:,1])) < mesh.elem_size[1]/2:
      num_extremes += 1
    if abs(coords[2] - min(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2 or abs(coords[2] - max(mesh.node_xyz[:,2])) < mesh.elem_size[2]/2:
      num_extremes += 1
    
    if num_extremes == 2:
      corner_nodes.append(node)
    elif num_extremes == 1:
      edge_nodes.append(node) 
    else:
      face_nodes.append(node)
  
  # Split into lists for clarity
  corner_nodes = np.array(corner_nodes)
  edge_nodes = np.array(edge_nodes)
  face_nodes = np.array(face_nodes)
  
  # Apply downward forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  force[3*face_nodes + 2] = -4.0  # Apply force in z direction (downward)
  force[3*edge_nodes + 2] = -2.0  
  force[3*corner_nodes + 2] = -1.0 
  
  # Normalize forces to achieve desired total load
  load_nodes = np.union1d(np.union1d(face_nodes, edge_nodes), corner_nodes)
  total_load = np.sum(force[3*load_nodes + 2])
  force[3*load_nodes + 2] *= appliedLoad/abs(total_load)
  print("Total force:", np.sum(force[3*load_nodes + 2]))
  mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None

  print('-----------------------------')
  # For a beam with rectangular cross-section under end load P
  # Maximum deflection = PL^3/(3EI) where I = (w*h^3)/12
  I = (L[1]*L[2]**3)/12
  print("Theoretical max deflection: {:.2g}".format(appliedLoad*L[0]**3/(3*mat_prop.youngs_modulus*I)))
  # Maximum bending stress = My/I where M = PL and y = h/2
  print("Theoretical max axial stress: {:.2g}".format(appliedLoad*L[0]*L[2]/(2*I)))
  print('-----------------------------')
 
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createShearBlockProblem(nDOFDesired: int = 10000, L: float = [1, 1, 1],shearForce = 1000):
  """Creates a shear problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
 
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes]).flatten().astype(int) # fixed in x direction
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  
  # Add forces on x=xMax plane with different magnitudes for edges and corners
  # that is consistent with numerical integration over the surface
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane

  load_dof = 3*load_nodes + 1 # y direction

  # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  force[load_dof] = shearForce/len(load_nodes)


  mesh.node_indices[load_nodes, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force



def createMitchellProblem(nDOFDesired: int = 10000, load1 = 5.6e4, load2 = 0):
  """
  See: Topology Optimization Benchmarks in 2D: Results for Minimum Compliance and Minimum Volume in Planar Stress Problems
S. Ivvan Valdez, et al. Arch Computat Methods Eng (2017) 24:803–839, DOI 10.1007/s11831-016-9190-3
  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/Mitchell/Mitchell.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  left_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane

  left_dofs = np.array([3 * left_nodes]).flatten().astype(int) # fixed in x direction

  bottom_nodes = mesh.getNodesOnBoundingBoxPlane(1,True) # y = 0 plane
  right_nodes = np.intersect1d(bottom_nodes, np.where(node_pts[:, 0] >= 0.9)[0]) # hard coded
  right_dofs = np.array([3 * right_nodes+1]).flatten().astype(int)# fixed in y direction
  
  fixed_dofs = np.union1d(left_dofs,right_dofs)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[left_nodes, 3] = 1
  mesh.node_indices[right_nodes, 3] = 1
  
  
  load_nodes_1 = np.intersect1d(bottom_nodes, np.where(node_pts[:, 0] <= 0.1))
  load_dof_1 = 3*load_nodes_1 + 1 # y direction

  load_nodes_2 = np.intersect1d(bottom_nodes, np.where((node_pts[:, 0] >= 0.25) & (node_pts[:, 0] <= 0.35))[0])
  load_dof_2 = 3*load_nodes_2 + 1 # y direction

  # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)

  force[load_dof_1] = -load1/len(load_nodes_1)
  force[load_dof_2] = -load2/len(load_nodes_2)

 
  mesh.node_indices[load_nodes_1, 3] = 2
  mesh.node_indices[load_nodes_2, 3] = 2
  
  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force


  # ----------------------------------------



def createInverterProblem(nDOFDesired: int = 10000):
  """
  See:Huang, X., Li, Y., Zhou, S.W. and Xie, Y.M., 2014. 
  Topology optimization of compliant mechanisms with desired structural stiffness. 
  Engineering Structures, 79, pp.13-21.
  Parameters:
  ----------
  
  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/Inverter/Inverter.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
  node_pts = mesh.node_xyz
  fix_ratio = 0.1 # ratio of the y height

  top_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) & (node_pts[:, 1] > ((1-fix_ratio) * np.max(node_pts[:, 1]))))[0]
  top_dofs = (3 * top_nodes[:, None] + np.arange(3)).flatten().astype(int) # fix all 3 directions
  bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) & (node_pts[:, 1] < (fix_ratio * np.max(node_pts[:, 1]))))[0]
  bottom_dofs = (3 * bottom_nodes[:, None] + np.arange(3)).flatten().astype(int) # fix all 3 directions

  fixed_dofs = np.union1d(top_dofs,bottom_dofs)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[top_nodes, 3] = 1
  mesh.node_indices[bottom_nodes, 3] = 1

  node_pts = mesh.node_xyz
  xMax = np.max(node_pts[:,0]) 
  yMid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  outputNodes = np.where((abs(node_pts[:, 0] - xMax) < mesh.elem_size[0]/2) & (abs(node_pts[:, 1] - yMid) < mesh.elem_size[1]/2))[0]
  if len(outputNodes) == 0:
    raise ValueError("No output nodes found. Check the mesh and node coordinates.")
 
  load_nodes = np.where((abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) & (abs(node_pts[:, 1] - np.mean(node_pts[:,1])) < 1.5*mesh.elem_size[1]))[0]
  load_dof = 3*load_nodes # x direction

  ## Add spring to output node and input nodes
  # Spring stiffness 0.1 (E = 1): the value the Reference results were made with. Commit f4e163e changed it to
  # 0.01, which makes a different (much softer-output) problem and objectives not comparable with the Reference.
  mesh.externalSprings = [(0.1,3*node) for node in np.concatenate((outputNodes, load_nodes))]
   
   # Apply forces according to node type
  force = np.zeros(3*mesh.num_nodes)
  
  load = 1
  force[load_dof] = load/len(load_nodes)
 
  mesh.node_indices[load_nodes, 3] = 2

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.create_material_with_defaults("dummyMaterial",youngs_modulus = 1.0, poissons_ratio = 0.3)
  
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createEdgeCantileverProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and y = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

# ----------------------------------------


def createEdgeCantileverConstraintMatrixProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1],totalLoad = 10000):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [0.1, 0.1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  
  C0 = np.zeros((3*len(fixed_nodes), mesh.num_nodes * 3))
  for i, node in enumerate(fixed_nodes):
    C0[3*i, 3*node] = 1
    C0[3*i+1, 3*node+1] = 1
    C0[3*i+2, 3*node+2] = 1

  constraint_matrix = spy_sprs.csr_matrix(C0)
  constraint_rhs = np.zeros(3*len(fixed_nodes))

  mesh.node_indices[fixed_nodes, 3] = 1
  # line defined by x = xMax, and z = 0 
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(2,True))
  load_dofs = 3 * load_nodes + 2  # z direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = [],
            dirichlet_values = [],
            constraint_matrix = constraint_matrix,
            constraint_rhs = constraint_rhs) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createShortCantileverMidLoadProblem(nDOFDesired: int = 10000,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/ShortCantilever/ShortCantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.05)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createShortCantileverTipLoadProblem(nDOFDesired: int = 10000, totalLoad = 5.8e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/ShortCantilever/ShortCantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
 
  y_tip_nodes = np.where(abs(node_pts[:,1]) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_tip_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  

def createCantileverMidLoadProblem(nDOFDesired: int = 10000,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/Cantilever/Cantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.05)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createCantileverTipLoadProblem(nDOFDesired: int = 10000,totalLoad = 5.8e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/Cantilever/Cantilever.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)

  
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
 
  y_tip_nodes = np.where(abs(node_pts[:,1]) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_tip_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")

  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createTwoBarProblem(nDOFDesired: int = 10000,totalLoad = 9e4):
  """Creates a edge loaded cantilever beam problem with approximate desired DOFs.

  Parameters:
  ----------
  nDOFDesired : int
    Desired number of degrees of freedom (default 10000)
  L : list of float
    Dimensions [Lx, Ly, Lz] of domain (default [1, 1, 0.1])
  youngs_modulus : float
    Young's modulus of material (default 2e11)
  poissons_ratio : float 
    Poisson's ratio of material (default 0.3)

  Returns:
  -------
  tuple
    (mesh, mat_prop, bc) containing:
    - mesh: Mesher object with grid discretization
    - mat_prop: Material properties object
    - bc: Boundary conditions with fixed left face and load on right face
  """
  stl_file = os.path.join(script_dir, '../Models/TwoBar/TwoBar.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  
  max_x_nodes = mesh.getNodesOnBoundingBoxPlane(0,False)
  # Get nodes with y coordinate near yMax/2
  node_pts = mesh.node_xyz
  y_mid = (np.max(node_pts[:,1]) + np.min(node_pts[:,1]))/2
  y_mid_nodes = np.where(abs(node_pts[:,1] - y_mid) < 0.1)[0]
  # Intersect the two sets to get nodes on the line

  load_nodes = np.intersect1d(max_x_nodes, y_mid_nodes)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------



def createMBBBeamProblem(nDOFDesired: int = 10000, load = 2.7e4):
  ''' 
    See: Topology Optimization Benchmarks in 2D: Results for Minimum Compliance and Minimum Volume in Planar Stress Problems
  S. Ivvan Valdez, et al. Arch Computat Methods Eng (2017) 24:803–839, DOI 10.1007/s11831-016-9190-3
'''
  stl_file = os.path.join(script_dir, '../Models/MBBBeam/MBBBeam.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  symmetry_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  symmetry_dofs = np.array([3 * symmetry_nodes]).flatten().astype(int)

  xMax = np.max(mesh.node_xyz[:,0])
  right_nodes=np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,True), np.where(mesh.node_xyz[:,0] == xMax)[0])
  right_dofs = np.array([3 * right_nodes+1, 3 * right_nodes + 2]).flatten().astype(int)
  
  fixed_dofs = np.union1d(symmetry_dofs,right_dofs)
  fixed_nodes = np.union1d(symmetry_nodes,right_nodes)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1

  xMin = np.min(mesh.node_xyz[:,0])
  load_nodes = np.intersect1d(mesh.getNodesOnBoundingBoxPlane(1,False), np.where(mesh.node_xyz[:,0] == xMin)[0])
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -load/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Aluminum") 
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createDistributedLoadProblem(nDOFDesired: int = 10000, L: float = [1.0, 0.5, 0.025]):
  stl_file = os.path.join(script_dir, '../Models/DistributedLoad/DistributedLoad.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  left_node =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,True))
  right_node =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  fixed_nodes = np.union1d(left_node,right_node)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1

 
  load_nodes = mesh.getNodesOnBoundingBoxPlane(1,False)
  load_dofs = 3 * load_nodes + 1  # y direction

  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -10000/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
def createBridgeProblem(nDOFDesired: None):
    # Define grid size and element size
    nelx, nely, nelz = 100, 50, 1
    Lx, Ly, Lz = nelx* 1.0, nely* 1.0, nelz * 1.0  # Example physical dimensions (adjust as needed)
    mesh = hex_mesher.HexMesher()
    mesh.grid_mesh(num_elems=(nelx, nely, nelz),
                   elem_size=(Lx/nelx, Ly/nely, Lz/nelz))
    mesh.createEdofMatStructural()

    node_pts = mesh.node_xyz
    bridge_length = np.max(node_pts[:, 0]) - np.min(node_pts[:, 0])

    # Fix left bottom edge (all 3 DOFs fixed)
    left_bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.min(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                                 (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    left_bottom_dofs = np.array([3 * left_bottom_nodes,
                                 3 * left_bottom_nodes + 1,
                                 3 * left_bottom_nodes + 2]).flatten().astype(int)

    # Fix right bottom edge (only y and z fixed, x free)
    right_bottom_nodes = np.where((np.abs(node_pts[:, 0] - np.max(node_pts[:, 0])) < mesh.elem_size[0]/2) &
                                  (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    right_bottom_dofs = np.array([3 * right_bottom_nodes + 1,  # y DOF
                                  3 * right_bottom_nodes + 2]).flatten().astype(int)

    # Combine fixed DOFs
    fixed_dofs = np.union1d(left_bottom_dofs, right_bottom_dofs)
    dirichlet_values = 0 * np.ones_like(fixed_dofs, dtype=float)
    mesh.node_indices[left_bottom_nodes, 3] = 1  # for plotting
    mesh.node_indices[right_bottom_nodes, 3] = 1  # for plotting

    # Apply edge loads
    force = np.zeros(3 * mesh.num_nodes)

    # Load at 1/3rd the length of the bridge
    load_nodes_1 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 3)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_1 + 1] = -1 / len(load_nodes_1)  # y direction
    mesh.node_indices[load_nodes_1, 3] = 2  # for plotting

    # Load at 1/2 the length of the bridge
    load_nodes_2 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + bridge_length / 2)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_2 + 1] = -2 / len(load_nodes_2)  # y direction
    mesh.node_indices[load_nodes_2, 3] = 2  # for plotting

    # Load at 2/3rd the length of the bridge
    load_nodes_3 = np.where((np.abs(node_pts[:, 0] - (np.min(node_pts[:, 0]) + 2 * bridge_length / 3)) < mesh.elem_size[0] / 2) &
                            (np.abs(node_pts[:, 1] - np.min(node_pts[:, 1])) < mesh.elem_size[1]/2))[0]
    force[3 * load_nodes_3 + 1] = -1 / len(load_nodes_3)  # y direction
    mesh.node_indices[load_nodes_3, 3] = 2  # for plotting

    # Define material properties
    mat_prop = mat_lib.create_material_with_defaults("CustomMaterial", youngs_modulus=1.0, poissons_ratio=0.3, mass_density=1.0)

    # Create boundary conditions
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    elem_body_force = None  
    return mesh, mat_prop, bc, elem_body_force

def createMultiloadProblem(nDOFDesired: int = 10000, L: float = [0.4, 0.2, 0.1]):
  # This is an example where a grid mesh is created, and a structural problem is posed on it.
  # For a perfect cube, an estimate of the number of elements is made, and a grid mesh is created.
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  load_node1 = mesh.get_nodes_from_locations([L[0], L[1]/2, L[2]/2])  
  load_dof1 = 3 * load_node1 + 1  
  load_node2 = mesh.get_nodes_from_locations([L[0]/2, L[1], L[2]/2])  
  load_dof2 = 3 * load_node2 + 1  

  force = np.zeros(3*mesh.num_nodes)
  force[load_dof1] = -1000
  force[load_dof2] = -10000
  mesh.node_indices[load_node1, 3] = 2
  mesh.node_indices[load_node2, 3] = 2 # for plotting

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createLBracketProblem(nDOFDesired: int = 10000, topload = 1000,midload = 0):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for an L-bracket
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. The problem includes fixed boundary conditions on the top
  surface and a distributed load on a portion of the right surface.
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/LBracket/LBracket.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
  triList0 = [16,17]
  fixed_nodes = mesh.get_nodes_on_triangles(triList0)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)
  if(abs(topload) > 0):
    triList1 = [12,13]
    topload_nodes = mesh.get_nodes_on_triangles(triList1)
    
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)

  if(abs(midload) > 0):
    triList2 = [4,5]
    midload_nodes = mesh.get_nodes_on_triangles(triList2)
    midload_dofs = 3 * midload_nodes + 1
    mesh.node_indices[midload_nodes, 3] = 2 # for plotting
    force[midload_dofs] = -midload/len(midload_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("SS304") 
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createGravityBarProblem(nDOFDesired: int = 10000, material_density = 7700):
  """Creates a structural problem setup for a vertical bar under gravity loading.
  This function sets up a finite element analysis problem for a vertical bar by:
  1. Reading an STL file of a vertical bar
  2. Creating a mesh with desired number of degrees of freedom
  3. Setting up fixed boundary conditions at the top surface
  4. Applying gravitational body forces
  Parameters
  ----------
  nDOFDesired : int, optional
    Desired number of degrees of freedom in the mesh (default is 10000)
  youngs_modulus : float, optional
    Young's modulus of the material in Pa (default is 2.1e11)
  poissons_ratio : float, optional  
    Poisson's ratio of the material (default is 0.28)
  material_density : float, optional
    Density of the material in kg/m³ (default is 7700)
  Returns
  -------
  tuple
    A tuple containing:
    - mesh: Mesh object with the discretized geometry
    - mat_prop: Material properties object
    - bc: Boundary conditions object
    - elem_body_force: Array of body forces on elements
  Notes
  -----
  - The STL file must be located at '../Models/VerticalBar/VerticalBar.STL'
  - The problem fixes all DOFs at the top surface (z = zMax plane)
  - Gravity acts in the negative z direction
  - SolidWorks validation shows maximum displacement of 1.7906e-9 m
  """

  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/VerticalBar/VerticalBar.STL')
  print("SolidWorks maximum displacement: 1.7906e-9 m")
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()
 
  fixed_nodes =  mesh.getNodesOnBoundingBoxPlane(2,False)  # z = zMax plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = np.zeros(3*mesh.num_elems)
  elem_body_force[2::3] = -9.81*material_density*np.prod(mesh.elem_size)    # Apply gravity in -z direction to each element

  boundaryForce = np.zeros(3*mesh.num_nodes) # no boundary force
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 
  mat_prop = mat_lib.get_material("Steel") 

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createTensilePlateProblem(nDOFDesired: int = 10000, L: float = [1.0, 1, 0.01]):
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()


  fixed_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,False))

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1
  force_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,False))
  force_dofs = np.array([3 * force_nodes])
  boundary_force = np.zeros(3*mesh.num_nodes)
  boundary_force[force_dofs] = 10000/len(force_nodes)
  elem_body_force = None
  mat_prop = mat_lib.get_material("Steel") 
 
  mesh.node_indices[force_nodes, 3] = 2 # for plotting
  bc = bound_cond.BC(force = boundary_force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

  
def createGravityPlateProblem(nDOFDesired: int = 10000, L: float = [1.0, 0.5, 0.01],
                               verticalForcePercent = 0):
  nVoxelsDesired = nDOFDesired/3    
  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))
  mesh.createEdofMatStructural()


  left_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,True), mesh.getNodesOnBoundingBoxPlane(1,True))
  right_nodes =np.intersect1d(mesh.getNodesOnBoundingBoxPlane(0,False), mesh.getNodesOnBoundingBoxPlane(1,True))
  fixed_nodes = np.union1d(left_nodes,right_nodes)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1
  elem_body_force = np.zeros(3*mesh.num_elems)
  mat_prop = mat_lib.get_material("Steel") 
  elem_body_force[1::3] = -9.81*mat_prop.mass_density*np.prod(mesh.elem_size)    # Apply gravity in -y direction to each element

  verticalForce = verticalForcePercent*np.linalg.norm(elem_body_force)
  boundary_force = np.zeros(3*mesh.num_nodes)
  node_pts = mesh.node_xyz
  # Apply a small force in the middle of the plate
  load_nodes = np.where((np.abs(node_pts[:, 0]-L[0]/2) < mesh.elem_size[0]/2) & (np.abs(node_pts[:, 1] - L[1]/2) <mesh.elem_size[1]/2))[0] # hard coded    
  load_dof = 3 * load_nodes + 1  # y direction
  boundary_force[load_dof] = -verticalForce/len(load_nodes)
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  bc = bound_cond.BC(force = boundary_force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 
  
  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------

def createArrowHeadProblem(nDOFDesired: int = 10000, totalLoad = 1000):
  """Creates a structural problem setup for an arrowhead-shaped bracket.

  This function sets up a finite element analysis problem for an arrowhead bracket by:
  1. Reading an STL file of the arrowhead geometry 
  2. Creating a mesh with desired number of degrees of freedom
  3. Setting fixed boundary conditions on the bottom surface
  4. Applying distributed load on the top surface

  Parameters
  ----------
  nDOFDesired : int, optional
    Desired number of degrees of freedom in the mesh (default is 10000)
  youngs_modulus : float, optional 
    Young's modulus of the material in Pa (default is 2.1e11)
  poissons_ratio : float, optional
    Poisson's ratio of the material (default is 0.3)
  totalLoad : float, optional
    Total applied force in N (default is 1000)

  Returns
  -------
  tuple
    A tuple containing:
    - mesh: Mesh object with the discretized geometry 
    - mat_prop: Material properties object
    - bc: Boundary conditions object

  Notes
  -----
  - The STL file must be located at '../Models/ArrowHead/ArrowHead3x3.STL'
  - The problem fixes all DOFs at z=0 plane
  - Load is distributed uniformly on the top surface (z=zMax plane)
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/ArrowHead/ArrowHead3x3.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(2,True)  # z = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
 
  load_nodes = mesh.getNodesOnBoundingBoxPlane(2,False)  # z = zMax plane
  load_dofs = 3 * load_nodes + 2
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createCompliantMechanismProblem(nDOFDesired: int = 10000, totalLoad = 1e3):
  """Creates a structural problem setup for an Compliant Mechanism 
  This function sets up a finite element mesh and boundary conditions for an Compliant Mechanism
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. 
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CompliantMechanism/CompliantMechanism.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts =mesh.node_indices[:, :3]*mesh.elem_size + mesh.origin
  fixed_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) > 20))[0] # the two end faces of the mechanism
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  load_nodes = np.where((node_pts[:, 0] == np.min(node_pts[:, 0])) & (abs(node_pts[:, 1] - 55) < 20))[0] # the middle face of the mechanism    
  load_dofs = 3 * load_nodes  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = -totalLoad/len(load_nodes)

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ---------------------------------------
  
def createBeamSurfaceLoadProblem(nDOFDesired: int = 20000, L: float = [0.1, 0.01, 0.01],totalLoad = 30000):
  # This is for large deformation
  nVoxelsDesired = int(nDOFDesired/5)
  print(nVoxelsDesired)

  # Let the number of voxels be proportional to the length in each direction
  alpha = (nVoxelsDesired/(L[0]*L[1]*L[2]))**(1/3)
  nelx = round(alpha*L[0])+1
  nely = round(alpha*L[1])
  nelz = round(alpha*L[2])
  mesh = hex_mesher.HexMesher()
  mesh.grid_mesh(num_elems = (nelx, nely, nelz),
                  elem_size = (L[0]/nelx, L[1]/nely, L[2]/nelz))

  mesh.createEdofMatStructural()
  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)

  mesh.node_indices[fixed_nodes, 3] = 1 # this is for plotting

  # line defined by x = xMax
  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane 
  load_dofs = 3 * load_nodes + 2  # z direction
  mesh.node_indices[load_nodes, 3] = 2
  load_per_dof = -totalLoad/len(load_nodes)

  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  bc = bound_cond.BC(force = force,
            fixed_dofs = fixed_dofs,
            dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force
  # ----------------------------------------


  # ----------------------------------------

def createFilletedBeamProblem(nDOFDesired=50000, totalLoad = 1):
  stl_file = os.path.join(script_dir, '../Models/FilletedBeam/FilletedBeam.STL')

  mesh = hex_mesher.HexMesher()
  nElemsDesired = round(nDOFDesired/3)    # estimate
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  fixed_nodes = mesh.getNodesOnBoundingBoxPlane(0,True) # x = 0 plane
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  load_nodes = mesh.getNodesOnBoundingBoxPlane(0,False) # x = xMax plane    
  # Get node coordinates
  node_centers = mesh.node_xyz[load_nodes]

  # Calculate center of face 
  face_center = np.mean(node_centers, axis=0)

  # Calculate torque vector for each node
  load_dofs = []
  force_values = []

  for node in load_nodes:
    node_pos = mesh.node_xyz[node] - face_center
    # Cross product with [1,0,0] to get perpendicular direction
    torque_dir = np.cross([1,0,0], node_pos)
    # Normalize
    if np.linalg.norm(torque_dir) > 0:
      torque_dir = torque_dir / np.linalg.norm(torque_dir)
      # Add force components
      load_dofs.extend([3*node + i for i in range(3)])
      force_values.extend(totalLoad/len(load_nodes) * torque_dir)
 
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = force_values

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------
  
def createCentrifugalPlateProblem(nDOFDesired: int = 10000,
                               rpm = 10000, verticalLoad = 100):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting


   # Apply centrifugal and gravity on all elements
  elem_body_force = np.zeros(3*mesh.num_elems)
  omega = 2*np.pi*rpm/60
  mat_prop = mat_lib.get_material("Steel")
  for e in range(mesh.num_elems):
    center = mesh.elem_centers[e]
    elem_body_force[3*e:3*e+2] = (mat_prop.mass_density * np.prod(mesh.elem_size)) * omega**2 * center[:2]
  
  boundaryForce = np.zeros(3*mesh.num_nodes)   
  outerRadius = 0.05
  # Apply vertical load on the outer circumference elements
  vertical_load_nodes = mesh.get_nodes_within_annular_region(centerPt, axis, outerRadius - mesh.elem_size[0] * 0.707,
                                                             outerRadius + mesh.elem_size[0] * 0.707)
  vertical_load_dofs = 3 * vertical_load_nodes + 2  # z direction
  boundaryForce[vertical_load_dofs] = -verticalLoad / len(vertical_load_nodes)

  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  
  return mesh, mat_prop, bc, elem_body_force
# ----------------------------------------

def createTorquePlateProblem(nDOFDesired: int = 10000, torque =  500):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/CircularPlateHole/CircularPlateHole.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()


  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01
  fixed_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = None
  
  outerRadius = 0.05
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply torque force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates 
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    if r > 0:
      # Unit vector in tangential direction (perpendicular to radial)
      tangent_dir = np.array([-node_pos[1], node_pos[0]])/r
      # Apply tangential force to create torque
      boundaryForce[3*node] = torque/(r*len(load_nodes)) * tangent_dir[0]
      boundaryForce[3*node + 1] = torque/(r*len(load_nodes)) * tangent_dir[1]
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  
  return mesh, mat_prop, bc, elem_body_force

# ----------------------------------------

def createThreeHoleBracketProblem(nDOFDesired: int = 10000, totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/ThreeHoleBracket/ThreeHoleBracket.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix top hole
  centerPt = [0.015,0.065,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_1 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  # fix bottom hole
  centerPt = [0.015,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_2 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  fixed_nodes = np.union1d(fixed_nodes_1,fixed_nodes_2)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = None
  
  # load on right hole
  centerPt = [0.065,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)     
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  
  return mesh, mat_prop, bc, elem_body_force


def createThreeHoleBracketThickProblem(nDOFDesired: int = 10000, totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/ThreeHoleBracket/ThreeHoleBracketThick.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix top hole
  centerPt = [0.015,0.065,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_1 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  # fix bottom hole
  centerPt = [0.015,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  fixed_nodes_2 = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)  
  
  fixed_nodes = np.union1d(fixed_nodes_1,fixed_nodes_2)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  elem_body_force = None
  
  # load on right hole
  centerPt = [0.065,0.015,0.020]
  axis = [0,0,1]
  radius = 0.005
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,radius-mesh.elem_size[0]*0.707,
                                                     radius+mesh.elem_size[0]*0.707)     
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  
  return mesh, mat_prop, bc, elem_body_force


# ----------------------------------------

def createBliskQuarterProblem(nDOFDesired: int = 10000,rpm = 0,radialForce = 0,
																	   downwardForce = 10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskQuarter.STL')


  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # fix inner radius
  centerPt = [0,0,0]
  axis = [0,0,1]
  innerRadius = 0.01085
  inner_cylinder_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,innerRadius-mesh.elem_size[0]*0.707,
                                                     innerRadius+mesh.elem_size[0]*0.707)  
  fixed_dofs = np.array([3 * inner_cylinder_nodes,
              3 * inner_cylinder_nodes + 1,
              3 * inner_cylinder_nodes + 2]).flatten().astype(int)
  

  # Find nodes on z = zMax bounding box
  xMin = np.min(mesh.node_xyz[:, 0])
  xMin_nodes = np.where(np.abs(mesh.node_xyz[:, 2] - xMin) < mesh.elem_size[2]/2)[0]
  xMin_dofs = np.array([ 3 * xMin_nodes]).flatten().astype(int)

   # Find nodes on y = yMin bounding box
  yMin = np.min(mesh.node_xyz[:, 1])
  yMin_nodes = np.where(np.abs(mesh.node_xyz[:, 1] - yMin) < mesh.elem_size[1]/2)[0]
  yMin_dofs = np.array([ 3 * yMin_nodes + 1]).flatten().astype(int)

  fixed_dofs = np.union1d(fixed_dofs, yMin_dofs)
  fixed_dofs = np.union1d(fixed_dofs, xMin_dofs)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[inner_cylinder_nodes, 3] = 1 # for plotting

  elem_body_force = np.zeros(3*mesh.num_elems)
  if (abs(rpm) > 0):
    print("Applying centrifugal force at ",rpm," rpm")
    omega = 2*np.pi*rpm/60
   
    for e in range(mesh.num_elems):
      center = mesh.elem_centers[e]
      elem_body_force[3*e:3*e+2] = (mat_prop.mass_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]

    print("total body force ",np.sum(elem_body_force[3::3]))
  # Apply centrifugal force on each node on the circumference
  # this is in addition to the body force
  outerRadius = 0.0565
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)    
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
  # Apply radial force on each node on the circumference 
  
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,1:3] # get y,z coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
   
    # Unit vector in radial direction
    radial_dir = node_pos/r
      # Add x and y dofs with force components
    boundaryForce[3*node] = radialForce/len(load_nodes)* radial_dir[1]
    boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[0]
    boundaryForce[3*node + 2] =  -downwardForce/len(load_nodes)  
    
  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  return mesh, mat_prop, bc, elem_body_force


def createGEGrabCADProblem(nDOFDesired: int = 50000, axialLoad = 10000): 
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/GEGrabCad/GEGrabCAD.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixedTri = list(range(3245, 3533))   # the two bolt-hole walls (as Models/GEGrabCad/GEGrabCAD.pyto); 3533 is the flange's end face
  fixed_nodes = mesh.get_nodes_on_triangles(fixedTri)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  
  forceTri = [5071,5072,5073,5074,5075,5076,5077,5078,5079,5080,5081,5082,5083,5084,5085,5086,5087,5088,5089,5090,5091,5092,5093,5094,5095,5096,5097,5098,5099,5100,5101,5102,5103,5104,5105,5106,5107,5108,5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119,5120,5121,5122,5123,5124,5125,5126,5127,5128,5129,5130,5131,5132,5133,5134,5135,5136,5137,5138,5139,5140,5141,5142,5143,5144,5145,5146,5147,5148,5149,5150,5151,5152,5153,5154,5155,5156,5157,5158,5159,5160,5161,5162,5163,5164,5165,5166,5167,5168,5169,5170,5171,5172,5173,5174,5175,5176,5177,5178,5179,5180,5181,5182,5183,5184,5185,5186,5187,5188,5189,5190,5191,5192,5193,5194,5195,5196,5197,5198,5199,5200,5201,5202,5203,5204,5205,5206,5207,5208,5209,5210,5211,5212,5213,5214,5215,5216,5217,5218,5219,5220,5221,5222,5223,5224,5225,5226,5227,5228,5229,5230,5231,5232,5233,5234,5235,5236,5237,5238,5239,5240,5241,5242,5243,5244,5245,5246,5247,5248,5249,5250,5251,5252,5253,5254,5255,5256,5257,5258,5259,5260,5261,5262,5263,5264,5265,5266,5267,5268,5269,5270,5271,5272,5273,5274,5275,5276,5277,5278,5279,5280,5281,5282,5283,5284,5285,5286,6351,6352,6353,6354,6355,6356,6357,6358,6359,6360,6361,6362,6363,6364,6365,6366,6367,6368,6369,6370,6371,6372,6373,6374,6375,6376,6377,6378,6379,6380,6381,6382,6383,6384,6385,6386,6387,6388,6389,6390,6391,6392,6393,6394,6395,6396,6397,6398,6399,6400,6401,6402,6403,6404,6405,6406,6407,6408,6409,6410,6411,6412,6413,6414,6415,6416,6417,6418,6419,6420,6421,6422,6423,6424,6425,6426,6427,6428,6429,6430,6431,6432,6433,6434,6435,6436,6437,6438,6439,6440,6441,6442,6443,6444,6445,6446,6447,6448,6449,6450,6451,6452,6453,6454,6455,6456,6457,6458,6459,6460,6461,6462,6463,6464,6465,6466,6467,6468,6469,6470,6471,6472,6473,6474,6475,6476,6477,6478,6479,6480,6481,6482,6483,6484,6485,6486,6487,6488,6489,6490,6491,6492,6493,6494,6495,6496,6497,6498,6499,6500,6501,6502,6503,6504,6505,6506,6507,6508,6509,6510,6511,6512,6513,6514,6515,6516,6517,6518,6519,6520,6521,6522,6523,6524,6525,6526,6527,6528,6529,6530,6531,6532,6533,6534,6535,6536,6537,6538,6539,6540,6541,6542,6543,6544,6545,6546,6547,6548,6549,6550,6551,6552,6553,6554,6555,6556,6557,6558,6559,6560,6561,6562,6563,6564,6565,6566]
  load_nodes = mesh.get_nodes_on_triangles(forceTri)
  
  
  load_dofs = 3 * load_nodes   # x direction

  load_per_dof = axialLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof

  mat_prop = mat_lib.get_material("Nitronic60")

  nElems = mesh.num_elems
  elemVolume = mesh.elem_size[0]*mesh.elem_size[1]*mesh.elem_size[2]
  totalMass = nElems * elemVolume * mat_prop.mass_density
  print("Total mass of GEGrabCAD: {:.2f} kg".format(totalMass))
  elem_body_force = None

  # All constraints are implemented using the constraint matrix
  # Therefore fixed_dofs and dirichlet_values are empty
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  return mesh, mat_prop, bc, elem_body_force


# ----------------------------------------

def createBliskSectionProblem(nDOFDesired: int = 50000, rpm = 0, radialForce =0, downwardForce = 10000 ): 
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskSection.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()

  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  fixedTri = [1354,1355]
  
  fixed_nodes = mesh.get_nodes_on_triangles(fixedTri)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten() # This is needed if the dofs are being retained for TO
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  C0 = lil_matrix((3*len(fixed_nodes), mesh.num_nodes * 3))
  for i, node in enumerate(fixed_nodes):
    C0[3*i, 3*node] = 1
    C0[3*i+1, 3*node+1] = 1
    C0[3*i+2, 3*node+2] = 1
  C0 = C0.tocsr()
  
  
  # Nodes on these triangles are to subject to sliding boundary condition
  # i.e. d.n = 0 where n is the normal to the surface of the triangle
  triSet1 = [1321, 1322, 1323, 1324, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684]
  sliding_nodes_1 = mesh.get_nodes_on_triangles(triSet1)
  normal_1 = mesh.stlGeom.get_triangle_normal(triSet1[0])
  
  udof1 = 3 * sliding_nodes_1
  vdof1 = 3 * sliding_nodes_1 + 1
  wdof1 = 3 * sliding_nodes_1 + 2
  # Create constraint matrix for sliding boundary conditions
  # First surface normal constraint
  C1 = lil_matrix((len(sliding_nodes_1), mesh.num_nodes * 3))
  for i, node in enumerate(sliding_nodes_1):
    C1[i, udof1[i]] = normal_1[0]
    C1[i, vdof1[i]] = normal_1[1]
    C1[i, wdof1[i]] = normal_1[2]
  C1 = C1.tocsr()

  triSet2 = [1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804]
  sliding_nodes_2 = mesh.get_nodes_on_triangles(triSet2)
  udof2 = 3 * sliding_nodes_2
  vdof2 = 3 * sliding_nodes_2 + 1 
  wdof2 = 3 * sliding_nodes_2 + 2
  # Second surface normal constraint
  C2 = lil_matrix((len(sliding_nodes_2), mesh.num_nodes * 3))
  normal_2 = mesh.stlGeom.get_triangle_normal(triSet2[0])
  for i, node in enumerate(sliding_nodes_2):
    C2[i, udof2[i]] = normal_2[0]
    C2[i, vdof2[i]] = normal_2[1]
    C2[i, wdof2[i]] = normal_2[2]
  C2 = C2.tocsr()

  # Combine constraints
  constraint_matrix = spy_sprs.vstack((spy_sprs.csr_matrix(C0), 
                       spy_sprs.csr_matrix(C1),
                       spy_sprs.csr_matrix(C2)))
  constraint_rhs = np.zeros(3*len(fixed_nodes) + len(sliding_nodes_1) + len(sliding_nodes_2))

  total_mesh_volume = np.prod(mesh.elem_size) * mesh.num_elems # * 0.0283168 # ft3 to m3
  print("total mesh volume in m3",total_mesh_volume)
  mat_prop=mat_lib.get_material("Steel")
  material_density = mat_prop.mass_density # * 16.0185 # lb/ft3 to kg/m3
  total_mass = material_density * total_mesh_volume
  print("total mass in kg",total_mass)


  elem_body_force = None
  if (abs(rpm) > 0):
    elem_body_force = np.zeros(3*mesh.num_elems)
    print("Applying centrifugal force at ",rpm," rpm")
    omega = 2*np.pi*rpm/60
    for e in range(mesh.num_elems):
      center = mesh.elem_centers[e]
      # Add centrifugal force to each element in xy plane
      elem_body_force[3*e:3*e+2] = (material_density*np.prod(mesh.elem_size)) * omega**2 *  center[:2]

    print("total body force ",np.linalg.norm(elem_body_force))

  axis = [0,0,1] # z-axis
  centerPt = [0,0,0] # center of the blisk section
  outerRadius = 0.0565
  load_nodes = mesh.get_nodes_within_annular_region(centerPt,axis,outerRadius-mesh.elem_size[0]*0.707,
                                                    outerRadius+mesh.elem_size[0]*0.707)  
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  boundaryForce = np.zeros(3*mesh.num_nodes) 
 
  print("Applying radial force of ",radialForce," N on ", len(load_nodes), " nodes on outer circumference")
  # Apply radial force on each node on the circumference 
  for node in load_nodes:
    node_pos = mesh.node_xyz[node,:2] # get x,y coordinates
    r = np.sqrt(np.sum(node_pos**2)) # distance from center
    # Unit vector in radial direction
    radial_dir = node_pos/r
    # Add x and y dofs with force components
    boundaryForce[3*node] = radialForce/len(load_nodes) * radial_dir[0]  
    boundaryForce[3*node + 1] = radialForce/len(load_nodes) * radial_dir[1]
    boundaryForce[3*node + 2] = downwardForce/len(load_nodes)
  
  print("Total applied radial force ",np.sum(boundaryForce[0::3]),np.sum(boundaryForce[1::3]))
  

  mesh.node_indices[sliding_nodes_1, 3] = 1 # for plotting
  mesh.node_indices[sliding_nodes_2, 3] = 1 # for plotting

  # All constraints are implemented using the constraint matrix

  bc = bound_cond.BC(force = boundaryForce,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values
                     ,constraint_matrix=constraint_matrix,constraint_rhs=constraint_rhs) 

  return mesh, mat_prop, bc, elem_body_force

  # ----------------------------------------


def createBliskPressureLoadingProblem(nDOFDesired: int = 50000, pressure = 1000000, loadingMode = 2):
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
    stl_file = os.path.join(script_dir, '../Models/BliskModel/BliskSection.STL')

    stl_geom = STLGeom(stl_file)
    [area, stl_volume, cg, inertia] = stl_geom.compute_mass_properties()
  
    print("STL Volume: ", stl_volume)

    nElemsDesired = nDOFDesired/3    # estimate
    mesh = hex_mesher.HexMesher()

    mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
    mesh.createEdofMatStructural()
    mesh_vol = mesh.num_elems * np.prod(mesh.elem_size)
    print("Mesh Volume: ", mesh_vol)
    print("Vol error: ", (mesh_vol - stl_volume)/stl_volume * 100, "%")
  
    # Fixed boundary condition at x = xMin plane (all DOFs fixed)
  # Find all nodes with x coordinate close to xMin
    xMin = np.min(mesh.node_xyz[:, 0])
    fixed_nodes = np.where(np.abs(mesh.node_xyz[:, 0] - xMin) < mesh.elem_size[0]/2)[0]

    fixed_dofs = np.array([3 * fixed_nodes,
                3 * fixed_nodes + 1,
                3 * fixed_nodes + 2]).flatten().astype(int)
    
    mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
    dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
   
    # load on the blade surface
    bladeStartRadius = 0.0575
    bladeEndRadius = 0.0725
    
    triList = []
    for t in range(stl_geom.stl_n_triangles):
        normal = stl_geom.tri_normals[t]
        center = stl_geom.get_triangle_center(t)
        if np.dot(normal, [0, 1,0]) > 0.05:  # Check if normal is in +y direction
            # Check if the triangle center is within the annular region
            if (bladeStartRadius <= np.linalg.norm(center[:2]) <= bladeEndRadius):
                triList.append(t)
  
    centerPt = [0, 0, 0]  # Center of the blisk section
    axis = [0, 0, 1]  # z-axis
    blade_nodes = mesh.get_nodes_within_annular_region(centerPt, axis, bladeStartRadius,
                                                      bladeEndRadius)
    
    blade_vertices = mesh.node_xyz[blade_nodes]
    
    force = np.zeros(3*mesh.num_nodes)
    print("Computing pressure force on blade nodes...")
    for tri in triList:
        area = stl_geom.tri_areas[tri]
        normal = stl_geom.tri_normals[tri]
        if (loadingMode == 1):
            # Apply pressure force in  -z direction
            forceTri= -pressure * area * np.array([0,0,1])
        elif (loadingMode == 2):
            # Apply pressure force in the normal direction
            forceTri = -pressure * area * np.array(normal)
        elif (loadingMode == 3):
            # Apply pressure force in the y direction, proportional to y component of normal
            forceTri = -pressure * area * np.array([0,1, 0])*np.dot(normal, [0, 1, 0])
        elif (loadingMode == 4):
            # Apply pressure force with random scaling factor
            x, y, z = stl_geom.get_triangle_center(tri)
            random_factor = (np.exp((x-bladeStartRadius)/bladeStartRadius)-0.9)
            forceTri = -pressure * area * np.array([0,0,1]) * random_factor
        else:
            raise ValueError("Invalid loading mode.")

        distances = stl_geom.find_points_triangle_distances_vectorized(blade_vertices, tri)
        # Find nodes close to the triangle and distribute force among them
        close_nodes_indices = np.where(distances < mesh.elem_size[0]*0.707)[0]
        if len(close_nodes_indices) > 0:
          # Get the actual node indices from the blade_nodes array
          close_nodes = blade_nodes[close_nodes_indices]
          # Distribute force equally among close nodes
          mesh.node_indices[close_nodes, 3] = 2 # for plotting
          for node_idx in close_nodes:
            force[3*node_idx:3*node_idx+3] += forceTri / len(close_nodes)
    
  
    bc = bound_cond.BC(force=force, fixed_dofs=fixed_dofs, dirichlet_values=dirichlet_values)
    # Calculate and print total force in each direction
    total_force_x = np.sum(force[0::3])
    total_force_y = np.sum(force[1::3])
    total_force_z = np.sum(force[2::3])
    print(f"Total force in x direction: {total_force_x:.4e} N")
    print(f"Total force in y direction: {total_force_y:.4e} N")
    print(f"Total force in z direction: {total_force_z:.4e} N")
    print(f"Total force magnitude: {np.sqrt(total_force_x**2 + total_force_y**2 + total_force_z**2):.4e} N")
    mat_prop = mat_lib.get_material("Steel")
    elem_body_force = None
    return mesh, mat_prop, bc, elem_body_force


def createKnuckleAssemblyProblem(nDOFDesired: int = 10000, youngs_modulus = [2e11,2e11], 
                               poissons_ratio = [0.28,0.28], totalLoad =  10000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/KnuckleAssembly/KnuckleAssembly.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  fixed_nodes_1 = np.where(node_pts[:, 0] == np.min(node_pts[:, 0]))[0]
  fixed_nodes_2 = np.where(node_pts[:, 0] == np.max(node_pts[:, 0]))[0]
  fixed_nodes = np.union1d(fixed_nodes_1,fixed_nodes_2)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]))[0]       
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  # There are 2 components in the assembly
  # Assign material properties to each component
  steel = mat_lib.get_material("Steel")
  aluminum = mat_lib.get_material("Aluminum")
  mat_prop = [steel if cid == 1 else aluminum for cid in mesh.elemComponentId]

  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force


def createTableProblem(nDOFDesired: int = 10000, totalLoad =  100000):
 
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/Table/Table.STL')

  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  node_pts = mesh.node_xyz
  fixed_nodes = np.where(node_pts[:, 1] == np.min(node_pts[:, 1]))[0]

  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting
  
  load_nodes = np.where(node_pts[:, 1] == np.max(node_pts[:, 1]))[0]       
  
  mesh.node_indices[load_nodes, 3] = 2 # for plotting
  load_dofs = 3 * load_nodes + 1  # y direction

  load_per_dof = -totalLoad/len(load_nodes)
  force = np.zeros(3*mesh.num_nodes)
  force[load_dofs] = load_per_dof
  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  
  elem_body_force = None
  return mesh, mat_prop, bc, elem_body_force



def createLBracketThickProblem(nDOFDesired: int = 80000,topload = 1000,midload = 0):
  """Creates a structural problem setup for an L-bracket topology optimization.
  This function sets up a finite element mesh and boundary conditions for an L-bracket
  structural problem from an STL file. The mesh is created with approximately the desired
  number of degrees of freedom. The problem includes fixed boundary conditions on the top
  surface and a distributed load on a portion of the right surface.
  Args:
    nDOFDesired (int, optional): Desired number of degrees of freedom for the mesh. 
                  Defaults to 10000.
  Returns:
    tuple: A tuple containing:
      - mesh (Mesher): Mesh object with the L-bracket discretization
      - mat_prop (StructuralMaterial): Material properties object with structural parameters
      - bc (BC): Boundary conditions object with forces and constraints
  Notes:
    - The mesh is created from an STL file located at '../Models/LBracket/LBracket.STL'
    - Fixed boundary conditions are applied at y = yMax
    - Load is applied in the -y direction on nodes where y > 0.039 and x > 0.09
    - Total applied load is 1000 units distributed equally among loaded nodes
    - Material properties are set to E = 2.1e5 and ν = 0.3
  """
  # Read the STL model, create a mesh of desired size, and a structural problem is posed on it.
  stl_file = os.path.join(script_dir, '../Models/LBracketThick/LBracketThick.STL')
  nElemsDesired = nDOFDesired/3    # estimate
  mesh = hex_mesher.HexMesher()
  
  mesh.createMeshFromSTLFile(stl_file, nElemsDesired=nElemsDesired)
  mesh.createEdofMatStructural()

  # Supports/loads are selected by GEOMETRY, not by hard-coded STL triangle index.
  # The model was re-tessellated (32 -> 596 triangles) and the old indices [16,17],
  # [12,13], [4,5] then silently pointed at unrelated patches of the y = 0.5 surface
  # (5 colinear fixed nodes instead of the whole top face) -- a different problem.
  vecs = mesh.stlGeom.mesh.vectors  # (n_triangles, 3 vertices, xyz)
  x, y, z = vecs[..., 0], vecs[..., 1], vecs[..., 2]
  eps = 1e-4
  fixed_tris = np.where((np.abs(y - y.max()) < eps).all(axis=1))[0]  # top face, y = yMax
  fixed_nodes = mesh.get_nodes_on_triangles(fixed_tris)
  fixed_dofs = np.array([3 * fixed_nodes,
              3 * fixed_nodes + 1,
              3 * fixed_nodes + 2]).flatten().astype(int)
  dirichlet_values = 0*np.ones_like(fixed_dofs, dtype = float)
  mesh.node_indices[fixed_nodes, 3] = 1 # for plotting

  force = np.zeros(3*mesh.num_nodes)
  if(abs(topload) > 0):
    # narrow strip at the end of the foot: top surface y = 0.5, x >= 0.975
    topload_tris = np.where((np.abs(y - 0.5) < eps).all(axis=1) & (x >= 0.975 - eps).all(axis=1))[0]
    topload_nodes = mesh.get_nodes_on_triangles(topload_tris)
    topload_dofs = 3 * topload_nodes + 1  
    mesh.node_indices[topload_nodes, 3] = 2 # for plotting
    force[topload_dofs] = -topload/len(topload_nodes)
  if(abs(midload) > 0):
    # narrow strip at mid-height of the foot's end face: x = xMax, |y - 0.25| <= 0.0125
    midload_tris = np.where((np.abs(x - x.max()) < eps).all(axis=1) & (np.abs(y - 0.25) <= 0.0125 + eps).all(axis=1))[0]
    midload_nodes = mesh.get_nodes_on_triangles(midload_tris)
    midload_dofs = 3 * midload_nodes + 1
    mesh.node_indices[midload_nodes, 3] = 2 # for plotting
    force[midload_dofs] = -midload/len(midload_nodes)
    

  bc = bound_cond.BC(force = force,fixed_dofs = fixed_dofs,dirichlet_values = dirichlet_values) 

  mat_prop = mat_lib.get_material("Steel")
  elem_body_force = None

  return mesh, mat_prop, bc, elem_body_force


