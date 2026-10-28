"""Optimization routines for topology optimization."""
import enum
import numpy as np
from pyto.topopt.filters import *
import matplotlib.pyplot as plt
import pyto.solve.numpy_backend as lin_solv
import pyto.core.hex_mesher as hex_mesher
import pyto.physics.structural.hex_structural_fea as hex_structural_fea
import pyto.physics.hex_element_stiffness as hex_element_stiffness
from pyto.autodiff.material_model import *
from dataclasses import dataclass, field
from typing import Tuple
import pyto.physics.thermal.hex_thermal_fea as hex_thermal_fea 
import pyto.solve.deflation as deflation
import pyto.solve.numpy_backend as linear_solvers
from pyto.autodiff.material_model import *

DIRECT_SOLVER_DOF_CUTOFF = 100000 #  dof limit for direct solver, for greater number of dof, iterative solver is used


class TO_METHODS(enum.Enum):
	DENSITYMMA = enum.auto()

class GRADIENT_METHOD(enum.Enum):  # how MMA gets the sensitivities
	AUTODIFF = "autodiff"      # torch autograd through the FE solve (adjoint); any objective/constraint
	ANALYTICAL = "analytical"  # derived (adjoint) sensitivities: compliance, p-norm stress, volume fraction


def gradient_method(value) -> "GRADIENT_METHOD":
	"""GRADIENT_METHOD from the enum or its name/value ("autodiff", "analytical"; "manual" is the old name of
	"analytical" and is still accepted, e.g. in saved specs). None means AUTODIFF."""
	if value is None or isinstance(value, GRADIENT_METHOD):
		return value or GRADIENT_METHOD.AUTODIFF
	key = str(value).strip().lower()
	if key == "manual":
		return GRADIENT_METHOD.ANALYTICAL
	for m in GRADIENT_METHOD:
		if key in (m.value, m.name.lower()):
			return m
	raise ValueError(f"Unknown gradient method {value!r}; use GRADIENT_METHOD.AUTODIFF or GRADIENT_METHOD.ANALYTICAL.")


class FEA_MODE(enum.Enum):
	STRUCTURAL = enum.auto()
	THERMAL = enum.auto()
	THERMO_STRUCTURAL = enum.auto()

class TO_QOI(enum.Enum): # Topology optimization; Various Quantity of Interest
	VOLUME_FRACTION = enum.auto() # Volume fraction
	VOLUME_FRACTION_MIN = enum.auto() # Lower bound on volume fraction (constraint: 1 - mean(x)/limit <= 0); e.g. keeps self-weight problems from collapsing
	VOLUME = enum.auto() # Volume total
	MASS = enum.auto() # Mass total
	COMPLIANCE = enum.auto()
	PNORM_STRESS = enum.auto()
	MAX_VONMISES_STRESS = enum.auto()
	STRESS_FAILURE_FACTOR = enum.auto() # opposite of safety factor
	GVECTOR = enum.auto() # g'* u
	GFUNCTION = enum.auto() # user-defined torch function fn(sol, x, fe_solver); MMA only (see autodiff/qoi/user_function.py)
	COST = enum.auto() # Generic cost function
	TEMPERATURE_FAILURE_FACTOR = enum.auto() 
	MAX_CRITICALITY = enum.auto() # Captures availability of material
	MEAN_CRITICALITY = enum.auto() # Captures availability of material
	FATIGUE_FAILURE_FACTOR = enum.auto() # opposite of safety factor for fatigue
	PBR = enum.auto() # Oxidation index or MMTO


@dataclass(slots=True) # avoid accidental modification
class TOParams: # These are the default parameters
    Comment: str = "" # Comment for the topology optimization problem
    Objective: Tuple[TO_QOI, None] = (TO_QOI.COMPLIANCE,None) # Tuple of objective type and auxiliary scalar/vector/function	
    Constraints: list[Tuple[TO_QOI, None, float]] = field(
        default_factory=lambda: [(TO_QOI.VOLUME_FRACTION, None, 0.5)]
    )# Collection of tuples of constraint type, auxiliary scalar/vector/function, and upper bound
    nDOFDesired: int = 25000 # Desired number of degrees of freedom in the finite element problem
    APPLY_FILTER_TO_DENSITY: bool = True # Apply filter to density
    HeavisideProjection: bool = False # Project the filtered density with a smooth Heaviside step (MMA driver); beta
    # starts at 1 and doubles every HeavisideBetaInterval iterations up to HeavisideBetaMax (Ooms et al., Sect. 3.4)
    HeavisideBetaInterval: int = 50
    HeavisideBetaMax: float = 32.0
    RelativeFilterRadius: float = 1.5 #relative to the element size
    XSymmetry: bool = False # Desired symmetry in YZ plane
    YSymmetry: bool = False
    ZSymmetry: bool = False
    XAxisAngularSymmetry: int = 0 # Desired symmetry sectors about X axis
    YAxisAngularSymmetry: int = 0
    ZAxisAngularSymmetry: int = 0
    ExtrudeX: bool = False # Should the design be extrudable in X direction
    ExtrudeY: bool = False
    ExtrudeZ: bool = False
    KeepFixedElems: bool = False # Should the elements with Dirichlet dof be retained?
    RemoveHangingElems: bool = False # Should the hanging elements be removed?
    AMBuildDir: str = '' # Direction of AM build, '','X','Y','Z'
    ElemsToKeep: list[int] = None # List of additional elements to retain in the design
    MaxIterations: int = 150 # Maximum number of iterations
    Enforce_Constraints_MMA: bool = False # Should the constraints be enforced more strongly in GCMMA?
    Eliminate_Hanging_Elements: bool = False # Should the hanging elements be eliminated after optimization?
    MaterialsExcelFile: str = '' # Path to the Excel file containing material properties for MMTO problems
    materialModel: MaterialModel = MaterialModel.SIMP # Material model for density based topology optimization
    # What the user wrote, for progress printing (set by compile_spec; None for benchmark problems):
    # ObjectiveLabel = (expression, "minimize"/"maximize"), ConstraintLabels = [(expression, "<="/">=", bound), ...]
    ObjectiveLabel: tuple = None
    ConstraintLabels: list = None
    Gradient: GRADIENT_METHOD = GRADIENT_METHOD.AUTODIFF # MMA sensitivities: AUTODIFF (autograd) or ANALYTICAL
    # (topopt/manual_sensitivities.py: compliance, p-norm stress, volume fraction); strings are accepted too
    # Thermo-mechanical problems only (see physics/thermoelastic/thermostructural_fea.py)
    ThermalReferenceTemperature: float = 23.0 # Stress-free temperature of the structure
    ConductivityPenalty: float = None # SIMP exponent of the conductivity (None = module default)
    ConductivityVoidRatio: float = None # Void conductivity relative to solid (None = module default)
    stress_scaling: float = 1.0 # Adaptive stress-normalization factor for MAX_VONMISES_STRESS/STRESS_FAILURE_FACTOR
    # constraints. Lives here (per optimization run) rather than as a module
    # global or a mutable function attribute, so two separate optimization
    # runs in the same process don't silently share/carry over this state.

def find_elements_with_forces(mesh: hex_mesher.HexMesher, force,nDOFPerNode) -> np.ndarray:
	"""Find all elements that have nodes on which force has been applied.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with applied forces.
	"""
	force_dofs = np.where(force != 0)[0]
	forced_nodes = set(force_dofs // nDOFPerNode)  # Convert DOFs to node indices
	elements_with_forces = []

	# Vectorized approach: check which elements contain forced nodes
	elem_nodes = mesh.elemArray  # Shape: (num_elems, nodes_per_elem)
	# Create a mask where True indicates a forced node
	is_forced = np.isin(elem_nodes, list(forced_nodes))
	# Find elements that have at least one forced node
	elements_with_forces = np.where(is_forced.any(axis=1))[0]

	return np.array(elements_with_forces)

    
def find_elements_with_fixedDOF(mesh, bc,nDOFPerNode ) -> np.ndarray:
	"""Find all elements that have nodes with fixed degrees of freedom.
	
	Args:
		mesh: The mesh object.
		bc: The boundary conditions object.
	
	Returns:
		Array of element indices that have nodes with fixed degrees of freedom.
	"""
	fixed_dofs = bc.fixed_dofs
	fixed_nodes = set(fixed_dofs // nDOFPerNode)  # Convert DOFs to node indices
	elements_with_fixed_dofs = []

	# Vectorized approach: check which elements contain fixed nodes
	elem_nodes = mesh.elemArray  # Shape: (num_elems, nodes_per_elem)
	# Create a mask where True indicates a fixed node
	is_fixed = np.isin(elem_nodes, list(fixed_nodes))
	# Find elements that have at least one fixed node
	elements_with_fixed_dofs = np.where(is_fixed.any(axis=1))[0]

	return np.array(elements_with_fixed_dofs)


def createFilters(fe_solver: hex_structural_fea.HexStructuralFEA,to_params):
	# Create  filters
	H, Hs = createSmoothingFilter(fe_solver.mesh, rel_filter_radius=to_params.RelativeFilterRadius)
	# Accumulate all other filters
	if to_params.XSymmetry:
		HX = createXSymmetryFilter(fe_solver.mesh)
		H = H*HX
	if to_params.YSymmetry:
		HY = createYSymmetryFilter(fe_solver.mesh)
		H = H*HY
	if to_params.ZSymmetry:
		HZ = createZSymmetryFilter(fe_solver.mesh)
		H = H*HZ
	if to_params.XAxisAngularSymmetry > 0:
		HAAX = createXAngularSymmetryFilter(fe_solver.mesh, to_params.XAxisAngularSymmetry)
		H = H*HAAX
	if to_params.YAxisAngularSymmetry > 0:
		HAAY = createYAngularSymmetryFilter(fe_solver.mesh, to_params.YAxisAngularSymmetry)
		H = H*HAAY
	if to_params.ZAxisAngularSymmetry >	0:
		HAZ = createZAngularSymmetryFilter(fe_solver.mesh, to_params.ZAxisAngularSymmetry)
		H = H*HAZ
	if (to_params.ExtrudeY):
		HEY = createYExtrudeFilter(fe_solver.mesh)
		H = H*HEY
	if (to_params.ExtrudeX):
		HEX = createXExtrudeFilter(fe_solver.mesh)
		H = H*HEX
	if (to_params.ExtrudeZ):
		HEZ = createZExtrudeFilter(fe_solver.mesh)
		H = H*HEZ
	Hs = np.array(H.sum(1)).squeeze()
	return H, Hs
