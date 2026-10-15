"""Legacy, hand-derived NumPy discrete-adjoint sensitivity formulas.

TEST-ORACLE ONLY. Nothing in pyto.topopt (the optimizer drivers) imports
from this module -- it exists so tests/ can verify the torch-native QOI
functions in pyto.autodiff.qoi agree with an independently-derived
analytic formula. See plan Phase 4: before this split, these functions
and their torch-native twins lived in the same file
(topopt_obj_cons_sensitivities.py) and were called inconsistently by
different drivers, which is what caused most of the bugs found in the
initial analysis of this codebase.

Formulas are unchanged from before the split (only import locations
changed) -- they are the ground truth these files exist to preserve.
"""
import numpy as np

from pyto.autodiff.material_model import (
    _PNORM_EXPONENT,
    get_structural_material_model_scaling,
    get_thermal_material_model_scaling,
    get_structural_material_model_sensitivity,
    get_thermal_material_model_sensitivity,
)
import pyto.solve.numpy_backend as linear_solvers


def compute_volume_constraint_and_gradient(x: np.ndarray,
											 volfracUpper: float,
											 )-> np.ndarray:
	"""Compute the volume constraint.

	Args:
		density: Array of (num_elems,) containing the element densities.
		volfrac: The target volume fraction.

	Returns: The volume constraint. The constraint is satisfied when the
		returned value is zero. The constraint is inactive when the returned
		value is negative.
	"""

	volConstraint = ((np.mean(x)/volfracUpper) - 1.0)
	volConstraint_gradient = np.ones_like(x) / volfracUpper/ x.size
	return volConstraint, volConstraint_gradient


def compute_compliance(sol: np.ndarray, x: np.ndarray,
				fe_solver, KE,
				material_model = None) -> np.ndarray:
	"""Compute the  compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)

	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")

	compliance = np.sum(materialScaling * ce)

	return compliance


def compute_compliance_and_gradient(sol: np.ndarray, x: np.ndarray,
				fe_solver, KE,
				material_model = None) -> np.ndarray:
	"""Compute the  compliance objective.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(sol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		materialScaling = get_structural_material_model_scaling(x, material_model)
		compliance_grad = -get_structural_material_model_sensitivity(x,material_model) * ce

	elif (nRows == 8): # thermal hex
		materialScaling = get_thermal_material_model_scaling(x, material_model)
		compliance_grad = -get_thermal_material_model_sensitivity(x,material_model) * ce
	else:
		raise ValueError("Invalid number of rows in element stiffness matrix.")

	compliance = np.sum(materialScaling * ce)
	return compliance, compliance_grad


def compute_pnorm_stress_and_sensitivity(sol: np.ndarray, x, fe_solver, KE, material_model):
	"""p-norm von Mises stress, its gradient w.r.t. x, and the max von Mises stress.

	Same definition as autodiff/qoi/stress.py (strain at the element centre, relaxed stress
	(eps + (1 - eps) x^0.5) D B u_e, P = (sum vm_e^p)^(1/p)); discrete adjoint as in
	"An efficient 146-line 3D sensitivity analysis code of stress based topology optimization written
	in MATLAB", Optim. Eng. (2022) 23:1733-1757. Structural (no thermal strain) only.
	The adjoint system is solved with fe_solver.adjoint_solve (the last solve's matrix and factors) when the solver
	has one; otherwise fe_solver.stiff_mtrx must be the SciPy stiffness of the last solve.
	"""
	from pyto.autodiff.material_model import _EVOID_RELATIVE
	mesh = fe_solver.mesh
	qStress, p = 0.5, float(_PNORM_EXPONENT)
	E, nu = float(fe_solver.mat_prop.youngs_modulus), float(fe_solver.mat_prop.poissons_ratio)
	D = E / ((1 + nu) * (1 - 2 * nu)) * np.array([
		[1 - nu, nu, nu, 0, 0, 0], [nu, 1 - nu, nu, 0, 0, 0], [nu, nu, 1 - nu, 0, 0, 0],
		[0, 0, 0, (1 - 2 * nu) / 2, 0, 0], [0, 0, 0, 0, (1 - 2 * nu) / 2, 0], [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]])
	gradN = (1 / 8) * np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
								[-1, -1, 1, 1, -1, -1, 1, 1],
								[-1, -1, -1, -1, 1, 1, 1, 1]]) * (2.0 / np.asarray(mesh.elem_size, float))[:, None]
	B = np.zeros((6, 24))                      # strain = B u_e, u_e ordered (u, v, w) per node
	u, v, w = np.arange(8) * 3, np.arange(8) * 3 + 1, np.arange(8) * 3 + 2
	B[0, u], B[1, v], B[2, w] = gradN[0], gradN[1], gradN[2]
	B[3, u], B[3, v] = gradN[1], gradN[0]
	B[4, u], B[4, w] = gradN[2], gradN[0]
	B[5, v], B[5, w] = gradN[2], gradN[1]
	M = np.zeros((6, 6))                       # vm^2 = s^T M s
	M[:3, :3] = [[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]]
	M[3, 3] = M[4, 4] = M[5, 5] = 3.0

	x = np.maximum(np.asarray(x, float), 1e-12)
	ue = sol[mesh.edofMat]                     # (n, 24)
	sigma = ue @ (D @ B).T                     # unrelaxed stress (n, 6)
	c = _EVOID_RELATIVE + (1 - _EVOID_RELATIVE) * x ** qStress
	s = c[:, None] * sigma
	vm2 = np.einsum("ei,ij,ej->e", s, M, s)
	vm = np.sqrt(vm2 + 1e-16)
	P = np.sum(vm ** p) ** (1 / p)
	dP_dvm = P ** (1 - p) * vm ** (p - 1)
	# explicit: vm is linear in c, dvm/dc = vm2 / (c vm)
	explicit = dP_dvm * vm2 / (c * vm) * (1 - _EVOID_RELATIVE) * qStress * x ** (qStress - 1)
	# adjoint: dP/du_e = dP/dvm * c B^T D^T M s / vm
	g_e = (dP_dvm * c / vm)[:, None] * ((s @ M.T) @ (D @ B))
	g = np.zeros(fe_solver.bc.num_dofs)
	np.add.at(g, mesh.edofMat, g_e)
	if hasattr(fe_solver, "adjoint_solve"):
		adjointSol = fe_solver.adjoint_solve(g)
	else:
		adjointSol = linear_solvers.solve(fe_solver.stiff_mtrx, g, fe_solver.solver, fe_solver.bc,
										  dsolver=fe_solver.dsolver, **fe_solver.kwargs)
	ce = ((adjointSol[mesh.edofMat] @ KE) * ue).sum(1)
	implicit = -get_structural_material_model_sensitivity(x, material_model) * ce
	return P, explicit + implicit, vm.max()


def compute_solution_dotproduct_and_gradient(sol: np.ndarray, x,fe_solver,KE,material_model,g: np.ndarray,
				) -> np.ndarray:
	"""Compute the objective g'* sol, and its gradient.

	Args:
		density: Array of (num_elems,) containing the element densities.
		fe_solver: The structural FEA solver object.
		penal: The penalization factor for the SIMP method.

	Returns: The compliance objective value.
	"""
	obj = np.dot(sol, g)


	adjointSol =  -linear_solvers.solve(fe_solver.stiff_mtrx,
                      g,
                      fe_solver.solver,
                      fe_solver.bc,
                      dsolver = fe_solver.dsolver,
                      **fe_solver.kwargs)

	dofMat = fe_solver.mesh.edofMat
	num_elems = fe_solver.mesh.num_elems
	nRows = KE.shape[0]
	ce = (np.dot(adjointSol[dofMat].reshape(num_elems, nRows), KE) * sol[dofMat].reshape(num_elems, nRows)).sum(1)

	if (nRows == 24): # structural hex
		compliance_grad = get_structural_material_model_sensitivity(x,material_model) * ce

	elif (nRows == 8): # thermal hex
		compliance_grad = get_thermal_material_model_sensitivity(x,material_model) * ce
	return obj, compliance_grad


# NOTE: the topological-sensitivity functions that used to live here
# (compute_objective_topological_sensitivity_compliance,
# computeStructuralTopologicalSensitivity, compute_pnorm_stress_and_TS,
# computeThermalTopologicalSensitivity) moved to
# pyto.topopt.topological_sensitivity in Phase 5 (they are not NumPy twins
# of torch-native functions, so they do not belong with the test-oracle-only
# functions above).
