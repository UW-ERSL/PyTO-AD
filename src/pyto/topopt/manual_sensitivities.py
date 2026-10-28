"""Analytical (derived, "manual") gradients for the MMA driver, as an alternative to automatic differentiation.

to_params.Gradient = GRADIENT_METHOD.ANALYTICAL makes MMA use these instead of torch.autograd. They exist only where a derived
sensitivity is available and verified against autograd (tests/test_manual_sensitivities.py):

  objective    COMPLIANCE (structural, thermal, thermo-structural), PNORM_STRESS (structural), VOLUME_FRACTION
  constraints  the same, VOLUME_FRACTION as <= or >= (VOLUME_FRACTION_MIN)

Everything else (mass, max stress, failure factor, user expressions, Python functions) needs automatic
differentiation; so do design-dependent body forces and per-problem conductivity laws. manual_support() says why.
Values are computed by the same functions as the autograd path, so both modes optimize exactly the same problem.
"""
import contextlib
import io

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.topopt.common import TO_QOI

MANUAL_OBJECTIVES = {TO_QOI.COMPLIANCE, TO_QOI.PNORM_STRESS, TO_QOI.VOLUME_FRACTION}
MANUAL_CONSTRAINTS = {TO_QOI.COMPLIANCE, TO_QOI.PNORM_STRESS, TO_QOI.VOLUME_FRACTION, TO_QOI.VOLUME_FRACTION_MIN}
_AD_ONLY_HINT = " (analytical gradients cover compliance, p-norm stress and volume fraction; use automatic differentiation)"


def _physics(fe_solver):
    from pyto.autodiff.qoi.responses import physics_of
    return physics_of(fe_solver)


def formulation_support(to_params) -> tuple:
    """manual_support() without the solver: only which objective/constraint types have a manual gradient."""
    obj = to_params.Objective[0]
    if obj not in MANUAL_OBJECTIVES:
        return False, f"No analytical gradient for the objective {getattr(obj, 'name', obj)}{_AD_ONLY_HINT}."
    for c in to_params.Constraints:
        if c[0] not in MANUAL_CONSTRAINTS:
            return False, f"No analytical gradient for the constraint {getattr(c[0], 'name', c[0])}{_AD_ONLY_HINT}."
    return True, ""


def _types(to_params):
    return {to_params.Objective[0]} | {c[0] for c in to_params.Constraints}


def manual_support(to_params, fe_solver) -> tuple:
    """(True, "") if every objective/constraint has a manual gradient for this solver, else (False, reason)."""
    ok, reason = formulation_support(to_params)
    if not ok:
        return ok, reason
    types = _types(to_params)
    physics = _physics(fe_solver)
    if TO_QOI.PNORM_STRESS in types and physics != "structural":
        return False, "The analytical p-norm stress gradient is for structural problems only."
    if types & {TO_QOI.COMPLIANCE, TO_QOI.PNORM_STRESS} and getattr(fe_solver, "elem_body_force", None) is not None:
        return False, "The analytical gradients do not include design-dependent body forces."
    if TO_QOI.COMPLIANCE in types:
        thermal = fe_solver.thermal_fea if physics == "thermo-structural" else fe_solver if physics == "thermal" else None
        if thermal is not None and (getattr(thermal, "conductivity_penalty", None) is not None
                                    or getattr(thermal, "conductivity_void_ratio", None) is not None):
            return False, "The analytical compliance gradient assumes the default conductivity law."
        bcs = [fe_solver.bc] + ([fe_solver.thermal_fea.bc] if physics == "thermo-structural" else [])
        for bc in bcs:
            if _varying_prescribed(bc) and np.any(bc.force):
                return False, ("The analytical compliance gradient covers load-driven or prescribed-value-driven "
                               "problems, not both at once (varying prescribed values together with loads).")
    return True, ""


def _varying_prescribed(bc) -> bool:
    """Prescribed values that actually drive the field. A uniform value (the same on every fixed dof) is a constant
    temperature offset / rigid translation, which K ignores (K.1 = 0), so it counts as no prescribed value
    (e.g. HeatPlate: heat load + all fixed nodes at 23 C behaves as load-driven)."""
    v = np.asarray(bc.dirichlet_values, dtype=float)
    return v.size > 0 and np.ptp(v) > 1e-12 * max(1.0, np.abs(v).max())


def _compliance_gradient(sol, x, fe_solver, material_model):
    """dC/dx for C = u^T K u at the last solve (sol = that solve's solution)."""
    from pyto.autodiff.reference_adjoint import compute_compliance_and_gradient
    physics = _physics(fe_solver)
    if physics == "thermo-structural":
        from pyto.autodiff.material_model import _SIMP_STRUCTURAL_PENALTY, _SIMP_THERMAL_PENALTY
        from pyto.physics.thermoelastic.topopt_thermostructural_sensitivity import ThermoElasticSensitivity
        with contextlib.redirect_stdout(io.StringIO()):               # the analyser prints diagnostics
            return ThermoElasticSensitivity(fe_solver.thermal_fea, fe_solver).compute_compliance_sensitivity(
                x, fe_solver.temperature.detach().cpu().numpy(), sol, p=_SIMP_STRUCTURAL_PENALTY,
                q=_SIMP_THERMAL_PENALTY, material_model=material_model, solver=fe_solver.solver)
    # Load-driven: dC/dx = -dk/dx u^T ke u. Driven only by prescribed values (no load, e.g. FourCornersThermal): the
    # energy rises with stiffness and the sign flips (checked against autograd).
    sign = -1.0 if (not np.any(fe_solver.bc.force) and _varying_prescribed(fe_solver.bc)) else 1.0
    KE = fe_solver.elem_stiff[0]
    mesh = fe_solver.mesh
    saved = mesh.edofMat          # shared connectivity: make it match this solver (thermal meshes carry both)
    mesh.edofMat = mesh.edofMatThermal if physics == "thermal" else mesh.edofMatStructural
    try:
        return sign * compute_compliance_and_gradient(sol, x, fe_solver, KE, material_model)[1]
    finally:
        mesh.edofMat = saved


def _pnorm_gradient(sol, x, fe_solver, material_model):
    from pyto.autodiff.reference_adjoint import compute_pnorm_stress_and_sensitivity
    # adjoint solve on the forward solve's matrix and factors (fe_solver.adjoint_solve): same solver, no new
    # factorization, no conversion of the torch stiffness to SciPy
    return compute_pnorm_stress_and_sensitivity(sol, x, fe_solver, fe_solver.elem_stiff[0], material_model)[1]


def manual_gradients(to_params, sol, x, fe_solver, material_model=MaterialModel.SIMP):
    """Gradients w.r.t. the physical density x (numpy) of the objective and of every constraint, in the exact form
    compute_objective_and_gradient / compute_constraint_and_gradient evaluate them. sol and x belong to the last
    solve. Returns (dobj (n,), dcons (m, n))."""
    ok, reason = manual_support(to_params, fe_solver)
    if not ok:
        raise ValueError(f"Analytical gradient not available: {reason}")
    sol = sol.detach().cpu().numpy() if isinstance(sol, torch.Tensor) else np.asarray(sol)
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    n = x.size
    cache = {}

    def compliance():
        if "C" not in cache:
            cache["C"] = _compliance_gradient(sol, x, fe_solver, material_model)
        return cache["C"]

    def pnorm():
        if "P" not in cache:
            cache["P"] = _pnorm_gradient(sol, x, fe_solver, material_model)
        return cache["P"]

    obj = to_params.Objective[0]
    d_obj = {TO_QOI.COMPLIANCE: compliance, TO_QOI.PNORM_STRESS: pnorm,
             TO_QOI.VOLUME_FRACTION: lambda: np.full(n, 1.0 / n)}[obj]()

    rows = []
    for ctype, _param, limit in to_params.Constraints:
        if ctype == TO_QOI.VOLUME_FRACTION:                        # mean(x)/limit - 1
            rows.append(np.full(n, 1.0 / (n * limit)))
        elif ctype == TO_QOI.VOLUME_FRACTION_MIN:                  # 1 - mean(x)/limit
            rows.append(np.full(n, -1.0 / (n * limit)))
        elif ctype == TO_QOI.PNORM_STRESS:                         # P/limit - 1
            rows.append(pnorm() / limit)
        else:                                                      # COMPLIANCE: C/limit - 1
            rows.append(compliance() / limit)
    return np.asarray(d_obj, dtype=float), np.array(rows, dtype=float).reshape(len(rows), n)
