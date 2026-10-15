"""Which methods can solve a formulation, where their gradient comes from, and a gradient check (plan S4).

The GUI uses method_support() to enable only the methods that can solve the problem the user built (with the reason
shown for the others), gradient_source() to state how each method gets its gradient, check_gradient() to compare the
autograd gradient with finite differences before a long run, and cost_estimate() to show how many linear solves an
iteration needs.
"""
from dataclasses import dataclass

import numpy as np
import torch

from pyto.autodiff.material_model import MaterialModel
from pyto.topopt.common import GRADIENT_METHOD, gradient_method
from pyto.topopt.spec import OptimizationSpec, _key
from pyto.topopt.expressions import compile_expression

GRADIENT_SOURCE = {
    "MMA": "Automatic differentiation: PyTorch autograd through the FE solve (adjoint), any objective/constraint",
    "OC": "Automatic differentiation: PyTorch autograd (compliance with a volume constraint only)",
}


def gradient_source(method: str, gradient: str = "autodiff") -> str:
    if gradient_method(gradient) is GRADIENT_METHOD.ANALYTICAL:
        return "Analytical sensitivities: compliance, p-norm stress and volume fraction only"
    return GRADIENT_SOURCE[method]




def manual_gradient_support(spec: OptimizationSpec, selections: dict = None, body_force: bool = False) -> tuple:
    """(True, "") if MMA/OC can use analytical gradients for this formulation, else (False, reason).

    The solver-dependent limits (per-problem conductivity laws, prescribed values together with loads) are checked
    again at run time by manual_sensitivities.manual_support."""
    from pyto.topopt.manual_sensitivities import formulation_support
    from pyto.topopt.spec import PYTHON_PREFIX, compile_spec
    exprs = [spec.objective.expression] + [c.expression for c in spec.active_constraints]
    if any(e.strip().startswith(PYTHON_PREFIX) for e in exprs):
        return False, "Python functions need automatic differentiation."
    try:
        to_params = compile_spec(spec, selections)
    except Exception as e:
        return False, f"Fix the formulation first ({e})."
    ok, reason = formulation_support(to_params)
    if ok and body_force and any(_key(e) in ("Compliance()", "StressPNorm()") for e in exprs):
        return False, "The analytical gradients do not include design-dependent body forces."
    return ok, reason


def _is_classical(spec: OptimizationSpec) -> bool:
    """Minimize compliance subject to exactly one volume-fraction upper bound."""
    cons = spec.active_constraints
    return (spec.objective.sense == "minimize" and _key(spec.objective.expression) == "Compliance()"
            and len(cons) == 1 and _key(cons[0].expression) == "VolumeFraction()" and cons[0].op == "<="
            and cons[0].bound > 0)


def method_support(spec: OptimizationSpec, physics: str, body_force: bool = False) -> dict:
    """{method: (supported, reason)} for a formulation.

    physics: 'structural', 'thermal' or 'thermo-structural'; body_force: a design-dependent (self-weight) load is
    applied. MMA solves every formulation. OC only minimizes compliance with one volume constraint, and is not
    used for thermo-structural problems (the thermally loaded compliance is not monotone and OC 2-cycles) or with
    design-dependent body forces (it collapses on self-weight).
    """
    out = {"MMA": (True, "")}
    if not _is_classical(spec):
        out["OC"] = (False, "OC supports only minimize Compliance() with a single 'VolumeFraction() <= bound' "
                            "constraint; use MMA.")
    elif physics == "thermo-structural":
        out["OC"] = (False, "OC is not available for thermo-structural problems (the thermally loaded compliance is "
                            "not monotone and OC 2-cycles); use MMA.")
    elif body_force:
        out["OC"] = (False, "OC does not handle design-dependent body forces (self-weight); use MMA.")
    else:
        out["OC"] = (True, "")
    return out

def solution_dependent_count(spec: OptimizationSpec, selections: dict = None) -> int:
    """Number of objective/constraint functions whose gradient needs an adjoint solve (volume and mass do not)."""
    n = 0
    for e in [spec.objective.expression] + [c.expression for c in spec.active_constraints]:
        k = _key(e)
        if k == "MaxStress()" or e.strip().startswith("python:"):   # stress built-in / user code: assume it does
            n += 1
        else:
            n += int(compile_expression(e, selections).depends_on_solution)
    return n


def cost_estimate(spec: OptimizationSpec, selections: dict = None, forward_s: float = None,
                  backward_s: float = None) -> str:
    """Per-iteration linear solves (and seconds, if single forward/backward timings are given) for MMA."""
    if gradient_method(spec.method.gradient) is GRADIENT_METHOD.ANALYTICAL:
        k = int(any(_key(e) == "StressPNorm()"
                    for e in [spec.objective.expression] + [c.expression for c in spec.active_constraints]))
        return (f"Per iteration: 1 forward FE solve + {k} adjoint solve{'s' if k != 1 else ''} "
                f"(analytical: compliance is self-adjoint, p-norm stress needs one).")
    k = solution_dependent_count(spec, selections)
    text = f"Per iteration: 1 forward FE solve + {k} adjoint solve{'s' if k != 1 else ''}"
    if forward_s is not None and backward_s is not None:
        text += f" (about {forward_s + k * backward_s:.2f} s)"
    return text + "."


@dataclass
class GradientCheck:
    name: str
    max_rel_error: float        # max |FD - AD| over checked elements / max |AD| over them
    ok: bool


def check_gradient(to_params, fe_solver, n_elements: int = 5, rel_step: float = 1e-4, tol: float = 1e-3,
                   seed: int = 0) -> list:
    """Compare the gradients of the objective and every constraint (autograd, or the analytical ones when
    to_params.Gradient is GRADIENT_METHOD.ANALYTICAL) with central finite differences.

    Evaluated at a random interior design (0.3..0.9) on n_elements random elements, with the same material law the
    MMA driver uses. The adaptive stress normalization (to_params.stress_scaling) is reset before every evaluation so
    the finite differences see the same function as autograd. Returns one GradientCheck per function.
    """
    from pyto.autodiff.qoi import compute_constraint_and_gradient, compute_objective_and_gradient
    from pyto.topopt.drivers._shared import compute_element_stiffness
    KE = torch.tensor(compute_element_stiffness(fe_solver))
    mm = MaterialModel.SIMP
    rng = np.random.default_rng(seed)
    x0 = torch.tensor(0.3 + 0.6 * rng.random(fe_solver.mesh.num_elems), dtype=torch.float64)
    elems = rng.choice(fe_solver.mesh.num_elems, size=min(n_elements, fe_solver.mesh.num_elems), replace=False)
    scaling0 = to_params.stress_scaling
    last = {}

    def values(x):
        to_params.stress_scaling = scaling0
        sol = last["sol"] = fe_solver.solve(x, mm)
        obj = compute_objective_and_gradient(to_params, sol, x, fe_solver, KE, mm)
        cons, _ = compute_constraint_and_gradient(to_params, sol, x, fe_solver, KE, mm)
        return [obj.reshape(())] + [c.reshape(()) for c in cons.reshape(-1)]

    try:
        if gradient_method(to_params.Gradient) is GRADIENT_METHOD.ANALYTICAL:
            from pyto.topopt.manual_sensitivities import manual_gradients
            with torch.no_grad():
                vals = values(x0.clone())
            d_obj, d_cons = manual_gradients(to_params, last["sol"], x0, fe_solver, mm)
            ad = [torch.as_tensor(g) for g in [d_obj] + list(d_cons)]
        else:
            x = x0.clone().requires_grad_(True)
            vals = values(x)
            ad = [torch.autograd.grad(v, x, retain_graph=True, allow_unused=True)[0] for v in vals]
            ad = [torch.zeros_like(x0) if g is None else g.detach() for g in ad]
        fd = [np.zeros(len(elems)) for _ in vals]
        for j, e in enumerate(elems):
            h = rel_step * max(1.0, abs(float(x0[e])))
            up, dn = x0.clone(), x0.clone()
            up[e] += h
            dn[e] -= h
            with torch.no_grad():
                vu, vd = values(up), values(dn)
            for i in range(len(vals)):
                fd[i][j] = (vu[i].item() - vd[i].item()) / (2 * h)
    finally:
        to_params.stress_scaling = scaling0

    names = ["objective"] + [f"constraint {i + 1}" for i in range(len(vals) - 1)]
    out = []
    for name, g, f in zip(names, ad, fd):
        a = g[torch.as_tensor(elems)].numpy()
        scale = max(np.abs(a).max(), np.abs(f).max(), 1e-300)
        err = float(np.abs(a - f).max() / scale)
        out.append(GradientCheck(name, err, err <= tol))
    return out
