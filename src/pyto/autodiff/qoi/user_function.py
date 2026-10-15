"""User-defined objective/constraint (TO_QOI.GFUNCTION).

The user supplies a Python function written with torch operations:

    def my_qoi(sol, x, fe_solver) -> torch scalar
        sol       : (n_dof,) FE solution (displacement, temperature, ...) -- part of the autograd graph
        x         : (n_elems,) physical (filtered) density used in the solve
        fe_solver : the FE solver, for mesh, bc, material, element stiffness, ...

and sets it in the (type, param[, limit]) slot the other QOIs use:

    to_params.Objective   = (TO_QOI.GFUNCTION, my_qoi)
    to_params.Constraints = [(TO_QOI.GFUNCTION, my_qoi, limit)]   # my_qoi <= limit  (limit > 0)
                            [(TO_QOI.GFUNCTION, my_g, None)]      # my_g   <= 0      (already normalized)

The gradient is computed by autograd through the FE solve (adjoint), so no sensitivity has to be written. The
function must use torch operations only (NumPy breaks the gradient path) and must depend only on its
arguments. Only the MMA driver supports it; OC handles compliance with a volume constraint only.
"""
import torch


def evaluate_user_function(fn, sol: torch.Tensor, x: torch.Tensor, fe_solver, role: str = "objective") -> torch.Tensor:
    """Call the user function and check that its result is a differentiable torch scalar."""
    if not callable(fn):
        raise TypeError(f"GFUNCTION {role}: expected a function fn(sol, x, fe_solver) in the parameter slot, "
                        f"got {type(fn).__name__}.")
    name = getattr(fn, "__name__", repr(fn))
    value = fn(sol, x, fe_solver)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"GFUNCTION {role} '{name}' returned {type(value).__name__}; it must return a torch "
                        f"tensor built with torch operations on sol/x so autograd can differentiate it.")
    if value.numel() != 1:
        raise ValueError(f"GFUNCTION {role} '{name}' returned a tensor of shape {tuple(value.shape)}; "
                         f"it must return a single value (reduce with .sum(), .mean(), a p-norm, ...).")
    if (sol.requires_grad or x.requires_grad) and not value.requires_grad:
        raise RuntimeError(f"GFUNCTION {role} '{name}' has no gradient path to the design: it was probably "
                           f"computed with NumPy (.numpy(), .item(), float(...)) or under torch.no_grad(). "
                           f"Write it with torch operations on sol and x.")
    return value.reshape(())


def user_function_constraint(fn, limit, sol, x, fe_solver) -> torch.Tensor:
    """Constraint value in the <= 0 form every other constraint uses: fn/limit - 1, or fn itself if limit is None."""
    value = evaluate_user_function(fn, sol, x, fe_solver, role="constraint")
    if limit is None:
        return value
    if limit <= 0:
        raise ValueError(f"GFUNCTION constraint limit must be > 0 (got {limit}); for a constraint of the form "
                         f"g <= 0 pass limit None.")
    return value / limit - 1.0
