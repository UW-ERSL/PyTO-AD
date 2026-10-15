# User-defined functions

This page shows how to optimize **any quantity you can compute with PyTorch**: write a Python function, and PyTO differentiates it through the FE solve for you. Use it when [expressions](expressions.md) aren't enough, for example for a custom weighting of displacements, a field computed element by element, or quantities from several regions combined in a non-standard way.

The full, runnable example is [`docs/examples/user_defined_function.py`](../examples/user_defined_function.py).

## The function
```text
def my_quantity(sol, x, fe_solver):
    ...
    return value            # a torch scalar
```
| Argument | What it is |
|---|---|
| `sol` | the FE solution of this iteration (torch, part of the autograd graph): displacements (3 per node, `sol.reshape(-1, 3)`) for structural and thermo-structural solvers, temperatures (1 per node) for thermal ones |
| `x` | the physical density (after filtering and projection), one value per element (torch) |
| `fe_solver` | the solver: `fe_solver.mesh`, `fe_solver.bc`, `fe_solver.mat_prop`, `fe_solver.elem_stiff`, and for coupled solvers `fe_solver.temperature` |

Two examples, both from the example file:
```python
import numpy as np
import torch

def tip_deflection_squared(sol, x, fe_solver):
    """Mean squared displacement along the load direction at the loaded nodes (m^2)."""
    f = torch.as_tensor(fe_solver.bc.force)
    loaded = torch.nonzero(f).flatten()
    direction = torch.sign(f[loaded])
    return ((sol[loaded] * direction) ** 2).mean()

def material_in_left_half(sol, x, fe_solver):
    """Volume fraction of the left half of the model (depends on the design only)."""
    left = torch.as_tensor(fe_solver.mesh.elem_centers[:, 0] < np.median(fe_solver.mesh.elem_centers[:, 0]))
    return x[left].mean()
```

## Rules
1. **Torch operations on `sol` and `x` only.** NumPy, `.numpy()`, `.item()` or `float(...)` on them breaks the gradient path, and PyTO stops with *"has no gradient path to the design"*.
2. **Return one value**: reduce arrays with `.sum()`, `.mean()` or a p-norm. Returning a vector gives *"it must return a single value"*.
3. **Only depend on the arguments.** Constant data (node lists, weights) can be computed outside the function or from `fe_solver`, as NumPy, then converted with `torch.as_tensor`.
4. **Keep it smooth.** `max`, `abs` or `where` on the design give gradients that jump, and MMA may stall. Prefer p-norms (`(v ** p).sum() ** (1 / p)`) and squares.
5. **Units:** SI, like everything in PyTO. Scale the result so its value is of order 1–100; MMA behaves better.

## Way A: directly in TOParams
```python
from pyto.topopt.common import TO_QOI

to_params.Objective = (TO_QOI.GFUNCTION, tip_deflection_squared)
to_params.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3),
                         (TO_QOI.GFUNCTION, material_in_left_half, 0.45)]     # fn <= 0.45
```
Constraint forms:

| Tuple | Means |
|---|---|
| `(TO_QOI.GFUNCTION, fn, limit)` with `limit > 0` | `fn ≤ limit` (stored as `fn/limit − 1 ≤ 0`) |
| `(TO_QOI.GFUNCTION, g, None)` | `g ≤ 0`; you normalize it yourself |

To **maximize**, minimize the negative: `lambda sol, x, fe: -fn(sol, x, fe)`. For a **`>=`** constraint, write it as `lambda sol, x, fe: 1 - fn(sol, x, fe) / bound` with limit `None`.

## Way B: through a spec (what the GUI stores)
```python
from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec

spec = OptimizationSpec(
    objective=ObjectiveSpec("python:/path/to/my_functions.py::tip_deflection_squared"),
    constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3)])
to_params = compile_spec(spec, allow_code=True)          # loading the file runs it: opt in explicitly
```
- The reference format is `python:<path to .py>::<function name>`.
- `compile_spec` handles `maximize` and `>=` for you, as for expressions.
- In the GUI, choose **Type: Python function...** and enter the same reference. The GUI asks for confirmation before running the file.

## Check the gradient before a long run
```python
from pyto.topopt.capabilities import check_gradient

for r in check_gradient(to_params, fe):
    print(r.name, f"{r.max_rel_error:.1e}", "OK" if r.ok else "MISMATCH")
```
It compares the autograd gradient with central finite differences on 5 random elements. For the example above:
```
objective: 2.3e-07 OK
constraint 1: 7.1e-10 OK
constraint 2: 5.3e-10 OK
```
A mismatch usually means a non-smooth operation, or data changing between calls (for example randomness inside the function).

## Methods
User-defined functions need **MMA**. OC only handles compliance with one volume constraint, and the [analytical gradients](gradients.md) don't cover user functions.

## Useful building blocks
| Want | Torch code |
|---|---|
| Displacements as (nodes, 3) | `u = sol.reshape(-1, 3)` |
| Displacements of some nodes | `u[torch.as_tensor(nodes)]` |
| Element strain energy (structural) | `ue = sol[torch.as_tensor(fe_solver.mesh.edofMatStructural)]`, then `torch.einsum("ei,ij,ej->e", ue, torch.as_tensor(fe_solver.elem_stiff[0]), ue)`, times the SIMP scaling of `x` |
| Temperature of a coupled solver | `fe_solver.temperature` |
| Smooth maximum | `(v.abs() ** p).sum() ** (1 / p)` |
| Region mask | `torch.as_tensor(fe_solver.mesh.elem_centers[:, 0] < 0.5)` |
| A built-in quantity | `from pyto.autodiff.qoi import responses`, then `responses.compliance(sol, x, fe_solver)`, `responses.displacement(sol, x, fe_solver, nodes=..., component="y")` |

For a quantity you'll reuse, or want in the GUI's list, register it as a response instead: [Add a response](../extending/add-a-response.md).

---
Next: [Running the optimizers](running-optimizers.md) · [Back to contents](../README.md)
