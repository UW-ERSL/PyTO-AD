# Add an optimization method

This recipe adds a new optimization algorithm (a **driver**) next to MMA and OC, or in place of one. It is for developers trying a new update rule, a different constraint handling, a population-based method, or a machine-learning approach.

## The driver contract
```text
def topopt_mymethod(fe_solver, to_params, maxIterations=100, ...,
                    print_progress=True, progress_callback=None, iteration_callback=None):
    ...
    return solution, history, success, message, n_fea
```
| Part | Requirement |
|---|---|
| `fe_solver` | any PyTO FE solver; call `fe_solver.solve(x_phys, material_model)` |
| `to_params` | read the formulation (`Objective`, `Constraints`) and settings (filter radius, symmetry, `MaxIterations`, `Gradient`) from it |
| `solution` | numpy solution of the final design |
| `history` | dict of lists, at least `objective` and `volfrac`, one entry per iteration; `constraint_1`, ... in the normalized `g ≤ 0` form if you handle constraints |
| `success`, `message`, `n_fea` | bool; `"No errors."` or a warning or error; number of FE solves |
| the design | leave it on the mesh: `fe_solver.mesh.setPseudoDensity(torch_tensor)` (plots and post-processing read `mesh.elemPseudoDensity`) |
| `iteration_callback(history)` | call it after each iteration if given (the GUI's live plot) |
| `progress_callback(message)` | send messages to it instead of `print` if given (the GUI's log) |

## The building blocks
```python
from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.qoi import compute_constraint_and_gradient, compute_objective_and_gradient
from pyto.topopt.drivers._shared import setup_driver_state, torch_sparse_filter

state = setup_driver_state(fe_solver, to_params)   # H, Hs (filter incl. symmetry/extrusion), KE, forced elements
H, Hs = torch_sparse_filter(state["H"], state["Hs"])
KE = torch.tensor(state["KE"])

x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
x_phys = torch.sparse.mm(H, x_t.unsqueeze(1)).squeeze(1) / Hs        # density filter
sol = fe_solver.solve(x_phys, MaterialModel.SIMP)
f = compute_objective_and_gradient(to_params, sol, x_phys, fe_solver, KE, MaterialModel.SIMP)
c, _ = compute_constraint_and_gradient(to_params, sol, x_phys, fe_solver, KE, MaterialModel.SIMP)  # (m, 1), g <= 0
(df_dx,) = torch.autograd.grad(f, x_t, retain_graph=True)             # gradient through filter and solve
```
- `compute_objective_and_gradient` and `compute_constraint_and_gradient` evaluate **every** kind of objective and constraint PyTO knows: built-ins, expressions, user functions. So a new driver gets all formulations for free.
- For the constraint Jacobian, call `torch.autograd.grad(c[i, 0], x_t, retain_graph=True)` per row, as `mma.py` does.
- Analytical gradients instead of autograd: `manual_gradients(to_params, sol, x_phys, fe_solver, MaterialModel.SIMP)` from `pyto.topopt.manual_sensitivities`, then chain through the filter as `manual_step` in `mma.py` does.

## Worked example: projected steepest descent
[`docs/examples/custom_driver.py`](../examples/custom_driver.py) is a complete driver in about 50 lines. Each iteration:
1. takes a gradient step with a move limit;
2. uses bisection to find the shift that brings the filtered design back onto the volume budget;
3. records history and calls the callback.

The core loop:
```python
for it in range(maxIterations):
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    x_phys = torch.sparse.mm(H, x_t.unsqueeze(1)).squeeze(1) / Hs
    fe_solver.mesh.setPseudoDensity(x_phys.detach())
    sol = fe_solver.solve(x_phys, mm)
    f = compute_objective_and_gradient(to_params, sol, x_phys, fe_solver, KE, mm)
    (g,) = torch.autograd.grad(f, x_t)
    g = g.numpy() / (np.abs(g.numpy()).max() + 1e-300)
    lo, hi = -1.0, 1.0                                   # bisection on the volume shift
    for _ in range(60):
        mu = 0.5 * (lo + hi)
        x_new = np.clip(x - move * g + mu, np.maximum(0.0, x - move), np.minimum(1.0, x + move))
        lo, hi = (mu, hi) if filtered_mean(x_new) < V else (lo, mu)
    ...
```
On the benchmark cantilever (5 000 dofs, 30 % volume), it lowers the compliance from 251.6 J to 55.3 J in 60 iterations, holding the volume at exactly 0.300. MMA reaches 19 J on the same problem: the example shows the structure of a driver, not a competitive method.

## Registration checklist
A driver works as soon as you call it from a script. To make it a first-class method, register it in these places. There is currently no single registry; one is planned.

| # | File | What to add |
|---|---|---|
| 1 | `topopt/common.py` | a member of `TO_METHODS`, e.g. `MYMETHOD = enum.auto()` |
| 2 | `topopt/spec.py` | its name in `METHODS` (the GUI's Method list is built from this tuple) |
| 3 | `topopt/capabilities.py` | its support rules in `method_support()` (which formulations and physics it handles, and why not otherwise), a line in `GRADIENT_SOURCE` |
| 4 | `examples_benchmarks/topopt_run_benchmarks.py` | a branch in `runTOMethodOnBenchmarks` next to the MMA/OC calls (with `continue` for problems it can't run), and the enum in the default list of `run_all_benchmarks_isolated` |
| 5 | `gui/PyTOGUI.py` | a branch in `StructuralTopOptWindow._run_optimization` calling your driver with the GUI's callbacks |
| 6 | `tests/test_drivers_smoke.py` | a test: a few iterations on the structural fixture run and lower the objective |

## Replacing a method
To use your algorithm **instead of** MMA for everything, point the `MMA` branch in steps 4 and 5 at your driver. Keep MMA available under another name, so you can compare results on the benchmark sweep ([Comparing results](../postprocessing/comparing-results.md)).

## Design notes
- **Constraints:** MMA handles any number of inequality constraints. Simpler methods can handle one volume constraint by bisection, as in the example and OC. General constraints need a penalty, an augmented Lagrangian, or a subproblem solver.
- **Grey designs:** density methods leave intermediate densities. Use `binarize_topology`-style thresholding at the end, or a projection with continuation.
- **Sensitivity filtering:** OC filters sensitivities (`H(x·g)/Hs/x`) rather than densities. Either choice works, but be consistent: filtered densities need the gradient chained through the filter.
- **Stopping:** combine an iteration cap, a design-change tolerance, and an objective-change tolerance.

---
Next: [Add a material model](add-a-material-model.md) · [Back to contents](../README.md)
