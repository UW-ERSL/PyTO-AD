# Running the optimizers

This page describes PyTO's four optimization methods: what each can solve, how to call it, and what it returns. It is for anyone running optimizations from code.

## Which method
| Method | Function | Solves | Gradient |
|---|---|---|---|
| **MMA** (Method of Moving Asymptotes) | `topopt_mma` in `pyto.topopt.drivers.mma` | any objective and any number of constraints, all physics | autograd, or [analytical](gradients.md) for compliance, p-norm stress and volume fraction |
| **OC** (Optimality Criteria) | `topopt_optimality_criteria` in `pyto.topopt.drivers.oc` | minimize compliance with one volume limit | autograd, or analytical |

Rules of thumb:
- **Use MMA** unless you have a reason not to. It is the most general and the best tested.
- **OC** is fast for classical compliance problems. It is unreliable when loads depend on the design: thermal expansion (it oscillates), and pure self-weight (the design collapses).

`pyto.topopt.capabilities.method_support(spec, physics, body_force)` returns, for a formulation, which methods apply and why not. The GUI uses it to enable its method list.

## MMA
```python
from pyto.topopt.drivers.mma import topopt_mma

u, history, success, message, n_fea = topopt_mma(
    fe_solver, to_params,
    maxMMAIterations=150,        # iteration limit
    timeLimitSecs=36000,         # wall-clock limit
    move_limit=0.2,              # max change of a density per iteration
    kkt_tol=1e-6, objective_tol=1e-4, constraint_tol=1e-4,   # convergence tests
    print_progress=True,         # print each iteration
    binarize_topology=True,      # round the final design to 0/1 (keeping its volume) and evaluate it
    progress_callback=None,      # fn(message: str): receives log messages instead of print
    iteration_callback=None)     # fn(history: dict): called after every iteration
```
- **Convergence:** it stops when the objective changes by less than `objective_tol` and the constraints hold within `constraint_tol`, when the KKT residual is below `kkt_tol`, or at the iteration or time limit. With Heaviside projection on, the objective test is off, so β can finish its continuation.
- **Final evaluation:** the returned design is evaluated once more with the filter turned off, because the design is already physical. With `binarize_topology=True` it is first thresholded to 0/1 at the level that keeps the volume. So the **last history entry is the delivered design**; the one before it is the last continuous iterate.
- **Safeguard:** if the 0/1 design is more than twice as bad as the last continuous iterate (thresholding cut a load or heat path), the continuous design is returned instead and `message` says so (MMA and OC).
- **Stress constraints:** if the final peak von Mises stress exceeds a stress limit by more than 1 %, `message` says so.

## OC
```python
from pyto.topopt.drivers.oc import topopt_optimality_criteria

u, history, success, message, n_fea = topopt_optimality_criteria(
    fe_solver, to_params, maxIterations=250, move=0.2, move_tol=0.05, rel_conv_tol=1e-4,
    print_progress=True, binarize_topology=True, progress_callback=None, iteration_callback=None)
```
`to_params` must be *minimize `COMPLIANCE`* with exactly one `VOLUME_FRACTION` constraint. `to_params.materialModel` selects SIMP, RAMP or SIMP+ for OC.

## What comes back
| Return | Meaning |
|---|---|
| `u` | solution of the final design (numpy): displacements or temperatures |
| `history` | per-iteration values, below |
| `success` | `False` if the run failed |
| `message` | `"No errors."`, or a warning or error |
| `n_fea` | number of FE solves |

| `history` key | Content |
|---|---|
| `objective` | objective per iteration, in physical units (for `maximize`, the minimized value, i.e. −f) |
| `volfrac` | mean physical density per iteration |
| `constraint_1`, `constraint_2`, ... | MMA only: each constraint in its normalized form `g ≤ 0` (`value/limit − 1` for a `<=` constraint) |
| `change` | OC only: largest density change per iteration |

The design itself is on the mesh: `fe_solver.mesh.elemPseudoDensity`. For plots and evaluation, see [Part 3](../postprocessing/history-and-convergence.md).

## Callbacks: live monitoring
```python
def on_iteration(history):
    print(len(history["objective"]), history["objective"][-1])

topopt_mma(fe, to_params, iteration_callback=on_iteration, print_progress=False)
```
`iteration_callback` gets the whole history after every iteration (MMA and OC); the GUI draws its live plot from it. `progress_callback` receives the text messages the driver would print.

## Performance
- **Mesh size** dominates. Start around 5 000–25 000 unknowns (`nDOFDesired`), then refine.
- **Solver:** PARDISO is several times faster than SPSOLVE. Its factorization is reused for the gradient's adjoint solve.
- **Solves per iteration:** one FE solve, plus one adjoint solve per objective or constraint that depends on the solution. Volume and mass need none. `pyto.topopt.capabilities.cost_estimate(spec)` states it for a formulation.
- **Analytical gradients** avoid building the autograd graph and are faster where available; see [Gradients](gradients.md).

---
Next: [Gradients](gradients.md) · [Back to contents](../README.md)
