# Quick start: notebook

This page explains `main.ipynb`, the notebook that runs any benchmark problem with MMA (automatic differentiation or analytical sensitivities) and shows everything about the run. It is for users who want to explore the built-in problems, or check a result, without writing code.

## Open it
1. Open `main.ipynb` in VS Code or Jupyter, with the **PyToLib** kernel.
2. Run it from the repository root; the first cell adds `src/` to the import path.

## The first cell: choose problem and method
```python
PROBLEM = StructuralTOExamples.MBBBeam
# PROBLEM = StructuralTOExamples.LBracketTopLoad_Stress_Vol       # p-norm stress objective (MMA only)
# PROBLEM = ThermalTOExamples.HeatPlate
# PROBLEM = ThermoStructuralTOExamples.MBBBeamNoHeat              # coupled problems: MMA recommended
METHOD = TO_METHODS.DENSITYMMA          # DENSITYMMA or DENSITYOCM
GRADIENT = GRADIENT_METHOD.AUTODIFF     # GRADIENT_METHOD.AUTODIFF or GRADIENT_METHOD.ANALYTICAL
SOLVER = None                           # None -> as the sweep: PARDISO up to 100k dofs, DPCG above;
                                        # or Solvers.PARDISO, Solvers.SPSOLVE (macOS), Solvers.DPCG, Solvers.PCG
MAX_ITERS = None                        # None -> the problem's to_params.MaxIterations (what the sweep uses)
BINARIZE = True                         # the sweep binarizes the results (not with Heaviside projection)
```
- **`PROBLEM`**: any member of `StructuralTOExamples`, `ThermalTOExamples` or `ThermoStructuralTOExamples`. The full list is in [Benchmark problems](../reference/benchmarks.md).
- **`METHOD`**: MMA, which handles every formulation.
- **`GRADIENT`**: automatic differentiation or analytical sensitivities. Both give the same gradient; analytical is faster where available (compliance, p-norm stress, volume fraction), and the build cell stops with the reason elsewhere. See [Gradients](../code/gradients.md).
- **`SOLVER`**: `None` chooses as the benchmark sweep does (PARDISO up to 100 000 dofs, DPCG with its deflation space above). Set it to force one, e.g. `Solvers.SPSOLVE` on macOS. Both gradient methods use the chosen solver for the FE solve and the adjoint. See [Installation](installation.md#linear-solvers).
- **`MAX_ITERS`**: leave `None` to use the problem's own setting.
- **`BINARIZE`**: round the final design to 0/1 and evaluate it again, as the benchmark sweep does.

Then **Run All**.

## What the cells show
| Cell | Shows |
|---|---|
| Build | solver, number of unknowns, objective, constraints, supports and loads; a picture of the setup |
| Optimize | success, message, number of FE solves, time and time per FE solve |
| Results | objective and volume fraction of the last continuous design and of the final design |
| Topology | the final design, as in the benchmark result images |
| Fields | density, deformation and von Mises stress (structural), temperature (thermal and coupled) |
| Convergence | objective and volume fraction per iteration |

The build step uses `build_benchmark()`, the same function the benchmark sweep uses. So one notebook run is exactly one row of the sweep, and a result can be checked against the saved sweep results.

---
Next: [Concepts](../code/concepts.md) · [Back to contents](../README.md)
