# Troubleshooting

This page lists common problems and error messages, with their cause and fix. Search it for the text of your message.

## Installation and start-up
| Symptom / message | Cause | Fix |
|---|---|---|
| `The GUI needs ..., which this Python (...) does not have, and the conda environment 'PyToLib' was not found` | `run_gui.py` couldn't find the environment | create it ([Installation](../getting-started/installation.md)), or set `PYTO_CONDA_ENV=<your env>` |
| `ModuleNotFoundError: No module named 'pyto'` in your script | `src/` isn't on the import path | `export PYTHONPATH=/path/to/PyTO/src`, or `sys.path.insert(0, ".../src")` |
| no windows appear, or plots are empty | headless variables are set (`PYVISTA_OFF_SCREEN`, `MPLBACKEND=Agg`) | unset them for interactive use; `run_gui.py` clears them itself |
| `pypardiso` fails to install (macOS) | not available on macOS | skip it; use `Solvers.SPSOLVE` (GUI: **Solver Type → SPSOLVE**) |
| warning: `Solvers.PYAMG is not implemented for the torch-autodiff sparse solve path ... falling back` | PYAMG isn't available during optimization | use PARDISO, SPSOLVE or DPCG; the result is correct anyway |
| warning: `Maximum iterations reached in DPCG` or `PCG did not converge` | the iterative solve stopped early (very stiff contrasts, or no deflation space) | use PARDISO for problems below ~100 000 dofs; for large ones build the deflation space as `build_benchmark` does |

## Setting up a problem
| Symptom / message | Cause | Fix |
|---|---|---|
| `BC is under-constrained: at least one rigid-body mode ... is left free` | supports don't stop the part moving or rotating in the direction of a load | add supports; e.g. fix z on a flat 2.5D model, or add a second support point |
| `For a structural optimization, please apply: supports ...; a load ...` (GUI) | supports or loads missing for the physics | apply them in **Structural Loads** / **Thermal Loads** / **Body force** |
| missing elements or holes after **Generate Mesh** | the STL isn't watertight, or walls are thinner than an element | repair the STL; use more elements |
| a load or support has no effect | the selected nodes don't lie on the mesh (e.g. a coordinate window narrower than half an element) | widen the window; check `len(nodes)`; draw with `plot_mesh(plot_bc=True)` |
| thermal temperatures of 1e8 and up | heat input on nodes that became void | put heat sources only where material will stay ([Thermal problems](../code/thermal-and-thermostructural.md#thermal-problems)) |

## Formulation
| Message | Cause | Fix |
|---|---|---|
| `Unknown selection 'X' in Displacement(). Defined selections: ...` | name typo, or the load or support doesn't exist | use a listed name |
| `max() is not differentiable; use smooth_max(a, b, ...) or a pnorm aggregate.` (also `min`, `abs`) | not allowed in expressions | use `smooth_max`, `pnorm` or `magnitude` |
| `constraint 1: a >= bound must be > 0 ...` | `>=` needs a positive bound | rewrite the constraint (e.g. flip the sign of the quantity and use `<=`) |
| `Minimizing volume/mass without constraints: the optimum is an empty design` | ill-posed | add a stiffness, displacement or stress constraint |
| `WARNING: Minimizing compliance without a volume or mass limit` | the answer is the full block | add `VolumeFraction() <= ...` |
| `X not available for thermal problems` | the quantity doesn't exist for that physics | see the physics column in [Expressions](../code/expressions.md#quantities) |
| `Analytical gradient not available: ... Use automatic differentiation.` | no analytical gradient for this quantity or problem | switch **Gradient** to automatic differentiation ([Gradients](../code/gradients.md#analytical-gradients)) |
| `GFUNCTION objective '...' has no gradient path to the design` | a Python function used NumPy, `.item()` or `float()` on the solution | use torch operations ([User-defined functions](../code/user-defined-functions.md#rules)) |

## Results
| Symptom | Cause | Fix |
|---|---|---|
| message `the 0/1 (binarized) design was much worse than the continuous one ... so the continuous design is returned` | thresholding would have cut load or heat paths (typical for grey designs, e.g. heat conduction) | the returned grey design is fine to use; for a 0/1 part use Heaviside projection or a higher conductivity penalty ([Thresholding](../postprocessing/thresholding-and-evaluation.md#when-thresholding-goes-wrong)) |
| a constraint is `violated` at the end | the bound is too tight for the material, or too few iterations | loosen it, allow more material, or run more iterations |
| `Warning: stress limit exceeded by x%` | the final 0/1 design's peak stress is above the limit | a slightly tighter bound, or `StressFailureFactor` with a margin |
| objective oscillates | move limit too large, or a design-dependent load | lower **Move limit** |
| the design uses less material than allowed | normal for heated or self-weight problems (material adds load) | nothing to fix ([Thermo-structural coupling](../theory/thermo-structural.md#why-its-different-to-optimize)) |
| grey, blurry design | filter radius large relative to the part, or few iterations | smaller **Filter radius**, more iterations, Heaviside projection |
| checkerboard pattern | no density filter (e.g. `APPLY_FILTER_TO_DENSITY = False`) | keep the filter on |
| STL recovery returns the original STL (`Boolean subtraction failed`) | the original STL isn't watertight | repair the STL ([Export](../postprocessing/export.md#smooth-stl)) |

## Still stuck
- Run the tests (`pytest tests -q`) to see whether the installation itself is healthy.
- Reproduce the problem with a small mesh (`nDOFDesired=5000`) and `check_gradient`.
- Open an issue with the message, the formulation (`spec.to_json()`), and how to reproduce it.

---
Next: [Testing](testing.md) · [Back to contents](../README.md)
