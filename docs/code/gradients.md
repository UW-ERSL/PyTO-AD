# Gradients

This page explains how PyTO computes the gradient (sensitivity) of objectives and constraints, how to choose between automatic differentiation and analytical gradients, and how to check a gradient. It is for anyone who wants to trust, speed up or extend the optimization.

## Automatic differentiation (default)
For MMA, PyTO builds each objective and constraint as a PyTorch computation from the design `x`:

filter → projection → FE solve → quantity.

`torch.autograd` then differentiates backwards through all of it. The FE solve is a custom autograd function (`pyto.autodiff.sparse_solve`) whose backward pass solves the **adjoint system** with the same matrix. So:
- **any** quantity written with torch operations (built-in, [expression](expressions.md) or [Python function](user-defined-functions.md)) gets an exact gradient, with no derivation needed;
- the cost is about **one extra linear solve per quantity that depends on the solution**. Volume and mass need none, and with PARDISO the factorization is reused.

## Analytical gradients
For the most common quantities, PyTO also has analytical (adjoint) sensitivities, derived by hand, in `pyto.topopt.manual_sensitivities`. They give the same values without building the autograd graph.

| Quantity | Analytical gradient | Physics |
|---|---|---|
| `COMPLIANCE` (objective or constraint) | yes, needs no adjoint solve (self-adjoint) | structural, thermal, thermo-structural |
| `PNORM_STRESS` (objective or constraint) | yes, one adjoint solve | structural |
| `VOLUME_FRACTION` (`<=`), `VOLUME_FRACTION_MIN` (`>=`) | yes, trivial | all |
| everything else (mass, max stress, failure factor, expressions, Python functions) | no, automatic differentiation only | – |

Also not covered: design-dependent body forces (self-weight), per-problem conductivity laws (`ConductivityPenalty`, `ConductivityVoidRatio`), and compliance of problems driven by varying prescribed values together with loads. `manual_support(to_params, fe)` returns `(False, reason)` for these.

Choose it with the `GRADIENT_METHOD` enum:
```python
from pyto.topopt.common import GRADIENT_METHOD

to_params.Gradient = GRADIENT_METHOD.ANALYTICAL      # default GRADIENT_METHOD.AUTODIFF
```
or `MethodSpec(gradient="analytical")` in a spec, or **Method → Gradient → Analytical** in the GUI. The strings `"autodiff"` and `"analytical"` are accepted too, and so is the old name `"manual"` (e.g. in saved projects); `gradient_method(value)` converts any of these to the enum.

It applies to **MMA**: the analytical gradient is taken with respect to the physical density and chained back through the Heaviside projection and the density filter by hand.

### Same solver path in both modes
Both modes run the **same forward FE solve** (same assembly, same linear solver). The analytical adjoint solve then uses that solve's matrix and solver object through `fe.adjoint_solve(rhs)`:
- **PARDISO / SPSOLVE:** the forward factorization is reused, so there's no second factorization;
- **DPCG / PCG:** the same preconditioner and deflation space are reused.

It's the same reuse the autograd backward pass gets, so timings compare the gradient methods, not two different solver set-ups.

### Why use analytical gradients
- **Speed:** no autograd graph, and no autograd pass back through the stiffness assembly. MMA, PARDISO, same iterates in both modes, seconds per iteration:

  | Problem | dofs | AD | Analytical | Speed-up |
  |---|---|---|---|---|
  | LBracketTopLoad_Stress_Vol (p-norm stress) | 38 961 | 3.47 | 2.15 | 1.62 |
  | BiClamp (thermo-structural compliance) | 33 750 | 2.62 | 1.99 | 1.32 |
  | HeatPlate (thermal compliance) | 9 245 | 1.03 | 0.82 | 1.26 |
  | MBBBeam (compliance) | 59 220 | 4.76 | 4.12 | 1.16 |

  These are 5-iteration runs, each in a fresh process; a 20-iteration L-bracket run gave 1.60 against 1.36 s/iteration. Rerun them with the [comparison benchmark](#comparing-the-two-on-the-benchmarks).
- **Reference:** an independent derivation to check autograd against, and a starting point for methods that need derived sensitivities.

### How they are verified
`tests/test_manual_sensitivities.py` compares every analytical gradient with autograd. Agreement is at the 1e-14 level: 9e-15 for compliance, 1.7e-14 for p-norm stress, and below 1e-6 for thermo-structural compliance. The tests also run full MMA optimizations both ways and require the same iterates. To add an analytical gradient of your own: [Add an analytical sensitivity](../extending/add-a-manual-sensitivity.md).

## Checking a gradient
```python
from pyto.topopt.capabilities import check_gradient

for r in check_gradient(to_params, fe, n_elements=5, rel_step=1e-4, tol=1e-3):
    print(r.name, f"{r.max_rel_error:.1e}", "OK" if r.ok else "MISMATCH")
```
- It compares the gradient in use (autograd, or analytical if `to_params.Gradient` is `GRADIENT_METHOD.ANALYTICAL`) with central finite differences on `n_elements` random elements.
- The comparison is made at a random interior design (densities 0.3–0.9).
- It perturbs MMA's own design variables, so the finite differences pass through the same density filter (with symmetry and extrusion) and, if on, the Heaviside projection (at β = 1, where a run starts) as the optimizer's gradient. `through_filter=False` perturbs the physical density instead, which checks only the response and the solve.
- The GUI's **Check gradient** button runs the same function.

A mismatch points to one of:
- a non-smooth quantity (`max`, `abs`, `where`);
- a function that uses NumPy on the solution;
- a finite-difference step that's too large or too small for a strongly nonlinear quantity. Try `rel_step=1e-5`.

## Cost estimate
```python
from pyto.topopt.capabilities import cost_estimate
print(cost_estimate(spec, selections))
# "Per iteration: 1 forward FE solve + 2 adjoint solves."
```
It counts the linear solves one MMA iteration needs. With analytical gradients it counts only what the analytical path needs (0 for compliance, 1 for p-norm stress).

## Measured cost
`tests/test_gradient_cost.py` times forward, forward+backward and backward separately for a given problem size:
```bash
PYTHONPATH=src python tests/test_gradient_cost.py --ndof 25000 --repeat 10 --solver PARDISO
```
Results for this repository are summarized in [Validation](../reference/validation.md).

## Comparing the two on the benchmarks
`pyto.examples_benchmarks.gradient_comparison` optimizes every sweep problem that has analytical sensitivities twice, once with each gradient method, with MMA.
- **Same optimization as the sweep:** the same problem, mesh, solver, iteration limit and binarization (`build_benchmark` and the same driver call). Only `to_params.Gradient` differs, so both runs take the same path. Every pair is checked: same number of FE solves, final objectives within 1e-6.
- **Measured per iteration:** each iteration is split into
  - FE solve (assembly + linear solve);
  - gradient (AD backward pass or analytical formula);
  - adjoint solves;
  - optimizer update.

  Each run also records peak memory per iteration, the AD graph size and solve residuals. Reported values are medians over iterations 2 and later; iteration 1 holds the driver's set-up.
- **Repeats and order:** `--repeats N` runs each pair N times (3 for publication numbers). AD and analytical alternate, so slow drifts of the machine affect both alike. The report shows the spread across repeats.
- **Quiet machine:** before each run the script waits, for at most `--wait-quiet` minutes, until other programs use less than `--max-busy` % of the CPUs. A run that still starts on a busy machine is marked ⚠ in the report.
- **Isolation:** every run is its own process, under a memory cap.

Run it from the repository root, with nothing else running:
```bash
export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
nohup systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \
    python -m pyto.examples_benchmarks.gradient_comparison --repeats 3 > gradient_comparison.log 2>&1 &
python -m pyto.examples_benchmarks.gradient_comparison --problems LBracketTopLoad_Stress_Vol MBBBeam --max-iters 5
python -m pyto.examples_benchmarks.gradient_comparison --solver DPCG        # both modes with DPCG
```
Output, in `Results/GradientComparison_<tag>/`:
- `runs.csv`: one row per run, with per-iteration medians, totals, memory, residuals and the CPU use before start.
- `pairs.csv` and `GRADIENT_COMPARISON_<tag>.md`: per method and problem, AD and analytical side by side:
  - time per iteration, and AD's extra time in %;
  - the phase split for each method;
  - peak memory and AD graph size;
  - the same-path check and the spread across repeats.
- `runs/<method>/<problem>/<gradient>_r<k>/`: `iterations.csv`, `result.json` and the topology image of each run.

A stopped comparison continues with the same `--tag`. `--max-iters N` shortens every run for a quick trial; leave it out for full runs.

---
Next: [Thermal and thermo-structural problems](thermal-and-thermostructural.md) · [Back to contents](../README.md)
