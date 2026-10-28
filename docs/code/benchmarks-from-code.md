# Benchmarks from code

This page shows how to use PyTO's 46 ready-made problems: build one, run it with any method, and run many in a sweep. It is for anyone checking results, comparing methods, or looking for a starting point close to their own problem.

## Build one problem
```python
from pyto.examples_benchmarks.topopt_run_benchmarks import build_benchmark
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples

fe, to_params, fea_mode = build_benchmark(StructuralTOExamples.MBBBeam)
```
`build_benchmark` returns:
- the FE solver: structural, thermal or thermo-structural;
- the problem's `TOParams`: formulation, filter, symmetry, iterations;
- the `FEA_MODE`.

It picks PARDISO below 100 000 unknowns and DPCG above. Pass `solver=Solvers.SPSOLVE` to override, for example on macOS.

It is exactly what the benchmark sweep and `main.ipynb` use, so a result from your script can be compared with a sweep result directly.

For a **smaller** version while experimenting, call the problem builder with `nDOFDesired`:
```python
from pyto.examples_benchmarks.topopt_structural_benchmarks import getStructuralTOProblem

mesh, material, bc, body_force, to_params = getStructuralTOProblem(StructuralTOExamples.MBBBeam, nDOFDesired=5000)
```
(`getThermalTOProblem` returns the same; `getThermoStructuralTOProblem` returns `mesh, material, structural_bc, thermal_bc, body_force, to_params`.) Then create the solver yourself, as in [Quick start: code](../getting-started/quickstart-code.md).

## Run it
```python
from pyto.topopt.drivers.mma import topopt_mma

u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params,
                                                 maxMMAIterations=to_params.MaxIterations)
```
Any method works where it applies; see [Running the optimizers](running-optimizers.md). The problem's formulation can be changed before running, e.g. `to_params.Constraints = [...]`, or by compiling a spec with `base=to_params`.

## The problems
Grouped as in the sweep; every name is a member of the enum shown:

| Group (list in `topopt_run_benchmarks`) | Problems |
|---|---|
| `benchmarks_structural_2_5D_problems_1` | `StructuralTOExamples`: Mitchell, CantileverTipLoad, CantileverMidLoad, MBBBeam, Bridge, TwoBar |
| `benchmarks_structural_2_5D_problems_2` | LBracketTopLoad, LBracketMidLoad, TorquePlate, DistributedLoad, ThreeHoleBracket |
| `benchmarks_structural_3D_problems` | EdgeCantilever, ThreeHoleBracketThick, Multiload, LBracketThickTopLoad, LBracketThickMidLoad, Table |
| `benchmarks_structural_noncompliance` | CantileverMidLoadVolumeCompliance, LBracketTopLoad_Stress_Vol, LBracketTopLoad_Vol_Stress, LBracketMidLoad_Vol_Stress, LBracketTopLoad_Mass_StressFF, Inverter (compliant mechanism) |
| `benchmarks_structural_bodyforce` | GravityPlate (self-weight), CentrifugalPlate |
| `benchmarks_structural_casestudies` | GEGrabCAD |
| `benchmarks_thermal_2_5D_problems` | `ThermalTOExamples`: HeatPlate, FourCornersThermal, BridgeThermal |
| `benchmarks_thermostructural_problems` | `ThermoStructuralTOExamples`: BiClamp, BiClampDT0, BiClampDT4, BiClampDT10, MBBBeam, MBBBeamNoHeat |
| `benchmarks_not_in_sweep` (buildable, not run by default) | Mitchell_2, Mitchell_3, ShortCantileverTipLoad, ShortCantileverMidLoad, TensilePlate, EdgeCantileverLargeDOF, EdgeCantileverConstraintMatrix, LBracketThickTopLoad_Vol_Stress, LBracketThickTopLoad_Stress_Vol, LBracketMidLoad_Vol_StressFailureFactor_Compliance, KnuckleAssembly |

The full table, with physics and formulation per problem, is in [Benchmark problems](../reference/benchmarks.md). To list the members yourself: `[m.name for m in StructuralTOExamples]`.

## Run a sweep
Every (method, problem) pair runs in its own process: memory is fully returned between runs, which matters for long sweeps. From the repository root:
```bash
export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
nohup python -m pyto.examples_benchmarks.topopt_run_benchmarks > benchmark_run.log 2>&1 &     # everything: hours
```
A subset:
```python
from pyto.examples_benchmarks.topopt_run_benchmarks import *
run_all_benchmarks_isolated(optimizationMethods=[TO_METHODS.DENSITYMMA],
                            problems=benchmarks_structural_noncompliance)
```
Results go to `Results/Results_<date>/<category>/<method>/`:
- one image per problem;
- a summary CSV per method;
- combined summary tables.

The sweep runs **MMA**. Reading and comparing results: [Comparing results](../postprocessing/comparing-results.md).

To time automatic differentiation against analytical sensitivities on the same problems (MMA, each run in its own process), use `python -m pyto.examples_benchmarks.gradient_comparison`; see [Gradients](gradients.md#comparing-the-two-on-the-benchmarks).

### Solver scaling study
`python -m pyto.examples_benchmarks.solver_scaling` finds how far each linear solver goes with automatic differentiation and with analytical sensitivities, and what each costs in time and memory. It runs three structural problems: EdgeCantilever, the thick L-bracket with p-norm stress, and GEGrabCAD. Each problem is refined from 25k up to 3.2M dofs, and every step runs an MMA optimization capped at 10 iterations (`--max-iters`) once per solver (PARDISO, SPSOLVE, DPCG, DPCG on the GPU, PCG) and gradient method.

Runs have no time limit by default; with `--time-limit SECONDS` every run is stopped after that time. A stopped run's time per iteration comes from the iterations it completed, and the report shows how many of the planned evaluations ran plus an estimate for the full run. A solver/gradient pair stops climbing at its first failure: fewer than 3 iterations within the time limit, the memory cap (70 GB by default) or out of memory, a linear solve that does not converge, or an error.

Each iteration records:
- time per phase;
- peak memory;
- the size of the AD graph;
- solver factor memory;
- CG iterations and solve residuals;
- the objective.

At each mesh size, iteration 1 of every solver and gradient method is compared with the PARDISO automatic-differentiation run.

Results go to `Results/SolverScaling_<date>/`: summary CSV, checks CSV, a report with plots, and per-run iteration histories.

Run it from the repository root, inside a memory cgroup:

```bash
export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
nohup systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \
    python -m pyto.examples_benchmarks.solver_scaling > solver_scaling.log 2>&1 &
```

If it stops, start it again with the same `--tag`; finished runs are skipped. `--problems`, `--solvers`, `--gradients`, `--dofs`, `--max-iters` and `--time-limit` narrow the study.

## Adding a benchmark problem
1. Write a builder in `examples_benchmarks/hex_structural_examples.py` (or the thermal / thermo-structural file). Copy an existing one; each returns `mesh, mat_prop, bc, elem_body_force`.
2. Add a member to `StructuralTOExamples` (in `topopt_structural_benchmarks.py`), and a branch in `getStructuralTOProblem` that sets its `TOParams`.
3. Add it to one of the `benchmarks_*` lists in `topopt_run_benchmarks.py` to include it in the sweep.

---
Next: [Part 2: the GUI](../gui/overview.md) · [Part 4: extending](../extending/architecture.md) · [Back to contents](../README.md)
