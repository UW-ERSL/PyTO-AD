# Reproducing the paper

Every number, table and figure of the PyTO-AD paper comes from a script in this repository. This page lists them, with the command to run and the folder each one writes to.

Results are not part of the repository. Every script writes to a folder under `Results/` (or `docs/paper/figures/generated/` for figures), and those folders are git-ignored. Run the scripts in the order of this page when you start from an empty `Results/`: some figures read the output of earlier runs.

## Environment

Install the package in a fresh environment, or use the conda environment of [Installation](../getting-started/installation.md):

```bash
pip install -e .                      # from the repository root; or: conda activate PyToLib
```

Run every command from the repository root (except the `docs/paper/fig_gen` scripts, see below), with:

```bash
export PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
```

## Machine notes

- Run one heavy job at a time. Do not start a second benchmark, test run or figure render while a timing run is going.
- Timing runs need an idle machine. `gradient_comparison` waits before each run until other programs use less than 5 % of the CPUs, and flags runs that started on a busy machine.
- PyTorch uses all cores by default. The paper's runs used 12 threads on a 12-core (20-thread) workstation with 78 GB of RAM.
- Large runs can use tens of GB. Start them in their own memory cgroup, so a run that runs out of memory is stopped without taking the desktop down:

  ```bash
  systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 python -m <module> <arguments>
  ```

- Long runs can be stopped and continued: `gradient_comparison` and `solver_scaling` skip runs already recorded under the same `--tag`.

## Benchmark sweep

The benchmark tables and the design galleries come from the full sweep of the benchmark problems (MMA, AD gradients).

| What | Command | Output |
|---|---|---|
| Full sweep, every problem in its own process (hours) | `python -m pyto.examples_benchmarks.topopt_run_benchmarks` | `Results/Results_<date>/<category>/<method>/` |
| One problem, or a few | `main.ipynb`, or `run_all_benchmarks_isolated(problems=...)` | same |

See [Benchmarks from code](../code/benchmarks-from-code.md) and [Quick start: notebook](../getting-started/quickstart-notebook.md).

## AD against analytical gradients: time and memory

`pyto.examples_benchmarks.gradient_comparison` runs every benchmark problem that has an analytical gradient twice, once with AD and once with the analytical gradient, and compares time per iteration, peak memory and the final objective. Each run is its own process. Use a different `--tag` for each session; the tags below are examples.

| What | Command | Output |
|---|---|---|
| Time per iteration, session 1 (no instrumentation) | `python -m pyto.examples_benchmarks.gradient_comparison --plain --repeats 3 --tag plain_s1` | `Results/GradientComparison_plain_s1/` |
| Time per iteration, session 2 (start it hours after session 1) | `python -m pyto.examples_benchmarks.gradient_comparison --plain --repeats 3 --tag plain_s2` | `Results/GradientComparison_plain_s2/` |
| Per-phase times, AD graph size, peak memory (instrumented) | `python -m pyto.examples_benchmarks.gradient_comparison --repeats 1 --tag instr` | `Results/GradientComparison_instr/` |
| Memory over 10 iterations | `python -m pyto.examples_benchmarks.gradient_comparison --repeats 1 --max-iters 10 --tag memory` | `Results/GradientComparison_memory/` |
| One-thread runs (bit-identical repeats) | `OMP_NUM_THREADS=1 python -m pyto.examples_benchmarks.gradient_comparison --problems LBracketTopLoad_Stress_Vol --repeats 3 --tag 1thread` | `Results/GradientComparison_1thread/` |

Notes:

- Each session runs every problem 3 times per gradient method and takes many hours. `--problems` and `--max-iters` give short trial runs.
- The script has no thread option. PyTorch reads `OMP_NUM_THREADS` when it starts, and every run process inherits it; the report header shows the thread count used (`torch threads`).
- OC instead of MMA: add `--methods OC`.
- Each output folder holds `runs.csv`, `pairs.csv`, a Markdown report and one folder per run.

The per-iteration measurement is in `pyto.examples_benchmarks._instrument`.

Two small analyses explain details of these runs:

| What | Command | Output |
|---|---|---|
| Threshold tie on ThreeHoleBracketThick (two 0/1 designs from one continuous design) | `python docs/paper/analysis/tie_check.py` | `Results/BinarizationTie/` |
| Every step of an iteration timed separately (where the time goes; about 10 min) | `python docs/paper/analysis/phase_check.py` | `Results/C2_phase_check/` |

## Solver scaling

`pyto.examples_benchmarks.solver_scaling` refines the mesh of EdgeCantilever step by step and runs 10 MMA iterations per size with each linear solver (PARDISO, DPCG, DPCG on the GPU) and each gradient method. The whole ladder takes days.

```bash
systemd-run --user --scope -p MemoryMax=55G -p MemorySwapMax=0 \
    python -m pyto.examples_benchmarks.solver_scaling --tag scaling
python -m pyto.examples_benchmarks.solver_scaling --problems EdgeCantilever --dofs 25000 50000 --max-iters 3   # short trial
```

Output: `Results/SolverScaling_<tag>/` (summary CSV, checks CSV, report, plots, one folder per run).

## Finite-difference step check

AD gradients against central finite differences over nine decades of step, for compliance, displacement p-norm, p-norm stress and the inverter output.

```bash
python -m pyto.examples_benchmarks.gradient_fd_check
```

Output: `Results/GradientFD/fd_vs_step.csv`. The figure is drawn by `make_paper_figures`.

## Review designs

0/1 designs of thermo-elastic BiClamp at 4 and 10 K and of the stress-constrained L-brackets (with and without projection).

```bash
python -m pyto.examples_benchmarks.review_designs --tag review            # or: --cases BiClampDT4 BiClampDT10
```

Output: `Results/ReviewDesigns_<tag>/<case>/` and `summary.csv`.

## AD showcase

One L-bracket with four objective strings (compliance, p-norm stress, displacement, mass with a compliance limit), plus the inverter and EdgeCantilever. It feeds the formulations figure and two cases of the body-fitted re-analysis.

```bash
python -m pyto.examples_benchmarks.ad_showcase run        # optimizations, about 35 min
python -m pyto.examples_benchmarks.ad_showcase render     # GIFs and PNGs from the saved runs
```

Output: `Results/AD_Showcase/` (another folder: set `PYTO_AD_SHOWCASE_OUT`).

## Body-fitted re-analysis

`pyto.examples_benchmarks.body_fitted_study` turns optimized voxel designs into smooth STL parts, meshes them with tetrahedra and analyses them again. It needs the AD showcase (L-bracket and EdgeCantilever) and the GUI session (see below) first.

```bash
python -m pyto.examples_benchmarks.body_fitted_study all          # optimize, recover, analyze, images
python -m pyto.examples_benchmarks.body_fitted_study voxeltet     # like-for-like check
python -m pyto.examples_benchmarks.body_fitted_study table        # prints the paper's table
```

Stages can be run one at a time (`optimize`, `recover`, `analyze`, `voxeltet`, `images`, `table`), for all cases or for the cases named after the stage. Output: `Results/BodyFitted/<case>/` (another folder: set `PYTO_BODY_FITTED_OUT`).

## Data figures

`pyto.examples_benchmarks.make_paper_figures` draws the time, memory, scaling and finite-difference plots from the recorded results. It runs no optimization. The result folders it reads are named at the top of the script; change them there to use your own runs.

```bash
python -m pyto.examples_benchmarks.make_paper_figures
```

Output: `Results/PaperFigures/fig_*.pdf` and `.png`, and `figures_manifest.csv` (figure, data source, notes).

## Figure scripts in `docs/paper/fig_gen`

The diagrams, renders and design figures are drawn by the scripts in `docs/paper/fig_gen`. Run them from that folder:

```bash
cd docs/paper/fig_gen
export PYTHONPATH=../../../src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true
python fig01_graphical_abstract.py
```

Each script writes a PDF and a PNG to `docs/paper/figures/generated/` (git-ignored; set `PYTO_PAPER_FIGURES` to write elsewhere). Optimization runs are cached in `docs/paper/fig_gen/_cache/` (git-ignored). A script reruns an optimization only when its cache file is missing; delete a cache file to rerun it. The shared style is in `_style.py` and `_render.py`, and the figure conventions are in `docs/paper/fig_gen/FIGURE_GUIDE.md`.

| Script | Figure | Inputs |
|---|---|---|
| `run_round2_designs.py threehole ge` | none (runs the fine-mesh designs) | writes `_cache/round2_threehole.npz` and `_cache/round2_ge.npz` |
| `fig01_graphical_abstract.py` | graphical abstract: STL, stress, design, recovered part | `_cache/round2_threehole.npz` |
| `fig_intro_gegrabcad.py` | GE bracket in the introduction | `_cache/round2_ge.npz` |
| `fig02_quickstart.py` | the two quick-start designs | runs `docs/examples/quickstart.py` and `quickstart_displacement.py` at a finer mesh; cached in `_cache/quickstart_*.npz` |
| `gui_session_threehole.py` | none (records the GUI session) | writes `Results/GUISession_ThreeHole/` |
| `fig04_05_gui.py` | GUI formulation and STL recovery | `Results/GUISession_ThreeHole/raw/` |
| `fig06_teaching_grid.py` | MBB beam, filter radius against SIMP exponent | 9 runs, cached in `_cache/teaching_*.npz` |
| `fig13_formulations.py` | one L-bracket, four objectives | `Results/AD_Showcase/` |
| `fig_sweep_galleries.py` | design galleries and thermo-structural designs | the benchmark sweep and the review designs (folders named at the top of the script) |
| `fig17_casestudy.py` | body-fitted re-analysis of recovered parts | `Results/BodyFitted/` |

The GUI session drives the real GUI on the screen for a few minutes. It needs a display (`DISPLAY` set, not off-screen) and runs at double scale:

```bash
cd docs/paper/fig_gen
QT_SCALE_FACTOR=2 PYTHONPATH=../../../src python gui_session_threehole.py
```

`--keep-recovered` records the screen captures again but keeps the recovered STL, project and logs of the earlier session.

## Quick-start listings

The listings of the paper are the tested scripts in `docs/examples/`:

```bash
python docs/examples/quickstart.py                 # compliance, three-hole bracket
python docs/examples/quickstart_displacement.py    # the same with a displacement objective
```

They are part of the test suite ([Testing](testing.md)).
