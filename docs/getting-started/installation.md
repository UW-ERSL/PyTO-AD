# Installation

This page sets up PyTO-AD and checks that it works. There are two ways to install it:

| | A. Clone the repository | B. Install with pip from GitHub |
|---|---|---|
| Use it for | everything: the GUI, your own scripts, the tutorial problems, the examples, the tests and the paper's studies | your own STL parts, from Python or the GUI |
| GUI | `python run_gui.py` (finds the `PyToLib` environment itself) | `pyto-gui` |
| Tutorial problems, `docs/examples`, tests | yes | no: they read STL files from `Models/` in the repository |
| Update | `git pull` | `pip install --force-reinstall ...` ([Updating](#updating)) |

Both need the same [requirements](#requirements). If unsure, use A.

## Requirements
- **Python 3.10.** Other versions may work but are not tested. Make a fresh environment for PyTO-AD rather than
  installing into a base or system Python: some dependencies have no ready-made packages for newer Pythons on every
  platform, and a base environment that is not writeable makes pip install for the user only, with its commands
  outside `PATH`.
- **conda** (Anaconda, Miniconda or Miniforge). The GUI launcher `run_gui.py` looks for a conda environment named
  **`PyToLib`** (set `PYTO_CONDA_ENV` to use another name).
- **Operating system:** tested on Linux (Ubuntu, x86-64, Python 3.10.19); all results in the paper were produced
  there, and the test suite runs there. On Windows (Intel CPU) the GUI and `main.ipynb` have been run; Windows is
  not part of the release tests.
- **macOS:** not part of the release tests. On an Apple-silicon Mac, with the installation below, `main.ipynb`
  runs and the GUI starts, but loading an STL file in the GUI does not work there yet. See [macOS](#macos).
  PARDISO is not available on macOS (see [Linear solvers](#linear-solvers)).
- **Pinned versions:** `requirements-lock.txt` lists the exact package versions of the tested environment
  (`pip install -r requirements-lock.txt`). With newer `pyvista` releases the voxel meshes differ slightly from the
  tested ones (see [Updating](#updating)).
- **Memory:** a few GB for problems up to about 100 000 degrees of freedom.

## A. Clone the repository
1. Get the code:
   ```bash
   git clone https://github.com/UW-ERSL/PyTO-AD
   cd PyTO-AD
   ```
2. Create and activate the environment:
   ```bash
   conda create -n PyToLib python=3.10
   conda activate PyToLib
   ```
3. Optional but recommended: install the **CPU build of PyTorch** first. It is much smaller than the default GPU
   build, and PyTO-AD doesn't need a GPU.
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   ```
4. Install the dependencies:
   ```bash
   pip install -r requirements.txt          # requirements-dev.txt adds pytest, for the test suite
   pip install -e .                         # optional: `import pyto` from anywhere, and the pyto-gui command
   ```
5. Check the installation:
   ```bash
   python run_gui.py --check
   python docs/examples/quickstart.py
   ```
   - The first command should print `OK: ... has all GUI modules.`
   - The second runs a small optimization in about 10 seconds and writes `quickstart_topology.png`.
6. Start the GUI:
   ```bash
   python run_gui.py
   ```
   Any Python can run it: if the current one lacks the GUI's modules, it restarts itself in the `PyToLib`
   environment. It adds `src/` to the import path, so no `PYTHONPATH` is needed. See
   [Quick start: GUI](quickstart-gui.md).

## B. Install with pip from GitHub
Use this to run PyTO-AD on your own STL parts without cloning the repository.

1. Create and activate a fresh environment (as in A, step 2), and optionally install the CPU build of PyTorch
   (A, step 3).
2. Install the package:
   ```bash
   pip install "pyto-ad @ git+https://github.com/UW-ERSL/PyTO-AD.git"
   ```
   This installs the `main` branch. For another branch or a tagged release, add it after the URL, e.g.
   `"pyto-ad @ git+https://github.com/UW-ERSL/PyTO-AD.git@v1.0"`.
3. Start the GUI with `pyto-gui`, or use the library from Python: the package is `pyto-ad`, imported as `pyto`.
4. Check what is installed:
   ```bash
   pip show pyto-ad
   ```

If `pyto-gui` is not found, the install went into a user folder that is not on `PATH` (pip says so in a warning);
activate the environment you installed into, or run `python -m pyto.gui.PyTOGUI` instead.

What you get and what you don't:

| Works after a pip install | Needs a clone (A) |
|---|---|
| Voxel meshing of your STL, FE solvers, MMA | Tutorial and benchmark problems (`getStructuralTOProblem`, ...) |
| AD gradients, expressions, `OptimizationSpec`, `check_gradient` | `docs/examples` quick-start scripts |
| Analytical gradients, STL recovery, VTU export | The benchmark sweep and the paper's studies and figures |
| The GUI (`pyto-gui`) | The test suite |

The right-hand column reads its STL files from `Models/` in the repository, which the package does not include.

## macOS
**Status: not part of the release tests.** On an Apple-silicon Mac, with the steps below, `main.ipynb` runs (with
SciPy's solver) and the GUI (`pyto-gui`) starts, but loading an STL file in the GUI does not work there yet.

What works, in this order:
1. **Use a native Apple-silicon conda** (Miniforge for arm64). An Intel Anaconda on an Apple-silicon Mac runs every
   environment under Rosetta, and several dependencies then fail (see the errors below). Check first:
   ```bash
   python -c "import platform; print(platform.machine())"     # must print arm64
   conda info | grep platform                                  # must show osx-arm64
   ```
   To install Miniforge: `curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`
   and `bash Miniforge3-MacOSX-arm64.sh`; then reopen the terminal.
2. **Create the environment and install PyTO-AD:**
   ```bash
   conda create -n pytoad python=3.10 -y
   conda activate pytoad
   pip install "pyto-ad @ git+https://github.com/UW-ERSL/PyTO-AD.git"
   ```
3. **Take SciPy and PyTorch from conda-forge**, replacing pip's builds:
   ```bash
   pip uninstall -y scipy torch
   conda install -c conda-forge scipy pytorch -y
   pyto-gui
   ```
   pip's SciPy for Python 3.10 does not load on recent macOS, and pip's PyTorch carries its own OpenMP runtime, which
   clashes with the one conda-forge's packages use. With both from conda-forge there is one OpenMP runtime.
4. **For `main.ipynb`:** install a Jupyter kernel in the same environment (`conda install -c conda-forge ipykernel`)
   and pick the `pytoad` environment as the notebook's kernel. The notebook uses SciPy's solver on macOS by itself.

Errors seen on the way, and their causes:

| Error | Cause | Fix |
|---|---|---|
| `xcrun: error: unable to load libxcrun ... need 'x86_64'` when pip runs `git` | Intel (x86_64) conda under Rosetta | arm64 conda (step 1); or `conda install git`, or install from the `.zip` URL |
| `Failed building wheel for llvmlite` / `numba` | no Intel-Mac build of llvmlite for that Python; pip tries to compile it | arm64 conda; or `conda install -c conda-forge numba llvmlite` |
| `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` (from torch) | the last Intel-Mac PyTorch is built for NumPy 1 | arm64 conda (current PyTorch) |
| `ImportError ... _spropack ... __DATA/__thread_bss has a zero-fill section type` (from scipy) | pip's SciPy build for Python 3.10 on recent macOS | SciPy from conda-forge (step 3) |
| `OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` | two OpenMP runtimes: pip's PyTorch and conda-forge's | PyTorch from conda-forge (step 3); do not set `KMP_DUPLICATE_LIB_OK`, which may give wrong results |

## Updating
**A clone (A):**
```bash
cd PyTO-AD
git pull
pip install -r requirements.txt            # only needed when requirements.txt changed
```
With `pip install -e .` the package follows the folder, so nothing else is needed.

**A pip install (B):** reinstall from GitHub. The version number may not have changed, so a plain
`pip install --upgrade` can skip it; `--force-reinstall` makes pip fetch the new code, and `--no-deps` keeps it
from downloading PyTorch and the other dependencies again:
```bash
conda activate PyToLib                     # the environment you installed into
pip install --force-reinstall --no-deps "pyto-ad @ git+https://github.com/UW-ERSL/PyTO-AD.git"
```
Add `@<branch or tag>` after the URL if you installed one (step B2). If the update also needs newer dependencies,
leave out `--no-deps`.

**pyvista:** PyTO-AD meshes STL files with `pyvista.voxelize` while that function exists (pyvista 0.46 and
earlier, as tested). Newer pyvista releases remove it; PyTO-AD then uses `DataSetFilters.voxelize` on the same
grid, which keeps slightly fewer boundary voxels, so results differ a little from the tested ones. To reproduce the
published results exactly, install the pinned versions (`pip install -r requirements-lock.txt`).

**Uninstalling (B):** `pip uninstall pyto-ad`.

## Optional packages
Listed at the end of `requirements.txt`; install them only if you need them.

| Package | For |
|---|---|
| `cupy-cuda12x` (or `cupy-cuda11x`) | GPU deflation solver |
| `petsc4py` | PETSc linear solvers (Linux/macOS) |
| `pypiwin32` | SolidWorks interface (Windows) |

## Linear solvers
Every FE solver takes a `solver=` argument from `pyto.solve.solvers.Solvers`.

| Solver | When |
|---|---|
| `PARDISO` | Default; fastest direct solver (Intel MKL). Not on macOS. |
| `SPSOLVE` | SciPy direct solver; works everywhere. Use it on macOS or if PARDISO is missing. |
| `PETSC` | PETSc solvers, if `petsc4py` is installed. |
| `DPCG` | Deflated preconditioned conjugate gradients, for large problems (the benchmarks use it above 100 000 dofs). Needs a `DeflationSolver` with its deflation matrix built for the mesh (`build_benchmark` does this); without one it runs as plain Jacobi PCG. |
| `PCG` | Jacobi-preconditioned conjugate gradients. |
| `SPLU` | the same as `SPSOLVE` during optimization. |
| `PYAMG` | analysis only; during **optimization** PyTO warns and falls back to `SPSOLVE`. |

## Using PyTO without installing it as a package
PyTO is used straight from the repository: the code lives in `src/pyto`.
- `run_gui.py` and the examples in `docs/examples/` add `src/` to the import path themselves.
- For your own scripts, either do the same (`sys.path.insert(0, "path/to/PyTO/src")`) or set `PYTHONPATH`:
  ```bash
  export PYTHONPATH=/path/to/PyTO/src        # Windows PowerShell: $env:PYTHONPATH="C:\path\to\PyTO\src"
  ```

## Running without a screen (servers, scripts)
PyVista opens windows by default. For scripts, benchmark runs and tests, set:
```bash
export MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true QT_QPA_PLATFORM=offscreen
```
or in Python, before plotting: `import pyvista; pyvista.OFF_SCREEN = True`.

---
Next: [Quick start: code](quickstart-code.md) · [Quick start: GUI](quickstart-gui.md) · [Back to contents](../README.md)
