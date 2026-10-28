# Installation

This page sets up a Python environment for PyTO and checks that it works. It is for everyone, whether you plan to use the GUI or the library.

## Requirements
- **Python 3.10.** Other versions may work but are not tested.
- **conda** (Anaconda, Miniconda or Miniforge). The GUI launcher looks for a conda environment named **`PyToLib`**.
- **Operating system:** tested on Linux (Ubuntu, x86-64, Python 3.10.19); all results in the paper were produced there. The code also targets Windows and macOS, but they are not part of the release tests. On macOS the PARDISO solver is not available, and PyTO uses SciPy's solver instead.
- **Pinned versions:** `requirements-lock.txt` lists the exact package versions of the tested environment (`pip install -r requirements-lock.txt`).
- **Memory:** a few GB for problems up to about 100 000 degrees of freedom.

## Steps

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

3. Optional but recommended: install the **CPU build of PyTorch** first. It is much smaller than the default GPU build, and PyTO doesn't need a GPU.
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cpu torch
   ```

4. Install PyTO's dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For running the test suite too, use `pip install -r requirements-dev.txt` instead. It includes everything above plus pytest.

5. Check the installation:
   ```bash
   python run_gui.py --check
   python docs/examples/quickstart.py
   ```
   - The first command should print `OK: ... has all GUI modules.`
   - The second runs a small optimization in about 10 seconds and writes `quickstart_topology.png`.

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
