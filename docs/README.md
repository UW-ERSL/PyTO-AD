# PyTO documentation

PyTO is a Python library for **3D topology optimization** on voxel (hexahedral) meshes, with a desktop GUI on top. These pages explain how to use it from code, how to use the GUI, how to work with results, and how to extend it with new physics, filters, objectives and optimization methods.

New here? Start with **[Installation](getting-started/installation.md)**, then the **[code quick start](getting-started/quickstart-code.md)**.

---

## Getting started
| Page | What you learn |
|---|---|
| [Installation](getting-started/installation.md) | Set up the conda environment and check it |
| [Quick start: code](getting-started/quickstart-code.md) | Your first optimization in about 30 lines of Python |
| [Quick start: GUI](getting-started/quickstart-gui.md) | Your first optimization by clicking |
| [Quick start: notebook](getting-started/quickstart-notebook.md) | Run any benchmark problem from `main.ipynb` |

## Part 1: Using PyTO from Python
| Page | What you learn |
|---|---|
| [Concepts](code/concepts.md) | The objects (mesh, boundary conditions, FE solver, density, responses, formulation, driver) and how they connect |
| [A problem from scratch](code/problem-from-scratch.md) | Build mesh, material, supports, loads and solver yourself, then optimize |
| [Selecting nodes and faces](code/selecting-nodes-and-faces.md) | Find the nodes and elements for supports, loads and named selections |
| [Formulations with specs](code/formulation-with-spec.md) | Describe objective, constraints and method as data (`OptimizationSpec`) |
| [Expressions](code/expressions.md) | Write objectives and constraints as formulas; full list of quantities |
| [User-defined functions](code/user-defined-functions.md) | Any objective or constraint as a Python (torch) function |
| [Running the optimizers](code/running-optimizers.md) | MMA and OC (Pareto and LevelSet as secondary methods): arguments, results, callbacks |
| [Gradients](code/gradients.md) | Automatic differentiation vs analytical gradients, gradient checks, cost |
| [Thermal and thermo-structural problems](code/thermal-and-thermostructural.md) | Heat conduction and coupled problems from code |
| [Benchmarks from code](code/benchmarks-from-code.md) | Use the 46 ready-made problems, run sweeps |

## Part 2: Using the GUI
| Page | What you learn |
|---|---|
| [Overview](gui/overview.md) | The main window and the order of work |
| [Geometry and materials](gui/geometry-and-materials.md) | Load an STL, choose a material, units |
| [Loads and supports](gui/loads-and-supports.md) | Pick faces or nodes, box select, torque, body force, thermal loads |
| [Analysis](gui/analysis.md) | Mesh, analyse, and look at deformation, stress and temperature |
| [Optimization setup](gui/optimization-setup.md) | Objective, constraints, method, gradient; validate and run |
| [Manufacturing options](gui/manufacturing-options.md) | Symmetry, cyclic symmetry, extrusion, keep-solid |
| [Projects](gui/projects.md) | Save and load your work |
| [From the GUI to code](gui/gui-to-code.md) | Reproduce a GUI setup in a Python script |

## Part 3: Results and post-processing
| Page | What you learn |
|---|---|
| [History and convergence](postprocessing/history-and-convergence.md) | What the optimizer returns and how to read it |
| [Fields and plots](postprocessing/fields-and-plots.md) | Deformation, stress, temperature and density pictures |
| [Thresholding and evaluation](postprocessing/thresholding-and-evaluation.md) | Turn the grey design into a 0/1 part and report honest numbers |
| [Export](postprocessing/export.md) | VTU for ParaView, smooth STL, tetrahedral mesh |
| [Comparing results](postprocessing/comparing-results.md) | Benchmark sweep outputs and comparison reports |

## Part 4: Extending PyTO
| Page | What you learn |
|---|---|
| [Architecture](extending/architecture.md) | How the package is organized and how one optimization step flows |
| [Add a response](extending/add-a-response.md) | A new quantity usable in expressions and the GUI |
| [Add an analytical sensitivity](extending/add-a-manual-sensitivity.md) | A derived gradient, verified against autograd |
| [Add a filter](extending/add-a-filter.md) | A new filter, projection or manufacturing constraint |
| [Add a physics](extending/add-a-physics.md) | A new FE solver that the optimizers can use |
| [Add an optimization method](extending/add-an-optimization-method.md) | A new driver next to (or instead of) MMA |
| [Add a material model](extending/add-a-material-model.md) | A new density-to-stiffness interpolation |
| [Machine-learning approaches](extending/ml-approaches.md) | Neural reparameterization (TOuNN-style) and other ML ideas |
| [Testing your extension](extending/testing-your-extension.md) | The checks that show your addition is correct |

## Part 5: Theory
| Page | What you learn |
|---|---|
| [Density method](theory/density-method.md) | SIMP interpolation, density filter, Heaviside projection |
| [Sensitivities](theory/sensitivities.md) | The adjoint method and automatic differentiation through the solve |
| [Stress](theory/stress.md) | Relaxed stress, p-norm aggregation and its adjoint |
| [Thermo-structural coupling](theory/thermo-structural.md) | Thermal strain, coupled compliance, what changes for optimization |

## Reference
| Page | What you find |
|---|---|
| [Benchmark problems](reference/benchmarks.md) | Every benchmark problem and how to run it |
| [Validation](reference/validation.md) | Comparison with published results, gradient accuracy and cost |
| [Troubleshooting](reference/troubleshooting.md) | Error messages and what to do |
| [Testing](reference/testing.md) | Running the test suite |

---

**Runnable examples** used throughout these pages are in [`examples/`](examples/). Run any of them from the repository root, for example `python docs/examples/quickstart.py`.

**License:** PyTO is open source under the [GNU General Public License v3.0](../LICENSE).

**Not available yet:** transient thermal analysis, modal analysis, tetrahedral FE analysis and large-deformation analysis. Some files for these exist in the source tree, but they are unfinished and not supported.
