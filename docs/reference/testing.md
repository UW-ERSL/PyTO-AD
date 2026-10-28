# Testing

This page explains how to run PyTO's test suite and what each test file covers. It is for contributors and anyone checking an installation.

## Run
```bash
pip install -r requirements-dev.txt
PYTHONPATH=src MPLBACKEND=Agg PYVISTA_OFF_SCREEN=true QT_QPA_PLATFORM=offscreen pytest tests -q
```
- **Time:** about a minute. `pytest tests --collect-only -q` lists the tests.
- **Environment variables:** they keep plots and Qt windows off screen.
- **A subset:** `pytest tests/test_responses.py -q`, or `-k name` for single tests.
- **Documentation examples:** `tests/test_docs_examples.py` runs every script in `docs/examples/`, so the docs stay correct.

## What each file covers
| File | Covers |
|---|---|
| `conftest.py` | shared fixtures: small structural (ShortCantileverTipLoad) and thermal (HeatPlate) problems |
| `test_sparse_solve_gradcheck.py` | the differentiable linear solve (`torch.autograd.gradcheck`), PARDISO factor reuse, DPCG vs direct, `adjoint_solve` reusing the forward factorization |
| `test_qoi_gradients.py` | built-in objectives and constraints vs finite differences |
| `test_responses.py` | the response registry: values vs independent calculations, gradients vs finite differences |
| `test_expressions.py` | the expression compiler: values, gradients, rejected input, error messages |
| `test_spec.py` | `OptimizationSpec`: validation, compilation, normalization, JSON round trip, progress labels |
| `test_user_defined_qoi.py` | Python-function objectives and constraints: gradients, error messages, driver support |
| `test_capabilities.py` | method support rules, gradient check, cost estimate |
| `test_manual_sensitivities.py` | analytical gradients vs autograd, full MMA runs both ways, `GRADIENT_METHOD`, and the AD-vs-analytical benchmark script |
| `test_drivers_smoke.py` | each driver runs a few iterations and improves the objective |
| `test_thermoelastic.py` | thermal force, coupled solve and its gradients |
| `test_gradient_cost.py` | timing of forward and backward (a smoke test under pytest; a benchmark as a script) |
| `test_stl_recovery.py` | the STL-recovery CNN |
| `test_imports_smoke.py` | every module imports |
| `test_gui_api.py`, `test_gui_optimization_setup.py`, `test_gui_topopt_window.py`, `test_gui_node_loads.py` | GUI logic, and real windows driven headless: formulation panel, optimization window, node and box selection |
| `test_docs_examples.py` | the scripts in `docs/examples/` |

## Writing tests
See [Testing your extension](../extending/testing-your-extension.md): what to compare against, finite-difference tips, and fixtures.

---
[Back to contents](../README.md)
