# Formulations with specs

This page explains `OptimizationSpec`: a formulation (objective, constraints, method, filtering, manufacturing options) written as plain data. The GUI edits one, project files store one, and `compile_spec` turns one into the `TOParams` the optimizers read. Use specs when you want formulas, validation, or a formulation you can save.

## The parts
```python
from pyto.topopt.spec import (ConstraintSpec, ManufacturingSpec, MethodSpec, ObjectiveSpec, OptimizationSpec,
                              RegularizationSpec, compile_spec, validate)

spec = OptimizationSpec(
    objective=ObjectiveSpec("Compliance()", "minimize"),                 # expression, "minimize" | "maximize"
    constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.3),          # expression, "<=" | ">=", bound
                 ConstraintSpec("StressFailureFactor()", "<=", 0.5, enabled=False)],   # kept but unused
    method=MethodSpec("MMA", max_iterations=150, move_limit=0.2, gradient="autodiff"),
    regularization=RegularizationSpec(filter_radius=1.5, heaviside=False),
    manufacturing=ManufacturingSpec(extrude="", symmetry=["X"], cyclic_z=0, keep_selection=""))
```

| Part | Fields |
|---|---|
| `ObjectiveSpec` | `expression`, `sense` |
| `ConstraintSpec` | `expression`, `op` (`<=` / `>=`), `bound` (SI units), `enabled` |
| `MethodSpec` | `name` (`MMA`), `max_iterations`, `move_limit`, `gradient` (`autodiff` / `analytical`; old name `manual` accepted) |
| `RegularizationSpec` | `filter_radius` (in elements), `heaviside` (smooth 0/1 projection, MMA) |
| `ManufacturingSpec` | `extrude` (`""`, `X`, `Y`, `Z`), `symmetry` (planes normal to these axes), `cyclic_z` (sectors about z), `keep_selection` (name of an element selection kept solid) |

The expression language is described in [Expressions](expressions.md). An expression can also reference a Python function: `"python:path/to/file.py::function_name"`; see [User-defined functions](user-defined-functions.md).

## Selections
Names used inside expressions (`Displacement(Tip, y, mean)`) and in `keep_selection` come from a dict you pass along:
```python
selections = {"Tip": {"nodes": tip_nodes}, "Flange": {"elements": flange_elements}}
```
`All` always means the whole model. How to find the nodes and elements: [Selecting nodes and faces](selecting-nodes-and-faces.md).

## Validate
```python
issues = validate(spec, physics="structural", selections=selections)     # or "thermal", "thermo-structural"
for issue in issues:
    print(issue)            # "ERROR: ..." stops compile_spec; "WARNING: ..." is advice
```
`validate` checks, without solving anything:
- every expression parses;
- every quantity exists for the physics;
- selections exist;
- bounds are usable (a `>=` bound must be > 0);
- the problem is well posed. For example, minimizing volume with no constraint is an error, and minimizing compliance with no volume or mass limit gets a warning.

## Compile
```python
to_params = compile_spec(spec, selections, base=None, allow_code=False)
```
- **`base`**: an existing `TOParams` whose other settings (mesh size, material model, thermal settings) are kept. Benchmark problems pass their own.
- **`allow_code=True`**: required when an expression references a Python file. Loading the file runs it, so this is opt-in.
- It raises `ValueError` for a formulation that can't be compiled.

What compilation does:
1. **Built-in quantities** (`Compliance()`, `VolumeFraction()`, `Mass()`, `StressPNorm()`, and the constraints `MaxStress()`, `StressFailureFactor()`) become the built-in types (`TO_QOI.COMPLIANCE`, ...). That keeps the benchmarks' tested behaviour, and allows [analytical gradients](gradients.md).
2. **Anything else** becomes a `TO_QOI.GFUNCTION`: a torch function compiled from the expression, differentiated by autograd (MMA).
3. **Constraints are normalized** to the form `g ≤ 0` that MMA expects:

   | You write | Stored as |
   |---|---|
   | `f <= b` with b > 0 | `f / b − 1 ≤ 0` |
   | `f <= b` with b ≤ 0 | `f − b ≤ 0` |
   | `f >= b` (b > 0 required) | `1 − f / b ≤ 0` |
   | `maximize f` | minimize `−f` |
4. `ObjectiveLabel` and `ConstraintLabels` record what you wrote, so the iteration printout shows `Displacement(Tip, y, mean): -0.00128 <= 2e-07  ok` instead of the internal form.

## Save and load
```python
text = spec.to_json()
spec2 = OptimizationSpec.from_json(text)      # also from_dict / to_dict
```
Older saved specs without newer fields (such as `gradient`) load with the defaults.

## From settings back to a spec
`spec_from_to_params(to_params, method="MMA")` describes an existing `TOParams` (for example a benchmark's) as a spec, returning `(spec, selections)`. Python-function objectives can't be turned back into text and raise `NotImplementedError`.

## Without a spec
Setting `TOParams` directly is equivalent for the built-in types:
```python
from pyto.topopt.common import TOParams, TO_QOI

p = TOParams()
p.Objective = (TO_QOI.COMPLIANCE, None)
p.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3),         # (type, parameter, limit): value <= limit
                 (TO_QOI.VOLUME_FRACTION_MIN, None, 0.1),     # mean density >= 0.1
                 (TO_QOI.PNORM_STRESS, None, 2e8)]            # p-norm von Mises stress <= 200 MPa
p.RelativeFilterRadius = 1.5
p.XSymmetry = True
p.MaxIterations = 100
```
Other useful `TOParams` fields:

| Field | Meaning |
|---|---|
| `HeavisideProjection`, `HeavisideBetaInterval`, `HeavisideBetaMax` | smooth 0/1 projection; β doubles every interval up to the max |
| `ExtrudeX/Y/Z`, `XSymmetry/YSymmetry/ZSymmetry`, `X/Y/ZAxisAngularSymmetry` | manufacturing filters |
| `ElemsToKeep` | elements kept solid |
| `KeepFixedElems` | read by the benchmark builders: they add the elements at supports to `ElemsToKeep` |
| `Gradient` | `GRADIENT_METHOD.AUTODIFF` (default) or `GRADIENT_METHOD.ANALYTICAL` |
| `ThermalReferenceTemperature`, `ConductivityPenalty`, `ConductivityVoidRatio` | thermo-structural settings |

---
Next: [Expressions](expressions.md) · [Back to contents](../README.md)
