# Expressions

This page describes the formula language for objectives and constraints, with the full list of quantities. The same language is used in code (`ObjectiveSpec`, `ConstraintSpec`) and in the GUI's **Expression...** fields. It is for anyone who wants an objective or constraint beyond the built-in ones, without writing a gradient.

An expression is parsed against a whitelist and never executed as code, so it is safe to store in project files. PyTorch autograd differentiates it through the FE solve.

## Using an expression
In code:
```python
from pyto.topopt.spec import ConstraintSpec, ObjectiveSpec, OptimizationSpec, compile_spec, validate

selections = {"Tip": {"nodes": tip_nodes}}
spec = OptimizationSpec(
    objective=ObjectiveSpec("Displacement(Tip, magnitude, pnorm) / 1e-3 + 0.1 * VolumeFraction()", "minimize"),
    constraints=[ConstraintSpec("StressFailureFactor()", "<=", 0.5)])
print(validate(spec, "structural", selections))
to_params = compile_spec(spec, selections)
```
To evaluate one expression on its own (e.g. to check a value):
```python
from pyto.topopt.expressions import compile_expression

expr = compile_expression("Displacement(Tip, y, mean)", selections)
value = expr(fe.solve(), x, fe)          # a torch scalar; x = density tensor (e.g. torch.ones(n))
print(expr.responses, expr.physics, expr.depends_on_solution)
```
In the GUI: **TopOpt Execute → Objective → Type: Expression...**, or the expression column of the **Constraints** tab; see [Optimization setup](../gui/optimization-setup.md).

## Quantities
| Quantity | Parameters | Physics | Unit | Meaning |
|---|---|---|---|---|
| `Compliance()` | none | structural, thermal, thermo-structural | J | uᵀKu, the usual stiffness measure; lower = stiffer. For thermal problems: thermal compliance. |
| `VolumeFraction(sel)` | element selection (optional) | all | – | mean density of the region, 0..1 |
| `Mass()` | none | structural, thermo-structural | kg | total mass |
| `Displacement(sel, component, aggregate, p)` | nodes; `x`/`y`/`z`/`magnitude`; `mean`/`sum`/`pnorm`; p | structural, thermo-structural | m | displacement at the selected nodes |
| `Temperature(sel, aggregate, p)` | nodes; `mean`/`sum`/`pnorm`; p | thermal, thermo-structural | K | temperature at the selected nodes |
| `ReactionForce(sel, component)` | nodes; `x`/`y`/`z` | structural, thermo-structural | N | sum of support reactions over the selected fixed nodes |
| `StrainEnergy(sel)` | element selection (optional) | structural | J | ½ uᵀKu of a region |
| `StressPNorm()` | none | structural, thermo-structural | Pa | p-norm of the von Mises stress, a smooth stand-in for the peak stress |
| `StressFailureFactor()` | none | structural, thermo-structural | – | `StressPNorm() / YieldStrength`; ≤ 1 keeps the smoothed peak below yield |
| `MechanicalCompliance()` | none | structural, thermo-structural | J | work of the mechanical loads only; leaves out the thermal-expansion load |
| `ThermalCompliance()` | none | thermal, thermo-structural | W·K | TᵀKT of the temperature field |
| `MaxStress()` | none | structural, thermo-structural | Pa | **constraint only, on its own** (`MaxStress() <= 2e8`): a p-norm rescaled every iteration toward the true maximum von Mises stress |

### Parameters
- **Selections** are names from the `selections` dict (code) or the GUI's load and support names (`Load1`, `Support1`, ...). `All`, or empty brackets, means the whole model.
- **Defaults:** `Displacement(Tip)` means `Displacement(Tip, magnitude, mean)`, and `Displacement(Tip, y)` means `Displacement(Tip, y, mean)`.
- **`pnorm` exponent:** 8 unless given, as in `Displacement(Tip, y, pnorm, 12)`. A larger p is closer to the true maximum but less smooth.
- **Keywords** work too: `Displacement(nodes=Tip, component=y, aggregate=pnorm, p=16)`. `selection=` is accepted as an alias of `nodes=`/`elements=`.

| Aggregate | Gives | Use for |
|---|---|---|
| `mean` | average over the selection | overall deflection or temperature of a face |
| `sum` | sum over the selection | totals |
| `pnorm` | smooth upper bound of the maximum | peak values (worst node) |

## Maths
| Allowed | Example |
|---|---|
| numbers, scientific notation | `2`, `0.5`, `1e-3` |
| `+ - * / **`, unary minus, brackets | `(Compliance() / 10) ** 0.5`, `-Compliance()` |
| `sqrt`, `exp`, `log` | `sqrt(Compliance())` |
| `smooth_max(a, b, ..., p=8)` | `smooth_max(Displacement(Tip, y, pnorm) / 1e-3, StressFailureFactor())` |
| material constants | `YieldStrength`, `YoungsModulus`, `Density` (from the solver's material) |

| Rejected | Why | Use instead |
|---|---|---|
| `max(...)` | not differentiable | `smooth_max(...)` or `pnorm` |
| `min(...)` | not differentiable | a constraint, or `-smooth_max(-a, -b)` |
| `abs(...)` | not differentiable at 0 | the `magnitude` component, or square the quantity |
| attributes, indexing, strings, lambdas, imports, ... | not in the whitelist | a [Python function](user-defined-functions.md) |

## Examples
### Objectives
| Goal | Expression | Sense |
|---|---|---|
| Stiffest part | `Compliance()` | minimize |
| Least deflection at a load acting in −y | `Displacement(Tip, y, mean)` | **maximize** (see below) |
| Smallest peak displacement, in mm | `Displacement(Tip, magnitude, pnorm) / 1e-3` | minimize |
| Stiff and low stress | `Compliance() + 0.5 * StressPNorm() / YieldStrength` | minimize |
| Stiffness plus a material penalty | `Compliance() / 10 + 2 * VolumeFraction()` | minimize |
| The worse of displacement and stress | `smooth_max(Displacement(Tip, y, pnorm) / 1e-3, StressFailureFactor())` | minimize |
| Lightest part (needs a performance constraint) | `Mass()` | minimize |
| Coolest hot spot | `Temperature(HotFace, pnorm)` | minimize |
| Best heat conduction | `ThermalCompliance()` | minimize |
| Stiff against the mechanical load of a heated part | `MechanicalCompliance()` | minimize |

### Constraints
| Meaning | Expression | Op | Bound |
|---|---|---|---|
| At most 30 % material | `VolumeFraction()` | `<=` | `0.3` |
| At least 20 % material | `VolumeFraction()` | `>=` | `0.2` |
| Stress below half of yield | `StressFailureFactor()` | `<=` | `0.5` |
| Peak stress below 200 MPa | `MaxStress()` | `<=` | `2e8` |
| Peak displacement at most 1 mm | `Displacement(Tip, magnitude, pnorm)` | `<=` | `1e-3` |
| Mass at most 2 kg | `Mass()` | `<=` | `2` |
| Hot spot at most 350 K | `Temperature(HotFace, pnorm)` | `<=` | `350` |
| Less than 30 % material in a region | `VolumeFraction(Flange)` | `<=` | `0.3` |

### Complete setups
1. **Lightest bracket that is stiff enough:** minimize `VolumeFraction()`, subject to `Displacement(Tip, magnitude, pnorm) <= 1e-3` and `StressFailureFactor() <= 0.5`.
2. **Stiffest part that doesn't yield:** minimize `Compliance()`, subject to `VolumeFraction() <= 0.4` and `StressFailureFactor() <= 1`.
3. **Heat sink:** minimize `Temperature(HotFace, pnorm)`, subject to `VolumeFraction() <= 0.3`.
4. **Heated structure:** minimize `MechanicalCompliance()`, subject to `VolumeFraction() <= 0.4` and `Temperature(HotFace, pnorm) <= 400`.

## Things to watch
- **Units are SI** (m, Pa, K, N, J, kg). A 1 mm limit is `1e-3`.
- **Scale terms you add together.** Divide each by a typical value (`/ 1e-3`, `/ 10`), or the larger term decides alone. Validate in the GUI, or evaluate the expression once, to see the start values.
- **Displacement components have a sign.** Under a load in −y, `y` is negative, so *minimizing* `Displacement(Tip, y, mean)` makes the part bend **more**. Maximize it, or minimize `Displacement(Tip, magnitude, pnorm)`.
- **Use `pnorm` for peaks**, not `mean`.
- **Minimizing volume or mass needs a constraint**, or the answer is empty; `validate` reports an error. Minimizing compliance with no volume or mass limit gives a full block; `validate` warns.
- **`>=` needs a bound > 0**, because `f >= b` is stored as `1 − f/b ≤ 0`. Other cases are rejected.
- **A constraint violated at the end:** the final iteration printout marks it `violated`. Loosen the bound, allow more material, or run more iterations.
- **Stress quantities** cost more per iteration and react strongly to small changes; start from a loose bound.

## Gradient and methods
- An expression that is exactly a built-in quantity (`Compliance()`, `VolumeFraction()`, `Mass()`, `StressPNorm()`, plus the stress constraints) becomes the built-in type, which keeps [analytical gradients](gradients.md) available where they apply.
- Any other expression uses automatic differentiation and needs MMA.

## Adding your own quantity
New quantities (for example a region's temperature gradient) are registered once in `pyto/autodiff/qoi/responses.py`. They are then usable in expressions and the GUI: [Add a response](../extending/add-a-response.md).

---
Next: [User-defined functions](user-defined-functions.md) · [Back to contents](../README.md)
