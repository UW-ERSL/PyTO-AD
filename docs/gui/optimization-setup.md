# Optimization setup

This page explains the **TopOpt Execute** window: the objective, the constraints, the method and gradient, checking the formulation, and running and watching the optimization. It is for GUI users.

The window's upper part has four tabs, **Objective**, **Constraints**, **Method** and **Iterations**. Below them are **Validate formulation**, the live plot, **Show topology every iteration**, **Optimize** and **STOP OPTIMIZATION!**.

## Objective tab
| Control | Meaning |
|---|---|
| **Sense** | Minimize or Maximize |
| **Quantity** | a built-in quantity (Compliance, Volume fraction, Mass, Displacement, Temperature, Reaction force, Strain energy, Stress (p-norm), Stress failure factor, Mechanical compliance, Thermal compliance: only those valid for your physics), **Expression...**, or **Python function...** |
| **Selection** | for quantities at a place: `All`; each applied support or load, `Support1`, `Support2`, ..., `Load1`, ...; their unions `Supports` and `Loads`; thermal faces `FixedTemp1`, ..., `HeatSource1`, ...; and `FixedFaces` when **Keep Fixed Faces** is on |
| **Component** | magnitude, x, y or z (Displacement, Reaction force) |
| **Aggregate** | mean, p-norm (smooth max), sum |
| **Expression** | the formula; filled in from the choices above, or typed |
| **Choose file...** | for **Python function...**: pick a `.py` file and a function |

Choosing a built-in quantity writes its formula into **Expression**, e.g. `Displacement(Load1, y, pnorm)`. Edit it freely: scale it, add terms. The full language and the list of quantities are in [Expressions](../code/expressions.md).

## Constraints tab
A table with one row per constraint:

| Column | Meaning |
|---|---|
| **On** | untick to keep the row without using it |
| **Quantity / expression** | any expression, e.g. `VolumeFraction()`, `StressFailureFactor()`, `Displacement(Load1, magnitude, pnorm)` |
| (operator) | `<=` or `>=` |
| **Bound** | the limit, **always in SI units** (m, Pa, K, N, J, kg) |

- **Add** inserts a row for the quantity chosen next to it.
- **Remove selected** deletes rows.
- `MaxStress()` is available as a constraint on its own: `MaxStress() <= 2e8`.
- Constraints from older projects' performance options are imported as rows, switched off.

## Method tab
| Control | Meaning |
|---|---|
| **Method** | MMA or OC. A method that can't solve your formulation is **disabled**, and the tooltip says why (e.g. OC only minimizes compliance with one volume limit). A project saved with another method opens with MMA |
| **Gradient** | **Automatic differentiation** (default) or **Analytical**. Analytical is offered only for compliance, p-norm stress and volume fraction with MMA or OC; otherwise it's greyed out with the reason. See [Gradients](../code/gradients.md) |
| **Max iterations**, **Move limit** | iteration cap; the largest density change per iteration (MMA, OC) |
| **Filter radius (elements)** | the density filter radius, in element sizes (1.5 by default). Larger means thicker members |
| **Heaviside projection** | pushes the design to 0/1 during the run (MMA); needs more iterations |
| **Stress-free temperature** | thermo-structural problems only |
| **Cost** | the linear solves one iteration needs |
| **Check gradient (finite differences)** | compares the gradient in use with finite differences on a few elements |

## Validate formulation
It checks everything without optimizing:
1. every expression parses;
2. every quantity exists for the physics;
3. selections exist;
4. bounds are usable, and the problem is well posed. For example, minimizing volume with no constraint is an error.

It then runs **one FE solve** at a uniform design and shows the objective and each constraint's value. Use those values to choose sensible bounds and scale factors.

## Optimize
**Optimize** validates again, builds the problem from your current loads, and starts the run in the background.
- **Live plot:** objective and constraints (in their normalized form, ≤ 0 = satisfied) after every iteration.
- **Iterations tab:** each iteration's numbers, with each constraint as you wrote it and `ok` or `violated`:
  ```
  Iteration: 12
  Min. Objective (Compliance()): 20.4
  Constraint 1 (VolumeFraction()): 0.3 <= 0.3  ok
  ```
- **Show topology every iteration:** redraws the design in the 3D view each iteration (off by default: slower on big meshes). It can be toggled during the run.
- **STOP OPTIMIZATION!** ends the run early.

At the end, the message area gives the final objective, the volume fraction, and a summary of the solid part (maximum displacement, stress against yield, temperature). The design is exported as a VTU next to your STL. Then:
- look at deformation and stress in **Analysis → Show results → Optimized design** ([Analysis](analysis.md));
- make a smooth STL in **TopOpt Postprocess** ([Export](../postprocessing/export.md)).

## Examples
| Goal | Objective | Constraints |
|---|---|---|
| Stiffest part | Minimize `Compliance()` | `VolumeFraction() <= 0.3` |
| Lightest part that is stiff enough and doesn't yield | Minimize `VolumeFraction()` | `Displacement(Load1, magnitude, pnorm) <= 1e-3`, `StressFailureFactor() <= 0.5` |
| Least deflection at the load (load in −y) | **Maximize** `Displacement(Load1, y, mean)` | `VolumeFraction() <= 0.3` |
| Coolest hot spot (thermal) | Minimize `Temperature(HeatSource1, pnorm)` | `VolumeFraction() <= 0.3` |

More examples and pitfalls, e.g. why a downward displacement is *maximized*, are in [Expressions](../code/expressions.md#things-to-watch).

---
Next: [Manufacturing options](manufacturing-options.md) · [Back to contents](../README.md)
