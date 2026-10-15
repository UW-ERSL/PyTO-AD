# Add a material model

This recipe adds a new **interpolation law**: how an element's stiffness or conductivity depends on its density $x \in [0, 1]$. It is for developers experimenting with penalization schemes, or with different void treatments for their physics.

## Where the laws live
`autodiff/material_model.py` holds the `MaterialModel` enum (`SIMP`, `RAMP`, `SIMPPLUS`) and, for each physics, three functions that must agree:

| Function | Kind | Used by |
|---|---|---|
| `get_structural_material_model_scaling_torch(x, model)` | torch, differentiable | the FE solve (autograd path) |
| `get_structural_material_model_scaling(x, model)` | numpy | derived sensitivities, post-processing |
| `get_structural_material_model_sensitivity(x, model)` | numpy, the derivative $dE/dx$ | analytical gradients, OC |

The thermal set (`get_thermal_...`) is the same with the conductivity constants. The torch version also takes the per-problem `penalty` and `void_ratio`.

The current laws, with $E_v = 10^{-8}$ (void floor), $p = 3$ (structural) or $1$ (thermal), and $q = 5$ (RAMP):

| Model | $E(x) / E_0$ |
|---|---|
| `SIMP` | $E_v + x^p (1 - E_v)$ |
| `RAMP` | $E_v + \dfrac{x (1 - E_v)}{1 + q(1 - x)}$ |
| `SIMPPLUS` | average of SIMP and $E_v x$ |

## Steps
1. **Enum:** add a member to `MaterialModel`, e.g. `SINH = enum.auto()`.
2. **Torch law:** add a branch to `get_structural_material_model_scaling_torch` (and the thermal one if it applies), using torch operations.
3. **NumPy law and derivative:** add the same branch to `get_structural_material_model_scaling` and `get_structural_material_model_sensitivity`, with the analytic derivative.
4. **Use it:**
   - **OC** reads `to_params.materialModel`.
   - **MMA** currently sets `MaterialModel.SIMP` inside the driver. To use another law there, read `to_params.materialModel` instead, in `topopt/drivers/mma.py` (`material_model = MaterialModel.SIMP` near the top of `topopt_mma`), and likewise in the other drivers.
5. **Test** that the three functions agree. For random `x` in (0, 1):
   - the torch and NumPy laws are equal;
   - autograd of the torch law equals the NumPy derivative;
   - both agree with a finite-difference derivative.

   A small parametrized test in `tests/` covers all models at once.

## Things to keep
- **A void floor** ($E(0) > 0$) keeps the stiffness matrix non-singular when elements are empty.
- **$E(1) = 1$**, so a solid element has the material's real stiffness.
- **Penalization:** $E(x) < x$ for intermediate $x$, so grey material is inefficient and the optimum is close to 0/1. With $E(x) = x$, designs stay grey.
- **Stress relaxation:** stress constraints use their own relaxation factor $(E_v + (1 - E_v)x^{0.5})$ in `autodiff/qoi/stress.py`; change it only together with the manual p-norm gradient.
- **Global constants:** penalties are module-level constants (`_SIMP_STRUCTURAL_PENALTY` and others). Continuation schemes change them through the `set_` / `increment_` helpers in the same file. Remember to reset them between runs in the same process.

---
Next: [Machine-learning approaches](ml-approaches.md) · [Back to contents](../README.md)
