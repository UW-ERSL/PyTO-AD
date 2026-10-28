# Add an analytical sensitivity

This recipe adds a derived (hand-written) gradient for a quantity, so MMA can run it with `Gradient = "manual"`. It is for developers who need speed, a reference to check autograd against, or a sensitivity for a method that can't use autograd.

You don't need this to optimize a new quantity: [a response](add-a-response.md) or [a user-defined function](../code/user-defined-functions.md) is differentiated automatically. An analytical sensitivity is extra work, and it must be proven equal to autograd.

## How analytical gradients fit in
- `topopt/manual_sensitivities.py` decides **which** quantities have an analytical gradient and **evaluates** it.
- The formulas live in `autodiff/reference_adjoint/legacy_sensitivities.py`, in NumPy.
- MMA calls `manual_gradients(to_params, sol, x_phys, fe, material_model)`. It returns `(d_obj, d_cons)`: the gradients with respect to the **physical** density, for the objective and for each constraint in its normalized form.
- The chain rule through the Heaviside projection and the density filter is then applied in `drivers/mma.py` (`manual_step`).

## Step 1: derive
Most quantities follow the **adjoint** template. For a quantity $f(\mathbf{u}(\mathbf{x}), \mathbf{x})$ with $\mathbf{K}(\mathbf{x})\mathbf{u} = \mathbf{F}$ and a design-independent load:

$$\frac{df}{dx_e} = \frac{\partial f}{\partial x_e} - \boldsymbol{\lambda}_e^{T}\,\frac{\partial \mathbf{K}_e}{\partial x_e}\,\mathbf{u}_e, \qquad \mathbf{K}\boldsymbol{\lambda} = \frac{\partial f}{\partial \mathbf{u}}$$

- The first term is the **explicit** dependence on the density; the second is the **implicit** one through the solution.
- With SIMP, $\partial \mathbf{K}_e/\partial x_e = E'(x_e)\,\mathbf{K}^0_e$, where `get_structural_material_model_sensitivity(x, material_model)` gives $E'$.
- **Self-adjoint case (compliance):** $f = \mathbf{F}^T\mathbf{u}$ gives $\boldsymbol{\lambda} = \mathbf{u}$ and no extra solve: $df/dx_e = -E'(x_e)\,\mathbf{u}_e^T\mathbf{K}^0_e\mathbf{u}_e$.
- **Design-dependent loads** (self-weight, thermal expansion) add a term $\boldsymbol{\lambda}^T\,\partial\mathbf{F}/\partial x_e$.

## Step 2: implement the formula
Add a function to `autodiff/reference_adjoint/legacy_sensitivities.py`, and export it in that package's `__init__.py`:
```python
def compute_my_quantity_and_gradient(sol, x, fe_solver, KE, material_model):
    """Value and d/dx (numpy), at the solver's last solve."""
    ...
    g = ...                                               # df/du, (num_dofs,)
    lam = fe_solver.adjoint_solve(g)                      # adjoint solve: last solve's matrix and factors
    ce = ((lam[mesh.edofMat] @ KE) * sol[mesh.edofMat]).sum(1)                  # lambda_e^T K0 u_e
    return value, explicit - get_structural_material_model_sensitivity(x, material_model) * ce
```
- **`fe_solver.adjoint_solve(g)`** solves $\mathbf{K}\boldsymbol{\lambda} = \mathbf{g}$ with the forward solve's own matrix and solver object, zeroing the fixed dofs. That means the same PARDISO/SPSOLVE factors (no second factorization) or the same DPCG preconditioner, the same path the autograd backward pass takes, so the two gradient modes stay comparable. `compute_pnorm_stress_and_sensitivity` is the worked example.
- **Match the definition exactly.** The formula must differentiate *the same function* the torch path evaluates: the same element-centre strain, relaxation factor, void floor and p. Any difference shows up as a mismatch in step 4.

## Step 3: register it
In `topopt/manual_sensitivities.py`:
1. Add the `TO_QOI` type to `MANUAL_OBJECTIVES` and/or `MANUAL_CONSTRAINTS`.
2. In `manual_support()`, return `(False, reason)` for the cases your formula doesn't cover: other physics, body forces, and so on. The reason is shown in the GUI tooltip and in errors.
3. In `manual_gradients()`, add a cached branch like `pnorm()`, and the constraint form. A `<=` constraint is stored as `value/limit − 1`, so its gradient is `d_value / limit`.

For the GUI, `capabilities.manual_gradient_support` reads `formulation_support` automatically. Only if your formula has a limit visible at the spec level (like body force) does it need a line there. `cost_estimate` in the same file counts adjoint solves for the analytical path, so update it if your quantity needs one.

## Step 4: verify against autograd
Add a case to `tests/test_manual_sensitivities.py`:
```python
def test_my_quantity(structural_fe_solver):
    _compare(_params((TO_QOI.MY_QUANTITY, None), (TO_QOI.VOLUME_FRACTION, None, 0.5)),
             structural_fe_solver, tol=1e-7)
```
- `_compare` evaluates the autograd gradient and your analytical gradient at a random interior design and requires them to agree.
- Expect agreement at the **1e-12** level. Anything worse than about 1e-7 means the two paths compute different functions.
- Then add a full-run check, as `test_mma_manual_pnorm_matches_autodiff` does: a few MMA iterations with each gradient must give the same objective history.

## Worked example: the p-norm stress gradient
The p-norm von Mises stress, used by `StressPNorm()` and stress constraints, is:

$$\sigma_e = \big(\varepsilon_v + (1-\varepsilon_v)\,x_e^{q}\big)\,\mathbf{D}\mathbf{B}\mathbf{u}_e, \quad \sigma^{vm}_e = \sqrt{\sigma_e^T\mathbf{M}\sigma_e}, \quad P = \Big(\sum_e (\sigma^{vm}_e)^p\Big)^{1/p}$$

with $q = 0.5$ (stress relaxation), $\varepsilon_v = 10^{-8}$ (void floor), $p = 6$, and $\mathbf{M}$ the von Mises quadratic form.

The adjoint gradient has:
- an **explicit term** through the relaxation factor: $\frac{\partial P}{\partial \sigma^{vm}_e}\,\frac{(\sigma^{vm}_e)^2}{c_e\,\sigma^{vm}_e}\,c_e'(x_e)$;
- an **adjoint load** $\partial P/\partial \mathbf{u} = \sum_e \frac{\partial P}{\partial \sigma^{vm}_e}\,\frac{c_e}{\sigma^{vm}_e}\,\mathbf{B}^T\mathbf{D}^T\mathbf{M}\sigma_e$;
- one adjoint solve.

It is implemented in `compute_pnorm_stress_and_sensitivity` (`legacy_sensitivities.py`), vectorized over elements.

**What went wrong in the old version**, a checklist for your own derivations. The previous formula was 94 % off autograd because it:
1. used the reference-element strain matrix **B** without the $2/h$ scaling to the real element size;
2. read stresses from the plotting post-process, which isn't the function being optimized;
3. hard-coded the SIMP exponent $p=3$ instead of using the material law, which has a void floor;
4. used a relaxation factor without the void floor.

After the rewrite it agrees with autograd to **1.7e-14**, and a 4-iteration MMA run gives the same iterates with both gradients.

## Pitfalls
- **Sign conventions.** Problems driven only by prescribed values (no load) flip the compliance sign; see `compliance_sign` and `_compliance_gradient`.
- **Shared mesh tables:** the mesh has both `edofMatStructural` and `edofMatThermal`, and `mesh.edofMat` points at one of them. Set it explicitly while your formula runs, as `_compliance_gradient` does.
- **Stateful quantities** (e.g. `MAX_VONMISES_STRESS`, which rescales itself between iterations) can't have a fixed analytical gradient. Leave them to autograd.

---
Next: [Add a filter](add-a-filter.md) · [Back to contents](../README.md)
