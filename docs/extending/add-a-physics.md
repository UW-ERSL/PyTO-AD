# Add a physics

This recipe explains what a new FE solver needs so the optimizers, responses and gradients can use it: another PDE (electrical or magnetic potential, acoustics, flow in porous media, ...), another element, or a coupling of existing solvers. It is for developers extending PyTO's physics.

## The solver contract
The drivers, responses and gradient code only rely on the following. Anything else is up to you.

| Member | Type | Used for |
|---|---|---|
| `mesh` | `HexMesher` | geometry, element connectivity, the current design (`mesh.elemPseudoDensity`) |
| `bc` | `BC` | loads (`bc.force`, one entry per dof), fixed dofs and their values; `bc.num_dofs` is the system size |
| `mat_prop` | `Material` | material data for element matrices and responses |
| `solver`, `dsolver`, `kwargs` | from the constructor | linear solver selection |
| `elem_stiff` | numpy `(n_materials, k, k)` | the element matrix of a **solid** element (k = dofs per element) |
| `mesh.edofMat` | `(num_elems, k)` | element → global dof table for this physics; the responses read it |
| `solve(x, material_model)` | → torch `(num_dofs,)` | **differentiable** solve for a density; `x=None` means fully solid |
| `stiff_mtrx` | torch sparse | the assembled matrix of the last solve |
| `adjoint_solve(rhs)` | → numpy | solves with the matrix and solver object of the last solve (same factors / preconditioner); analytical sensitivities use it. Pass `keep=self._last_solve` to `sparse_solve.solve` and copy the method from `HexThermalFEA` |
| `postprocess()` | – | derived fields for plots (fluxes, stresses) |
| `plot_mesh(...)`, `plot_elem_field(...)` | – | pictures; the plotting in `gui/hex_plotter.py` can be reused |

### The one hard rule: a differentiable solve
`solve` must build the matrix from `x` with **torch operations**, and solve with `pyto.autodiff.sparse_solve.solve`:
```python
from pyto.autodiff.sparse_solve import solve as sparse_spsolve
from pyto.core.bc import apply_dirichlet_bc_torch

def solve(self, x=None, material_model=None):
    if x is None:
        x = torch.ones(self.mesh.num_elems, dtype=torch.float64)
    scale = my_material_law_torch(x, material_model)                  # (E,) torch, e.g. SIMP
    vals = torch.einsum("mij,m->mij", self.elem_stiff_torch, scale).reshape(-1)
    agg = torch.zeros(self._n_unique, dtype=x.dtype).index_add_(0, self._scatter_inverse, vals)
    self.stiff_mtrx = torch.sparse_coo_tensor(self._unique_indices, agg, (ndof, ndof))
    K_bc, f_bc = apply_dirichlet_bc_torch(self.stiff_mtrx, self._bc_force_torch, self.bc)
    self.sol = sparse_spsolve(K_bc, f_bc, solver=self.solver)
    return self.sol
```
This is a fragment: the assembly indices (`_unique_indices`, `_scatter_inverse`, `_n_unique`) are built once in `__init__` from `mesh.edofMat`. Copy that part from `HexThermalFEA.__init__` (scalar, 8 dofs per element) or `HexStructuralFEA.__init__` (vector, 24). `sparse_solve` gives the gradient its adjoint solve for free. NumPy anywhere between `x` and the solution breaks the gradient.

## Steps
1. **Element matrix.** Add it to `physics/hex_element_stiffness.py`, next to `hex8_stiffness_matrix_structural` / `_thermal`: a k × k matrix for a solid element of size `mesh.elem_size`. For a scalar diffusion-type PDE it is the thermal matrix with your coefficient.
2. **Dof table.** If the dofs per node differ from 1 or 3, add a `createEdofMat...` method to `HexMesher` modelled on `createEdofMatThermal` / `createEdofMatStructural`. Point `mesh.edofMat` to it in your constructor.
3. **Material law.** The density-to-coefficient interpolation, in torch. Reuse the structural or thermal law from `autodiff/material_model.py`, or [add one](add-a-material-model.md).
4. **Solver class** in `physics/<name>/hex_<name>_fea.py`, with the contract above. Start by copying `physics/thermal/hex_thermal_fea.py`: it is the smallest complete example.
5. **Register it where the code branches on the physics:**

   | Where | What |
   |---|---|
   | `physics_of()` in `autodiff/qoi/responses.py` | return a name for your solver (e.g. `"electric"`); check subclasses before parents |
   | `Response(...)` physics sets in `responses.py` | add the name to the responses that make sense (e.g. `Compliance`, `VolumeFraction`) and add new responses for your fields |
   | `compute_compliance_torch` in `autodiff/qoi/compliance.py` | choose the right material law for your solver (it uses the element matrix and `edofMat`) |
   | `compute_element_stiffness` in `topopt/drivers/_shared.py` | return your element matrix; it raises `TypeError` for unknown solvers |
   | `nDOFPerNode = 3 if isinstance(...HexStructuralFEA) else 1` | in `_shared.py`, `mma.py`, `oc.py`: correct only for 1 or 3 dofs per node |
   | `method_support()` in `topopt/capabilities.py` | which methods your physics supports (MMA always, if the solve is differentiable) |
   | `manual_support()` in `topopt/manual_sensitivities.py` | refuse analytical gradients until you've [added one](add-a-manual-sensitivity.md) |
   | `validate()` in `topopt/spec.py` | accepts any physics name the responses know |

6. **GUI (optional):** a loads window and an analysis button in `gui/PyTOGUI.py`, and a branch in `StructuralTopOptWindow.build_problem`. This is the biggest part; do it last.
7. **Tests:** see below.

## Coupling existing solvers
`physics/thermoelastic/thermostructural_fea.py` is the template for multiphysics. `ThermoStructuralFEA`:
- **subclasses** the structural solver, so every structural response, plot and driver path works unchanged;
- **holds** a `HexThermalFEA` as `self.thermal_fea`;
- in `solve(x)`, solves the temperature, turns it into a load (`get_thermoelastic_force_torch`), and calls the structural `solve` with `extra_force=`. It all stays in torch, so autograd follows the whole chain, including the thermal solve.

To add, say, piezoelectric or electro-thermal coupling, follow the same pattern:
1. Pick the "output" physics as the parent class.
2. Solve the other physics first.
3. Pass its effect in through `extra_force` (or an equivalent hook).
4. Keep the intermediate field on the object, as `self.temperature` is, so responses can use it.

## Tests
- **Patch tests:** a uniform-coefficient problem with a known analytic answer (e.g. a linear potential between two faces).
- **Gradient:** `check_gradient` on compliance and on one of your new responses.
- **Driver smoke test:** a few MMA iterations run and lower the objective (`tests/test_drivers_smoke.py` style).
- For coupled solvers: the coupled solve at `x = 1` matches solving the two physics by hand.

See [Testing your extension](testing-your-extension.md).

---
Next: [Add an optimization method](add-an-optimization-method.md) · [Back to contents](../README.md)
