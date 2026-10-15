# Thermal and thermo-structural problems

This page shows how to set up **steady heat conduction** and **coupled thermo-structural** problems in code, and what is different about optimizing them. It assumes you've read [A problem from scratch](problem-from-scratch.md).

## Thermal problems
The unknown is the temperature, **one dof per node**. Loads are heat inputs in W per node; supports are nodes held at a fixed temperature.

```python
import numpy as np
import pyto.autodiff.sparse_solve as sparse_solve
import pyto.core.bc as bound_cond
import pyto.core.mat_lib as mat_lib
from pyto.core.hex_mesher import HexMesher
from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA
from pyto.topopt.common import TOParams, TO_QOI
from pyto.topopt.drivers.mma import topopt_mma

mesh = HexMesher()
mesh.grid_mesh(num_elems=(30, 30, 1), elem_size=(0.1 / 30, 0.1 / 30, 0.005))    # 0.1 x 0.1 x 0.005 m plate
mesh.createEdofMatThermal()                                                      # 1 dof per node
xyz = mesh.node_xyz

sink = np.where(xyz[:, 1] > xyz[:, 1].max() - 1e-9)[0]                          # top edge held at 20 C
source = np.where((xyz[:, 1] < 1e-9) & (np.abs(xyz[:, 0] - 0.05) < 0.005))[0]   # small patch, bottom centre
heat = np.zeros(mesh.num_nodes)
heat[source] = 10.0 / len(source)                                                # 10 W into that patch

bc = bound_cond.BC(force=heat, fixed_dofs=sink, dirichlet_values=np.full(len(sink), 20.0))
fe = HexThermalFEA(mesh=mesh, mat_prop=mat_lib.get_material("Steel"), bc=bc, solver=sparse_solve.Solvers.SPSOLVE)
T = fe.solve()                                   # temperatures of the full plate (max about 40 C here)

p = TOParams()
p.Objective = (TO_QOI.COMPLIANCE, None)          # thermal compliance: lower = better heat conduction
p.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.3)]
u, history, ok, message, n = topopt_mma(fe, to_params=p, maxMMAIterations=30,
                                        binarize_topology=False, print_progress=False)
# thermal compliance 2216 -> 1540 in 30 iterations
```

Things specific to thermal problems:
- **Put heat sources where material will be.** A heat input on a node that becomes void has nowhere to go, and its temperature explodes. So avoid spreading heat over the whole design domain; that is a different problem, with design-dependent heat generation.
- **Conductivity law:** the conductivity of an element with density `x` follows its own SIMP-type law (penalty 1 by default). `HexThermalFEA(conductivity_penalty=..., conductivity_void_ratio=...)` overrides it per problem.
- **Problems driven only by prescribed temperatures** (no heat input; for example hot and cold faces) are optimized for *maximum* conduction. PyTO flips the sign automatically (`compliance_sign`) and still reports the true value.
- **Quantities:** `Compliance()` / `ThermalCompliance()`, `Temperature(sel, mean|pnorm)`, `VolumeFraction()`. See [Expressions](expressions.md).
- **Grey designs and binarization:** with conductivity penalty 1, thermal designs stay mostly grey, and thresholding them to 0/1 can cut the conduction path (in this example, 1.1e10 instead of 1540). MMA and OC detect this and return the continuous design with a warning in `message`; the examples here pass `binarize_topology=False` to skip thresholding altogether. See [Thresholding and evaluation](../postprocessing/thresholding-and-evaluation.md#when-thresholding-goes-wrong).

## Thermo-structural problems
`ThermoStructuralFEA` solves the temperature first. It then applies the thermal-expansion load and solves the structure. It **is** a structural solver (it subclasses `HexStructuralFEA`), so everything structural works on it; the temperature is on `fe.temperature`.

```python
from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA

mesh = HexMesher()
mesh.grid_mesh(num_elems=(40, 20, 1), elem_size=(0.2 / 40, 0.1 / 20, 0.005))
mesh.createEdofMatStructural()
mesh.createEdofMatThermal()                      # both tables: two physics on one mesh
xyz = mesh.node_xyz

left = mesh.getNodesOnBoundingBoxPlane(0, True)
tip = np.where((xyz[:, 0] > xyz[:, 0].max() - 1e-9) & (np.abs(xyz[:, 1] - 0.05) < 0.006))[0]
f = np.zeros(3 * mesh.num_nodes)
f[3 * tip + 1] = -100.0 / len(tip)
fixed = np.concatenate([3 * left, 3 * left + 1, 3 * left + 2])
structural_bc = bound_cond.BC(force=f, fixed_dofs=fixed, dirichlet_values=np.zeros(len(fixed)))

all_nodes = np.arange(mesh.num_nodes)
thermal_bc = bound_cond.BC(force=np.zeros(mesh.num_nodes), fixed_dofs=all_nodes,
                           dirichlet_values=np.full(mesh.num_nodes, 70.0))      # uniformly at 70 C

fe = ThermoStructuralFEA(mesh=mesh, mat_prop=mat_lib.get_material("Titanium"),
                         structural_bc=structural_bc, thermal_bc=thermal_bc,
                         solver=sparse_solve.Solvers.SPSOLVE,
                         thermoElasticReferenceTemperature=20.0)                # stress-free temperature
u = fe.solve()                                   # displacement; fe.temperature holds the temperature

p = TOParams()
p.Objective = (TO_QOI.COMPLIANCE, None)
p.Constraints = [(TO_QOI.VOLUME_FRACTION, None, 0.4)]
u, history, ok, message, n = topopt_mma(fe, to_params=p, maxMMAIterations=20,
                                        binarize_topology=False, print_progress=False)
# compliance 1.28 -> 0.214; the design uses only 15 % material although 40 % is allowed
```

What is different:
- **The thermal load depends on the design.** Material that expands also pushes on the structure. So adding material doesn't always help, and the optimizer may use **less** material than allowed, as in this example (15 % of a 40 % budget). Compliance can also rise when material is added.
- **Methods:** use **MMA**. OC oscillates on these problems; `method_support` reports this.
- **Stress** is computed from the *elastic* strain: total strain minus the free thermal strain α(T − T_ref).
- **Extra quantities:**
  - `MechanicalCompliance()`: the work of the mechanical loads only, a common objective for heated structures (Ooms et al. 2023).
  - `ThermalCompliance()` and `Temperature(...)`: the thermal half.
- **Settings in `TOParams`** (read by `build_benchmark`): `ThermalReferenceTemperature`, `ConductivityPenalty`, `ConductivityVoidRatio`.
- **Constructor arguments:** `conductivity_penalty` and `conductivity_void_ratio` can also be passed to `ThermoStructuralFEA` directly.

The published benchmarks (BiClamp after Rodrigues & Fernandes 1995; MBB beam after Ooms et al. 2023) are ready-made: `ThermoStructuralTOExamples.BiClamp`, `ThermoStructuralTOExamples.MBBBeam`. See [Benchmarks from code](benchmarks-from-code.md) and [Validation](../reference/validation.md).

Background: [Thermo-structural coupling](../theory/thermo-structural.md).

---
Next: [Benchmarks from code](benchmarks-from-code.md) · [Back to contents](../README.md)
