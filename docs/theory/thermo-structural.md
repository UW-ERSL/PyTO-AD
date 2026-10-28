# Thermo-structural coupling

This page explains the one-way thermo-mechanical coupling in PyTO, and why optimizing heated structures behaves differently from purely mechanical problems. It is short and practical.

## The model
1. **Heat conduction:** $\mathbf{K}_T(\tilde{\mathbf{x}})\,\mathbf{T} = \mathbf{Q}$, with fixed temperatures on some faces. The conductivity is interpolated with its own penalty and void ratio.
2. **Thermal load:** a temperature rise above the stress-free temperature $T_{ref}$ makes material expand. In an element, with average temperature $\bar{T}_e$:
   $$\mathbf{f}^{th}_e = s(\tilde{x}_e)\,\frac{E\alpha}{1 - 2\nu}\,(\bar{T}_e - T_{ref})\,\mathbf{b}_e$$
   where $s$ is the structural stiffness interpolation and $\mathbf{b}_e$ integrates the element's shape-function gradients (`get_thermoelastic_force_torch` in `physics/thermal/hex_thermal_fea.py`).
3. **Elasticity:** $\mathbf{K}(\tilde{\mathbf{x}})\,\mathbf{u} = \mathbf{F}_{mech} + \mathbf{f}^{th}(\tilde{\mathbf{x}}, \mathbf{T})$.
4. **Stress** from the elastic strain: $\boldsymbol{\varepsilon} - \alpha(\bar{T}_e - T_{ref})[1,1,1,0,0,0]^T$ ([Stress](stress.md)).

The coupling is one-way: temperature affects the structure, and deformation doesn't affect the temperature. `ThermoStructuralFEA.solve` runs steps 1–3 in torch, so autograd differentiates through both solves and both adjoints.

## Why it's different to optimize
- **The load depends on the design.** Material that expands pushes on its neighbours and supports. Adding material can therefore *increase* compliance, which never happens in load-driven mechanical problems.
- **Less material than allowed.** The volume constraint is often inactive: in the example in [Thermal and thermo-structural problems](../code/thermal-and-thermostructural.md), the optimizer used 15 % of a 40 % budget.
- **Non-monotone objective.** Adding material does not always help: it also adds thermal load. Update rules that assume monotone, negative sensitivities (optimality criteria) oscillate on these problems; MMA does not assume this.
- **Choice of objective:**

  | Objective | Includes the thermal load? | Typical use |
  |---|---|---|
  | total compliance $\mathbf{F}^T\mathbf{u}$ (`Compliance()`) | yes | classical formulation (Rodrigues & Fernandes 1995) |
  | mechanical compliance $\mathbf{F}_{mech}^T\mathbf{u}$ (`MechanicalCompliance()`) | no | stiffness against the service load while the part is hot (Ooms et al. 2023) |

  Stress constraints are often the real design driver for hot parts: `StressFailureFactor() <= 1`.

## Validation in PyTO
- **BiClamp** (Rodrigues & Fernandes 1995): designs follow the paper's trend as the temperature rise grows (ΔT = 0, 1, 4, 10 K).
- **MBB beam** (Ooms et al. 2023): the mechanical case is within 0.8 % of the paper.

Images and details: [Validation](../reference/validation.md).

## References
- Rodrigues & Fernandes, *A material based model for topology optimization of thermoelastic structures*, IJNME 38 (1995).
- Ooms, Vantyghem, Thienpont, Van Coile, De Corte, *Thermoelastic topology optimization of structural components at elevated temperatures considering transient heat conduction*, 2023.
- Gao & Zhang, *Topology optimization involving thermo-elastic stress loads*, SMO 42 (2010).

---
Next: [Reference: Benchmark problems](../reference/benchmarks.md) · [Back to contents](../README.md)
