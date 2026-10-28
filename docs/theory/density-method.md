# Density method

This page summarizes the density-based topology optimization that MMA implements in PyTO: the material interpolation, the filter and the projection. It is short and practical; for full derivations see the references at the end.

## Design variables
The design domain is divided into $N$ voxel elements. Each has a density $x_e \in [0, 1]$: 1 is solid, 0 is void. The optimizer changes $\mathbf{x}$; the FE analysis sees the **physical** density $\tilde{\mathbf{x}}$ obtained after filtering and projection.

## Material interpolation (SIMP)
The stiffness of element $e$ is scaled by its physical density:

$$E_e = E_0\left(E_v + \tilde{x}_e^{\,p}\,(1 - E_v)\right), \qquad p = 3,\; E_v = 10^{-8}$$

- $p > 1$ penalizes intermediate densities: half the material gives an eighth of the stiffness. The optimum is therefore close to 0/1.
- The void floor $E_v$ keeps the stiffness matrix solvable when elements are empty.
- Thermal conductivity uses the same form, with $p = 1$ by default; thermo-structural problems can set their own penalty and void ratio.
- Self-weight scales with $\tilde{x}_e$ (linearly), not with the stiffness law.

Code: `autodiff/material_model.py`. How to add a law: [Add a material model](../extending/add-a-material-model.md).

## Density filter
Each physical density is a weighted average of the densities within a radius $r$:

$$\bar{x}_e = \frac{\sum_{i \in N_e} H_{ei}\, x_i}{\sum_{i \in N_e} H_{ei}}, \qquad H_{ei} = \max(0,\; r - \lVert \mathbf{c}_e - \mathbf{c}_i \rVert)$$

- It removes checkerboard patterns and makes the result independent of the mesh.
- It sets a minimum member size of about $2r$.
- In PyTO, $r$ = **filter radius** × element size (`RelativeFilterRadius`, default 1.5).
- Symmetry, cyclic symmetry and extrusion are extra averaging matrices multiplied into $\mathbf{H}$ ([Add a filter](../extending/add-a-filter.md)).

## Heaviside projection (optional)
The filter leaves grey transitions. A smooth step pushes them toward 0 or 1 (Ooms et al. 2023, eq. 33, with $\eta = 0.5$):

$$\tilde{x}_e = \frac{\tanh(\beta\eta) + \tanh(\beta(\bar{x}_e - \eta))}{\tanh(\beta\eta) + \tanh(\beta(1 - \eta))}$$

$\beta$ starts at 1 and doubles every `HeavisideBetaInterval` iterations (50) up to `HeavisideBetaMax` (32). A small $\beta$ keeps the problem smooth early on; a large $\beta$ gives a crisp design at the end.

## The optimization problem
$$\min_{\mathbf{x}} \; f(\tilde{\mathbf{x}}, \mathbf{u}) \quad \text{s.t.} \quad g_j(\tilde{\mathbf{x}}, \mathbf{u}) \le 0,\;\; 0 \le x_e \le 1, \quad \text{where } \mathbf{K}(\tilde{\mathbf{x}})\,\mathbf{u} = \mathbf{F}$$
- **MMA** (Svanberg 1987) solves a sequence of convex approximations. It handles any number of constraints.
- Both need the gradients $df/dx_e$, $dg_j/dx_e$: see [Sensitivities](sensitivities.md).

## References
- Bendsøe & Sigmund, *Topology Optimization: Theory, Methods and Applications*, Springer, 2003.
- Sigmund, *A 99 line topology optimization code written in Matlab*, SMO 21 (2001).
- Svanberg, *The method of moving asymptotes*, IJNME 24 (1987).
- Ooms, Vantyghem, Thienpont, Van Coile, De Corte, *Thermoelastic topology optimization of structural components at elevated temperatures considering transient heat conduction*, 2023.

---
Next: [Sensitivities](sensitivities.md) · [Back to contents](../README.md)
