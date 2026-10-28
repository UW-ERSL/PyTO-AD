# Thresholding and evaluation

This page explains how to turn an optimized (partly grey) density into a clean 0/1 part, and how to report numbers that belong to that part. It is for anyone publishing results, or passing a design to CAD or manufacturing.

## Why the grey design's numbers aren't the part's numbers
Density methods such as MMA end with some intermediate densities, where the SIMP law makes grey material weak. The objective reported for that design describes the grey structure, not the part you'd build: a 0/1 version has a different compliance, volume and stress.

## Thresholding
```python
import numpy as np
import torch
from pyto.autodiff.material_model import MaterialModel

x = np.asarray(fe.mesh.elemPseudoDensity)
x01 = torch.as_tensor((x > 0.5).astype(float))           # fixed threshold 0.5
with torch.no_grad():
    u01 = fe.solve(x01, MaterialModel.SIMP)                # analyse the 0/1 design
compliance01 = float(np.dot(fe.bc.force, u01.numpy()))     # load-driven structural compliance
```
From the post-processing example (cantilever, 30 % volume):
```
grey design compliance 25.54 J at volume 0.300; 0/1 design 17.36 J at volume 0.319
```
The 0/1 part is stiffer here, but it uses a little more material than the limit. A fixed 0.5 threshold doesn't preserve volume.

**Volume-preserving threshold:** the drivers' `binarize_topology=True` sorts the densities and picks the threshold that keeps the mean density. The last history entry is then the 0/1 design's value.

## When thresholding goes wrong
Thresholding keeps the densest elements. In a mostly grey design those need not connect the loads to the supports, or the heat source to the sink. The 0/1 design is then broken and its value meaningless; for example, thermal compliance 1.1e10 instead of 1540 for a thin plate whose design was 81 % grey. Heat conduction with the default conductivity penalty of 1 is the typical case: grey material conducts well, so the design stays grey.

**Safeguard:** MMA compares the thresholded design with the last continuous one. If the 0/1 value is more than twice as bad (or, for a maximized quantity, less than half), it returns the **continuous** design instead. `message` then says so, for example:
```
Warning: the 0/1 (binarized) design was much worse than the continuous one (objective 1.142e+10 vs 1539;
grey fraction 0.81), so the continuous design is returned.
```
The rule is `binarization_acceptable` in `topopt/drivers/_shared.py`.

To get a usable 0/1 part in such cases:
- **Heaviside projection** (`to_params.HeavisideProjection = True`, MMA) pushes the design toward 0/1 during the optimization, so little changes at the end. It needs more iterations, because β is doubled every `HeavisideBetaInterval` iterations (default 50) up to `HeavisideBetaMax` (32).
- **A higher conductivity penalty** for thermal problems (`HexThermalFEA(conductivity_penalty=3)`, or `to_params.ConductivityPenalty` for thermo-structural benchmarks), so grey material stops paying off. On the thin plate above, penalty 3 lowers the grey fraction from 81 % to 13 %, and the 0/1 design (1450) is then better than the continuous one.
- **A finer mesh or a larger filter radius**, so members are several elements thick.
- **`Eliminate_Hanging_Elements`** (`TOParams`, MMA) keeps only the largest connected solid component after binarization, removing floating islands.

## Reporting checklist
- [ ] State whether values are for the grey design or the 0/1 design.
- [ ] Give the volume fraction of the design you report on.
- [ ] Report stresses on the solid region only (`final_design_summary`; see [Fields and plots](fields-and-plots.md)).
- [ ] Check connectivity: one solid component connecting loads and supports.
- [ ] For publication, re-analyse the final geometry (smooth STL or 0/1 voxels) with a finer mesh ([Export](export.md)).

---
Next: [Export](export.md) · [Back to contents](../README.md)
