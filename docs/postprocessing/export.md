# Export

This page shows how to get a result out of PyTO: as a VTU file for ParaView, as a smooth STL for CAD and printing, or as a tetrahedral mesh for re-analysis. It is for anyone passing designs to other tools.

The steps are taken from [`docs/examples/postprocess.py`](../examples/postprocess.py), which runs all of them in about 15 seconds.

## VTU (ParaView, VisIt)
```python
mesh.export_vtu_mesh(elem_field=mesh.elemPseudoDensity, density_field="density", file_name="design.vtu")
```
- Writes the hex mesh with the element field (and its node average) under the name `density_field`.
- Any per-element array can be exported the same way, e.g. `fe.vonMisesStress` with `density_field="von_mises"`.
- `mask_low_pseudodensity=True` keeps only elements with density > 0.5.
- In ParaView, use **Threshold** on `density` (e.g. ≥ 0.5) or **Contour** at 0.5 to see the part.

## Smooth STL
Voxel results have staircase surfaces. PyTO's STL recovery produces a smooth part that keeps the original STL's outer surfaces:
1. The voxel density is smoothed, then the **void** isosurface is extracted with marching cubes. Two smoothers:
   - `"cnn"` (default): a small 3D convolutional network is fitted, for a few seconds, to this density; its reconstruction is the smoothed field. It starts from random weights, so two runs differ slightly (seed `torch` to repeat a run).
   - `"gaussian"`: a Gaussian filter of width `sigma` element sizes (default 0.5). Deterministic and nearly instant. Wider than about 0.5 erodes thin members.

   On two test designs the two gave the same volume, overlap with the voxel design and smoothness (`Results/SmoothingComparison`).
2. That void surface is **subtracted** from the original design-domain STL (a boolean operation with `trimesh` and `manifold3d`). Faces that were never optimized, like mounting faces, keep their exact CAD shape.

```python
import pyvista
from pyto.io.topopt_stl_recovery import extract_isosurface, subtract_voids_from_stl

voids = extract_isosurface(pyvista.read("design.vtu"), isovalue=0.5, method="cnn")   # or method="gaussian", sigma=0.5
part = subtract_voids_from_stl(mesh.stlMesh.triangulate().compute_normals(), voids)   # mesh made from an STL
part.save("design.stl")
```
- `mesh.stlMesh` is the STL the mesh was made from. For a box mesh (`grid_mesh`) there is none: build a box STL first, e.g. `pyvista.Box(bounds=...).triangulate()`.
- If the boolean subtraction cannot run or returns no part, recovery fails with an error instead of returning a part:
  - an input STL that is not a closed volume (e.g. non-watertight, or bodies touching along edges) raises `ValueError` ("Not all meshes are volumes!"); repair the STL, or export the voxels instead;
  - an empty result (the voids cover the whole domain) raises `RecoveryError`.
- Check the part against the design: `recovered_volume_check(part, voxel_volume)` returns `(ok, ratio)`, the part's volume over the voxel design's (elements with density at least 0.5), with `ok` false outside 1 ± 0.3.
- In the GUI: **TopOpt Postprocess → Apply Postprocess** does this and writes `<name>_optimized.stl` next to your STL. A failed recovery shows a warning and saves nothing; a volume ratio outside 1 ± 0.3 shows a warning and logs the ratio. Choose the smoother there (**Smoothing**: CNN or Gaussian filter, with its width).

## Tetrahedral mesh
For re-analysis in another FE program, or to check the smooth part:
```python
from pyto.core.tet_mesher import TetMesher

tet = TetMesher()
tet.createTetMeshFromSTLFile("design.stl", nElemsDesired=20000)      # uses tetgen
# tet.node_xyz (nodes, 3), tet.elems (elements, 4)
```
- The STL must be watertight; the GUI checks this first and reports non-manifold edges.
- In the GUI: **TopOpt Postprocess → Generate TetMesh**.
- PyTO's own FE analysis is hex-only. Tetrahedral FE analysis inside PyTO is not supported.

## Images
All `plot_*` methods take `save_path=`; see [Fields and plots](fields-and-plots.md). In the GUI, **Display Options → Save Image** saves the 3D view.

## Projects
The GUI's **Projects** window saves the whole setup: STL path, units, material, loads and supports, and the formulation. See [Projects](../gui/projects.md). From code, save the formulation with `spec.to_json()` ([Formulations with specs](../code/formulation-with-spec.md#save-and-load)).

---
Next: [Comparing results](comparing-results.md) · [Back to contents](../README.md)
