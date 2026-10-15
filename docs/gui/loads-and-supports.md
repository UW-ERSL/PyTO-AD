# Loads and supports

This page explains how to apply forces, torques, supports, thermal loads and body forces in the GUI, and the three ways to select where they act. It is for GUI users setting up a problem.

## Selecting where a load acts
The **Structural Loads** window has a **Selection** mode:

| Mode | You click | Selects |
|---|---|---|
| **Facet** | a face of the part | the whole flat or curved CAD face (triangles grown from the one you click, up to a feature angle) |
| **Triangle** | STL triangles | exactly the triangles you click |
| **Node** | points on the surface | single nodes; or use **Box select nodes** |

**Node mode:**
- **Before meshing,** the candidates are points sampled densely on the STL surface. **After Generate Mesh,** they are the mesh's surface nodes. Clicking snaps to the nearest candidate.
- **Box select nodes:** click the button, then drag a rectangle in the 3D view. Every candidate inside it is selected, **through the whole depth** of the part. A click adds single nodes; a right-click clears the selection.
- A box is stored as a **region**, not as a node list, so it still selects the right nodes after you remesh at a different size.

Faces and nodes are turned into mesh nodes when the problem is built: at analysis and at optimization.

## Structural loads and supports
1. Set **Load Set**. Each load or support you apply is stored as its own set, named `Load1`, `Load2`, `Support1`, ... These names are what you use in expressions ([Optimization setup](optimization-setup.md)).
2. Choose the selection mode and select.
3. Choose **Load Type** and click **Apply**:

| Load Type | Input | Effect |
|---|---|---|
| **Force** | **Force Components** X, Y, Z (in the display unit, e.g. N) | the **total** force, shared equally by the selected nodes |
| **Torque** | **Magnitude** | a torque about the axis of the selected face (planar or cylindrical); needs Facet or Triangle mode |
| **Fixed XYZ** | – | fixes all three displacements |
| **Fixed X**, **Fixed Y**, **Fixed Z** | – | fixes one displacement component (rollers, symmetry planes) |

Arrows (loads) and markers (supports) appear in the 3D view; **Display Options** shows or hides them.

**Supports must stop rigid-body motion** in the directions the loads act. Otherwise the solver stops with *"BC is under-constrained ..."* instead of returning a meaningless answer.

## Thermal loads
The **Thermal Loads** window uses **Facet** or **Triangle** selection:

| Type | Input | Effect |
|---|---|---|
| **Temperature** | temperature (display unit) | holds the selected surface at that temperature |
| **Total Heat** | W | heat input, shared equally by the selected surface nodes |
| **Heat Flux** | W per display area (e.g. W/m², W/mm²) | heat input of flux × (area of the selected faces), shared equally by their surface nodes |

Place heat inputs on surfaces that will stay material. Heat put on void can't flow away, and the temperatures blow up ([Thermal problems](../code/thermal-and-thermostructural.md#thermal-problems)).

A model with both structural and thermal loads is **thermo-structural**. The stress-free temperature is **Zero-strain T** in the Analysis window, and **Stress-free temperature** on the TopOpt Execute Method tab.

## Body force
**Body force** applies an acceleration in m/s² (X, Y, Z), e.g. Y = −9.81 for gravity. It acts on every element in proportion to its density, so it changes as material is removed. Self-weight problems are handled by MMA; OC is disabled for them.

## Checking before you optimize
- **Analysis** shows the deformation and stress of the full design. That's a quick check that loads and supports do what you meant.
- **Optimize** refuses to start while supports or loads are missing for the physics, and says what: *"For a structural optimization, please apply: supports (Fixed XYZ/X/Y/Z on faces or nodes); a load (force, torque or body force)."*

---
Next: [Analysis](analysis.md) · [Back to contents](../README.md)
