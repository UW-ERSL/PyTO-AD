"""Quick start with a different objective: quickstart.py minimizing the
deflection at the load.

The objective is the p-norm of the displacement magnitude over the nodes of
the loaded hole, a smooth stand-in for their largest deflection, at 25 %
volume. PyTO-AD has no hand-written gradient for this quantity; automatic
differentiation provides it. Lines changed from quickstart.py are marked.

Run from the repository root:
    python docs/examples/quickstart_displacement.py
"""
import os
import sys

import numpy as np                                               # changed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

import pyvista
pyvista.OFF_SCREEN = True          # save images instead of opening windows

from pyto.autodiff.sparse_solve import Solvers
from pyto.examples_benchmarks.topopt_structural_benchmarks import (
    StructuralTOExamples, getStructuralTOProblem)
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.topopt.capabilities import check_gradient
from pyto.topopt.drivers.mma import topopt_mma
from pyto.topopt.spec import (ConstraintSpec, ManufacturingSpec,
                              ObjectiveSpec, OptimizationSpec, compile_spec)

# 1. Problem: the bracket STL, voxelized; two holes fixed, load on the third
mesh, material, bc, body_force, to_params = getStructuralTOProblem(
    StructuralTOExamples.ThreeHoleBracketThick, nDOFDesired=20000)

# 2. Finite element analysis with the PARDISO direct solver
#    (use Solvers.SPSOLVE where PARDISO is missing)
fe = HexStructuralFEA(mesh=mesh, mat_prop=material, bc=bc,
                      solver=Solvers.PARDISO, elem_body_force=body_force)

# 3. Formulation: minimize the deflection of the loaded hole with at most
#    25 % of the material, symmetric about z
load = {"LoadHole": {"nodes":                                 # changed
                     np.unique(np.flatnonzero(bc.force) // 3)}}
spec = OptimizationSpec(
    objective=ObjectiveSpec(
        "Displacement(LoadHole, magnitude, pnorm)"),             # changed
    constraints=[ConstraintSpec("VolumeFraction()", "<=", 0.25)],
    manufacturing=ManufacturingSpec(symmetry=["Z"]))
to_params = compile_spec(spec, selections=load, base=to_params)  # changed

# 4. Check the gradients, then optimize with MMA
#    (gradients by automatic differentiation)
for r in check_gradient(to_params, fe):
    print(f"gradient check {r.name}: relative error {r.max_rel_error:.1e}")
u, history, success, message, n_fea = topopt_mma(
    fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
    print_progress=False)
print(f"deflection (p-norm) {history['objective'][-1]:.4g} m, "  # changed
      f"volume {history['volfrac'][-1]:.3f}, {n_fea} FE solves")

# 5. Save a picture of the design
out = os.path.join(os.getcwd(),
                   "quickstart_displacement_topology.png")       # changed
fe.plot_mesh(save_path=out, plot_bc=None, title=None)
print("saved", out)
