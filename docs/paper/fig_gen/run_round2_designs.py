"""Optimizations behind the advisor round-2 figures: fine-mesh designs of two STL parts.

  threehole  ThreeHoleBracketThick, about 200k requested DOF (fig:graphical_abstract)
  ge         GEGrabCAD, about 300k requested DOF (fig:intro_gegrabcad)

Each part is set up as its tutorial problem (supports, load, volume fraction, symmetry) at the finer mesh, with the
PARDISO direct solver and MMA with AD gradients, as in the benchmark sweep. Saved per part in _cache/round2_<name>.npz:
mesh, final (0/1) density, supports and loads, the solid full-domain FE fields (von Mises stress, displacement
magnitude) and the run statistics. The figures are rendered from these files by fig01_graphical_abstract.py and
fig_sweep_galleries.py, so a re-render never re-optimizes.

Run from docs/paper/fig_gen:  python run_round2_designs.py threehole ge
"""
import json
import os
import sys
import time

import numpy as np
import torch

from _style import CACHE

from pyto.autodiff.material_model import MaterialModel
from pyto.autodiff.sparse_solve import Solvers
from pyto.examples_benchmarks.topopt_structural_benchmarks import StructuralTOExamples, getStructuralTOProblem
from pyto.physics.structural.hex_structural_fea import HexStructuralFEA
from pyto.topopt.drivers.mma import topopt_mma

CASES = {"threehole": (StructuralTOExamples.ThreeHoleBracketThick, 200000),
         "ge": (StructuralTOExamples.GEGrabCAD, 300000)}


def run(name):
    problem, n_dof = CASES[name]
    mesh, material, bc, body_force, to_params = getStructuralTOProblem(problem, nDOFDesired=n_dof)
    fe = HexStructuralFEA(mesh=mesh, mat_prop=material, bc=bc, solver=Solvers.PARDISO, elem_body_force=body_force)
    ones = torch.ones(mesh.num_elems, dtype=torch.float64)
    with torch.no_grad():                                        # solid full domain: the FE result panel
        fe.solve(ones, MaterialModel.SIMP)
    fe.postprocess()
    vm_solid = np.asarray(fe.vonMisesStress, float).copy()
    u = np.asarray(fe.sol if not torch.is_tensor(fe.sol) else fe.sol.detach().numpy(), float).reshape(-1, 3)
    umag_solid = np.linalg.norm(u, axis=1)
    t0 = time.perf_counter()
    _u, history, success, message, n_fea = topopt_mma(fe, to_params=to_params, maxMMAIterations=to_params.MaxIterations,
                                                      binarize_topology=not to_params.HeavisideProjection,
                                                      print_progress=False)
    wall = time.perf_counter() - t0
    stats = dict(problem=problem.name, dofs=int(3 * mesh.num_nodes), elements=int(mesh.num_elems), fe_solves=int(n_fea),
                 compliance=float(np.ravel(history["objective"][-1])[0]), volume=float(history["volfrac"][-1]),
                 volume_limit=float(to_params.Constraints[0][2]), success=bool(success), message=message, opt_time_s=wall)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f"round2_{name}.npz"), x=np.asarray(mesh.elemPseudoDensity, float),
                        node_xyz=np.asarray(mesh.node_xyz), elems=np.asarray(mesh.elemArray)[:, :8].astype(np.int64),
                        fixed=np.unique(np.asarray(bc.fixed_dofs) // 3), force=np.asarray(bc.force, float),
                        vm_solid=vm_solid, umag_solid=umag_solid, elem_size=np.asarray(mesh.elem_size, float),
                        stats=json.dumps(stats))
    print(name, json.dumps(stats), flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:] or list(CASES):
        run(name)
